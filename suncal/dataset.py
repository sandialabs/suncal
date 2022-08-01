''' Class for holding a measured data set and performing basic calculations on the data '''
from collections import namedtuple

import numpy as np
from scipy import stats
from dateutil.parser import parse
import yaml

from . import distributions
from . import output
from . import report
from . import plotting
from . import ttable


# Keep these functions at top-level for easier library use
def autocorrelation(x):
    ''' Calculate autocorrelation

        Parameters
        ----------
        x: array
            Autocorrelated data

        Returns
        -------
        rho: array
            Autocorrelation vs lag array. Same length as x.

        Notes
        -----
        Implements equation 10 in Zhang, Metrologia 43, S276.
        Same as Rh in NIST https://www.itl.nist.gov/div898/handbook/eda/section3/autocopl.htm
    '''
    # This is pretty close, but not exactly np.correlate(x, x, mode='full') normalized.
    rho = np.zeros(len(x))
    xbar = x.mean()
    denom = sum((x-xbar)**2)
    for i in range(len(x)):
        rho[i] = sum((x[:len(x)-i] - xbar) * (x[i:] - xbar)) / denom
    return rho


def uncert_autocorrelated(x, conf=.95):
    ''' Calculate standard uncertainty in x accounting for autocorrelation.

        Parameters
        ----------
        x: array
            Sampled data
        conf: float
            Confidence (0-1) for finding nc cutoff

        Returns
        -------
        uncert: float
            Standard uncertainty accounting for autocorrelation
        r: float
            Multiplier for converting variance into autocorrelation-corrected variance
        r_unc: float
            Multiplier for converting uncertainty into autocorrelation-corrected uncertainty
        nc: int
            Cut-off lag
    '''
    n = len(x)
    if n > 3:
        rho = autocorrelation(x)
        sigr = _sigma_rhok(rho)

        # Limit lag to be in 95% limits
        k = ttable.t_factor(conf, n)
        nc = np.argwhere(np.abs(rho) > k * sigr)
        if len(nc) > 0:
            nc = nc.max()
            nc = min(nc, n//4)  # Also limit lag to be n//4
        else:
            nc = 0

        i = np.arange(1, nc+1)
        r = 1 + 2/n*sum((n-i) * rho[1:nc+1])  # Skip the rho[0] == 1 point.
        unc = np.sqrt(np.var(x, ddof=1) / n * r)
    else:
        unc = np.nan
        r = np.nan
        nc = np.nan

    Result = namedtuple('AutoCorrUncert', ['uncert', 'r', 'r_unc', 'nc'])
    return Result(unc, r, np.sqrt(r), nc)


def _sigma_rhok(rho):
    ''' Calculate sigma_rho parameter used for autocorrelation confidence band. '''
    # Eq. 14 in Zhang
    n = len(rho)
    if n == 0:
        return np.array([0])
    sigma_rhok = np.zeros(n)
    for k in range(n):
        sigma_rhok[k] = np.sqrt((1 + 2 * sum(rho[1:k+1]**2))/n)
    return sigma_rhok


class DataSet(object):
    ''' Class for storing and statistics on measured data sets

        Parameters
        ----------
        data: array
            1D or 2D array of data
        colnames: list of string, numeric, or datetime
            Names for each column
        name: string
            Name for the dataset
    '''
    def __init__(self, data=None, colnames=None, name='data'):
        self.name = name
        self.description = ''
        self.set_data(data, colnames)
        self.out = DataSetOutput(self)

    def set_data(self, data, colnames=None):
        ''' Set the data array '''
        if data is None:
            self.data = np.array([[]])
            self.colnames = []
        else:
            self.data = np.atleast_2d(data)
            if colnames is None:
                colnames = [str(i) for i in range(self.data.shape[0])]
            self.colnames = colnames

    @property
    def colnames(self):
        ''' Get column names '''
        return self._colnames

    @colnames.setter
    def colnames(self, value):
        ''' Set column names, and parse strings into float or date values if possible. '''
        if all(hasattr(v, 'month') for v in value):
            cols = value
            coltype = 'date'
            self._colnames = [v.strftime('%Y-%m-%d') for v in value]
        else:
            try:
                cols = [float(c) for c in value]
                coltype = 'float'
            except (TypeError, ValueError):
                try:
                    cols = [parse(c) for c in value]
                    coltype = 'date'
                except (ValueError, OverflowError):
                    cols = np.arange(len(value))
                    coltype = 'str'

            self._colnames = value  # Raw column names
        self._pcolnames = cols  # Parsed numeric column names
        self.coltype = coltype  # type of columns

    def colnames_parsed(self):
        ''' Get parsed column names (converted to float or datetime if possible) '''
        return self._pcolnames

    def get_column(self, colname=None):
        ''' Get one column of data '''
        return self.data[self._colidx(colname)]

    def _colidx(self, name):
        ''' Get column index from name '''
        if name is None or name not in self.colnames:
            return 0
        else:
            return self.colnames.index(name)

    def _colname(self, name=None):
        ''' Get column name '''
        if name is None:
            return self.colnames[0]
        else:
            return name

    def ncolumns(self):
        ''' Get number of columns/groups in data set '''
        return len(self.colnames)

    def maxrows(self):
        ''' Return longest column length '''
        return max(len(c) for c in self.data)

    def histogram(self, colname=None, bins='auto'):
        ''' Get histogram of the column data '''
        return np.histogram(self.data[self._colidx(colname)], bins=bins)

    def stats(self, colname=None):
        ''' Get summary statistics for one column '''
        dat = self.get_column(colname)
        mean = np.nanmean(dat)
        n = len(dat)
        stdev = np.nanstd(dat, ddof=1)
        sem = stdev / np.sqrt(n)
        Result = namedtuple('ColumnStats', ['name', 'mean', 'stdev', 'sem', 'N', 'df'])
        return Result(self._colname(colname), mean, stdev, sem, n, n-1)

    def group_stats(self):
        ''' Get summary statistics for each column '''
        try:
            groupvar = np.nanvar(self.data, axis=1, ddof=1)
            groupstd = np.sqrt(groupvar)
            groupmean = np.nanmean(self.data, axis=1)
            groupN = np.count_nonzero(np.isfinite(self.data), axis=1)
            groupsem = groupstd / np.sqrt(groupN)
        except TypeError:  # Could be datetime
            ncol = self.ncolumns()
            groupvar = np.full(ncol, np.nan)
            groupstd = np.full(ncol, np.nan)
            groupmean = np.full(ncol, np.nan)
            groupN = np.full(ncol, np.nan)
            groupsem = np.full(ncol, np.nan)

        Result = namedtuple('GroupStats', ['name', 'mean', 'var', 'stdev', 'sem', 'N', 'df'])
        return Result(self.colnames, groupmean, groupvar, groupstd, groupsem, groupN, groupN - 1)

    def pooled_stats(self):
        ''' Get summary statistics for all columns (pooled variance, etc.) '''
        ntot = np.count_nonzero(np.isfinite(self.data))
        gstats = self.group_stats()
        poolvar = sum(gstats.var * gstats.df) / sum(gstats.df)
        poolstd = np.sqrt(poolvar)
        pooldf = sum(gstats.df)
        reproducibility = np.std(gstats.mean, ddof=1)  # Standard deviation of group means
        reproducibility_df = len(self.colnames) - 1
        reprod_ofmean = reproducibility / np.sqrt(len(self.colnames))   # sR / sqrt(M)
        poolstd_ofmean = poolstd / np.sqrt(ntot)                        # sr / sqrt(N*M)

        grandmean = np.nanmean(self.data)
        allvar = np.nanvar(self.data, ddof=1)
        allstd = np.sqrt(allvar)
        alldf = ntot - 1

        Result = namedtuple('PooledStats', ['mean', 'poolvar', 'poolstd', 'pooldf', 'pool_ofmean',
                                            'reproducibility', 'reproducibilitydf', 'reprod_ofmean',
                                         'allvar', 'allstd', 'N', 'alldf'])
        return Result(grandmean, poolvar, poolstd, pooldf, poolstd_ofmean,
                      reproducibility, reproducibility_df, reprod_ofmean,
                      allvar, allstd, ntot, alldf)

    def anova(self, conf=.95):
        ''' Analysis of Variance (one-way)

            Parameters
            ----------
            conf: float
                Level of confidence as fraction (0-1) for critical f value

            Returns
            -------
            f: float
                F-statistic
            fcrit: float
                Critical F value.
            p: float
                P value
            test: bool
                True if the groups are statistically the same (f < fcrit and p > 0.05).
        '''
        # NOTE: scipy.f_oneway can do this, but only with full 2D data.
        # if data is summarized as group means/standard deviations,
        # this one will work. This version also provides intermediate (SS, MS) data output
        gstats = self.group_stats()
        pstats = self.pooled_stats()
        n = gstats.N  # Number of measurements in each group
        ngroups = self.ncolumns()
        ntot = pstats.N
        SSbetween = sum(n*(gstats.mean - pstats.mean)**2)
        SSwithin = sum((n-1)*gstats.var)
        MSbetween = SSbetween / (ngroups - 1)
        MSwithin = SSwithin / (ntot-ngroups)
        F = MSbetween / MSwithin
        P = stats.f.sf(F, dfn=(ngroups-1), dfd=(ntot-ngroups))
        fcrit = stats.f.ppf(q=conf, dfn=ngroups-1, dfd=ntot-ngroups)
        test = (F < fcrit) and (P > 0.05)
        AnovaResult = namedtuple('AnovaResult', ['F', 'P', 'Fcrit', 'passfail', 'SSbet', 'SSwit', 'MSbet', 'MSwit'])
        return AnovaResult(F, P, fcrit, test, SSbetween, SSwithin, MSbetween, MSwithin)

    def fit_dist(self, colname=None, distname='normal'):
        ''' Fit a distribution and return distribution parameters dictionary '''
        fitdist = distributions.get_distribution(distname)
        data = self.get_column(colname)
        params = fitdist.fit(data)
        return params

    def correlation(self):
        ''' Get correlation matrix between columns '''
        return np.corrcoef(self.data)

    def autocorrelation(self, colname=None):
        ''' Get autocorrelation array rho(lag) '''
        data = self.get_column(colname)
        return autocorrelation(data)

    def autocorrelation_uncert(self, colname=None):
        ''' Get uncertainty adjusted for autocorrelation '''
        data = self.get_column(colname)
        return uncert_autocorrelated(data)

    def summarize(self, name=''):
        ''' Convert the DataSet into an DataSetSummary object by finding
            mean and standard deviation of each group.

            Parameters
            ----------
            name: string
                Name for the Array object

            Returns
            -------
            summary: DataSetSummary object
                Summarized data set
        '''
        gstats = self.group_stats()
        return DataSetSummary(self.colnames, gstats.mean, gstats.stdev, nmeas=gstats.N)

    def to_array(self):
        ''' Summarize the DataSet as an array with columns for x, y, stdev(y), u(y). '''
        gstats = self.group_stats()
        percent = gstats.sem/gstats.mean
        dset = DataSet(np.vstack((self._pcolnames, gstats.mean, gstats.sem, percent, percent*100, gstats.stdev)),
                       colnames=['x', 'y', 'u(y)', 'u(y)/y', 'u(y)/y*100%', 'stdev(y)'])
        dset.coltype = self.coltype
        return dset

    @classmethod
    def from_config(cls, config):
        ''' Create new DataSet from configuration dictionary '''
        if 'nmeas' in config:
            newdat = DataSetSummary.from_config(config)
        else:
            newdat = cls(name=config.get('name', 'data'))
            newdat.colnames = config['colnames']
            newdat.data = np.array(config['data'])
            newdat.description = config.get('desc', '')
        return newdat

    def calculate(self):
        ''' "Calculate" returning the output, for compatibility with other project classes. '''
        return self.out

    def get_output(self):
        ''' Get output report '''
        return self.out

    def get_config(self):
        ''' Get the dataset configuration dictionary '''
        d = {}
        d['mode'] = 'data'
        d['name'] = self.name
        d['colnames'] = self.colnames
        d['data'] = self.data.astype('float').tolist()
        d['desc'] = self.description
        return d

    @classmethod
    def from_configfile(cls, fname):
        ''' Read and parse the configuration file. Returns a new UncertRisk
            instance.

            Parameters
            ----------
            fname: string or file
                File name or open file object to read configuration from
        '''
        try:
            try:
                yml = fname.read()  # fname is file object
            except AttributeError:
                with open(fname, 'r') as fobj:  # fname is string
                    yml = fobj.read()
        except UnicodeDecodeError:
            # file is binary, can't be read as yaml
            return None

        try:
            config = yaml.safe_load(yml)
        except yaml.YAMLError:
            return None  # Can't read YAML

        u = cls.from_config(config[0])  # config yaml is always a list
        return u

    def save_config(self, fname):
        ''' Save configuration to file.

            Parameters
            ----------
            fname: string or file
                File name or file object to save to
        '''
        d = self.get_config()
        out = yaml.dump([d], default_flow_style=False)
        try:
            fname.write(out)
        except AttributeError:
            with open(fname, 'w') as f:
                f.write(out)


class DataSetSummary(DataSet):
    ''' Dataset given by 3 rows: mean, stdev, and N for each group.

        Parameters
        ----------
        colnames: list
            List of column/group names
        means: float array
            1D array of group mean values
        stds: float array
            1D array of group standard deviations
        nmeas: float array
            1D array of number of measurements in each group (degf+1)
    '''
    ROW_MEAN = 0
    ROW_STD = 1
    ROW_NMEAS = 2
    MAX_ROWS = 3

    def __init__(self, colnames=None, means=None, stds=None, nmeas=None, name='data'):
        super().__init__(name=name)
        self.colnames = colnames                   # Parse them in setter
        try:
            self.data = np.vstack((means, stds, nmeas)).T
        except ValueError:
            self.data = np.array([[]])

    def _means(self):
        ''' Get mean of each group '''
        if self.data.shape[1] == 0:
            return []
        return self.data[:, self.ROW_MEAN]

    def _stds(self):
        ''' Get standard deviation of each group '''
        if self.data.shape[1] == 0:
            return []
        return self.data[:, self.ROW_STD]

    def _nmeas(self):
        ''' Get number of measurements in each group '''
        if self.data.shape[1] == 0:
            return []
        return self.data[:, self.ROW_NMEAS]

    def maxrows(self):
        ''' Get maximum number of rows in data table (always 3) '''
        return self.MAX_ROWS

    def ncolumns(self):
        ''' Get number of columns/groups in data set '''
        return len(self._means())

    def stats(self, colname=None):
        ''' Get statistics for one column/group '''
        idx = self._colidx(colname)
        Result = namedtuple('ColumnStats', ['name', 'mean', 'stdev', 'sem', 'N', 'df'])
        return Result(self._colname(colname),
                   self._means()[idx],
                   self._stds()[idx],
                   self._stds()[idx]/np.sqrt(self._nmeas()[idx]),
                   self._nmeas()[idx],
                   self._nmeas()[idx]-1)

    def group_stats(self):
        ''' Get statistics for all groups '''
        Result = namedtuple('GroupStats', ['name', 'mean', 'var', 'stdev', 'sem', 'N', 'df'])
        return Result(self.colnames,
                   self._means(),
                   self._stds()**2,
                   self._stds(),
                   self._stds()/np.sqrt(self._nmeas()),
                   self._nmeas(),
                   self._nmeas()-1)

    def pooled_stats(self):
        ''' Get pooled statistics '''
        Result = namedtuple('PooledStats', ['mean', 'poolvar', 'poolstd', 'pooldf', 'pool_ofmean',
                                            'reproducibility', 'reproducibilitydf', 'reprod_ofmean',
                                            'allvar', 'allstd', 'N', 'alldf'])
        nmeas = self._nmeas()
        means = self._means()
        stds = self._stds()

        ntot = sum(nmeas)
        groupvar = stds**2
        grandmean = sum(nmeas * means)/sum(nmeas)
        poolvar = sum(groupvar * (nmeas-1)) / sum(nmeas - 1)
        poolstd = np.sqrt(poolvar)
        pooldf = sum(nmeas - 1)
        poolstd_ofmean = poolstd/np.sqrt(ntot)
        reproducibility = np.std(means, ddof=1)
        reproducibility_df = len(self.colnames) - 1
        reprod_ofmean = reproducibility / np.sqrt(reproducibility_df+1)
        alldf = ntot - 1
        return Result(grandmean,
                      poolvar,
                      poolstd,
                      pooldf,
                      poolstd_ofmean,
                      reproducibility,
                      reproducibility_df,
                      reprod_ofmean,
                      None,
                      None,
                      ntot,
                      alldf)

    def get_config(self):
        ''' Get configuration dictionary '''
        d = super().get_config()
        d['nmeas'] = self._nmeas().astype('float').tolist()
        d['means'] = self._means().astype('float').tolist()
        d['stds'] = self._stds().astype('float').tolist()
        d['summary'] = True
        return d

    @classmethod
    def from_config(cls, config):
        ''' Load DataSetSummary from configuration dictionary '''
        newdat = cls(colnames=config.get('colnames', config.get('groupnames', [])), means=np.array(config['means']),
                     stds=np.array(config['stds']), nmeas=np.array(config['nmeas']), name=config.get('name', 'data'))
        newdat.description = config.get('desc', '')
        return newdat


class DataSetOutput(output.Output):
    ''' Report generator for DataSet objects '''
    def __init__(self, dataset):
        self.dataset = dataset

    def get_dataset(self, name=None):
        ''' Get a DataSet from this output. If name is None, return list of available datasets. '''
        names = ['Columns']
        if len(self.dataset.colnames) > 1:
            names.append('Summarized Array')

        if name is None:
            return names

        elif name in names:
            if name == 'Summarized Array':
                return self.dataset.to_array()
            else:
                return self.dataset

        else:
            raise ValueError('{} not found in output'.format(name))
        return names

    def get_dists(self):
        ''' Get dictionary of distributions in this dataset '''
        d = {}
        colnames = self.dataset.colnames

        # Individual columns are returned as sampled data
        for col in colnames:
            d[col] = {'samples': self.dataset.get_column(col)}

        # Pooled stats returned as mean/std/df dictionary
        if len(colnames) > 1:
            pstats = self.dataset.pooled_stats()
            d['Repeatability'] = {'mean': pstats.mean, 'std': pstats.poolstd,
                                  'sem': pstats.pool_ofmean, 'df': pstats.pooldf}
            d['Reproducibility'] = {'mean': pstats.mean, 'std': pstats.reproducibility,
                                    'sem': pstats.reprod_ofmean,
                                    'df': pstats.reproducibilitydf}
        return d

    def report(self, **kwargs):
        ''' Generate summary report '''
        rows = []
        names = self.dataset.colnames
        gstats = self.dataset.group_stats()
        meanstrs = report.Number.number_array(gstats.mean, fmin=0)
        for g, gmean, gvar, gstd, gsem, df in zip(names, meanstrs, gstats.var, gstats.stdev, gstats.sem, gstats.df):
            rows.append([format(g), gmean, report.Number(gvar, fmin=0), report.Number(gstd, fmin=0), report.Number(gsem, fmin=0), format(df)])

        rpt = report.Report(**kwargs)
        rpt.table(rows, hdr=['Group', 'Mean', 'Variance', 'Std. Dev.', 'Std. Error', 'Deg. Freedom'])
        return rpt

    def report_all(self, **kwargs):
        ''' Report summary, pooled statistics, anova '''
        rpt = self.report(**kwargs)
        rpt.append(self.report_pooled(**kwargs))
        rpt.append(self.report_anova(**kwargs))
        return rpt

    def report_column(self, colname=None, **kwargs):
        ''' Report statistics for one column '''
        st = self.dataset.stats(colname)
        dat = self.dataset.get_column(colname)
        q025, q25, q75, q975 = np.quantile(dat, (.025, .25, .75, .975))
        rows = [['Mean', report.Number(st.mean, fmin=0)],
                ['Standard Deviation', report.Number(st.stdev, fmin=3)],
                ['Std. Error of the Mean', report.Number(st.sem, fmin=3)],
                ['Deg. Freedom', format(st.df, '.2f')],
                ['Minimum',  report.Number(dat.min(), fmin=3)],
                ['First Quartile', report.Number(q25, fmin=3)],
                ['Median', report.Number(np.median(dat), fmin=3)],
                ['Third Quartile', report.Number(q75, fmin=3)],
                ['Maximum', report.Number(dat.max(), fmin=3)],
                ['95% Coverage Interval', '{}, {}'.format(report.Number(q025, fmin=3), report.Number(q975, fmin=3))]
                ]
        rpt = report.Report(**kwargs)
        rpt.table(rows, hdr=['Parameter', 'Value'])
        return rpt

    def report_pooled(self, **kwargs):
        ''' Report pooled statistics and grand mean '''
        pstats = self.dataset.pooled_stats()
        rows = []
        rows.append(['Grand Mean', report.Number(pstats.mean, matchto=pstats.poolstd, fmin=0), '-'])
        rows.append(['Repeatability (Pooled Standard Deviation)', report.Number(pstats.poolstd), report.Number(pstats.pooldf, fmin=0)])
        rows.append(['Reproducibility', report.Number(pstats.reproducibility), report.Number(pstats.reproducibilitydf, fmin=0)])
        rows.append(['Standard Deviation of All Measurements', report.Number(pstats.allstd), report.Number(pstats.alldf, fmin=0)])
        rpt = report.Report(**kwargs)
        rpt.table(rows, hdr=['Statistic', 'Value', 'Degrees of Freedom'])
        return rpt

    def report_anova(self, **kwargs):
        ''' Report analysis of variance '''
        aresult = self.dataset.anova()
        hdr = ['Source', 'SS', 'MS', 'F', 'F crit (95%)', 'p-value']
        rows = [['Between Groups', report.Number(aresult.SSbet), report.Number(aresult.MSbet),
                report.Number(aresult.F, fmt='decimal'), report.Number(aresult.Fcrit, fmt='decimal'),
                report.Number(aresult.P, fmt='decimal')],
                ['Within Groups', report.Number(aresult.SSwit), report.Number(aresult.MSwit), '-', '-', '-'],
                ['Total', report.Number(aresult.SSbet+aresult.SSwit), '-', '-', '-', '-']]
        rpt = report.Report(**kwargs)
        rpt.table(rows, hdr=hdr)
        rpt.table([['F < Fcrit?', format(aresult.F < aresult.Fcrit)], ['p > 0.05?', format(aresult.P > 0.05)]],
                  hdr=['Test', 'Statistically equivalent (95%)?'])
        return rpt

    def report_correlation(self, **kwargs):
        ''' Report correlation coefficients between columns '''
        rpt = report.Report(**kwargs)
        if len(self.dataset.colnames) < 2:
            rpt.txt('Add columns to compute correlation.')
            return rpt

        corr = self.dataset.correlation()
        names = self.dataset.colnames
        rows = []
        for name, corrow in zip(names, corr):
            rows.append([name] + [report.Number(f) for f in corrow])
        rpt.table(rows, hdr=['-'] + names)
        return rpt

    def report_autocorrelation(self, colname=None, **kwargs):
        ''' Report of autocorrelation for one column '''
        acor = self.dataset.autocorrelation_uncert(colname=colname)
        rows = [['r (variance)', report.Number(acor.r, fmin=0)],
                ['r (uncertainty)', report.Number(acor.r_unc, fmin=0)],
                ['nc', str(acor.nc)],
                ['uncertainty', report.Number(acor.uncert)]]
        rpt = report.Report(**kwargs)
        rpt.table(rows, hdr=['Parameter', 'Value'])
        return rpt

    def plot_groups(self, plot=None):
        ''' Plot each group with errorbars.

            Parameters
            ----------
            plot: Figure or Axis
                Either matplotlib figure or axis object. If omitted,
                new figure will be created.
        '''
        fig, ax = plotting.initplot(plot)
        summary = self.dataset.summarize()
        gstats = summary.group_stats()
        x = summary.colnames_parsed()
        y = gstats.mean
        if len(x) != len(y): return  # Nothing to plot
        uy = gstats.stdev

        ax.errorbar(x, y, yerr=uy, marker='o', ls='', capsize=4)
        if self.dataset.coltype == 'str':
            ax.set_xticks(x)
            ax.set_xticklabels(self.dataset.colnames)
        ax.set_xlabel('Group')
        ax.set_ylabel('Value')

    def plot_histogram(self, colname=None, plot=None, fit=None, qqplot=False, bins='sqrt', points=None, coverage=None):
        ''' Plot a histogram, with optional distribution fit and qq-plot, of one column

            Parameters
            ----------
            colname: string
                Name of column to plot
            plot: object
                Either matplotlib figure or axis object. If omitted,
                new figure will be created.
            fit: string
                Name of distribution to fit
            qqplot: bool
                Show a Q-Q normal probability plot
            bins: int or string
                Number of bins for histogram (see numpy.histogram_bin_edges)
            points: int
                Number of points to show in Q-Q plot (reduce for speed)
            coverage: array
                List of coverage probabilities to plot as vertical lines
        '''
        fig, ax = plotting.initplot(plot)
        data = self.dataset.get_column(colname)
        if colname is None and len(self.dataset.colnames) > 0:
            colname = self.dataset.colnames[0]
        plotting.fitdist(data, fit, plot=fig, qqplot=qqplot, bins=bins, points=points, coverage=coverage, xlabel=colname)

    def plot_autocorrelation(self, colname=None, plot=None, nmax=None, conf=.95):
        ''' Plot autocorrelation vs lag for one column

            Parameters
            ----------
            colname: string
                Name of column to plot
            plot: object
                Figure or axis to plot on
            nmax: int
                Maximum lag (upper x limit to plot)
            conf: float
                Confidence level (0-1) for confidence bands
        '''
        fig, ax = plotting.initplot(plot)

        x = self.dataset.get_column(colname)
        rho = self.dataset.autocorrelation(colname)

        if nmax is None:
            nmax = len(x)

        k = ttable.t_factor(conf, np.inf)
        ax.plot(rho[:nmax+1], marker='o')
        z = k/np.sqrt(len(x))
        ax.axhline(0, ls='-', color='black')
        ax.axhline(z, ls=':', color='black')
        ax.axhline(-z, ls=':', color='black')
        sig = _sigma_rhok(rho)
        ax.plot(k*sig[:nmax+1], ls='--', color='red')
        ax.plot(-k*sig[:nmax+1], ls='--', color='red')

    def plot_lag(self, colname=None, lag=1, plot=None):
        ''' Plot lag-plot for column

            Parameters
            ----------
            colname: string
                Name of column to plot
            plot: object
                Figure or axis to plot on
            lag: int
                Lag value to plot
        '''
        fig, ax = plotting.initplot(plot)
        x = self.dataset.get_column(colname)
        ax.plot(x[lag:], x[:len(x)-lag], ls='', marker='o')

    def plot_scatter(self, col1, col2, plot=None):
        ''' Scatter plot between two columns

            Parameters
            ----------
            col1: string
                Name of column 1 data
            col2: string
                Name of column 2 data
            plot: object
                Figure or axis to plot on
        '''
        fig, ax = plotting.initplot(plot)
        x = self.dataset.get_column(col1)
        y = self.dataset.get_column(col2)
        ax.scatter(x, y, marker='.')
        ax.set_xlabel(col1)
        ax.set_ylabel(col2)
