''' Grouped arrays and analysis of variance '''

from collections import namedtuple
import numpy as np
from scipy import stats
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import yaml

from . import output
from . import uarray


class AnovaOutput(output.Output):
    ''' Output object for ANOVA results '''
    def __init__(self, array):
        self.array = array

    def get_array(self, name=None):
        ''' Get resulting array, one x, y, uy point for each group. If name is None,
            return a list of possible arrays (just one in this case). There's only
            one possible array, so name is ignored if not None.
        '''
        if name is None:
            return ['Grouped Statistics']
        else:
            return self.array.to_array()

    def get_dists(self, name=None, **kwargs):
        ''' Return a distribution from the output. For ANOVA, this will be t-distribution from one group
            specified by kwargs and using standard error of the mean.
        '''
        gnames = self.group_names()
        if name is None:
            # Return list of groups
            return gnames + ['Repeatability', 'Reproducibility']
        elif name in gnames:
            groupidx = self.group_names().index(name)
            mean = self.group_mean()[groupidx]
            std = self.group_std()[groupidx]
            degf = self.group_df()[groupidx]
            sem = std / np.sqrt(degf+1)
            return {'mean': mean, 'std': sem, 'df': degf}
        elif name == 'Repeatability':
            return {'mean': self.grand_mean(), 'std': self.std_pooled(), 'df': self.df_pooled()}
        elif name == 'Reproducibility':
            return {'mean': self.grand_mean(), 'std': self.reproducibility(), 'df': self.df_reproducibility()}
        else:
            return None

    def report(self, **kwargs):
        ''' Generate ANOVA report '''
        if len(self.array) == 0:
            return output.MDstring('No data.')

        aresult = self.anova()

        rpt = output.MDstring('### Group values:\n\n')
        rows = []
        groupnames = self.group_names()
        meanstrs = output.formatter.f_array(self.group_mean(), **kwargs)
        for g, gmean, gvar, gstd, df in zip(groupnames, meanstrs, self.group_var(), self.group_std(), self.group_df()):
            rows.append([format(g), gmean, output.formatter.f(gvar, **kwargs), output.formatter.f(gstd, **kwargs), format(df)])
        rpt += output.md_table(rows, hdr=['Group', 'Mean', 'Variance', 'Std. Dev.', 'Deg. Freedom'], **kwargs)

        rpt += '### Statistics:\n\n'
        rows = []
        rows.append(['Grand Mean', output.formatter.f(self.grand_mean(), matchto=self.std_pooled(), fmin=0, **kwargs), '-'])
        rows.append(['Pooled Standard Deviation (repeatability)', output.formatter.f(self.std_pooled(), **kwargs), format(int(self.df_pooled()), 'd')])
        rows.append(['Reproducibility', output.formatter.f(self.reproducibility(), **kwargs), format(int(self.df_reproducibility()), 'd')])
        rows.append(['Standard Deviation of All Measurements', output.formatter.f(self.std_all(), **kwargs), format(int(self.df_all()), 'd')])
        rpt += output.md_table(rows, hdr=['Statistic', 'Value', 'Degrees of Freedom'], **kwargs)

        rpt += '### One-way analysis of variance\n\n'
        hdr = ['Source', 'SS', 'MS', 'F', 'F crit (95%)', 'p-value']
        rows = [['Between Groups', format(aresult.SSbet, '.5g'), format(aresult.MSbet, '.5g'), format(aresult.F, '.5g'), format(aresult.Fcrit, '.5g'), format(aresult.P, '.5g')],
                ['Within Groups', format(aresult.SSwit, '.5g'), format(aresult.MSwit, '.5g'), '-', '-', '-'],
                ['Total', format(aresult.SSbet+aresult.SSwit, '.5g'), '-', '-', '-', '-']]
        rpt += output.md_table(rows, hdr=hdr, **kwargs) + '\n\n'

        rpt += output.md_table([['F < Fcrit?', format(aresult.F < aresult.Fcrit)], ['p > 0.05?', format(aresult.P > 0.05)]],
                                hdr=['Test', 'Statistically equivalent (95%)?'])
        return rpt

    def report_all(self, **kwargs):
        ''' Generate combined report with statistics and plot '''
        r = self.report(**kwargs)
        with mpl.style.context(output.mplcontext):
            plt.ioff()
            fig, ax = plt.subplots()
            self.plot(ax=ax)
            r.add_fig(fig)
        return r

    def plot(self, ax=None):
        ''' Plot errorbar of groups '''
        if ax is None:
            fig, ax = plt.subplots()
        arr = self.array.to_array()
        if self.array.is_date():
            try:
                x = mdates.num2date(arr.x)
            except (TypeError, ValueError):
                x = arr.x
        else:
            x = arr.x
        ax.errorbar(x, arr.y, yerr=arr.uy, marker='o', ls='', capsize=4)
        if self.array.is_str():
            ax.set_xticks(arr.x)
            ax.set_xticklabels(self.array.group_names())
        ax.set_xlabel('Group')
        ax.set_ylabel('Value')

    # Leave most calculations as part of array grouped for easier access in interactive mode.
    # Just wrap the functions into self.array.
    def anova(self, conf=.95):
        return self.array.anova(conf=conf)

    def to_array(self):
        return self.to_array()

    def grand_mean(self):
        return self.array.grand_mean()

    def var_pooled(self):
        return self.array.var_pooled()

    def std_pooled(self):
        return self.array.std_pooled()

    def df_pooled(self):
        return self.array.df_pooled()

    def var_all(self):
        return self.array.var_all()

    def std_all(self):
        return self.array.std_all()

    def df_all(self):
        return self.array.df_all()

    def reproducibility(self):
        return self.array.reproducibility()

    def df_reproducibility(self):
        return self.array.df_reproducibility()

    def group_names(self):
        return self.array.group_names()

    def group_mean(self):
        return self.array.group_mean()

    def group_var(self):
        return self.array.group_var()

    def group_std(self):
        return self.array.group_std()

    def group_df(self):
        return self.array.group_df()

    def ngroups(self):
        return self.array.ngroups()

    def ntot(self):
        return self.array.ntot()


class ArrayGrouped(object):
    ''' Array of measured data points. Points are measured in groups (x-values), where
        each x value has multiple y value measurements.

        Allows for analysis of variance of the measured data as well as converting to
        a standard Array type for curve fitting based on group mean and variance.

        Parameters
        ----------
        xvals: list
            list of group names
        yvals: float array
            2D array of y values

        Notes
        -----
        if xvals and/or yvals is not given, values may be added using add_group(). Internally,
        data is stored as NaN-padded array.
    '''
    def __init__(self, xvals=None, yvals=None, name='anova'):
        if xvals is None and yvals is None:
            xvals = np.array([])
            yvals = np.array([[]])
        elif xvals is None:
            # No x values given, use integer to number columns of y
            xvals = np.arange(yvals.shape[1])

        if yvals is None:
            yvals = np.array([[]])

        self.measx = xvals  # 1D array of x values
        self.measy = yvals  # 2D array of y values
        self.out = None
        self.name = name
        self.description = ''

    def __len__(self):
        ''' Length (number of groups) '''
        return self.ngroups()

    def calculate(self):
        ''' Calculate the ANOVA, returning an AnovaOutput object. '''
        self.out = AnovaOutput(self)
        return self.out

    def get_output(self):
        return self.out

    def is_date(self):
        ''' Returns true if x-values are date/time '''
        return any([hasattr(x, 'toordinal') for x in self.measx])

    def is_str(self):
        ''' Returns true if x-values are strings '''
        return any([isinstance(x, str) for x in self.measx])

    def to_array(self, name=''):
        ''' Convert the ArrayGrouped data into an Array object by finding
            mean and standard deviation of each group.

            Parameters
            ----------
            name: string
                Name for the Array object

            Returns
            -------
            arr: Array object
                Array object containing mean x and y values from each group
                in ArrayGrouped.
        '''
        y = self.group_mean()
        uy = self.group_std()
        if self.is_str():
            x = np.arange(len(self.measx))
        elif self.is_date():
            measx = []
            for x in self.measx:
                try:
                    measx.append(mdates.date2num(x))
                except AttributeError:
                    measx.append(np.nan)
            x = np.array(measx)
        else:
            x = np.array(self.measx)
        return uarray.Array(x, y, uy=uy, name=name, xdate=self.is_date())

    def add_group(self, group, xval=None):
        ''' Add a group of measurements.

            Parameters
            ----------
            group: 1D array
                y-values for the group to add
            xval: string, float, or datetime
                x-value associated with this group. If not provided,
                an integer index will be used.
        '''
        if self.measy.shape[1] == 0:
            self.measy = np.array([group])
        else:
            if len(group) > self.measy.shape[1]:
                self.measy = np.pad(self.measy, ((0, 0), (0, len(group)-self.measy.shape[1])),
                                    'constant', constant_values=np.nan)
            elif len(group) < self.measy.shape[1]:
                group = np.pad(group, (0, self.measy.shape[1]-len(group)),
                               'constant', constant_values=np.nan)

            self.measy = np.vstack((self.measy, group))

        if len(self.measx) == 0:
            if xval is None:
                self.measx = [0]
            else:
                self.measx = [xval]
        else:
            if xval is None:
                try:
                    xval = float(self.measx[-1]) + 1
                except ValueError:
                    xval = 0
            self.measx.append(xval)

    def group_names(self):
        ''' Return list of group names '''
        if self.is_date():
            try:
                names = [d.strftime('%d-%b-%Y') for d in self.measx]
            except AttributeError:
                names = [str(d) for d in self.measx]

        elif not self.is_str():
            names = [str(x) for x in self.measx]
        else:
            names = self.measx
        return names

    def get_group(self, groupval):
        ''' Get values from a group '''
        idx = self.measx.index(groupval)
        return self.measy[:, idx]

    def var_pooled(self):
        ''' Pooled variance. sum((ni-1)*si**2) / sum(ni-1) '''
        return sum(self.group_var() * self.group_df()) / sum(self.group_df())

    def std_pooled(self):
        ''' Pooled standard deviation '''
        return np.sqrt(self.var_pooled())

    def df_pooled(self):
        ''' Degrees of freedom for pooled variance/standard deviation. '''
        return sum(self.group_df())

    def grand_mean(self):
        ''' Grand mean of all measurements '''
        return np.nanmean(self.measy)

    def std_all(self):
        ''' Standard deviation of ALL measurements '''
        return np.nanstd(self.measy, ddof=1)

    def var_all(self):
        ''' Variance of ALL measurements '''
        return np.nanvar(self.measy, ddof=1)

    def df_all(self):
        ''' Degrees of freedom of ALL measurements. ntot() - 1. '''
        return self.ntot() - 1

    def reproducibility(self):
        ''' Reproducibility - standard deviation of each group's mean '''
        return np.std(self.group_mean(), ddof=1)

    def df_reproducibility(self):
        ''' Degrees of freedom of reproducibility value (ngroups - 1) '''
        return self.ngroups() - 1

    def group_mean(self):
        ''' Mean of each group '''
        return np.nanmean(self.measy, axis=1)

    def group_var(self):
        ''' Variance of each group '''
        return np.nanvar(self.measy, axis=1, ddof=1)

    def group_std(self):
        ''' Sample standard deviation of each group '''
        return np.nanstd(self.measy, axis=1, ddof=1)

    def group_df(self):
        ''' Degrees of freedom for each group '''
        return np.count_nonzero(np.isfinite(self.measy), axis=1) - 1

    def ngroups(self):
        ''' Number of groups in data set '''
        return len(self.measx)

    def ntot(self):
        ''' Total number of measurements '''
        return np.count_nonzero(np.isfinite(self.measy))

    def anova(self, conf=.95):
        ''' Analysis of Variance (one-way)

            Parameters
            ----------
            conf: float
                Confidence as fraction (0-1) for critical f value

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
        n = self.group_df() + 1  # Number of measurements in each group
        ngroups = self.ngroups()
        ntot = self.ntot()
        SSbetween = sum(n*(self.group_mean() - self.grand_mean())**2)
        SSwithin = sum((n-1)*self.group_var())
        MSbetween = SSbetween / (ngroups - 1)
        MSwithin = SSwithin / (ntot-ngroups)
        F = MSbetween / MSwithin
        P = stats.f.sf(F, dfn=(ngroups-1), dfd=(ntot-ngroups))
        fcrit = stats.f.ppf(q=conf, dfn=ngroups-1, dfd=ntot-ngroups)
        test = (F < fcrit) and (P > 0.05)
        AnovaResult = namedtuple('AnovaResult', ['F', 'P', 'Fcrit', 'passfail', 'SSbet', 'SSwit', 'MSbet', 'MSwit'])
        return AnovaResult(F, P, fcrit, test, SSbetween, SSwithin, MSbetween, MSwithin)

    def set_xy(self, data):
        ''' Set (and overwrite) ArrayGrouped data using 2-column array,
            where first column is the group index.
        '''
        groups = list(np.unique(data[:, 0]))
        gvals = []
        for g in groups:
            gvals.append(data[data[:, 0] == g, 1])

        def padnans(v, fillval=np.nan):
            # https://stackoverflow.com/questions/40569220/efficiently-convert-uneven-list-of-lists-to-minimal-containing-array-padded-with
            lens = np.array([len(item) for item in v])
            mask = lens[:, None] > np.arange(lens.max())
            out = np.full(mask.shape, fillval)
            out[mask] = np.concatenate(v)
            return out
        self.measx = groups
        self.measy = padnans(gvals)

    @classmethod
    def from_xy(cls, data):
        ''' Create a new ArrayGrouped from 2-column array, where first column
            is the group index.
        '''
        newarr = cls()
        newarr.set_xy(data)
        return newarr

    def get_config(self):
        d = {}
        d['mode'] = 'anova'
        d['name'] = self.name
        d['groupnames'] = self.measx
        d['groupvals'] = self.measy.astype('float').tolist()
        d['xdate'] = self.is_date()
        d['desc'] = self.description
        return d

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

    @classmethod
    def from_config(cls, config):
        ''' Create new ArrayGrouped or ArrayGroupedSummary from configuration dictionary '''
        if 'nmeas' in config:
            newanova = ArrayGroupedSummary.from_config(config)
        else:
            newanova = cls(name=config.get('name', 'anova'))
            newanova.measx = config['groupnames']
            newanova.measy = np.array(config['groupvals'])
            newanova.description = config.get('desc', '')
        return newanova

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
        except yaml.scanner.ScannerError:
            return None  # Can't read YAML

        u = cls.from_config(config[0])  # config yaml is always a list
        return u


class ArrayGroupedSummary(ArrayGrouped):
    ''' Grouped Array where only summary statistics are given.

        Parameters
        ----------
        xvals: list
            list of group names
        yvals: float array
            1D array of y mean values
        ystds: float array
            1D array of y standard deviations
        nmeas: float array
            1D array of number of measurements in each group (degf+1)

    '''
    def __init__(self, xvals=None, ymeans=None, ystds=None, nmeas=None, name='anova'):
        super(ArrayGroupedSummary, self).__init__(xvals, np.array([ymeans]).transpose(), name=name)
        self.nmeas = nmeas
        self.ystds = ystds

    def grand_mean(self):
        ''' Grand mean of all measurements '''
        return sum(self.nmeas * self.measy.flatten())/sum(self.nmeas)

    def group_var(self):
        ''' Variance of each group '''
        return self.ystds**2

    def group_std(self):
        ''' Standard deviation of each group '''
        return self.ystds

    def group_df(self):
        ''' Degrees of freedom for each group '''
        return self.nmeas - 1

    def ntot(self):
        ''' Total number of measurements '''
        return sum(self.nmeas)

    def get_config(self):
        ''' Get configuration dictionary '''
        d = super(ArrayGroupedSummary, self).get_config()
        d['groupvals'] = self.measy.astype('float').flatten().tolist()
        d['ystds'] = self.ystds.astype('float').tolist()
        d['nmeas'] = self.nmeas.astype('int').tolist()
        return d

    @classmethod
    def from_config(cls, config):
        ''' Load ArrayGroupedSummary from configuration dictionary '''
        newanova = cls(xvals=config['groupnames'], ymeans=np.array(config['groupvals']),
                       ystds=np.array(config['ystds']), nmeas=np.array(config['nmeas']), name=config.get('name', 'anova'))
        newanova.description = config.get('desc', '')
        return newanova
