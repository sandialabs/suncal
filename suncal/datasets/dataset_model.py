''' Data Set Model, processing ANOVA, etc. and handling column names, dates '''

from collections import namedtuple
import numpy as np
from scipy import stats
from dateutil.parser import parse

from ..common import distributions
from .report.dataset import ReportDataSet
from . import dataset


ColumnStats = namedtuple('ColumnStats', ['name', 'mean', 'standarddev', 'standarderr', 'N', 'degf'])


class DataSet:
    ''' Data Set Model (same as DataSetResults)

        Args:
            data (array): 2D data to compute stats. May contain NaNs if groups
              have different sizes.
            colnames (array): names for each column
    '''
    def __init__(self, data=None, colnames=None):
        if data is None:
            self.data = np.array([[]])
            self.colnames = []
        else:
            self.data = np.atleast_2d(data)
            if colnames is None:
                self.colnames = [str(i) for i in range(self.data.shape[0])]
            else:
                self.colnames = colnames
        self.report = ReportDataSet(self)

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

    def _colidx(self, name):
        ''' Get column index from name '''
        if name is None or name not in self.colnames:
            return 0
        return self.colnames.index(name)

    def _colname(self, name=None):
        ''' Get column name '''
        if name is None:
            return self.colnames[0]
        return name

    def get_column(self, colname=None):
        ''' Get one column of data '''
        return self.data[self._colidx(colname)]

    def ncolumns(self):
        ''' Get number of columns/groups in data set '''
        return len(self.colnames)

    def maxrows(self):
        ''' Return longest column length '''
        return max(len(c) for c in self.data)

    def histogram(self, colname=None, bins='auto'):
        ''' Get histogram of the column data

            Args:
                colname: name of column
                bins: Number of histogram bins, passed to np.histogram.
        '''
        return np.histogram(self.data[self._colidx(colname)], bins=bins)

    def column_stats(self, column_name):
        ''' Get statistics for one column

            Args:
                column_name: name of column

            Returns:
                name: name of column
                mean: mean value of column
                stdandarddev: standard deviation of column
                standarderr: standard error of the mean
                N: number of data points in column
                degf: degrees of freedom
        '''
        data = self.get_column(column_name)
        mean = np.nanmean(data)
        n = len(data)
        stdev = np.nanstd(data, ddof=1)
        sem = stdev / np.sqrt(n)
        return ColumnStats(self._colname(column_name), mean, stdev, sem, n, n-1)

    def mean(self):
        ''' Mean of all data (excluding NaNs) '''
        return np.nanmean(self.data)

    def group_stats(self):
        ''' Get summary statistics for each column/group

            Returns:
                mean (array): mean of each group
                variance (array): variance of each group
                stdandarddev (array): standard deviation of each group
                standarderror (array): standard error of the mean of each group
                num_measurements (array): number of (non-NaN) points in each group
                degf (array): degrees of freedom of each group
        '''
        GroupStats = namedtuple('GroupStats', ['mean', 'variance', 'standarddev',
                                               'standarderror', 'num_measurements', 'degf'])

        try:
            gstats = dataset.group_stats(self.data)
        except TypeError:  # Could be datetime
            ncol = self.ncolumns()
            gstats = GroupStats(
                np.full(ncol, np.nan),
                np.full(ncol, np.nan),
                np.full(ncol, np.nan),
                np.full(ncol, np.nan),
                np.full(ncol, np.nan),
                np.full(ncol, np.nan))
        return gstats

    def pooled_stats(self):
        ''' Calculate pooled standard deviation/variance/error for the 2D data

            Returns:
                mean (float): Mean of all data points
                reproducibility (float): Reproducibility standard deviation
                repeatability (float): Repeatability (pooled) standard deviation
                reproducibility_degf (float): Degrees of freedom of reproducibility
                repeatability_degf (float): Degrees of freedom of repeatability
        '''
        return dataset.pooled_stats(self.data)

    def anova(self, conf=.95):
        ''' Analysis of Variance (one-way)

            Returns:
                f (float): F-statistic
                fcrit (float): Critical F value.
                p (float): P value
                test (bool): True if the groups are statistically the same (f < fcrit and p > 0.05).
                SSbetween (float): Sum-of-squares of between-group variation
                SSwithin (float): Sum-of-squares of within-group variation
                MSbetween (float): Between-group variation
                MSwithin (float): Within-group variation
        '''
        return dataset.anova(self.data, conf=conf)

    def standarderror(self, conf=.95):
        ''' Compute standard error of the mean of 2D data. Checks whether reproducibility is
            significant using ANOVA with conf confidnece in F-test.

            Returns:
                standarderror (float): Standard error of the mean of the data
                degf (float): Degrees of freedom
                standarddeviation (float): Standard deviation
                reprod_significant (bool): If reproducibility between groups is significant and was
                  used to determine standard error.
        '''
        return dataset.standarderror(self.data, conf=conf)

    def fit_dist(self, colname=None, distname='normal'):
        ''' Fit a distribution and return distribution parameters dictionary

            Args:
                colname: name of column
                distname: name of probability distribution

            Returns:
                params: dictionary of fit parameters
        '''
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
        return dataset.autocorrelation(data)

    def autocorrelation_uncert(self, colname=None):
        ''' Get uncertainty adjusted for autocorrelation '''
        data = self.get_column(colname)
        return dataset.uncert_autocorrelated(data)

    def summarize(self):
        ''' Convert the DataSet into an DataSetSummary object by finding
            mean and standard deviation of each group.
        '''
        gstats = self.group_stats()
        return DataSetSummary(self.colnames, gstats.mean, gstats.standarddev, nmeas=gstats.N)

    def to_array(self):
        ''' Summarize the DataSet as an array with columns for x, y, stdev(y), u(y). '''
        gstats = self.group_stats()
        percent = gstats.standarderror/gstats.mean
        dset = DataSet(np.vstack((self._pcolnames, gstats.mean, gstats.standarderror,
                                  percent, percent*100, gstats.standarddev)),
                       colnames=['x', 'y', 'u(y)', 'u(y)/y', 'u(y)/y*100%', 'stdev(y)'])
        dset.coltype = self.coltype
        return dset

    def calculate(self):
        ''' Run calculation. Just returns self here. '''
        return self  # DataSet model is same as DataSetResult


class DataSetSummary(DataSet):
    ''' Dataset given by 3 rows: mean, stdev, and N for each group.

        Args:
            colnames (list): column/group names
            means (array): group mean values
            stds (array): group standard deviations
            nmeas (array): number of measurements in each group (degf+1)
    '''
    ROW_MEAN = 0
    ROW_STD = 1
    ROW_NMEAS = 2
    MAX_ROWS = 3

    def __init__(self, colnames, means, stds, nmeas):
        try:
            data = np.vstack((means, stds, nmeas)).T
        except ValueError:
            data = np.array([[]])
        super().__init__(data, colnames)

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

    def column_stats(self, column_name):
        ''' Get statistics for one column/group '''
        idx = self._colidx(column_name)
        return ColumnStats(self._colname(column_name),
                           self._means()[idx],
                           self._stds()[idx],
                           self._stds()[idx]/np.sqrt(self._nmeas()[idx]),
                           self._nmeas()[idx],
                           self._nmeas()[idx]-1)

    def group_stats(self):
        ''' Get statistics for all groups

            Returns:
                mean (array): mean of each group
                variance (array): variance of each group
                stdandarddev (array): standard deviation of each group
                standarderror (array): standard error of the mean of each group
                num_measurements (array): number of (non-NaN) points in each group
                degf (array): degrees of freedom of each group
        '''
        Result = namedtuple('GroupStats', ['name', 'mean', 'variance', 'standarddev', 'standarderror', 'N', 'degf'])
        return Result(self.colnames,
                      self._means(),
                      self._stds()**2,
                      self._stds(),
                      self._stds()/np.sqrt(self._nmeas()),
                      self._nmeas(),
                      self._nmeas()-1)

    def pooled_stats(self):
        ''' Pooled standard deviation

            Returns:
                mean (float): Mean of all data points
                reproducibility (float): Reproducibility standard deviation
                repeatability (float): Repeatability (pooled) standard deviation
                reproducibility_degf (float): Degrees of freedom of reproducibility
                repeatability_degf (float): Degrees of freedom of repeatability
        '''
        ngroups = self.ncolumns()
        groupmeans = self._means()
        groupvars = self._stds()**2
        degfs = self._nmeas()-1
        pooled_var = sum(groupvars * degfs) / sum(degfs)
        reprod_std = np.std(groupmeans, ddof=1)  # Standard deviation of group means, s(xj)
        reprod_df = ngroups - 1
        repeat_std = np.sqrt(pooled_var)  # Pooled standard deviation
        repeat_df = sum(degfs)
        PooledStats = namedtuple('PooledStats', ['mean', 'reproducibility', 'repeatability',
                                                 'reproducibility_degf', 'repeatability_degf'])
        return PooledStats(np.nanmean(groupmeans), reprod_std, repeat_std, reprod_df, repeat_df)

    def standarderror(self, conf=.95):
        ''' Compute standard error of the mean the data. Checks whether reproducibility is
            significant using ANOVA with conf confidnece in F-test.

            Returns:
                standarderror (float): Standard error of the mean of the data
                degf (float): Degrees of freedom
                standarddeviation (float): Standard deviation
                reprod_significant (bool): If reproducibility between groups is significant and was
                  used to determine standard error.
        '''
        pstats = self.pooled_stats()
        anova_result = self._anova(conf=conf)
        ngroups = self.ncolumns()

        if anova_result.F > anova_result.Fcrit:
            # Reproducibility is significant
            sem = pstats.reproducibility / np.sqrt(ngroups)
            sem_degf = ngroups - 1
            reprod_significant = True
            sem_std = pstats.reproducibility  # Standard deviation used to compute SEM
        else:
            # Reproducibility negligible
            sem_degf = anova_result.degf_a + anova_result.degf_b
            sem_std = (anova_result.degf_a*anova_result.sa2 + anova_result.degf_b*anova_result.sb2)/sem_degf
            sem = np.sqrt(sem_std/anova_result.N)
            reprod_significant = False

        StandardError = namedtuple(
            'StandardError', ['standarderror', 'degf', 'standarddeviation', 'reprod_significant'])
        return StandardError(sem, sem_degf, sem_std, reprod_significant)

    def _anova(self, conf=0.95):
        ''' Compute ANOVA variables for use in pooled stats and anova reports

            Returns:
                mean: mean of all data points
                N: number of data points
                sa2: variability of daily means
                sb2: variability of daily observations
                degf_a: degrees of freedom of sa2
                degf_b: degrees of freedom of sb2
                F: Test F value (sa2/sb2)
                P: Test P value
                fcrit: Critical F value given conf
                gstats: Group statistics calculated from group_statistics()
        '''
        gstats = self.group_stats()
        N = sum(gstats.N)
        mean = sum(gstats.mean)/self.ncolumns()
        sb2 = sum(gstats.variance * gstats.degf) / sum(gstats.degf)   # Pooled variance
        degf_b = sum(gstats.degf)
        degf_a = len(self.colnames) - 1
        sa2 = sum((gstats.degf+1)*(mean - gstats.mean)**2) / degf_a
        F = sa2 / sb2
        P = stats.f.sf(F, dfn=degf_a, dfd=degf_b)
        fcrit = stats.f.ppf(q=conf, dfn=degf_a, dfd=degf_b)

        Result = namedtuple('_AnovaStats', ['mean', 'N', 'sa2', 'sb2', 'degf_a', 'degf_b',
                                            'F', 'P', 'Fcrit', 'gstats'])
        return Result(mean, N, sa2, sb2, degf_a, degf_b, F, P, fcrit, gstats)

    def anova(self, conf=.95):
        ''' Analysis of Variance (one-way)

            Returns:
                f (float): F-statistic
                fcrit (float): Critical F value.
                p (float): P value
                test (bool): True if the groups are statistically the same (f < fcrit and p > 0.05).
                SSbetween (float): Sum-of-squares of between-group variation
                SSwithin (float): Sum-of-squares of within-group variation
                MSbetween (float): Between-group variation
                MSwithin (float): Within-group variation
        '''
        anova = self._anova(conf=conf)
        ngroups = self.ncolumns()
        test = (anova.F < anova.Fcrit) and (anova.P > 0.05)
        MSbetween = anova.sa2
        MSwithin = anova.sb2
        SSbetween = MSbetween * (ngroups-1)
        SSwithin = MSwithin * (anova.N-ngroups)

        AnovaResult = namedtuple('AnovaResult', ['F', 'P', 'Fcrit', 'reprod_significant',
                                                 'SSbet', 'SSwit', 'MSbet', 'MSwit'])

        return AnovaResult(anova.F, anova.P, anova.Fcrit,
                           test, SSbetween, SSwithin, MSbetween, MSwithin)
