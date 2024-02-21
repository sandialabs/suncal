''' Data Set Model, processing ANOVA, etc. and handling column names, dates '''
from dataclasses import dataclass
import numpy as np
from scipy import stats
from dateutil.parser import parse

from ..common import reporter
from .report.dataset import ReportDataSet
from . import dataset


@reporter.reporter(ReportDataSet)
@dataclass
class DataSetResult:
    ''' Results of DataSet statistics

        Attributes:
            data: The data used for calculation
            colnames: Names, as-entered, of each column
            colvals: Values, as-parsed, of each column
            groups: Statistics for each column/group
            pooled: Pooled statistics for the data set
            uncertainty: Estimate of uncertainty/standard error
            anova: Analysis of Variance
            autocorrelation: Autocorrelation uncertainty for each column
            correlation: Correlation matrix between the columns
    '''
    data: list[list[float]]
    colnames: list[str]   # String/Value
    colvals: list[float]  # Parsed
    groups: dataset.GroupSummary  # For each group
    pooled: dataset.PooledResult  # For the whole data set
    uncertainty: dataset.StandardErrorResult
    anova: dataset.AnovaResult
    autocorrelation: list[dataset.AutoCorrelationResult]  # For each column
    correlation: list[list[float]]  # Correlation between columns

    def array(self):
        ''' Convert 2D to array of mean/unc '''

    @property
    def mean(self):
        ''' Grand mean of all data points '''
        return np.mean(self.data)

    @property
    def ncolumns(self):
        return len(self.colnames)

    @property
    def maxrows(self):
        ''' Return longest column length '''
        return max(len(c) for c in self.data)

    @property
    def totalN(self):
        return np.count_nonzero(np.isfinite(self.data))

    def histogram(self, bins=20):
        ''' Get histogram of entire set '''
        return np.histogram(self.data, bins=bins)

    def groupidx(self, name) -> int:
        ''' Get index of the group by name '''
        if name is None:
            return 0
        try:
            idx = self.colnames.index(name)
        except ValueError:
            idx = self.colvals.index(name)
        return idx

    def group_acorr(self, name) -> dataset.AutoCorrelationResult:
        ''' Get autocorrelation for the group by name '''
        idx = self.groupidx(name)
        return self.autocorrelation[idx]

    def group(self, name) -> dataset.GroupResult:
        ''' Get stats for one group '''
        if name is None:
            name = self.colnames[0]
        idx = self.groupidx(name)
        return dataset.GroupResult(
            name,
            self.colvals[idx],
            self.data[idx],
            self.groups.means[idx],
            self.groups.variances[idx],
            self.groups.std_devs[idx],
            self.groups.std_errs[idx],
            self.groups.counts[idx],
            self.groups.degfs[idx],
            self.autocorrelation[idx] if self.autocorrelation is not None else None)


class DataSet:
    ''' Data Set Model '''
    def __init__(self, data=None, column_names=None):
        if data is None:
            data = [[]]
        self.data = np.atleast_2d(data)
        if column_names is None:
            self.colnames = list(range(1, max(1, len(self.data)+1)))
        else:
            self.colnames = column_names
        self._result = None

    @property
    def result(self):
        ''' Results of DataSet statistics '''
        if self._result is None:
            self.calculate()
        return self._result

    @property
    def colnames(self):
        ''' Get column names '''
        return self._colnames

    def get_column(self, name=None):
        ''' Get column data by name '''
        if name is None:
            idx = 0
        else:
            idx = self.colnames.index(name)
        return self.data[idx]

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

    def clear(self):
        ''' Clear the data '''
        self.colnames = []
        self.data = np.atleast_2d([[]])

    def calculate(self):
        ''' Calculate all the statistics '''
        pooled = dataset.pooled_stats(self.data)
        groups = dataset.group_stats(self.data)
        anova = dataset.anova(self.data)
        stderr = dataset.standarderror(self.data)
        autocorr = [dataset.uncert_autocorrelated(column) for column in self.data]
        self._result = DataSetResult(
            self.data,
            self._colnames,
            self._pcolnames,
            groups=groups,
            pooled=pooled,
            uncertainty=stderr,
            anova=anova,
            autocorrelation=autocorr,
            correlation=np.corrcoef(self.data))
        return self._result

    def summarize(self) -> 'DataSetSummary':
        ''' Convert the DataSet into an DataSetSummary object by finding
            mean and standard deviation of each group.
        '''
        gstats = self.result.groups
        return DataSetSummary(gstats.means,
                              gstats.std_devs,
                              nmeas=gstats.counts,
                              column_names=self.colnames)

    def to_array(self) -> dict[str, list[float]]:
        ''' Summarize the DataSet as x, y, uy values '''
        gstats = self.result.groups
        return {
            'x': self._pcolnames,
            'y': gstats.means,
            'u(y)': gstats.std_errs}


class DataSetSummary:
    ''' Dataset given by 3 rows: mean, stdev, and N for each group.

        Args:
            means (array): group mean values
            stds (array): group standard deviations
            nmeas (array): number of measurements in each group (degf+1)
            column_names (list): column/group names
    '''

    def __init__(self, means, stdevs, nmeas, column_names=None):
        self.means = np.asarray(means)
        self.stdevs = np.asarray(stdevs)
        self.nmeas = np.asarray(nmeas)
        if column_names is None:
            self.colnames = list(range(len(self.means)))
        else:
            self.colnames = column_names
        self._result = None

    @property
    def result(self):
        ''' Results of DataSet statistics '''
        if self._result is None:
            self.calculate()
        return self._result

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

    def calculate(self) -> DataSetResult:
        ''' Calculate statistics for the summarized values '''
        pooled = self._pooled_stats()
        groups = self._group_stats()
        anova = self._anova()
        stderr = self._standarderror()
        self._result = DataSetResult(
            np.array([[]]),
            self._colnames,
            self._pcolnames,
            groups=groups,
            pooled=pooled,
            uncertainty=stderr,
            anova=anova,
            autocorrelation=None,
            correlation=None)
        return self._result

    def _group_stats(self) -> dataset.GroupSummary:
        ''' Get statistics for all groups '''
        variance = self.stdevs**2
        sems = self.stdevs/np.sqrt(self.nmeas)
        degf = self.nmeas - 1
        return dataset.GroupSummary(
            means=self.means,
            counts=self.nmeas,
            variances=variance,
            std_devs=self.stdevs,
            std_errs=sems,
            degfs=degf)

    def _pooled_stats(self) -> dataset.PooledResult:
        ''' Pooled standard deviation '''
        ngroups = len(self.means)
        degfs = self.nmeas - 1
        variances = self.stdevs**2
        pooled_var = sum(variances * degfs) / sum(degfs)
        reprod_std = np.std(self.means, ddof=1)  # Standard deviation of group means, s(xj)
        reprod_df = ngroups - 1
        repeat_std = np.sqrt(pooled_var)  # Pooled standard deviation
        repeat_df = sum(degfs)
        return dataset.PooledResult(
            np.mean(self.means),
            reprod_std,
            repeat_std,
            reprod_df,
            repeat_df)

    def _standarderror(self, conf=.95) -> dataset.StandardErrorResult:
        ''' Compute standard error of the mean of the summarized data.
            Checks whether reproducibility is significant using ANOVA
            with conf confidnece in F-test.
        '''
        pstats = self._pooled_stats()
        anova_result = self._anova(conf=conf)
        ngroups = len(self.means)
        ntotal = sum(self.nmeas)

        if anova_result.f > anova_result.fcrit:
            # Reproducibility is significant
            sem = pstats.reproducibility / np.sqrt(ngroups)
            sem_degf = ngroups - 1
            reprod_significant = True
            sem_std = pstats.reproducibility  # Standard deviation used to compute SEM
        else:
            # Reproducibility negligible
            sem_degf = anova_result.degf_msbet + anova_result.degf_mswit
            sem_std = (anova_result.degf_msbet*anova_result.mean_sumsq_between +
                       anova_result.degf_mswit*anova_result.mean_sumsq_within)/sem_degf
            sem = np.sqrt(sem_std/ntotal)
            reprod_significant = False
        return dataset.StandardErrorResult(
            sem,
            sem_std,
            sem_degf,
            reprod_significant)

    def _anova(self, conf=.95) -> dataset.AnovaResult:
        ''' Analysis of Variance (one-way) for summarized statistics '''
        gstats = self._group_stats()
        ncolumns = len(gstats.means)
        N = sum(gstats.counts)
        mean = np.mean(gstats.means)
        sb2 = sum(gstats.variances * gstats.degfs) / sum(gstats.degfs)   # Pooled variance
        degf_b = sum(gstats.degfs)
        degf_a = ncolumns - 1
        sa2 = sum((gstats.degfs+1)*(mean - gstats.means)**2) / degf_a
        f = sa2 / sb2
        p = stats.f.sf(f, dfn=degf_a, dfd=degf_b)
        fcrit = stats.f.ppf(q=conf, dfn=degf_a, dfd=degf_b)
        SSbetween = sa2 * (ncolumns-1)
        SSwithin = sb2 * (N-ncolumns)
        return dataset.AnovaResult(
            f,
            fcrit,
            p,
            SSbetween,
            SSwithin,
            sa2,
            sb2,
            degf_a,
            degf_b)
