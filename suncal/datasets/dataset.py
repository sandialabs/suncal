''' Functions for analysis of variance and converting 1D or 2D data into uncertainties

    Reference: GUM Appendix H.5
'''
from typing import Sequence
from dataclasses import dataclass

import numpy as np
from scipy import stats

from ..common import ttable


@dataclass
class AutoCorrelationResult:
    ''' Results of Autocorrelation calculation

    Attributes:
        rho: Autocorrelation vs lag array. Same length as x.
        uncert: Standard uncertainty accounting for autocorrelation
        r: Multiplier for converting variance into autocorrelation-corrected variance
        r_unc: Multiplier for converting uncertainty into
            autocorrelation-corrected uncertainty
        nc: Cut-off lag
    '''
    rho: Sequence[float]
    uncert: float
    r: float
    r_unc: float
    nc: int


@dataclass
class GroupResult:
    ''' Stats for one column

    Attributes:
        name: Name of the column/group
        column: Parsed name of the column/group
        values: Data values
        mean: Mean value
        variance: Variance
        std_dev: Standard Deviation
        std_err: Standard Error of the Meean
        num_meas: Number of measurements in the group
        degf: Degrees of Freedom
        autocorrelation: Autocorrelation info
    '''
    name: str  # String name
    column: float  # Parsed name
    values: list[float]
    mean: float
    variance: float
    std_dev: float
    std_err: float
    num_meas: float
    degf: float
    autocorrelation: AutoCorrelationResult

    def histogram(self, bins):
        ''' Get histogram of the column '''
        return np.histogram(self.values, bins=bins)


@dataclass
class GroupSummary:
    ''' Stats for all groups

        Attributes:
            means: Mean of each group
            counts: Number of (non-NaN) points in each group
            variances: Variance of each group
            std_devs: Standard deviation of each group
            std_errs: Standard error of the mean of each group
            degfs: Degrees of freedom of each group
    '''
    means: Sequence[float]
    counts: Sequence[float]
    variances: Sequence[float]
    std_devs: Sequence[float]
    std_errs: Sequence[float]
    degfs: Sequence[float]


@dataclass
class PooledResult:
    ''' Pooled Statistics of 2D data set

        Attributes:
            mean: Grand mean of the data
            reproducibility: Reproducibility (stdev of group means)
            repeatability: Repeatability (pooled standard deviation)
            reprod_degf: Degrees of freedom of reproducibility
            repeat_degf: Degrees of freedom of repeatability
    '''
    mean: float
    reproducibility: float
    repeatability: float
    reprod_degf: float
    repeat_degf: float


@dataclass
class AnovaResult:
    ''' Analysis of Variance Result

        Attributes:
            f: F-statistic
            fcrit: Critical F value.
            p: P statistic
            sumsq_between: Sum-of-squares of between-group variation
            sumsq_within: Sum-of-squares of within-group variation
            mean_sumsq_between: Mean sum-of-squares between-group variation
            mean_sumsq_within: Mean sum-of-squares within-group variation
            degf_msbet: Degrees of freedom of mean_sumsq_between
            degf_mswit: Degrees of freedom of mean_sumsq_within
    '''
    f: float
    fcrit: float
    p: float
    sumsq_between: float
    sumsq_within: float
    mean_sumsq_between: float  # sa2
    mean_sumsq_within: float  # sb2
    degf_msbet: float
    degf_mswit: float


@dataclass
class StandardErrorResult:
    ''' Estimate of uncertainty/standard error. Considers repeatability
        and reproducibility (if significant).

        Attributes:
            stderr: Standard error of the mean of the data
            stdev: Standard deviation of the data
            stderr_degf: Degrees of freedom of the standard error estimate
            reprod_significant: If reproducibility between groups is
                statistically significant (F > Fcritical) and was
                used to determine standard error.
    '''
    stderr: float
    stdev: float
    stderr_degf: float
    reprod_significant: bool


def autocorrelation(x: Sequence[float]) -> list[float]:
    ''' Calculate autocorrelation

        Args:
            x (array): Data with possible autocorrelation

        Returns:
            rho (array): Autocorrelation vs lag array. Same length as x.

        Notes:
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


def uncert_autocorrelated(x: Sequence[float], conf: float = .95) -> AutoCorrelationResult:
    ''' Calculate standard uncertainty in x accounting for autocorrelation.

        Args:
            x (array): Sampled data
            conf (float): Confidence (0-1) for finding nc cutoff lag

        Reference:
            Zhang, Metrologia 43, S276.
    '''
    n = len(x)
    if n > 3:
        rho = autocorrelation(x)
        sigr = sigma_rhok(rho)

        # Limit lag to be in 95% limits
        k = ttable.k_factor(conf, n)
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
        rho = np.nan
        unc = np.nan
        r = np.nan
        nc = np.nan
    return AutoCorrelationResult(rho, unc, r, np.sqrt(r), nc)


def sigma_rhok(rho: Sequence[float]) -> Sequence[float]:
    ''' Calculate sigma_rho parameter used for autocorrelation confidence band. '''
    # Eq. 14 in Zhang
    n = len(rho)
    if n == 0:
        return np.array([0])
    _sigma_rhok = np.zeros(n)
    for k in range(n):
        _sigma_rhok[k] = np.sqrt((1 + 2 * sum(rho[1:k+1]**2))/n)
    return _sigma_rhok


def anova(data: Sequence[Sequence[float]],
          conf: float = 0.95) -> AnovaResult:
    ''' Calculate one-way analysis of variance

        Args:
            data: Data array (2D). May contain NaNs if groups have
                different lengths
            conf: Level of confidence for ANOVA critical F value
    '''
    # NOTE: scipy.f_oneway can do this, but only with full 2D data.
    N = np.count_nonzero(np.isfinite(data))
    ncolumns = data.shape[0]
    mean = np.nanmean(data)
    gstats = group_stats(data)
    sb2 = sum(gstats.variances * gstats.degfs) / sum(gstats.degfs)   # Pooled variance
    degf_b = sum(gstats.degfs)
    degf_a = ncolumns - 1
    sa2 = sum((gstats.degfs+1)*(mean - gstats.means)**2) / degf_a
    f = sa2 / sb2
    p = stats.f.sf(f, dfn=degf_a, dfd=degf_b)
    fcrit = stats.f.ppf(q=conf, dfn=degf_a, dfd=degf_b)
    ss_between = sa2 * (ncolumns-1)
    ss_within = sb2 * (N-ncolumns)
    return AnovaResult(
        f,
        fcrit,
        p,
        sumsq_between=ss_between,
        sumsq_within=ss_within,
        mean_sumsq_between=sa2,
        mean_sumsq_within=sb2,
        degf_msbet=degf_a,
        degf_mswit=degf_b)


def group_stats(data: Sequence[Sequence[float]]) -> GroupSummary:
    ''' Caluclate statistics for each group (column) in the 2D data

        Args:
            data (array): 2D data to compute stats. May contain NaNs if groups
              have different sizes.
    '''
    return GroupSummary(
        means=np.nanmean(data, axis=1),
        counts=(count := np.count_nonzero(np.isfinite(data), axis=1)),
        variances=(var := np.nanvar(data, axis=1, ddof=1)),
        std_devs=(std := np.sqrt(var)),
        std_errs=std / np.sqrt(count),
        degfs=count-1)


def pooled_stats(data: Sequence[Sequence[float]]) -> PooledResult:
    ''' Calculate pooled standard dev., variance, standard error for the 2D data

        Args:
            data (array): 2D data to compute stats. May contain NaNs if groups
              have different sizes.
    '''
    groupmeans = np.nanmean(data, axis=1)
    groupvars = np.nanvar(data, axis=1, ddof=1)
    degfs = np.count_nonzero(np.isfinite(data), axis=1) - 1
    pooled_var = sum(groupvars * degfs) / sum(degfs)   # Pooled variance

    ngroups = data.shape[0]
    if ngroups > 1:
        reprod_std = np.std(groupmeans, ddof=1)  # Standard deviation of group means, s(xj)
        reprod_df = ngroups - 1
        repeat_std = np.sqrt(pooled_var)  # Pooled standard deviation
        repeat_df = sum(degfs)
    else:
        reprod_std = np.nan
        reprod_df = 0
        repeat_std = np.nan
        repeat_df = 0
    return PooledResult(
        np.nanmean(groupmeans),
        reprod_std,
        repeat_std,
        reprod_df,
        repeat_df)


def standarderror(data: Sequence[Sequence[float]], conf: float = 0.95) -> StandardErrorResult:
    ''' Compute standard error of the mean of 2D data. Checks whether reproducibility is
        significant using ANOVA F-test.

        Args:
            data (array): 2D data to compute stats. May contain NaNs if groups
              have different sizes.
            conf (float): Confidence for ANOVA Critical F value
    '''
    anova_result = anova(data, conf=conf)
    ngroups = data.shape[0]

    if anova_result.f > anova_result.fcrit:
        # Reproducibility is significant
        pstats = pooled_stats(data)
        sem = pstats.reproducibility / np.sqrt(ngroups)
        sem_degf = ngroups - 1
        reprod_significant = True
        sem_std = pstats.reproducibility  # Standard deviation used to compute SEM
    else:
        # Reproducibility negligible
        N = np.count_nonzero(np.isfinite(data))
        sem_degf = anova_result.degf_msbet + anova_result.degf_mswit
        sem_var = (anova_result.degf_msbet*anova_result.mean_sumsq_between +
                   anova_result.degf_mswit*anova_result.mean_sumsq_within)/sem_degf
        sem = np.sqrt(sem_var/N)
        sem_std = np.sqrt(sem_var)
        reprod_significant = False

    return StandardErrorResult(
        sem,
        sem_std,
        sem_degf,
        reprod_significant)
