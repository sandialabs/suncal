''' Functions for analysis of variance and converting 1D or 2D data into uncertainties

    Reference: GUM Appendix H.5
'''

from collections import namedtuple

import numpy as np
from scipy import stats

from ..common import ttable


def autocorrelation(x):
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


def uncert_autocorrelated(x, conf=.95):
    ''' Calculate standard uncertainty in x accounting for autocorrelation.

        Args:
            x (array): Sampled data
            conf (float): Confidence (0-1) for finding nc cutoff lag

        Returns:
            uncert (float): Standard uncertainty accounting for autocorrelation
            r (float): Multiplier for converting variance into autocorrelation-corrected variance
            r_unc (float): Multiplier for converting uncertainty into
              autocorrelation-corrected uncertainty
            nc (int): Cut-off lag

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
        unc = np.nan
        r = np.nan
        nc = np.nan

    Result = namedtuple('AutoCorrUncert', ['uncert', 'r', 'r_unc', 'nc'])
    return Result(unc, r, np.sqrt(r), nc)


def sigma_rhok(rho):
    ''' Calculate sigma_rho parameter used for autocorrelation confidence band. '''
    # Eq. 14 in Zhang
    n = len(rho)
    if n == 0:
        return np.array([0])
    _sigma_rhok = np.zeros(n)
    for k in range(n):
        _sigma_rhok[k] = np.sqrt((1 + 2 * sum(rho[1:k+1]**2))/n)
    return _sigma_rhok


def _anova(data, conf=0.95):
    ''' One-way analysis of variance

        Args:
            conf (float): Confidence for F-Test

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

        Reference:
            GUM H.5
    '''
    N = np.count_nonzero(np.isfinite(data))
    ncolumns = data.shape[0]
    mean = np.nanmean(data)
    gstats = group_stats(data)
    sb2 = sum(gstats.variance * gstats.degf) / sum(gstats.degf)   # Pooled variance
    degf_b = sum(gstats.degf)
    degf_a = ncolumns - 1
    sa2 = sum((gstats.degf+1)*(mean - gstats.mean)**2) / degf_a
    F = sa2 / sb2
    P = stats.f.sf(F, dfn=degf_a, dfd=degf_b)
    fcrit = stats.f.ppf(q=conf, dfn=degf_a, dfd=degf_b)
    Result = namedtuple('_AnovaStats', ['mean', 'N', 'sa2', 'sb2', 'degf_a', 'degf_b',
                                        'F', 'P', 'Fcrit', 'gstats'])
    return Result(mean, N, sa2, sb2, degf_a, degf_b, F, P, fcrit, gstats)


def anova(data, conf=.95):
    ''' Analysis of Variance (one-way)

        Args:
            conf (float): Level of confidence as fraction (0-1) for critical f value

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
    # NOTE: scipy.f_oneway can do this, but only with full 2D data.
    # if data is summarized as group means/standard deviations,
    # this one will work. This version also provides intermediate (SS, MS) data output
    ngroups = data.shape[0]
    anova_result = _anova(data, conf=conf)
    MSbetween = anova_result.sa2
    MSwithin = anova_result.sb2
    SSbetween = MSbetween * (ngroups-1)
    SSwithin = MSwithin * (anova_result.N-ngroups)
    test = (anova_result.F < anova_result.Fcrit) and (anova_result.P > 0.05)
    AnovaResult = namedtuple('AnovaResult', ['F', 'P', 'Fcrit', 'reprod_significant',
                                             'SSbet', 'SSwit', 'MSbet', 'MSwit'])
    r = AnovaResult(anova_result.F, anova_result.P, anova_result.Fcrit,
                    test, SSbetween, SSwithin, MSbetween, MSwithin)
    return r


def group_stats(data):
    ''' Caluclate statistics for each group (column) in the 2D data

        Args:
            data (array): 2D data to compute stats. May contain NaNs if groups
              have different sizes.

        Returns:
            mean (array): mean of each group
            variance (array): variance of each group
            stdandarddev (array): standard deviation of each group
            standarderror (array): standard error of the mean of each group
            num_measurements (array): number of (non-NaN) points in each group
            degf (array): degrees of freedom of each group
    '''
    groupvar = np.nanvar(data, axis=1, ddof=1)
    groupstd = np.sqrt(groupvar)
    groupmean = np.nanmean(data, axis=1)
    groupN = np.count_nonzero(np.isfinite(data), axis=1)
    groupsem = groupstd / np.sqrt(groupN)
    GroupStats = namedtuple('GroupStats', ['mean', 'variance', 'standarddev',
                                           'standarderror', 'N', 'degf'])
    return GroupStats(groupmean, groupvar, groupstd, groupsem, groupN, groupN - 1)


def pooled_stats(data):
    ''' Calculate pooled standard dev., variance, standard error for the 2D data

        Args:
            data (array): 2D data to compute stats. May contain NaNs if groups
              have different sizes.

        Returns:
            mean (float): Mean of all data points
            reproducibility (float): Reproducibility standard deviation
            repeatability (float): Repeatability (pooled) standard deviation
            reproducibility_degf (float): Degrees of freedom of reproducibility
            repeatability_degf (float): Degrees of freedom of repeatability
    '''
    # single mean, std, etc. for entire data set

    groupmeans = np.nanmean(data, axis=1)
    groupvars = np.nanvar(data, axis=1, ddof=1)
    degfs = np.count_nonzero(np.isfinite(data), axis=1) - 1
    pooled_var = sum(groupvars * degfs) / sum(degfs)   # Pooled variance

    ngroups = data.shape[0]
    reprod_std = np.std(groupmeans, ddof=1)  # Standard deviation of group means, s(xj)
    reprod_df = ngroups - 1
    repeat_std = np.sqrt(pooled_var)  # Pooled standard deviation
    repeat_df = sum(degfs)
    PooledStats = namedtuple('PooledStats', ['mean', 'reproducibility', 'repeatability',
                                             'reproducibility_degf', 'repeatability_degf'])
    return PooledStats(np.nanmean(groupmeans), reprod_std, repeat_std, reprod_df, repeat_df)


def standarderror(data, conf=0.95):
    ''' Compute standard error of the mean of 2D data. Checks whether reproducibility is
        significant using ANOVA F-test.

        Args:
            data (array): 2D data to compute stats. May contain NaNs if groups
              have different sizes.
            conf (float): Confidence for ANOVA F-Test

        Returns:
            standarderror (float): Standard error of the mean of the data
            degf (float): Degrees of freedom
            standarddeviation (float): Standard deviation
            reprod_significant (bool): If reproducibility between groups is significant and was
              used to determine standard error.
    '''
    pstats = pooled_stats(data)
    anova_result = _anova(data, conf=conf)
    ngroups = data.shape[0]

    if anova_result.F > anova_result.Fcrit:
        # Reproducibility is significant
        sem = pstats.reproducibility / np.sqrt(ngroups)
        sem_degf = ngroups - 1
        reprod_significant = True
        sem_std = pstats.reproducibility  # Standard deviation used to compute SEM
    else:
        # Reproducibility negligible
        sem_degf = anova_result.degf_a + anova_result.degf_b
        sem_var = (anova_result.degf_a*anova_result.sa2 + anova_result.degf_b*anova_result.sb2)/sem_degf
        sem = np.sqrt(sem_var/anova_result.N)
        sem_std = np.sqrt(sem_var)
        reprod_significant = False

    StandardError = namedtuple('StandardError', ['standarderror', 'degf', 'standarddeviation', 'reprod_significant'])
    return StandardError(sem, sem_degf, sem_std, reprod_significant)
