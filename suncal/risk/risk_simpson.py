''' Compute risk by Simpson integration of discrete PDF points. Can be much faster than scipy.quad integration,
    but may be less accurate for certain distributions.
'''
from collections import namedtuple
import numpy as np
from scipy.integrate import simpson

from ..common import distributions


def integration_limit_infinities(dist_proc, dist_test, LL, UL):
    ''' Get suitable integration limits for plus and minus infinities.

        Args:
            dist_proc (stats.rv_frozen or distributions.Distribution):
              Distribution of possible unit under test values from process
            dist_test (stats.rv_frozen or distributions.Distribution):
              Distribution of possible test measurement values
            LL (float): Lower specification limit (absolute)
            UL (float): Upper specification limit (absolute)

        Returns:
            proc.lo: minus infinity approximation for process
            proc.hi: plus infinity approxmiation for process
            test.lo: minus infinity approximation for test
            test.hi: plus infinity approxmiation for test
    '''
    Limit = namedtuple('Limit', 'lo hi')
    Limits = namedtuple('IntegrationLimits', 'proc test')
    minus_inf_proc, plus_inf_proc = dist_proc.interval(1-1E-12)
    minus_inf_test = LL - dist_test.std() * 7
    plus_inf_test = UL + dist_test.std() * 7
    return Limits(Limit(minus_inf_proc, plus_inf_proc),
                  Limit(minus_inf_test, plus_inf_test))


def PFA(dist_proc, dist_test, LL, UL, GBL=0, GBU=0, testbias=0, N=5001):
    ''' Calculate Probability of False Accept (Consumer Risk) using
        sampled distributions and Simpson integration.

        Args:
            dist_proc (array): Sampled values from process distribution
            dist_test (array): Sampled values from test measurement distribution
            LL (float):Lower specification limit (absolute)
            UL (float): Upper specification limit (absolute)
            GBL (float): Lower guardband, as offset. Test limit is LL + GBL.
            GBU (float): Upper guardband, as offset. Test limit is UL - GBU.
            testbias (float): Bias (difference between distribution median and expected value)
              in test distribution
            N (int): Number of divisions for Simpson numerical integration

        Returns:
            PFA (float): Probability of False Accept
    '''
    limits = integration_limit_infinities(dist_proc, dist_test, LL, UL)
    test_expected = dist_test.median() - testbias
    dtest_kwds = distributions.get_distargs(dist_test)
    locorig = dtest_kwds.pop('loc', 0)
    dtest_kwds.pop('loc', None)

    def integrand(t):
        d = dist_test.dist(loc=t-(test_expected-locorig), **dtest_kwds)
        return (d.cdf(UL-GBU) - d.cdf(LL+GBL)) * dist_proc.pdf(t)

    c = 0
    if limits.proc.lo < LL:
        procvals = np.linspace(limits.proc.lo, LL, N)
        jointpdf = integrand(procvals)
        c += simpson(jointpdf, x=procvals)

    if limits.proc.hi > UL:
        procvals = np.linspace(UL, limits.proc.hi, N)
        jointpdf = integrand(procvals)
        c += simpson(jointpdf, x=procvals)

    return c


def PFR(dist_proc, dist_test, LL, UL, GBL=0, GBU=0, testbias=0, N=5001):
    ''' Calculate Probability of False Accept (Consumer Risk) using
        sampled distributions and Simpson integration.

        Args:
            dist_proc (array): Sampled values from process distribution
            dist_test (array): Sampled values from test measurement distribution
            LL (float):Lower specification limit (absolute)
            UL (float): Upper specification limit (absolute)
            GBL (float): Lower guardband, as offset. Test limit is LL + GBL.
            GBU (float): Upper guardband, as offset. Test limit is UL - GBU.
            testbias (float): Bias (difference between distribution median and expected value)
              in test distribution
            N (int): Number of divisions for Simpson numerical integration

        Returns:
            PFA (float): Probability of False Accept
    '''
    test_expected = dist_test.median() - testbias
    dtest_kwds = distributions.get_distargs(dist_test)
    locorig = dtest_kwds.pop('loc', 0)

    def integrand1(t):
        d = dist_test.dist(loc=t-(test_expected-locorig), **dtest_kwds)
        return d.cdf(LL+GBL) * dist_proc.pdf(t)

    def integrand2(t):
        d = dist_test.dist(loc=t-(test_expected-locorig), **dtest_kwds)
        return (1-d.cdf(UL-GBU)) * dist_proc.pdf(t)

    minus_inf, plus_inf = dist_proc.interval(1-1E-12)
    if not np.isfinite(LL):
        x = np.linspace(minus_inf, UL, N)
    elif not np.isfinite(UL):
        x = np.linspace(LL, plus_inf, N)
    else:
        x = np.linspace(LL, UL, N)

    c1 = simpson(integrand1(x), x=x)
    c2 = simpson(integrand2(x), x=x)
    return c1 + c2


def PFA_conditional(dist_proc, dist_test, LL, UL, GBL=0, GBU=0, testbias=0, N=5001):
    ''' Conditional probability of false accept (CPFA), sometimes denoted CFAR
        (Conditional False Accept Risk). Calculated using Simpson integration

        Args:
            dist_proc (stats.rv_frozen or distributions.Distribution):
              Distribution of possible unit under test values from process
            dist_test (stats.rv_frozen or distributions.Distribution):
              Distribution of possible test measurement values
            LL (float):Lower specification limit (absolute)
            UL (float): Upper specification limit (absolute)
            GBL (float): Lower guardband, as offset. Test limit is LL + GBL.
            GBU (float): Upper guardband, as offset. Test limit is UL - GBU.
            testbias (float): Bias (difference between distribution median and expected value)
              in test distribution
            N (int): Number of divisions for Simpson numerical integration
    '''
    # CPFA = 1 - P(IT & Accepted) / P(Accepted)
    limits = integration_limit_infinities(dist_proc, dist_test, LL, UL)
    test_expected = dist_test.median() - testbias
    dtest_kwds = distributions.get_distargs(dist_test)
    locorig = dtest_kwds.pop('loc', 0)
    dtest_kwds.pop('loc', None)

    def integrand(t):
        d = dist_test.dist(loc=t-(test_expected-locorig), **dtest_kwds)
        return (d.cdf(UL-GBU) - d.cdf(LL+GBL)) * dist_proc.pdf(t)

    minus_inf, plus_inf = dist_proc.interval(1-1E-12)
    if not np.isfinite(LL):
        intol_procvals = np.linspace(minus_inf, UL, N)
    elif not np.isfinite(UL):
        intol_procvals = np.linspace(LL, plus_inf, N)
    else:
        intol_procvals = np.linspace(LL, UL, N)

    intol_and_accepted = simpson(integrand(intol_procvals), x=intol_procvals)

    all_procvals = np.linspace(limits.proc.lo, limits.proc.hi, N)
    accepted = simpson(integrand(all_procvals), x=all_procvals)
    return 1 - intol_and_accepted / accepted
