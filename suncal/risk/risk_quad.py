''' False accept/reject risk calculations using scipy.quad integration. Slower, but
    potentially more accurate than the Simpson integration method for some distribution types.
'''

import numpy as np
from scipy.integrate import quad, dblquad

from ..common import distributions
from .risk_simpson import integration_limit_infinities


def PFA(dist_proc, dist_test, LL, UL, GBL=0, GBU=0, testbias=0):
    ''' Calculate unconditional global Probability of False Accept for arbitrary
        process and test distributions. Uses scipy.quad integration.

        Probability a DUT is OOT and Accepted.

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

        Returns:
            PFA (float): Probability of False Accept
    '''
    test_expected = dist_test.median() - testbias
    dtest_kwds = distributions.get_distargs(dist_test)
    locorig = dtest_kwds.pop('loc', 0)

    limits = integration_limit_infinities(dist_proc, dist_test, LL, UL)

    # Use CDF() to compute the inner integral, just need one quad()
    def integrand(t):
        d = dist_test.dist(loc=t-(test_expected-locorig), **dtest_kwds)
        return (d.cdf(UL-GBU) - d.cdf(LL+GBL)) * dist_proc.pdf(t)

    c1, _ = quad(integrand, limits.proc.lo, LL, epsabs=.001, epsrel=.001)
    if (dist_test.mean() == dist_test.median()
        and dist_proc.mean() == dist_proc.median()
        and np.isclose(dist_proc.mean(), (UL+LL)/2)):
            # Integration is symmetric
            c2 = c1
    else:
        c2, _ = quad(integrand, UL, limits.proc.hi, epsabs=.001, epsrel=.001)

    return c1 + c2


def PFR(dist_proc, dist_test, LL, UL, GBL=0, GBU=0, testbias=0):
    ''' Calculate unconditional global Probability of False Accept for arbitrary
        process and test distributions. Uses scipy.quad integration.

        Probability a DUT is OOT and Accepted.

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

    c1, _ = quad(integrand1, LL, UL, epsabs=.001, epsrel=.001)
    c2, _ = quad(integrand2, LL, UL, epsabs=.001, epsrel=.001)
    return c1 + c2


def PFA_dblquad(dist_proc, dist_test, LL, UL, GBL=0, GBU=0, testbias=0):
    ''' Calculate PFA using scipy.dqlquad instead of using distribution CDF
        to perform one of the integrals.
    '''
    # Strip loc keyword from test distribution so it can be changed,
    # but shift loc so the median (expected) value starts at the spec limit.
    test_expected = dist_test.median() - testbias
    kwds = distributions.get_distargs(dist_test)
    locorig = kwds.pop('loc', 0)
    def integrand(y, t):
        return dist_test.dist.pdf(y, loc=t-(test_expected-locorig), **kwds) * dist_proc.pdf(y)

    limits = integration_limit_infinities(dist_proc, dist_test, LL, UL)
    c1, _ = dblquad(integrand, LL+GBL, UL-GBU, gfun=UL, hfun=limits.proc.hi)
    c2, _ = dblquad(integrand, LL+GBL, UL-GBU, gfun=limits.proc.lo, hfun=LL)
    return c1 + c2


def PFR_dblquad(dist_proc, dist_test, LL, UL, GBL=0, GBU=0, testbias=0):
    ''' Calculate PFR using scipy.dqlquad instead of using distribution CDF
        to perform one of the integrals.
    '''
    # Strip loc keyword from test distribution so it can be changed,
    # but shift loc so the median (expected) value starts at the spec limit.
    expected = dist_test.median() - testbias
    kwds = distributions.get_distargs(dist_test)
    locorig = kwds.pop('loc', 0)

    def integrand(y, t):
        return dist_test.dist.pdf(y, loc=t-(expected-locorig), **kwds) * dist_proc.pdf(y)

    limits = integration_limit_infinities(dist_proc, dist_test, LL, UL)
    p1, _ = dblquad(integrand, UL-GBU, limits.test.hi, gfun=LL, hfun=UL)
    p2, _ = dblquad(integrand, limits.test.lo, LL+GBL, gfun=LL, hfun=UL)
    return p1 + p2
