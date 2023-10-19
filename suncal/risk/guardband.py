''' Compute guardbands to target a specific PFA '''

from contextlib import suppress
import numpy as np
from scipy import stats
from scipy.optimize import brentq, minimize_scalar

from .risk import PFA, PFA_conditional, PFR, specific_risk
from ..common import distributions


def target(dist_proc, dist_test, LL, UL, target_PFA, testbias=0):
    ''' Calculate (symmetric) guardband required to meet a target PFA value, for
        arbitrary distributions.

        Args:
            dist_proc (stats.rv_frozen or distributions.Distribution):
              Distribution of possible unit under test values from process
            dist_test (stats.rv_frozen or distributions.Distribution):
              Distribution of possible test measurement values
            LL (float): Lower specification limit (absolute)
            UL (float): Upper specification limit (absolute)
            target_PFA (float): Probability of false accept required
            testbias (float): Bias (difference between distribution median and expected value)
              in test distribution

        Returns:
            GB (float): Guardband offset required to meet target PFA. Symmetric on upper and
              lower limits, such that lower test limit is LL+GB and upper test limit is UL-GB.

        Notes:
            Uses Brent's Method to find zero of PFA(dist_proc, dist_test, LL, UL, GBU=x, GBL=x)-target_PFA.
    '''
    w = UL-(LL+UL)/2
    if not np.isfinite(w):
        w = np.nanmax([x for x in [abs(LL), abs(UL), max(dist_proc.std()*4, dist_test.std()*4)] if np.isfinite(x)])

    try:
        gb, r = brentq(lambda x: PFA(dist_proc, dist_test, LL, UL,
                                     GBU=x, GBL=x, testbias=testbias)-target_PFA,
                       a=-w/2, b=w/2, full_output=True)
    except ValueError:
        return np.nan  # Problem solving

    if r.converged:
        return gb
    else:
        return np.nan


def target_conditional(dist_proc, dist_test, LL, UL, target_PFA, testbias=0):
    ''' Calculate (symmetric) guardband required to meet a target Conditional PFA
        value, for arbitrary distributions.

        Args:
            dist_proc (stats.rv_frozen or distributions.Distribution):
              Distribution of possible unit under test values from process
            dist_test (stats.rv_frozen or distributions.Distribution):
              Distribution of possible test measurement values
            LL (float): Lower specification limit (absolute)
            UL (float): Upper specification limit (absolute)
            target_PFA (float): Probability of false accept required
            testbias (float): Bias (difference between distribution median and expected value)
              in test distribution

        Returns:
            GB (float): Guardband offset required to meet target PFA. Symmetric on upper and
              lower limits, such that lower test limit is LL+GB and upper test limit is UL-GB.

        Notes:
            Uses Brent's Method to find zero of PFA(dist_proc, dist_test, LL, UL, GBU=x, GBL=x)-target_PFA.
    '''
    w = UL-(LL+UL)/2
    if not np.isfinite(w):
        w = np.nanmax([x for x in [abs(LL), abs(UL), max(dist_proc.std()*4, dist_test.std()*4)] if np.isfinite(x)])

    try:
        gb, r = brentq(lambda x: PFA_conditional(dist_proc, dist_test, LL, UL,
                                                 GBU=x, GBL=x, testbias=testbias)-target_PFA,
                       a=-w/2, b=w/2, full_output=True)
    except ValueError:
        return np.nan  # Problem solving

    if r.converged:
        return gb
    else:
        return np.nan


def specific(dtest, LL, UL, target):
    ''' Calculate guardband based on maximum specific risk

        Args:
            dtest (stats.frozen or distributions.Distribution): Test measurement distribution
            LL (float): Lower specification limit
            UL (float): Upper specification limit
            target (float): Target maximum specific risk

        Returns:
            GBL: Lower guardband limit
            GBU: Upper guardband limit
    '''
    kwds = distributions.get_distargs(dtest)
    w = (UL-LL)
    xx = np.linspace(LL-w/2, UL+w/2, num=500)
    if not np.isfinite(w):
        w = dtest.std() * 8
        xx = np.linspace(dtest.mean()-w if not np.isfinite(LL) else LL-w/2,
                         dtest.mean()+w if not np.isfinite(UL) else UL+w/2,
                         num=500)

    fa_lower = np.empty(len(xx))
    fa_upper = np.empty(len(xx))
    for i, loc in enumerate(xx):
        kwds.update({'loc': loc})
        dtestswp = dtest.dist(**kwds)
        fa_lower[i] = specific_risk(dtestswp, LL=LL, UL=np.inf).total
        fa_upper[i] = specific_risk(dtestswp, LL=-np.inf, UL=UL).total
    fa = fa_lower + fa_upper

    GBL = np.nan
    GBU = np.nan
    with suppress(IndexError):
        GBL = xx[np.where(fa <= target)[0][0]]
    with suppress(IndexError):
        GBU = xx[np.where(fa <= target)[0][-1]]

    return GBL-LL, UL-GBU


def guardbandfactor_to_offset(gbf, LL, UL):
    ''' Convert guardband factor into offset from spec limits

        Args:
            gbf: guardband factor (0-1)
            LL: Lower limit
            UL: Upper limit

        Returns:
            Offset from specification limit (symmetric)
    '''
    return (UL-LL)/2 * (1-gbf)


def optimize(dproc, dtest, LL, UL, target, allow_negative=False, conditional=False):
    ''' Compute (possibly) asymmetric guardband limits that achieve the
        target PFA while minimizing PFR

        Args:
            dtest (stats.frozen or distributions.Distribution): Test measurement distribution
            LL (float): Lower specification limit
            UL (float): Upper specification limit
            target (float): Target maximum specific risk
            allow_negative (bool): Allow negative guardbands (accepting
                product outside the limits)
            conditional (bool): Target is a conditional PFA

        Returns:
            gbl: lower guardband, as offset from LL
            gbu: upper guardband, as offset from UL

        Acceptance limits are LL+gbl, UL-gbu.
    '''
    if allow_negative:
        bounds = -(UL-LL)/2, (UL-LL)/2
    else:
        bounds = 0, (UL-LL)/2

    _PFA = PFA if not conditional else PFA_conditional

    def _pfr(gbl):
        # Find GBU that acheives target PFA at this GBL
        try:
            gbu = brentq(lambda x: _PFA(dproc, dtest, LL, UL, gbl, GBU=x)-target, *bounds)
        except ValueError:  # probably PFA already < target
            return 1
        # return PFR at this gbl/gbu
        return PFR(dproc, dtest, LL, UL, gbl, gbu)

    gbl_result = minimize_scalar(_pfr, bounds=bounds)
    if not gbl_result.success:
        print(gbl_result)
        gbl = 0
    else:
        gbl = gbl_result.x

    try:
        gbu = brentq(lambda x: _PFA(dproc, dtest, LL, UL, gbl, GBU=x)-target, *bounds)
    except ValueError:
        gbu = 0.

    return gbl, gbu
