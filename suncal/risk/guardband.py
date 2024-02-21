''' Compute guardbands to target a specific PFA '''

import logging
import numpy as np
from scipy.optimize import minimize, fsolve

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
    try:
        gb, _, ier, msg = fsolve(lambda x: PFA(dist_proc, dist_test, LL, UL,
                                               GBU=x, GBL=x, testbias=testbias)-target_PFA,
                                 x0=.1, full_output=True)
    except ValueError:
        return np.nan  # Problem solving

    if ier == 1:
        return gb[0]
    else:
        logging.warning(msg)
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
    try:
        gb, _, ier, msg = fsolve(lambda x: PFA_conditional(dist_proc, dist_test, LL, UL,
                                                           GBU=x, GBL=x, testbias=testbias)-target_PFA,
                                 x0=.1, full_output=True)
    except ValueError:
        return np.nan  # Problem solving

    if ier == 1:
        return gb[0]
    else:
        logging.warning(msg)
        return np.nan


def target_pfr(dist_proc, dist_test, LL, UL, target_pfr, testbias=0):
    ''' Calculate (symmetric) guardband required to meet a target PFR value, for
        arbitrary distributions.

        Args:
            dist_proc (stats.rv_frozen or distributions.Distribution):
              Distribution of possible unit under test values from process
            dist_test (stats.rv_frozen or distributions.Distribution):
              Distribution of possible test measurement values
            LL (float): Lower specification limit (absolute)
            UL (float): Upper specification limit (absolute)
            target_pfr (float): Probability of false accept required
            testbias (float): Bias (difference between distribution median and expected value)
              in test distribution

        Returns:
            GB (float): Guardband offset required to meet target PFA. Symmetric on upper and
              lower limits, such that lower test limit is LL+GB and upper test limit is UL-GB.

        Notes:
            Uses Brent's Method to find zero of PFR(dist_proc, dist_test, LL, UL, GBU=x, GBL=x)-target_PFA.
    '''
    try:
        gb, _, ier, msg = fsolve(lambda x: PFR(dist_proc, dist_test, LL, UL,
                                               GBU=x, GBL=x, testbias=testbias)-target_pfr,
                                 x0=.1, full_output=True)

    except ValueError:
        return np.nan  # Problem solving

    if ier == 1:
        return gb[0]
    else:
        logging.warning(msg)
        return np.nan


def specific(dtest, LL, UL, target, testbias=0):
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
    expected = dtest.median()
    locorig = kwds.pop('loc', 0)
    nom = expected-locorig+testbias
    GBL = GBU = 0
    if np.isfinite(UL):
        AU, _, ier, msg = fsolve(
            lambda x: specific_risk(dtest.dist(loc=x-nom, **kwds), LL=-np.inf, UL=UL).upper-target,
            x0=UL-abs(UL)/20, full_output=True)
        if ier == 1:
            AU = AU[0]
            GBU = UL-AU
        else:
            logging.warning(msg)

    if (testbias == 0 and np.isfinite(LL) and np.isfinite(UL) and
            (dtest.median() == dtest.mean())):
        GBL = GBU  # Symmetric distribution, save some time

    elif np.isfinite(LL):
        AL, _, ier, msg = fsolve(
            lambda x: specific_risk(dtest.dist(loc=x-nom, **kwds), LL=LL, UL=np.inf).lower-target,
            x0=LL+(abs(LL)/20), full_output=True)
        if ier == 1:
            AL = AL[0]
            GBL = AL-LL
        else:
            logging.warning(msg)

    return GBL, GBU


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
    _PFA = PFA if not conditional else PFA_conditional

    def _pfr(gbl):
        # Find GBU that acheives target PFA at this GBL
        try:
            gbu = fsolve(lambda x: _PFA(dproc, dtest, LL, UL, gbl, GBU=x)-target, x0=.1)[0]
        except ValueError:  # probably PFA already < target
            return 1
        # return PFR at this gbl/gbu
        return PFR(dproc, dtest, LL, UL, gbl, gbu)

    gbl_result = minimize(_pfr, x0=.1)
    if not gbl_result.success:
        gbl = 0
    else:
        gbl = gbl_result.x[0]

    try:
        gbu = fsolve(lambda x: _PFA(dproc, dtest, LL, UL, gbl, GBU=x)-target, x0=.1)[0]
    except ValueError:
        gbu = 0.

    if not allow_negative:
        gbl = max(gbl, 0)
        gbu = max(gbu, 0)

    return gbl, gbu


def mincost(dproc, dmeas, cost_pfa, cost_pfr, LL=-1, UL=1):
    ''' Guardband by minimizing total expected cost, where

        cost = cost_pfa * pfa + cost_pfr * pfr

        solve for symmetric guardband limits that minimize the
        cost function.

        Args:
            dproc (stats.rv_frozen or distributions.Distribution):
              Distribution of possible unit under test values from process
            dmeas (stats.rv_frozen or distributions.Distribution):
              Distribution of possible test measurement values
            cost_pfa (float): Cost of a false accept
            cost_pfr (float): Cost of a false reject
            LL (float): Lower specification limit (absolute)
            UL (float): Upper specification limit (absolute)

        Returns:
            Guardband GB as offsets from tolerance, so A = UL - GBU
            and A = LL + GBL

        See also:
            suncal.risk.guardband_tur.mincost for symmetric/normal systems
            based on TUR and ITP.
    '''
    def total_cost(GB):
        # Setting AL=AU.
        pfa = PFA(dproc, dmeas, LL, UL, GB, GB)
        pfr = PFR(dproc, dmeas, LL, UL, GB, GB)
        return cost_pfa * pfa + cost_pfr * pfr
    result = minimize(total_cost, UL*.1)
    # result.fun gives expected cost value
    if result.success:
        gbofst = result.x[0]
        gbofst = min(gbofst, (UL-LL)/2)
    else:
        gbofst = 0
    return gbofst, gbofst


def minimax(dmeas, cost_pfa, cost_pfr_u, cost_pfr_l, LL=-1, UL=1):
    ''' Guardband by minimizing the maximum possible cost,
        which occurs when the true value is on the upper or lower
        limit. Potentially different FR costs above and below. Assumes
        no probability of rejecting a DUT for measuring >UL when its
        true value is <LL, or vice versa.

        Args:
            dmeas (stats.rv_frozen or distributions.Distribution):
              Distribution of possible test measurement values
            cost_pfa (float): Cost of a false accept
            cost_pfr_u (float): Cost of a false reject occuring above
                the upper limit
            cost_pfr_l (float): Cost of a false reject occuring below
                the lower limit
            LL (float): Lower specification limit (absolute)
            UL (float): Upper specification limit (absolute)

        Returns:
            Guardband GB as offsets from tolerance, so A = UL - GBU
            and A = LL + GBL

        See also:
            suncal.risk.guardband_tur.mincost for symmetric/normal systems
            based on TUR and ITP.
    '''
    kwds = distributions.get_distargs(dmeas)

    # assume all params are keyword arguments
    if len(dmeas.args) > 0:
        raise ValueError

    # Shift dmeas to TU and TL
    expected = dmeas.median()
    locorg = kwds.pop('loc', 0)
    _dmeasU = dmeas.dist(loc=UL-(expected-locorg), **kwds)
    _dmeasL = dmeas.dist(loc=LL-(expected-locorg), **kwds)

    # Minimax on top side
    def mmx_top(GB):
        AU = UL-GB
        return _dmeasU.cdf(AU) - cost_pfr_u / (cost_pfr_u + cost_pfa)

    # Minimax on bottom side
    def mmx_bot(GB):
        AL = LL+GB
        return (1-_dmeasL.cdf(AL)) - cost_pfr_l / (cost_pfr_l + cost_pfa)

    # For normal dist, will be same.
    gb_top = fsolve(mmx_top, x0=.1)[0]
    gb_bot = fsolve(mmx_bot, x0=.1)[0]
    return gb_bot, gb_top
