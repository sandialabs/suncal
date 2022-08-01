''' Calculations for risk, including probability of false accept (PFA) or consumer risk,
    and probability of false reject (PFR) or producer risk.

    The PFA and PFR functions take arbitrary distributions and perform the false accept/
    false reject double integrals numerically. Distributions can be either frozen instances
    of scipy.stats or random samples (e.g. Monte Carlo output of a forward uncertainty
    propagation calculation). PFAR_MC will find both PFA and PFR using a Monte Carlo method.

    The functions PFA_norm and PFR_norm assume normal distributions and take
    TUR and in-tolerance-probability (itp) as inputs. The normal assumption will
    make these functions much faster than the generic PFA and PFR functions.
    The functions PFA_deaver and PFR_deaver use the equations in Deaver's "How to
    "Maintain Confidence" paper, which require specification limits in terms of
    standard deviations of the process distribution, and use a slightly different
    definition for TUR. These functions are provided for convenience when working
    with this definition.

    The guardband and guardband_norm functions can be used to determine the guardband
    required to meet a specified PFA, or apply one of the common guardband calculation
    techniques.

    The Risk and RiskOutput classes are included mainly for use with the GUI and
    project interface, and provide consistent wrappers around the other risk
    calculation functions.
'''

from collections import namedtuple
from contextlib import suppress
import numpy as np
import yaml
from scipy import stats
from matplotlib.ticker import FormatStrFormatter
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.integrate import dblquad
from scipy.optimize import brentq, fsolve

from . import report
from . import output
from . import plotting
from . import distributions


def specific_risk(dist, LL, UL):
    ''' Calculate specific risk and process capability index for the distribution.

        Parameters
        ----------
        dist: stats.rv_frozen or distributions.Distribution
            Distribution of possible unit under test values
        LL: float
            Lower specification limit
        UL: float
            Upper specification limit

        Returns
        -------
        Cpk: float
            Process capability index. Cpk > 1.333 indicates process is capable of
            meeting specifications.
        risk_total: float
            Total risk (0-1 range) of nonconformance
        risk_lower: float
            Risk of nonconformance below LL
        risk_upper: float
            Risk of nonconformance above UL

        Notes
        -----
        Normal distributions use the standard definition for cpk:

            min( (UL - x)/(3 sigma), (x - LL)/(3 sigma) )

        Non-normal distributions use the proportion nonconforming:

            min( norm.ppf(risk_lower)/3, norm.ppf(risk_upper)/3 )

        (See https://www.qualitydigest.com/inside/quality-insider-article/process-performance-indices-nonnormal-distributions.html)
    '''
    LL, UL = min(LL, UL), max(LL, UL)  # make sure LL < UL
    risk_lower = dist.cdf(LL)
    risk_upper = 1 - dist.cdf(UL)
    risk_total = risk_lower + risk_upper
    if hasattr(dist, 'dist') and hasattr(dist.dist, 'name') and dist.dist.name == 'norm':
        # Normal distributions can use the standard definition of cpk, process capability index
        cpk = min((UL-dist.mean())/(3*dist.std()), (dist.mean()-LL)/(3*dist.std()))
    else:
        # Non-normal distributions use fractions out.
        # See https://www.qualitydigest.com/inside/quality-insider-article/process-performance-indices-nonnormal-distributions.html
        cpk = max(0, min(abs(stats.norm.ppf(risk_lower))/3, abs(stats.norm.ppf(risk_upper))/3))
        if risk_lower > .5 or risk_upper > .5:
            cpk = -cpk
    Result = namedtuple('SpecificRisk', ['cpk', 'total', 'lower', 'upper'])
    return Result(cpk, risk_total, risk_lower, risk_upper)


def guardband_norm(method, TUR, **kwargs):
    ''' Get guardband factor for the TUR (applies to normal-only risk).

        Parameters
        ----------
        method: string
            Guardband method to apply. One of: 'dobbert', 'rss',
            'rp10', 'test', '4:1', 'pfa', 'mincost', 'minimax'.
        TUR: float
            Test Uncertainty Ratio

        Keyword Arguments
        -----------------
        pfa: float (optional)
            Target PFA (for method 'pfa'. Defaults to 0.008)
        itp: float (optional)
            In-tolerance probability (for method 'pfa', '4:1', 'mincost',
            and 'minimax'. Defaults to 0.95)
        CcCp: float (optional)
            Ratio of cost of false accept to cost of part (for methods
            'mincost' and 'minimax')

        Returns
        -------
        guardband: (float)
            Guardband factor. Acceptance limit = Tolerance Limit * guardband.

        Notes
        -----
        Dobbert's method maintains <2% PFA for ANY itp at the TUR.
        RSS method: GB = sqrt(1-1/TUR**2)
        test method: GB = 1 - 1/TUR  (subtract the 95% test uncertainty)
        rp10 method: GB = 1.25 - 1/TUR (similar to test, but less conservative)
        pfa method: Solve for GB to produce desired PFA
        4:1 method: Solve for GB that results in same PFA as 4:1 at this itp
        mincost method: Minimize the total expected cost due to false decisions (Ref Easterling 1991)
        minimax method: Minimize the maximum expected cost due to false decisions (Ref Easterling 1991)
    '''
    if method == 'dobbert':
        # Dobbert Eq. 4 for Managed Guardband, maintains max PFA 2% for any itp.
        M = 1.04 - np.exp(0.38 * np.log(TUR) - 0.54)
        GB = 1 - M / TUR
    elif method == 'rss':
        # The common RSS method
        GB = np.sqrt(1-1/TUR**2)
    elif method == 'test':
        # Subtract the test uncertainty from the spec limit
        GB = 1 - 1/TUR if TUR <= 4 else 1
    elif method == 'rp10':
        # Method described in NCSLI RP-10
        GB = 1.25 - 1/TUR if TUR <= 4 else 1
    elif method in ['pfa', '4:1']:
        # Calculate guardband for specific PFA
        itp = kwargs.get('itp', 0.95)
        if method == 'pfa':
            pfa_target = kwargs.get('pfa', .008)
        else:
            pfa_target = PFA_norm(itp, TUR=4)
        # In normal case, this is faster than guardband() method
        GB = fsolve(lambda x: PFA_norm(itp, TUR, GB=x)-pfa_target, x0=.8)[0]
    elif method == 'mincost':
        itp = kwargs.get('itp', 0.95)
        Cc_over_Cp = kwargs.get('CcCp', 10)
        conf = 1 - (1 / (1 + Cc_over_Cp))
        sigtest = 1/TUR/2
        sigprod = 1/stats.norm.ppf((1+itp)/2)
        k = stats.norm.ppf(conf) * np.sqrt(1 + sigtest**2/sigprod**2) - sigtest/sigprod**2
        GB = 1 - k * sigtest
    elif method == 'minimax':
        Cc_over_Cp = kwargs.get('CcCp', 10)
        conf = 1 - (1 / (1 + Cc_over_Cp))
        k = stats.norm.ppf(conf)
        GB = 1 - k * (1/TUR/2)
    else:
        raise ValueError('Unknown guardband method {}.'.format(method))
    return GB


def guardband(dist_proc, dist_test, LL, UL, target_PFA, testbias=0, approx=False):
    ''' Calculate (symmetric) guardband required to meet a target PFA value, for
        arbitrary distributions.

        Parameters
        ----------
        dist_proc: stats.rv_frozen or distributions.Distribution
            Distribution of possible unit under test values from process
        dist_test: stats.rv_frozen or distributions.Distribution
            Distribution of possible test measurement values
        LL: float
            Lower specification limit (absolute)
        UL: float
            Upper specification limit (absolute)
        target_PFA: float
            Probability of false accept required
        testbias: float
            Bias (difference between distribution median and expected value)
            in test distribution
        approx: bool
            Approximate the integral using discrete probability distribution.
            Faster than using scipy.integrate.

        Returns
        -------
        GB: float
            Guardband offset required to meet target PFA. Symmetric on upper and
            lower limits, such that lower test limit is LL+GB and upper
            test limit is UL-GB.

        Notes
        -----
        Uses Brent's Method to find zero of PFA(dist_proc, dist_test, LL, UL, GBU=x, GBL=x)-target_PFA.
    '''
    w = UL-(LL+UL)/2
    if not np.isfinite(w):
        w = np.nanmax([x for x in [abs(LL), abs(UL), max(dist_proc.std()*4, dist_test.std()*4)] if np.isfinite(x)])

    try:
        gb, r = brentq(lambda x: PFA(dist_proc, dist_test, LL, UL, GBU=x, GBL=x, testbias=testbias, approx=approx)-target_PFA, a=-w/2, b=w/2, full_output=True)
    except ValueError:
        return np.nan  # Problem solving

    if r.converged:
        return gb
    else:
        return np.nan


def guardband_specific(dtest, LL, UL, target):
    ''' Calculate guardband based on maximum specific risk

        Parameters
        ----------
        dtest : stats.frozen or distributions.Distribution
            Test measurement distribution
        LL : float
            Lower specification limit
        UL : float
            Upper specification limit
        target : float
            Target maximum specific risk
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
    ''' Convert guardband factor into offset from spec limits '''
    return (UL-LL)/2 * (1-gbf)


def get_sigmaproc_from_itp(itp, bias=0):
    ''' Get process standard deviation from in-tolerance probability.
        Assumes normal distribution, but accounts for bias.
    '''
    if bias == 0:
        return 1/stats.norm.ppf((1+itp)/2)
    else:
        # With bias, PDF is asymetric, can't just use PPF
        
        def sp_risk(mu, sigma):
            ''' Simplified version of specific risk function '''
            rl = stats.norm(mu, sigma).cdf(-1)
            ru = 1 - stats.norm(mu, sigma).cdf(1)
            return rl + ru

        # fsolve is sensitive to initial guess x0 - try another x0 if first one doesn't work
        out = fsolve(lambda x: (1-sp_risk(bias, x))-itp, x0=.1, full_output=1)
        if out[2] != 1:
            out = fsolve(lambda x: (1-sp_risk(bias, x))-itp, x0=.8, full_output=1)
        if out[2] != 1:
            return 0
        return out[0][0]



def PFA_norm(itp, TUR, GB=1, sig0=None, biastest=0, biasproc=0, observeditp=False, **kwargs):
    ''' PFA for normal distributions in terms of TUR and
        in-tolerance probability

        Parameters
        ----------
        itp: float
            In-tolerance probability (0-1 range). A-priori distribution of
            process.
        TUR: float
            Test Uncertainty Ratio. Spec Limit / (2*Test Uncertainty)
        GB: float or string (optional)
            Guardband Factor. If GB is numeric, GB = K, where acceptance
            limit A = T * K. In Dobbert's notation, K = 1 - M/TUR where
            A = T - U*M. GB = 1 implies no guardbanding.
            If GB is a string, it can be one of options in guardband_norm
            method. kwargs passed to guardband_norm.
        sig0: float
            Process standard deviation in terms of #SL, overrides itp
        biastest: float
            Bias/shift in the test measurement distribution, in percent of SL
        biasproc: float
            Bias/shift in the process distribution, in percent of SL
        observed: bool
            Consider itp as the "observed" itp. True itp will be adjusted
            to account for measurement uncertainty in observing itp. See
            Mimbs "Using Reliability to Meet Z540.3's 2% Rule", NCSLI 2011.
    '''
    # Convert itp to stdev of process
    # This is T in equation 2 in Dobbert's Guardbanding Strategy, with T = 1.
    sigma0 = sig0 if sig0 is not None else get_sigmaproc_from_itp(itp, biasproc)
    sigmatest = 1/TUR/2

    if observeditp:
        sigma0 = np.sqrt(sigma0**2 - sigmatest**2)

    try:
        GB = float(GB)
    except ValueError:  # String
        GB = guardband_norm(GB, TUR, itp=itp, **kwargs)

    A = GB  # A = T * GB = 1 * GB
    if biastest == 0 and biasproc == 0:
        c, _ = dblquad(lambda y, t: np.exp(-(y-biasproc)**2/2/sigma0**2)*np.exp(-(y-t+biastest)**2/2/sigmatest**2),
                       -A, A, gfun=1, hfun=np.inf)
        c *= 2  # Symmetric both sides
    else:
        c1, _ = dblquad(lambda y, t: np.exp(-(y-biasproc)**2/2/sigma0**2)*np.exp(-(y-t+biastest)**2/2/sigmatest**2),
                        -A, A, gfun=1, hfun=np.inf)
        c2, _ = dblquad(lambda y, t: np.exp(-(y-biasproc)**2/2/sigma0**2)*np.exp(-(y-t+biastest)**2/2/sigmatest**2),
                        -A, A, gfun=-np.inf, hfun=-1)
        c = c1 + c2
    c = c / (2 * np.pi * sigmatest * sigma0)
    return c


def PFR_norm(itp, TUR, GB=1, sig0=None, biastest=0, biasproc=0, observeditp=False, **kwargs):
    ''' PFR for normal distributions in terms of TUR and
        in-tolerance probability

        Parameters
        ----------
        itp: float
            In-tolerance probability (0-1 range). A-priori distribution of
            process.
        TUR: float
            Test Uncertainty Ratio. Spec Limit / (2*Test Uncertainty)
        GB: float or string (optional)
            Guardband Factor. If GB is numeric, GB = K, where acceptance
            limit A = T * K. In Dobbert's notation, K = 1 - M/TUR where
            A = T - U*M. GB = 1 implies no guardbanding.
            If GB is a string, it can be one of options in guardband_norm
            method. kwargs passed to guardband_norm.
        sig0: float
            Process standard deviation in terms of #SL, overrides itp
        biastest: float
            Bias/shift in the test measurement distribution
        biasproc: float
            Bias/shift in the process distribution
        observed: bool
            Consider itp as the "observed" itp. True itp will be adjusted
            to account for measurement uncertainty in observing itp. See
            Mimbs "Using Reliability to Meet Z540.3's 2% Rule", NCSLI 2011.
    '''
    sigma0 = sig0 if sig0 is not None else get_sigmaproc_from_itp(itp, biasproc)
    sigmatest = 1/TUR/2

    if observeditp:
        sigma0 = np.sqrt(sigma0**2 - sigmatest**2)

    try:
        GB = float(GB)
    except ValueError:  # String
        GB = guardband_norm(GB, TUR, itp=itp, **kwargs)

    A = GB
    if biastest == 0 and biasproc == 0:
        c, _ = dblquad(lambda y, t: np.exp(-(y-biasproc)**2/2/sigma0**2)*np.exp(-(y-t+biastest)**2/2/sigmatest**2),
                       A, np.inf, gfun=-1, hfun=1)
        c *= 2  # Symmetric both sides
    else:
        c1, _ = dblquad(lambda y, t: np.exp(-(y-biasproc)**2/2/sigma0**2)*np.exp(-(y-t+biastest)**2/2/sigmatest**2),
                        A, np.inf, gfun=-1, hfun=1)
        c2, _ = dblquad(lambda y, t: np.exp(-(y-biasproc)**2/2/sigma0**2)*np.exp(-(y-t+biastest)**2/2/sigmatest**2),
                        -np.inf, -A, gfun=-1, hfun=1)
        c = c1 + c2
    c = c / (2 * np.pi * sigmatest * sigma0)
    return c


def PFA_deaver(SL, TUR, GB=1):
    ''' Calculate Probability of False Accept (Consumer Risk) for normal
        distributions given spec limit and TUR, using Deaver's equation.

        Parameters
        ----------
        sigma: float
            Specification Limit in terms of standard deviations, symmetric on
            each side of the mean
        TUR: float
            Test Uncertainty Ratio (sigma_uut / sigma_test). Note this is
            definition used by Deaver's papers, NOT the typical SL/(2*sigma_test) definition.
        GB: float (optional)
            Guardband factor (0-1) with 1 being no guardband

        Returns
        -------
        PFA: float
            Probability of False Accept

        Reference
        ---------
        Equation 6 in Deaver - How to Maintain Confidence
    '''
    c, _ = dblquad(lambda y, t: np.exp(-(y*y + t*t)/2) / np.pi, SL, np.inf, gfun=lambda t: -TUR*(t+SL*GB), hfun=lambda t: -TUR*(t-SL*GB))
    return c


def PFR_deaver(SL, TUR, GB=1):
    ''' Calculate Probability of False Reject (Producer Risk) for normal
        distributions given spec limit and TUR, using Deaver's equation.

        Parameters
        ----------
        SL: float
            Specification Limit in terms of standard deviations, symmetric on
            each side of the mean
        TUR: float
            Test Uncertainty Ratio (sigma_uut / sigma_test). Note this is
            definition used by Deaver's papers, NOT the typical SL/(2*sigma_test) definition.
        GB: float (optional)
            Guardband factor (0-1) with 1 being no guardband

        Returns
        -------
        PFR: float
            Probability of False Reject

        Reference
        ---------
        Equation 7 in Deaver - How to Maintain Confidence
    '''
    p, _ = dblquad(lambda y, t: np.exp(-(y*y + t*t)/2) / np.pi, -SL, SL, gfun=lambda t: TUR*(GB*SL-t), hfun=lambda t: np.inf)
    return p


def PFA(dist_proc, dist_test, LL, UL, GBL=0, GBU=0, testbias=0, approx=False):
    ''' Calculate Probability of False Accept (Consumer Risk) for arbitrary
        process and test distributions.

        Parameters
        ----------
        dist_proc: stats.rv_frozen or distributions.Distribution
            Distribution of possible unit under test values from process
        dist_test: stats.rv_frozen or distributions.Distribution
            Distribution of possible test measurement values
        LL: float
            Lower specification limit (absolute)
        UL: float
            Upper specification limit (absolute)
        GBL: float
            Lower guardband, as offset. Test limit is LL + GBL.
        GBU: float
            Upper guardband, as offset. Test limit is UL - GBU.
        testbias: float
            Bias (difference between distribution median and expected value)
            in test distribution
        approx: bool
            Approximate using discrete probability distribution. This
            uses trapz integration so it may be faster than letting
            scipy integrate the actual pdf function.

        Returns
        -------
        PFA: float
            Probability of False Accept
    '''
    if approx:
        xx = np.linspace(dist_proc.median() - dist_proc.std()*8, dist_proc.median() + dist_proc.std()*8, num=1000)
        xx2 = np.linspace(dist_test.median() - dist_test.std()*8,  dist_test.median() + dist_test.std()*8, num=1000)
        procpdf = dist_proc.pdf(xx)
        testpdf = dist_test.pdf(xx2)
        return _PFA_discrete((xx, procpdf), (xx2, testpdf), LL, UL, GBL=GBL, GBU=GBU, testbias=testbias)

    else:
        # Strip loc keyword from test distribution so it can be changed,
        # but shift loc so the median (expected) value starts at the spec limit.
        test_expected = dist_test.median() - testbias
        kwds = distributions.get_distargs(dist_test)
        locorig = kwds.pop('loc', 0)

        def integrand(y, t):
            return dist_test.dist.pdf(y, loc=t-(test_expected-locorig), **kwds) * dist_proc.pdf(y)

        c1, _ = dblquad(integrand, LL+GBL, UL-GBU, gfun=UL, hfun=np.inf)
        c2, _ = dblquad(integrand, LL+GBL, UL-GBU, gfun=-np.inf, hfun=LL)
        return c1 + c2


def _PFA_discrete(dist_proc, dist_test, LL, UL, GBL=0, GBU=0, testbias=0):
    ''' Calculate Probability of False Accept (Consumer Risk) using
        sampled distributions.

        Parameters
        ----------
        dist_proc: array
            Sampled values from process distribution
        dist_test: array
            Sampled values from test measurement distribution
        LL: float
            Lower specification limit (absolute)
        UL: float
            Upper specification limit (absolute)
        GBL: float
            Lower guardband, as offset. Test limit is LL + GBL.
        GBU: float
            Upper guardband, as offset. Test limit is UL - GBU.
        testbias: float
            Bias (difference between distribution median and expected value)
            in test distribution

        Returns
        -------
        PFA: float
            Probability of False Accept
    '''
    if isinstance(dist_proc, tuple):
        procx, procy = dist_proc
        dy = procx[1]-procx[0]
    else:
        procy, procx = np.histogram(dist_proc, bins='auto', density=True)
        dy = procx[1]-procx[0]
        procx = procx[1:] - dy/2

    if isinstance(dist_test, tuple):
        testx, testy = dist_test
        dx = testx[1]-testx[0]
    else:
        testy, testx = np.histogram(dist_test, bins='auto', density=True)
        dx = testx[1]-testx[0]
        testx = testx[1:] - dx/2

    expected = np.median(testx) - testbias
    c = 0
    for t, ut in zip(procx[np.where(procx > UL)], procy[np.where(procx > UL)]):
        idx = np.where(testx+t-expected < UL-GBU)
        c += np.trapz(ut*testy[idx], dx=dx)

    for t, ut in zip(procx[np.where(procx < LL)], procy[np.where(procx < LL)]):
        idx = np.where(testx+t-expected > LL+GBL)
        c += np.trapz(ut*testy[idx], dx=dx)

    c *= dy
    return c


def PFR(dist_proc, dist_test, LL, UL, GBL=0, GBU=0, testbias=0, approx=False):
    ''' Calculate Probability of False Reject (Producer Risk) for arbitrary
        process and test distributions.

        Parameters
        ----------
        dist_proc: stats.rv_frozen or distribution.Distribution instance
            Distribution of possible unit under test values from process
        dist_test: stats.rv_frozen or distribution.Distribution instance
            Distribution of possible test measurement values
        LL: float
            Lower specification limit (absolute)
        UL: float
            Upper specification limit (absolute)
        GBL: float
            Lower guardband, as offset. Test limit is LL + GBL.
        GBU: float
            Upper guardband, as offset. Test limit is UL - GBU.
        testbias: float
            Bias (difference between distribution median and expected value)
            in test distribution
        approx: bool
            Approximate using discrete probability distribution. This
            uses trapz integration so it may be faster than letting
            scipy integrate the actual pdf function.

        Returns
        -------
        PFR: float
            Probability of False Reject
    '''
    if approx:
        xx = np.linspace(dist_proc.median() - dist_proc.std()*8, dist_proc.median() + dist_proc.std()*8, num=1000)
        xx2 = np.linspace(dist_test.median() - dist_test.std()*8,  dist_test.median() + dist_test.std()*8, num=1000)
        procpdf = dist_proc.pdf(xx)
        testpdf = dist_test.pdf(xx2)
        return _PFR_discrete((xx, procpdf), (xx2, testpdf), LL, UL, GBL=GBL, GBU=GBU, testbias=testbias)

    else:
        # Strip loc keyword from test distribution so it can be changed,
        # but shift loc so the median (expected) value starts at the spec limit.
        expected = dist_test.median() - testbias
        kwds = distributions.get_distargs(dist_test)
        locorig = kwds.pop('loc', 0)

        def integrand(y, t):
            return dist_test.dist.pdf(y, loc=t-(expected-locorig), **kwds) * dist_proc.pdf(y)

        p1, _ = dblquad(integrand, UL-GBU, np.inf, gfun=LL, hfun=UL)
        p2, _ = dblquad(integrand, -np.inf, LL+GBL, gfun=LL, hfun=UL)
        return p1 + p2


def _PFR_discrete(dist_proc, dist_test, LL, UL, GBL=0, GBU=0, testbias=0):
    ''' Calculate Probability of False Reject (Producer Risk) using
        sampled distributions.

        Parameters
        ----------
        dist_proc: array
            Sampled values from process distribution
        dist_test: array
            Sampled values from test measurement distribution
        LL: float
            Lower specification limit (absolute)
        UL: float
            Upper specification limit (absolute)
        GBL: float
            Lower guardband, as offset. Test limit is LL + GBL.
        GBU: float
            Upper guardband, as offset. Test limit is UL - GBU.
        testbias: float
            Bias (difference between distribution median and expected value)
            in test distribution

        Returns
        -------
        PFR: float
            Probability of False Reject
    '''
    if isinstance(dist_proc, tuple):
        procx, procy = dist_proc
        dy = procx[1]-procx[0]
    else:
        procy, procx = np.histogram(dist_proc, bins='auto', density=True)
        dy = procx[1]-procx[0]
        procx = procx[1:] - dy/2

    if isinstance(dist_test, tuple):
        testx, testy = dist_test
        dx = testx[1]-testx[0]
    else:
        testy, testx = np.histogram(dist_test, bins='auto', density=True)
        dx = testx[1]-testx[0]
        testx = testx[1:] - dx/2

    expected = np.median(testx) - testbias
    c = 0
    for t, ut in zip(procx[np.where((procx > LL) & (procx < UL))], procy[np.where((procx > LL) & (procx < UL))]):
        idx = np.where(testx+t-expected > UL-GBU)
        c += np.trapz(ut*testy[idx], dx=dx)
        idx = np.where(testx+t-expected < LL+GBL)
        c += np.trapz(ut*testy[idx], dx=dx)

    c *= dy
    return c


def PFAR_MC(dist_proc, dist_test, LL, UL, GBL=0, GBU=0, N=100000, testbias=0):
    ''' Probability of False Accept/Reject using Monte Carlo Method

        dist_proc: stats.rv_frozen
            Distribution of possible unit under test values from process
        dist_test: stats.rv_frozen
            Distribution of possible test measurement values
        LL: float
            Lower specification limit (absolute)
        UL: float
            Upper specification limit (absolute)
        GBL: float
            Lower guardband, as offset. Test limit is LL + GBL.
        GBU: float
            Upper guardband, as offset. Test limit is UL - GBU.
        N: int
            Number of Monte Carlo samples
        testbias: float
            Bias (difference between distribution median and expected value)
            in test distribution

        Returns
        -------
        PFA: float
            False accept probability
        PFR: float
            False reject probability
        proc_samples: array (optional)
            Monte Carlo samples for uut
        test_samples: array (optional)
            Monte Carlo samples for test measurement
    '''
    Result = namedtuple('MCRisk', ['pfa', 'pfr', 'process_samples', 'test_samples'])
    proc_samples = dist_proc.rvs(size=N)
    expected = dist_test.median() - testbias
    kwds = distributions.get_distargs(dist_test)
    locorig = kwds.pop('loc', 0)
    try:
        # Works for normal stats distributions, but not rv_histograms
        test_samples = dist_test.dist.rvs(loc=proc_samples-(expected-locorig), size=N, **kwds)
    except TypeError:
        # Works for histograms, but not regular distributions...
        test_samples = dist_test.dist(**kwds).rvs(loc=proc_samples-(expected-locorig), size=N)
    except ValueError:
        # Invalid parameter in kwds
        test_samples = np.array([])
        return Result(np.nan, np.nan, None, None)

    FA = np.count_nonzero(((test_samples < UL-GBU) & (test_samples > LL+GBL)) & ((proc_samples > UL) | (proc_samples < LL))) / N
    FR = np.count_nonzero(((test_samples > UL-GBU) | (test_samples < LL+GBL)) & ((proc_samples < UL) & (proc_samples > LL))) / N
    return Result(FA, FR, proc_samples, test_samples)


def PFA_sweep_simple(xvar='itp', zvar='TUR', xvals=None, zvals=None,
                     GBFdflt=1, itpdflt=.95, TURdflt=4, sig0=None, tbias=0, pbias=0, risk='PFA'):
    ''' Sweep PFA vs. itp, tur, or gbf for producing risk curves in simple mode

        Parameters
        ----------
        xvar : string
            Sweep variable for x axis - 'itp', 'tur', 'gbf', 'tbias', 'pbias'
        zvar : string
            Step variable - 'itp', 'tur', 'gbf', 'tbias', 'pbias'
        xvals : array
            List of sweep values for x axis
        zvals : array
            List of step values for step/z axis
        GBFdlft : float
            Default guardband value, if gbf is not being swept
        itpdflt : float
            Default itp value, if itp is not being swept
        TURdflt : float
            Default tur value, if tur is not being swept
        sig0: float
            Process standard deviation in terms of #SL, overrides itp
        tbias : float
            Default test measurement bias
        pbias : float
            Default process distribution bias
        risk : string
            Calculate 'PFA' or 'PFR'

        Returns
        -------
        risk : array
            2D array (shape len(xvals) x len(zvals)) of risk values
    '''
    assert xvar.lower() in ['itp', 'tur', 'gbf', 'tbias', 'pbias', 'sig0']
    assert zvar.lower() in ['itp', 'tur', 'gbf', 'tbias', 'pbias', 'sig0', 'none']
    assert risk.lower() in ['pfa', 'pfr']

    if zvar == 'none':
        zvals = [None]
        xx = np.array([xvals])
        zz = np.array([])
    else:
        xx, zz = np.meshgrid(xvals, zvals)
    riskfunc = PFR_norm if risk.lower() == 'pfr' else PFA_norm

    if xvar.lower() == 'itp':
        itp = xx
    elif zvar.lower() == 'itp':
        itp = zz
    else:
        itp = np.full(xx.shape, itpdflt)

    if xvar.lower() == 'tur':
        TUR = xx
    elif zvar.lower() == 'tur':
        TUR = zz
    else:
        TUR = np.full(xx.shape, TURdflt)

    if xvar.lower() == 'gbf':
        GBF = xx
    elif zvar.lower() == 'gbf':
        GBF = zz
    else:
        GBF = np.full(xx.shape, GBFdflt)

    if xvar.lower() == 'tbias':
        tbias = xx
    elif zvar.lower() == 'tbias':
        tbias = zz
    else:
        tbias = np.full(xx.shape, tbias)

    if xvar.lower() == 'pbias':
        pbias = xx
    elif zvar.lower() == 'pbias':
        pbias = zz
    else:
        pbias = np.full(xx.shape, pbias)

    if xvar.lower() == 'sig0':
        sig0 = 1/xx
    elif zvar.lower() == 'sig0':
        sig0 = 1/zz
    else:
        sig0 = np.full(xx.shape, 1/sig0 if sig0 is not None else None)

    curves = np.empty_like(xx)
    for zidx in range(len(zvals)):
        for xidx in range(len(xvals)):
            curves[zidx, xidx] = riskfunc(itp[zidx, xidx], TUR[zidx, xidx], GBF[zidx, xidx],
                                          sig0=sig0[zidx, xidx],
                                          biastest=tbias[zidx, xidx], biasproc=pbias[zidx, xidx])
    return curves


def PFA_sweep(xvarparam, zvarparam, xvardist=None, zvardist=None, xvals=None, zvals=None,
              dist_proc=None, dist_test=None, LL=-1, UL=1, GBL=0, GBU=0, testbias=0,
              risk='PFA', approx=True):
    ''' Sweep PFA vs. any distribution parameter for producing risk curves

        Parameters
        ----------
        xvarparam : string
            Name of distribution parameter for sweep variable, or 'gb', 'gbl', 'gbu'
        zvarparam : string
            Name of distribution parameter for step variable, or 'gb', 'gbl', 'gbu'
        xvardist : Distribution object
            Distribution to change in x sweep
        zvardist : Distribution object
            Distribution to change in z step
        xvals : array
            List of sweep values for x axis
        zvals : array
            List of step values for step/z axis
        dist_proc : Distribution object
            Process distribution
        dist_test : Distribution object
            Test measurement distribution
        LL, UL : float
            Lower and upper specification limits
        GBL, GBU : float
            Lower and upper guardbands, absolute
        testbias : float
            Bias in test measurement
        risk : string
            Calculate 'PFA' or 'PFR'
        approx : bool
            Use trapezoidal approximation for integral (faster but less accurate)

        Returns
        -------
        risk : array
            2D array (shape len(xvals) x len(zvals)) of risk values
    '''
    riskfunc = PFR if risk == 'pfr' else PFA

    xx, zz = np.meshgrid(xvals, zvals)
    curves = np.empty_like(xx)
    for zidx, z in enumerate(zvals):
        if zvarparam.lower() == 'gb':
            GBL = z
            GBU = z
        elif zvarparam.lower() == 'gbl':
            GBL = z
        elif zvarparam.lower() == 'gbu':
            GBU = z
        elif zvarparam.lower() == 'bias':
            testbias = z
        else:
            zvardist.update_kwds(**{zvarparam: z})

        for xidx, x in enumerate(xvals):
            if xvarparam.lower() == 'gb':
                GBL = x
                GBU = x
            elif xvarparam.lower() == 'gbl':
                GBL = x
            elif xvarparam.lower() == 'gbu':
                GBU = x
            elif xvarparam.lower() == 'bias':
                testbias = x
            else:
                xvardist.update_kwds(**{xvarparam: x})

            curves[zidx, xidx] = riskfunc(dist_proc, dist_test, LL, UL, GBL, GBU, testbias, approx=True)
    return curves


class Risk(object):
    ''' Class incorporating risk calculations. Mostly useful for implementing the GUI.
        Risk Functions in this module should be used for manual data analysis.
    '''
    def __init__(self, name='risk'):
        self.name = name
        self.description = ''
        self.procdist = distributions.get_distribution('normal', std=.51021346)  # For 95% itp starting value
        self.testdist = distributions.get_distribution('normal', std=.125)
        self.testbias = 0              # Offset between testdist median and measurement result
        self.speclimits = (-1.0, 1.0)  # Upper/lower specification limits
        self.guardband = (0, 0)        # Guardband offset (A = speclimits - guardband)
        self.out = RiskOutput(self)
        self.cost_FA = None  # Cost of false accept
        self.cost_FR = None  # Cost of false reject

    def set_testdist(self, testdist, testbias=None):
        ''' Set the test distribution

            Parameters
            ----------
            testdist: distributions.Distribution
                Test distribution instance
        '''
        self.testdist = testdist
        if testbias is not None:
            self.testbias = testbias

    def set_procdist(self, procdist):
        ''' Set the process distribution

            Parameters
            ----------
            procdist: distributions.Distribution
                Process distribution instance
        '''
        self.procdist = procdist

    def set_speclimits(self, LL, UL):
        ''' Set specification limits

            Parameters
            ----------
            LL: float
                Lower specification limit, in absolute units
            UL: float
                Upper specification limit, in absolute units
        '''
        self.speclimits = LL, UL

    def set_gbf(self, gbf):
        ''' Set guardband factor

            Parameters
            ----------
            gbf: float
                Guardband factor as multiplier of specification
                limit. Acceptance limit A = T * gbf.
        '''
        rng = (self.speclimits[1] - self.speclimits[0])/2
        gb = rng * (1 - gbf)
        self.guardband = gb, gb

    def set_guardband(self, GBL, GBU):
        ''' Set relative guardband

            Parameters
            ----------
            GBL: float
                Lower guardband offset. Acceptance limit A = LL + GBL
            GBU: float
                Upper guardband offset. Acceptance limit A = UL - GBU
        '''
        self.guardband = GBL, GBU

    def set_itp(self, itp):
        ''' Set in-tolerance probability by adjusting process distribution
            with specification limits of +/-
            Parameters
            ----------
            itp: float
                In-tolerance probability (0-1)
        '''
        self.to_simple()
        sigma = self.speclimits[1] / stats.norm.ppf((1+itp)/2)
        self.procdist = distributions.get_distribution('normal', loc=0, std=sigma)

    def set_tur(self, tur):
        ''' Set test uncertainty ratio by adjusting test distribution

            Parameters
            ----------
            tur: float
                Test uncertainty ratio (> 0)
        '''
        self.to_simple()
        sigma = 1/tur/2
        median = self.testdist.median()
        self.testdist = distributions.get_distribution('normal', loc=median, std=sigma)

    def set_testmedian(self, median):
        ''' Set median of test measurement

            Parameters
            ----------
            median: float
                Median value of a particular test measurement result
        '''
        sigma = self.testdist.std()
        self.testdist = distributions.get_distribution('normal', loc=median, std=sigma)

    def set_costs(self, FA, FR):
        ''' Set cost of false accept and reject for cost-minimization techniques '''
        self.cost_FA = FA
        self.cost_FR = FR

    def get_testmedian(self):
        ''' Get test measurement median '''
        return self.testdist.median()

    def is_simple(self):
        ''' Check if simplified normal-only functions can be used '''
        if self.procdist is None or self.testdist is None:
            return False
        if self.procdist.median() != 0 or self.testbias != 0:
            return False
        if self.procdist.name != 'normal' or self.testdist.name != 'normal':
            return False
        if self.speclimits[1] != 1 or self.speclimits[0] != -1:
            return False
        if self.guardband[1] != self.guardband[0]:
            return False
        return True

    def to_simple(self):
        ''' Convert to simple, normal-only form. '''
        if self.is_simple():
            return  # Already in simple form

        # Get existing parameters
        tur = self.get_tur() if self.testdist is not None else 4
        itp = self.get_itp() if self.procdist is not None else 0.95
        median = self.testdist.median() if self.testdist is not None else 0
        gbf = self.get_gbf()

        # Convert to normal/symmetric
        self.set_speclimits(-1, 1)
        sigma0 = self.speclimits[1] / stats.norm.ppf((1+itp)/2)
        self.procdist = distributions.get_distribution('normal', loc=0, std=sigma0)
        sigmat = 1/tur/2
        self.testdist = distributions.get_distribution('normal', loc=median, std=sigmat)
        self.set_gbf(gbf)

    def get_tur(self):
        ''' Get test uncertainty ratio.
            Speclimit range / Expanded test measurement uncertainty.
        '''
        rng = (self.speclimits[1] - self.speclimits[0])/2   # Half the interval
        TL = self.testdist.std() * 2   # k=2
        return rng/TL

    def get_itp(self):
        ''' Get in-tolerance probability '''
        return 1 - self.proc_risk()

    def get_guardband(self):
        ''' Get guardband as offset GBF where A = T - GBF '''
        return self.guardband

    def get_speclimits(self):
        ''' Get specification limits as absolute values '''
        return self.speclimits

    def get_gbf(self):
        ''' Get guardband as multiplier GB where A = T * GB '''
        gb = self.guardband[1] - (self.guardband[1] - self.guardband[0])/2
        rng = (self.speclimits[1] - self.speclimits[0])/2
        gbf = 1 - gb / rng
        return gbf

    def get_testdist(self):
        ''' Get the test distribution '''
        return self.testdist

    def get_procdist(self):
        ''' Get the process distribution '''
        return self.procdist

    def calc_guardband(self, method, pfa=None):
        ''' Set guardband using a predefined method

        Parameters
        ----------
        method: string
            Guardband method to apply. One of: 'dobbert', 'rss',
            'rp10', 'test', '4:1', 'pfa', 'minimax', 'mincost', 'specific'.
        pfa: float (optional)
            Target PFA (for method 'pfa' and 'specific'. Defaults to 0.008)

        Notes
        -----
        Dobbert's method maintains <2% PFA for ANY itp at the TUR.
        RSS method: GB = sqrt(1-1/TUR**2)
        test method: GB = 1 - 1/TUR  (subtract the 95% test uncertainty)
        rp10 method: GB = 1.25 - 1/TUR (similar to test, but less conservative)
        pfa method: Solve for GB to produce desired PFA
        4:1 method: Solve for GB that results in same PFA as 4:1 at this itp
        mincost method: Minimize the total expected cost due to false decisions (Ref Easterling 1991)
        minimax method: Minimize the maximum expected cost due to false decisions (Ref Easterling 1991)
        '''
        CcCp = None
        if method in ['minimax', 'mincost']:
            if (self.cost_FA is None or self.cost_FR is None):
                raise ValueError('Minimax and Mincost methods must set costs of false accept/reject using `set_costs`.')
            CcCp = self.cost_FA/self.cost_FR

        if method == 'specific':
            gbl, gbu = guardband_specific(self.get_testdist(), *self.get_speclimits(), pfa)
            self.set_guardband(gbl, gbu)
        elif self.is_simple() or method != 'pfa':
            itp = self.get_itp() if self.get_procdist() is not None else None
            gbf = guardband_norm(method, self.get_tur(), pfa=pfa, itp=itp, CcCp=CcCp)
            self.set_gbf(gbf)
        else:
            gb = guardband(self.get_procdist(), self.get_testdist(), *self.get_speclimits(), pfa, self.get_testbias(), approx=True)
            self.set_guardband(gb, gb)   # Returns guardband as offset

    def PFA(self, approx=True):
        ''' Calculate probability of false acceptance (consumer risk).

            Parameters
            ----------
            approx: bool
                Use trapezoidal integration approximation for speed

            Returns
            -------
            PFA: float
                Probability of false accept (0-1)
        '''
        return PFA(self.procdist, self.testdist, *self.speclimits,
                   *self.guardband, self.testbias, approx)

    def PFR(self, approx=True):
        ''' Calculate probability of false reject (producer risk).

            Parameters
            ----------
            approx: bool
                Use trapezoidal integration approximation for speed

            Returns
            -------
            PFR: float
                Probability of false reject (0-1)
        '''
        return PFR(self.procdist, self.testdist, *self.speclimits,
                   *self.guardband, self.testbias, approx)

    def proc_risk(self):
        ''' Calculate total process risk, risk of process distribution being outside
            specification limits
        '''
        return specific_risk(self.procdist, *self.speclimits)[1]

    def cpk(self):
        ''' Get process risk and CPK values

        Returns
        -------
        Cpk: float
            Process capability index. Cpk > 1.333 indicates process is capable of
            meeting specifications.
        risk_total: float
            Total risk (0-1 range) of nonconformance
        risk_lower: float
            Risk of nonconformance below LL
        risk_upper: float
            Risk of nonconformance above UL
        '''
        return specific_risk(self.procdist, *self.speclimits)

    def test_risk(self):
        ''' Calculate PFA or PFR of the specific test measurement defined by dist_test
            including its median shift. Does not consider process dist. If median(testdist)
            is within spec limits, PFA is returned. If median(testdist) is outside spec
            limits, PFR is returned.

            Returns
            -------
            PFx: float
                Probability of false accept or reject
            accept: bool
                Accept or reject this measurement
        '''
        med = self.testdist.median() + self.testbias
        LL, UL = self.speclimits
        LL, UL = min(LL, UL), max(LL, UL)  # Make sure LL < UL
        accept = (med >= LL + self.guardband[0] and med <= UL - self.guardband[1])

        if med >= LL + self.guardband[0] and med <= UL - self.guardband[1]:
            PFx = self.testdist.cdf(LL) + (1 - self.testdist.cdf(UL))
        else:
            PFx = abs(self.testdist.cdf(LL) - self.testdist.cdf(UL))
        return PFx, accept

    # Extra functions for GUI
    def get_procdist_args(self):
        ''' Get dictionary of arguments for process distribution '''
        args = self.procdist.kwds.copy()
        args.update({'dist': self.procdist.name})
        return args

    def get_testdist_args(self):
        ''' Get dictionary of arguments for test distribution '''
        args = self.testdist.kwds.copy()
        args.update({'dist': self.testdist.name})
        return args

    def get_testbias(self):
        ''' Get bias in test distribution '''
        return self.testbias

    def set_testbias(self, bias=0):
        ''' Set bias in test distribution '''
        self.testbias = bias

    # Stuff to make it compatible with UncertCalc projects
    def calculate(self):
        ''' "Calculate" values, returning RiskOutput object '''
        self.out = RiskOutput(self)
        return self.out

    def get_output(self):
        ''' Get output object (or None if not calculated yet) '''
        return self.out

    def get_config(self):
        ''' Get configuration dictionary '''
        d = {}
        d['mode'] = 'risk'
        d['name'] = self.name
        d['desc'] = self.description
        d['bias'] = self.testbias

        if self.procdist is not None:
            d['distproc'] = self.procdist.get_config()

        if self.testdist is not None:
            d['disttest'] = self.testdist.get_config()

        d['GBL'] = self.guardband[0]
        d['GBU'] = self.guardband[1]
        d['LL'] = self.speclimits[0]
        d['UL'] = self.speclimits[1]
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
        ''' Load Risk object from config dictionary '''
        newrisk = cls(name=config.get('name', 'risk'))
        newrisk.description = config.get('desc', '')
        newrisk.set_speclimits(config.get('LL', 0), config.get('UL', 0))
        newrisk.set_guardband(config.get('GBL', 0), config.get('GBU', 0))
        newrisk.set_testbias(config.get('bias', 0))

        dproc = config.get('distproc', None)
        if dproc is not None:
            dist_proc = distributions.from_config(dproc)
            newrisk.set_procdist(dist_proc)
        else:
            newrisk.procdist = None

        dtest = config.get('disttest', None)
        if dtest is not None:
            dist_test = distributions.from_config(dtest)
            newrisk.set_testdist(dist_test)
        else:
            newrisk.testdist = None
        return newrisk

    @classmethod
    def from_configfile(cls, fname):
        ''' Read and parse the configuration file. Returns a new Risk
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


class RiskOutput(output.Output):
    ''' Output object for risk calculation. Just a reporting wrapper around
        Risk object for parallelism with other calculator modes.
    '''
    def __init__(self, risk):
        self.risk = risk

    def report(self, **kwargs):
        ''' Generate report of risk calculation '''
        hdr = []
        cols = []
        cost = None

        if self.risk.get_procdist() is not None:
            cpk, risk_total, risk_lower, risk_upper = self.risk.cpk()
            hdr.extend(['Process Risk'])   # No way to span columns at this point...
            cols.append([('Process Risk: ', report.Number(risk_total*100, fmt='auto'), '%'),
                         ('Upper limit risk: ', report.Number(risk_upper*100, fmt='auto'), '%'),
                         ('Lower limit risk: ', report.Number(risk_lower*100, fmt='auto'), '%'),
                         ('Process capability index (Cpk): ', report.Number(cpk))])
            if self.risk.cost_FA is not None:
                cost = self.risk.cost_FA * risk_total  # Everything accepted - no false rejects

        if self.risk.get_testdist() is not None:
            val = self.risk.get_testdist().median() + self.risk.get_testbias()
            PFx, accept = self.risk.test_risk()  # Get PFA/PFR of specific measurement

            hdr.extend(['Specific Measurement Risk'])
            cols.append([
                ('TUR: ', report.Number(self.risk.get_tur(), fmt='auto')),
                ('Measured value: ', report.Number(val)),
                'Result: {}'.format('ACCEPT' if accept else 'REJECT'),
                ('Specific F{} Risk: '.format('A' if accept else 'R'), report.Number(PFx*100, fmt='auto'), '%'),
                ])

        if self.risk.get_testdist() is not None and self.risk.get_procdist() is not None:
            hdr.extend(['Global Risk'])
            pfa = self.risk.PFA()
            pfr = self.risk.PFR()
            cols.append([
                ('Total PFA: ', report.Number(pfa*100, fmt='auto'), '%'),
                ('Total PFR: ', report.Number(pfr*100, fmt='auto'), '%'), '-', '-'])
            if self.risk.cost_FA is not None and self.risk.cost_FR is not None:
                cost = self.risk.cost_FA * pfa + self.risk.cost_FR * pfr

        rpt = report.Report()
        if len(hdr) > 0:
            rows = list(map(list, zip(*cols)))  # Transpose cols->rows
            rpt.table(rows=rows, hdr=hdr)

        if cost is not None:
            costrows = [['Cost of false accept', report.Number(self.risk.cost_FA)],
                        ['Cost of false reject', report.Number(self.risk.cost_FR)],
                        ['Expected cost', report.Number(cost)]]
            rpt.table(costrows, hdr=['Cost', 'Value'])
        return rpt

    def report_all(self, **kwargs):
        ''' Report with table and plots '''
        if kwargs.get('mc', False):
            with plt.style.context(plotting.plotstyle):
                fig = plt.figure()
            r = self.report_montecarlo(fig=fig, **kwargs)
            r.plot(fig)
            plt.close(fig)
        else:
            with plt.style.context(plotting.plotstyle):
                fig = plt.figure()
                self.plot_dists(fig)
            r = report.Report(**kwargs)
            r.plot(fig)
            plt.close(fig)
            r.txt('\n\n')
            r.append(self.report(**kwargs))
        return r

    def plot_dists(self, plot=None):
        ''' Plot risk distributions '''
        with plt.style.context(plotting.plotstyle):
            fig, ax = plotting.initplot(plot)
            fig.clf()

            procdist = self.risk.get_procdist()
            testdist = self.risk.get_testdist()

            nrows = (procdist is not None) + (testdist is not None)
            plotnum = 0
            LL, UL = self.risk.get_speclimits()
            GBL, GBU = self.risk.get_guardband()

            # Add some room on either side of distributions
            pad = 0
            if procdist is not None:
                pad = max(pad, procdist.std() * 3)
            if testdist is not None:
                pad = max(pad, testdist.std() * 3)

            xmin = xmax = pad
            if np.isfinite(LL):
                xmin = LL-pad
            elif procdist:
                xmin = procdist.mean() - pad*2
            elif testdist:
                xmin = testdist.mean() - pad*2
            if np.isfinite(UL):
                xmax = UL+pad
            elif procdist:
                xmax = procdist.mean() + pad*2
            elif testdist:
                xmax = testdist.mean() + pad*2

            x = np.linspace(xmin, xmax, 300)
            if procdist is not None:
                yproc = procdist.pdf(x)
                ax = fig.add_subplot(nrows, 1, plotnum+1)
                ax.plot(x, yproc, label='Process Distribution', color='C0')
                ax.axvline(LL, ls='--', label='Specification Limits', color='C2')
                ax.axvline(UL, ls='--', color='C2')
                ax.fill_between(x, yproc, where=((x <= LL) | (x >= UL)), alpha=.5, color='C0')
                ax.set_ylabel('Probability Density')
                ax.set_xlabel('Value')
                ax.legend(loc='upper left', fontsize=10)
                if self.risk.is_simple():
                    ax.xaxis.set_major_formatter(FormatStrFormatter(r'%.1fSL'))
                plotnum += 1

            if testdist is not None:
                ytest = testdist.pdf(x)
                median = self.risk.get_testmedian()
                measured = median + self.risk.get_testbias()
                ax = fig.add_subplot(nrows, 1, plotnum+1)
                ax.plot(x, ytest, label='Test Distribution', color='C1')
                ax.axvline(measured, ls='--', color='C1')
                ax.axvline(median, ls=':', lw=.5, color='lightgray')
                ax.axvline(LL, ls='--', label='Specification Limits', color='C2')
                ax.axvline(UL, ls='--', color='C2')
                if GBL != 0 or GBU != 0:
                    ax.axvline(LL+GBL, ls='--', label='Guardband', color='C3')
                    ax.axvline(UL-GBU, ls='--', color='C3')

                if measured > UL-GBU or measured < LL+GBL:   # Shade PFR
                    ax.fill_between(x, ytest, where=((x >= LL) & (x <= UL)), alpha=.5, color='C1')
                else:  # Shade PFA
                    ax.fill_between(x, ytest, where=((x <= LL) | (x >= UL)), alpha=.5, color='C1')

                ax.set_ylabel('Probability Density')
                ax.set_xlabel('Value')
                ax.legend(loc='upper left', fontsize=10)
                if self.risk.is_simple():
                    ax.xaxis.set_major_formatter(FormatStrFormatter(r'%.1fSL'))
            fig.tight_layout()
        return fig

    def plot_joint(self, plot=None):
        ''' Plot risk distributions '''
        with plt.style.context(plotting.plotstyle):
            fig, ax = plotting.initplot(plot)
            fig.clf()

            procdist = self.risk.get_procdist()
            testdist = self.risk.get_testdist()

            pad = 0
            pad = max(pad, procdist.std() * 3)
            pad = max(pad, testdist.std() * 3)

            LL, UL = self.risk.get_speclimits()
            GBL, GBU = self.risk.get_guardband()

            xmin = LL-pad if np.isfinite(LL) else procdist.mean() - pad*2
            xmax = UL+pad if np.isfinite(UL) else procdist.mean() + pad*2

            x = y = np.linspace(xmin, xmax, 300)
            xx, yy = np.meshgrid(x, y)
            pdf1 = procdist.pdf(xx)
            expected = testdist.median()
            kwds = distributions.get_distargs(testdist)
            locorig = kwds.pop('loc', 0)
            pdf2 = testdist.dist.pdf(yy, loc=xx-(expected-locorig), **kwds)

            ax1 = plt.subplot2grid((5, 5), loc=(1, 0), colspan=4, rowspan=4, fig=fig)
            ax1.contourf(xx, yy, (pdf1*pdf2)**.5, levels=20, cmap='Blues')
            ax1.contour(xx, yy, (pdf1*pdf2)**.5, levels=20, colors='blue', linewidths=.1)
            ax1.axvline(LL, color='black', ls='--', lw=1)
            ax1.axhline(LL, color='black', ls='--', lw=1)
            ax1.axvline(UL, color='black', ls='--', lw=1)
            ax1.axhline(UL, color='black', ls='--', lw=1)
            ax1.axhline(LL+GBL, color='gray', ls='--', lw=1)
            ax1.axhline(UL-GBU, color='gray', ls='--', lw=1)

            procpdf = procdist.pdf(x)
            testpdf = testdist.pdf(y)

            ax2 = plt.subplot2grid((5, 5), loc=(0, 0), colspan=4, sharex=ax1, fig=fig)
            ax2.plot(x, procpdf)
            ax2.fill_between(x, procpdf, where=x>UL, color='C0', alpha=.25)
            ax2.fill_between(x, procpdf, where=x<LL, color='C0', alpha=.25)
            ax2.axvline(LL, color='black', ls='--', lw=1)
            ax2.axvline(UL, color='black', ls='--', lw=1)
            
            ax3 = plt.subplot2grid((5, 5), loc=(1, 4), rowspan=4, sharey=ax1, fig=fig)
            ax3.plot(testpdf, y, color='C1')
            ax3.fill_betweenx(y, testpdf, where=y>UL, color='C1', alpha=.25)
            ax3.fill_betweenx(y, testpdf, where=y<LL, color='C1', alpha=.25)
            ax3.axhline(testdist.mean(), ls='--', lw=1, color='C1')
            ax3.axhline(LL, color='black', ls='--', lw=1)
            ax3.axhline(UL, color='black', ls='--', lw=1)

            fig.subplots_adjust(hspace=0.05, wspace=.05)
            plt.setp(ax2.get_xticklabels(), visible=False)
            plt.setp(ax2.get_yticklabels(), visible=False)
            plt.setp(ax3.get_xticklabels(), visible=False)
            plt.setp(ax3.get_yticklabels(), visible=False)

            ax1.set_xlabel('Actual Product')
            ax1.set_ylabel('Test Result')
            ax1.set_ylim(y.min(), y.max())

            LLGB = LL+GBL if np.isfinite(LL+GBL) else xmin
            ULGB = UL-GBU if np.isfinite(UL-GBU) else xmax
            LL1 = LL if np.isfinite(LL) else xmin
            UL1 = UL if np.isfinite(UL) else xmax
            ax1.fill_between(x, LLGB, ULGB, where=(x<LL), color='C1', alpha=.15, label='False Accept')
            ax1.fill_between(x, LLGB, ULGB, where=(x>UL), color='C1', alpha=.15)
            ax1.fill_betweenx(y, LL1, UL1, where=(x<LL+GBL), color='C2', alpha=.15, label='False Reject')
            ax1.fill_betweenx(y, LL1, UL1, where=(x>UL-GBU), color='C2', alpha=.15)
            ax1.legend(loc='upper left', fontsize=10)
            ax1.set_xlim(xmin, xmax)
            ax1.set_ylim(xmin, xmax)

            if self.risk.is_simple():
                ax1.xaxis.set_major_formatter(FormatStrFormatter(r'%.1fSL'))
                ax1.yaxis.set_major_formatter(FormatStrFormatter(r'%.1fSL'))
        return fig


    def report_montecarlo(self, fig=None, **kwargs):
        ''' Run Monte-Carlo risk and return report. If fig is provided, plot it. '''
        N = kwargs.get('samples', 100000)
        LL, UL = self.risk.get_speclimits()
        GB = self.risk.get_guardband()
        pfa, pfr, psamples, tsamples = PFAR_MC(self.risk.get_procdist(), self.risk.get_testdist(),
                                               LL, UL, *GB, N=N, testbias=self.risk.get_testbias())

        LLplot = np.nan if not np.isfinite(LL) else LL
        ULplot = np.nan if not np.isfinite(UL) else UL
        if fig is not None:
            fig.clf()
            ax = fig.add_subplot(1, 1, 1)
            if psamples is not None:
                ifa1 = (tsamples > LL+GB[0]) & (tsamples < UL-GB[1]) & ((psamples < LL) | (psamples > UL))
                ifr1 = ((tsamples < LL+GB[0]) | (tsamples > UL-GB[1])) & ((psamples > LL) & (psamples < UL))
                good = np.logical_not(ifa1 | ifr1)
                ax.plot(psamples[good], tsamples[good], marker='o', ls='', markersize=2, color='C0', label='Correct Decision', rasterized=True)
                ax.plot(psamples[ifa1], tsamples[ifa1], marker='o', ls='', markersize=2, color='C1', label='False Accept', rasterized=True)
                ax.plot(psamples[ifr1], tsamples[ifr1], marker='o', ls='', markersize=2, color='C2', label='False Reject', rasterized=True)
                ax.axvline(LLplot, ls='--', lw=1, color='black')
                ax.axvline(ULplot, ls='--', lw=1, color='black')
                ax.axhline(LLplot+GB[0], lw=1, ls='--', color='gray')
                ax.axhline(ULplot-GB[1], lw=1, ls='--', color='gray')
                ax.axhline(LLplot, ls='--', lw=1, color='black')
                ax.axhline(ULplot, ls='--', lw=1, color='black')
                ax.legend(loc='upper left', fontsize=10)
                ax.set_xlabel('Actual Product')
                ax.set_ylabel('Test Result')
                pdist = self.risk.get_procdist()
                tdist = self.risk.get_testdist()
                xmin = np.nanmin([LLplot, pdist.mean()-5*pdist.std()])
                xmax = np.nanmax([ULplot, pdist.mean()+5*pdist.std()])
                ymin = np.nanmin([LLplot, pdist.mean()-5*pdist.std()-tdist.std()])
                ymax = np.nanmax([ULplot, pdist.mean()+5*pdist.std()+tdist.std()])
                ax.set_xlim(xmin, xmax)
                ax.set_ylim(ymin, ymax)
                fig.tight_layout()

        rpt = report.Report(**kwargs)
        rpt.txt('- TUR: ')
        rpt.num(self.risk.get_tur(), fmt='auto', end='\n')
        rpt.txt('- Total PFA: ')
        rpt.num(pfa*100, fmt='auto', end='%\n')
        rpt.txt('- Total PFR: ')
        rpt.num(pfr*100, fmt='auto', end='%\n')
        return rpt

    def report_gbsweep(self, plot=None, **kwargs):
        ''' Plot PFA/R vs guardband '''
        with plt.style.context(plotting.plotstyle):
            fig, ax = plotting.initplot(plot)
            fig.clf()
            LL, UL = self.risk.get_speclimits()
            simple = self.risk.is_simple()
            dtest = self.risk.get_testdist()
            dproc = self.risk.get_procdist()

            if dproc is not None and dtest is not None:
                if simple:
                    gbrange = np.linspace(0, 1, num=26)
                    xlabel = 'Guardband Factor'
                else:
                    gbrange = np.linspace(0, dtest.std()*3, num=26)
                    xlabel = 'Guardband'

                pfa = np.empty(len(gbrange))
                pfr = np.empty(len(gbrange))
                for i, gb in enumerate(gbrange):
                    if simple:
                        self.risk.set_gbf(gb)
                    else:
                        self.risk.set_guardband(gb, gb)  # Always symmetric
                    pfa[i] = self.risk.PFA()
                    pfr[i] = self.risk.PFR()

                ax1 = fig.add_subplot(2, 1, 1)
                ax1.plot(gbrange, pfa*100)
                ax1.set_ylabel('False Accept %')
                ax2 = fig.add_subplot(2, 1, 2)
                ax2.plot(gbrange, pfr*100)
                ax2.set_ylabel('False Reject %')
                ax2.set_xlabel(xlabel)
                ax1.get_shared_x_axes().join(ax1, ax2)
                fig.tight_layout()
                if simple:
                    ax1.set_xlim(1, 0)  # Go backwards
                    ax2.set_xlim(1, 0)
                plt.close(fig)

            rpt = report.Report(**kwargs)
            rpt.hdr(f'Global Risk vs {xlabel}', level=2)
            hdr = [xlabel, 'False Accept %', 'False Reject %']
            gbrange = [report.Number(g) for g in gbrange]
            pfa = [report.Number(p*100) for p in pfa]
            pfr = [report.Number(p*100) for p in pfr]
            rows = list(map(list, zip(gbrange, pfa, pfr)))
            if simple:
                rows = list(reversed(rows))
            rpt.table(rows=rows, hdr=hdr)
            return rpt

        rpt = report.Report(**kwargs)
        rpt.text('No test distribution')
        return rpt

    def report_probconform(self, plot=None, **kwargs):
        ''' Plot Probability of Conformance plot '''
        with plt.style.context(plotting.plotstyle):
            fig, ax = plotting.initplot(plot)
            fig.clf()
            LL, UL = self.risk.get_speclimits()
            dtest = self.risk.get_testdist()

            if dtest is not None:
                kwds = distributions.get_distargs(dtest)
                bias = self.risk.get_testbias()
                w = (UL-LL)
                xx = np.linspace(LL-w/2, UL+w/2, num=500)
                if not np.isfinite(w):
                    w = dtest.std() * 4
                    xx = np.linspace(dtest.mean()-w if not np.isfinite(LL) else LL-w/2,
                                     dtest.mean()+w if not np.isfinite(UL) else UL+w/2,
                                     num=500)
                fa_lower = np.empty(len(xx))
                fa_upper = np.empty(len(xx))
                for i, loc in enumerate(xx):
                    dtest.set_median(loc-bias)
                    kwds = distributions.get_distargs(dtest)
                    dtestswp = dtest.dist(**kwds)  # Pass in rv_continuous, could be faster than Distribution object
                    fa_lower[i] = specific_risk(dtestswp, LL=LL, UL=np.inf).total
                    fa_upper[i] = specific_risk(dtestswp, LL=-np.inf, UL=UL).total
                fa = 1-(fa_lower + fa_upper)

                ax = fig.add_subplot(1, 1, 1)
                ax.plot(xx, fa*100, color='C1')
                ax.set_ylabel('Probability of Conformance %')
                ax.set_xlabel('Measurement Result')
                ax.axvline(LL, ls='--', label='Specification Limits', color='C2')
                ax.axvline(UL, ls='--', color='C2')
                GBL, GBU = self.risk.get_guardband()
                if GBL != 0 or GBU != 0:
                    ax.axvline(LL+GBL, ls='--', label='Guardband', color='C3')
                    ax.axvline(UL-GBU, ls='--', color='C3')
                fig.tight_layout()
                plt.close(fig)

                rpt = report.Report(**kwargs)
                rpt.hdr('Probability of Conformance', level=2)
                hdr = ['Measurement Result', 'Probability of Conformance %']
                xx = report.Number.number_array(xx[::10])
                pc = [report.Number(p) for p in fa[::10]*100]
                rows = list(map(list, zip(xx, pc)))
                rpt.table(rows, hdr=hdr)
                return rpt

            rpt = report.Report(**kwargs)
            rpt.text('No test distribution')
            return rpt

    def report_sweep(self, plot=None, **kwargs):
        ''' Plot PFA(R) sweep (simple mode only for now) '''
        xvar = kwargs.get('xvar')
        zvar = kwargs.get('zvar')
        xvals = kwargs.get('xvals')
        zvals = kwargs.get('zvals')
        yvar = kwargs.get('yvar', 'PFA')
        threed = kwargs.get('threed', False)
        logy = kwargs.get('logy', False)
        gbmode = kwargs.get('gbf', None)
        sig0 = kwargs.get('sig0', None)
        tbias = kwargs.get('tbias', 0)
        pbias = kwargs.get('pbias', 0)

        gbf = self.risk.get_gbf() if gbmode is None else gbmode

        labels = {'tur': 'TUR', 'itp': 'In-Tolerance Probability %', 'tbias': 'Test Measurement Bias',
                  'pbias': 'Process Distribution Bias', 'gbf': 'GBF', 'sig0': r'$SL/\sigma_0$'}

        rpt = report.Report(**kwargs)

        with plt.style.context(plotting.plotstyle):
            fig, ax = plotting.initplot(plot)
            fig.clf()

            yvars = [yvar.lower()] if yvar.lower() != 'both' else ['pfa', 'pfr']
            for k, yvar in enumerate(yvars):
                curves = PFA_sweep_simple(xvar, zvar, xvals, zvals,
                                          GBFdflt=gbf, itpdflt=self.risk.get_itp(),
                                          TURdflt=self.risk.get_tur(), risk=yvar,
                                          sig0=sig0,
                                          tbias=tbias, pbias=pbias) * 100

                xlabel = labels.get(xvar, 'x')
                zlabel = labels.get(zvar, 'z')
                ylabel = '{} %'.format(yvar.upper())

                xplot = xvals if xvar.lower() not in ['itp', 'tbias', 'pbias'] else xvals * 100
                zplot = np.zeros(len(xvals)) if xvar == 'none' else zvals if zvar.lower() not in ['itp', 'tbias', 'pbias'] else zvals * 100

                if threed:
                    ax = fig.add_subplot(1, len(yvars), k+1, projection='3d')
                    xx, zz = np.meshgrid(xplot, zplot)
                    ax.plot_surface(xx, zz, curves, cmap='coolwarm')
                    ax.set_zlabel(ylabel)
                    ax.set_ylabel(zlabel)
                else:
                    ax = fig.add_subplot(1, len(yvars), k+1)
                    for i in range(len(zvals)):
                        ax.plot(xplot, curves[i, :], label=str(zplot[i]))
                    ax.set_ylabel(ylabel)

                ax.set_xlabel(xlabel)
                if zvar != 'none' and not threed:
                    ax.legend(title=zlabel, fontsize=10)

                if logy:
                    ax.set_yscale('log')

                rpt.hdr(ylabel, level=2)
                if zvar == 'none':
                    hdr = [xlabel, ylabel]
                else:
                    hdr = [xlabel] + ['{}={}'.format(zlabel, z) for z in zvals]
                rows = [[report.Number(xvals[i])] + [report.Number(p) for p in row] for i, row in enumerate(curves.transpose())]
                rpt.table(rows, hdr)

            fig.tight_layout()
            plt.close(fig)
        return rpt
