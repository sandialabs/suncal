''' Calculations for risk, including probability of false accept (PFA) or consumer risk,
    and probability of false reject (PFR) or producer risk.

    The PFA and PFR functions take arbitrary distributions and perform the false accept/
    false reject double integrals numerically. Distributions can be either frozen instances
    of scipy.stats or random samples (e.g. Monte Carlo output of a forward uncertainty
    propagation calculation). PFAR_MC will find both PFA and PFR using a Monte Carlo method.

    The functions PFA_norm and PFR_norm assume normal distributions and take
    TUR and in-tolerance-probability (itp) as inputs. The normal assumption will
    make these functions much faster than the generic PFA and PFR functions.

    The guardband and guardband_norm functions can be used to determine the guardband
    required to meet a specified PFA, or apply one of the common guardband calculation
    techniques.
'''

from collections import namedtuple
from contextlib import suppress
import numpy as np
from scipy import stats
from scipy.integrate import dblquad
from scipy.optimize import brentq, fsolve

from ..common import distributions


def specific_risk(dist, LL, UL):
    ''' Calculate specific risk and process capability index for the distribution.

        Args:
            dist (stats.rv_frozen or distributions.Distribution): Distribution
              of possible unit under test values
            LL (float): Lower specification limit
            UL (float): Upper specification limit

        Returns:
            Cpk (float): Process capability index. Cpk > 1.333 indicates
                process is capable of meeting specifications.
            risk_total (float): Total risk (0-1 range) of nonconformance
            risk_lower (float): Risk of nonconformance below LL
            risk_upper (float): Risk of nonconformance above UL

        Notes:
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

        Args:
            method (str): Guardband method to apply. One of: 'dobbert', 'rss'/'rds',
              'rp10', 'test', '4:1', 'pfa', 'mincost', 'minimax'.
            TUR (float): Test Uncertainty Ratio
            pfa (float): Target PFA (for method 'pfa'. Defaults to 0.008)
            itp (float): In-tolerance probability (for method 'pfa', '4:1', 'mincost',
              and 'minimax'. Defaults to 0.95)
            CcCp (float): Ratio of cost of false accept to cost of part (for methods
              'mincost' and 'minimax')

        Returns:
            guardband (float): Guardband factor. Acceptance limit = Tolerance Limit * guardband.

        Notes:
            Dobbert's method maintains <2% PFA for ANY itp at the TUR.
            RDS (same as RSS) method: GB = sqrt(1-1/TUR**2)
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
    elif method in ['rds', 'rss']:
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
        raise ValueError(f'Unknown guardband method {method}.')
    return GB


def guardband(dist_proc, dist_test, LL, UL, target_PFA, testbias=0, approx=False):
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
            approx (bool): Approximate the integral using discrete probability distribution.
              Faster than using scipy.integrate.

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
                                     GBU=x, GBL=x, testbias=testbias, approx=approx)-target_PFA,
                       a=-w/2, b=w/2, full_output=True)
    except ValueError:
        return np.nan  # Problem solving

    if r.converged:
        return gb
    else:
        return np.nan


def guardband_specific(dtest, LL, UL, target):
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
    ''' PFA for normal distributions in terms of TUR and in-tolerance probability

        Args:
            itp (float): In-tolerance probability (0-1 range). A-priori distribution of process.
            TUR (float): Test Uncertainty Ratio. Spec Limit / (2*Test Uncertainty)
            GB (float or string): Guardband Factor. If GB is numeric, GB = K, where
              acceptance limit A = T * K. In Dobbert's notation, K = 1 - M/TUR where
              A = T - U*M. GB = 1 implies no guardbanding.
              If GB is a string, it can be one of options in guardband_norm
              method. kwargs passed to guardband_norm.
            sig0 (float): Process standard deviation in terms of #SL, overrides itp
            biastest (float): Bias/shift in the test measurement distribution, in percent of SL
            biasproc (float): Bias/shift in the process distribution, in percent of SL
            observed (bool): Consider itp as the "observed" itp. True itp will be adjusted
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
    ''' PFR for normal distributions in terms of TUR and in-tolerance probability

        Args:
            itp (float): In-tolerance probability (0-1 range). A-priori distribution of process.
            TUR (float): Test Uncertainty Ratio. Spec Limit / (2*Test Uncertainty)
            GB (float or string): Guardband Factor. If GB is numeric, GB = K, where
              acceptance limit A = T * K. In Dobbert's notation, K = 1 - M/TUR where
              A = T - U*M. GB = 1 implies no guardbanding.
              If GB is a string, it can be one of options in guardband_norm
              method. kwargs passed to guardband_norm.
            sig0 (float): Process standard deviation in terms of #SL, overrides itp
            biastest (float): Bias/shift in the test measurement distribution, in percent of SL
            biasproc (float): Bias/shift in the process distribution, in percent of SL
            observed (bool): Consider itp as the "observed" itp. True itp will be adjusted
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


def PFA(dist_proc, dist_test, LL, UL, GBL=0, GBU=0, testbias=0, approx=False):
    ''' Calculate Probability of False Accept (Consumer Risk) for arbitrary
        process and test distributions.

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
            approx (bool): Approximate using discrete probability distribution. This
              uses trapz integration so it may be faster than letting scipy integrate
              the actual pdf function.

        Returns:
            PFA (float): Probability of False Accept
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

        Args:
            dist_proc (array): Sampled values from process distribution
            dist_test (array): Sampled values from test measurement distribution
            LL (float):Lower specification limit (absolute)
            UL (float): Upper specification limit (absolute)
            GBL (float): Lower guardband, as offset. Test limit is LL + GBL.
            GBU (float): Upper guardband, as offset. Test limit is UL - GBU.
            testbias (float): Bias (difference between distribution median and expected value)
              in test distribution

        Returns:
            PFA (float): Probability of False Accept
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
            approx (bool): Approximate using discrete probability distribution. This
              uses trapz integration so it may be faster than letting scipy integrate
              the actual pdf function.

        Returns:
            PFR (float): Probability of False Reject
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

        Args:
            dist_proc (array): Sampled values from process distribution
            dist_test (array): Sampled values from test measurement distribution
            LL (float):Lower specification limit (absolute)
            UL (float): Upper specification limit (absolute)
            GBL (float): Lower guardband, as offset. Test limit is LL + GBL.
            GBU (float): Upper guardband, as offset. Test limit is UL - GBU.
            testbias (float): Bias (difference between distribution median and expected value)
              in test distribution

        Returns:
            PFR (float): Probability of False Reject
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
            approx (bool): Approximate using discrete probability distribution. This
              uses trapz integration so it may be faster than letting scipy integrate
              the actual pdf function.
            N (int): Number of Monte Carlo samples

        Returns:
            PFA: False accept probability
            PFR: False reject probability
            proc_samples: Monte Carlo samples for uut
            test_samples: Monte Carlo samples for test measurement
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

    FA = np.count_nonzero(((test_samples < UL-GBU) & (test_samples > LL+GBL)) &
                          ((proc_samples > UL) | (proc_samples < LL))) / N
    FR = np.count_nonzero(((test_samples > UL-GBU) | (test_samples < LL+GBL)) &
                          ((proc_samples < UL) & (proc_samples > LL))) / N
    return Result(FA, FR, proc_samples, test_samples)
