''' Calculations for risk, including probability of false accept (PFA) or consumer risk,
    and probability of false reject (PFR) or producer risk.
'''
from collections import namedtuple
import numpy as np
from scipy import stats
from scipy.optimize import root_scalar

from . import risk_simpson
from ..common import distributions


SpecificRisk = namedtuple('SpecificRisk', ['cpk', 'total', 'lower', 'upper'])


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
    return SpecificRisk(cpk, risk_total, risk_lower, risk_upper)


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

    out = root_scalar(lambda x: (1-sp_risk(bias, x))-itp,
                      bracket=(1E-9, 10),
                      x0=.1)
    if out.converged:
        return out.root
    return 0.


def get_sigmaproc_from_itp_arb(dist, param, itp, LL, UL):
    ''' Get process standard deviation from itp for arbitrary (non-normal)
        distributions

        Args:
            dist (stats.rv_frozen or distributions.Distribution):
                Probability distribution with parameters defined
            param (str): Name of the distribution parameter to adjust to
                meet the itp value
            itp (float): In-tolerance probability target (0-1)
            LL (float): Lower tolerance limit
            UL (float): Upper tolerance limit

        Returns:
            param (float): Value of parameter that results in itp% of the
                distribution falling between LL and UL. Returns None if
                no solution is found.
    '''
    fixedargs = dist.kwds
    currentval = fixedargs.pop(param, 1)
    def sp_risk(**kwargs):
        ''' Simplified version of specific risk function '''
        rl = distributions.get_distribution(dist.name, **kwargs).cdf(LL)
        ru = 1 - distributions.get_distribution(dist.name, **kwargs).cdf(UL)
        return rl + ru

    try:
        out = root_scalar(lambda x: (1-sp_risk(**{param: x}, **fixedargs))-itp,
                          bracket=(0, currentval*100),
                          x0=currentval)
    except ValueError:  # Not bracketed
        return None

    if out.converged:
        return out.root
    return None


def get_sigmaproc_from_cpk(dist, param, cpk, LL, UL):
    ''' Get process standard deviation from Process Capability Index '''
    fixedargs = dist.kwds
    currentval = fixedargs.pop(param, 1)
    def new_cpk(**kwargs):
        ''' Cpk for the test distribution '''
        d = distributions.get_distribution(dist.name, **kwargs)
        return specific_risk(d, LL, UL).cpk

    try:
        out = root_scalar(lambda x: new_cpk(**{param: x}, **fixedargs)-cpk,
                          bracket=(0, currentval*100),
                          x0=currentval)
    except ValueError:  # Not bracketed
        return None

    if out.converged:
        return out.root
    return None


def PFA(dist_proc, dist_test, LL, UL, GBL=0, GBU=0, testbias=0):
    ''' Calculate unconditional global Probability of False Accept for arbitrary
        process and test distributions.

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
    return risk_simpson.PFA(dist_proc, dist_test, LL, UL, GBL, GBU, testbias)


def PFR(dist_proc, dist_test, LL, UL, GBL=0, GBU=0, testbias=0):
    ''' Calculate global Probability of False Reject (Producer Risk) for arbitrary
        process and test distributions.

        Probability a DUT is in tolerance and rejected.

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
            PFR (float): Probability of False Reject
    '''
    return risk_simpson.PFR(dist_proc, dist_test, LL, UL, GBL, GBU, testbias)


def PFA_conditional(dist_proc, dist_test, LL, UL, GBL=0, GBU=0, testbias=0):
    ''' Calculate conditional global Probability of False Accept for arbitrary
        process and test distributions.

        Probability a DUT is OOT given it was Accepted.

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
    return risk_simpson.PFA_conditional(dist_proc, dist_test, LL, UL, GBL, GBU, testbias)


def PFA_norm(itp, TUR, GB=1, sig0=None, biastest=0, biasproc=0, observeditp=False):
    ''' PFA for normal distributions in terms of TUR and in-tolerance probability

        Args:
            itp (float): In-tolerance probability (0-1 range). A-priori distribution of process.
            TUR (float): Test Uncertainty Ratio. Spec Limit / (2*Test Uncertainty)
            GB (float): Guardband Factor. Acceptance limit A = T * GB.
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

    GB = 1-GB  # Guardband factor to absolute guardband width
    dtest = stats.norm(loc=0, scale=sigmatest)
    dproc = stats.norm(loc=biasproc, scale=sigma0)
    return PFA(dproc, dtest, -1, 1, GB, GB, testbias=biastest)


def PFA_norm_conditional(itp, TUR, GB=1, sig0=None, biastest=0, biasproc=0, observeditp=False):
    ''' Conditional PFA for normal distributions given TUR, ITP, and guardband

        Args:
            itp (float): In-tolerance probability (0-1 range). A-priori distribution of process.
            TUR (float): Test Uncertainty Ratio. Spec Limit / (2*Test Uncertainty)
            GB (float): Guardband Factor. Acceptance limit A = T * GB.
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

    GB = 1-GB  # Guardband factor to absolute guardband width
    dtest = stats.norm(loc=0, scale=sigmatest)
    dproc = stats.norm(loc=biasproc, scale=sigma0)
    return PFA_conditional(dproc, dtest, -1, 1, GB, GB, testbias=biastest)


def PFR_norm(itp, TUR, GB=1, sig0=None, biastest=0, biasproc=0, observeditp=False):
    ''' PFR for normal distributions in terms of TUR and in-tolerance probability

        Args:
            itp (float): In-tolerance probability (0-1 range). A-priori distribution of process.
            TUR (float): Test Uncertainty Ratio. Spec Limit / (2*Test Uncertainty)
            GB (float): Guardband Factor. Acceptance limit A = T * GB.
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

    GB = 1-GB  # Guardband factor to absolute guardband width
    dtest = stats.norm(loc=0, scale=sigmatest)
    dproc = stats.norm(loc=biasproc, scale=sigma0)
    return PFR(dproc, dtest, -1, 1, GB, GB, testbias=biastest)
