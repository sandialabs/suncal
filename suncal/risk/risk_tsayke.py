''' Compute risk using Tsay-Ke (2023) approximation to the Bivariate Normal Integral.

    Reference: Tsay-Ke, A simple approximation for the bivariate normal integral,”
        Communications in Statistics-Simulation and Computation,
        vol. 52, no. 4, pp. 1462-1475.

    Adapt PFA integral into standard bivariate normal and use Tsay-Ke's formula to integrate.
'''
from scipy.special import erf
import numpy as np


def bivariate_norm_cdf(p, q, rho):
    ''' Compute integral of standard bivariate normal distribution
        over (-inf..p) and (-inf..q) with correlation rho.

        Because rho>0 for PFA problems, only cases 4 and 5 are implemented.
    '''
    cx = -1.0950081470333
    cy = -0.75651138383854
    d = np.sqrt(1-rho**2)
    a = -rho/d
    b = p/d
    aqplusb = a*q + b
    z = 1-a**2*cy
    sqrtz = np.sqrt(z)
    sqrt2 = np.sqrt(2)
    if a >= 0:
        raise NotImplementedError

    if aqplusb >= 0:
        result = (
            0.5 + 0.5 * erf(q/sqrt2)
            - 1/(4*sqrtz) * np.exp((a**2*cx**2+2*sqrt2*b*cx+2*b**2*cy)/(4*z))
            * (1 + erf((sqrt2*q - sqrt2*a**2*cy*q - sqrt2*a*b*cy - a*cx)/(2*sqrtz)))
        )

    else:
        result = (
            0.5 - 0.5 * erf(b/(sqrt2*a))
            - 1/(4*sqrtz) * np.exp((a**2*cx**2+2*sqrt2*b*cx+2*b**2*cy)/(4*z))
            * (1 - erf((sqrt2*b + a**2*cx)/(2*a*sqrtz)))
            + 1/(4*sqrtz) * np.exp((a**2*cx**2-2*sqrt2*b*cx+2*b**2*cy)/(4*z))
            * (erf((sqrt2*q-sqrt2*a**2*cy*q - sqrt2*a*b*cy + a*cx)/(2*sqrtz))
            + erf((-(a**2)*cx + sqrt2*b)/(2*a*sqrtz)))
            )
    return result


def PFA(sigmap, sigmam, mup, TL, TU, AL, AU, testbias=0):
    ''' Calculate Probability of False Accept (Consumer Risk) using
        sampled distributions and Simpson integration.

        Returns:
            PFA (float): Probability of False Accept
    '''
    center = (TU+TL)/2
    center_a = (AU+AL)/2
    if testbias == 0 and np.isclose(mup, center) and np.isclose(mup, center_a):
        T = TU - center
        A = AU - center
        return PFA_symmetric(sigmap, sigmam, T, A)
    else:
        return PFA_asymmetric(sigmap, sigmam, mup, TL, TU, AL, AU, testbias=testbias)


def PFA_symmetric(sigmap, sigmam, T, A):
    ''' PRobability of false accept for symmetric tolerances '''
    sp = sigmap
    st = np.sqrt(sigmam**2 + sigmap**2)
    rho = np.sqrt(sigmap**2 / (sigmap**2+sigmam**2))
    return 2*(bivariate_norm_cdf(A/st, -T/sp, rho) - bivariate_norm_cdf(-A/st, -T/sp, rho))


def PFA_asymmetric(sigmap, sigmam, mup, TL, TU, AL, AU, testbias=0):
    ''' Probability of false accept for asymmetric tolerances '''
    sp = sigmap
    st = np.sqrt(sigmam**2 + sigmap**2)
    rho = np.sqrt(sigmap**2 / (sigmap**2+sigmam**2))

    if not np.isfinite(AU):
        AU = 1E99
    if not np.isfinite(AL):
        AL = -1E99
    if not np.isfinite(TU):
        TU = 1E99
    if not np.isfinite(TL):
        TL = -1E99

    pfaleft = bivariate_norm_cdf((AU-mup-testbias)/st, (TL-mup)/sp, rho) - bivariate_norm_cdf((AL-mup-testbias)/st, (TL-mup)/sp, rho)
    pfarght = bivariate_norm_cdf((-AL+mup+testbias)/st, (-TU+mup)/sp, rho) - bivariate_norm_cdf((-AU+mup+testbias)/st, (-TU+mup)/sp, rho)
    return np.nansum((pfaleft+pfarght))


def PFR(sigmap, sigmam, mup, TL, TU, AL, AU, testbias=0):
    ''' Calculate Probability of False Reject (Producer Risk) using
        sampled distributions and Simpson integration.
    '''
    center = (TU+TL)/2
    center_a = (AU+AL)/2
    if testbias == 0 and np.isclose(mup, center) and np.isclose(mup, center_a):
        T = TU - center
        A = AU - center
        return PFR_symmetric(sigmap, sigmam, T, A)
    else:
        return PFR_asymmetric(sigmap, sigmam, mup, TL, TU, AL, AU, testbias=testbias)


def PFR_symmetric(sigmap, sigmam, T, A):
    ''' Probability of false accept for asymmetric tolerances '''
    sp = sigmap
    st = np.sqrt(sigmam**2 + sigmap**2)
    rho = np.sqrt(sigmap**2 / (sigmap**2+sigmam**2))
    return 2*(bivariate_norm_cdf(-A/st, T/sp, rho) - bivariate_norm_cdf(-A/st, -T/sp, rho))


def PFR_asymmetric(sigmap, sigmam, mup, TL, TU, AL, AU, testbias=0):
    ''' Probability of false accept for asymmetric tolerances '''
    sp = sigmap
    st = np.sqrt(sigmam**2 + sigmap**2)
    rho = np.sqrt(sigmap**2 / (sigmap**2+sigmam**2))

    if not np.isfinite(AU):
        AU = 1E99
    if not np.isfinite(AL):
        AL = -1E99
    if not np.isfinite(TU):
        TU = 1E99
    if not np.isfinite(TL):
        TL = -1E99

    pfr1 = bivariate_norm_cdf((AL-mup-testbias)/st, (TU-mup)/sp, rho) - bivariate_norm_cdf((AL-mup-testbias)/st, (TL-mup)/sp, rho)
    pfr2 = bivariate_norm_cdf((-AU+mup+testbias)/st, (-TL+mup)/sp, rho) - bivariate_norm_cdf((-AU+mup+testbias)/st, (-TU+mup)/sp, rho)
    return pfr1+pfr2


def PFA_conditional(sigmap, sigmam, mup, TL, TU, AL, AU, testbias=0):
    ''' Conditional probability of false accept (CPFA), sometimes denoted CFAR
        (Conditional False Accept Risk).
    '''
    # CPFA = 1 - P(IT & Accepted) / P(Accepted)
    def normcdf(x):
        return 0.5 * (1 + erf(x/np.sqrt(2)))

    sp = sigmap
    st = np.sqrt(sigmam**2 + sigmap**2)
    rho = np.sqrt(sigmap**2 / (sigmap**2+sigmam**2))

    p_accepted = normcdf((AU-mup-testbias)/st) - normcdf((AL-mup-testbias)/st)

    p_it_accept = (
        bivariate_norm_cdf((AU-mup-testbias)/st, (TU-mup)/sp, rho)
        - bivariate_norm_cdf((AL-mup-testbias)/st, (TU-mup)/sp, rho)
        - bivariate_norm_cdf((AU-mup-testbias)/st, (TL-mup)/sp, rho)
        + bivariate_norm_cdf((AL-mup-testbias)/st, (TL-mup)/sp, rho)
        )

    return 1 - p_it_accept/p_accepted
