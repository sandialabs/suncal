''' TUR-based guardbands, returning guardband factor as multipler of
    the plus-minus tolerance
'''
import numpy as np
from scipy import stats
from scipy.optimize import fsolve

from .risk import PFA_norm


def dobbert(tur: float) -> float:
    ''' Calculate guardband factor using Dobbert's method, also known as
        "Method 6". Guarantees 2% PFA or less for any in-tolerance
        probability.

        GBF = 1 - (1.04 - exp(0.38 log(tur) - 0.54)) / TUR

        Args:
            tur: Test Uncertainty Ratio (Tolerance / Uncertainty)
    '''
    M = 1.04 - np.exp(0.38 * np.log(tur) - 0.54)
    return 1 - M / tur


def rss(tur: float) -> float:
    ''' Calculate guardband factor using RSS method

        GBF = sqrt(1-1/tur**2)

        Args:
            tur: Test Uncertainty Ratio (Tolerance / Uncertainty)
    '''
    return np.sqrt(1-1/tur**2)


def test95(tur: float) -> float:
    ''' Calculate guardband using 95% test uncertainty method
        (same as subtracting the 95% uncertainty from the tolerance)

        GBF = 1 - 1/TUR

        Args:
            tur: Test Uncertainty Ratio (Tolerance / Uncertainty)
    '''
    return 1-1/tur if tur <= 4 else 1


def rp10(tur: float) -> float:
    ''' Calculate guardband using NCSLI RP-10 (similar to test95, but
        less conservative)

        GBF = 1.25 - 1/TUR

        Args:
            tur: Test Uncertainty Ratio (Tolerance / Uncertainty)
    '''
    return 1.25 - 1/tur if tur <= 4 else 1


def four_to_1(tur: float, itp: float = 0.95) -> float:
    ''' Calculate guardband that results in the same PFA as if the TUR
        was 4:1.

        Args:
            tur: Test Uncertainty Ratio (Tolerance / Uncertainty)
            itp: In-tolerance probability (also called
                end-of-period-reliability)
    '''
    target = PFA_norm(itp, TUR=4)
    return pfa_target(tur, itp, target)


def pfa_target(tur: float, itp: float = 0.95, pfa: float = 0.08) -> float:
    ''' Calculate guardband required to acheive the desired PFA

        Args:
            tur: Test Uncertainty Ratio (Tolerance / Uncertainty)
            itp: In-tolerance probability (also called
                end-of-period-reliability)
            pfa: Desired Probability of False Accept
    '''
    return fsolve(lambda x: PFA_norm(itp, tur, GB=x)-pfa, x0=.8)[0]


def mincost(tur: float, itp: float = 0.95, cc_over_cp: float = 10) -> float:
    ''' Calculate guardband using the Mincost method, minimizing the
        total expected cost due to all false decisions.
        Reference Easterling 1991.

        Args:
            tur: Test Uncertainty Ratio (Tolerance / Uncertainty)
            itp: In-tolerance probability (also called
                end-of-period-reliability)
            cc_over_cp: Ratio of cost of a false accept to cost of the part
    '''
    conf = 1 - (1 / (1 + cc_over_cp))
    sigtest = 1/tur/2
    sigprod = 1/stats.norm.ppf((1+itp)/2)
    k = stats.norm.ppf(conf) * np.sqrt(1 + sigtest**2/sigprod**2) - sigtest/sigprod**2
    return 1 - k * sigtest


def minimax(tur: float, cc_over_cp: float = 10) -> float:
    ''' Calculate guardband by minimizing the maximum expected cost due
        to all false decisions
        Reference Easterling 1991.

        Args:
            tur: Test Uncertainty Ratio (Tolerance / Uncertainty)
            cc_over_cp: Ratio of cost of a false accept to cost of the part
    '''
    conf = 1 - (1 / (1 + cc_over_cp))
    k = stats.norm.ppf(conf)
    return 1 - k * (1/tur/2)
