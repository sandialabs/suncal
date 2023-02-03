''' Deaver used different definition for TUR and Specification Limit.
    These functions allow recreating plots in his papers

    The functions PFA_deaver and PFR_deaver use the equations in Deaver's "How to
    "Maintain Confidence" paper, which require specification limits in terms of
    standard deviations of the process distribution, and use a slightly different
    definition for TUR. These functions are provided for convenience when working
    with this definition.
'''

import math
from scipy.integrate import dblquad


def PFA_deaver(SL, TUR, GB=1):
    ''' Calculate Probability of False Accept (Consumer Risk) for normal
        distributions given spec limit and TUR, using Deaver's equation.

        Args:
            SL (float): Specification Limit in terms of standard deviations,
              symmetric on each side of the mean
            TUR (float): Test Uncertainty Ratio (sigma_uut / sigma_test). Note this is
              definition used by Deaver's papers, NOT the typical SL/(2*sigma_test) definition.
            GB (float): Guardband factor (0-1) with 1 being no guardband

        Returns:
            PFA (float): Probability of False Accept

        Reference:
            Equation 6 in Deaver - How to Maintain Confidence
    '''
    c, _ = dblquad(lambda y, t: math.exp(-(y*y + t*t)/2) / math.pi, SL, math.inf,
                   gfun=lambda t: -TUR*(t+SL*GB), hfun=lambda t: -TUR*(t-SL*GB))
    return c


def PFR_deaver(SL, TUR, GB=1):
    ''' Calculate Probability of False Reject (Producer Risk) for normal
        distributions given spec limit and TUR, using Deaver's equation.

        Args:
            SL (float): Specification Limit in terms of standard deviations,
              symmetric on each side of the mean
            TUR (float): Test Uncertainty Ratio (sigma_uut / sigma_test). Note this is
              definition used by Deaver's papers, NOT the typical SL/(2*sigma_test) definition.
            GB (float): Guardband factor (0-1) with 1 being no guardband

        Returns:
            PFR (float): Probability of False Reject

        Reference:
            Equation 7 in Deaver - How to Maintain Confidence
    '''
    p, _ = dblquad(lambda y, t: math.exp(-(y*y + t*t)/2) / math.pi, -SL, SL,
                   gfun=lambda t: TUR*(GB*SL-t), hfun=lambda t: math.inf)
    return p
