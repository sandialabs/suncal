''' Functions for calculating t-table values '''

import numpy as np
from scipy.special import nctdtridf
from scipy import stats


def t_factor(conf, degf):
    ''' Return tp(v) given confidence (0-1) and degrees of freedom.

        Parameters
        ----------
        conf: float
            Level of confidence (0-1).
        degf: float
            Degrees of freedom

        Returns
        -------
        tp: float
            Value of tp(v)
    '''
    if not np.isfinite(degf):
        degf = 1E99
    return stats.t.ppf(1-(1-conf)/2, df=degf)


def t_onetail(conf, degf):
    if not np.isfinite(degf):
        degf = 1E99
    return stats.t.ppf(conf, df=degf)


def confidence(tp, degf):
    ''' Get confidence value given tp and degrees of freedom. Inverse of t_factor.

        Parameters
        ----------
        tp: float
            Value of tp(v)
        degf: float
            Degrees of freedom

        Returns
        -------
        conf: float
            Confidence value in the range (0-1).
    '''
    if not np.isfinite(degf):
        degf = 1E99
    return 1+2*(stats.t.cdf(tp, df=degf)-1)


def degf(tp, conf):
    ''' Calculate degrees of freedom given tp and confidence

        Parameters
        ----------
        tp: float
            Value of tp(v)
        conf: float
            Level of confidence (0-1).

        Returns
        -------
        degf: float
            Degrees of freedom
    '''
    # Non-central t distribution with non-centrality parameter of 0.
    df = nctdtridf(1-(1-conf)/2, 0, tp)
    return df if df < 1E12 else np.inf
