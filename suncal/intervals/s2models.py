''' Reliability Model functions for Binomial Method S2

    See Appendix D of NCSLI-RP1.
'''

import numpy as np
from scipy import stats
from scipy.optimize import curve_fit


def R_exp(t, theta):
    ''' Exponential reliability model '''
    return np.exp(-theta * t)


def R_weibull(t, theta1, theta2):
    ''' Weibull reliability model '''
    return np.exp(-(theta1*t)**theta2)


def R_expmixed(t, theta1, theta2):
    ''' Mixed exponential reliability model '''
    return (1+theta1*t)**(-theta2)


def R_walk(t, theta1, theta2):
    ''' Random walk reliability model '''
    Q = 1/np.sqrt(theta1+theta2*t)
    return 2*(stats.norm.cdf(Q)) - 1


def R_restrictedwalk(t, theta1, theta2, theta3):
    ''' Restricted random walk reliability model '''
    Q = 1/np.sqrt(theta1+theta2*(1-np.exp(-theta3*t)))
    return 2*(stats.norm.cdf(Q)) - 1


def R_gamma(t, theta):
    ''' Modified gamma reliability model '''
    return np.exp(-theta*t) * sum([1, theta*t, ((theta*t)**2)/2, ((theta*t)**3)/6])


def R_mortality(t, theta1, theta2):
    ''' Mortality drift reliability model '''
    return np.exp(-(theta1*t + theta2*t**2))


def R_warranty(t, theta1, theta2):
    ''' Warranty reliability model '''
    return 1/(1 + np.exp(theta1*(t-theta2)))


def R_drift(t, theta1, theta2, theta3):
    ''' Drift reliability model '''
    # -1 (-0.5 for each cdf) to shift to the right place (PHI(x) in RP1 = cdf(x)-0.5)
    # RP1 seems to have wrong theta values for the plot in D-11, but 2.5, 2.5, and 0.5 seem to match?
    return stats.norm.cdf(theta1 + theta3*t) + stats.norm.cdf(theta2 - theta3*t) - 1


def R_lognorm(t, theta1, theta2):
    ''' Lognormal reliability model '''
    return 1 - stats.norm.cdf(np.log(theta1*t)/theta2)


# Functions for determining a reasonable initial guess for curve fit
def _condition(y):
    ''' Condition the reliability data to remove exact 1's and 0's that
        don't play well in some of these models
    '''
    y = y.copy()
    y[y >= 1] = .99999
    y[y <= 0] = .00001
    return y


def guess_exp(t, y):
    ''' Generate initial guess and bounds for exponential model '''
    y = _condition(y)
    logy = np.nan_to_num(-np.log(y))  # Linearize and fit straight line
    try:
        theta1 = curve_fit(lambda x, a: x*a, t, logy)[0][0]
    except RuntimeError:
        theta1 = 0.5
    return (theta1,), ((1E-6, 10.),)


def guess_weibull(t, y):
    ''' Generate initial guess for Weibull model '''
    y = _condition(y)
    logy = np.nan_to_num(np.log(-np.log(y)))  # Linearize
    logt = np.nan_to_num(np.log(t))
    coef = np.polyfit(logt, logy, deg=1)
    theta2 = coef[0]
    theta1 = np.exp(coef[1]/theta2)
    try:
        theta1, theta2 = curve_fit(R_weibull, t, y, p0=(theta1, theta2))[0]
    except RuntimeError:
        pass  # Keep original guess

    # theta1 = Constant failure rate. Shouldn't be more than 1
    # theta2 = Shape parameter related to curvature. Higher = steeper.
    return (theta1, theta2), ((0, 1.), (0, 100.))


def guess_expmixed(t, y):
    ''' Generate initial guess for mixed exponential model '''
    # Use Approximation: (1+x)**r ~= (1+xr)
    y = _condition(y)
    theta1theta2 = curve_fit(lambda x, a: x*a, t, 1-y)[0][0]

    # theta2 should be less than 1
    theta2 = .5
    theta1 = theta1theta2/theta2
    return (theta1, theta2), ((0, 1E4), (0, 1))


def guess_walk(t, y):
    ''' Generate initial guess for random walk model '''
    idx = (y > .001)  # Zeros aren't good. Drop them.
    y, t = y[idx], t[idx]
    yy = stats.norm.ppf((y+1)/2)**-2    # Invert the cdf
    theta2, theta1 = np.polyfit(t, yy, deg=1)
    return (theta1, theta2), ((-1E6, 1E6), (-1E6, 1E6))


def guess_rwalk(t, y):
    ''' Generate initial guess for restricted random walk model '''
    # Rough approximation
    y = _condition(y)
    yy = (stats.norm.ppf((y+1)/2))**-2
    theta1 = .0513  # ~stats.norm.ppf((.999+1)/2)**-2   # t->0
    t1plust2 = yy[-1] if np.isfinite(yy[-1]) else np.nanmean(yy)    # t->inf
    theta2 = t1plust2 - theta1
    theta3 = 1/t.mean()   # ~ decay rate

    # Fine tune
    try:
        theta1, theta2, theta3 = curve_fit(R_restrictedwalk, t, y, p0=(theta1, theta2, theta3))[0]
    except RuntimeError:
        pass  # Keep original guess
    return (theta1, theta2, theta3), ((0, 1E6), (0, 1E6), (0, 1))


def guess_gamma(t, y):
    ''' Generate initial guess for modified gamma model '''
    y = _condition(y)
    logy = np.nan_to_num(-np.log(y))   # Ignore the SUM terms to get a rough guess
    try:
        theta1 = curve_fit(lambda x, a: x*a, t, logy)[0][0]
    except RuntimeError:
        theta1 = 0.5
    else:
        # Then fine tune with full model
        try:
            theta1 = curve_fit(R_gamma, t, y, p0=(theta1,))[0][0]
        except RuntimeError:
            pass  # Keep original guess
    return (theta1,), ((0, 1),)


def guess_mortality(t, y):
    ''' Generate initial guess for mortality drift model '''
    y = _condition(y)
    logy = -np.nan_to_num(np.log(y))  # Quadratic after linearizing
    try:
        theta1, theta2 = curve_fit(lambda x, a, b: b*x**2+a*x, t, logy)[0]
    except RuntimeError:
        theta1, theta2 = 0.5
    return (theta1, theta2), ((0, 1), (0, 1))


def guess_warranty(t, y):
    ''' Generate initial guess for warranty model '''
    y = _condition(y)
    ylog = np.log(1/y-1)  # Invert/linearize
    theta1, theta1theta2 = np.polyfit(t, ylog, deg=1)
    theta2 = -theta1theta2/theta1

    # Fine tune
    try:
        theta1, theta2 = curve_fit(R_warranty, t, y, p0=(theta1, theta2))[0]
    except RuntimeError:
        pass
    return (theta1, theta2), ((0, 1E5), (0, 1E5))


def guess_drift(t, y):
    ''' Generate initial guess for drift model '''
    t1overt3 = t.mean()  # Inflection point ~= theta1/theta3
    theta1 = theta2 = 2
    theta3 = theta1/t1overt3
    return (theta1, theta2, theta3), ((0, 1E5), (0, 1E5), (0, 1E5))


def guess_lognorm(t, y):
    ''' Generate initial guess for lognormal model '''
    # Rough approx
    ythresh = (y.max()+y.min())/2
    tthresh = t[np.abs(y-ythresh).argmin()]
    theta1 = 1/tthresh
    theta2 = 1

    # Fine tune
    try:
        theta1, theta2 = curve_fit(R_lognorm, t, y, p0=(theta1, theta2))[0]
    except RuntimeError:
        pass
    return (theta1, theta2), ((0, 1E5), (0, 1E5))


models = {'Exponential': R_exp,
          'Weibull': R_weibull,
          'Mixed Exponential': R_expmixed,
          'Random Walk': R_walk,
          'Restricted Walk': R_restrictedwalk,
          'Modified Gamma': R_gamma,
          'Mortality Drift': R_mortality,
          'Warranty': R_warranty,
          'Drift': R_drift,
          'Log Normal': R_lognorm}

guessers = {'Exponential': guess_exp,
            'Weibull': guess_weibull,
            'Mixed Exponential': guess_expmixed,
            'Random Walk': guess_walk,
            'Restricted Walk': guess_rwalk,
            'Modified Gamma': guess_gamma,
            'Mortality Drift': guess_mortality,
            'Warranty': guess_warranty,
            'Drift': guess_drift,
            'Log Normal': guess_lognorm}