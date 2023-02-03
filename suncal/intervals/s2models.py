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
def guess_exp(t, y):
    ''' Generate initial guess for exponential model '''
    y = np.nan_to_num(-np.log(y))  # Linearize and fit straight line
    theta1 = curve_fit(lambda x, a: x*a, t, y)[0][0]
    return theta1


def guess_weibull(t, y):
    ''' Generate initial guess for Weibull model '''
    y[y == 1] = .99999
    t = np.nan_to_num(np.log(t))
    y = np.nan_to_num(np.log(-np.log(y)))  # Linearize
    coef = np.polyfit(t, y, deg=1)
    theta2 = coef[0]
    theta1 = np.exp(coef[1]/theta2)
    return theta1, theta2


def guess_expmixed(t, y):
    ''' Generate initial guess for mixed exponential model '''
    # Use Approximation: (1+x)**r ~= (1+xr)
    y = 1-y
    theta1theta2 = curve_fit(lambda x, a: x*a, t, y)[0][0]
    theta1 = theta1theta2/2
    theta2 = theta1theta2/theta1
    return theta1, theta2


def guess_walk(t, y):
    ''' Generate initial guess for random walk model '''
    yy = stats.norm.ppf((y+1)/2)**-2    # Invert the cdf
    theta1 = yy[yy > 0][0]**2
    theta2 = 1/t[np.argmin(abs(y-(y.max()+y.min())/2))]
    return theta1, theta2


def guess_rwalk(t, y):
    ''' Generate initial guess for restricted random walk model '''
    y = (stats.norm.ppf((y+1)/2))**-2
    theta1 = np.nanmin(y)   # t->0
    t1plust2 = y[-1] if np.isfinite(y[-1]) else np.nanmean(y)    # t->inf
    theta2 = t1plust2 - theta1
    theta3 = 1/t.mean()     # ~ decay rate
    return theta1, theta2, theta3


def guess_gamma(t, y):
    ''' Generate initial guess for modified gamma model '''
    yy = np.nan_to_num(-np.log(y))   # Ignore the SUM terms...
    theta1 = curve_fit(lambda x, a: x*a, t, yy)[0][0]
    return theta1


def guess_mortality(t, y):
    ''' Generate initial guess for mortality drift model '''
    yy = np.nan_to_num(np.log(y))  # Quadratic after linearizing
    theta1, theta2 = curve_fit(lambda x, a, b: b*x**2-a*x, t, yy)[0]
    return theta1, theta2


def guess_warranty(t, y):
    ''' Generate initial guess for warranty model '''
    yy = np.nan_to_num(np.log(1/y-1))  # Invert/linearize
    theta1, theta1theta2 = np.polyfit(t, yy, deg=1)
    theta2 = -theta1theta2/theta1
    return theta1, theta2


def guess_drift(t, y):
    ''' Generate initial guess for drift model '''
    t1overt3 = t.mean()  # Inflection point ~= theta1/theta3
    theta1 = theta2 = 2
    theta3 = theta1/t1overt3
    return theta1, theta2, theta3


def guess_lognorm(t, y):
    ''' Generate initial guess for lognormal model '''
    ythresh = (y.max()+y.min())/2
    tthresh = t[np.abs(y-ythresh).argmin()]
    theta1 = 1/tthresh
    theta2 = 1/(t.max()/2)
    return theta1, theta2


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