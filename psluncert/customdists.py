''' Customized distributions

Most functions generate a scipy.stats distribution but shift the loc and
scale parameters to something more user-friendly.

Functions get_config, from_config convert a distribution to/from a dictionary
of parameters with 'dist' keyword defining the distribution name and other
keywords defining the distribution parameters.

The get_dist function gets a distribution by name, selecting either a customized
one from this module or a standard one from scipy.stats.
'''

import numpy as np
import scipy.stats as stats


def get_config(dist):
    ''' Get configuration dictionary for the distribution. Uses customized
        parameter names if distribution was made with a function in this module.
    '''
    if hasattr(dist, 'customkwds'):
        config = dist.customkwds
        config.update({'median': dist.median()})
    elif isinstance(dist, stats.rv_histogram):
        config = {'dist': 'histogram', 'hist': list(dist._histogram[0]), 'edges': list(dist._histogram[1])}
    else:
        config = {'dist': dist.dist.name}
        config.update(dist.kwds)
    return config


def from_config(config):
    ''' Get a distribution instance from the configuration dictionary. Opposite of
        get_config.
    '''
    config = config.copy()
    name = config.pop('dist', 'norm')
    if name == 'histogram':
        dist = stats.rv_histogram((config.get('hist'), config.get('edges')))
    elif name in globals() and 'loc' not in config:   # if loc in args, it's a stats distribution (ie uniform, t, arcsine that are in both)
        median = config.pop('median', 0)
        dist = globals()[name](**config)
        dist.kwds['loc'] = median - (dist.median() - dist.kwds['loc'])
    else:
        dist = getattr(stats, name)(**config)
    return dist


def get_dist(name):
    ''' Get root distribution rv_continuous given name '''
    try:
        dist = getattr(stats, name)
    except AttributeError:
        dist = {'normal': stats.norm,
                'triangular': stats.triang,
                'resolution': stats.uniform,
                'histogram': hist,
                'curvtrap': ctrap}.get(name, None)
    return dist


# NOTE: MUST specify loc=, scale= as KEYWORD arguments
def uniform(a):
    ''' Uniform distribution from -a to +a. '''
    dist = stats.uniform(loc=-a, scale=a*2)
    dist.customkwds = {'dist': 'uniform', 'a': a}   # Save the original parameters for saving/loading
    return dist


def normal(std):
    ''' Normal (Gaussian) distribution. Specify uncertainty and either k or confidence in percent. '''
    dist = stats.norm(loc=0, scale=std)
    dist.customkwds = {'dist': 'normal', 'std': std}
    return dist


def triangular(a):
    ''' Symmetric triangular distribution from -a to 0 to a. '''
    dist = stats.triang(loc=-a, scale=a*2, c=.5)
    dist.customkwds = {'dist': 'triangular', 'a': a}
    return dist


def t(std, df):
    ''' Student's T distribution. Specify uncertainty and either k or confidence in percent. Specify degrees of freedom (>2) in Measured Quantities table. '''
    if df <= 2:
        df = 2.001
    dist = stats.t(loc=0, scale=std/np.sqrt(df/(df-2)), df=df)
    dist.customkwds = {'dist': 't', 'std': std, 'df': df}
    return dist


def arcsine(a):
    ''' Arcsine distribution from -a to +a '''
    dist = stats.arcsine(loc=-a, scale=a*2)
    dist.customkwds = {'dist': 'arcsine', 'a': a}
    return dist


def resolution(a):
    ''' Uncertainty due to digital equipment resolution. The 'a' parameter is
        increment of least-significant digit in a digital display.

        Same as a uniform distribution with half-width a/2.
    '''
    dist = stats.uniform(loc=-a/2, scale=a)
    dist.customkwds = {'dist': 'resolution', 'a': a}
    return dist


def curvtrap(a, d):
    ''' Curvilinear Trapezoid Distribution

        a: half-width of trapezoid (excluding curvature)
        d: curvature (i.e. uncertainty in a, must be less than a)
    '''
    loc = -a-d
    scale = 2*a + 2*d
    dist = ctrap(loc=loc, scale=scale, d=d/scale)
    dist.customkwds = {'dist': 'curvtrap', 'a': a, 'd': d}
    return dist


def poisson(v):
    ''' Poisson discrete distribution with variance v. In general, mean should equal variance. '''
    dist = stats.poisson(loc=-v, mu=v)
    dist.customkwds = {'dist': 'poisson', 'v': v}
    return dist


def hist(data):
    ''' Histogram-based distribution

        data: 1D array of data
    '''
    hist, edges = np.histogram(data, bins='auto')
    dist = stats.rv_histogram((hist, edges))
    dist.customkwds = {'dist': 'histogram', 'hist': list(hist), 'edges': list(edges)}
    return dist


def piecewise(x, pdf):
    ''' Piecewise distribution, uses rv_histogram

        x: x values (bin edges)
        pdf: pdf values for bin
    '''
    if len(x) == len(pdf):
        # Bin edges should be n+1 length
        x = np.append(x, x[-1] + (x[1]-x[0]))
    dist = stats.rv_histogram((pdf, x))
    dist.customkwds = {'dist': 'piecewise', 'pdf': pdf}
    return dist


# Curvilinear Trapezoid Distribution
# Scipy doesn't have this one, so need to subclass rv_continuous and make one ourselves.
# This one uses loc, scale, and d as parameters,
# but curvtrap() function above translates that to half-width a and curvature d.
class _ctrap_gen(stats.rv_continuous):
    ''' Curvilinear Trapezoidal Distribution.

        Like a trapezoid that:
            starts at loc,
            increases to loc+2*d,
            flat to loc + scale - 2*d
            decreases to loc + scale
        Except sides are curved by parameter d (d<scale/2)
    '''
    def _argcheck(self, d):
        return d < 0.25

    def _pdf(self, x, d):
        ''' Probability Density Function.

            See GUM-Supplement 1, 6.4.3.2, eq. 3
            Using loc=0, scale=1, a=d, b=1-d.
        '''
        c = 1/(4*d)
        w = 0.5 - d
        condlist = [x < d*2, x < 1-d*2, x <= 1]
        choicelist = [
            c * np.log((w + d)/(0.5-x)),   # 0.5 is loc + scale/2
            c * np.log((w + d)/(w-d)),
            c * np.log((w + d)/(x-0.5))]
        return np.select(condlist, choicelist)

    def _stats(self, d):
        ''' Compute statistics. If this isn't defined, it will try to integrate PDF numerically
            and take forever.

            See GUM-Supplement 1, 6.4.3.3 for variance
        '''
        return 0.5, (1-2*d)**2/12 + d**2/9, None, None  # mean, variance, [skew, kertosis]

    def _rvs(self, d):
        ''' Generate random variates.

            See GUM-Supplement 1, 6.4.3.4.
        '''
        a_s = self._random_state.uniform(low=0, high=d*2, size=self._size)
        r2 = self._random_state.uniform(size=self._size)
        b_s = 1 - a_s
        return a_s + (b_s - a_s)*r2
ctrap = _ctrap_gen(a=0, b=1, name='ctrap')
