''' Probability Distribution Manager

The Distribution class wraps the different kinds of scipy.stats distributions
(rv_continuous, rv_discrete, and rv_histogram) so they all behave the same.
For example, it provides a pdf() function that works with all three.

Use get_distribution(), given a distribution name, to return an instance
of Distribution class.
'''

import inspect
import numpy as np
import scipy.stats as stats

from . import ttable


def get_argnames(name):
    ''' Get argument names for the distribution '''
    if name in _aliases:
        return _aliases[name].argnames
    else:
        dist = getattr(stats, name)
        return list(inspect.signature(dist._parse_args).parameters.keys())


def get_distribution(name, **kwds):
    ''' Get an instance of the Distribution class.

        Parameters
        ----------
        name: string
            Name of the distribution
        kwds: keyword arguments
            Arguments used to set up the distribution
    '''
    if name in _aliases:
        return _aliases[name](name, **kwds)
    elif isinstance(name, stats.rv_continuous) or isinstance(name, stats.rv_discrete) or isinstance(name, stats.rv_histogram):
        d = Distribution(name.name)
        d.dist = name
        d.update_kwds(**kwds)
        return d
    return Distribution(name, **kwds)


def from_config(config):
    ''' Load a Distribution instance from a config dictionary. '''
    config = config.copy()
    name = config.pop('dist', 'normal')
    return get_distribution(name, **config)


def get_distargs(dist):
    ''' Get dictionary of arguments defining this Distribution instance. '''
    try:
        return dist.get_distargs().copy()  # Instance of Distribution class
    except AttributeError:
        # stats.rv_ class
        kwds = dist.kwds.copy()
        if len(dist.args) > 0:
            argnames = dist.dist.shapes.split(', ') if dist.dist.shapes is not None else []
            argnames = argnames + ['loc', 'scale']
            for i, arg in enumerate(dist.args):
                kwds[argnames[i]] = arg
        return kwds


def fittable(name):
    ''' Determine if the distribution has a fit() function. '''
    dist = get_distribution(name)
    return dist.can_fit()


class Distribution(object):
    ''' Distribution class for handling stats distributions

        Works for rv_continuous, rv_discrete, and rv_histogram, plus
        can be subclassed to work on distributions with wrapped arguments
        (for example define a uniform distribution using median and a
        parameters rather than loc and scale).

        Parameters
        ----------
        name: string
            Name of the distribution
        kwds: keyword arguments
            Keywords passed to the scipy distribution
    '''
    showshift = False

    def __init__(self, name, **kwds):
        self.name = name
        self.kwds = kwds
        self.distargs = None  # Set in update_kwds
        self.update_kwds(**kwds)

    def __getattr__(self, name):
        ''' Get attribute. Passed to the frozen distribution so this class
            behaves similar to rv_frozen. Allows access to things like ppf(),
            cdf(), rvs(), etc.
        '''
        dfrozen = self.dist(**self.distargs)
        return getattr(dfrozen, name)

    def update_kwds(self, **kwds):
        ''' Update the distribution keywords.

            Parameters
            ----------
            kwds: keyword arguments
                Arguments passed to scipy.stats distribution

            Subclass this to provide wrappers around arguments and
            allow definition of things like uniform from median and "a"
            rather than loc and scale.
        '''
        self.dist = getattr(stats, self.name)
        self.argnames = get_argnames(self.name)
        self.kwds.update(kwds)  # User-specified arguments
        self.distargs = self.kwds  # Arguments needed for scipy.stats

        # Remove unused arguments and set defaults for required args not in kwds
        [self.distargs.pop(k) for k in list(self.distargs.keys()) if k not in self.argnames]
        [self.distargs.setdefault(k, 1) for k in self.argnames if k not in self.distargs]

    def set_median(self, median):
        ''' Set the median value of the distribution. Calculates the correct "loc" keyword. '''
        zeroargs = self.distargs.copy()
        zeroargs['loc'] = 0
        zeromed = self.dist(**zeroargs).median()
        self.distargs['loc'] = median - zeromed
        if 'median' in self.kwds:
            self.kwds['median'] = median
        elif 'loc' in self.kwds:
            self.kwds['loc'] = self.distargs['loc']

    def set_mean(self, mean):
        ''' Set the mean value of the distribution. Calculates the correct "loc" keyword. '''
        zeroargs = self.distargs.copy()
        zeroargs['loc'] = 0
        zeromean = self.dist(**zeroargs).mean()
        self.distargs['loc'] = mean - zeromean
        if 'mean' in self.kwds:
            self.kwds['mean'] = mean
        elif 'loc' in self.kwds:
            self.kwds['loc'] = self.distargs['loc']

    def set_shift(self, shift):
        ''' Set shift of the distribiton by setting loc parameter '''
        self.kwds['shift'] = shift
        self.distargs['loc'] = shift

    def get_distargs(self):
        ''' Return arguments used scipy.stats distribution '''
        return self.distargs

    def get_config(self):
        ''' Get configuration dictionary (args plus distribution name) '''
        d = self.kwds.copy()
        d.update({'dist': self.name})
        return d

    def fit(self, x):
        ''' Find best fitting parameters for the distribution to the sampled x data.
            Changes the parameters of this Distribution instance.
        '''
        if not self.can_fit():
            raise ValueError(f'Distribution `{self.name}` is not fitable.')

        params = self.dist.fit(x)
        pnames = list(inspect.signature(self.dist._parse_args).parameters.keys())
        self.kwds = dict(zip(pnames, params))
        self.update_kwds(**self.kwds)
        return self.kwds

    def can_fit(self):
        ''' Determine whether this Distribution can be fit to sampled data. '''
        return hasattr(self.dist, 'fit')

    def pdf(self, x):
        ''' Get probability density at x. Uses sampling/histogram technique for discrete distributions. '''
        _dfrozen = self.dist(**self.distargs)
        try:
            pdf = _dfrozen.pdf(x)
        except AttributeError:
            # Discrete dist
            try:
                _x = np.atleast_1d(x)
                samples = _dfrozen.rvs(10000).astype(float)
                xx = np.concatenate((_x, [_x[-1] + np.diff(_x)[-1]]))
                pdf, xx = np.histogram(samples, bins=xx, density=True)
            except ValueError:
                pdf = np.full(len(_x), np.nan)

            if np.isscalar(x):
                pdf = pdf[0]
        return pdf

    def helpstr(self):
        ''' Get a help string for this Distribution. '''
        helpstr = self.dist.__class__.__doc__.split('\n')
        helpstr = '\n'.join(l for l in helpstr if not l.startswith('    %'))
        return helpstr


class DNormal(Distribution):
    ''' Normal distribution defined by one of the following sets of parameters:

        - unc and conf
        - unc and k
        - std (same as unc with k=1)
        - scale (same as std)
    '''
    dist = stats.norm
    argnames = ['std']

    def update_kwds(self, **kwds):
        ''' Update the keywords for the distribution. '''
        self.kwds.update(kwds)
        if 'unc' in self.kwds:
            if 'conf' in self.kwds:
                # Calculate confidence based on infinite degf since this is normal distribution
                k = ttable.t_factor(self.kwds['conf'], np.inf)
                std = self.kwds['unc']/k
            else:
                k = self.kwds.get('k', 1)
                std = self.kwds['unc']/k
        elif 'std' in self.kwds:
            std = self.kwds.get('std')
        elif 'scale' in self.kwds:
            std = self.kwds.get('scale')
        else:
            std = 1
        std = max(std, 1E-99)  # don't allow 0 standard deviation - to prevent nans

        self.distargs = {'loc': self.kwds.get('median', self.kwds.get('loc', 0)), 'scale': std}

    def fit(self, x):
        ''' Find best fitting parameters for the distribution to the sampled x data.
            Changes the parameters of this Distribution instance.
        '''
        loc, scale = self.dist.fit(x)
        self.distargs = {'loc': loc, 'scale': scale}
        self.kwds = {'median': loc, 'std': scale}
        return self.kwds

    def helpstr(self):
        return 'Normal (Gaussian) distribution. Specify uncertainty and either k or confidence in percent.'


class Dt(Distribution):
    ''' Student's T distribution defined by one of the following sets of parameters:

        - unc, conf, df
        - unc, k, df
        - std, df (same as unc with k=1 and df=inf)
        - scale, df (same as std)
    '''
    dist = stats.t
    argnames = ['std', 'df']

    def update_kwds(self, **kwds):
        ''' Update the keywords for the distribution. '''
        self.kwds.update(kwds)
        df = self.kwds.get('df', np.inf)
        df = max(2.00001, df)  # No div/0 or negative scales later
        if 'unc' in self.kwds:   # UNC and DF, but no conf or K......
            if 'conf' in self.kwds:
                k = ttable.t_factor(self.kwds['conf'], df)
                std = self.kwds['unc']/k
            else:
                k = self.kwds.get('k', 1)
                std = self.kwds['unc']/k
        elif 'std' in self.kwds:
            std = self.kwds.get('std')
        elif 'scale' in self.kwds:
            std = self.kwds.get('scale') * np.sqrt(df/(df-2))
        else:
            std = 1

        std = max(std, 1E-99)  # don't allow 0 standard deviation - to prevent nans
        scale = std / np.sqrt(df/(df-2))
        self.distargs = {'loc': self.kwds.get('median', self.kwds.get('loc', 0)), 'scale': scale, 'df': df}

    def fit(self, x):
        ''' Find best fitting parameters for the distribution to the sampled x data.
            Changes the parameters of this Distribution instance.
        '''
        df, loc, scale = self.dist.fit(x)
        self.distargs = {'loc': loc, 'scale': scale, 'df': df}
        self.kwds = {'median': loc, 'std': scale * np.sqrt(df/(df-2)), 'df': df}
        return self.kwds

    def helpstr(self):
        return "Student's T distribution. Specify uncertainty and either k or confidence in percent."


class DUniform(Distribution):
    ''' Uniform distribution defined by median and half-width "a". '''
    dist = stats.uniform
    argnames = ['a']  # 'median' is also an argname, but not required.

    def update_kwds(self, **kwds):
        ''' Update the keywords for the distribution. '''
        self.kwds.update(kwds)
        a = self.kwds.get('a', 1)
        median = self.kwds.get('median', self.kwds.get('loc', 0))
        self.distargs = {'loc': median-a, 'scale': a*2}

    def fit(self, x):
        ''' Find best fitting parameters for the distribution to the sampled x data. '''
        loc, scale = self.dist.fit(x)
        self.distargs = {'loc': loc, 'scale': scale}
        self.kwds = {'a': scale/2, 'median': loc+scale/2}
        return self.kwds

    def helpstr(self):
        return 'Uniform distribution from -a to +a.'


class DTriangular(Distribution):
    ''' Symmetric triangular distribution from -a to 0 to +a. '''
    dist = stats.triang
    argnames = ['a']

    def update_kwds(self, **kwds):
        ''' Update the keywords for the distribution. '''
        self.kwds.update(kwds)
        a = self.kwds.get('a', 1)
        median = self.kwds.get('median', self.kwds.get('loc', 0))
        self.distargs = {'loc': median-a, 'scale': a*2, 'c': 0.5}

    def fit(self, x):
        ''' Find best fitting parameters for the distribution to the sampled x data. '''
        c, loc, scale = self.dist.fit(x, f0=0.5)  # Keep c parameter 0.5 for symmetric triangle
        self.distargs = {'loc': loc, 'scale': scale, 'c': 0.5}
        self.kwds = {'a': scale/2, 'median': loc+scale/2}
        return self.kwds

    def helpstr(self):
        return 'Symmetric triangular distribution from -a to 0 to a.'


class DArcsine(Distribution):
    ''' Arcsine distribution from -a to 0 to +a. '''
    dist = stats.arcsine
    argnames = ['a']

    def update_kwds(self, **kwds):
        ''' Update the keywords for the distribution. '''
        self.kwds.update(kwds)
        a = self.kwds.get('a', 1)
        median = self.kwds.get('median', self.kwds.get('loc', 0))
        self.distargs = {'loc': median-a, 'scale': a*2}

    def fit(self, x):
        ''' Find best fitting parameters for the distribution to the sampled x data. '''
        loc, scale = self.dist.fit(x)
        self.distargs = {'loc': loc, 'scale': scale}
        self.kwds = {'a': scale/2, 'median': loc+scale/2}
        return self.kwds

    def helpstr(self):
        return 'Arcsine distribution from -a to +a'


class DResolution(Distribution):
    ''' Resolution distribution, same as uniform with half-width a/2 '''
    dist = stats.uniform
    argnames = ['a']

    def update_kwds(self, **kwds):
        ''' Update the keywords for the distribution. '''
        self.kwds.update(kwds)
        a = self.kwds.get('a', 1)
        median = self.kwds.get('median', self.kwds.get('loc', 0))
        self.distargs = {'loc': median-a/2, 'scale': a}

    def fit(self, x):
        ''' Find best fitting parameters for the distribution to the sampled x data. '''
        loc, scale = self.dist.fit(x)
        self.distargs = {'loc': loc, 'scale': scale}
        self.kwds = {'a': scale, 'median': loc+scale}
        return self.kwds

    def helpstr(self):
        return ''' Uncertainty due to digital equipment resolution. The 'a' parameter is\nincrement of least-significant digit in a digital display.\n\nSame as a uniform distribution with half-width a/2.'''


class DExpon(Distribution):
    ''' Exponential distribution defined by lambda parameter '''
    dist = stats.expon
    showshift = True
    argnames = ['lambda']

    def update_kwds(self, **kwds):
        ''' Update keywords for the distribution '''
        lam = kwds.get('lambda', 1)
        self.kwds.update(kwds)
        self.distargs = {'scale': 1/lam}

    def fit(self, x):
        ''' Find best fitting parameters for the distribution to the sampled x data '''
        loc, scale = self.dist.fit(x)
        self.distargs = {'loc': loc, 'scale': scale}
        self.kwds = {'lambda': 1/scale, 'shift': loc}
        return self.kwds

    def helpstr(self):
        return '''Exponential distribution with rate parameter lambda. (lambda > 0)

PDF(x) = lambda * exp(lambda*x)'''


class DPoisson(Distribution):
    ''' Poisson discrete distribution defined by parameter "v". '''
    dist = stats.poisson
    argnames = ['v']
    showshift = True

    def update_kwds(self, **kwds):
        ''' Update the keywords for the distribution. '''
        self.kwds.update(kwds)
        v = self.kwds.get('v', 1)
        self.distargs = {'loc': -v, 'mu': v}  # Since Median value will be applied later, shift loc to -v to compensate.

    def helpstr(self):
        return 'Poisson discrete distribution with variance v. In general, measured value equals the variance.'


class DBinom(Distribution):
    ''' Binomial distribution defined by "n" and "p". '''
    # Need this one since stats.binom() will crash if n is float. This just casts it to int.
    dist = stats.binom
    argnames = ['n', 'p']

    def update_kwds(self, **kwds):
        self.kwds.update(kwds)
        n = self.kwds.get('n', 1)
        p = self.kwds.get('p', 0.5)
        self.distargs = {'n': int(n), 'p': p}

    def helpstr(self):
        return 'Binomial discrete distribution.'


class DHistogram(Distribution):
    ''' Histogram approximation to a distribution. Keywords can be one of:

        - histogram: tuple of (hist, edges) resulting from np.histogram
        - hist, edges: histogram data resulting from np.histogram
        - data: sampled data. histogram calculated from np.histogram(data, bins='auto')
    '''
    dist = stats.rv_histogram
    argnames = ['data']

    def update_kwds(self, **kwds):
        ''' Update the keywords for the distribution. '''
        self.kwds = kwds
        if 'histogram' in self.kwds:
            hist, edges = self.kwds['histogram']
            self.kwds['hist'] = list(hist)
            self.kwds['edges'] = list(edges)
            self.distargs = {'histogram': self.kwds['histogram']}
        elif 'data' in self.kwds:
            hist, edges = np.histogram(kwds.get('data'), bins='auto')
            self.kwds['hist'] = list(hist)
            self.kwds['edges'] = list(edges)
            self.distargs = {'histogram': (list(hist), list(edges))}
        elif 'hist' in self.kwds and 'edges' in self.kwds:
            hist = self.kwds['hist']
            edges = self.kwds['edges']
            self.distargs = {'histogram': (list(hist), list(edges))}
        else:
            self.distargs = {'histogram': ([1], [0, 1])}

    def fit(self, x):
        ''' Find best fitting parameters for the distribution to the sampled x data. '''
        hist, edges = np.histogram(x, bins='auto')
        self.kwds = {'hist': hist, 'edges': edges}
        self.distargs = {'histogram': (list(hist), list(edges))}
        return self.distargs

    def set_median(self, median):
        ''' Set the median value of the histogram distribution '''
        hist, edges = self.distargs['histogram']
        zeromed = self.dist(**self.distargs).median()
        shift = median - zeromed
        newedges = edges + shift
        self.distargs['histogram'] = hist, newedges
        self.kwds['edges'] = newedges
        self.kwds['median'] = median

    def get_config(self):
        ''' Get configuration dictionary (args plus distribution name) '''
        d = {'hist': list(self.kwds['hist']),
             'edges': list(self.kwds['edges']),
             'dist': self.name}
        return d

    def helpstr(self):
        return 'Histogram approximation to a probability distribution.'


class DPiecewise(DHistogram):
    ''' Piecewise distribution approximation. Keywords of x and pdf. '''
    dist = stats.rv_histogram
    argnames = ['x', 'pdf']

    def update_kwds(self, **kwds):
        ''' Update the keywords for the distribution. '''
        x = kwds.get('x')
        pdf = kwds.get('pdf')
        if len(x) == len(pdf):
            # Bin edges should be n+1 length
            x = np.append(x, x[-1] + (x[1]-x[0]))
        self.kwds = {'hist': pdf, 'edges': x}
        self.distargs = {'histogram': (pdf, x)}


class DGamma(Distribution):
    ''' Gamma distribution defined by alpha and beta. '''
    dist = stats.gamma
    argnames = ['alpha', 'beta']
    showshift = True

    def update_kwds(self, **kwds):
        self.kwds.update(kwds)
        a = self.kwds.get('alpha', 1.0)
        b = self.kwds.get('beta', 1.0)
        self.distargs = {'a': a, 'scale': 1/b}
    
    def fit(self, x):
        ''' Find best fitting parameters for the distribution to the sampled x data. '''
        s, loc, scale = self.dist.fit(x)
        self.distargs = {'a': s, 'loc': loc, 'scale': scale}
        self.kwds = {'alpha': s, 'beta': 1/scale, 'shift': loc}
        return self.kwds

    def helpstr(self):
        return '''Gamma distribution with shape parameters alpha and beta.

When based on measurement data with mean y and standard deviaion s, alpha = y^2/s^2 and beta = y/s^2.

PDF(x) = (beta^alpha * x^(alpha-1) * exp(beta*x)) / (Gamma(alpha)) for x, beta, alpha > 0'''


class DLognorm(Distribution):
    ''' Lognormal distribution defined by mean and standard deviation
        of natural log of random variable X.
    '''
    dist = stats.lognorm
    argnames = ['mu', 'std']
    showshift = True

    def update_kwds(self, **kwds):
        self.kwds.update(kwds)
        std = self.kwds.get('std', 1.0)  # std is stdev of ln(X)
        mu = self.kwds.get('mu', 1.0)    # mu is expectation of ln(X)
        self.distargs = {'s': std, 'scale': np.exp(mu)}
    
    def fit(self, x):
        ''' Find best fitting parameters for the distribution to the sampled x data. '''
        s, loc, scale = self.dist.fit(x)
        self.distargs = {'s': s, 'scale': scale, 'loc': loc}
        self.kwds = {'std': s, 'mu': np.log(scale), 'shift': loc}
        return self.kwds

    def helpstr(self):
        return 'Lognormal distribution where mu and std are the expected value and\nstandard deviation of the natural log of the random variable.'


class DBeta(Distribution):
    ''' Beta Distribution '''
    dist = stats.beta
    argnames = ['a', 'b', 'scale']
    showshift = True

    def update_kwds(self, **kwds):
        self.kwds.update(kwds)
        a = self.kwds.get('a', 1.0)
        b = self.kwds.get('b', 1.0)
        scale = self.kwds.get('scale', 1)
        self.distargs = {'a': a, 'b': b, 'scale': scale}

    def fit(self, x):
        ''' Find best fitting parameters for the distribution to the sampled x data. '''
        a, b, loc, scale = self.dist.fit(x)
        self.distargs = {'a': a, 'b': b, 'loc': loc, 'scale': scale}
        self.kwds = {'a': a, 'b': b, 'shift': loc}
        return self.kwds

    def helpstr(self):
        return 'Beta distribution with positive shape parameters a and b.'


# Curvilinear Trapezoid Distribution
# Scipy doesn't have this one, so need to subclass rv_continuous and make one ourselves.
# This one uses loc, scale, and d as parameters,
# but DCurvTrap class translates that to half-width a and curvature d.
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

    def _rvs(self, d, size=None, random_state=None):
        ''' Generate random variates.

            See GUM-Supplement 1, 6.4.3.4.
        '''
        a_s = random_state.uniform(low=0, high=d*2, size=size)
        r2 = random_state.uniform(size=size)
        b_s = 1 - a_s
        return a_s + (b_s - a_s)*r2
ctrap = _ctrap_gen(a=0, b=1, name='ctrap')


class DCurvTrap(Distribution):
    ''' Curvilinear trapezoidal distribution, defined by parameters "a" and "d", where d < a. '''
    dist = ctrap  # Defined below
    argnames = ['a', 'd']

    def update_kwds(self, **kwds):
        ''' Update the keywords for the distribution. '''
        self.kwds.update(kwds)
        a = self.kwds.get('a', 1)
        d = self.kwds.get('d', 0.5)
        median = self.kwds.get('median', 0)
        loc = -a-d+median
        scale = 2*a + 2*d
        self.distargs = {'loc': loc, 'scale': scale, 'd': d/scale}

    def fit(self, x):
        ''' Find best fitting parameters for the distribution to the sampled x data. '''
        xmax = x.max()
        xmin = x.min()

        # self.dist.fit doesn't actually fit the "d" parameter
        # here we make a complete guess. Won't be great, but will return a valid distribution.
        loc = xmin
        scale = xmax-xmin
        d = (xmax-xmin)/6

        median = loc+scale/2
        self.distargs = {'loc': loc, 'scale': scale, 'd': d/scale}
        self.kwds = {'median': median, 'a': d*2, 'd': d}
        return self.kwds

    def helpstr(self):
        return 'Curvilinear Trapezoid Distribution\n\n    a: half-width of trapezoid (excluding curvature)\n\nd: curvature (i.e. uncertainty in a, must be less than a)'


# Lookup Distribution subclass from name.
_aliases = {
    'uniform': DUniform,
    'normal': DNormal,
    't': Dt,
    'triangular': DTriangular,
    'gamma': DGamma,
    'beta': DBeta,
    'lognorm': DLognorm,
    'expon': DExpon,
    'curvtrap': DCurvTrap,
    'ctrap': DCurvTrap,
    'arcsine': DArcsine,
    'resolution': DResolution,
    'poisson': DPoisson,
    'binom': DBinom,
    'histogram': DHistogram,
    'hist': DHistogram,
    'piecewise': DPiecewise,
    }
