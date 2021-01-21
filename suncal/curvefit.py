''' Calculations for finding uncertainty in fitted curves.

The CurveFit class computes fit function and uncertainty for any
arbitrary curve function.

CurveFitParam objects specify a single coefficient from a CurveFit as the value
of interest to pull, for example the slope value, into an UncertCalc instance.
'''

from collections import namedtuple
import inspect
import numpy as np
import sympy
from scipy import odr
import scipy.optimize
import yaml

from . import out_curvefit
from . import uncertarray
from . import uparser
from .uncertarray import Array  # Explicitly import Array so it can be accessed via curvefit.Array


Fit = namedtuple('Fit', ['coeff', 'covariance'])

FitResids = namedtuple('FitResiduals', ['residuals', 'Syx', 'r', 'F', 'SSres', 'SSreg'])
FitOut = namedtuple('FitOutput', ['coeff', 'uncert', 'covariance', 'degf', 'residuals', 'samples', 'acceptance'], defaults=(None,)*7)


class CurveFit(object):
    ''' Fitting an arbitrary function curve to measured data points and computing
        uncertainty in the fit parameters.

        Parameters
        ----------
        arr: Array object
            The array of data points to operate on

        func: string or callable
            Function to fit data to. For common functions, give a string name of the function,
            one of: (line, quad, cubic, poly, exp, decay, log, logistic). A 'poly' must also provide
            the polyorder argument.

            Any other string will be evaluated as an expression, and must contain an 'x' variable.

            If func is callable, arguments must take the form func(x, *args) where x is array
            of independent variable and *args are parameters of the fit. For example a quadratic
            fit using a lambda function:
            lambda x, a, b, c: return a + b*x + c*x**2

        name: string
            Name of the function

        desc: string
            Description of the function

        polyorder: int
            Order for polynomial fit. Only required for fit == 'poly'.

        p0: list, optional
            Initial guess for function parameters

        method: string, optional
            Method passed to scipy.curve_fit

        bounds: 2-tuple, optional
            Upper and Lower bounds for fit parameters, passed to scipy.curve_fit. If specified,
            will also set priors for MCMC method to uniform between bounds. (Ignored with ODR
            fitting method).

        odr: bool
            Force use of orthogonal regression

        absolute_sigma: boolean
            Treat uncertainties in an absolute sense. If false, only relative
            magnitudes matter.

        Notes
        -----
        Uses scipy.optimize.curve_fit or scipy.odr to do the fitting, depending on
        if the array has uncertainty in x (or if odr parameter is True). p0 is required if ODR is used.

        Saving and loading to config file is only supported when func is given as a string.
    '''
    def __init__(self, arr, func='line', name='curvefit', desc='', polyorder=None, p0=None, method=None,
                 bounds=None, odr=None, seed=None, absolute_sigma=True):
        self.seed = seed
        self.samples = 5000
        self.outputs = {}
        self.arr = arr
        self.name = name
        self.desc = desc
        self.out = None
        self.xname = 'x'
        self.yname = 'y'
        self.absolute_sigma = absolute_sigma
        self.set_fitfunc(func, polyorder=polyorder, p0=p0, method=method, bounds=bounds, odr=odr)

    def set_fitfunc(self, func, polyorder=2, method=None, bounds=None, odr=None, p0=None):
        ''' Set up fit function '''
        self.fitname = func
        self.polyorder = polyorder
        self.odr = odr
        self.bounds = bounds
        self.p0 = p0

        if callable(func):
            self.fitname = 'callable'

        elif self.fitname == 'line':
            self.expr = sympy.sympify('a + b*x')
            def func(x, b, a):
                return a + b*x

        elif self.fitname == 'exp':  # Full exponential
            self.expr = sympy.sympify('c + a * exp(x/b)')
            def func(x, a, b, c):
                return c + a * np.exp(x/b)

        elif self.fitname == 'decay':  # Exponential decay to zero (no c parameter)
            self.expr = sympy.sympify('a * exp(-x/b)')
            def func(x, a, b):
                return a * np.exp(-x/b)

        elif self.fitname == 'decay2':  # Exponential decay, using rate lambda rather than time constant tau
            self.expr = sympy.sympify('a * exp(-x*b)')
            def func(x, a, b):
                return a * np.exp(-x*b)

        elif self.fitname == 'log':
            self.expr = sympy.sympify('a + b * log(x-c)')
            def func(x, a, b, c):
                return a + b * np.log(x-c)

        elif self.fitname == 'logistic':
            self.expr = sympy.sympify('a / (1 + exp((x-c)/b)) + d')
            def func(x, a, b, c, d):
                return d + a / (1 + np.exp((x-c)/b))

        elif self.fitname == 'quad' or (func == 'poly' and polyorder == 2):
            self.expr = sympy.sympify('a + b*x + c*x**2')
            def func(x, a, b, c):
                return a + b*x + c*x*x

        elif self.fitname == 'cubic' or (func == 'poly' and polyorder == 3):
            self.expr = sympy.sympify('a + b*x + c*x**2 + d*x**3')
            def func(x, a, b, c, d):
                return a + b*x + c*x*x + d*x*x*x

        elif self.fitname == 'poly':
            def func(x, *p):
                return np.poly1d(p[::-1])(x)  # Coeffs go in reverse order (...e, d, c, b, a)

            polyorder = int(polyorder)
            if polyorder < 1 or polyorder > 12:
                raise ValueError('Polynomial order out of range')
            varnames = [chr(ord('a')+i) for i in range(polyorder+1)]
            self.expr = sympy.sympify('+'.join(v+'*x**{}'.format(i) for i, v in enumerate(varnames)))

            # variable *args must have initial guess for scipy
            if self.p0 is None:
                self.p0 = np.ones(polyorder+1)
        else:
            # actual expression as string
            func, self.expr, _ = self.parse_math(self.fitname)

        self.func = func

        if self.fitname == 'poly' and polyorder > 3:
            # poly def above doesn't have named arguments, so the inspect won't find them. Name them here.
            self.pnames = varnames
        else:
            self.pnames = list(inspect.signature(self.func).parameters.keys())[1:]
        self.numparams = len(self.pnames)

        if self.fitname == 'callable':
            self.expr = sympy.sympify('f(x, ' + ', '.join(self.pnames) + ')')

        if self.bounds is None:
            bounds = (-np.inf, np.inf)
        else:
            bounds = self.bounds
            self.set_mcmc_priors([lambda x, a=blow, b=bhi: (x > a) & (x <= b) for blow, bhi in zip(bounds[0], bounds[1])])

        if self.fitname == 'line' and not odr:
            # use generic LINE fit for lines with no odr
            self.fitfunc = lambda x, y, ux, uy, absolute_sigma=self.absolute_sigma: genlinefit(x, y, ux, uy, absolute_sigma=absolute_sigma)
        else:
            self.fitfunc = lambda x, y, ux, uy, absolute_sigma=self.absolute_sigma: genfit(self.func, x, y, ux, uy, p0=self.p0, method=method, bounds=bounds, odr=odr, absolute_sigma=absolute_sigma)

        return self.expr

    @classmethod
    def parse_math(cls, expr):
        ''' Check expr string for a valid curvefit function including an x variable
            and at least one fit parameter.

            Returns
            -------
            func: callable
                Lambdified function of expr
            symexpr: sympy object
                Sympy expression of function
            argnames: list of strings
                Names of arguments (except x) to function
        '''
        uparser.parse_math(expr)  # Will raise if not valid expression
        symexpr = sympy.sympify(expr)
        argnames = sorted(str(s) for s in symexpr.free_symbols)
        if 'x' not in argnames:
            raise ValueError('Expression must contain "x" variable.')
        argnames.remove('x')
        if len(argnames) == 0:
            raise ValueError('Expression must contain one or more parameters to fit.')
        func = sympy.lambdify(['x'] + argnames, symexpr, 'numpy')  # Make sure to specify 'numpy' so nans are returned instead of complex numbers
        ParsedMath = namedtuple('ParsedMath', ['function', 'sympyexpr', 'argnames'])
        return ParsedMath(func, symexpr, argnames)

    def clear(self):
        ''' Clear the sampled points '''
        self.arr.clear()
        self.outputs = {}

    def get_output(self):
        ''' Get output object '''
        return self.out

    def run_uyestimate(self):
        if (self.arr.uy_estimate is None and (not self.arr.has_ux() and not self.arr.has_uy())):
            # Estimate uncertainty using residuals if uy not provided. LSQ method does this already,
            # do the same for GUM and MC.
            self.arr.uy_estimate = self.estimate_uy()

    def calculate(self, **kwargs):
        ''' Calculate curve fit by different methods and display the results.
            Only least-squares analytical method is calculated by default.

            Keyword Arguments
            -----------------
            lsq: bool
                Calculate analytical Least Squares method
            gum: bool
                Calculate GUM method
            mc: bool
                Calculate Monte Carlo method
            mcmc: bool
                Calculate Markov-Chain Monte Carlo method
            samples: int
                Number of Monte Carlo samples

            Returns
            -------
            FuncOutput object
        '''
        self.run_uyestimate()
        self.samples = kwargs.get('samples', self.samples)
        self.out = out_curvefit.CurveFitOutput(self, self.arr, **kwargs)
        return self.out

    def calc_LSQ(self):
        ''' Calculate analytical Least-Squares curve fit and uncertainty.

            Returns
            -------
            CurveFitOutput object
        '''
        uy = np.zeros(len(self.arr.x)) if not self.arr.has_uy() else self.arr.uy
        coeff, cov = self.fitfunc(self.arr.x, self.arr.y, self.arr.ux, uy)

        resids = (self.arr.y - self.func(self.arr.x, *coeff))  # All residuals (NOT squared)
        sigmas = np.sqrt(np.diag(cov))
        degf = len(self.arr.x) - len(coeff)
        if self.absolute_sigma or not self.arr.has_uy():
            w = np.full(len(self.arr.x), 1)  # Unweighted residuals in Syx
        else:
            w = (1/uy**2)  # Determine weighted Syx
            w = w/sum(w) * len(self.arr.y)      # Normalize weights so sum(wi) = N
        SSres = sum(w*resids**2)   # Sum-of-squares of residuals
        Syx = np.sqrt(SSres/degf)  # Standard error of the estimate (based on residuals)
        SSreg = sum(w*(self.func(self.arr.x, *coeff) - sum(w*self.arr.y)/sum(w))**2)
        r = np.sqrt(1-SSres/(SSres+SSreg))
        resids = FitResids(resids, Syx, r, SSreg*degf/SSres, SSres, SSreg)
        out = FitOut(coeff, sigmas, cov, degf, resids)
        self.outputs['lsq'] = out
        return out

    def sample(self, samples=1000):
        ''' Generate Monte Carlo samples '''
        self.arr.clear()
        self.arr.sample(samples)

    def estimate_uy(self):
        ''' Calculate an estimate for uy using residuals of fit for when uy is not given.
            This is what linefit() method does behind the scenes, this function allows the
            same behavior for GUM and Monte Carlo.
        '''
        pcoeff, _ = self.fitfunc(self.arr.x, self.arr.y, ux=None, uy=None)
        uy = np.sqrt(np.sum((self.func(self.arr.x, *pcoeff) - self.arr.y)**2)/(len(self.arr.x) - len(pcoeff)))
        uy = np.full(len(self.arr.x), uy)
        return uy

    def calc_MC(self, samples=1000, sensitivity=False):
        ''' Calculate Monte Carlo curve fit and uncertainty.

            Parameters
            ----------
            samples: int
                Number of Monte Carlo samples

            Returns
            -------
            CurveFitOutput object
        '''
        self.run_uyestimate()
        uy = self.arr.uy if self.arr.uy_estimate is None else self.arr.uy_estimate
        if self.arr.xsamples is None or self.arr.ysamples is None or self.arr.xsamples.shape[1] != samples:
            self.sample(samples)

        self.samplecoeffs = np.zeros((samples, self.numparams))
        for i in range(samples):
            self.samplecoeffs[i], _ = self.fitfunc(self.arr.xsamples[:, i], self.arr.ysamples[:, i], ux=None, uy=None)

        coeff = self.samplecoeffs.mean(axis=0)
        sigma = self.samplecoeffs.std(axis=0, ddof=1)

        resids = (self.arr.y - self.func(self.arr.x, *coeff))
        degf = len(self.arr.x) - len(coeff)
        if self.absolute_sigma or not self.arr.has_uy():
            w = np.full(len(self.arr.x), 1)  # Unweighted residuals in Syx
        else:
            w = (1/uy**2)  # Determine weighted Syx
            w = w/sum(w) * len(self.arr.y)   # Normalize weights so sum(wi) = N
        cov = np.cov(self.samplecoeffs.T)
        SSres = sum(w*resids**2)   # Sum-of-squares of residuals
        Syx = np.sqrt(SSres/degf)  # Standard error of the estimate (based on residuals)
        SSreg = sum(w*(self.func(self.arr.x, *coeff) - sum(w*self.arr.y)/sum(w))**2)
        r = np.sqrt(1-SSres/(SSres+SSreg))

        resids = FitResids(resids, Syx, r, SSreg*degf/SSres, SSres, SSreg)
        out = FitOut(coeff, sigma, cov, degf, resids, self.samplecoeffs)

        self.outputs['mc'] = out
        return out

    def calc_MCMC(self, samples=10000, burnin=0.2):
        ''' Calculate Markov-Chain Monte Carlo (Metropolis-in-Gibbs algorithm)
            fit parameters and uncertainty

            Parameters
            ----------
            samples: int
                Total number of samples to generate
            burnin: float
                Fraction of samples to reject at start of chain

            Returns
            -------
            CurveFitOutput object

            Notes
            -----
            Currently only supported with constant u(y) and u(x) = 0.
        '''
        if self.seed is not None:
            np.random.seed(self.seed)

        self.run_uyestimate()
        uy = self.arr.uy if self.arr.uy_estimate is None else self.arr.uy_estimate

        if self.arr.has_ux():
            print('WARNING - MCMC algorithm ignores u(x) != 0')
        if np.max(uy) != np.min(uy):
            print('WARNING - MCMC algorithm with non-constant u(y). Using mean.')

        # Find initial guess/sigmas
        p, cov = self.fitfunc(self.arr.x, self.arr.y, self.arr.ux, uy)
        up = np.sqrt(np.diag(cov))
        if not all(np.isfinite(up)):
            raise ValueError('MCMC Could not determine initial sigmas. Try providing p0.')

        if all(uy == 0):
            # Sigma2 is unknown. Estimate from residuals and vary through trace.
            resids = (self.arr.y - self.func(self.arr.x, *p))
            sig2 = resids.var(ddof=1)
            sresid = np.std(np.array([self.arr.y, self.func(self.arr.x, *p)]), axis=0)
            sig2sig = 2*np.sqrt(sig2)
            sig2lim = np.percentile(sresid, 5)**2, np.percentile(sresid, 95)**2
            varysigma = True
        else:
            # Sigma2 (variance of data) is known. Use it and don't vary sigma during trace.
            sig2 = uy.mean()**2
            varysigma = False

        if not hasattr(self, 'priors') or self.priors is None:
            priors = [lambda x: 1 for i in range(len(self.pnames))]
        else:
            priors = [p if p is not None else lambda x: 1 for p in self.priors]

        for pidx in range(len(p)):
            if priors[pidx](p[pidx]) <= 0:
                # Will get div/0 below
                raise ValueError('Initial prior for parameter {} is < 0'.format(self.pnames[pidx]))

        accepts = np.zeros(len(p))
        self.mcmccoeffs = np.zeros((samples, self.numparams))
        self.sig2trace = np.zeros(samples)
        for i in range(samples):
            for pidx in range(len(p)):
                pnew = p.copy()
                pnew[pidx] = pnew[pidx] + np.random.normal(scale=up[pidx])

                Y = self.func(self.arr.x, *p)  # Value using p (without sigma that was appended to p)
                Ynew = self.func(self.arr.x, *pnew)  # Value using pnew

                # NOTE: could use logpdf, but it seems slower than manually writing it out:
                # problog = stat.norm.logpdf(self.arr.y, loc=I, scale=np.sqrt(sig2).sum()
                problog = -1/(2*sig2) * sum((self.arr.y - Y)**2)
                problognew = -1/(2*sig2) * sum((self.arr.y - Ynew)**2)

                r = np.exp(problognew-problog) * priors[pidx](pnew[pidx]) / priors[pidx](p[pidx])
                if r >= np.random.uniform():
                    p = pnew
                    accepts[pidx] += 1

            if varysigma:
                sig2new = sig2 + np.random.normal(scale=sig2sig)
                if (sig2new < sig2lim[1] and sig2new > sig2lim[0]):
                    Y = self.func(self.arr.x, *p)
                    ss2 = sum((self.arr.y - Y)**2)
                    problog = -1/(2*sig2) * ss2
                    problognew = -1/(2*sig2new) * ss2
                    if np.exp(problognew - problog) >= np.random.uniform():
                        sig2 = sig2new

            self.mcmccoeffs[i, :] = p
            self.sig2trace[i] = sig2
        burnin = int(burnin * samples)
        self.mcmccoeffs = self.mcmccoeffs[burnin:, :]
        self.sig2trace = self.sig2trace[burnin:]

        coeff = self.mcmccoeffs.mean(axis=0)
        sigma = self.mcmccoeffs.std(axis=0, ddof=1)
        resids = (self.arr.y - self.func(self.arr.x, *coeff))
        degf = len(self.arr.x) - len(coeff)
        if self.absolute_sigma or not self.arr.has_uy():
            w = np.full(len(self.arr.x), 1)  # Unweighted residuals in Syx
        else:
            w = (1/uy**2)  # Determine weighted Syx
            w = w/sum(w) * len(self.arr.y)   # Normalize weights so sum(wi) = N
        cov = np.cov(self.mcmccoeffs.T)
        SSres = sum(w*resids**2)   # Sum-of-squares of residuals
        Syx = np.sqrt(SSres/degf)  # Standard error of the estimate (based on residuals)
        SSreg = sum(w*(self.func(self.arr.x, *coeff) - sum(w*self.arr.y)/sum(w))**2)
        r = np.sqrt(1-SSres/(SSres+SSreg))
        resids = FitResids(resids, Syx, r, SSreg*degf/SSres, SSres, SSreg)
        out = FitOut(coeff, sigma, cov, degf, resids, self.mcmccoeffs, accepts/samples)
        self.outputs['mcmc'] = out
        return out

    def set_mcmc_priors(self, priors):
        ''' Set prior distribution functions for each input to be used in
            Markov-Chain Monte Carlo.

            Parameters
            ----------
            priors: list of callables
                List of functions, one for each fitting parameter. Each function must
                take a possible fit parameter as input and return the probability
                of that parameter from 0-1.

            Notes
            -----
            If set_mcmc_priors is not called, all priors will return 1.
        '''
        assert len(priors) == len(self.pnames)
        self.priors = priors

    def calc_GUM(self):
        ''' Calculate curve fit and uncertainty using GUM Approximation.

            Returns
            -------
            output: CurveFitOutput object
        '''
        self.run_uyestimate()
        uy = self.arr.uy if self.arr.uy_estimate is None else self.arr.uy_estimate

        coeff, cov, grad = uncertarray._GUM(lambda x, y: self.fitfunc(x, y, ux=None, uy=None)[0], self.arr.x, self.arr.y, self.arr.ux, uy)
        sigmas = np.sqrt(np.diag(cov))
        resids = (self.arr.y - self.func(self.arr.x, *coeff))
        degf = len(self.arr.x) - len(coeff)
        if self.absolute_sigma or not self.arr.has_uy():
            w = 1  # Unweighted residuals in Syx
        else:
            w = (1/uy**2)  # Determine weighted Syx
            w = w/sum(w) * len(self.arr.y)   # Normalize weights so sum(wi) = N
        SSres = sum(w*resids**2)
        Syx = np.sqrt(SSres/degf)  # Standard error of the estimate (based on residuals)
        SSreg = sum(w*(self.func(self.arr.x, *coeff) - self.arr.y.mean())**2)
        r = np.sqrt(1-SSres/(SSres+SSreg))

        resids = FitResids(resids, Syx, r, SSreg*degf/SSres, SSres, SSreg)
        out = FitOut(coeff, sigmas, cov, degf, resids)

        self.outputs['gum'] = out
        return out

    def get_config(self):
        if self.fitname == 'callable':
            raise ValueError('Saving CurveFit only supported for line, poly, and exp fits.')

        d = {}
        d['mode'] = 'curvefit'
        d['curve'] = self.fitname
        d['name'] = self.name
        d['desc'] = self.desc
        d['odr'] = self.odr
        d['xname'] = self.xname
        d['yname'] = self.yname
        d['xdates'] = self.arr.xdate
        d['abssigma'] = self.absolute_sigma
        if self.fitname == 'poly':
            d['order'] = self.polyorder
        if self.p0 is not None:
            d['p0'] = self.p0
        if self.bounds is not None:
            d['bound0'] = self.bounds[0]
            d['bound1'] = self.bounds[1]

        d['arrx'] = self.arr.x.astype('float').tolist()  # Can't yaml numpy arrays, use list
        d['arry'] = self.arr.y.astype('float').tolist()
        if self.arr.has_ux():
            d['arrux'] = list(self.arr.ux)
        if self.arr.has_uy():
            d['arruy'] = list(self.arr.uy)
        return d

    def save_config(self, fname):
        ''' Save configuration to file.

            Parameters
            ----------
            fname: string or file
                File name or file object to save to
        '''
        d = self.get_config()
        out = yaml.dump([d], default_flow_style=False)
        try:
            fname.write(out)
        except AttributeError:
            with open(fname, 'w') as f:
                f.write(out)

    @classmethod
    def from_config(cls, config):
        arr = Array(np.asarray(config.get('arrx'), dtype=float),
                    np.asarray(config.get('arry'), dtype=float),
                    ux=config.get('arrux', 0.),
                    uy=config.get('arruy', 0.),
                    xdate=config.get('xdates', False))
        newfit = cls(arr, config['curve'],
                     polyorder=config.get('order', 2),
                     name=config.get('name', None),
                     desc=config.get('desc', ''),
                     p0=config.get('p0', None),
                     odr=config.get('odr', None),
                     seed=config.get('seed', None),
                     absolute_sigma=config.get('abssigma', True))
        newfit.xname = config.get('xname', 'x')
        newfit.yname = config.get('yname', 'y')
        return newfit

    @classmethod
    def from_configfile(cls, fname):
        ''' Read and parse the configuration file. Returns a new UncertRisk
            instance.

            Parameters
            ----------
            fname: string or file
                File name or open file object to read configuration from
        '''
        try:
            try:
                yml = fname.read()  # fname is file object
            except AttributeError:
                with open(fname, 'r') as fobj:  # fname is string
                    yml = fobj.read()
        except UnicodeDecodeError:
            # file is binary, can't be read as yaml
            return None

        try:
            config = yaml.safe_load(yml)
        except yaml.YAMLError:
            return None  # Can't read YAML

        u = cls.from_config(config[0])  # config yaml is always a list
        return u


# Functions for fitting curves
#------------------------------------------------------------
def odrfit(func, x, y, ux, uy, p0=None, absolute_sigma=True):
    ''' Fit the curve using scipy's orthogonal distance regression (ODR)

        Parameters
        ----------
        func: callable
            The function to fit. Must take x as first argument, and other
            parameters as remaining arguments.
        x, y: arrays
            X and Y data to fit
        ux, uy: arrays
            Standard uncertainty in x and y
        p0: array
            Initial guess of parameters.
        absolute_sigma: boolean
            Treat uncertainties in an absolute sense. If false, only relative
            magnitudes matter.

        Returns
        -------
        pcoeff: array
            Coefficients of best fit curve
        pcov: array
            Covariance of coefficients. Standard error of coefficients is
            np.sqrt(np.diag(pcov)).
    '''
    # Wrap the function because ODR puts params first, x last
    def odrfunc(B, x):
        return func(x, *B)

    if ux is not None and all(ux == 0):
        ux = None
    if uy is not None and all(uy == 0):
        uy = None

    model = odr.Model(odrfunc)
    mdata = odr.RealData(x, y, sx=ux, sy=uy)
    modr = odr.ODR(mdata, model, beta0=p0)
    mout = modr.run()
    if mout.info != 1:
        print('Warning - ODR failed to converge')

    if absolute_sigma:
        # SEE: https://github.com/scipy/scipy/issues/6842.
        # If this issue is fixed, these options may be swapped!
        cov = mout.cov_beta
    else:
        cov = mout.cov_beta*mout.res_var
    ODR = namedtuple('ODR', ['coeff', 'covariance'])
    return ODR(mout.beta, cov)


def genfit(func, x, y, ux, uy, p0=None, method=None, bounds=(-np.inf, np.inf), odr=None, absolute_sigma=True):
    ''' Generic curve fit. Selects scipy.optimize.curve_fit if ux==0 or scipy.odr otherwise.

        Parameters
        ----------
        func: callable
            The function to fit
        x, y: arrays
            X and Y data to fit
        ux, uy: arrays
            Standard uncertainty in x and y
        p0: array-like
            Initial guess parameters
        absolute_sigma: boolean
            Treat uncertainties in an absolute sense. If false, only relative
            magnitudes matter.

        Returns
        -------
        pcoeff: array
            Coefficients of best fit curve
        pcov: array
            Covariance of coefficients. Standard error of coefficients is
            np.sqrt(np.diag(pcov)).
    '''
    if odr or not (ux is None or all(ux == 0)):
        return odrfit(func, x, y, ux, uy, p0=p0, absolute_sigma=absolute_sigma)
    else:
        if uy is None or all(uy == 0):
            return Fit(*scipy.optimize.curve_fit(func, x, y, p0=p0, bounds=bounds))
        else:
            return Fit(*scipy.optimize.curve_fit(func, x, y, sigma=uy, absolute_sigma=absolute_sigma, p0=p0, bounds=bounds))


def genlinefit(x, y, ux, uy, absolute_sigma=True):
    ''' Generic straight line fit. Uses linefit() if ux==0 or linefitYork otherwise.

        Parameters
        ----------
        func: callable
            The function to fit
        x, y: arrays
            X and Y data to fit
        ux, uy: arrays
            Standard uncertainty in x and y
        absolute_sigma: boolean
            Treat uncertainties in an absolute sense. If false, only relative
            magnitudes matter.

        Returns
        -------
        pcoeff: array
            Coefficients of best fit curve
        pcov: array
            Covariance of coefficients. Standard error of coefficients is
            np.sqrt(np.diag(pcov)).
    '''
    if ux is None or all(ux == 0):
        return linefit(x, y, sig=uy, absolute_sigma=absolute_sigma)
    else:
        return linefitYork(x, y, sigx=ux, sigy=uy, absolute_sigma=absolute_sigma)


def linefit(x, y, sig, absolute_sigma=True):
    ''' Fit a line with uncertainty in y (but not x)

        Parameters
        ----------
        x: array
            X values of fit
        y: array
            Y values of fit
        sig: array
            uncertainty in y values
        absolute_sigma: boolean
            Treat uncertainties in an absolute sense. If false, only relative
            magnitudes matter.

        Returns
        -------
        coeff: array
            Coefficients of line fit [slope, intercept].
        cov: array 2x2
            Covariance matrix of fit parameters. Standard error is
            np.sqrt(np.diag(cov)).

        Note
        ----
        Returning coeffs and covariance so the return value matches scipy.optimize.curve_fit.
        With sig=0, this algorithm estimates a sigma using the residuals.

        References
        ----------
        [1] Numerical Recipes in C, The Art of Scientific Computing. Second Edition.
            W. H. Press, S. A. Teukolsky, W. T. Vetterling, B. P. Flannery.
            Cambridge University Press. 2002.
    '''
    sig = np.atleast_1d(sig)
    if len(sig) == 1:
        sig = np.full(len(x), sig[0])
    if all(sig) > 0:
        wt = 1./sig**2
        ss = sum(wt)
        sx = sum(x*wt)
        sy = sum(y*wt)
        sxoss = sx/ss
        t = (x-sxoss)/sig
        st2 = sum(t*t)
        b = sum(t*y/sig)/st2
    else:
        sx = sum(x)
        sy = sum(y)
        ss = len(x)
        sxoss = sx/ss
        t = (x-sxoss)
        st2 = sum(t*t)
        b = sum(t*y)/st2
    a = (sy-sx*b)/ss
    siga = np.sqrt((1+sx*sx/(ss*st2))/ss)
    sigb = np.sqrt(1/st2)

    resid = sum((y-a-b*x)**2)
    syx = np.sqrt(resid/(len(x)-2))
    cov = -sxoss * sigb**2
    if not all(sig) > 0:
        siga = siga * syx
        sigb = sigb * syx
        cov = cov * syx*syx
    elif not absolute_sigma:
        # See note in scipy.optimize.curve_fit for absolute_sigma parameter.
        chi2 = sum(((y-a-b*x)/sig)**2)/(len(x)-2)
        siga, sigb, cov = np.sqrt(siga**2*chi2), np.sqrt(sigb**2*chi2), cov*chi2
    #rab = -sxoss * sigb / siga  # Correlation can be computed this way
    return Fit(np.array([b, a]), np.array([[sigb**2, cov], [cov, siga**2]]))


def linefitYork(x, y, sigx=None, sigy=None, rxy=None, absolute_sigma=True):
    ''' Find a best-fit line through the x, y points having
        uncertainties in both x and y. Also accounts for
        correlation between the uncertainties. Uses York's algorithm.

        Parameters
        ----------
        x: array
            X values to fit
        y: array
            Y values to fit
        sigx: array or float
            Uncertainty in x values
        sigy: array or float
            Uncertainty in y values
        rxy: array or float, optional
            Correlation coefficient between sigx and sigy
        absolute_sigma: boolean
            Treat uncertainties in an absolute sense. If false, only relative
            magnitudes matter.

        Returns
        -------
        coeff: array
            Coefficients of line fit [slope, intercept].
        cov: array 2x2
            Covariance matrix of fit parameters. Standard error is
            np.sqrt(np.diag(cov)).

        Note
        ----
        Returning coeffs and covariance so the return value matches scipy.optimize.curve_fit.
        Implemented based on algorithm in [1] and pseudocode in [2].

        References
        ----------
        [1] York, Evensen. Unified equations for the slope, intercept, and standard
            errors of the best straight line. American Journal of Physics. 72, 367 (2004)
        [2] Wehr, Saleska. The long-solved problem of the best-fit straight line:
            application to isotopic mixing lines. Biogeosciences. 14, 17-29 (2017)
    '''
    # Condition inputs so they're all float64 arrays
    x = x.astype(np.float64)
    y = y.astype(np.float64)
    if sigx is None or len(np.nonzero(sigx)[0]) == 0:
        sigx = np.full(len(x), 1E-99, dtype=np.float64)   # Don't use 0, but a really small number
    elif np.isscalar(sigx):
        sigx = np.full_like(x, sigx)

    if sigy is None or len(np.nonzero(sigy)[0]) == 0:
        sigy = np.full(len(y), 1E-99, dtype=np.float64)
    elif np.isscalar(sigy):
        sigy = np.full_like(x, sigy)

    sigy = np.maximum(sigy, 1E-99)
    sigx = np.maximum(sigx, 1E-99)

    if rxy is None:
        rxy = np.zeros_like(y)
    elif np.isscalar(rxy):
        rxy = np.full_like(x, rxy)

    _, b0 = np.polyfit(x, y, deg=1)  # Get initial estimate for slope
    T = 1E-15
    b = b0
    bdiff = np.inf

    wx = 1./sigx**2
    wy = 1./sigy**2
    alpha = np.sqrt(wx*wy)
    while bdiff > T:
        bold = b
        w = alpha**2/(b**2 * wy + wx - 2*b*rxy*alpha)
        sumw = sum(w)
        X = sum(w*x)/sumw
        Y = sum(w*y)/sumw
        U = x - X
        V = y - Y
        beta = w * (U/wy + b*V/wx - (b*U + V)*rxy/alpha)
        Q1 = sum(w*beta*V)
        Q2 = sum(w*beta*U)
        b = Q1/Q2
        bdiff = abs((b-bold)/bold)
    a = Y - b*X

    # Uncertainties
    xi = X + beta
    xbar = sum(w*xi) / sumw
    sigb = np.sqrt(1./sum(w * (xi - xbar)**2))
    siga = np.sqrt(xbar**2 * sigb**2 + 1/sumw)
    #resid = sum((y-b*x-a)**2)

    # Correlation bw a, b
    #rab = -xbar * sigb / siga
    cov = -xbar * sigb**2

    if not absolute_sigma:
        # See note in scipy.optimize.curve_fit for absolute_sigma parameter.
        chi2 = sum(((y-a-b*x)*np.sqrt(w))**2)/(len(x)-2)
        siga, sigb, cov = np.sqrt(siga**2*chi2), np.sqrt(sigb**2*chi2), cov*chi2
    return Fit(np.array([b, a]), np.array([[sigb**2, cov], [cov, siga**2]]))
