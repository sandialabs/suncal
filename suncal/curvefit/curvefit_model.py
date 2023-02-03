''' Curve Fit Model '''

from collections import namedtuple
import inspect
import numpy as np
import sympy

from ..common import uparser
from . import uncertarray
from .curvefit import fit, linefit
from .results.curvefit import CurveFitResults, CurveFitResultsCombined


FitResids = namedtuple('FitResiduals', ['residuals', 'Syx', 'r', 'F', 'SSres', 'SSreg'])
FitResults = namedtuple('FitResults', ['coeff', 'uncert', 'covariance',
                                       'degf', 'residuals', 'samples', 'acceptance'], defaults=(None,)*7)
FitSetup = namedtuple('FitSetup', ['points', 'expression', 'function', 'modelname', 'coeffnames', 'xname', 'yname'])


class CurveFit:
    ''' Fitting an arbitrary function curve to measured data points and computing
        uncertainty in the fit parameters.

        Args:
            arr (Array): The array of data points to operate on
            func (string or callable): Function to fit data to. For common functions, give a string
                name of the function, one of: (line, quad, cubic, poly, exp, decay, log, logistic).
                A 'poly' must also provide the polyorder argument.
                Any other string will be evaluated as an expression, and must contain an 'x' variable.
                If func is callable, arguments must take the form func(x, *args) where x is array
                of independent variable and *args are parameters of the fit. For example a quadratic
                fit using a lambda function:
                lambda x, a, b, c: return a + b*x + c*x**2
            polyorder (int): Order for polynomial fit. Only required for fit == 'poly'.
            p0 (list): Initial guess for function parameters
            bounds (2-tuple): Upper and Lower bounds for fit parameters, passed to scipy.curve_fit.
                If specified, will also set priors for MCMC method to uniform between bounds. (Ignored
                with ODR fitting method).
            odr (bool): Force use of orthogonal regression
            absolute_sigma (boolean): Treat uncertainties in an absolute sense. If false, only relative
                magnitudes matter.

        Notes:
            Uses scipy.optimize.curve_fit or scipy.odr to do the fitting, depending on
            if the array has uncertainty in x (or if odr parameter is True). p0 is required if ODR is used.
    '''
    def __init__(self, arr, func='line', polyorder=None, p0=None,
                 bounds=None, odr=None, absolute_sigma=True):
        self.arr = arr
        self.xname = 'x'
        self.yname = 'y'
        self.absolute_sigma = absolute_sigma
        self.set_fitfunc(func, polyorder=polyorder, p0=p0, bounds=bounds, odr=odr)

    def fitsetup(self):
        ''' Get setup parameters. See class for arguments '''
        return FitSetup(self.arr, self.expr, self.func, self.modelname, self.pnames, self.xname, self.yname)

    def set_fitfunc(self, func, polyorder=2, bounds=None, odr=None, p0=None):
        ''' Set up fit function '''
        self.modelname = func
        self.polyorder = polyorder
        self.odr = odr
        self.bounds = bounds
        self.p0 = p0

        if callable(func):
            self.modelname = 'callable'

        elif self.modelname == 'line':
            self.expr = sympy.sympify('a + b*x')

            def func(x, b, a):
                return a + b*x

        elif self.modelname == 'exp':  # Full exponential
            self.expr = sympy.sympify('c + a * exp(x/b)')

            def func(x, a, b, c):
                return c + a * np.exp(x/b)

        elif self.modelname == 'decay':  # Exponential decay to zero (no c parameter)
            self.expr = sympy.sympify('a * exp(-x/b)')

            def func(x, a, b):
                return a * np.exp(-x/b)

        elif self.modelname == 'decay2':  # Exponential decay, using rate lambda rather than time constant tau
            self.expr = sympy.sympify('a * exp(-x*b)')

            def func(x, a, b):
                return a * np.exp(-x*b)

        elif self.modelname == 'log':
            self.expr = sympy.sympify('a + b * log(x-c)')

            def func(x, a, b, c):
                return a + b * np.log(x-c)

        elif self.modelname == 'logistic':
            self.expr = sympy.sympify('a / (1 + exp((x-c)/b)) + d')

            def func(x, a, b, c, d):
                return d + a / (1 + np.exp((x-c)/b))

        elif self.modelname == 'quad' or (func == 'poly' and polyorder == 2):
            self.expr = sympy.sympify('a + b*x + c*x**2')

            def func(x, a, b, c):
                return a + b*x + c*x*x

        elif self.modelname == 'cubic' or (func == 'poly' and polyorder == 3):
            self.expr = sympy.sympify('a + b*x + c*x**2 + d*x**3')

            def func(x, a, b, c, d):
                return a + b*x + c*x*x + d*x*x*x

        elif self.modelname == 'poly':
            def func(x, *p):
                return np.poly1d(p[::-1])(x)  # Coeffs go in reverse order (...e, d, c, b, a)

            polyorder = int(polyorder)
            if polyorder < 1 or polyorder > 12:
                raise ValueError('Polynomial order out of range')
            varnames = [chr(ord('a')+i) for i in range(polyorder+1)]
            self.expr = sympy.sympify('+'.join(f'{v}*x**{i}' for i, v in enumerate(varnames)))

            # variable *args must have initial guess for scipy
            if self.p0 is None:
                self.p0 = np.ones(polyorder+1)
        else:
            # actual expression as string
            func, self.expr, _ = self.parse_math(self.modelname)

        self.func = func

        if self.modelname == 'poly' and polyorder > 3:
            # poly def above doesn't have named arguments, so the inspect won't find them. Name them here.
            self.pnames = varnames
        else:
            self.pnames = list(inspect.signature(self.func).parameters.keys())[1:]
        self.numparams = len(self.pnames)

        if self.modelname == 'callable':
            self.expr = sympy.sympify('f(x, ' + ', '.join(self.pnames) + ')')

        if self.bounds is None:
            bounds = (-np.inf, np.inf)
        else:
            bounds = self.bounds
            self.set_mcmc_priors(
                [lambda x, a=blow, b=bhi: (x > a) & (x <= b) for blow, bhi in zip(bounds[0], bounds[1])])

        if self.modelname == 'line' and not odr:
            # use generic LINE fit for lines with no odr
            self.fitcalc = (lambda x, y, ux, uy, absolute_sigma=self.absolute_sigma:
                            linefit(x, y, ux, uy, absolute_sigma=absolute_sigma))
        else:
            self.fitcalc = (lambda x, y, ux, uy, absolute_sigma=self.absolute_sigma:
                            fit(self.func, x, y, ux, uy, p0=self.p0, bounds=bounds,
                                odr=odr, absolute_sigma=absolute_sigma))

        return self.expr

    def parse_math(self, expr):
        ''' Check expr string for a valid curvefit function including an x variable
            and at least one fit parameter.

            Args:
                func (callable): Lambdified function of expr
                symexpr (sympy): Sympy expression of function
                argnames (list of strings): Names of arguments (except x) to function
        '''
        uparser.parse_math(expr)  # Will raise if not valid expression
        symexpr = sympy.sympify(expr)
        argnames = sorted(str(s) for s in symexpr.free_symbols)
        if 'x' not in argnames:
            raise ValueError('Expression must contain "x" variable.')
        argnames.remove('x')
        if len(argnames) == 0:
            raise ValueError('Expression must contain one or more parameters to fit.')
        # Make sure to specify 'numpy' so nans are returned instead of complex numbers
        func = sympy.lambdify(['x'] + argnames, symexpr, 'numpy')
        ParsedMath = namedtuple('ParsedMath', ['function', 'sympyexpr', 'argnames'])
        return ParsedMath(func, symexpr, argnames)

    def clear(self):
        ''' Clear the sampled points '''
        self.arr.clear()

    def run_uyestimate(self):
        ''' Generate estimate of y uncertainty using residuals '''
        if (self.arr.uy_estimate is None and (not self.arr.has_ux() and not self.arr.has_uy())):
            # Estimate uncertainty using residuals if uy not provided. LSQ method does this already,
            # do the same for GUM and MC.
            self.arr.uy_estimate = self.estimate_uy()

    def calculate_lsq(self):
        ''' Calculate analytical Least-Squares curve fit and uncertainty.

            Returns:
                CurveFitResults instance
        '''
        uy = np.zeros(len(self.arr.x)) if not self.arr.has_uy() else self.arr.uy
        coeff, cov = self.fitcalc(self.arr.x, self.arr.y, self.arr.ux, uy)

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
        out = FitResults(coeff, sigmas, cov, degf, resids)
        return CurveFitResults(out, self.fitsetup())

    def sample(self, samples=1000):
        ''' Generate Monte Carlo samples '''
        self.arr.clear()
        self.arr.sample(samples)

    def estimate_uy(self):
        ''' Calculate an estimate for uy using residuals of fit for when uy is not given.
            This is what linefit() method does behind the scenes, this function allows the
            same behavior for GUM and Monte Carlo.
        '''
        pcoeff, _ = self.fitcalc(self.arr.x, self.arr.y, ux=None, uy=None)
        uy = np.sqrt(np.sum((self.func(self.arr.x, *pcoeff) - self.arr.y)**2)/(len(self.arr.x) - len(pcoeff)))
        uy = np.full(len(self.arr.x), uy)
        return uy

    def monte_carlo(self, samples=1000):
        ''' Calculate Monte Carlo curve fit and uncertainty.

            Args:
                samples (int): Number of Monte Carlo samples

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
            self.samplecoeffs[i], _ = self.fitcalc(self.arr.xsamples[:, i], self.arr.ysamples[:, i], ux=None, uy=None)

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
        out = FitResults(coeff, sigma, cov, degf, resids, self.samplecoeffs)
        return CurveFitResults(out, self.fitsetup())

    def markov_chain_monte_carlo(self, samples=10000, burnin=0.2):
        ''' Calculate Markov-Chain Monte Carlo (Metropolis-in-Gibbs algorithm)
            fit parameters and uncertainty

            Args:
                samples (int): Total number of samples to generate
                burnin (float): Fraction of samples to reject at start of chain

            Returns:
                CurveFitResults instance

            Notes:
                Currently only supported with constant u(y) and u(x) = 0.
        '''
        self.run_uyestimate()
        uy = self.arr.uy if self.arr.uy_estimate is None else self.arr.uy_estimate

        if self.arr.has_ux():
            print('WARNING - MCMC algorithm ignores u(x) != 0')
        if np.max(uy) != np.min(uy):
            print('WARNING - MCMC algorithm with non-constant u(y). Using mean.')

        # Find initial guess/sigmas
        p, cov = self.fitcalc(self.arr.x, self.arr.y, self.arr.ux, uy)
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

        for pidx, pval in enumerate(p):
            if priors[pidx](pval) <= 0:
                # Will get div/0 below
                raise ValueError(f'Initial prior for parameter {self.pnames[pidx]} is < 0')

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
                if (sig2lim[1] > sig2new > sig2lim[0]):
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
        out = FitResults(coeff, sigma, cov, degf, resids, self.mcmccoeffs, accepts/samples)
        return CurveFitResults(out, self.fitsetup())

    def set_mcmc_priors(self, priors):
        ''' Set prior distribution functions for each input to be used in
            Markov-Chain Monte Carlo.

            Args:
                priors (list of callables): List of functions, one for each fitting
                    parameter. Each function must take a possible fit parameter as input
                    and return the probability of that parameter from 0-1.

            Notes:
                If set_mcmc_priors is not called, all priors will return 1.
        '''
        assert len(priors) == len(self.pnames)
        self.priors = priors

    def calculate_gum(self):
        ''' Calculate curve fit and uncertainty using GUM Approximation.

            Returns:
                CurveFitResults instance
        '''
        self.run_uyestimate()
        uy = self.arr.uy if self.arr.uy_estimate is None else self.arr.uy_estimate

        coeff, cov, _ = uncertarray._GUM(lambda x, y: self.fitcalc(x, y, ux=None, uy=None)[0],
                                         self.arr.x, self.arr.y, self.arr.ux, uy)
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
        out = FitResults(coeff, sigmas, cov, degf, resids)
        return CurveFitResults(out, self.fitsetup())

    def calculate(self):
        ''' Calculate Fit using Least Squares method. Same as calculate_lsq(). '''
        return self.calculate_lsq()

    def calculate_all(self, lsq=True, montecarlo=True, markov=True, gum=True):
        ''' Calculate all methods and return ReportCurveFitCombined '''
        outlsq = outgum = outmc = outmcmc = None
        if lsq:
            outlsq = self.calculate_lsq()
        if gum:
            outgum = self.calculate_gum()
        if montecarlo:
            outmc = self.monte_carlo()
        if markov:
            outmcmc = self.markov_chain_monte_carlo()
        return CurveFitResultsCombined(outlsq, outgum, outmc, outmcmc)
