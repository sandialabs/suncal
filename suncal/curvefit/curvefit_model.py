''' Curve Fit Model '''
from typing import Literal
from dataclasses import dataclass
from collections import namedtuple
import warnings
import inspect
import numpy as np
import sympy

from . import uncertarray
from .curvefit import fit, linefit
from .fitparse import fit_callable
from .results.curvefit import CurveFitResults, CurveFitResultsCombined, WaveformFeatureResult, WaveformFeatures
from . import waveform



FitResids = namedtuple('FitResiduals', ['residuals', 'Syx', 'r', 'F', 'SSres', 'SSreg'])
FitResults = namedtuple('FitResults', ['coeff', 'uncert', 'covariance',
                                       'degf', 'residuals', 'samples', 'acceptance'], defaults=(None,)*7)
FitSetup = namedtuple('FitSetup', ['points', 'expression', 'function', 'modelname', 'coeffnames', 'xname', 'yname'])


WaveCalcs = Literal['max', 'min', 'pkpk', 'rise', 'fall', 'fwhm', 'thresh']


@dataclass
class WaveCalc:
    ''' Define one waveform feature calculation '''
    calc: WaveCalcs
    clip: tuple[float, float] = None
    thresh: float = None
    tolerance: 'Limit' = None


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
                 bounds=None, odr=None, absolute_sigma=True, predictor_var='x'):
        self.arr = arr
        self.xname = 'x'
        self.yname = 'y'
        self.absolute_sigma = absolute_sigma
        self.tolerances = {}
        self.predictions: dict[str, tuple[float, 'Limit']] = {}
        self.wavecalcs: dict[str, WaveCalc] = {}
        self.set_fitfunc(func, polyorder=polyorder, p0=p0, bounds=bounds, odr=odr, predictor_var=predictor_var)

    def fitsetup(self):
        ''' Get setup parameters. See class for arguments '''
        return FitSetup(self.arr, self.expr, self.func, self.modelname, self.pnames, self.xname, self.yname)

    def _initial_guess(self):
        ''' Make a reasonable initial guess based on the model and data '''
        x, y = self.arr.x, self.arr.y
        if self.modelname == 'decay':
            b, a = np.polyfit(x, np.log(abs(y)), deg=1)   # Fit line to (x, log(y))
            p0 = [np.exp(a), -1/b]
        elif self.modelname == 'decay2':
            b, a = np.polyfit(x, np.log(abs(y)), deg=1)
            p0 = [np.exp(a), -b]
        elif self.modelname == 'exp':
            b, a = np.polyfit(x, np.log(abs(y)), deg=1)
            p0 = [np.exp(a), -1/b, 0]
        elif self.modelname == 'log':
            if all(np.sign(x)):
                b, a = np.polyfit(np.log(x), y, deg=1)
                p0 = [a, b, 0]
            else:
                b, a = np.polyfit(np.log(x-x.min()+1), y, deg=1)
                p0 = [a, b, x.min()]
        elif self.modelname == 'logisitic':
            p0 = [y.max()-y.min(), (x[-1]-x[0])/2, x.mean(), y.min()]
        else:
            p0 = np.ones(self.numparams)
        return p0

    def set_fitfunc(self, func, polyorder=2, bounds=None, odr=None, p0=None, predictor_var='x'):
        ''' Set up fit function '''
        self.modelname = func
        self.polyorder = polyorder
        self.odr = odr
        self.bounds = bounds
        self.predictor_var = predictor_var

        if callable(func):
            self.modelname = 'callable'
            self.func = func
            self.pnames = list(inspect.signature(self.func).parameters.keys())[1:]
            self.expr = sympy.sympify('f(x, ' + ', '.join(self.pnames) + ')')
        else:
            self.func, self.expr = fit_callable(self.modelname, self.polyorder, self.predictor_var)

            if self.modelname == 'poly' and self.polyorder > 3:
                # poly def above doesn't have named arguments, so the inspect won't find them. Name them here.
                self.pnames = [chr(ord('a')+i) for i in range(self.polyorder+1)]
            else:
                self.pnames = list(inspect.signature(self.func).parameters.keys())[1:]

        self.numparams = len(self.pnames)

        if self.bounds is None:
            bounds = (-np.inf, np.inf)
        else:
            bounds = self.bounds
            self.set_mcmc_priors(
                [lambda x, a=blow, b=bhi: (x > a) & (x <= b) for blow, bhi in zip(bounds[0], bounds[1])])

        if p0 is None:
            self.p0 = self._initial_guess()
        else:
            self.p0 = p0

        if self.modelname == 'line' and not odr:
            # use generic LINE fit for lines with no odr
            self.fitcalc = (lambda x, y, ux, uy, absolute_sigma=self.absolute_sigma:
                            linefit(x, y, ux, uy, absolute_sigma=absolute_sigma))
        else:
            self.fitcalc = (lambda x, y, ux, uy, absolute_sigma=self.absolute_sigma:
                            fit(self.func, x, y, ux, uy, p0=self.p0, bounds=bounds,
                                odr=odr, absolute_sigma=absolute_sigma))

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

        resids = self.arr.y - self.func(self.arr.x, *coeff)  # All residuals (NOT squared)
        sigmas = np.sqrt(np.diag(cov))
        degf = len(self.arr.x) - len(coeff)
        if self.absolute_sigma or not self.arr.has_uy():
            w = np.full(len(self.arr.x), 1)  # Unweighted residuals in Syx
        else:
            w = 1/uy**2  # Determine weighted Syx
            w = w/sum(w) * len(self.arr.y)      # Normalize weights so sum(wi) = N
        SSres = sum(w*resids**2)   # Sum-of-squares of residuals
        Syx = np.sqrt(SSres/degf)  # Standard error of the estimate (based on residuals)
        SSreg = sum(w*(self.func(self.arr.x, *coeff) - sum(w*self.arr.y)/sum(w))**2)
        r = np.sqrt(1-SSres/(SSres+SSreg))
        resids = FitResids(resids, Syx, r, SSreg*degf/SSres, SSres, SSreg)
        out = FitResults(coeff, sigmas, cov, degf, resids)
        return CurveFitResults(out, self.fitsetup(), self.tolerances, self.predictions)

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

        resids = self.arr.y - self.func(self.arr.x, *coeff)
        degf = len(self.arr.x) - len(coeff)
        if self.absolute_sigma or not self.arr.has_uy():
            w = np.full(len(self.arr.x), 1)  # Unweighted residuals in Syx
        else:
            w = 1/uy**2  # Determine weighted Syx
            w = w/sum(w) * len(self.arr.y)   # Normalize weights so sum(wi) = N
        cov = np.cov(self.samplecoeffs.T)
        SSres = sum(w*resids**2)   # Sum-of-squares of residuals
        Syx = np.sqrt(SSres/degf)  # Standard error of the estimate (based on residuals)
        SSreg = sum(w*(self.func(self.arr.x, *coeff) - sum(w*self.arr.y)/sum(w))**2)
        r = np.sqrt(1-SSres/(SSres+SSreg))

        resids = FitResids(resids, Syx, r, SSreg*degf/SSres, SSres, SSreg)
        out = FitResults(coeff, sigma, cov, degf, resids, self.samplecoeffs)
        return CurveFitResults(out, self.fitsetup(), self.tolerances, self.predictions)

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
            warnings.warn('MCMC algorithm ignores u(x) != 0')
        if np.max(uy) != np.min(uy):
            warnings.warn('MCMC algorithm with non-constant u(y). Using mean.')

        # Find initial guess/sigmas
        p, cov = self.fitcalc(self.arr.x, self.arr.y, self.arr.ux, uy)
        up = np.sqrt(np.diag(cov))
        if not all(np.isfinite(up)):
            raise ValueError('MCMC Could not determine initial sigmas. Try providing p0.')

        if all(uy == 0):
            # Sigma2 is unknown. Estimate from residuals and vary through trace.
            resids = self.arr.y - self.func(self.arr.x, *p)
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
                if sig2lim[1] > sig2new > sig2lim[0]:
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
            w = 1/uy**2  # Determine weighted Syx
            w = w/sum(w) * len(self.arr.y)   # Normalize weights so sum(wi) = N
        cov = np.cov(self.mcmccoeffs.T)
        SSres = sum(w*resids**2)   # Sum-of-squares of residuals
        Syx = np.sqrt(SSres/degf)  # Standard error of the estimate (based on residuals)
        SSreg = sum(w*(self.func(self.arr.x, *coeff) - sum(w*self.arr.y)/sum(w))**2)
        r = np.sqrt(1-SSres/(SSres+SSreg))
        resids = FitResids(resids, Syx, r, SSreg*degf/SSres, SSres, SSreg)
        out = FitResults(coeff, sigma, cov, degf, resids, self.mcmccoeffs, accepts/samples)
        return CurveFitResults(out, self.fitsetup(), self.tolerances, self.predictions)

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
        resids = self.arr.y - self.func(self.arr.x, *coeff)
        degf = len(self.arr.x) - len(coeff)
        if self.absolute_sigma or not self.arr.has_uy():
            w = 1  # Unweighted residuals in Syx
        else:
            w = 1/uy**2  # Determine weighted Syx
            w = w/sum(w) * len(self.arr.y)   # Normalize weights so sum(wi) = N
        SSres = sum(w*resids**2)
        Syx = np.sqrt(SSres/degf)  # Standard error of the estimate (based on residuals)
        SSreg = sum(w*(self.func(self.arr.x, *coeff) - self.arr.y.mean())**2)
        r = np.sqrt(1-SSres/(SSres+SSreg))

        resids = FitResids(resids, Syx, r, SSreg*degf/SSres, SSres, SSreg)
        out = FitResults(coeff, sigmas, cov, degf, resids)
        return CurveFitResults(out, self.fitsetup(), self.tolerances, self.predictions)

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
        outwave = self.calculate_wave()
        return CurveFitResultsCombined(outlsq, outgum, outmc, outmcmc, outwave)

    def calculate_wave(self) -> WaveformFeatures:
        ''' Calculate waveform features '''
        results = {}
        for name, wcalc in self.wavecalcs.items():
            wave = self.arr
            if wcalc.clip:
                wave = wave.slice(*wcalc.clip)

            func = {
                'max': lambda w: waveform.extrema.maximum(w),
                'min': lambda w: waveform.extrema.minimum(w),
                'rise': waveform.pulse.u_rise_time,
                'fall': waveform.pulse.u_fall_time,
                'thresh_rise': lambda w, thresh=wcalc.thresh: waveform.threshold.threshold_crossing_uncertainty(
                                            w, thresh, direction='rising'),
                'thresh_fall': lambda w, thresh=wcalc.thresh: waveform.threshold.threshold_crossing_uncertainty(
                                            w, thresh, direction='falling'),
                'pkpk': waveform.extrema.peak_peak,
                'fwhm': waveform.pulse.u_pulse_width,
             }.get(wcalc.calc)

            assert func is not None

            result = func(wave)
            poc = None
            if wcalc.tolerance:
                poc = wcalc.tolerance.probability_conformance_95(result.low, result.high)
            results[name] = WaveformFeatureResult(
                wcalc.calc,
                result,
                wcalc.clip,
                wcalc.tolerance,
                wcalc.thresh,
                poc
            )

        return WaveformFeatures(results, self.arr)
