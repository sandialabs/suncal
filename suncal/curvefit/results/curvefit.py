''' Results of curve fit calculation '''

from dataclasses import dataclass
import numpy as np
import sympy
from scipy import stats
from scipy import interpolate, optimize

from ...common import reporter
from ...common.ttable import k_factor
from ..report.curvefit import ReportCurveFit, ReportCurveFitCombined


@reporter.reporter(ReportCurveFit)
class CurveFitResults:
    ''' Results of curve fit

        Args:
            fitresults: CurveFitResults instance
            fitsetup: tuple data from CurveFit.fitsetup()
    '''
    def __init__(self, fitresults, fitsetup):
        self.coeffs = fitresults.coeff
        self.uncerts = fitresults.uncert
        self.covariance = fitresults.covariance
        self.degf = fitresults.degf
        self.residuals = fitresults.residuals
        self.samples = fitresults.samples
        self.acceptance = fitresults.acceptance
        self.setup = fitsetup
        self.correlation = self.covariance / self.uncerts[:, None] / self.uncerts[None, :]

    def y(self, x):
        ''' Predict Y value at X '''
        return self.setup.function(x, *self.coeffs)

    def confidence_band(self, x, k=1, conf=None):
        ''' Get confidence band uncertainty at x values

            Args:
                x (float or array): value(s) at which to compute
                k (float): Coverage factor
                conf (float): Level of confidence, overrides k
        '''
        if conf is not None:
            k = k_factor(conf, self.degf)

        dp = self.uncerts / 1E6
        band = []
        for xval in np.atleast_1d(x):
            grad = optimize.approx_fprime(self.coeffs, lambda p, xx=xval: self.setup.function(xx, *p), epsilon=dp)
            band.append(grad.T @ np.atleast_2d(self.covariance) @ grad)
        band = k * np.sqrt(np.array(band))
        return band[0] if np.isscalar(x) else band

    def prediction_band(self, x, k=1, conf=None, mode='Syx'):
        ''' Get prediction band uncertainty at x values

            Args:
                x (float or array): value(s) at which to compute
                k (float): Coverage factor
                conf (float): Level of confidence, overrides k
                mode (str): How to apply uncertainty in new measurement. 'Syx'
                    will use Syx calculated from residuals. 'sigy' uses user-provided
                    y-uncertainty, extrapolating between values as necessary. 'sigylast'
                    uses last sigy value (useful when x is time and fit is being predicted
                    into the future)
        '''
        if mode not in ['Syx', 'sigy', 'sigylast']:
            raise ValueError('Prediction band mode must be Syx, sigy, or sigylast')

        sigy = self.setup.points.uy

        if mode == 'Syx' or (np.isscalar(sigy) and sigy == 0):
            uy = self.residuals.Syx
        elif np.isscalar(sigy):
            uy = sigy
        elif mode == 'sigy':
            if sigy.min() == sigy.max():  # All elements equal
                uy = sigy[0]
            else:
                arrx = self.setup.points.x
                if not np.all(np.diff(arrx) > 0):  # np.interp requires sorted data
                    idx = np.argsort(arrx)
                    arrx = arrx[idx]
                    sigy = sigy[idx]
                uy = interpolate.interp1d(arrx, sigy, fill_value='extrapolate')(x)
        elif mode == 'sigylast':
            uy = sigy[-1]

        if conf is not None:
            k = k_factor(conf, self.degf)
        return k * np.sqrt(self.confidence_band(x, k=1)**2 + uy**2)

    def confidence_band_distribution(self, x):
        ''' Get distribution of confidence band at x value

            Args:
                x (float): value at which to compute
        '''
        std = self.confidence_band(x)
        y = self.setup.points.y(x)
        degf = self.degf
        degf = min(2.0001, degf)    # Standard dev not defined for df < 2
        if degf > 1000:
            return stats.norm(loc=y, scale=std)
        return stats.t(loc=y, scale=std/np.sqrt(degf/(degf-2)), df=degf)

    def prediction_band_distribution(self, x, mode='Syx'):
        ''' Get distribution of prediction band at x value

            Args:
                x (float): value at which to compute
                mode (str): How to apply uncertainty in new measurement. 'Syx'
                    will use Syx calculated from residuals. 'sigy' uses user-provided
                    y-uncertainty, extrapolating between values as necessary. 'sigylast'
                    uses last sigy value (useful when x is time and fit is being predicted
                    into the future)
        '''
        std = self.prediction_band(x, mode=mode)
        y = self.y(x)
        degf = self.degf
        degf = max(2.0001, degf)    # Standard dev not defined for df < 2
        if degf > 1000:
            return stats.norm(loc=y, scale=std)
        return stats.t(loc=y, scale=std/np.sqrt(degf/(degf-2)), df=degf)

    def fit_expr(self, subs=True, n=4, full=True):
        ''' Get Sympy expression of fit function with coefficients

            Args:
                subs (bool): Substitute calculated fit parameter values in result
                n (int): Number of decimals to include
                full (bool): Include the "y ="
        '''
        expr = self.setup.expression
        if subs:
            expr = expr.subs(dict(zip(self.setup.coeffnames, self.coeffs))).evalf(n=n)
        if full:
            expr = sympy.Eq(sympy.Symbol('y'), expr)
        return expr

    def confidence_expr(self, subs=True, n=4, full=True):
        ''' Return sympy expression for confidence interval as function of x

            Args:
                subs (bool): Substitute calculated fit parameter values in result
                n (int): Number of decimals to include
                full (bool): Show full expression, ie "U_conf = ...". When full==false,
                    just the right hand side is returned

            Notes:
                This expression will only match u_conf exactly when using residuals
                Syx to determine y-uncertainty.
        '''
        if self.setup.modelname == 'line':
            expr = sympy.sympify('S_yx * sqrt(1 / n + (x-xbar)**2 * (sigma_b/S_yx)^2)')
            if subs:
                expr = expr.subs({'S_yx': self.residuals.Syx,
                                  'n': len(self.setup.points),
                                  'xbar': self.setup.points.x.mean(),
                                  'sigma_b': self.uncerts[0]}).evalf(n=n)

            if full:
                expr = sympy.Eq(sympy.Symbol('u_{conf}'), expr)
        else:
            raise NotImplementedError('uconf expression only implemented for line fits.')
        return expr

    def prediction_expr(self, subs=True, n=4, full=True, mode='Syx'):
        ''' Return sympy expression for prediction interval as function of x

            Args:
                subs (bool): Substitute calculated fit parameter values in result
                n (int): Number of decimals to include
                full (bool): Show full expression, ie "U_conf = ...". When full==false,
                    just the right hand side is returned
                mode (str): How to apply uncertainty in new measurement. 'Syx'
                    will use Syx calculated from residuals. 'sigy' uses user-provided
                    y-uncertainty, extrapolating between values as necessary. 'sigylast'
                    uses last sigy value (useful when x is time and fit is being predicted
                    into the future)

            Notes:
                This expression will only match u_conf exactly when using residuals
                Syx to determine y-uncertainty.
        '''
        if self.setup.modelname == 'line':
            subsdict = {'n': len(self.setup.points),
                        'xbar': self.setup.points.x.mean(),
                        'sigma_b': self.uncerts[0],
                        'S_yx': self.residuals.Syx}
            if mode == 'Syx':
                expr = sympy.sympify('S_yx * sqrt(1 + 1 / n + (x-xbar)**2 * (sigma_b/S_yx)^2)')
            else:
                expr = sympy.sympify('sqrt(u_y^2 + S_yx^2 * (1 / n + (x-xbar)**2 * (sigma_b/S_yx)^2))')
                sigy = self.setup.points.uy
                if mode == 'sigy' and sigy.min() == sigy.max():
                    # Interpolate, but all elements are equal
                    subsdict.update({'u_y': sigy[0]})
                elif mode == 'sigylast':
                    subsdict.update({'u_y': sigy[-1]})

            if subs:
                expr = expr.subs(subsdict).evalf(n=n)
            if full:
                expr = sympy.Eq(sympy.Symbol('u_{pred}'), expr)
        else:
            raise NotImplementedError('upred expression only implemented for line fits.')
        return expr

    def interval_expression(self):
        ''' Return sympy expressions for uncertainty components
            along a calibration curve (see GUM F). The equations
            compute a value and uncertianty valid over the entire interval.

            Returned values are variances, RSS'd together give the interval uncertainty
            with reportd value of y+bbar
        '''
        x, t1, t2 = sympy.symbols('x t1 t2')
        oneovert1t2 = 1 / (t2-t1)

        # Variables with _int are intermediate integrals, without have been evaluated
        # with _expr are the GUM expressions
        b = self.setup.expression
        bbar_expr = oneovert1t2 * sympy.Integral(sympy.Symbol('b'), (x, t1, t2))
        bbar_int = oneovert1t2 * sympy.Integral(b, (x, t1, t2))
        bbar = bbar_int.doit()
        ubbar_expr = oneovert1t2 * sympy.Integral((sympy.Symbol('b') - sympy.Symbol('bbar'))**2, (x, t1, t2))
        ubbar_int = oneovert1t2 * sympy.Integral((b-bbar)**2, (x, t1, t2))
        ubbar = ubbar_int.doit()
        uconf = sympy.Symbol('u_conf')
        ub_expr = oneovert1t2 * sympy.Integral(uconf**2, (x, t1, t2))
        if self.setup.modelname == 'line':
            ub_int = oneovert1t2 * sympy.Integral(self.confidence_expr(subs=False, full=False)**2, (x, t1, t2))
            ub = ub_int.doit()
        else:
            # Can't integrate nonlinear uconf
            ub = None
            ub_int = None

        return {'bbar_expr': bbar_expr,
                'bbar': bbar,
                'ubbar_expr': ubbar_expr,
                'ubbar': ubbar,
                'ub_expr': ub_expr,
                'ub': ub}

    def interval_uncertainty(self, t1, t2):
        ''' Calculate a single value and standard uncertainty that applies to the entire
            interval from t1 to t2. See GUM F.2.4.5.
        '''
        uy = self.residuals.Syx
        # sigy can be scalar or array function of x. Interpolate/average (linearly) over interval if necessary
        if not np.isscalar(uy):
            uy1, uy2 = interpolate.interp1d(self.setup.points.x, uy, fill_value='extrapolate')([t1, t2])
            uy = np.sqrt((uy1**2 + uy2**2) / 2)  # Average over interval

        subs = {'t1': t1,
                't2': t2,
                'S_yx': uy,
                'n': len(self.setup.points.x),
                'xbar': self.setup.points.x.mean(),
                'sigma_b': self.uncerts[0]}
        subs.update(dict(zip(self.setup.coeffnames, self.coeffs)))

        components = self.interval_expression()
        value = components['bbar'].subs(subs).evalf()

        ubbar = components['ubbar'].subs(subs).evalf()
        ub = components['ub']
        if ub is None:
            # Need to numerically integrate uconf
            xx = np.linspace(t1, t2, num=200)
            ub = 1/(t2-t1) * np.trapz(self.confidence_band(xx)**2, xx)
        else:
            ub = ub.subs(subs).evalf()
        uncert = np.sqrt(float(ubbar + ub + uy**2))
        return float(value), float(uncert)

    def test_t(self, paramname, nominal=0, conf=.95, verbose=False):
        ''' T-test of fit parameter is statistically different from nominal.
            Use defaults to test slope of linear fit.

            Args:
                pidx (int): Index of fit parameter to test
                nominal (float): Test that mean[pidx] is statistically different from nominal.
                    For example, nominal=0 to test that a slope parameter is different
                    than 0.
                conf (float): Level of confidence (0-1) of range
                verbose (boolean): Print values

            Returns:
                p (boolean): Test result
        '''
        paramidx = self.setup.coeffnames.index(paramname)
        t = abs(self.coeffs[paramidx] - nominal) / self.uncerts[paramidx]
        ta = stats.t.ppf(1-(1-conf)/2, df=self.degf)
        if verbose:
            print(f'{paramname}: {t} > {ta} --> {t>ta}')
        return t > ta

    def test_t_range(self, paramname, nominal=0, conf=.95, verbose=False):
        ''' Find confidence range of parameter values, test that it does not contain nominal
            value. Use defaults to test linear fit slope is different from 0 with 95% confidence.

            Args:
                pidx (int): Index of fit parameter to test
                nominal (float): Test that mean[pidx] is statistically different from nominal.
                    For example, nominal=0 to test that a slope parameter is different
                    than 0.
                conf (float): Level of confidence (0-1) of range
                verbose (boolean): Print values

            Returns:
                p (boolean): Test result
        '''
        paramidx = self.setup.coeffnames.index(paramname)
        b1 = self.coeffs[paramidx]
        sb = self.uncerts[paramidx]
        ta = stats.t.ppf(1-(1-conf)/2, df=self.degf)
        ok = not (b1 - ta*sb < nominal < b1 + ta*sb)
        if verbose:
            print(f'[{b1 - ta*sb} < {paramname} < {b1 + ta*sb}] \u2285 {nominal} --> {ok}')
        return ok


@reporter.reporter(ReportCurveFitCombined)
@dataclass
class CurveFitResultsCombined:
    ''' Results of multiple curve fit calculation methods

        Attributes:
            lsq: Results of Least Squares curve fit
            gum: Results of GUM method curve fit
            montecarlo: Results of Monte Carlo curve fit
            markov: Results of Markov Chain Monte Carlo method
    '''
    lsq: CurveFitResults
    gum: CurveFitResults
    montecarlo: CurveFitResults
    markov: CurveFitResults

    def method(self, name):
        ''' Get results from one method
        
            Args:
                method (str): Name of method. Can be lsq, gum, montecarlo, or markov
        '''
        return getattr(self, name)

    @property
    def setup(self):
        ''' Calculation setup from one of the Results '''
        if self.lsq is not None:
            return self.lsq.setup
        if self.gum is not None:
            return self.gum.setup
        if self.montecarlo is not None:
            return self.montecarlo.setup
        if self.markov is not None:
            return self.markov.setup