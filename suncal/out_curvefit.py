''' Output reports for curve fitting calculations '''

import sympy
from scipy import stats
from scipy import interpolate, optimize
import numpy as np
from contextlib import suppress
from dateutil.parser import parse
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from . import report
from . import output
from . import out_uncert
from . import plotting
from .ttable import t_factor


class CurveFitOutputLSQ(output.Output):
    ''' Results from least squares (and maybe GUM?) method calculation '''
    def __init__(self, model, inputs):
        self.model = model
        self.inputs = inputs  # Array object
        self.name = self.model.name
        self.desc = self.model.desc
        self.paramnames = self.model.pnames
        self.fitfunc = self.model.func
        self.predmode = 'Syx'
        self._calccoeffs()

    def _calccoeffs(self):
        ''' Run calculation here (so it can be subclassed) '''
        result = self.model.calc_LSQ()
        self.coeffs = result.coeff
        self.uncerts = result.uncert
        self.cov = result.covariance
        self.degf = result.degf
        self.residuals = result.residuals
        self.cor = self.cov / self.uncerts[:, None] / self.uncerts[None, :]  # See numpy code for corrcoeff

    def _index(self, idx):
        return self.paramnames.index(idx) if isinstance(idx, str) else idx

    def y(self, x):
        ''' Predict y value for given x value '''
        return self.fitfunc(x, *self.coeffs)

    def u_conf(self, x, k=1, conf=None):
        ''' Confidence band uncertainty.

            Parameters
            ---------
            x: float or array
                X-value(s) at which to calculate
            k: float
                k-value to apply
            conf: float
                Level of confidence (0 to 1). Overrides value of k parameter
        '''
        if conf is not None:
            k = t_factor(conf, self.degf)

        dp = self.uncerts / 1E6
        band = []
        for xval in np.atleast_1d(x):
            grad = optimize.approx_fprime(self.coeffs, lambda p: self.fitfunc(xval, *p), epsilon=dp)
            band.append(grad.T @ np.atleast_2d(self.cov) @ grad)
        band = k * np.sqrt(np.array(band))
        return band[0] if np.isscalar(x) else band

    def u_pred(self, x, k=1, conf=None, mode=None, **kwargs):
        ''' Calculate prediction band for fit curve for arbitrary nonlinear regression.

            Parameters
            ----------
            x: float or array
                x-value at which to determine confidence band
            k: float
                k-value for expanding prediction band
            conf: float
                Level of confidence (0 to 1). Overrides value of k parameter.
            mode: string
                How to apply uncertainty in new measurement. 'Syx' will use Syx calculated from
                residuals. 'sigy' uses user-provided y-uncertainty, extrapolating between
                values as necessary. 'sigylast' uses last sigy value (useful when x is time
                and fit is being predicted into the future)

            Returns
            -------
            upred: array
                Prediction band at the points in x array. Interval will be
                y +/- k * uconf.

            Reference
            ---------
            Christopher Cox and Guangqin Ma. Asymptotic Confidence Bands for Generalized
            Nonlinear Regression Models. Biometrics Vol. 51, No. 1 (March 1995) pp 142-150.
        '''
        mode = self.predmode if mode is None else mode
        if mode not in ['Syx', 'sigy', 'sigylast']:
            raise ValueError('Prediction band mode must be Syx, sigy, or sigylast')

        sigy = self.inputs.uy

        if mode == 'Syx' or (np.isscalar(sigy) and sigy == 0):
            uy = self.residuals.Syx
        elif np.isscalar(sigy):
            uy = sigy
        elif mode == 'sigy':
            if sigy.min() == sigy.max():  # All elements equal
                uy = sigy[0]
            else:
                arrx = self.inputs.x
                if not np.all(np.diff(arrx) > 0):  # np.interp requires sorted data
                    idx = np.argsort(arrx)
                    arrx = arrx[idx]
                    sigy = sigy[idx]
                uy = interpolate.interp1d(arrx, sigy, fill_value='extrapolate')(x)
        elif mode == 'sigylast':
            uy = sigy[-1]

        if conf is not None:
            k = t_factor(conf, self.degf)
        return k * np.sqrt(self.u_conf(x, k=1)**2 + uy**2)

    def u_pred_dist(self, x, mode=None):
        ''' Get prediction band distribution at x.

            Parameters
            ----------
            x: float
                X-value at which to calculate distribution

            Returns
            -------
            dist: scipy.rv_frozen
                Frozen distribution of prediction band value
        '''
        std = self.u_pred(x, mode=mode)
        y = self.y(x)
        if self.degf > 1000:
            return stats.norm(loc=y, scale=std)
        else:
            if self.degf < 2:
                self.degf = 2.001  # Standard dev not defined for df < 2
            return stats.t(loc=y, scale=std/np.sqrt(self.degf/(self.degf-2)), df=self.degf)

    def u_conf_dist(self, x):
        ''' Get confidence band distribution at x.

            Parameters
            ----------
            x: float
                X-value at which to calculate distribution

            Returns
            -------
            dist: scipy.rv_frozen
                Frozen distribution of confidence band value
        '''
        std = self.u_conf(x)
        y = self.y(x)
        if self.degf > 1000:
            return stats.norm(loc=y, scale=std)
        else:
            if self.degf < 2:
                self.degf = 2.001  # Standard dev not defined for df < 2
            return stats.t(loc=y, scale=std/np.sqrt(self.degf/(self.degf-2)), df=self.degf)

    def expr(self, subs=True, n=4, full=True):
        ''' Return sympy expression for fit function

            Parameters
            ----------
            subs: bool
                Substitute calculated fit parameter values in result
            n: int
                Number of decimals to include
            full: bool
                Include the "y ="
        '''
        expr = self.model.expr
        if subs:
            expr = expr.subs(dict(zip(self.paramnames, self.coeffs))).evalf(n=n)
        if full:
            expr = sympy.Eq(sympy.Symbol('y'), expr)
        return expr

    def expr_uconf(self, subs=True, n=4, full=True):
        ''' Return sympy expression for confidence interval

            Parameters
            ----------
            subs: bool
                Substitute calculated fit parameter values in result
            n: int
                Number of decimals to include
            full: bool
                Show full expression, ie "U_conf = ...". When full==false,
                just the right hand side is returned

            Notes
            -----
            This expression will only match u_conf exactly when using residuals
            Syx to determine y-uncertainty.
        '''
        if self.model.fitname == 'line':
            expr = sympy.sympify('S_yx * sqrt(1 / n + (x-xbar)**2 * (sigma_b/S_yx)^2)')
            if subs:
                expr = expr.subs({'S_yx': self.residuals.Syx,
                                  'n': len(self.inputs),
                                  'xbar': self.inputs.x.mean(),
                                  'sigma_b': self.uncerts[0]}).evalf(n=n)

            if full:
                expr = sympy.Eq(sympy.Symbol('u_{conf}'), expr)
        else:
            raise NotImplementedError('uconf expression only implemented for line fits.')
        return expr

    def expr_upred(self, subs=True, n=4, full=True, mode=None):
        ''' Return sympy expression for prediction interval

            Parameters
            ----------
            subs: bool
                Substitute calculated fit parameter values in result
            n: int
                Number of decimals to include
            full: bool
                Show full expression, ie "U_conf = ...". When full==false,
                just the right hand side is returned
            mode: string
                How to apply uncertainty in new measurement. 'Syx' will use Syx calculated from
                residuals. 'sigy' uses user-provided y-uncertainty, extrapolating between
                values as necessary. 'sigylast' uses last sigy value (useful when x is time
                and fit is being predicted into the future)

            Notes
            -----
            This expression will only match u_conf exactly when using residuals
            Syx to determine y-uncertainty.
        '''
        if self.model.fitname == 'line':
            mode = self.predmode if mode is None else mode
            subsdict = {'n': len(self.inputs),
                        'xbar': self.inputs.x.mean(),
                        'sigma_b': self.uncerts[0],
                        'S_yx': self.residuals.Syx}
            if mode == 'Syx':
                expr = sympy.sympify('S_yx * sqrt(1 + 1 / n + (x-xbar)**2 * (sigma_b/S_yx)^2)')
            else:
                expr = sympy.sympify('sqrt(u_y^2 + S_yx^2 * (1 / n + (x-xbar)**2 * (sigma_b/S_yx)^2))')
                sigy = self.inputs.uy
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

    def expr_interval(self):
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
        b = self.model.expr
        bbar_expr = oneovert1t2 * sympy.Integral(sympy.Symbol('b'), (x, t1, t2))
        bbar_int = oneovert1t2 * sympy.Integral(b, (x, t1, t2))
        bbar = bbar_int.doit()
        ubbar_expr = oneovert1t2 * sympy.Integral((sympy.Symbol('b') - sympy.Symbol('bbar'))**2, (x, t1, t2))
        ubbar_int = oneovert1t2 * sympy.Integral((b-bbar)**2, (x, t1, t2))
        ubbar = ubbar_int.doit()
        uconf = sympy.Symbol('u_conf')
        ub_expr = oneovert1t2 * sympy.Integral(uconf**2, (x, t1, t2))
        if self.model.fitname == 'line':
            ub_int = oneovert1t2 * sympy.Integral(self.expr_uconf(subs=False, full=False)**2, (x, t1, t2))
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

    def uncert_interval(self, t1, t2):
        ''' Calculate a single value and standard uncertainty that applies to the entire
            interval from t1 to t2. See GUM F.2.4.5.
        '''
        uy = self.residuals.Syx
        # sigy can be scalar or array function of x. Interpolate/average (linearly) over interval if necessary
        if not np.isscalar(uy):
            uy1, uy2 = interpolate.interp1d(self._inputx(), uy, fill_value='extrapolate')([t1, t2])
            uy = np.sqrt((uy1**2 + uy2**2) / 2)  # Average over interval

        subs = {'t1': t1,
                't2': t2,
                'S_yx': uy,
                'n': len(self._inputx()),
                'xbar': self._inputx().mean(),
                'sigma_b': self.uncerts[0]}
        subs.update(dict(zip(self.paramnames, self.coeffs)))

        components = self.expr_interval()
        value = components['bbar'].subs(subs).evalf()

        ubbar = components['ubbar'].subs(subs).evalf()
        ub = components['ub']
        if ub is None:
            # Need to numerically integrate uconf
            xx = np.linspace(t1, t2, num=200)
            ub = 1/(t2-t1) * np.trapz(self.u_conf(xx)**2, xx)
        else:
            ub = ub.subs(subs).evalf()
        uncert = np.sqrt(float(ubbar + ub + uy**2))
        return float(value), float(uncert)

    def report(self, **kwargs):
        ''' Generate/return default report

            Keyword Arguments
            -----------------
            See report.Report

            Returns
            -------
            report.Report
        '''
        hdr = ['Parameter', 'Nominal', 'Standard Uncertainty']
        rows = []
        for name, val, unc in zip(self.paramnames, self.coeffs, self.uncerts):
            rows.append([name,
                         report.Number(val, matchto=unc),
                         report.Number(unc)])
        r = report.Report(**kwargs)
        r.table(rows, hdr=hdr)
        return r

    def report_fit(self, **kwargs):
        ''' Report goodness-of-fit values r-squared and Syx. Note r is not a good predictor
            of fit for nonlinear models, but we report it anyway.
        '''
        rows = []
        rows.append([report.Number(self.residuals.r, fmt='decimal'),
                     report.Number(self.residuals.r**2, fmt='decimal'),
                     report.Number(self.residuals.Syx),
                     report.Number(self.residuals.F, fmt='auto')])
        r = report.Report(**kwargs)
        r.table(rows, ['r', 'r-squared', 'Standard Error (Syx)', 'F-value'])
        return r

    def report_correlation(self, **kwargs):
        ''' Report table of correlation coefficients '''
        hdr = ['Parameter'] + self.paramnames
        rows = []
        for idx, row in enumerate(self.cor):
            rows.append([self.paramnames[idx]] + [report.Number(v) for v in row])
        r = report.Report(**kwargs)
        r.table(rows, hdr=hdr)
        return r

    def report_residuals(self, **kwargs):
        ''' Report a plot of residuals, histogram, and normal-probability '''
        with plt.style.context(plotting.plotstyle):
            fig = plt.figure()
            ax = fig.add_subplot(2, 2, 1)
            self.plot_points(ax=ax, ls='', marker='o')
            self.plot_fit(ax=ax)
            ax.set_title('Fit Line')
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax2 = fig.add_subplot(2, 2, 2)
            self.plot_residuals(ax=ax2, hist=True)
            ax2.set_title('Residual Histogram')
            ax2.set_xlabel(r'$\Delta$ y')
            ax2.set_ylabel('Probability')
            ax3 = fig.add_subplot(2, 2, 3)
            self.plot_residuals(ax=ax3, hist=False)
            ax3.axhline(0, color='C1')
            ax3.set_title('Raw Residuals')
            ax3.set_xlabel('x')
            ax3.set_ylabel(r'$\Delta$ y')
            ax4 = fig.add_subplot(2, 2, 4)
            self.plot_normprob(ax=ax4)
            ax4.set_title('Normal Probability')
            ax4.set_xlabel('Theoretical quantiles')
            ax4.set_ylabel('Ordered sample values')
            fig.tight_layout()
        r = report.Report(**kwargs)
        r.plot(fig)
        plt.close(fig)
        r.append(self.report_residtable(**kwargs))
        return r

    def report_residtable(self, k=2, conf=None, **kwargs):
        ''' Report table of measured values and their residuals '''
        r = report.Report(**kwargs)
        kstr = '(k={})'.format(k) if conf is None else '({:.4g}%)'.format(conf*100)
        hdr = ['Measured x', 'Measured y', 'Predicted y', 'Residual', 'Confidence Band {}'.format(kstr), 'Prediction Band {}'.format(kstr)]
        x, y = self.inputs.x, self.inputs.y
        if self.inputs.xdate:
            xstring = mdates.num2date(x)
            xstring = [k.strftime('%d-%b-%Y') for k in xstring]
        else:
            xstring = [str(k) for k in x]
        resid = self.residuals.residuals
        rows = []
        for i in range(len(x)):
            confband = self.u_conf(x[i], k=k, conf=conf)
            rows.append(['{}'.format(xstring[i]),
                         report.Number(y[i], matchto=confband),
                         report.Number(self.y(x[i]), matchto=confband),
                         report.Number(resid[i], matchto=confband),
                         report.Number(confband),
                         report.Number(self.u_pred(x[i], k=k, conf=conf))])
        r.table(rows, hdr=hdr)
        return r

    def report_interval_uncert(self, t1, t2, k=2, conf=None, plot=True, **kwargs):
        ''' Report the value and uncertainty that applies to the
            entire interval from t1 to t2.
        '''
        confstr = ''
        if conf is not None:
            k = t_factor(conf, self.degf)
            confstr = ', {:.4g}%'.format(conf*100)

        if self.inputs.xdate and isinstance(t1, str):
            t1 = mdates.date2num(parse(t1))
            t2 = mdates.date2num(parse(t2))

        if self.inputs.xdate:
            t1str = mdates.num2date(t1).strftime('%d-%b-%Y')
            t2str = mdates.num2date(t2).strftime('%d-%b-%Y')
        else:
            t1str, t2str = report.Number.number_array([t1, t2], thresh=8)

        value, uncert = self.uncert_interval(t1, t2)
        r = report.Report(**kwargs)
        r.txt('For the interval {} to {}:\n\n'.format(t1str, t2str))
        r.add('Value = ', report.Number(value, fmin=2), ' ± ', report.Number(uncert*k, fmin=2), '(k={:.3g}{})'.format(k, confstr))
        if plot:
            with plt.style.context(plotting.plotstyle):
                fig, ax = plt.subplots()
                self.plot_interval_uncert(t1, t2, ax=ax, k=k, conf=conf, mode=kwargs.get('mode', self.predmode))
                r.plot(fig)
        return r

    def report_interval_uncert_eqns(self, subs=False, n=4, **kwargs):
        ''' Report equations used to find interval uncertainty. '''
        params = self.expr_interval()
        b, bbar, ub, ubbar, uy, Syx, uc = sympy.symbols('b bbar u_b u_bbar u_y S_yx u_c')

        r = report.Report(**kwargs)
        r.add('Correction function: ', sympy.Eq(b, self.expr(subs=False, full=False)), '\n\n')
        r.add('Mean correction: ', sympy.Eq(bbar, params['bbar_expr']), '\n\n')
        if subs:
            r.add(report.Math('='), params['bbar'], '\n\n')
        r.add('Variance in mean correction: ', sympy.Eq(ubbar**2, params['ubbar_expr']), '\n\n')
        if subs:
            r.add(report.Math('='), params['ubbar'], '\n\n')
        r.add('Variance in correction: ', sympy.Eq(ub**2, params['ub_expr']), '\n\n')
        if subs:
            if params['ub'] is not None:
                r.add(report.Math('='),  params['ub'], '\n\n')
            else:
                r.txt('  (integrated numerically.)\n\n')
        r.add('Other variance: ', sympy.Eq(uy**2, Syx**2), '\n\n')
        r.add('Total uncertainty for interval: ', sympy.Eq(uc, sympy.sqrt(ubbar**2 + ub**2 + uy**2)), '\n\n')
        return r

    def report_confpred(self, **kwargs):
        ''' Report equations for confidence and prediction intervals, only for line fit. '''
        r = report.Report(**kwargs)
        if self.model.fitname == 'line':
            r.txt('Confidence interval:\n\n')
            r.sympy(self.expr_uconf(subs=False), end='\n\n')
            r.sympy(self.expr_uconf(subs=True), '\n\n')
            r.div()
            r.txt('Prediction interval:\n\n')
            r.sympy(self.expr_upred(subs=False), end='\n\n')
            r.sympy(self.expr_upred(subs=True), end='\n\n')
        return r

    def report_confpred_xval(self, xval, k=2, conf=None, plot=False, mode=None, **kwargs):
        ''' Report confidence and prediction intervals with table
            and optional plot showing a specific x value

            Parameters
            ----------
            xval: float or string, or array
                Value(s) (or date string(s)) at which to predict
            k: float
                k-value to apply to uncertainty intervals
            conf: float
                Level of confidence (0 to 1). Overrides value of k.
            plot: bool
                Include a plot showing the full curve fit with the predicted value
            mode: string
                Prediction band mode. One of 'Syx', 'sigy', or 'sigylast'.
        '''
        xval = np.atleast_1d(xval)
        x = []
        for val in xval:
            if self.inputs.xdate:
                with suppress(AttributeError, ValueError, OverflowError):
                    x.append(mdates.date2num(parse(val)))
            else:
                with suppress(ValueError):
                    x.append(float(val))

        x = np.asarray(x)
        y = self.y(x)
        uconf = self.u_conf(x, k=k, conf=conf)
        upred = self.u_pred(x, k=k, conf=conf, mode=mode)
        kstr = '(k={})'.format(k) if conf is None else '({:.4g}%)'.format(conf*100)
        hdr = ['x', 'y', 'confidence interval {}'.format(kstr), 'prediction interval {}'.format(kstr)]
        rows = []
        for i in range(len(x)):
            yi, yconfminus, yconfplus, ypredminus, ypredplus = report.Number.number_array([y[i], y[i]-uconf[i], y[i]+uconf[i], y[i]-upred[i], y[i]+upred[i]], fmin=2, **kwargs)
            rows.append([str(xval[i]),
                         yi,
                         ('±', report.Number(uconf[i], fmin=2, **kwargs), ' (', yconfminus, ', ', yconfplus, ')'),
                         ('±', report.Number(upred[i], fmin=2, **kwargs), ' (', ypredminus, ', ', ypredplus, ')')])

        r = report.Report(**kwargs)
        if plot:
            with plt.style.context(plotting.plotstyle):
                xx = self._full_xrange(np.nanmin(x), np.nanmax(x))
                fig, ax = plt.subplots()
                self.plot_points(ax=ax, marker='o', ls='')
                self.plot_fit(ax=ax, x=xx, ls='-', label='Fit')
                self.plot_pred(ax=ax, x=xx, k=k, conf=conf, ls='--', mode=mode)
                self.plot_pred_value(ax=ax, xval=x, k=k, conf=conf, mode=mode)
                ax.set_xlabel(self.model.xname)
                ax.set_ylabel(self.model.yname)
                ax.legend(loc='best')
                r.plot(fig)
        r.table(rows, hdr=hdr)
        return r

    def _inputx(self, dates=False):
        ''' Return x value as float or date '''
        xdata = self.inputs.x
        if dates:
            return mdates.num2date(xdata)
        else:
            return xdata

    def _full_xrange(self, x1=None, x2=None, num=200):
        ''' Get a linspace covering the full range of the curve fit, plus other
            points of interest.
        '''
        mn = self._inputx().min()
        mx = self._inputx().max()
        if x1 is not None:
            mn = min(mn, x1)
            mx = max(mx, x1)
        if x2 is not None:
            mn = min(mn, x2)
            mx = max(mx, x2)
        xx = np.linspace(mn, mx, num=num)
        return xx

    def plot_points(self, ax=None, ebar=False, **kwargs):
        ''' Plot the original data points used in the line/curve fit

            Parameters
            ----------
            ax: matplotlib axis
                Axis to plot on
            ebar: bool
                Show points with errorbars

            Keyword Arguments
            -----------------
            Passed to matplotlib plot() method.
        '''
        fig, ax = plotting.initplot(ax)

        xdata, ydata, ux, uy = self.inputs.x, self.inputs.y, self.inputs.ux, self.inputs.uy
        if self.inputs.xdate:
            xdata = mdates.num2date(xdata)

        kwargs.setdefault('marker', 'o')
        kwargs.setdefault('ls', '')
        if ebar:
            ax.errorbar(xdata, ydata, yerr=uy, xerr=ux, **kwargs)
        else:
            ax.plot(xdata, ydata, **kwargs)

    def plot_fit(self, x=None, ax=None, **kwargs):
        ''' Plot the fit line/curve

            Parameters
            ----------
            x: array
                x values to plot
            ax: matplotlib axis
                Axis to plot on

            Keyword Arguments
            -----------------
            Passed to matplotlib plot() method.
        '''
        fig, ax = plotting.initplot(ax)

        if x is None:
            x = self._full_xrange()

        yfit = self.y(x)

        if self.inputs.xdate:
            x = mdates.num2date(x)

        ax.plot(x, yfit, **kwargs)

    def plot_conf(self, x=None, ax=None, absolute=False, k=1, conf=None, **kwargs):
        ''' Plot confidence band

            Parameters
            ----------
            x: array
                x values to plot
            ax: matplotlib axis
                Axis to plot on
            absolute: bool
                If False, will plot uncertainty on top of nominal value
            k: float
                k-value
            conf: float
                Level of confidence (0 to 1). Overrides value of k.

            Keyword Arguments
            -----------------
            Passed to matplotlib plot() method.
        '''
        fig, ax = plotting.initplot(ax)

        if x is None:
            x = self._full_xrange()

        if 'label' not in kwargs:
            kstr = '(k={:.3g})'.format(k) if conf is None else '({:.4g}%)'.format(conf*100)
            kwargs['label'] = 'Confidence Band {}'.format(kstr)

        u_conf = self.u_conf(x, k=k, conf=conf)
        if not absolute:
            # Plot wrt fit line
            yfit = self.y(x)
            xx = np.append(np.append(x, [1]), x)
            if self.inputs.xdate:
                xx = mdates.num2date(xx)
            yy = np.append(np.append(yfit + u_conf, [np.nan]), yfit - u_conf)
            ax.plot(xx, yy, **kwargs)
        else:
            ax.plot(self._inputx(self.xdate), u_conf, **kwargs)

    def plot_pred(self, x=None, ax=None, absolute=False, k=1, conf=None, mode=None, **kwargs):
        ''' Plot prediction band

            Parameters
            ----------
            x: array
                x values to plot
            ax: matplotlib axis
                Axis to plot on
            absolute: bool
                If False, will plot uncertainty on top of nominal value
            k: float
                k-value
            conf: float
                Level of confidence (0 to 1). Overrides value of k.
            mode: string
                Prediction band mode. One of 'Syx', 'sigy', or 'sigylast'.

            Keyword Arguments
            -----------------
            Passed to matplotlib plot() method.
        '''
        fig, ax = plotting.initplot(ax)

        if x is None:
            x = self._full_xrange()

        if 'label' not in kwargs:
            kstr = '(k={:.3g})'.format(k) if conf is None else '({:.4g}%)'.format(conf*100)
            kwargs['label'] = 'Prediction Band {}'.format(kstr)

        u_pred = self.u_pred(x, k=k, conf=conf, mode=mode)
        if not absolute:
            # Plot wrt fit line
            yfit = self.y(x)
            xx = np.append(np.append(x, [1]), x)
            if self.inputs.xdate:
                xx = mdates.num2date(xx)
            yy = np.append(np.append(yfit + u_pred, [np.nan]), yfit - u_pred)
            ax.plot(xx, yy, **kwargs)
        else:
            ax.plot(self._inputx(self.xdate), u_pred, **kwargs)

    def plot_pred_value(self, xval, k=2, conf=None, ax=None, **kwargs):
        ''' Plot a single prediction band value as errorbar

            Parameters
            ----------
            xval: float or string or array
                X-value(s) (float or string date) to predict
            k: float
                K-value to apply to uncertainty
            conf: float
                Level of confidence (0 to 1). Overrides value of k.
            ax: matplotlib axis
                Axis to plot on
            mode: string
                Prediction band mode. One of 'Syx', 'sigy', or 'sigylast'.
        '''
        xval = np.asarray(xval)
        fig, ax = plotting.initplot(ax)

        if self.inputs.xdate and isinstance(xval, str):
            xfloat = np.array([parse(x).tooridnal() for x in xval])
            xplot = mdates.num2date(xfloat)
        else:
            xfloat = xval
            xplot = xval

        y = self.y(xfloat)
        upred = self.u_pred(xfloat, k=k, conf=conf, **kwargs)
        ax.errorbar(xplot, y, yerr=upred, marker='s', markersize=8, capsize=4, label='Predicted Value', ls='', color='C4')

    def plot_summary(self, ax=None, k=1, conf=None, **kwargs):
        ''' Plot a summary of the curve fit, including original points,
            fit line, and confidence/prediction bands.

            Parameters
            ----------
            ax: matplotlib axis
                Axis to plot on
            k: float
                k-value for uncertainty bands
            conf: float
                Level of confidence (0 to 1). Overrides value of k.
        '''
        fig, ax = plotting.initplot(ax)

        self.plot_points(ax=ax, marker='o', ls='')
        self.plot_fit(ax=ax, ls='-', label='Fit')
        self.plot_conf(ax=ax, k=k, conf=conf, ls=':')
        self.plot_pred(ax=ax, k=k, conf=conf, ls='--', mode=kwargs.pop('mode', self.predmode))
        ax.set_xlabel(self.model.xname)
        ax.set_ylabel(self.model.yname)
        ax.legend(loc='best')

    def plot_interval_uncert(self, t1, t2, ax=None, k=2, conf=None, mode=None, **kwargs):
        ''' Plot uncertainty valid for given interval (GUM F.2.4.5) '''
        fig, ax = plotting.initplot(ax)

        if self.inputs.xdate and isinstance(t1, str):
            t1 = mdates.date2num(parse(t1))
            t2 = mdates.date2num(parse(t2))

        if conf is not None:
            k = t_factor(conf, self.degf)

        self.plot_points(ax=ax, marker='o', ls='')
        xx = self._full_xrange(t1, t2)
        self.plot_fit(x=xx, ax=ax, label='Fit')
        self.plot_pred(x=xx, ax=ax, k=k, conf=conf, ls='--', color='C3', mode=mode)
        ax.axvline(t1, ls=':', color='black')
        ax.axvline(t2, ls=':', color='black')

        value, uncert = self.uncert_interval(t1, t2)
        ax.plot([t1, t2], [value + uncert*k, value + uncert*k], ls=':', color='C2')
        ax.plot([t1, t2], [value - uncert*k, value - uncert*k], ls=':', color='C2')
        ax.errorbar((t2+t1)/2, value, yerr=uncert*k, capsize=4, marker='s', color='C4', label='Interval Value')
        ax.legend(loc='best')

    def plot_residuals(self, ax=None, hist=False, **kwargs):
        ''' Plot standardized residual values. (in terms of standard deviations)

            Parameters
            ----------
            ax: Matplotlib axis
                Axis to plot on. Will be created if None.
            hist: boolean
                Plot as histogram (True) or scatter (False)

            Keyword Arguments
            -----------------
            Passed to matplotlib plot() method.
        '''
        fig, ax = plotting.initplot(ax)

        if hist:
            with suppress(ValueError):  # Can raise if resids has only one value or is empty
                ax.hist(self.residuals.residuals, **kwargs)
        else:
            # Default to no line, points only
            if 'ls' not in kwargs and 'linestyle' not in kwargs:
                kwargs['ls'] = ''
            if 'marker' not in kwargs:
                kwargs['marker'] = 'o'
            x = self.inputs.x
            if self.inputs.xdate:
                x = mdates.num2date(x)
            ax.plot(x, self.residuals.residuals, **kwargs)

    def get_normprob(self):
        ''' Get normalized probability of residuals. Values should fall on straight
            diagonal line if assumption of normality is valid.

            Returns
            -------
            resid: array
                Sorted normalized residuals
            Fi: array
                Cumulative frequency at each residual
            corr: float
                Correlation coefficient between cumulative frequency and residuals.
                Should be close to 1 for normality assumption.

            References
            ----------
            [1] Applied Regression & Analysis of Variance, 2nd edition. S. Glantz,
                B. Slinker. McGraw-Hill, 2001., pg 130
        '''
        resid = self.residuals.residuals.copy()
        sy = np.sqrt(sum(resid**2)/(len(resid)-2))
        resid.sort()
        resid = resid / sy  # Normalized
        Fi = (np.arange(1, len(resid)+1) - 0.5)/len(resid)
        cor = np.corrcoef(resid, Fi)[0, 1]
        return resid, Fi, cor

    def plot_normprob(self, ax=None, showfit=True, **kwargs):
        ''' Plot normal probability of residuals. Values should fall on straight
            diagonal line if assumption of normality is valid.

            Parameters
            ----------
            ax: Matplotlib axis
                Axis to plot on. Will be created if None.
            showfit: boolean
                Show a fit line

            Keyword Arguments
            -----------------
            fitargs: dict
                Dictionary of plot parameters for fit line

            Other keyword arguments passed to plot for residuals.

            References
            ----------
            [1] Applied Regression & Analysis of Variance, 2nd edition. S. Glantz,
                B. Slinker. McGraw-Hill, 2001., pg 130
        '''
        fig, ax = plotting.initplot(ax)

        resid, Fi, cor = self.get_normprob()
        kwargs.setdefault('marker', '.')
        kwargs.setdefault('ls', '')
        ax.plot(resid, Fi, **kwargs)

        if showfit and any(np.isfinite(resid)):
            fitargs = kwargs.get('fitargs', {})
            fitargs.setdefault('ls', ':')
            with suppress(np.linalg.linalg.LinAlgError):
                p = np.polyfit(resid, Fi, deg=1)
                x = np.linspace(resid.min(), resid.max(), num=10)
                ax.plot(x, np.poly1d(p)(x), **fitargs)

    def plot_correlation(self, params=None, fig=None, **kwargs):
        ''' Plot curve fit parameters against each other to see correlation between
            slope and intercept, for example.

            Parameters
            ----------
            params: list of int
                List of indexes into parameter names to include in plot
            fig: matplotlib figure
                Figure to plot on. Will be cleared.
            MCcontour:
                Plot Monte-Carlo samples as contour plot

            Keyword Arguments
            -----------------
            Passed to matplotlib plot() method.
        '''
        if fig is None:
            fig = plt.gcf()

        fig.clf()
        if params is None:
            params = list(range(len(self.paramnames)))

        num = len(params)
        if num < 2: return fig
        for row in range(num):
            for col in range(num):
                if col <= row: continue

                idx1, idx2 = params[row], params[col]
                ax = fig.add_subplot(num-1, num-1, row*(num-1)+col)
                cov = np.array([[self.cov[idx1, idx1], self.cov[idx1, idx2]],
                                [self.cov[idx2, idx1], self.cov[idx2, idx2]]])
                try:
                    rv = stats.multivariate_normal(np.array([self.coeffs[idx1], self.coeffs[idx2]]), cov=cov)
                except (np.linalg.linalg.LinAlgError, ValueError):  # Singular matrix
                    continue
                x, y = np.meshgrid(np.linspace(self.coeffs[idx1]-3*self.uncerts[idx1], self.coeffs[idx1]+3*self.uncerts[idx1]),
                                   np.linspace(self.coeffs[idx2]-3*self.uncerts[idx2], self.coeffs[idx2]+3*self.uncerts[idx2]))
                ax.contour(rv.pdf(np.dstack((x, y))), extent=[x.min(), x.max(), y.min(), y.max()], **kwargs)
                ax.set_xlabel(self.paramnames[idx1])
                ax.set_ylabel(self.paramnames[idx2])
        return fig

    def test_t(self, pidx=0, nominal=0, conf=.95, verbose=False):
        ''' T-test of fit parameter is statistically different from nominal.
            Use defaults to test slope of linear fit.

            Parameters
            ----------
            pidx: int
                Index of fit parameter to test
            nominal: float
                Test that mean[pidx] is statistically different from nominal.
                For example, nominal=0 to test that a slope parameter is different
                than 0.
            conf: float
                Level of confidence (0-1) of range
            verbose: boolean
                Print values

            Returns
            -------
            p: boolean
                Test result
        '''
        pidx = self._index(pidx)
        t = abs(self.coeffs[pidx] - nominal) / self.uncerts[pidx]
        ta = stats.t.ppf(1-(1-conf)/2, df=self.degf)
        if verbose:
            print('{}: {} > {} --> {}'.format(self.paramnames[pidx], t, ta, t > ta))
        return t > ta

    def test_t_range(self, pidx=0, nominal=0, conf=.95, verbose=False):
        ''' Find confidence range of parameter values, test that it does not contain nominal
            value. Use defaults to test linear fit slope is different from 0 with 95% confidence.

            Parameters
            ----------
            pidx: int
                Index of fit parameter to test
            nominal: float
                Range must not contain this value for test to pass. For example,
                nominal=0 to test that a slope parameter is different than 0.
            conf: float
                Level of confidence (0-1) of range
            verbose: boolean
                Print values

            Returns
            -------
            p: boolean
                Test result
        '''
        pidx = self._index(pidx)
        b1 = self.coeffs[pidx]
        sb = self.uncerts[pidx]
        ta = stats.t.ppf(1-(1-conf)/2, df=self.degf)
        ok = not (b1 - ta*sb < nominal < b1 + ta*sb)
        if verbose:
            print(r'[{} < {} < {}] {} {} --> {}'.format(b1 - ta*sb, self.paramnames[pidx], b1 + ta*sb, u'\u2285', nominal, ok))
        return ok


class CurveFitOutputMC(CurveFitOutputLSQ):
    ''' Results from a Monte Carlo curve fit '''
    def _calccoeffs(self):
        ''' Run calculation here (so it can be subclassed) '''
        result = self.model.calc_MC()
        self.coeffs = result.coeff
        self.uncerts = result.uncert
        self.cov = result.covariance
        self.degf = result.degf
        self.residuals = result.residuals
        self.samples = result.samples
        self.cor = self.cov / self.uncerts[:, None] / self.uncerts[None, :]  # See numpy code for corrcoeff

    def plot_samples(self, fig=None, **kwargs):
        ''' Plot samples for each parameter (value vs. sample number) '''
        fig, ax = plotting.initplot(fig)

        inpts = kwargs.pop('inpts', list(range(len(self.coeffs))))
        fig.clf()
        fig.subplots_adjust(**out_uncert.dfltsubplots)
        axs = plotting.axes_grid(len(inpts), fig)

        for ax, inptnum in zip(axs, inpts):
            samples = self.samples[:, inptnum]
            ax.plot(samples, **kwargs)
            ax.set_ylabel('$' + self.paramnames[inptnum] + '$')
            ax.set_xlabel('Sample #')
        fig.tight_layout()
        return fig

    def plot_xhists(self, fig=None, **kwargs):
        ''' Plot histograms of x samples

            Parameters
            ----------
            fig: matplotlib figure
                Figure to plot on. (Existing figure will be cleared)

            Keyword Arguments
            -----------------
            inpts: list
                List of integer indexes of inputs to plot

            Keyword Arguments
            -----------------
            Passed to matplotlib hist() method.
        '''
        fig, ax = plotting.initplot(fig)

        inpts = kwargs.pop('inpts', list(range(len(self.paramnames))))
        fig.clf()
        fig.subplots_adjust(**out_uncert.dfltsubplots)
        axs = plotting.axes_grid(len(inpts), fig)

        for ax, inptnum in zip(axs, inpts):
            samples = self.samples[:, inptnum]
            ax.hist(samples, **kwargs)
            ax.ticklabel_format(style='sci', axis='x', scilimits=(-4, 4), useOffset=False)
            ax.yaxis.set_visible(False)
            ax.set_xlabel('$' + self.paramnames[inptnum] + '$')
        fig.tight_layout()
        return fig


class CurveFitOutputMCMC(CurveFitOutputMC):
    ''' Results from a Markov Chain Monte Carlo curve fit '''
    def _calccoeffs(self):
        ''' Run calculation here '''
        result = self.model.calc_MCMC()
        self.coeffs = result.coeff
        self.uncerts = result.uncert
        self.cov = result.covariance
        self.degf = result.degf
        self.residuals = result.residuals
        self.samples = result.samples
        self.acceptance = result.acceptance
        self.cor = self.cov / self.uncerts[:, None] / self.uncerts[None, :]  # See numpy code for corrcoeff

    def report_acceptance(self, **kwargs):
        ''' Report acceptance rate (MCMC fits only) '''
        hdr = ['Parameter', 'Acceptance Rate']
        rows = []
        for p, v in zip(self.paramnames, self.acceptance):
            rows.append([p, '{:.2f}%'.format(v*100)])
        r = report.Report(**kwargs)
        r.table(rows, hdr=hdr)
        return r


class CurveFitOutputGUM(CurveFitOutputLSQ):
    ''' Results from a GUM curve fit '''
    def _calccoeffs(self):
        ''' Run calculation here '''
        result = self.model.calc_GUM()
        self.coeffs = result.coeff
        self.uncerts = result.uncert
        self.cov = result.covariance
        self.degf = result.degf
        self.residuals = result.residuals
        self.samples = result.samples
        self.cor = self.cov / self.uncerts[:, None] / self.uncerts[None, :]  # See numpy code for corrcoeff


class CurveFitOutput(output.Output):
    ''' Results from a set of different calculation methods on a curve-fit '''
    methods = {'gum': 'GUM Approximation',
               'mc': 'Monte Carlo',
               'lsq': 'Least Squares',
               'mcmc': 'Markov-Chain Monte Carlo'}

    def __init__(self, model, inputs, **kwargs):
        self.model = model
        self.inputs = inputs  # Array object
        self.paramnames = self.model.pnames
        self.fitfunc = self.model.func
        self.name = self.model.name
        self.desc = self.model.desc

        self.lsq = None
        self.gum = None
        self.mc = None
        self.mcmc = None
        if kwargs.get('lsq', True):
            self.lsq = CurveFitOutputLSQ(self.model, self.inputs)
        if kwargs.get('gum', False):
            self.gum = CurveFitOutputGUM(self.model, self.inputs)
        if kwargs.get('mc', False):
            self.mc = CurveFitOutputMC(self.model, self.inputs)
        if kwargs.get('mcmc', False):
            self.mcmc = CurveFitOutputMCMC(self.model, self.inputs)

    def expr(self, full=True):
        ''' Return sympy expression for fit function

            Parameters
            ----------
            subs: bool
                Substitute calculated fit parameter values in result
            n: int
                Number of decimals to include
            full: bool
                Include the "y ="
        '''
        expr = self.model.expr
        if full:
            expr = sympy.Eq(sympy.Symbol('y'), expr)
        return expr

    def report(self, **kwargs):
        ''' Generate report of mean/uncertainty values.

            Keyword Arguments
            -----------------
            See report.Report
        '''
        # rows are GUM, MC, LSQ, etc. Columns are means, uncertainties. Checks if baseoutput is CurveFit, and if so uses that format.
        hdr = ['Method (k=1)']
        if self.paramnames is None:
            hdr.extend(['Mean', 'Standard\nUncertainty'])
        else:
            hdr.extend(self.paramnames)
        rows = []
        for method in ['lsq', 'gum', 'mc', 'mcmc']:
            out = getattr(self, method)
            if out is not None:

                row = [self.methods[method]]
                for c, s in zip(out.coeffs, out.uncerts):
                    row.extend([(report.Number(c, matchto=s, fmin=0), ' ± ', report.Number(s))])
                rows.append(row)

        r = report.Report(**kwargs)
        r.table(rows, hdr)
        return r

    def report_all(self, k=2, conf=None, **kwargs):
        ''' Report all info on curve fit, including summary, residual plots, and correlations '''
        r = report.Report(**kwargs)
        r.hdr('Curve Fit', level=2)

        outs = [(getattr(self, m), m) for m in ['lsq', 'gum', 'mc', 'mcmc']]
        outs = [out for out in outs if out[0] is not None]
        for out, method in outs:

            if len(outs) > 1:
                r.hdr('Method: {}'.format(method), level=3)
            
            if kwargs.get('summary', True):
                r.append(out.report_summary(k=k, conf=conf, **kwargs))

            if kwargs.get('fitplot', True):
                fig, ax = plotting.initplot()
                out.plot_summary(ax=ax, k=k, conf=conf, **kwargs)
                r.plot(fig)
                plt.close(fig)
            
            if kwargs.get('goodness', True):
                r.append(out.report_fit(**kwargs))

            if kwargs.get('confpred', False):
                r.append(out.report_confpred(**kwargs))

            if kwargs.get('prediction', False):
                r.append(out.report_confpred_xval(kwargs['xvals'], k=k, conf=conf, plot=True, mode=kwargs.get('mode', 'Syx')))

            if kwargs.get('interval') is not None:
                r.append(out.report_interval_uncert(*kwargs.get('interval'), k=k, conf=conf))

            if kwargs.get('residuals', False):
                r.div()
                r.hdr('Residuals', level=3)
                r.append(out.report_residuals(k=k, conf=conf, **kwargs))

            if kwargs.get('correlations', False):
                r.div()
                r.hdr('Correlations', level=3)
                r.append(out.report_correlation())

        return r

    def get_dists(self):
        ''' Get distributions in this output. If name is none, return a list of
            available distribution names.
        '''
        dists = {}
        for method in ['lsq', 'gum', 'mc', 'mcmc']:
            if getattr(self, method) is not None:
                out = getattr(self, method)
                for pidx, param in enumerate(self.paramnames):
                    if 'mc' in method:
                        dists[f'{param} ({method.upper()})'] = {'samples': out.samples[:, pidx]}
                    else:
                        dists[f'{param} ({method.upper()})'] = {'mean': out.coeffs[pidx],
                                                                'std': out.uncerts[pidx],
                                                                'df': out.degf}
        for method in ['lsq', 'gum', 'mc', 'mcmc']:
            if getattr(self, method) is not None:
                out = getattr(self, method)
                dists[f'Confidence ({method.upper()})'] = \
                        {'xdates': self.inputs.xdate,
                         'function': lambda x, out=out: {'mean': out.y(x), 'std': out.u_conf(x), 'df': out.degf}}

                dists[f'Prediction ({method.upper()})'] = \
                        {'xdates': self.inputs.xdate,
                         'function': lambda x, out=out: {'mean': out.y(x), 'std': out.u_pred(x), 'df': out.degf}}
        return dists

