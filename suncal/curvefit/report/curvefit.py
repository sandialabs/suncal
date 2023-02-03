''' Reports for curve fitting calculations '''

from contextlib import suppress
import numpy as np
import sympy
from scipy import stats
from dateutil.parser import parse
import matplotlib.dates as mdates

from ...common import report, plotting
from ...common.ttable import k_factor


class ReportCurveFit:
    ''' Report curve fit results

        Args:
            fitresults: CurveFitResults instance
    '''
    def __init__(self, fitresults):
        self._results = fitresults
        self.plot = PlotCurveFit(self._results)

    def _repr_markdown_(self):
        return self.summary().get_md()

    def summary(self, **kwargs):
        ''' Report a summary of results '''
        hdr = ['Parameter', 'Nominal', 'Standard Uncertainty']
        rows = []
        for name, val, unc in zip(self._results.setup.coeffnames, self._results.coeffs, self._results.uncerts):
            rows.append([name,
                         report.Number(val, matchto=unc),
                         report.Number(unc)])
        r = report.Report(**kwargs)
        r.table(rows, hdr=hdr)
        return r

    def goodness_fit(self, **kwargs):
        ''' Report goodness-of-fit values r-squared and Syx. Note r is not a good predictor
            of fit for nonlinear models, but we report it anyway.
        '''
        residuals = self._results.residuals
        rows = []
        rows.append([report.Number(residuals.r, fmt='decimal'),
                     report.Number(residuals.r**2, fmt='decimal'),
                     report.Number(residuals.Syx),
                     report.Number(residuals.F, fmt='auto')])
        r = report.Report(**kwargs)
        r.table(rows, ['r', 'r-squared', 'Standard Error (Syx)', 'F-value'])
        return r

    def correlation(self, **kwargs):
        ''' Report table of correlation coefficients '''
        hdr = ['Parameter'] + self._results.setup.coeffnames
        rows = []
        for idx, row in enumerate(self._results.correlation):
            rows.append([self._results.setup.coeffnames[idx]] + [report.Number(v) for v in row])
        r = report.Report(**kwargs)
        r.table(rows, hdr=hdr)
        return r

    def residuals(self, **kwargs):
        ''' Report a plot of residuals, histogram, and normal-probability '''
        r = report.Report(**kwargs)
        with plotting.plot_figure() as fig:
            ax = fig.add_subplot(2, 2, 1)
            self.plot.points(ax=ax, ls='', marker='o')
            self.plot.fit(ax=ax)
            ax.set_title('Fit Line')
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax2 = fig.add_subplot(2, 2, 2)
            self.plot.residuals(ax=ax2, hist=True)
            ax2.set_title('Residual Histogram')
            ax2.set_xlabel(r'$\Delta$ y')
            ax2.set_ylabel('Probability')
            ax3 = fig.add_subplot(2, 2, 3)
            self.plot.residuals(ax=ax3, hist=False)
            ax3.axhline(0, color='C1')
            ax3.set_title('Raw Residuals')
            ax3.set_xlabel('x')
            ax3.set_ylabel(r'$\Delta$ y')
            ax4 = fig.add_subplot(2, 2, 4)
            self.plot.normprob(ax=ax4)
            ax4.set_title('Normal Probability')
            ax4.set_xlabel('Theoretical quantiles')
            ax4.set_ylabel('Ordered sample values')
            fig.tight_layout()
            r.plot(fig)
        r.append(self.residual_table(**kwargs))
        return r

    def residual_table(self, k=2, conf=None, **kwargs):
        ''' Report table of measured values and their residuals

            Args:
                k (float): Coverage factor
                conf (float): Level of confidence, overrides k
        '''
        rpt = report.Report(**kwargs)
        kstr = f'(k={k})' if conf is None else f'({conf*100:.4g}%)'
        hdr = ['Measured x', 'Measured y', 'Predicted y', 'Residual',
               f'Confidence Band {kstr}', f'Prediction Band {kstr}']
        x, y = self._results.setup.points.x, self._results.setup.points.y
        if self._results.setup.points.xdate:
            xstring = mdates.num2date(x)
            xstring = [k.strftime('%d-%b-%Y') for k in xstring]
        else:
            xstring = [str(k) for k in x]
        residuals = self._results.residuals.residuals
        rows = []
        for xs, xx, yy, resid in zip(xstring, x, y, residuals):
            confband = self._results.confidence_band(xx, k=k, conf=conf)
            rows.append([f'{xs}',
                         report.Number(yy, matchto=confband),
                         report.Number(self._results.y(xx), matchto=confband),
                         report.Number(resid, matchto=confband),
                         report.Number(confband),
                         report.Number(self._results.prediction_band(xx, k=k, conf=conf))])
        rpt.table(rows, hdr=hdr)
        return rpt

    def interval_uncert(self, t1, t2, k=2, conf=None, plot=True, **kwargs):
        ''' Report the value and uncertainty that applies to the
            entire interval from t1 to t2.

            Args:
                t1 (float): beginning of interval
                t2 (float): end of interval
                k (float): Coverage factor
                conf (float): Level of confidence, overrides k
                plot (bool): Include a plot of the curve
        '''
        confstr = ''
        if conf is not None:
            k = k_factor(conf, self._results.degf)
            confstr = f', {conf*100:.4g}%'

        if self._results.setup.points.xdate and isinstance(t1, str):
            t1 = mdates.date2num(parse(t1))
            t2 = mdates.date2num(parse(t2))

        if self._results.setup.points.xdate:
            t1str = mdates.num2date(t1).strftime('%d-%b-%Y')
            t2str = mdates.num2date(t2).strftime('%d-%b-%Y')
        else:
            t1str, t2str = report.Number.number_array([t1, t2], thresh=8)

        value, uncert = self._results.interval_uncertainty(t1, t2)
        r = report.Report(**kwargs)
        r.txt(f'For the interval {t1str} to {t2str}:\n\n')
        r.add('Value = ', report.Number(value, fmin=2), ' ± ',
              report.Number(uncert*k, fmin=2), f'(k={k:.3g}{confstr})')
        if plot:
            with plotting.plot_figure() as fig:
                ax = fig.add_subplot(1, 1, 1)
                self.plot.interval_uncert(t1, t2, ax=ax, k=k, conf=conf, mode=kwargs.get('mode', 'Syx'))
                r.plot(fig)
        return r

    def interval_uncert_eqns(self, subs=False, **kwargs):
        ''' Report equations used to find interval uncertainty.

            Args:
                subs (bool): Substitute values into equations
        '''
        params = self._results.interval_expression()
        b, bbar, ub, ubbar, uy, Syx, uc = sympy.symbols('b bbar u_b u_bbar u_y S_yx u_c')

        rpt = report.Report(**kwargs)
        rpt.add('Correction function: ', sympy.Eq(b, self._results.fit_expr(subs=False, full=False)), '\n\n')
        rpt.add('Mean correction: ', sympy.Eq(bbar, params['bbar_expr']), '\n\n')
        if subs:
            rpt.add(report.Math('='), params['bbar'], '\n\n')
        rpt.add('Variance in mean correction: ', sympy.Eq(ubbar**2, params['ubbar_expr']), '\n\n')
        if subs:
            rpt.add(report.Math('='), params['ubbar'], '\n\n')
        rpt.add('Variance in correction: ', sympy.Eq(ub**2, params['ub_expr']), '\n\n')
        if subs:
            if params['ub'] is not None:
                rpt.add(report.Math('='),  params['ub'], '\n\n')
            else:
                rpt.txt('  (integrated numerically.)\n\n')
        rpt.add('Other variance: ', sympy.Eq(uy**2, Syx**2), '\n\n')
        rpt.add('Total uncertainty for interval: ', sympy.Eq(uc, sympy.sqrt(ubbar**2 + ub**2 + uy**2)), '\n\n')
        return rpt

    def report_confpred(self, **kwargs):
        ''' Report equations for confidence and prediction intervals, only for line fit. '''
        rpt = report.Report(**kwargs)
        if self._results.setup.modelname == 'line':
            rpt.txt('Confidence interval:\n\n')
            rpt.sympy(self._results.confidence_expr(subs=False), end='\n\n')
            rpt.sympy(self._results.confidence_expr(subs=True), '\n\n')
            rpt.div()
            rpt.txt('Prediction interval:\n\n')
            rpt.sympy(self._results.prediction_expr(subs=False), end='\n\n')
            rpt.sympy(self._results.prediction_expr(subs=True), end='\n\n')
        return rpt

    def confpred_xval(self, xval, k=2, conf=None, plot=False, mode='Syx', **kwargs):
        ''' Report confidence and prediction bands with table
            and optional plot showing a specific x value

            Args:
                xval (float or string, or array): Value(s) (or date string(s)) at which to predict
                k (float): Coverage factor for bands
                conf (float): Level of confidence (0 to 1). Overrides value of k.
                plot (bool): Include a plot showing the full curve fit with the predicted value
                mode (string): Prediction band mode. One of 'Syx', 'sigy', or 'sigylast'.
        '''
        xval = np.atleast_1d(xval)
        x = []
        for val in xval:
            if self._results.setup.points.xdate:
                with suppress(AttributeError, ValueError, OverflowError):
                    x.append(mdates.date2num(parse(val)))
            else:
                with suppress(ValueError):
                    x.append(float(val))

        x = np.asarray(x)
        y = self._results.y(x)
        uconf = self._results.confidence_band(x, k=k, conf=conf)
        upred = self._results.prediction_band(x, k=k, conf=conf, mode=mode)
        kstr = f'(k={k})' if conf is None else f'({conf*100:.4g}%)'
        hdr = ['x', 'y', f'confidence interval {kstr}', f'prediction interval {kstr}']
        rows = []
        for i in range(len(x)):
            yi, yconfminus, yconfplus, ypredminus, ypredplus = report.Number.number_array(
                [y[i], y[i]-uconf[i], y[i]+uconf[i], y[i]-upred[i], y[i]+upred[i]], fmin=2, **kwargs)
            rows.append([str(xval[i]),
                         yi,
                         ('±', report.Number(uconf[i], fmin=2, **kwargs), ' (', yconfminus, ', ', yconfplus, ')'),
                         ('±', report.Number(upred[i], fmin=2, **kwargs), ' (', ypredminus, ', ', ypredplus, ')')])

        r = report.Report(**kwargs)
        if plot:
            with plotting.plot_figure() as fig:
                ax = fig.add_subplot(1, 1, 1)
                self.plot.summary_prediction(x, k=k, conf=conf, mode=mode, ax=ax)
                r.plot(fig)
        r.table(rows, hdr=hdr)
        return r

    def acceptance(self, **kwargs):
        ''' Report acceptance rate (MCMC fits only) '''
        if self._results.acceptance is None:
            raise ValueError('Acceptance report only for Markov Chain Monte Carlo')
        hdr = ['Parameter', 'Acceptance Rate']
        rows = []
        for param, value in zip(self._results.setup.coeffnames, self._results.acceptance):
            rows.append([param, f'{value*100:.2f}%'])
        rpt = report.Report(**kwargs)
        rpt.table(rows, hdr=hdr)
        return rpt


class PlotCurveFit:
    ''' Plot results of curve fit '''
    def __init__(self, fitresults):
        self._results = fitresults

    def _inputx(self, dates=False):
        ''' Return x value as float or date '''
        xdata = self._results.setup.points.x
        if dates:
            return mdates.num2date(xdata)
        return xdata

    def _full_xrange(self, x1=None, x2=None, num=200):
        ''' Get a linspace covering the full range of the curve fit, plus other
            points of interest.
        '''
        mn = self._results.setup.points.x.min()  # _inputx???
        mx = self._results.setup.points.x.max()
        if x1 is not None:
            mn = min(mn, x1)
            mx = max(mx, x1)
        if x2 is not None:
            mn = min(mn, x2)
            mx = max(mx, x2)
        xx = np.linspace(mn, mx, num=num)
        return xx

    def points(self, ax=None, ebar=False, **kwargs):
        ''' Plot the original data points used in the line/curve fit

            Args:
                ax (plt.axes): Axis to plot on
                ebar (bool): Show points with errorbars
                **kwargs: passed to matplotlib plot
        '''
        fig, ax = plotting.initplot(ax)

        xdata, ydata = self._results.setup.points.x, self._results.setup.points.y
        ux, uy = self._results.setup.points.ux, self._results.setup.points.uy
        if self._results.setup.points.xdate:
            xdata = mdates.num2date(xdata)

        kwargs.setdefault('marker', 'o')
        kwargs.setdefault('ls', '')
        if ebar:
            ax.errorbar(xdata, ydata, yerr=uy, xerr=ux, **kwargs)
        else:
            ax.plot(xdata, ydata, **kwargs)

    def fit(self, x=None, ax=None, **kwargs):
        ''' Plot the fit line/curve

            Args:
                x (array): x values to include in plot
                ax (plt.axes): Axis to plot on
                **kwargs: passed to matplotlib plot
        '''
        _, ax = plotting.initplot(ax)

        if x is None:
            x = self._full_xrange()

        yfit = self._results.y(x)

        if self._results.setup.points.xdate:
            x = mdates.num2date(x)

        ax.plot(x, yfit, **kwargs)

    def conf(self, x=None, ax=None, absolute=False, k=1, conf=None, **kwargs):
        ''' Plot confidence band

            Args:
                x (array): x values to include in plot
                ax (plt.axes): Axis to plot on
                absolute (bool): If False, will plot uncertainty on top of nominal value
                k (float): Coverage factor for bands
                conf (float): Level of confidence (0 to 1). Overrides value of k.
                **kwargs: passed to matplotlib plot
        '''
        _, ax = plotting.initplot(ax)

        if x is None:
            x = self._full_xrange()

        if 'label' not in kwargs:
            kstr = f'(k={k:.3g})' if conf is None else f'({conf*100:.4g}%)'
            kwargs['label'] = f'Confidence Band {kstr}'

        u_conf = self._results.confidence_band(x, k=k, conf=conf)
        if not absolute:
            # Plot wrt fit line
            yfit = self._results.y(x)
            xx = np.append(np.append(x, [1]), x)
            if self._results.setup.points.xdate:
                xx = mdates.num2date(xx)
            yy = np.append(np.append(yfit + u_conf, [np.nan]), yfit - u_conf)
            ax.plot(xx, yy, **kwargs)
        else:
            ax.plot(self._inputx(self._results.setup.points.xdate), u_conf, **kwargs)

    def pred(self, x=None, ax=None, absolute=False, k=1, conf=None, mode='Syx', **kwargs):
        ''' Plot prediction band

            Args:
                x (array): x values to include in plot
                ax (plt.axes): Axis to plot on
                absolute (bool): If False, will plot uncertainty on top of nominal value
                k (float): Coverage factor for bands
                conf (float): Level of confidence (0 to 1). Overrides value of k.
                mode (string): Prediction band mode. One of 'Syx', 'sigy', or 'sigylast'.
                **kwargs: passed to matplotlib plot
        '''
        _, ax = plotting.initplot(ax)

        if x is None:
            x = self._full_xrange()

        if 'label' not in kwargs:
            kstr = f'(k={k:.3g})' if conf is None else f'({conf*100:.4g}%)'
            kwargs['label'] = f'Prediction Band {kstr}'

        u_pred = self._results.prediction_band(x, k=k, conf=conf, mode=mode)
        if not absolute:
            # Plot wrt fit line
            yfit = self._results.y(x)
            xx = np.append(np.append(x, [1]), x)
            if self._results.setup.points.xdate:
                xx = mdates.num2date(xx)
            yy = np.append(np.append(yfit + u_pred, [np.nan]), yfit - u_pred)
            ax.plot(xx, yy, **kwargs)
        else:
            ax.plot(self._inputx(self._results.setup.points.xdate), u_pred, **kwargs)

    def pred_value(self, xval, k=2, conf=None, ax=None, **kwargs):
        ''' Plot a single prediction band value as errorbar

            Args:
                x (array): x values to include in plot
                ax (plt.axes): Axis to plot on
                k (float): Coverage factor for bands
                conf (float): Level of confidence (0 to 1). Overrides value of k.
                **kwargs: passed to matplotlib plot
        '''
        xval = np.asarray(xval)
        _, ax = plotting.initplot(ax)

        if self._results.setup.points.xdate and isinstance(xval, str):
            xfloat = np.array([parse(x).tooridnal() for x in xval])
            xplot = mdates.num2date(xfloat)
        else:
            xfloat = xval
            xplot = xval

        y = self._results.y(xfloat)
        upred = self._results.prediction_band(xfloat, k=k, conf=conf, **kwargs)
        ax.errorbar(xplot, y, yerr=upred, marker='s', markersize=8,
                    capsize=4, label='Predicted Value', ls='', color='C4')

    def summary_prediction(self, x, k=2, conf=None, ax=None, mode='Syx', **kwargs):
        ''' Plot summary of prediction band showing specific x values

            Args:
                x (array): x values to include in plot
                ax (plt.axes): Axis to plot on
                k (float): Coverage factor for bands
                conf (float): Level of confidence (0 to 1). Overrides value of k.
                **kwargs: passed to matplotlib plot
        '''
        _, ax = plotting.initplot(ax)
        xx = self._full_xrange(np.nanmin(x), np.nanmax(x))
        self.points(ax=ax, marker='o', ls='')
        self.fit(ax=ax, x=xx, ls='-', label='Fit')
        self.pred(ax=ax, x=xx, k=k, conf=conf, ls='--', mode=mode)
        self.pred_value(ax=ax, xval=x, k=k, conf=conf, mode=mode)
        ax.set_xlabel(self._results.setup.xname)
        ax.set_ylabel(self._results.setup.yname)
        ax.legend(loc='best')

    def summary(self, ax=None, k=1, conf=None, **kwargs):
        ''' Plot a summary of the curve fit, including original points,
            fit line, and confidence/prediction bands.

            Args:
                ax (plt.axes): Axis to plot on
                k (float): Coverage factor for bands
                conf (float): Level of confidence (0 to 1). Overrides value of k.
        '''
        _, ax = plotting.initplot(ax)

        self.points(ax=ax, marker='o', ls='')
        self.fit(ax=ax, ls='-', label='Fit')
        self.conf(ax=ax, k=k, conf=conf, ls=':')
        self.pred(ax=ax, k=k, conf=conf, ls='--', mode=kwargs.pop('mode', 'Syx'))
        ax.set_xlabel(self._results.setup.xname)
        ax.set_ylabel(self._results.setup.yname)
        ax.legend(loc='best')

    def interval_uncert(self, t1, t2, ax=None, k=2, conf=None, mode='Syx', **kwargs):
        ''' Plot uncertainty valid for given interval (GUM F.2.4.5)

            Args:
                t1 (float): beginning of interval
                t2 (float): end of interval
                ax (plt.axes): Axis to plot on
                k (float): Coverage factor
                conf (float): Level of confidence, overrides k
                mode (string): Prediction band mode. One of 'Syx', 'sigy', or 'sigylast'.
        '''
        _, ax = plotting.initplot(ax)

        if self._results.setup.points.xdate and isinstance(t1, str):
            t1 = mdates.date2num(parse(t1))
            t2 = mdates.date2num(parse(t2))

        if conf is not None:
            k = k_factor(conf, self._results.degf)

        self.points(ax=ax, marker='o', ls='')
        xx = self._full_xrange(t1, t2)
        self.fit(x=xx, ax=ax, label='Fit')
        self.pred(x=xx, ax=ax, k=k, conf=conf, ls='--', color='C3', mode=mode)
        ax.axvline(t1, ls=':', color='black')
        ax.axvline(t2, ls=':', color='black')

        value, uncert = self._results.interval_uncertainty(t1, t2)
        ax.plot([t1, t2], [value + uncert*k, value + uncert*k], ls=':', color='C2')
        ax.plot([t1, t2], [value - uncert*k, value - uncert*k], ls=':', color='C2')
        ax.errorbar((t2+t1)/2, value, yerr=uncert*k, capsize=4, marker='s', color='C4', label='Interval Value')
        ax.legend(loc='best')

    def residuals(self, ax=None, hist=False, **kwargs):
        ''' Plot standardized residual values. (in terms of standard deviations)

            Args:
                ax (plt.axes): Axis to plot on. Will be created if None.
                hist (boolean): Plot as histogram (True) or scatter (False)
                **kwargs: passed to matplotlib
        '''
        _, ax = plotting.initplot(ax)

        if hist:
            with suppress(ValueError):  # Can raise if resids has only one value or is empty
                ax.hist(self._results.residuals.residuals, **kwargs)
        else:
            # Default to no line, points only
            if 'ls' not in kwargs and 'linestyle' not in kwargs:
                kwargs['ls'] = ''
            if 'marker' not in kwargs:
                kwargs['marker'] = 'o'
            x = self._results.setup.points.x
            if self._results.setup.points.xdate:
                x = mdates.num2date(x)
            ax.plot(x, self._results.residuals.residuals, **kwargs)

    def _get_normprob(self):
        ''' Get normalized probability of residuals. Values should fall on straight
            diagonal line if assumption of normality is valid.

            Args:
                resid (array): Sorted normalized residuals
                Fi (array): Cumulative frequency at each residual
                corr (float): Correlation coefficient between cumulative frequency and
                    residuals. Should be close to 1 for normality assumption.

            References:
                [1] Applied Regression & Analysis of Variance, 2nd edition. S. Glantz,
                    B. Slinker. McGraw-Hill, 2001., pg 130
        '''
        resid = self._results.residuals.residuals.copy()
        sy = np.sqrt(sum(resid**2)/(len(resid)-2))
        resid.sort()
        resid = resid / sy  # Normalized
        Fi = (np.arange(1, len(resid)+1) - 0.5)/len(resid)
        cor = np.corrcoef(resid, Fi)[0, 1]
        return resid, Fi, cor

    def normprob(self, ax=None, showfit=True, **kwargs):
        ''' Plot normal probability of residuals. Values should fall on straight
            diagonal line if assumption of normality is valid.

            Args:
                ax (plt.axes): Axis to plot on. Will be created if None.
                showfit (boolean): Show a fit line
                fitargs (dict): Dictionary of plot parameters for fit line
                    Other keyword arguments passed to plot for residuals.

            References:
                [1] Applied Regression & Analysis of Variance, 2nd edition. S. Glantz,
                    B. Slinker. McGraw-Hill, 2001., pg 130
        '''
        _, ax = plotting.initplot(ax)

        resid, Fi, _ = self._get_normprob()
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

    def correlation(self, params=None, fig=None, **kwargs):
        ''' Plot curve fit parameters against each other to see correlation between
            slope and intercept, for example.

            Args:
                params (list of int): List of indexes into parameter names to include in plot
                fig (plt.Figure): Figure to plot on. Will be cleared.
                **kwargs: passed to matplotlib
        '''
        fig, _ = plotting.initplot(fig)
        fig.clf()
        if params is None:
            params = list(range(len(self._results.setup.coeffnames)))

        num = len(params)
        if num < 2:
            return fig

        covariance = self._results.covariance
        for row in range(num):
            for col in range(num):
                if col <= row:
                    continue

                idx1, idx2 = params[row], params[col]
                ax = fig.add_subplot(num-1, num-1, row*(num-1)+col)
                cov = np.array([[covariance[idx1, idx1], covariance[idx1, idx2]],
                                [covariance[idx2, idx1], covariance[idx2, idx2]]])
                try:
                    rv = stats.multivariate_normal(
                        np.array([self._results.coeffs[idx1], self._results.coeffs[idx2]]), cov=cov)
                except (np.linalg.linalg.LinAlgError, ValueError):  # Singular matrix
                    continue
                x, y = np.meshgrid(np.linspace(self._results.coeffs[idx1] - 3 * self._results.uncerts[idx1],
                                               self._results.coeffs[idx1] + 3 * self._results.uncerts[idx1]),
                                   np.linspace(self._results.coeffs[idx2] - 3 * self._results.uncerts[idx2],
                                               self._results.coeffs[idx2] + 3 * self._results.uncerts[idx2]))
                ax.contour(rv.pdf(np.dstack((x, y))), extent=[x.min(), x.max(), y.min(), y.max()], **kwargs)
                ax.set_xlabel(self._results.setup.coeffnames[idx1])
                ax.set_ylabel(self._results.setup.coeffnames[idx2])
        return fig

    def samples(self, fig=None, coeffnames=None, **kwargs):
        ''' Plot samples for each parameter (value vs. sample number) '''
        if self._results.samples is None:
            raise ValueError('Must run Monte Carlo calculation to generate samples plot')

        fig, _ = plotting.initplot(fig)

        if coeffnames is None:
            coeffnames = self._results.setup.coeffnames

        fig.clf()
        axs = plotting.axes_grid(len(coeffnames), fig)

        for ax, cname in zip(axs, coeffnames):
            idx = self._results.setup.coeffnames.index(cname)
            samples = self._results.samples[:, idx]
            ax.plot(samples, **kwargs)
            ax.set_ylabel(f'${cname}$')
            ax.set_xlabel('Sample #')
        fig.tight_layout()
        return fig

    def xhists(self, fig=None, coeffnames=None, **kwargs):
        ''' Plot histograms of x samples

            Args:
                fig (plt.Figure): Figure to plot on. (Existing figure will be cleared)
                coeffnames (list): List of names of coefficients to include
                **kwargs: passed to matplotlib
        '''
        if self._results.samples is None:
            raise ValueError('Must run Monte Carlo calculation to generate samples plot')

        fig, _ = plotting.initplot(fig)
        if coeffnames is None:
            coeffnames = self._results.setup.coeffnames
        fig.clf()
        axs = plotting.axes_grid(len(coeffnames), fig)

        for ax, cname in zip(axs, coeffnames):
            idx = self._results.setup.coeffnames.index(cname)
            samples = self._results.samples[:, idx]
            ax.hist(samples, **kwargs)
            ax.ticklabel_format(style='sci', axis='x', scilimits=(-4, 4), useOffset=False)
            ax.yaxis.set_visible(False)
            ax.set_xlabel(f'${cname}$')
        fig.tight_layout()
        return fig


class ReportCurveFitCombined:
    ''' Results from multiple curve fit calculation types

        Args:
            lsq: CurveFitResults of least-squares calculation
            montecarlo: CurveFitResults of Monte Carlo calculation
            markov: CurveFitResults of Markov-Chain MC calculation
            gum: CurveFitResults of GUM calculation
    '''
    # Link attribute names with nice formatted strings
    _METHODS = {'gum': 'GUM Approximation',
                'montecarlo': 'Monte Carlo',
                'lsq': 'Least Squares',
                'markov': 'Markov-Chain Monte Carlo'}

    def __init__(self, results):
        self._results = results
        self.lsq = ReportCurveFit(self._results.lsq) if self._results.lsq else None
        self.gum = ReportCurveFit(self._results.gum) if self._results.gum else None
        self.montecarlo = ReportCurveFit(self._results.montecarlo) if self._results.montecarlo else None
        self.markov = ReportCurveFit(self._results.markov) if self._results.markov else None

    def _repr_markdown_(self):
        return self.summary().get_md()

    @property
    def paramnames(self):
        ''' Get parameter names '''
        if self._results.lsq:
            return self._results.lsq.setup.coeffnames
        if self._results.montecarlo:
            return self._results.montecarlo.setup.coeffnames
        if self._results.markov:
            return self._results.markov.setup.coeffnames
        if self._results.gum:
            return self._results.gum.setup.coeffnames
        return None

    @property
    def points(self):
        ''' Get x,y Array '''
        if self._results.lsq:
            return self._results.lsq.setup.points
        if self._results.montecarlo:
            return self._results.montecarlo.setup.points
        if self._results.markov:
            return self._results.markov.setup.points
        if self._results.gum:
            return self._results.gum.setup.points
        return None

    def get_report(self, method):
        ''' Get report from a specific method '''
        assert method in ['lsq', 'montecarlo', 'markov', 'gum']
        return getattr(self, method)

    def summary(self, **kwargs):
        ''' Generate report of mean/uncertainty values. '''
        # rows are GUM, MC, LSQ, etc. Columns are means, uncertainties.
        # Checks if baseoutput is CurveFit, and if so uses that format.
        hdr = ['Method (k=1)']
        hdr.extend(self.paramnames)
        rows = []
        for method in self._METHODS:
            methodreport = getattr(self, method)
            if methodreport is not None:

                row = [self._METHODS[method]]
                for coef, std in zip(methodreport._results.coeffs, methodreport._results.uncerts):
                    row.extend([(report.Number(coef, matchto=std, fmin=0), ' ± ', report.Number(std))])
                rows.append(row)

        rpt = report.Report(**kwargs)
        rpt.table(rows, hdr)
        return rpt

    def all(self, k=2, conf=None, **kwargs):
        ''' Report all info on curve fit, including summary, residual plots, and correlations '''
        rpt = report.Report(**kwargs)
        rpt.hdr('Curve Fit', level=2)

        for method in self._METHODS:
            methodreport = getattr(self, method)
            if methodreport is None:
                continue

            rpt.hdr(f'Method: {self._METHODS[method]}', level=3)

            if kwargs.get('summary', True):
                rpt.append(methodreport.summary(k=k, conf=conf, **kwargs))

            if kwargs.get('fitplot', True):
                with plotting.plot_figure() as fig:
                    ax = fig.add_subplot(1, 1, 1)
                    methodreport.plot.summary(ax=ax, k=k, conf=conf, **kwargs)
                    rpt.plot(fig)

            if kwargs.get('goodness', True):
                rpt.append(methodreport.goodness_fit(**kwargs))

            if kwargs.get('confpred', False):
                rpt.append(methodreport.report_confpred(**kwargs))

            if kwargs.get('prediction', False):
                rpt.append(methodreport.confpred_xval(
                    kwargs['xvals'], k=k, conf=conf, plot=True, mode=kwargs.get('mode', 'Syx')))

            if kwargs.get('interval') is not None:
                rpt.append(methodreport.interval_uncert(*kwargs.get('interval'), k=k, conf=conf))

            if kwargs.get('residuals', False):
                rpt.div()
                rpt.hdr('Residuals', level=3)
                rpt.append(methodreport.residuals(k=k, conf=conf, **kwargs))

            if kwargs.get('correlations', False):
                rpt.div()
                rpt.hdr('Correlations', level=3)
                rpt.append(methodreport.correlation())
        return rpt
