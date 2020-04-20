''' Output reports for curve fitting calculations '''

from collections import namedtuple
import sympy
from scipy import stats
from scipy import interpolate
import numpy as np
from contextlib import suppress
from dateutil.parser import parse
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from . import report
from . import output
from . import out_uncert
from . import plotting
from .ttable import t_factor


class CurveFitOutput(output.Output):
    ''' Class to hold output of all calculations of a curve-fit calculation '''
    def __init__(self, baseoutputs=None, inputfunc=None, xdates=False):
        self._baseoutputs = baseoutputs
        self.name = inputfunc.name
        self.desc = inputfunc.desc
        self.pnames = self._baseoutputs[0].pnames
        self.inputfunc = inputfunc
        self.fitname = inputfunc.fitname
        self.xdates = xdates
        self.predmode = 'Syx'

        for b in self._baseoutputs:
            setattr(self, b._method, b)

    def get_output(self, method=None):
        ''' Get an output object by calculation method '''
        if method is not None:
            return getattr(self, method)
        else:
            return self

    def set_predmode(self, mode='Syx'):
        ''' Set prediction band mode.

            Parameters
            ----------
            mode: string
                Prediction band mode. One of 'Syx', 'sigy', or 'sigylast'.
        '''
        self.predmode = mode
        for b in self._baseoutputs:
            b.predmode = mode

    def report(self, **kwargs):
        ''' Generate report of mean/uncertainty values.

            Keyword Arguments
            -----------------
            See report.Report
        '''
        # rows are GUM, MC, LSQ, etc. Columns are means, uncertainties. Checks if baseoutput is CurveFit, and if so uses that format.
        hdr = ['Method']
        if self.pnames is None:
            hdr.extend(['Mean', 'Standard\nUncertainty'])
        else:
            for p in self.pnames:
                hdr.extend(['{}'.format(p), 'u({})'.format(p)])
        rows = []
        for out in self._baseoutputs:
            row = [out.method]
            for c, s in zip(out.coeffs, out.sigmas):
                row.extend([report.Number(c, matchto=s, fmin=0), report.Number(s)])
            rows.append(row)

        r = report.Report(**kwargs)
        r.sympy(self._baseoutputs[0].expr(subs=False), end='\n\n')
        r.table(rows, hdr)
        return r

    def report_summary(self, k=2, conf=None, **kwargs):
        ''' Report a summary of curve fit, including table of parameters and plot of fit line for each method '''
        r = self.report(**kwargs)
        with mpl.style.context(plotting.mplcontext):
            plt.ioff()
            fig = plt.figure()
            axes = plotting.axes_grid(len(self._baseoutputs), fig, maxcols=2)
            for ax, out in zip(axes, self._baseoutputs):
                out.plot_summary(ax=ax, k=k, conf=conf, **kwargs)
                if len(self._baseoutputs) > 1:
                    ax.set_title(out.method)
            fig.tight_layout()
        r.plot(fig)
        return r

    def report_fit(self, **kwargs):
        ''' Report goodness-of-fit values r-squared and Syx. Note r is not a good predictor
            of fit for nonlinear models, but we report it anyway.
        '''
        rows = []
        for out in self._baseoutputs:
            rows.append([out.method, report.Number(out.properties['r'], fmt='decimal'), report.Number(out.properties['r']**2, fmt='decimal'),
                         report.Number(out.properties['Syx']), report.Number(out.properties['F'], fmt='auto')])
        r = report.Report(**kwargs)
        r.table(rows, ['Method', 'r', 'r-squared', 'Standard Error (Syx)', 'F-value'])
        return r

    def report_all(self, xval=None, interval=None, k=2, conf=None, **kwargs):
        ''' Report all info on curve fit, including summary, residual plots, and correlations '''
        r = report.Report(**kwargs)
        r.hdr('Curve Fit Results', level=2)
        if interval is not None:
            r.append(self.report(**kwargs))
            r.txt('\n\n')
            r.append(self.report_fit(**kwargs))
            r.txt('\n\n')
            for out in self._baseoutputs:
                if len(self._baseoutputs) > 1:
                    r.hdr('Method: {}'.format(out.method))
                r.append(out.report_interval_uncert(t1=interval[0], t2=interval[1], k=k, conf=conf, **kwargs))  # Includes plot
                r.txt('\n\n')
                r.append(out.report_interval_uncert_eqns(**kwargs))
                r.txt('\n\n')

        elif xval is not None:
            r.append(self.report(**kwargs))  # Report, but don't plot here
            r.append(self.report_fit(**kwargs))
            r.txt('\n\n')
            for out in self._baseoutputs:
                if len(self._baseoutputs) > 1:
                    r.hdr('Method: {}'.format(out.method), level=2)
                r.append(out.report_confpred_xval(xval, plot=True, k=k, conf=conf, **kwargs))

        else:
            r.append(self.report_summary(k=k, conf=conf, **kwargs))
            r.append(self.report_fit(**kwargs))
            r.txt('\n\n')
            for out in self._baseoutputs:
                if len(self._baseoutputs) > 1:
                    r.hdr('Method: {}'.format(out.method), level=2)
                r.append(out.report_confpred(**kwargs))

        r.div()
        r.hdr('Residuals', level=3)
        r.append(self.report_residuals(k=k, conf=conf, **kwargs))
        return r

    def report_residuals(self, **kwargs):
        ''' Report of residual values for each method '''
        r = report.Report(**kwargs)
        for out in self._baseoutputs:
            if len(self._baseoutputs) > 1:
                r.hdr('Method: {}'.format(out.method), level=3)
            r.append(out.report_residuals(**kwargs))
        return r

    def report_correlation(self, **kwargs):
        ''' Report table and plot of correlations between fit parameters for each method '''
        r = report.Report(**kwargs)
        with mpl.style.context(output.mplcontext):
            plt.ioff()
            for out in self._baseoutputs:
                if len(self._baseoutputs) > 1:
                    r.hdr('Method: {}'.format(out.method), level=3)
                r.append(out.report_correlation(**kwargs))
                fig = plt.figure()
                out.plot_outputscatter(fig=fig)
                r.plot(fig)
        return r

    def get_dists(self):
        ''' Get distributions in this output. If name is none, return a list of
            available distribution names.
        '''
        dists = {}
        for method in self._baseoutputs:
            methodname = method._method.upper()
            baseout = self.get_output(method=method._method)
            for param in self.pnames:
                pidx = self.pnames.index(param)
                if 'mc' in method._method:
                    dists[f'{param} ({methodname})'] = {'samples': baseout.properties['samples'][:, pidx]}
                else:
                    dists[f'{param} ({methodname})'] = {'mean': baseout.coeffs[pidx], 'std': baseout.sigmas[pidx], 'df': baseout.degf}

        for method in self._baseoutputs:
            methodname = method._method.upper()
            baseout = self.get_output(method=method._method)
            dists[f'Confidence ({methodname})'] = {'xdates': self.xdates,
                                                   'function': lambda x, baseout=baseout: {'mean': baseout.y(x), 'std': baseout.u_conf(x), 'df': baseout.degf}}
            dists[f'Prediction ({methodname})'] = {'xdates': self.xdates,
                                                   'function': lambda x, baseout=baseout: {'mean': baseout.y(x), 'std': baseout.u_pred(x), 'df': baseout.degf}}

        return dists


class CurveFitMethodOutput(output.Output):
    ''' Output of one calculation method (LSQ, GUM, etc.)

        Parameters
        ----------
        method: string
            Calculation method (lsq, mc, mcmc, gum)

        Keyword Arguments
        -----------------
        coeffs: array
            Coeffecients for each curve fit parameter
        sigmas: array
            Uncertainties in each fit parameter
        pnames: array
            String names for each parameter
        desc: string
            Description
        func: callable
            Fitting function for curve fit
        degf: float
            Degrees of freedom for the fit
        samples: array
            Sampled values (for Monte Carlo fits)
        xdates: boolean
            Interpret x values as date in ordinal format
    '''
    def __init__(self, method, **kwargs):
        super().__init__()
        self._method = method
        self.coeffs = kwargs.get('coeffs')
        self.sigmas = kwargs.get('sigmas')
        self.pnames = kwargs.get('pnames')
        self.paramvalues = dict(zip(self.pnames, self.coeffs))
        self.desc = kwargs.get('desc', None)
        self.name = kwargs.get('name', None)
        self.xdates = kwargs.get('xdates', False)
        self.fitfunc = kwargs.get('func')
        self.degf = kwargs.get('degf')
        self.samples = kwargs.get('samples')
        self.predmode = 'Syx'
        self.properties = kwargs
        self._baseparams = []

        for i in range(len(self.pnames)):
            paramargs = {'mean': self.coeffs[i], 'uncert': self.sigmas[i], 'name': self.pnames[i]}
            if 'samples' in kwargs:
                paramargs['samples'] = kwargs['samples'][:, i]
            paramout = out_uncert.create_output(method=method, **paramargs)
            self._baseparams.append(paramout)

    @property
    def method(self):
        ''' Get method string '''
        return {'gum': 'GUM Approximation',
                'mc': 'Monte Carlo',
                'lsq': 'Least Squares',
                'mcmc': 'Markov-Chain Monte Carlo'}[self._method]

    def report(self, **kwargs):
        ''' Generate/return default report (same as repr)

            Keyword Arguments
            -----------------
            See report.Report

            Returns
            -------
            report.Report
        '''
        hdr = ['Parameter'] + self.pnames
        rowm = ['Mean'] + [report.Number(m, matchto=u) for m, u in zip(self.coeffs, self.sigmas)]
        rowu = ['Standard Uncertainty'] + [report.Number(u) for u in self.sigmas]
        r = report.Report(**kwargs)
        r.table([rowm, rowu], hdr=hdr)
        return r

    def _x(self, dates=False):
        ''' Return x value as float or date '''
        xdata, _, _, _ = self.properties.get('data')
        if dates:
            return mdates.num2date(xdata)
        else:
            return xdata

    def _y(self):
        ''' Return y value as float '''
        _, ydata, _, _ = self.properties.get('data')
        return ydata

    def _full_xrange(self, x1=None, x2=None, num=200):
        ''' Get a linspace covering the full range of the curve fit, plus other
            points of interest.
        '''
        mn = self._x().min()
        mx = self._x().max()
        if x1 is not None:
            mn = min(mn, x1)
            mx = max(mx, x1)
        if x2 is not None:
            mn = min(mn, x2)
            mx = max(mx, x2)
        xx = np.linspace(mn, mx, num=num)
        return xx

    def y(self, x):
        ''' Predict y value for given x value '''
        return self.fitfunc(x, *self.coeffs)

    def u_pred(self, x, k=1, conf=None, **kwargs):
        ''' Prediction band uncertainty

            Parameters
            ---------
            x: float or array
                X-value(s) at which to calculate
            k: float
                k-value to apply
            conf: float
                Level of confidence (0 to 1). Overrides value of k parameter

            Keyword Arguments
            -----------------
            mode: string
                Prediction band mode. One of 'Syx', 'sigy', or 'sigylast'.
        '''
        mode = kwargs.get('mode', self.predmode)
        if mode is None:
            mode = self.predmode
        if conf is not None:
            k = t_factor(conf, self.degf)
        return self.properties.get('u_pred')(x, mode=mode) * k

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
        return self.properties.get('u_conf')(x) * k

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

    def expanded(self, cov=.95, **kwargs):
        ''' Get expanded uncertainty for each parameter '''
        mins, maxs, ks = [], [], []
        for b in self._baseparams:
            mn, mx, k = b.expanded(cov=cov, **kwargs)
            mins.append(mn)
            maxs.append(mx)
            ks.append(k)
        Expanded = namedtuple('Expanded', ['minimum', 'maximum', 'k'])
        return Expanded(mins, maxs, ks)

    def get_pdf(self, x, idx=0, **kwargs):
        ''' Get probability density function for this output.

            Parameters
            ----------
            x: float or array
                X-values at which to compute PDF
            idx: int
                Index into means/uncerts if multiple
        '''
        return self._baseparams[idx].get_pdf(x)

    def plot_pdf(self, ax=None, **kwargs):
        ''' Plot probability density function

            Parameters
            ----------
            ax: matplotlib axis
                Axis to plot on

            Keyword Arguments
            -----------------
            fig: matplotlib figure
                Matplotlib figure to plot on. New axis will be created
            pidx: int
                Index of output parameter
            stddevs: float
                Number of standard deviations to include in x range
            **kwargs: dict
                Remaining keyword arguments passed to plot method.

            Returns
            -------
            ax: matplotlib axis
        '''
        pidx = kwargs.pop('pidx', list(range(len(self.coeffs))))
        pidx = np.atleast_1d(pidx)
        kwargs.setdefault('label', self.method)
        fig = kwargs.pop('fig', None)

        if ax is None and fig is None:
            fig = plt.gcf()
            if len(fig.axes) < len(pidx):
                ax = [fig.add_subplot(1, len(pidx), i+1) for i in range(len(pidx))]
            else:
                ax = fig.axes
        ax = np.atleast_1d(ax)
        assert len(ax) >= len(pidx)

        for p, axis in zip(pidx, ax):
            self._baseparams[p].plot_pdf(ax=axis, **kwargs)

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
        expr = self.properties['expr']
        if subs:
            expr = expr.subs(dict(zip(self.pnames, self.coeffs))).evalf(n=n)
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
        if self.properties['fitname'] == 'line':
            expr = sympy.sympify('S_yx * sqrt(1 / n + (x-xbar)**2 * (sigma_b/S_yx)^2)')
            if subs:
                expr = expr.subs({'S_yx': self.properties['Syx'],
                                  'n': len(self.properties['data'][0]),
                                  'xbar': self.properties['data'][0].mean(),
                                  'sigma_b': self.sigmas[0]}).evalf(n=n)

            if full:
                expr = sympy.Eq(sympy.Symbol('u_{conf}'), expr)
        else:
            raise NotImplementedError('uconf expression only implemented for line fits.')
        return expr

    def expr_upred(self, subs=True, n=4, full=True):
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

            Notes
            -----
            This expression will only match u_conf exactly when using residuals
            Syx to determine y-uncertainty.
        '''
        if self.properties['fitname'] == 'line':
            expr = sympy.sympify('S_yx * sqrt(1 + 1 / n + (x-xbar)**2 * (sigma_b/S_yx)^2)')
            if subs:
                expr = expr.subs({'S_yx': self.properties['Syx'],
                                  'n': len(self.properties['data'][0]),
                                  'xbar': self.properties['data'][0].mean(),
                                  'sigma_b': self.sigmas[0]}).evalf(n=n)
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
        b = self.properties['expr']  # Curve
        bbar_expr = oneovert1t2 * sympy.Integral(sympy.Symbol('b'), (x, t1, t2))
        bbar_int = oneovert1t2 * sympy.Integral(b, (x, t1, t2))
        bbar = bbar_int.doit()
        ubbar_expr = oneovert1t2 * sympy.Integral((sympy.Symbol('b') - sympy.Symbol('bbar'))**2, (x, t1, t2))
        ubbar_int = oneovert1t2 * sympy.Integral((b-bbar)**2, (x, t1, t2))
        ubbar = ubbar_int.doit()
        uconf = sympy.Symbol('u_conf')
        ub_expr = oneovert1t2 * sympy.Integral(uconf**2, (x, t1, t2))
        if self.properties['fitname'] == 'line':
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
                'ub': ub,
                }

    def uncert_interval(self, t1, t2):
        ''' Calculate a single value and standard uncertainty that applies to the entire
            interval from t1 to t2. See GUM F.2.4.5.
        '''
        uy = self.properties['Syx']
        # sigy can be scalar or array function of x. Interpolate/average (linearly) over interval if necessary
        if not np.isscalar(uy):
            uy1, uy2 = interpolate.interp1d(self._x(), uy, fill_value='extrapolate')([t1, t2])
            uy = np.sqrt((uy1**2 + uy2**2) / 2)  # Average over interval

        subs = {'t1': t1,
                't2': t2,
                'S_yx': uy,
                'n': len(self._x()),
                'xbar': self._x().mean(),
                'sigma_b': self.sigmas[0]}
        subs.update(dict(zip(self.pnames, self.coeffs)))

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

    def report_correlation(self, **kwargs):
        ''' Report table of correlation coefficients '''
        hdr = ['Parameter'] + self.pnames
        rows = []
        for idx, row in enumerate(self.properties['cor']):
            rows.append([self.pnames[idx]] + [report.Number(v) for v in row])
        r = report.Report(**kwargs)
        r.table(rows, hdr=hdr)
        return r

    def report_residuals(self, **kwargs):
        ''' Report a plot of residuals, histogram, and normal-probability '''
        with mpl.style.context(plotting.mplcontext):
            plt.ioff()
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
        r.append(self.report_residtable(**kwargs))
        return r

    def report_residtable(self, k=2, conf=None, **kwargs):
        ''' Report table of measured values and their residuals '''
        r = report.Report(**kwargs)
        kstr = '(k={})'.format(k) if conf is None else '({:.4g}%)'.format(conf*100)
        hdr = ['Measured x', 'Measured y', 'Predicted y', 'Residual', 'Confidence Band {}'.format(kstr), 'Prediction Band {}'.format(kstr)]
        x, y, _, _ = self.properties.get('data')
        if self.xdates:
            xstring = mdates.num2date(x)
            xstring = [k.strftime('%d-%b-%Y') for k in xstring]
        else:
            xstring = [str(k) for k in x]
        resid = self.properties.get('resids')
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

    def report_confpred(self, **kwargs):
        ''' Report equations for confidence and prediction intervals, only for line fit. '''
        r = report.Report(**kwargs)
        if self.properties['fitname'] == 'line':
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
            if self.xdates:
                with suppress(AttributeError, ValueError):
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
            rows.append([str(xval[i]), yi, '±{} [{}, {}]'.format(report.Number(uconf[i], fmin=2), yconfminus, yconfplus),
                         '±{} [{}, {}]'.format(report.Number(upred[i], fmin=2), ypredminus, ypredplus)])

        r = report.Report(**kwargs)
        if plot:
            with mpl.style.context(plotting.mplcontext):
                xx = self._full_xrange(np.nanmin(x), np.nanmax(x))
                fig, ax = plt.subplots()
                self.plot_points(ax=ax, marker='o', ls='')
                self.plot_fit(ax=ax, x=xx, ls='-', label='Fit')
                self.plot_pred(ax=ax, x=xx, k=k, conf=conf, ls='--', mode=mode)
                self.plot_pred_value(ax=ax, xval=x, k=k, conf=conf, mode=mode)
                ax.set_xlabel(self.properties['axnames'][0])
                ax.set_ylabel(self.properties['axnames'][1])
                ax.legend(loc='best')
                r.plot(fig)
        r.table(rows, hdr=hdr)
        r.append(self.report_confpred(**kwargs))
        return r

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
        if ax is None:
            fig = plt.gcf()
            ax = fig.gca()

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
            if self.xdates:
                xx = mdates.num2date(xx)
            yy = np.append(np.append(yfit + u_pred, [np.nan]), yfit - u_pred)
            ax.plot(xx, yy, **kwargs)
        else:
            ax.plot(self._x(self.xdates), u_pred, **kwargs)
        return ax

    def plot_pred_value(self, xval, k=2, conf=None, ax=None, mode=None):
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
        if ax is None:
            fig = plt.gcf()
            ax = fig.gca()

        if self.xdates and isinstance(xval, str):
            xfloat = np.array([parse(x).tooridnal() for x in xval])
            xplot = mdates.num2date(xfloat)
        else:
            xfloat = xval
            xplot = xval

        y = self.y(xfloat)
        upred = self.u_pred(xfloat, k=k, conf=conf, mode=mode)
        ax.errorbar(xplot, y, yerr=upred, marker='s', markersize=8, capsize=4, label='Predicted Value', ls='', color='C4')
        return ax

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
        if ax is None:
            fig = plt.gcf()
            ax = fig.gca()

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
            if self.xdates:
                xx = mdates.num2date(xx)
            yy = np.append(np.append(yfit + u_conf, [np.nan]), yfit - u_conf)
            ax.plot(xx, yy, **kwargs)
        else:
            ax.plot(self._x(self.xdates), u_conf, **kwargs)
        return ax

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
        if ax is None:
            fig = plt.gcf()
            ax = fig.gca()

        if x is None:
            x = self._full_xrange()

        yfit = self.y(x)

        if self.xdates:
            x = mdates.num2date(x)

        ax.plot(x, yfit, **kwargs)
        return ax

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
        if ax is None:
            fig = plt.gcf()
            ax = fig.gca()

        xdata, ydata, ux, uy = self.properties.get('data')
        if self.xdates:
            xdata = mdates.num2date(xdata)

        if ebar:
            ax.errorbar(xdata, ydata, yerr=uy, xerr=ux, **kwargs)
        else:
            ax.plot(xdata, ydata, **kwargs)
        return ax

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
        if ax is None:
            fig = plt.gcf()
            ax = fig.gca()

        self.plot_points(ax=ax, marker='o', ls='')
        self.plot_fit(ax=ax, ls='-', label='Fit')
        self.plot_conf(ax=ax, k=k, conf=conf, ls=':')
        self.plot_pred(ax=ax, k=k, conf=conf, ls='--', mode=kwargs.get('mode', self.predmode))
        ax.set_xlabel(self.properties['axnames'][0])
        ax.set_ylabel(self.properties['axnames'][1])
        ax.legend(loc='best')
        return ax

    def plot_interval_uncert(self, t1, t2, ax=None, k=2, conf=None, mode=None, **kwargs):
        ''' Plot uncertainty valid for given interval (GUM F.2.4.5) '''
        if ax is None:
            fig = plt.gcf()
            ax = fig.gca()

        if self.xdates and isinstance(t1, str):
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
        return ax

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
        if self._method not in ['mc', 'mcmc']:
            raise ValueError('plot_xhists only supported for Monte Carlo curve fit types')

        if fig is None:
            fig = plt.gcf()

        inpts = kwargs.pop('inpts', list(range(len(self._baseparams))))
        fig.clf()
        fig.subplots_adjust(**out_uncert.dfltsubplots)
        axs = plotting.axes_grid(len(inpts), fig)

        for ax, inptnum in zip(axs, inpts):
            samples = self._baseparams[inptnum].samples
            ax.hist(samples, **kwargs)
            ax.ticklabel_format(style='sci', axis='x', scilimits=(-4, 4), useOffset=False)
            ax.yaxis.set_visible(False)
            ax.set_xlabel('$' + self.pnames[inptnum] + '$')
        fig.tight_layout()
        return fig

    def report_interval_uncert(self, t1, t2, k=2, conf=None, plot=True, **kwargs):
        ''' Report the value and uncertainty that applies to the
            entire interval from t1 to t2.
        '''
        confstr = ''
        if conf is not None:
            k = t_factor(conf, self.degf)
            confstr = ', {:.4g}%'.format(conf*100)

        if self.xdates and isinstance(t1, str):
            t1 = mdates.date2num(parse(t1))
            t2 = mdates.date2num(parse(t2))

        if self.xdates:
            t1str = mdates.num2date(t1).strftime('%d-%b-%Y')
            t2str = mdates.num2date(t2).strftime('%d-%b-%Y')
        else:
            t1str, t2str = report.Number.number_array([t1, t2], thresh=8)

        value, uncert = self.uncert_interval(t1, t2)
        r = report.Report(**kwargs)
        r.txt('For the interval {} to {}:\n\n'.format(t1str, t2str))
        r.add('Value = ', report.Number(value, fmin=2), ' ± ', report.Number(uncert*k, fmin=2), '(k={:.3g}{})'.format(k, confstr))
        if plot:
            with mpl.style.context(output.mplcontext):
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

    def plot_outputscatter(self, params=None, fig=None, MCcontour=False, **kwargs):
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
            params = list(range(len(self.pnames)))

        num = len(params)
        if num < 2: return fig
        for row in range(num):
            for col in range(num):
                if col <= row: continue

                idx1, idx2 = params[row], params[col]
                ax = fig.add_subplot(num-1, num-1, row*(num-1)+col)
                if self._method == 'mc':
                    # Default to no line, points only
                    if not MCcontour:
                        if 'ls' not in kwargs and 'linestyle' not in kwargs:
                            kwargs['ls'] = ''
                        if 'marker' not in kwargs:
                            kwargs['marker'] = '.'
                        p, = ax.plot(self.properties['samples'][:, idx1], self.properties['samples'][:, idx2], **kwargs)
                    else:
                        bins = max(3, kwargs.pop('bins', 40))
                        counts, ybins, xbins = np.histogram2d(self.properties['samples'][:, idx2], self.properties['samples'][:, idx1], bins=bins)
                        ax.contour(counts, 10, extent=[xbins.min(), xbins.max(), ybins.min(), ybins.max()], **kwargs)

                else:  # else: generate 2D contour using mean/uncert and plot
                    cov = np.array([[self.properties['cov'][idx1, idx1], self.properties['cov'][idx1, idx2]],
                                    [self.properties['cov'][idx2, idx1], self.properties['cov'][idx2, idx2]]])
                    try:
                        rv = stats.multivariate_normal(np.array([self.coeffs[idx1], self.coeffs[idx2]]), cov=cov)
                    except (np.linalg.linalg.LinAlgError, ValueError):  # Singular matrix
                        continue
                    x, y = np.meshgrid(np.linspace(self.coeffs[idx1]-3*self.sigmas[idx1], self.coeffs[idx1]+3*self.sigmas[idx1]),
                                       np.linspace(self.coeffs[idx2]-3*self.sigmas[idx2], self.coeffs[idx2]+3*self.sigmas[idx2]))
                    ax.contour(rv.pdf(np.dstack((x, y))), extent=[x.min(), x.max(), y.min(), y.max()], **kwargs)
                ax.set_xlabel(self.pnames[idx1])
                ax.set_ylabel(self.pnames[idx2])
        return fig

    # Functions for diagnosing goodness of fit
    #-----------------------------------------
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
        if ax is None:
            fig = plt.gcf()
            ax = fig.gca()

        resid = self.properties['resids']
        sy = np.sqrt(sum(resid**2)/(len(resid)-2))
        resid = resid / sy  # Normalized

        if hist:
            with suppress(ValueError):  # Can raise if resids has only one value or is empty
                ax.hist(self.properties['resids'], **kwargs)
        else:
            # Default to no line, points only
            if 'ls' not in kwargs and 'linestyle' not in kwargs:
                kwargs['ls'] = ''
            if 'marker' not in kwargs:
                kwargs['marker'] = 'o'
            x = self.properties['data'][0]
            if self.xdates:
                x = mdates.num2date(x)
            ax.plot(x, self.properties['resids'], **kwargs)
        return ax

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
        resid = self.properties['resids'].copy()
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
        if ax is None:
            fig = plt.gcf()
            ax = fig.gca()

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

    def plot_samples(self, fig=None, **kwargs):
        ''' Plot samples for each parameter (value vs. sample number) '''
        if self._method not in ['mc', 'mcmc']:
            raise ValueError('Plot Samples for Monte Carlo methods only.')

        if fig is None:
            fig = plt.gcf()

        inpts = kwargs.pop('inpts', list(range(len(self.coeffs))))
        fig.clf()
        fig.subplots_adjust(**out_uncert.dfltsubplots)
        axs = plotting.axes_grid(len(inpts), fig)

        for ax, inptnum in zip(axs, inpts):
            samples = self._baseparams[inptnum].samples.magnitude
            ax.plot(samples, **kwargs)
            ax.set_ylabel('$' + self.pnames[inptnum] + '$')
            ax.set_xlabel('Sample #')
        fig.tight_layout()
        return fig

    def report_acceptance(self, **kwargs):
        ''' Report acceptance rate (MCMC fits only) '''
        if self._method != 'mcmc':
            raise ValueError('Acceptance Report for Markov-Chain MC method only.')

        if 'acceptance' in self.properties:
            hdr = ['Parameter', 'Acceptance Rate']
            rows = []
            for p, v in zip(self.pnames, self.properties['acceptance']):
                rows.append([p, '{:.2f}%'.format(v*100)])
            r = report.Report(**kwargs)
            r.table(rows, hdr=hdr)
        return r

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
        t = abs(self.coeffs[pidx] - nominal) / self.sigmas[pidx]
        ta = stats.t.ppf(1-(1-conf)/2, df=self.degf)
        if verbose:
            print('{}: {} > {} --> {}'.format(self.pnames[pidx], t, ta, t > ta))
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
        b1 = self.coeffs[pidx]
        sb = self.sigmas[pidx]
        ta = stats.t.ppf(1-(1-conf)/2, df=self.degf)
        ok = not (b1 - ta*sb < nominal < b1 + ta*sb)
        if verbose:
            print(r'[{} < {} < {}] {} {} --> {}'.format(b1 - ta*sb, self.pnames[pidx], b1 + ta*sb, u'\u2285', nominal, ok))
        return ok
