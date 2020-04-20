'''
The classes in this module hold the results of an Uncertainty Calculator calculation.
In addition to the raw calculated values, methods are available for generating
formatted reports and plots.
'''
from contextlib import suppress
from collections import namedtuple
import sympy
from scipy import stats
import numpy as np

from . import output
from . import report
from . import ureg
from . import plotting

import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.style.use('bmh')
dfltsubplots = {'wspace': .1, 'hspace': .1, 'left': .05, 'right': .95, 'top': .95, 'bottom': .05}


def create_output(method='gum', **kwargs):
    ''' Factory function for creating a BaseOutput '''
    if method.lower() in ['mc', 'mcmc']:
        return BaseOutputMC(method, **kwargs)
    else:
        return BaseOutput(method, **kwargs)


class BaseOutput(output.Output):
    ''' Class to hold results from one calculated parameter (one mean and uncertainty value)

        Parameters
        ----------
        method: string
            Method of calculation ['gum', 'mc']

        Keyword Arguments
        -----------------
        mean: float or Pint Quantity
            Mean value
        uncert: float or Pint Quantity
            Uncertainty value
        desc: string
            Description of this output
        name: string
            Name for this output
        props: array
            Proportions from calculation
        residual: float
            Residual proportion (due to correlation)
        degf: float
            Effective Degrees of Freedom
        units: Pint Quantity
            Units of mean and uncert
        properties: dict
            Other output attributes
    '''
    def __init__(self, method, **kwargs):
        self._method = method
        self.mean = kwargs.pop('mean')
        self.uncert = kwargs.pop('uncert')
        self.units = kwargs.pop('units', ureg.dimensionless)
        self.desc = kwargs.pop('desc', None)
        self.name = kwargs.pop('name', None)
        self.props = kwargs.pop('props', None)    # Proportions
        self.residprops = kwargs.pop('residual', None)
        self.sensitivity = kwargs.pop('sensitivity', None)
        self.degf = kwargs.pop('degf', np.inf)
        self.properties = kwargs
        if not hasattr(self.mean, 'units'):
            self.mean *= self.units
            self.uncert *= self.units

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
            Arguments for Report object

            Returns
            -------
            report.Report
        '''
        hdr = ['Parameter', 'Value']
        rowm = ['Mean', report.Number(self.mean, matchto=self.uncert)]
        rowu = ['Standard Uncertainty', report.Number(self.uncert)]
        r = report.Report(**kwargs)
        r.table([rowm, rowu], hdr=hdr)
        return r

    def report_expanded(self, covlist=None, normal=False, **kwargs):
        ''' Generate formatted report of expanded uncertainties

            Parameters
            ----------
            covlist: list of float
                Coverage intervals to include in report as fraction (i.e. 0.99 for 99%)
            normal: bool
                Calculate using normal distribution regardless of degrees of freedom

            Keyword Arguments
            -----------------
            covlistgum: list of float
                GUM Coverage intervals to include in report as fraction (i.e. 0.99 for 99%)

            Other kwargs passed to report.Report

            Returns
            -------
            report.Report
        '''
        covlist = kwargs.get('covlistgum', covlist)
        if covlist is None:
            covlist = [.99, .95, .90, .68]

        rows = []
        hdr = ['Interval', 'Min', 'Max', 'k', 'Deg. Freedom', 'Expanded Uncertainty']
        for cov in covlist:
            uncert, k = self.expanded(cov, normal=normal)
            row = ['{:.2f}%'.format(cov*100) if not isinstance(cov, str) else cov]
            row.append(report.Number(self.mean-uncert, matchto=uncert, **kwargs))
            row.append(report.Number(self.mean+uncert, matchto=uncert, **kwargs))
            row.append(format(k, '.3f'))
            row.append(format(self.degf, '.2f'))
            row.append(report.Number(uncert, **kwargs))
            rows.append(row)
        r = report.Report(**kwargs)
        r.table(rows, hdr)
        return r

    def report_warns(self, **kwargs):
        ''' Report of warnings raised during calculation

            Returns
            -------
            report.Report
        '''
        r = report.Report(**kwargs)
        if 'warns' in self.properties and self.properties.get('warns') is not None:
            for w in self.properties['warns']:
                r.txt('- ' + w + '\n')
        return r

    def report_derivation(self, solve=True, **kwargs):
        ''' Generate report of derivation of GUM method.

            Parameters
            ----------
            solve: bool
                Calculate values of each function.

            Returns
            -------
            report.Report
        '''
        N = kwargs.get('n', report.default_sigfigs) + 2  # Add a couple sigfigs since these are intermediate

        def combine(exprs, val):
            ''' Helper function for combining sympy expressions using multiple sympy.Eq()'s '''
            units = None
            with suppress(AttributeError):
                val, units = val.magnitude, val.units
            symexp = exprs[0]
            i = 1
            while i < len(exprs):
                symexp = sympy.Eq(symexp, exprs[i], evaluate=False)
                i += 1

            if solve:
                symexp = sympy.Eq(symexp, sympy.N(val, N), evaluate=False)
                return report.Math.from_sympy(symexp, units)
            else:
                return report.Math.from_sympy(symexp)

        rpt = report.Report(**kwargs)
        if self._method != 'gum':
            rpt.text('No derivation for {} method'.format(self.method))
            return rpt

        if 'symbolic' not in self.properties:
            rpt.txt('No symbolic solution computed for function.')
            return rpt

        symout = self.properties['symbolic']
        rpt.hdr('Model Equation:', level=3)
        rpt.sympy(symout['function'], end='\n\n')

        rpt.hdr('Input Definitions:', level=3)
        rows = []
        for var, val, uncert in zip(symout['var_symbols'], symout['var_means'], symout['var_uncerts']):
            vareqn = combine((var,), val)
            unceqn = combine((sympy.Symbol('u_{{{}}}'.format(str(var))),), uncert)
            rows.append([vareqn, unceqn])
        rpt.table(rows, hdr=['Variable', 'Std. Uncertainty'])

        if 'covsymbols' in symout:
            rpt.txt('Correlation coefficients:\n\n')
            for var, cov in zip(symout['covsymbols'], symout['covvals']):
                eq = var if not solve else sympy.Eq(var, cov).n(3)
                rpt.sympy(eq, end='\n\n')

        rpt.hdr('Sensitivity Coefficients:', level=3)
        for c, p1, p2, p3 in zip(symout['var_symbols'], symout['partials_raw'], symout['partials'], symout['partials_solved']):
            rpt.add(combine((sympy.Symbol('c_{}'.format(c)), p1, p2), p3), '\n\n')

        rpt.hdr('Combined uncertainty:', level=3)
        uname = sympy.Symbol('u_'+str(self.name) if self.name != '' and self.name is not None else 'u_c')
        uformula_cx = symout['unc_formula_sens']
        uformula = symout['unc_formula']
        uformula = sympy.Eq(uname, uformula, evaluate=False)
        uformula_cx = sympy.Eq(uname, uformula_cx, evaluate=False)
        rpt.sympy(uformula_cx, end='\n\n')
        rpt.sympy(uformula, end='\n\n')

        uformula = symout['uncertainty']  # Simplified formula
        rpt.add(combine((uname, uformula), symout['unc_val']), '\n\n')

        rpt.hdr('Effective degrees of freedom:', level=3)
        rpt.add(combine((sympy.Symbol('nu_eff'), symout['degf']), symout['degf_val']), '\n\n')
        return rpt

    def get_pdf(self, x, **kwargs):
        ''' Get probability density function for this output.

            Parameters
            ----------
            x: float or array
                X-values at which to compute PDF
        '''
        return stats.norm.pdf(x, loc=self.mean.magnitude, scale=self.uncert.magnitude)

    def plot_pdf(self, plot=None, **kwargs):
        ''' Plot probability density function

            Parameters
            ----------
            plot: object
                Matplotlib Figure or Axis to plot on

            Keyword Arguments
            -----------------
            fig: matplotlib figure
                Matplotlib figure to plot on. New axis will be created
            stddevs: float
                Number of standard deviations to include in x range
            **kwargs: dict
                Remaining keyword arguments passed to plot method.

            Returns
            -------
            ax: matplotlib axis
        '''
        stdevs = kwargs.pop('stddevs', 4)
        intervals = kwargs.pop('intervalsGUM', [])
        intervaltype = kwargs.pop('intTypeGUMt', True)  # Use student-t (vs. k-value) to compute interval
        kwargs.pop('intTypeMC', None)  # not used in gum pdf
        kwargs.pop('intervals', None)  # not used by gum (only MC)
        covcolors = kwargs.pop('covcolors', ['C4', 'C5', 'C6', 'C7', 'C8', 'C9'])
        kwargs.setdefault('label', self.method)
        kwargs.setdefault('color', 'C1')

        fig, ax = plotting.initplot(plot)
        mean = self.mean.magnitude
        uncert = self.uncert.magnitude
        units = self.units
        x = np.linspace(mean - stdevs*uncert, mean+stdevs*uncert, num=100)
        ax.plot(x, self.get_pdf(x), **kwargs)
        if self.name is not None:
            ax.set_xlabel(self.name)
        if units:
            unitstr = report.Unit(units).latex(bracket=True)
            ax.set_xlabel(ax.get_xlabel() + unitstr)

        if intervals:
            ilines, ilabels = [], []
            for covidx, cov in enumerate(intervals):
                u, k = self.expanded(cov, normal=not intervaltype)
                iline = ax.axvline(mean + u.magnitude, color=covcolors[covidx], ls='--')
                ax.axvline(mean - u.magnitude, color=covcolors[covidx], ls='--')
                ilines.append(iline)
                ilabels.append(cov)
            intervallegend = ax.legend(ilines, ilabels, fontsize=10, loc='center right', title='Intervals')
            ax.add_artist(intervallegend)

    def expanded(self, cov=0.95, **kwargs):
        ''' Calculate expanded uncertainty with coverage interval based on degf.

            Parameters
            ----------
            cov: float or string
                Coverage interval value, from 0-1 or string value as k-value or percent

            Keyword Arguments
            -----------------
            normal: bool
                Calculate using normal distribution, ignoring degrees of freedom

            Returns
            -------
            uncertainty: float
                Expanded uncertainty
            k: float
                k-value associated with the expanded uncertainty
        '''
        if isinstance(cov, str):
            if '%' in cov:
                cov = cov.strip('%')
                cov = float(cov) / 100
            elif 'k' in cov:
                _, cov = cov.split('=')
                cov = float(cov.strip())
                return self.uncert * cov, cov   # Can't numpy multiply different units

        if kwargs.get('normal', False):
            k = stats.norm.ppf(1-(1-cov)/2)
        else:
            d = max(1, min(self.degf, 1E6))   # Inf will return garbage. Use big number.
            k = stats.t.ppf(1-(1-cov)/2, d)   # Use half of interval for scipy to get both tails
        Expanded = namedtuple('Expanded', ['uncertainty', 'k'])
        return Expanded(k*self.uncert, k)


class BaseOutputMC(BaseOutput):
    ''' Base Output for Monte Carlo method. Differences include reporting symmetric coverage intervals and
        generating pdf from histogram of samples.
    '''
    def __init__(self, method, **kwargs):
        super().__init__(method=method, **kwargs)
        self.samples = kwargs.get('samples')
        if not hasattr(self.samples, 'units'):
            self.samples *= self.units

    def plot_converge(self, fig=None, ax1=None, ax2=None, **kwargs):
        ''' Plot Monte-Carlo convergence. Using the same sampled values, the mean
            and standard uncertainty will be recomputed at fixed intervals
            throughout the sample array.

            Parameters
            ----------
            fig: matplotlib figure instance, optional
                If omitted, a new figure will be created. Existing figure will be cleared.
            ax1, ax2: matplotlib axes
                To plot on an existing figure, use these two axes. fig parameter will be
                ignored.

            Keyword Arguments
            -----------------
            div: int
                Number of points to plot
            relative: bool
                Show as relative to final value
            meancolor: string
                Color for mean values
            unccolor: string
                Color for uncertainty values
            **kwargs: dict
                Remaining keyword arguments passed to plot method.
        '''
        div = kwargs.pop('div', 25)
        relative = kwargs.pop('relative', False)

        if ax1 is not None:
            fig = ax1.figure
            ax = [ax1, ax2]
        else:
            if fig is None:
                fig = plt.gcf()
            fig.clf()
            fig.subplots_adjust(**dfltsubplots)
            ax = [fig.add_subplot(1, 2, 1), fig.add_subplot(1, 2, 2)]

        meancolor = kwargs.pop('meancolor', 'C0')
        unccolor = kwargs.pop('unccolor', 'C1')
        kwargs.setdefault('marker', 'o')

        title = self.properties['latex'] if 'latex' in self.properties else self.names
        title = '$' + title + '$'
        Y = self.samples.magnitude
        unitstr = report.Unit(self.units).latex(bracket=True)
        step = len(Y)//div
        line = np.empty(div)
        steps = np.empty(div)
        for i in range(div):
            line[i] = Y[:step*(i+1)].mean()
            steps[i] = (i+1)*step
        if relative:
            line = line / line[-1]
        kwargs['color'] = meancolor
        ax[0].plot(steps, line, **kwargs)

        ax[0].set_title(title)
        if relative:
            ax[0].set_ylabel('Value (Relative to final)')
        else:
            ax[0].set_ylabel('Value' + unitstr)
        ax[0].set_xlabel('Samples')

        line = np.empty(div)
        steps = np.empty(div)
        for i in range(div):
            line[i] = Y[:step*(i+1)].std()
            steps[i] = (i+1)*step
        if relative:
            line = line / line[-1]
        kwargs['color'] = unccolor
        ax[1].plot(steps, line, **kwargs)
        ax[1].set_title(title)
        ax[1].set_xlabel('Samples')
        if relative:
            ax[1].set_ylabel('Uncertainty (Relative to final)')
        else:
            ax[1].set_ylabel('Uncertainty' + unitstr)
        fig.tight_layout()

    def get_pdf(self, x, **kwargs):
        ''' Get probability density function for this output. Based on histogram
            of output samples.

            Parameters
            ----------
            x: float or array
                X-values at which to compute PDF
            bins: int, default=100
                Number of bins for PDF histogram
        '''
        bins = max(3, kwargs.get('bins', 200))
        yy, xx = np.histogram(self.samples.magnitude, bins=bins, range=(x.min(), x.max()), density=True)
        xx = xx[:-1] + (xx[1]-xx[0])/2
        yy = np.interp(x, xx, yy)
        return yy

    def plot_pdf(self, plot=None, **kwargs):
        ''' Plot probability density function

            Parameters
            ----------
            plot: matplotlib axis or figure
                Axis to plot on

            Keyword Arguments
            -----------------
            bins: int
                Number of bins for histogram
            stddevs: float
                Number of standard deviations to include in x range
            intervals: list of float
                Coverage intervals to plot as vertical lines. Each float must be 0-1.
            hist: bool
                Show as histogram (True) or pdf (False)
            covcolors: list of string
                List of color strings for coverage interval lines

            Remaining kwargs passed to hist() or plot() method.

            Returns
            -------
            ax: matplotlib axis
        '''
        hist = kwargs.pop('hist', True)
        stdevs = kwargs.pop('stddevs', 4)
        kwargs.setdefault('label', self.method)
        intervals = kwargs.pop('intervals', [])
        intervalshortest = kwargs.pop('intTypeMC', 'Symmetric') == 'Shortest'
        covcolors = kwargs.pop('covcolors', ['C4', 'C5', 'C6', 'C7', 'C8', 'C9'])
        kwargs.pop('intTypeGUMt', None)   # Not used by MC
        kwargs.pop('intervalsGUM', None)  # Not used by MC

        fig, ax = plotting.initplot(plot)
        if hist:
            kwargs.setdefault('bins', 120)
            kwargs.setdefault('density', True)
            kwargs.setdefault('histtype', 'bar')
            kwargs.setdefault('ec', kwargs.get('color', 'C0'))
            kwargs.setdefault('fc', kwargs.get('color', 'C0'))
            kwargs.setdefault('linewidth', 2)

        mean = self.mean.magnitude
        uncert = self.uncert.magnitude
        units = self.units
        x = np.linspace(mean-stdevs*uncert, mean+stdevs*uncert, num=100)
        xmin, xmax = x.min(), x.max()
        if hist and np.isfinite(self.samples.magnitude).any():
            if np.isfinite(xmin) and np.isfinite(xmax) and xmax > xmin:
                kwargs['range'] = xmin, xmax
            ax.hist(self.samples.magnitude, **kwargs)
        elif np.isfinite(xmin) and np.isfinite(xmax):
            y = self.get_pdf(x, bins=kwargs.pop('bins', 200))
            ax.plot(x, y, **kwargs)

        if self.name is not None:
            ax.set_xlabel(self.name)
        if units:
            unitstr = report.Unit(self.units).latex(bracket=True)
            ax.set_xlabel(ax.get_xlabel() + unitstr)

        if intervals:
            ilines, ilabels = [], []
            for covidx, cov in enumerate(intervals):
                mins, maxs, _ = self.expanded(cov, shortest=intervalshortest)
                iline = ax.axvline(mins.magnitude, color=covcolors[covidx])
                ax.axvline(maxs.magnitude, color=covcolors[covidx])
                ilines.append(iline)
                ilabels.append(cov)

            intervallegend = ax.legend(ilines, ilabels, fontsize=10, loc='upper right', title='Monte Carlo\nIntervals')
            ax.add_artist(intervallegend)

    def plot_normprob(self, plot=None, points=None, **kwargs):
        '''
        Plot a normal probability plot of the Monte-Carlo samples vs.
        expected quantile. If data falls on straight line, data is
        normally distributed.

        Parameters
        ----------
        plot: matplotlib figure or axis
            Axis to plot on
        points: int
            Number of sample points to include for speed. Default is 100.
        '''
        fig, ax = plotting.initplot(plot)
        if points is None:
            points = min(100, len(self.samples))

        thin = len(self.samples.magnitude)//points
        plotting.probplot(self.samples[::thin].magnitude, ax=ax)

    def expanded(self, cov=0.95, **kwargs):
        ''' Calculate expanded uncertainty with coverage intervals based on degf.

            Parameters
            ----------
            cov: float or string
                Coverage interval fraction, 0-1 or string with percent, e.g. '95%'

            Keyword Arguments
            -----------------
            shortest: boolean
                Use shortest interval instead of symmetric interval.

            Returns
            -------
            umin: float
                Minimum uncertainty value of coverage range
            umax: float
                Maximum uncertainty value of coverage range
            k: float
                k-value associated with this expanded uncertainty
        '''
        # Generate percentiles from interval list
        if isinstance(cov, str):
            cov = cov.strip('%')
            cov = float(cov) / 100
        assert cov >= 0 and cov <= 1
        if len(self.samples) == 0:
            return np.nan, np.nan, np.nan

        if kwargs.get('shortest', False):
            y = np.sort(self.samples.magnitude)
            q = int(cov*len(y))  # constant number of points in coverage range
            # Shortest interval by looping
            rmin = np.inf
            ridx = 0
            for r in range(len(y)-q):
                if y[r+q] - y[r] < rmin:
                    rmin = y[r+q] - y[r]
                    ridx = r
            quant = (y[ridx]*self.units, y[ridx+q]*self.units)
        else:
            q = [100*(1-cov)/2, 100-100*(1-cov)/2]  # Get cov and 1-cov quantiles
            quant = np.nanpercentile(self.samples.magnitude, q)*self.units

        k = ((quant[1]-quant[0])/(2*self.uncert)).magnitude
        Expanded = namedtuple('Expanded', ['minimum', 'maximum', 'k'])
        return Expanded(quant[0], quant[1], k)

    def report_expanded(self, covlist=None, shortest=False, **kwargs):
        ''' Generate formatted report of expanded uncertainties

            Parameters
            ----------
            covlist: list of float
                Coverage intervals to include in report as fraction (i.e. 0.99 for 99%)

            Returns
            -------
            report.Report
        '''
        if covlist is None:
            covlist = [.99, .95, .90, .68]

        hdr = ['Interval', 'Min', 'Max', 'k']

        rows = []
        for cov in covlist:
            minval, maxval, kval = self.expanded(cov, shortest=shortest)
            if isinstance(cov, float):
                row = ['{:.2f}%'.format(cov*100)]
            else:
                row = [cov]
            row.append(report.Number(minval, matchto=self.uncert))
            row.append(report.Number(maxval, matchto=self.uncert))
            row.append(format(kval, '.3f'))
            rows.append(row)
        r = report.Report(**kwargs)
        r.txt('Shortest Coverage Intervals\n' if shortest else 'Symmetric Coverage Intervals\n')
        r.table(rows, hdr)
        return r


class FuncOutput(output.Output):
    ''' Class to hold output of all calculation methods (BaseOutputs list of GUM, MC)
        for a single function.

        Parameters
        ----------
        baseoutputs: list
            List of BaseOutput objects
        inputfunc: InputFunc object
            Function used to calculate outputs
    '''
    def __init__(self, baseoutputs=None, inputfunc=None):
        self._baseoutputs = baseoutputs
        self.name = inputfunc.name
        self.desc = inputfunc.desc
        self.inputfunc = inputfunc
        self.units = self._baseoutputs[0].units

        # Set each baseoutput as a property so they can be accessed via FuncOutput.gum, FuncOutput.mc, etc.
        for b in self._baseoutputs:
            setattr(self, b._method, b)

    def get_output(self, method=None):
        if method is not None:
            return getattr(self, method)
        else:
            return self

    def validate_gum(self, ndig=2, conf=0.95, full=False):
        ''' Validate GUM by comparing endpoints of 95% coverage interval.

            1. Express u(y) as "a x 10^r" where a has ndig digits and r is integer.
            2. delta = .5 * 10^r
            3. dlow = abs(ymean - uy_gum - ylow_mc); dhi = abs(ymean + uy_gum - yhi_mc)
            4. PASS if dlow < delta and dhi < delta

            Parameters
            ----------
            ndig: int
                Number of significant figures for comparison
            conf: float
                Level of confidence for comparison (0-1 range)
            full: boolean
                Return full set of values, including delta, dlow and dhigh

            Returns
            -------
            valid: boolean
                Validity of the GUM approximation compared to Monte-Carlo
            delta: float
                Allowable delta between GUM and MC
            dlow: float
                Low value abs(ymean - uy_gum - ylow_mc)
            dhi: float
                High value abs(ymean + uy_gum - yhi_mc)
            r: int
                r value used to find delta
            a: int
                a value. u(y) = a x 10^r

            References
            ----------
            GUM-S2, Section 8, also NPL Report DEM-ES-011, Chapter 8
        '''
        assert ndig > 0
        uc, _ = self.gum.expanded(conf)
        ucmin, ucmax, _ = self.mc.expanded(conf, shortest=True)
        r = np.floor(np.log10(np.abs(self.gum.uncert.magnitude))).astype(int) - (ndig-1)
        delta = 0.5 * 10.0**r * self.gum.units
        dlow = abs((self.gum.mean - uc) - ucmin)
        dhi = abs((self.gum.mean + uc) - ucmax)
        if not full:
            return (dlow < delta) & (dhi < delta)
        else:
            a = (self.gum.uncert/10.**r).astype(int)
            fullparams = {'delta': delta,
                          'dlow': dlow,
                          'dhi': dhi,
                          'r': r,
                          'a': a,
                          'gumlo': self.gum.mean-uc,
                          'gumhi': self.gum.mean+uc,
                          'mclo': ucmin,
                          'mchi': ucmax}
            return (dlow < delta) & (dhi < delta), fullparams

    def report(self, **kwargs):
        ''' Generate report of mean/uncertainty values.

            Keyword Arguments
            -----------------
            Passed to report.Report
        '''
        # rows are GUM, MC, LSQ, etc. Columns are means, uncertainties. Checks if baseoutput is CurveFit, and if so uses that format.
        hdr = ['Method', 'Mean', 'Standard Uncertainty']
        rows = []
        for out in self._baseoutputs:
            rows.append([out.method, report.Number(out.mean, matchto=out.uncert), report.Number(out.uncert)])
        r = report.Report(**kwargs)
        r.table(rows, hdr)
        return r

    def report_summary(self, **kwargs):
        ''' Generate report table with method, mean, stdunc, 95% range, and k.

            Keyword Arguments
            -----------------
            Passed to report.Report
        '''
        # rows are GUM, MC, LSQ, etc. Columns are means, uncertainties. Checks if baseoutput is CurveFit, and if so uses that format.
        conf = .95  # Level of confidence for expanded unceratinty
        hdr = ['Method', 'Mean', 'Std. Uncertainty', '95% Coverage', 'k', 'Deg. Freedom']
        rows = []
        for out in self._baseoutputs:
            row = [out.method]
            if out._method == 'mc':
                umin, umax, k = out.expanded(conf)
                rng95 = ('(', report.Number(umin, matchto=out.uncert), ', ', report.Number(umax, matchto=out.uncert), ')')
                row.extend([report.Number(out.mean, matchto=out.uncert), report.Number(out.uncert), rng95, format(k, '.3f'), '-'])
            else:
                unc, k = out.expanded(conf)
                rng95 = (u'± ', report.Number(unc))
                row.extend([report.Number(out.mean, matchto=out.uncert), report.Number(out.uncert), rng95, format(k, '.3f'), format(out.degf, '.1f')])
            rows.append(row)
        r = report.Report(**kwargs)
        r.table(rows, hdr)
        return r

    def report_warns(self, **kwargs):
        ''' Report of warnings raised during calculation

            Returns
            -------
            report.Report
        '''
        r = report.Report(**kwargs)
        for out in self._baseoutputs:
            r.append(out.report_warns(**kwargs))
        return r

    def report_expanded(self, covlist=None, normal=False, shortest=False, covlistgum=None, **kwargs):
        ''' Generate formatted report of expanded uncertainties

            Parameters
            ----------
            covlist: list of float
                Coverage intervals to include in report as fraction (i.e. 0.99 for 99%)
            normal: bool
                Calculate using normal distribution, ignoring degrees of freedom
            shortest: bool
                Use shortest interval instead of symmetric interval (MC only)
            covlistgum: list of string or float
                Coverage intervals for GUM. If not provided, will use same intervals as MC.
                String items may be expressed as "k=2" or as percents "95%".

            Keyword Arguments
            -----------------
            Passed to report.Report

            Returns
            -------
            report.Report
        '''
        if covlist is None:
            covlist = [.99, .95, .90, .68]
        if covlistgum is None:
            covlistgum = covlist

        r = report.Report(**kwargs)
        for out in self._baseoutputs:
            r.hdr(out.method, level=3)
            r.append(out.report_expanded(covlist, normal=normal, shortest=shortest, covlistgum=covlistgum, **kwargs))
        return r

    def report_derivation(self, solve=True, **kwargs):
        ''' Generate report of derivation of GUM method.

            Parameters
            ----------
            solve: bool
                Calculate values of each function.

            Keyword Arguments
            -----------------
            Passed to report.Report

            Returns
            -------
            report.Report
        '''
        return self.gum.report_derivation(solve=solve, **kwargs)  # Only GUM method has derivation

    def plot_converge(self, fig=None, ax1=None, ax2=None, **kwargs):
        ''' Plot Monte-Carlo convergence. Using the same sampled values, the mean
            and standard uncertainty will be recomputed at fixed intervals
            throughout the sample array.

            Parameters
            ----------
            fig: matplotlib figure instance, optional
                If omitted, a new figure will be created.
            ax1, ax2: matplotlib axes instances, optional
                ax1 for expected values, ax2 for uncertainty values

            Keyword Arguments
            -----------------
            div: int
                Number of points to plot
        '''
        return self.mc.plot_converge(fig=fig, ax1=ax1, ax2=ax2, **kwargs)  # Only MC has convergence plot

    def report_components(self, **kwargs):
        ''' Report the uncertainty components.

            Keyword Arguments
            -----------------
            Passed to report.Report
        '''
        cols = ['Variable', 'Component', 'Standard Uncertainty', 'Deg. Freedom', 'Description']
        rows = []
        for i, inpt in enumerate(self.inputfunc.get_basevars()):
            rows.append([report.Math(inpt.name), '', inpt.desc, report.Number(inpt.stdunc()), format(inpt.degf(), '.1f')])
            for u in inpt.uncerts:
                rows.append(['', report.Math(u.name),
                             u.desc if u.desc != '' else '--', report.Number(u.std()), format(u.degf, '.1f')])
        r = report.Report(**kwargs)
        r.table(rows, hdr=cols)
        return r

    def report_inputs(self, **kwargs):
        ''' Report of input values.

            Keyword Arguments
            -----------------
            Passed to report.Report
        '''
        rows = []
        cols = ['Variable', 'Mean', 'Std. Uncertainty', 'Deg. Freedom', 'Description']
        for i in self.inputfunc.get_basevars():
            rows.append([report.Math(i.name),
                         report.Number(i.mean(), matchto=i.stdunc()),
                         report.Number(i.stdunc()),
                         report.Number(i.degf()),
                         i.desc])
        r = report.Report(**kwargs)
        r.table(rows, hdr=cols)
        return r

    def report_sens(self, **kwargs):
        ''' Report sensitivity coefficients '''
        r = report.Report(**kwargs)
        inames = [i.name for i in self.inputfunc.get_basevars()]

        if (hasattr(self, 'gum') and hasattr(self, 'mc') and self.gum.sensitivity is not None and self.gum.props is not None and self.mc.sensitivity is not None and self.mc.props is not None):
            # Both GUM and MC in same tables
            rows = [[report.Math(name),
                     report.Number(sGUM, fmin=1),
                     format(pGUM, '.2f')+'%',
                     report.Number(sMC, fmin=1),
                     format(pMC, '.2f')+'%']
                    for name, sGUM, pGUM, sMC, pMC in zip(inames, self.gum.sensitivity, self.gum.props, self.mc.sensitivity, self.mc.props)]
            if hasattr(self.gum, 'residprops') and self.gum.residprops != 0:
                rows.append(['Correlations', '', format(self.gum.residprops, '.2f')+'%', '', ''])
            r.table(rows, hdr=['Variable', 'GUM Sensitivity', 'GUM Proportion', 'MC Sensitivity', 'MC Proportion'])

        elif hasattr(self, 'gum') and self.gum.sensitivity is not None and self.gum.props is not None:
            rows = [[name, report.Number(sGUM, fmin=1), format(pGUM, '.2f')+'%']
                    for name, sGUM, pGUM in zip(inames, self.gum.sensitivity, self.gum.props)]
            if hasattr(self, 'residprops') and self.gum.residprops != 0:
                rows.append(['Correlations', '', format(self.residprops, '.2f')+'%', '', ''])
            r.table(rows, hdr=['Variable', 'GUM Sensitivity', 'GUM Proportion'])

        elif hasattr(self, 'mc') and self.mc.sensitivity is not None and self.mc.props is not None:
            rows = [[name, report.Number(sMC, fmin=1), format(pMC, '.2f')+'%']
                    for name, sMC, pMC in zip(inames, self.mc.sensitivity, self.mc.props)]
            r.table(rows, hdr=['Variable', 'MC Sensitivity', 'MC Proportion'])
        return r

    def report_validity(self, conf=.95, ndig=1, **kwargs):
        ''' Validate the GUM results by comparing the 95% coverage limits with Monte-Carlo
            results.

            Parameters
            ----------
            conf: float
                Level of confidence (0-1) used for comparison
            ndig: int
                Number of significant digits for comparison

            References
            ----------
            GUM Supplement 1, Section 8
            NPL Report DEM-ES-011, Chapter 8
        '''
        r = report.Report(**kwargs)
        r.hdr('Comparison to Monte Carlo {:.2f}% Coverage'.format(conf*100), level=3)
        if not hasattr(self, 'gum') or not hasattr(self, 'mc'):
            r.txt('GUM and Monte Carlo result not run. No validity check can be made.')
        else:
            valid, params = self.validate_gum(ndig=ndig, conf=conf, full=True)
            deltastr = report.Number(params['delta'], fmin=1)
            r.txt(u'{:d} significant digit{}. δ = {}.\n\n'.format(ndig, 's' if ndig > 1 else '', deltastr))

            rows = []
            hdr = ['{:.2f}% Coverage'.format(conf*100), 'Lower Limit', 'Upper Limit']
            rows.append(['GUM', report.Number(params['gumlo'], matchto=params['dlow']), report.Number(params['gumhi'], matchto=params['dhi'])])
            rows.append(['MC', report.Number(params['mclo'], matchto=params['dlow']), report.Number(params['mchi'], matchto=params['dhi'])])
            rows.append(['abs(GUM - MC)', report.Number(params['dlow'], matchto=params['dlow']), report.Number(params['dhi'], matchto=params['dhi'])])
            rows.append([u'abs(GUM - MC) < δ', '<font color="green">PASS</font>' if params['dlow'] < params['delta'] else '<font color="red">FAIL</font>',
                                          '<font color="green">PASS</font>' if params['dhi'] < params['delta'] else '<font color="red">FAIL</font>'])
            r.table(rows, hdr=hdr)
        return r

    def plot_pdf(self, plot=None, **kwargs):
        ''' Plot probability density function of all methods

            Parameters
            ----------
            plot: matplotlib figure or axis
                Axis to plot on

            Keyword Arguments
            -----------------
            stddevs: float
                Number of standard deviations to include in x range
            label: string
                Label for curve on plot

            Returns
            -------
            ax: matplotlib axis
        '''
        fig, ax = plotting.initplot(plot)
        for b in self._baseoutputs:
            kwargs['label'] = b.method
            b.plot_pdf(plot=ax, **kwargs)
        unitstr = report.Unit(self._baseoutputs[0].units).latex(bracket=True)
        ax.set_xlabel(report.Math(self.name).latex() + unitstr)
        ax.legend(loc='best', fontsize=10)
        return ax


class CalcOutput(output.Output):
    ''' Class to hold FuncOutputs for every function in calculator

        Parameters
        ----------
        foutputs: list
            List of FuncOutput objects
        ucalc: UncertCalc object
            object that generated this output
        warns: list
            List of warning strings generated at the calculator level
    '''
    def __init__(self, foutputs=None, ucalc=None, warns=None):
        self.foutputs = foutputs
        self.ucalc = ucalc
        self.warns = warns

        for f in self.foutputs:
            if f.name and f.name.isidentifier() and not hasattr(self, f.name):
                setattr(self, f.name, f)

    def get_funcnames(self):
        ''' Return list of function names in this calc output '''
        return [f.name for f in self.foutputs]

    def get_output(self, fname=None, fidx=None, method=None):
        ''' Get an output object

            Parameters
            ----------
            fname: string
                Name of function to get
            fidx: int
                Index of function to get if no name given
            method: string
                Name of calculation method (e.g. 'gum' or 'mc')

            Notes
            -----
            If fname and fidx are not provided, the first function in the list
            will be returned.

            Returns
            -------
            BaseOutput object if method is provided, FuncOutput object otherwise
        '''
        if fname is None:
            fidx = 0 if fidx is None else fidx
            foutput = self.foutputs[fidx]
        else:
            foutput = getattr(self, fname)

        if method is not None:
            return getattr(foutput, method)
        else:
            return foutput

    def get_dists(self):
        ''' Get distributions in this output. If name is none, return a list of
            available distribution names.
        '''
        dists = {}
        for f in self.foutputs:
            for method in f._baseoutputs:
                name = '{} ({})'.format(f.name, method._method.upper())
                baseout = self.get_output(f.name, method=method._method)
                if 'mc' in method._method:
                    samples = baseout.properties['samples'].magnitude
                    expected = baseout.properties['expected'].magnitude
                    dists[name] = {'samples': samples,
                                   'median': np.median(samples),
                                   'expected': expected}
                else:
                    dists[name] = {'mean': baseout.mean.magnitude,
                                   'std': baseout.uncert.magnitude,
                                   'df': baseout.degf}
        return dists

    def report(self, **kwargs):
        ''' Generate report of all mean/uncerts for each function.

            Keyword Arguments
            -----------------
            Passed to report.Report
        '''
        r = report.Report(**kwargs)
        for f in self.foutputs:
            if f.inputfunc.show:
                r.sympy(f.inputfunc.full_func(), end='\n\n')
                r.append(f.report(**kwargs))
        return r

    def report_summary(self, **kwargs):
        ''' Generate report of all mean/uncerts for each function.

            Keyword Arguments
            -----------------
            Passed to report.Report
        '''
        r = self.report_summary_table(**kwargs)
        r.append(self.report_summary_plot(**kwargs))
        r.append(self.report_warns(**kwargs))
        return r

    def report_summary_table(self, **kwargs):
        ''' Generate summary table

            Keyword Arguments
            -----------------
            Passed to report.Report
        '''
        r = report.Report()
        for f in self.foutputs:
            if f.inputfunc.show:
                r.sympy(f.inputfunc.full_func(), end='<br>\n\n')
                r.append(f.report_summary(**kwargs))
        return r

    def report_summary_plot(self, **kwargs):
        ''' Generate summary plot

            Keyword Arguments
            -----------------
            Passed to report.Report
        '''
        with mpl.style.context(plotting.mplcontext):
            # TODO: suppress this plot from showing twice in jupyter
            plt.ioff()
            fig = plt.figure()
            self.plot_pdf(plot=fig)
        r = report.Report()
        r.plot(fig)
        return r

    def report_all(self, **kwargs):
        ''' Generate comprehensive report.

            Keyword Arguments
            -----------------
            summary: bool
            outputs: bool
            inputs: bool
            components: bool
            sens: bool
            expanded: bool
            gumderv: bool
            gumvalid: bool
            mchist: bool
            mcconv: bool
                Enable/disable plot sections
            outplotparams: dict
                Parameters for output plot
            expandedparams: dict
                Parameters for expanded uncertainty report
            mchistparams: dict
                Parameters for Monte Carlo histogram plot
        '''
        with mpl.style.context(plotting.mplcontext):
            plt.ioff()
            r = report.Report(**kwargs)
            if kwargs.get('summary', True):
                r.hdr('Summary', level=2)
                r.append(self.report_summary_table(**kwargs))
            if kwargs.get('outputs', True):
                fig = plt.figure()
                params = kwargs.get('outplotparams', {})
                joint = params.get('joint', False)
                if joint:
                    self.plot_outputscatter(plot=fig, **params.get('plotargs', {}))
                else:
                    self.plot_pdf(plot=fig, **params.get('plotargs', {}))
                r.plot(fig)
            if kwargs.get('inputs', True):
                r.hdr('Standardized Input Values', level=2)
                r.append(self.report_inputs(**kwargs))
                r.div()
            if kwargs.get('components', True):
                r.hdr('Uncertainty Budget', level=2)
                r.append(self.report_components(**kwargs))
                r.div()
            if kwargs.get('sens', True):
                r.hdr('Sensitivity Coefficients', level=2)
                r.append(self.report_sens(**kwargs))
                r.div()
            if kwargs.get('expanded', True):
                params = kwargs.get('expandedparams', {'intervalsgum': None, 'intervalsmc': None, 'norm': False, 'shortest': False})
                r.hdr('Expanded Uncertainties', level=2)
                r.append(self.report_expanded(covlist=params.get('intervalsmc', [0.95]),
                                              normal=params.get('norm', False), shortest=params.get('shortest', False),
                                              covlistgum=params.get('intervalsgum', None), **kwargs))
            if kwargs.get('gumderv', True) and hasattr(self.foutputs[0], 'gum'):
                solve = kwargs.get('gumvalues', False)
                r.hdr('GUM Derivation', level=2)
                r.append(self.report_derivation(solve=solve, **kwargs))
            if kwargs.get('gumvalid', True) and hasattr(self.foutputs[0], 'gum') and hasattr(self.foutputs[0], 'mc'):
                ndig = kwargs.get('gumvaliddig', 2)
                r.hdr('GUM Validity', level=2)
                r.append(self.report_validity(ndig=ndig, **kwargs))
            if kwargs.get('mchist', True) and hasattr(self.foutputs[0], 'mc'):
                params = kwargs.get('mchistparams', {})
                fig = plt.figure()
                plotparams = params.get('plotargs', {})
                if params.get('joint', False):
                    self.plot_xscatter(plot=fig, **plotparams)
                else:
                    self.plot_xhists(plot=fig, **plotparams)
                r.hdr('Monte Carlo Inputs', level=2)
                r.plot(fig)
            if kwargs.get('mcconv', True) and hasattr(self.foutputs[0], 'mc'):
                relative = kwargs.get('mcconvnorm', False)
                fig = plt.figure()
                self.plot_converge(fig, relative=relative)
                r.hdr('Monte Carlo Convergence', level=2)
                r.plot(fig)
        return r

    def report_func(self, **kwargs):
        ''' Report a list of functions in the calculator '''
        r = report.Report(**kwargs)
        for f in self.foutputs:
            if f.inputfunc.show:
                r.sympy(f.inputfunc.full_func(), end='\n\n')
        return r

    def report_warns(self, **kwargs):
        ''' Report warnings raised during calculation '''
        r = report.Report(**kwargs)
        if self.warns:   # Calculator-level warnings
            for w in self.warns:
                r.txt('- ' + w + '\n')

        for f in self.foutputs:
            r.append(f.report_warns(**kwargs))
        return r

    def report_expanded(self, covlist=None, normal=False, shortest=False, covlistgum=None, **kwargs):
        ''' Generate formatted report of expanded uncertainties

            Parameters
            ----------
            covlist: list of float
                Coverage intervals to include in report as fraction (i.e. 0.99 for 99%)
            normal: bool
                Calculate using normal distribution, ignoring degrees of freedom (GUM only)
            shortest: bool
                Use shortest interval instead of symmetric interval (MC only)
            covlistgum: list of string or float
                Coverage intervals for GUM. If not provided, will use same intervals as MC.
                String items may be expressed as "k=2" or as percents "95%".

            Keyword Arguments
            -----------------
            Passed to report.Report

            Returns
            -------
            report.Report
        '''
        if covlist is None:
            covlist = [.99, .95, .90, .68]
        if covlistgum is None:
            covlistgum = covlist

        r = report.Report()
        for i, f in enumerate(self.foutputs):
            if f.inputfunc.show:
                if len(self.foutputs) > 1:
                    if i > 0:
                        r.div()
                    r.sympy(f.inputfunc.full_func(), end='\n\n')
                r.append(f.report_expanded(covlist, normal=normal, shortest=shortest, covlistgum=covlistgum, **kwargs))
        return r

    def report_components(self, **kwargs):
        ''' Report the uncertainty components

            Keyword Arguments
            -----------------
            Passed to report.Report
        '''
        cols = ['Variable', 'Component', 'Description', 'Standard Uncertainty', 'Deg. Freedom']
        rows = []
        for i, inpt in enumerate(self.ucalc.get_baseinputs()):
            rows.append([report.Math(inpt.name), '-',
                         inpt.desc if inpt.desc else '--', report.Number(inpt.stdunc()), format(inpt.degf(), '.1f')])

            for u in inpt.uncerts:
                rows.append(['-', report.Math(u.name),
                             u.desc if u.desc else '--', report.Number(u.std()), format(u.degf, '.1f')])
        r = report.Report(**kwargs)
        r.table(rows, hdr=cols)
        return r

    def report_derivation(self, solve=True, **kwargs):
        ''' Generate report of derivation of GUM method.

            Parameters
            ----------
            solve: bool
                Calculate values of each function.

            Keyword Arguments
            -----------------
            Passed to report.Report

            Returns
            -------
            report.Report
        '''
        r = report.Report()
        for f in self.foutputs:
            if f.inputfunc.show:
                r.append(f.report_derivation(solve=solve, **kwargs))
                r.div()
        return r

    def report_validity(self, conf=.95, ndig=1, **kwargs):
        ''' Validate the GUM results by comparing the 95% coverage limits with Monte-Carlo
            results.

            Parameters
            ----------
            conf: float
                Level of confidence (0-1) for comparison
            ndig: int
                Number of significant digits for comparison

            References
            ----------
            GUM Supplement 1, Section 8
            NPL Report DEM-ES-011, Chapter 8
        '''
        r = report.Report(**kwargs)
        for f in self.foutputs:
            if f.inputfunc.show:
                if len(self.foutputs) > 1:
                    r.sympy(f.inputfunc.full_func(), end='\n\n')
                r.append(f.report_validity(conf=conf, ndig=ndig, **kwargs))
                r.div()
        return r

    def report_inputs(self, **kwargs):
        ''' Report of input values.

            Keyword Arguments
            -----------------
            Passed to report.Report
        '''
        rows = []
        cols = ['Variable', 'Mean', 'Std. Uncertainty', 'Deg. Freedom', 'Description']
        for i in self.ucalc.get_baseinputs():  # Merge all functions together
            rows.append([report.Math(i.name),
                         report.Number(i.mean(), matchto=i.stdunc()),
                         report.Number(i.stdunc()),
                         report.Number(i.degf()),
                         i.desc if i.desc != '' else '--'])
        r = report.Report(**kwargs)
        r.table(rows, hdr=cols)
        return r

    def report_allinputs(self, **kwargs):
        ''' Combined report of inputs, uncertainties, and sensitivity coefficients.

            Keyword Arguments
            -----------------
            Passed to report.Report
        '''
        r = report.Report(**kwargs)
        r.hdr('Input Measurements', level=2)
        r.append(self.report_inputs(**kwargs))
        r.div()
        r.hdr('Uncertainty Budget', level=2)
        r.append(self.report_components(**kwargs))
        r.div()
        r.hdr('Sensitivity Coefficients', level=2)
        r.append(self.report_sens(**kwargs))
        return r

    def report_sens(self, **kwargs):
        ''' Report sensitivity coefficients.

            Keyword Arguments
            -----------------
            Passed to report.Report
        '''
        r = report.Report(**kwargs)
        for f in self.foutputs:
            if f.inputfunc.show:
                if len(self.foutputs) > 1:
                    r.sympy(f.inputfunc.full_func(), end='\n\n')
                r.append(f.report_sens(**kwargs))
        return r

    def plot_pdf(self, plot=None, **kwargs):
        ''' Plot output probability (histogram and/or PDF curve)

            Parameters
            ----------
            plot: matplotlib figure or axis
                Figure or axis to plot on. Existing figure will be cleared. If omitted, a new figure/axis
                will be created in interactive mode.

            Keyword Arguments
            -----------------
            funcs: list of ints
                List of functions indexes to include
            stds : scalar
                Standard deviations above/below mean to show in x-scale [default 4]
            intervals: list of float
                Percent values for coverage intervals to show
            mccolor: string
                Color for monte carlo histogram
            gumcolor: string
                Color for GUM PDF
            covcolors: list of string
                Colors of coverage interval lines
            showmc: boolean
                Show Monte-Carlo histogram
            showgum: boolean
                Show GUM normal curve
            bins: int
                Number of bins in histogram
            contour: bool
                Approximate PDF as line instead of histogram
            legend: bool
                Show legend (GUM and MC lines). Note intervals legend is always shown when intervals==True.
            labelmode: string
                'name' or 'desc'. Set axis labels by variable name or description.
        '''
        funcs = kwargs.pop('funcs', [i for i, f in enumerate(self.foutputs) if f.inputfunc.show])
        stds = kwargs.pop('stddevs', 4)
        mccolor = kwargs.pop('mccolor', 'C0')
        gumcolor = kwargs.pop('gumcolor', 'C1')
        showmc = kwargs.pop('showmc', hasattr(self.foutputs[0], 'mc'))
        showgum = kwargs.pop('showgum', hasattr(self.foutputs[0], 'gum'))
        bins = max(3, kwargs.pop('bins', 120))
        contour = kwargs.pop('contour', False)
        labelmode = kwargs.pop('labelmode', 'name')  # or 'desc'
        showleg = kwargs.pop('legend', True)
        [kwargs.pop(k, None) for k in ['points', 'inpts', 'overlay', 'cmap', 'cmapmc', 'equal_scale']]  # Ignore these in pdf plot

        fig, ax = plotting.initplot(plot)
        fig.clf()
        fig.subplots_adjust(**dfltsubplots)
        if len(funcs) == 0: return fig

        rows = int(np.ceil(len(funcs)/3))
        cols = int(np.ceil(len(funcs)/rows))

        for plotid, func in enumerate([self.foutputs[i] for i in funcs]):
            ax = fig.add_subplot(rows, cols, plotid+1)

            try:
                mean = func.mc.mean.magnitude
                uncert = func.mc.uncert.magnitude
                units = func.mc.units
            except AttributeError:
                mean = func.gum.mean.magnitude
                uncert = func.gum.uncert.magnitude
                units = func.gum.units

            xlow = mean - stds * uncert
            xhi = mean + stds * uncert
            if not np.isfinite(xlow):  # Can happen if uncert is Nan for some reason, like with no input variables
                xlow = mean - 1
                xhi = mean + 1

            if showmc:
                kwargs['color'] = mccolor
                func.mc.plot_pdf(plot=ax, hist=not contour, bins=bins, stddevs=stds, **kwargs)

            if showgum:
                kwargs['color'] = gumcolor
                func.gum.plot_pdf(plot=ax, stddevs=stds, **kwargs)

            if np.isfinite(xlow) and np.isfinite(xhi) and xhi != xlow:
                ax.set_xlim(xlow, xhi)

            ax.ticklabel_format(style='sci', axis='x', scilimits=(-4, 4), useOffset=False)
            ax.set_ylabel('Probability Density')
            unitstr = report.Unit(units).latex(bracket=True)
            if labelmode == 'desc' and func.mc.desc is not None:
                ax.set_xlabel(func.mc.desc + unitstr)
            elif func.name:
                ax.set_xlabel(report.Math(func.name).latex() + unitstr)
            else:
                ax.set_xlabel(unitstr)

            if showleg:
                ax.legend(loc='upper left', fontsize=10)

        fig.tight_layout()

    def plot_outputscatter(self, plot=None, **kwargs):
        ''' Plot scatter plot of output functions (if more than one function in system)

            Parameters
            ----------
            plot: matplotlib figure or axis
                Figure or axis to plot on. Will be cleared.

            Keyword Arguments
            -----------------
            funcs: list of int
                List of function indexes to include
            color: string
                Color for scatter points
            contour: bool
                Show as contour lines instead of scatter points
            showgum: bool
                Show the GUM contour plot
            showmc: bool
                Show the MC scatter/contour plot
            points: int
                Number of points to include in scatter plot
            bins: int
                Number of bins for converting scatter to contour
            labelmode: string
                Either 'desc' or 'name', how to label axes
            overlay: bool
                Show MC and GUM plots on same axis
            equal_scale: bool
                Equalize the x and y limits of GUM and MC axes
            cmap: string
                Name of matplotlib colormap for GUM plot
            cmapmc: string
                Name of matplotlib colormap for MC plot
        '''
        funcs = kwargs.get('funcs', None)
        color = kwargs.get('color', 'C3')
        contour = kwargs.get('contour', False)
        showgum = kwargs.pop('showgum', hasattr(self.foutputs[0], 'gum'))
        showmc = kwargs.pop('showmc', hasattr(self.foutputs[0], 'mc'))
        points = kwargs.get('points', -1)
        bins = max(3, kwargs.get('bins', 40))
        labelmode = kwargs.get('labelmode', 'name')
        overlay = kwargs.get('overlay', False)
        equal_scale = kwargs.get('equal_scale', False)
        cmap = kwargs.get('cmap', 'viridis')
        cmapmc = kwargs.get('cmapmc', 'viridis')
        overlay = overlay and showgum and showmc

        fig, ax = plotting.initplot(plot)
        fig.clf()
        fig.subplots_adjust(**dfltsubplots)

        if funcs is None:
            funcs = list(range(len(self.foutputs)))

        num = len(funcs)
        if num < 2 or len(funcs) < 2:
            return fig  # Nothing to plot

        for row in range(num):
            for col in range(num):
                if col <= row: continue
                if overlay:
                    axgum = fig.add_subplot(num-1, num-1, row*(num-1)+col)
                    axmc = axgum
                elif showgum and showmc:
                    axgum = fig.add_subplot(num-1, (num-1)*2, row*(num-1)*2+col*2-1)
                    axmc = fig.add_subplot(num-1, (num-1)*2, row*(num-1)*2+col*2)
                elif showgum:
                    axgum = fig.add_subplot(num-1, num-1, row*(num-1)+col)
                elif showmc:
                    axmc = fig.add_subplot(num-1, num-1, row*(num-1)+col)
                else:
                    return  # Don't show either one!

                f1, f2 = funcs[row], funcs[col]

                if showgum:  # Only contour (no scatter)
                    x, y, h = self.ucalc.get_contour(f1, f2)
                    if h is not None:
                        axgum.contour(x.magnitude, y.magnitude, h, 10, cmap=plt.get_cmap(cmap))
                        axgum.locator_params(nbins=5)
                        if labelmode == 'desc':
                            axgum.set_xlabel(self.foutputs[f1].desc)
                            axgum.set_ylabel(self.foutputs[f2].desc)
                        else:
                            axgum.set_xlabel('$' + self.foutputs[f1].inputfunc.get_latex() + '$' + report.Unit(self.foutputs[f1].units).latex(bracket=True))
                            axgum.set_ylabel('$' + self.foutputs[f2].inputfunc.get_latex() + '$' + report.Unit(self.foutputs[f2].units).latex(bracket=True))
                        axgum.ticklabel_format(style='sci', axis='x', scilimits=(-4, 4), useOffset=False)
                        axgum.ticklabel_format(style='sci', axis='y', scilimits=(-4, 4), useOffset=False)

                if showmc:
                    axmc.locator_params(nbins=5)
                    x = self.foutputs[f1].mc.samples.magnitude
                    y = self.foutputs[f2].mc.samples.magnitude
                    mask = np.isfinite(x) & np.isfinite(y)
                    x = x[mask]
                    y = y[mask]
                    if contour:
                        counts, ybins, xbins = np.histogram2d(y, x, bins=bins)
                        if overlay:  # Use filled contour plot when showing both
                            axmc.contourf(counts, 10, extent=[xbins.min(), xbins.max(), ybins.min(), ybins.max()], cmap=plt.get_cmap(cmapmc))
                        else:
                            axmc.contour(counts, 10, extent=[xbins.min(), xbins.max(), ybins.min(), ybins.max()], cmap=plt.get_cmap(cmapmc))
                    else:
                        with suppress(ValueError):  # Raises in case where len(x) != len(y) when one output is constant
                            axmc.plot(x[:points], y[:points], marker='.', ls='', markersize=2, color=color, zorder=0)
                    if labelmode == 'desc':
                        axmc.set_xlabel(self.foutputs[f1].mc.desc)
                        axmc.set_ylabel(self.foutputs[f2].mc.desc)
                    else:
                        axmc.set_xlabel('$' + self.foutputs[f1].inputfunc.get_latex() + '$' + report.Unit(self.foutputs[f1].units).latex(bracket=True))
                        axmc.set_ylabel('$' + self.foutputs[f2].inputfunc.get_latex() + '$' + report.Unit(self.foutputs[f2].units).latex(bracket=True))
                    axmc.ticklabel_format(style='sci', axis='x', scilimits=(-4, 4), useOffset=False)
                    axmc.ticklabel_format(style='sci', axis='y', scilimits=(-4, 4), useOffset=False)

                if equal_scale and showgum and showmc and not overlay:
                    xlim1, xlim2 = axmc.get_xlim(), axgum.get_xlim()
                    ylim1, ylim2 = axmc.get_ylim(), axgum.get_ylim()
                    xlim = (min(xlim1[0], xlim2[0]), max(xlim1[1], xlim2[1]))
                    ylim = (min(ylim1[0], ylim2[0]), max(ylim1[1], ylim2[1]))
                    if np.isfinite(xlim[0]) and np.isfinite(xlim[1]):
                        axmc.set_xlim(xlim)
                        axgum.set_xlim(xlim)
                    if np.isfinite(ylim[0]) and np.isfinite(ylim[1]):
                        axmc.set_ylim(ylim)
                        axgum.set_ylim(ylim)

                if not overlay and showgum and showmc:
                    axmc.set_title('Monte-Carlo')
                    axgum.set_title('GUM Approximation')

        fig.tight_layout()

    def plot_xhists(self, plot=None, **kwargs):
        ''' Plot histograms for each input variable.

            Parameters
            ----------
            plot: matplotlib figure or axis
                If omitted, a new figure will be created. Existing figure will be cleared.

            Keyword Arguments
            -----------------
            color: string
                Color to use for histogram bars
            inpts: List of int
                List of input indexes to include. None= include all
            bins: int or 'auto'
                Number of bins for histograms
            labelmode: 'name' or 'desc'
                Label axes with input name or description
        '''
        color = kwargs.get('color', 'C0')
        inpts = kwargs.get('inpts', None)
        bins = max(3, kwargs.get('bins', 200))
        labelmode = kwargs.get('labelmode', 'name')

        fig, ax = plotting.initplot(plot)

        # fig.clf() doesn't reset subplots_adjust parameters that change on tight_layout.
        # Reset them here or we can get strange exceptions about negative width while
        # adding the new axes
        fig.clf()
        fig.subplots_adjust(**dfltsubplots)

        variables = self.ucalc.get_baseinputs()
        if inpts is not None:
            if len(inpts) == 0: return fig
            variables = [variables[i] for i in inpts]

        axs = plotting.axes_grid(len(variables), fig)
        for ax, inpt in zip(axs, variables):
            ax.hist(inpt.sampledvalues.magnitude, bins=bins, density=True, color=color, ec=color, histtype='bar', label=inpt.name)
            ax.ticklabel_format(style='sci', axis='x', scilimits=(-4, 4), useOffset=False)
            ax.yaxis.set_visible(False)
            unitstr = report.Unit(inpt.sampledvalues.units).latex(bracket=True)
            if labelmode == 'desc':
                ax.set_xlabel(inpt.desc + unitstr)
            else:
                ax.set_xlabel('$' + inpt.get_latex() + '$' + unitstr)
        fig.tight_layout()

    def plot_xscatter(self, plot=None, **kwargs):
        ''' Plot input samples against each other to look for correlation.

            Parameters
            ----------
            plot: matplotlib figure or axis
                If omitted, a new figure will be created.

            Keyword Arguments
            -----------------
            color: string
                Color to use for scatter plot
            contour: boolean
                Draw as contour (true) or histogram (false)
            cmapmc: string
                Colormap to use for contour plot
            inpts: List of int
                List of input indexes to include. None= include all
            points: int
                Limit number of scatter plot points for faster drawing
            bins: int
                Number of bins for histogram
            labelmode: string ['name' or 'desc']
                Label axes using input name or input description
        '''
        color = kwargs.get('color', 'C3')
        cmapmc = kwargs.get('cmapmc', 'viridis')
        contour = kwargs.get('contour', False)
        inpts = kwargs.get('inpts', None)
        points = kwargs.get('points', -1)
        bins = max(3, kwargs.get('bins', 200))
        labelmode = kwargs.get('labelmode', 'name')

        fig, ax = plotting.initplot(plot)
        fig.clf()
        fig.subplots_adjust(**dfltsubplots)

        variables = self.ucalc.get_baseinputs()
        if inpts is not None and len(inpts) > 1:
            variables = [variables[i] for i in inpts]
        elif inpts is not None:
            return fig  # Only one input, can't plot scatter

        num = len(variables)
        if num <= 1:
            return fig# 1 or 0 inputs to plot!

        LPAD = .1
        RPAD = .02
        w = (1-LPAD-RPAD)/(num-1)
        for row in range(num):
            for col in range(num):
                if col <= row: continue
                # Use add_axes instead of add_subplot because add_subplot tries to be too smart
                # and adjusts for axes labels etc. and will crash with lots of axes having long labels.
                ax = fig.add_axes([LPAD+w*(col-1), 1-w*(row+1), w, w])
                ax.locator_params(nbins=5)

                x = variables[col].sampledvalues.magnitude
                y = variables[row].sampledvalues.magnitude
                xunit = report.Unit(variables[col].units).latex(bracket=True)
                yunit = report.Unit(variables[row].units).latex(bracket=True)

                if contour:
                    counts, ybins, xbins = np.histogram2d(y, x, bins=bins)
                    ax.contour(counts, 10, extent=[xbins.min(), xbins.max(), ybins.min(), ybins.max()], cmap=plt.get_cmap(cmapmc))
                else:
                    ax.plot(x[:points], y[:points], marker='.', ls='', markersize=2,
                            color=color, label='{} vs. {}'.format(variables[col].name, variables[row].name))

                if col == row+1:
                    if labelmode == 'desc' and variables[col].desc != '' and variables[row].desc != '':
                        ax.set_xlabel(variables[col].desc + xunit)
                        ax.set_ylabel(variables[row].desc + yunit)
                    else:
                        ax.set_xlabel('$' + variables[col].get_latex() + '$' + xunit)
                        ax.set_ylabel('$' + variables[row].get_latex() + '$' + yunit)
                else:
                    ax.xaxis.set_visible(False)
                    ax.yaxis.set_visible(False)
                ax.ticklabel_format(style='sci', axis='x', scilimits=(-4, 4), useOffset=False)
                ax.ticklabel_format(style='sci', axis='y', scilimits=(-4, 4), useOffset=False)

    def plot_converge(self, plot=None, **kwargs):
        ''' Plot Monte-Carlo convergence. Using the same sampled values, the mean
            and standard uncertainty will be recomputed at fixed intervals
            throughout the sample array.

            Parameters
            ----------
            plot: matplotlib figure or axis
                If omitted, a new figure will be created.

            Keyword Arguments
            -----------------
            div: int
                Number of points to plot
        '''
        rows = len(self.foutputs)
        fig, ax = plotting.initplot(plot)
        fig.clf()
        fig.subplots_adjust(**dfltsubplots)
        cols = 2
        ax = []
        for i in range(rows):
            ax.append([fig.add_subplot(rows, cols, i*2+1), fig.add_subplot(rows, cols, i*2+2)])
        assert len(ax) == len(self.foutputs)

        for i, f in enumerate(self.foutputs):
            f.plot_converge(ax1=ax[i][0], ax2=ax[i][1], **kwargs)
        fig.tight_layout()
