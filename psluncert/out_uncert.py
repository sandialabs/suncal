'''
The classes in this module hold the results of an Uncertainty Calculator calculation.
In addition to the raw calculated values, methods are available for generating
formatted reports and plots.
'''
import sympy
from scipy import stats
import numpy as np

from . import output

try:
    import matplotlib as mpl
    import matplotlib.pyplot as plt
except ImportError:
    pass
else:
    mpl.style.use('bmh')

dfltsubplots = {'wspace': .1, 'hspace': .1, 'left': .05, 'right': .95, 'top': .95, 'bottom': .05}


def create_output(method='gum', **kwargs):
    ''' Factory function for creating a BaseOutput '''
    if method.lower() in ['mc', 'mcmc']:
        return BaseOutputMC(method, **kwargs)
    else:
        return BaseOutput(method, **kwargs)


class BaseOutput(output.Output):
    ''' Class to hold results from one calculation method (GUM, MC, or LSQ) on one function

        Parameters
        ----------
        method: string
            Method of calculation ['gum', 'mc', 'lsq', 'mcmc']

        Keyword Arguments
        -----------------
        mean: float or array
            Mean value(s)
        uncert: float or array
            Uncertainty value(s)
        pnames: list of string, optional
            List of parameters, same length as mean and uncert
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
        properties: dict
            Other output attributes
    '''
    def __init__(self, method, **kwargs):
        self._method = method
        self.mean = kwargs.pop('mean')
        self.uncert = kwargs.pop('uncert')
        self.mean = np.atleast_1d(self.mean)
        self.uncert = np.atleast_1d(self.uncert)
        self.pnames = kwargs.pop('pnames', None)
        self.desc = kwargs.pop('desc', None)
        assert len(self.mean) == len(self.uncert)
        if len(self.mean) > 1:
            assert len(self.pnames) == len(self.mean)

        self.name = kwargs.pop('name', None)
        self.props = kwargs.pop('props', None)    # Proportions
        if 'residual' in kwargs:
            self.residprops = kwargs.pop('residual', None)
        self.sensitivity = kwargs.pop('sensitivity', None)
        self.degf = kwargs.pop('degf', np.inf)
        self.properties = kwargs

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
            See NumFormatter()

            Returns
            -------
            report: MDstring
        '''
        hdr = ['-'] + self.pnames if self.pnames is not None else None
        rowm = ['Mean'] + [output.formatter.f(m, matchto=u, **kwargs) for m, u in zip(self.mean, self.uncert)]
        rowu = ['Standard Uncertainty'] + [output.formatter.f(u, **kwargs) for u in self.uncert]
        r = output.md_table([rowm, rowu], hdr=hdr, **kwargs)
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
            See NumFormatter()

            Returns
            -------
            report: MDstring
        '''
        covlist = kwargs.get('covlistgum', covlist)
        if covlist is None:
            covlist = [.99, .95, .90, .68]

        rows = []
        if self.pnames is not None:
            hdr = ['Interval'] + list(zip([['Min({})'.format(p) for p in self.pnames],
                                           ['Max({})'.format(p) for p in self.pnames],
                                           ['Degf({})'.format(p) for p in self.pnames],
                                           ['k({})'.format(p) for p in self.pnames],
                                           ['U({})'.format(p) for p in self.pnames]]))
        else:
            hdr = ['Interval', 'Min', 'Max', 'k', 'Deg. Freedom', 'Expanded Uncertainty']
        for cov in covlist:
            uncerts, k = self.expanded(cov, normal=normal)
            if isinstance(cov, float):
                row = ['{:.2f}%'.format(cov*100)]
            else:
                row = [cov]
            for idx, u in enumerate(uncerts):
                row.append(output.formatter.f(self.mean[idx]-u, matchto=u, **kwargs))
                row.append(output.formatter.f(self.mean[idx]+u, matchto=u, **kwargs))
                row.append(format(k, '.3f'))
                row.append(format(self.degf, '.2f'))
                row.append(output.formatter.f(u, matchto=u, **kwargs))
            rows.append(row)
        r = output.md_table(rows, hdr, **kwargs)
        return r

    def report_warns(self, **kwargs):
        ''' Report of warnings raised during calculation

            Returns
            -------
            report: MDstring
        '''
        r = output.MDstring()
        if 'warns' in self.properties and self.properties.get('warns') is not None:
            r = '\n'.join(self.properties['warns'])
        return r

    def report_derivation(self, solve=True, **kwargs):
        ''' Generate report of derivation of GUM method.

            Parameters
            ----------
            solve: bool
                Calculate values of each function.

            Returns
            -------
            report: MDstring
        '''
        N = kwargs.get('n', output.formatter.n)
        if self._method != 'gum':
            r = 'No derivation for {} method'.format(self.method)
            return output.MDstring(r)

        if 'symbolic' not in self.properties:
            r = 'No symbolic solution computed for function.'
            return output.MDstring(r)

        symout = self.properties['symbolic']
        r = output.MDstring('### Function:\n\n')
        r += output.sympyeqn(symout['function']) + '\n\n'

        r += 'GUM formula for combined uncertainty:\n\n'
        uname = sympy.Symbol('u_'+str(self.name) if self.name != '' and self.name is not None else 'u_c')
        uformula = symout['unc_formula']
        if self.name != '':
            uformula = sympy.Eq(uname, uformula, evaluate=False)
        r += output.sympyeqn(uformula) + '\n\n'

        r += '### Input Definitions:\n\n'
        rows = []
        for var, val, uncert in zip(symout['var_symbols'], symout['var_means'], symout['var_uncerts']):
            vareqn = var if not solve else sympy.Eq(var, val)
            uncsym = sympy.Symbol('u_{{{}}}'.format(str(var)))
            unceqn = uncsym if not solve else sympy.Eq(uncsym, sympy.sympify(uncert).n(N))
            rows.append([output.sympyeqn(vareqn), output.sympyeqn(unceqn)])
        r += output.md_table(rows, hdr=['Variable', 'Std. Uncertainty'], **kwargs)

        if 'covsymbols' in symout:
            r += 'Correlation coefficients:\n'
            for var, cov in zip(symout['covsymbols'], symout['covvals']):
                eq = var if not solve else sympy.Eq(var, cov).n(3)
                r += output.sympyeqn(eq) + '\n\n'

        r += '### Sensitivity Coefficients:\n\n'
        for c, p1, p2, p3 in zip(symout['var_symbols'], symout['partials_raw'], symout['partials'], symout['partials_solved']):
            eq = sympy.Eq(sympy.Symbol('c_{}'.format(c)), sympy.Eq(p1, p2, evaluate=False))
            if solve:
                eq = sympy.Eq(eq, p3.n(N), evaluate=False)
            r += output.sympyeqn(eq) + '\n\n'

        r += '### Simplified combined uncertainty:\n\n'
        uformula = symout['uncertainty']  # Simplified formula
        uformula = sympy.Eq(uname, uformula, evaluate=False)
        if solve:
            uformula = sympy.Eq(uformula, sympy.sympify(symout['unc_val']).n(N), evaluate=False)
        r += output.sympyeqn(uformula) + '\n\n'

        r += '### Effective degrees of freedom:\n\n'
        dformula = sympy.Eq(sympy.Symbol('nu_eff'), symout['degf'], evaluate=False)
        if solve:
            dformula = sympy.Eq(dformula, sympy.sympify(symout['degf_val']).n(5), evaluate=False)
        r += output.sympyeqn(dformula) + '\n\n'
        return r

    def get_pdf(self, x, idx=0, **kwargs):
        ''' Get probability density function for this output.

            Parameters
            ----------
            x: float or array
                X-values at which to compute PDF
            idx: int, optional
                Index into means/uncerts if multiple (use with CurveFitOutput only)
        '''
        return stats.norm.pdf(x, loc=self.mean[idx], scale=self.uncert[idx])

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
        pidx = kwargs.pop('pidx', list(range(len(self.mean))))
        pidx = np.atleast_1d(pidx)
        stdevs = kwargs.pop('stddevs', 4)
        fig = kwargs.pop('fig', None)
        intervals = kwargs.pop('intervalsGUM', [])
        intervalt = kwargs.pop('intTypeGUMt', True)  # Use student-t (vs. k-value) to compute interval
        kwargs.pop('intTypeMC', None)  # not used in gum pdf
        kwargs.pop('intervals', None)  # not used by gum (only MC)
        covcolors = kwargs.pop('covcolors', ['C4', 'C5', 'C6', 'C7', 'C8', 'C9'])
        kwargs.setdefault('label', self.method)

        if ax is None and fig is None:
            fig = plt.gcf()
            if len(fig.axes) < len(pidx):
                ax = [fig.add_subplot(1, len(pidx), i+1) for i in range(len(pidx))]
            else:
                ax = fig.axes
        ax = np.atleast_1d(ax)
        assert len(ax) >= len(pidx)

        for p, axis in zip(pidx, ax):
            x = np.linspace(self.mean[p] - stdevs*self.uncert[p], self.mean[p]+stdevs*self.uncert[p], num=100)
            axis.plot(x, self.get_pdf(x, idx=p), **kwargs)
            if self.pnames is not None:
                axis.set_xlabel(self.pnames[p])

            if intervals:
                ilines, ilabels = [], []
                for covidx, cov in enumerate(intervals):
                    u, k = self.expanded(cov, normal=not intervalt)
                    for p, axis in zip(pidx, ax):
                        iline = axis.axvline(self.mean[p] + u, color=covcolors[covidx], ls='--')
                        axis.axvline(self.mean[p] - u, color=covcolors[covidx], ls='--')
                    ilines.append(iline)
                    ilabels.append(cov)

                intervallegend = axis.legend(ilines, ilabels, fontsize=10, loc='center right', title='GUM\nIntervals')
                axis.add_artist(intervallegend)

    def expanded(self, cov=0.95, normal=False):
        ''' Calculate expanded uncertainty with coverage interval based on degf.

            Parameters
            ----------
            cov: float
                Coverage interval value, from 0-1
            normal: bool
                Calculate using normal distribution, ignoring degrees of freedom

            Returns
            -------
            unc: float
                Expanded uncertainty
            k: float
                k-value associated with this expanded uncertainty
        '''
        if isinstance(cov, str):
            if '%' in cov:
                cov = cov.strip('%')
                cov = float(cov) / 100
            elif 'k' in cov:
                _, cov = cov.split('=')
                cov = float(cov.strip())
                return self.uncert * cov, cov

        if normal:
            k = stats.norm.ppf(1-(1-cov)/2)
        else:
            d = max(1, min(self.degf, 1E6))   # Inf will return garbage. Use big number.
            k = stats.t.ppf(1-(1-cov)/2, d)   # Use half of interval for scipy to get both tails
        return self.uncert * k, k


class BaseOutputMC(BaseOutput):
    ''' Base Output for Monte Carlo method. Differences include reporting symmetric coverage intervals and
        generating pdf from histogram of samples.
    '''
    def __init__(self, method, **kwargs):
        super(BaseOutputMC, self).__init__(method=method, **kwargs)
        self.samples = kwargs.get('samples')
        if self.samples.ndim == 1:
            self.samples = np.expand_dims(self.samples, 1)

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
        rows = len(self.mean)

        if ax1 is not None:
            fig = ax1.figure
            ax = [[ax1, ax2]]
            assert rows == 1
        else:
            if fig is None:
                fig = plt.gcf()
            fig.clf()
            fig.subplots_adjust(**dfltsubplots)
            ax = [[fig.add_subplot(rows, 2, 1+i*2), fig.add_subplot(rows, 2, 2+i*2)] for i in range(rows)]

        meancolor = kwargs.pop('meancolor', 'C0')
        unccolor = kwargs.pop('unccolor', 'C1')
        kwargs.setdefault('marker', 'o')

        for param in range(rows):
            title = self.properties['latex'] if 'latex' in self.properties else self.pnames[param]
            title = '$' + title + '$'
            Y = self.samples[:, param]
            step = len(Y)//div
            line = np.empty(div)
            steps = np.empty(div)
            for i in range(div):
                line[i] = Y[:step*(i+1)].mean()
                steps[i] = (i+1)*step
            if relative:
                line = line / line[-1]
            kwargs['color'] = meancolor
            ax[param][0].plot(steps, line, **kwargs)

            ax[param][0].set_title(title)
            if relative:
                ax[param][0].set_ylabel('Value (Relative to final)')
            else:
                ax[param][0].set_ylabel('Value')
            ax[param][0].set_xlabel('Samples')

            line = np.empty(div)
            steps = np.empty(div)
            for i in range(div):
                line[i] = Y[:step*(i+1)].std()
                steps[i] = (i+1)*step
            if relative:
                line = line / line[-1]
            kwargs['color'] = unccolor
            ax[param][1].plot(steps, line, **kwargs)
            ax[param][1].set_title(title)
            ax[param][1].set_xlabel('Samples')
            if relative:
                ax[param][1].set_ylabel('Uncertainty (Relative to final)')
            else:
                ax[param][1].set_ylabel('Uncertainty')
        fig.tight_layout()

    def get_pdf(self, x, idx=0, **kwargs):
        ''' Get probability density function for this output. Based on histogram
            of output samples.

            Parameters
            ----------
            x: float or array
                X-values at which to compute PDF
            idx: int, optional
                Index into means/uncerts if multiple (use with CurveFitOutput only)
            bins: int, default=100
                Number of bins for PDF histogram
        '''
        bins = max(3, kwargs.get('bins', 200))
        yy, xx = np.histogram(self.samples[:, idx], bins=bins, range=(x.min(), x.max()), density=True)
        xx = xx[:-1] + (xx[1]-xx[0])/2
        yy = np.interp(x, xx, yy)
        return yy

    def plot_pdf(self, ax=None, **kwargs):
        ''' Plot probability density function

            Parameters
            ----------
            ax: matplotlib axis
                Axis to plot on

            Keyword Arguments
            -----------------
            pidx: int
                Index of output parameter
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
        pidx = kwargs.pop('pidx', list(range(len(self.mean))))
        pidx = np.atleast_1d(pidx)
        kwargs.setdefault('label', self.method)
        intervals = kwargs.pop('intervals', [])
        intervalshortest = kwargs.pop('intTypeMC', 'Symmetric') == 'Shortest'
        covcolors = kwargs.pop('covcolors', ['C4', 'C5', 'C6', 'C7', 'C8', 'C9'])
        kwargs.pop('intTypeGUMt', None)   # Not used by MC
        kwargs.pop('intervalsGUM', None)  # Not used by MC

        if ax is None:
            fig = plt.gcf()
            if len(fig.axes) < len(pidx):
                ax = [fig.add_subplot(1, len(pidx), i+1) for i in range(len(pidx))]
            else:
                ax = fig.axes
        ax = np.atleast_1d(ax)
        assert len(ax) >= len(pidx)

        if hist:
            kwargs.setdefault('bins', 120)
            kwargs.setdefault('density', True)
            kwargs.setdefault('histtype', 'bar')
            kwargs.setdefault('ec', kwargs.get('color', 'C0'))
            kwargs.setdefault('linewidth', 2)

        for p, axis in zip(pidx, ax):
            x = np.linspace(self.mean[p]-stdevs*self.uncert[p], self.mean[p]+stdevs*self.uncert[p], num=100)
            xmin, xmax = x.min(), x.max()
            if hist and np.isfinite(self.samples[:, p]).any():
                if np.isfinite(xmin) and np.isfinite(xmax) and xmax > xmin:
                    kwargs['range'] = xmin, xmax
                axis.hist(self.samples[:, p], **kwargs)
            elif np.isfinite(xmin) and np.isfinite(xmax):
                y = self.get_pdf(x, idx=p, bins=kwargs.pop('bins', 200))
                axis.plot(x, y, **kwargs)

            if self.pnames is not None:
                axis.set_xlabel(self.pnames[p])

        if intervals:
            ilines, ilabels = [], []
            for covidx, cov in enumerate(intervals):
                mins, maxs, _ = self.expanded(cov, shortest=intervalshortest)
                for p, axis in zip(pidx, ax):
                    iline = axis.axvline(mins[p], color=covcolors[covidx])
                    axis.axvline(maxs[p], color=covcolors[covidx])
                ilines.append(iline)
                ilabels.append(cov)

            intervallegend = axis.legend(ilines, ilabels, fontsize=10, loc='upper right', title='Monte Carlo\nIntervals')
            axis.add_artist(intervallegend)

    def plot_normprob(self, ax=None, pidx=0, points=None, **kwargs):
        '''
        Plot a normal probability plot of the Monte-Carlo samples vs.
        expected quantile. If data falls on straight line, data is
        normally distributed.

        Parameters
        ----------
        ax: matplotlib axis (optional)
            Axis to plot on
        pidx: int
            Index of parameter to plot
        points: int
            Number of sample points to include for speed. Default is 100.
        '''
        if ax is None:
            ax = plt.gca()
        if points is None:
            points = min(100, len(self.samples))
        thin = len(self.samples)//points
        output.probplot(self.samples[::thin, pidx], ax=ax)

    def expanded(self, cov=0.95, shortest=False):
        ''' Calculate expanded uncertainty with coverage intervals based on degf.

            Parameters
            ----------
            cov: float or string
                Coverage interval fraction, 0-1 or string with percent, e.g. '95%'
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
            n = [np.nan]*self.samples.shape[1]
            return n, n, n

        if shortest:
            y = np.sort(self.samples, axis=0)
            N = y.shape[0]
            q = int(cov*N)

            # Shortest interval by looping
            quant = np.zeros((2, y.shape[1]))
            k = np.zeros(y.shape[1])
            for i in range(y.shape[1]):
                rmin = np.inf
                ridx = 0
                for r in range(N-q):
                    if y[r+q][i] - y[r][i] < rmin:
                        rmin = y[r+q][i] - y[r][i]
                        ridx = r
                quant[1][i] = y[ridx][i]
                quant[0][i] = y[ridx+q][i]
        else:
            q = [100-100*(1-cov)/2, 100*(1-cov)/2]  # Get cov and 1-cov quantiles
            quant = np.nanpercentile(self.samples, q, axis=0)
        k = (quant[0]-quant[1])/(2*self.uncert)
        return np.atleast_1d(quant[1]), np.atleast_1d(quant[0]), k

    def report_expanded(self, covlist=None, shortest=False, **kwargs):
        ''' Generate formatted report of expanded uncertainties

            Parameters
            ----------
            covlist: list of float
                Coverage intervals to include in report as fraction (i.e. 0.99 for 99%)

            Returns
            -------
            report: MDstring
        '''
        if covlist is None:
            covlist = [.99, .95, .90, .68]

        hdr = ['Interval']
        if self.pnames is not None:
            for p in self.pnames:
                hdr.extend(['min({})'.format(p), 'max({})'.format(p), 'k({})'.format(p)])
        else:
            hdr.extend(['Min', 'Max', 'k']*len(self.mean))

        rows = []
        for cov in covlist:
            mins, maxes, kvals = self.expanded(cov, shortest=shortest)
            if isinstance(cov, float):
                row = ['{:.2f}%'.format(cov*100)]
            else:
                row = [cov]
            for i, (mx, mn, k) in enumerate(zip(maxes, mins, kvals)):
                minstr = output.formatter.f(mn, matchto=self.uncert[i], **kwargs)
                maxstr = output.formatter.f(mx, matchto=self.uncert[i], **kwargs)
                row.append(minstr)
                row.append(maxstr)
                row.append(format(k, '.3f'))
            rows.append(row)
        r = output.MDstring('Shortest Coverage Intervals\n' if shortest else 'Symmetric Coverage Intervals\n')
        r += output.md_table(rows, hdr, **kwargs)
        return r


class FuncOutput(output.Output):
    ''' Class to hold output of all calculation methods (BaseOutputs list of GUM, MC, LSQ)
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
        self.pnames = self._baseoutputs[0].pnames
        self.inputfunc = inputfunc

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
                Confidence interval to compare (0-1 range)
            full: boolean
                Return full set of values, including delta, dlow and dhigh

            Returns
            -------
            valid: boolean array
                Validity of the GUM approximation compared to Monte-Carlo for each parameter
            delta: float array
                Allowable delta between GUM and MC, for each parameter
            dlow: float array
                Low value abs(ymean - uy_gum - ylow_mc)
            dhi: float array
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
        r = np.floor(np.log10(np.abs(self.gum.uncert))).astype(int) - (ndig-1)
        delta = 0.5 * 10.0**r
        dlow = abs(self.gum.mean - uc - ucmin)
        dhi = abs(self.gum.mean + uc - ucmax)
        if not full:
            return (dlow < delta) & (dhi < delta)
        else:
            a = (self.gum.uncert / 10.**r).astype(int)
            fullparams = {'delta': delta,
                          'dlow': dlow,
                          'dhi': dhi,
                          'r': r,
                          'a': a,
                          'gumlo': self.gum.mean - uc,
                          'gumhi': self.gum.mean + uc,
                          'mclo': ucmin,
                          'mchi': ucmax}
            return (dlow < delta) & (dhi < delta), fullparams

    def validate_gum_attained_conf(self, N=1000, conf=.95):
        ''' Validate GUM method using Monte-Carlo to estimate attained confidence
            level, as described by NIST.

            Recalculate the GUM N times changing the nominal value within each input
            distribution and calculating 95% coverage interval. 95% of the computed
            coverage intervals should contain the initial nominal value.

            Parameters
            ----------
            N: int
                Number of samples to calculate
            conf: float
                Confidence level to compare (0-1)

            Returns
            -------
            conf: float array
                Approximate attained confidence level for each parameter. Should
                be near 95% for the GUM approach to be valid.

            References
            ----------
            MSC Training Course, GUM-1, 2018.

            Notes
            -----
            Will add to GUI when a real published reference is found. Not sure this
            slow calculation adds anything over comparison to MC results in validate_gum().
        '''
        if self.pnames is not None:
            attainedconf = np.zeros(len(self.pnames))
        else:
            attainedconf = np.zeros(1)
        for param in range(len(attainedconf)):
            gummean = self.gum.mean[param]
            nom_vars = [v.nom for v in self.inputfunc.get_basevars()]
            variables = self.inputfunc.get_basevars()

            N = 1000
            noms = np.zeros(N)
            uncs = np.zeros(N)

            for i in range(N):
                for v, nom in zip(variables, nom_vars):
                    v.nom = nom
                    v.nom = v.sample(1)
                out = self.inputfunc.calc_GUM(calc_sym=False)
                noms[i] = out.mean[0]
                uncs[i], _ = out.expanded(conf)

            # Restore state
            for v, nom in zip(variables, nom_vars):
                v.nom = nom

            attainedconf[param] = 1 - np.count_nonzero((noms - uncs > gummean) | (noms + uncs < gummean)) / N
        return attainedconf

    def report(self, **kwargs):
        ''' Generate report of mean/uncertainty values.

            Keyword Arguments
            -----------------
            See NumFormatter()
        '''
        # rows are GUM, MC, LSQ, etc. Columns are means, uncertainties. Checks if baseoutput is CurveFit, and if so uses that format.
        hdr = ['Method']
        if self.pnames is None:
            hdr.extend(['Mean', 'Standard Uncertainty'])
        else:
            for p in self.pnames:
                hdr.extend(['{}'.format(p), 'u({})'.format(p)])
        rows = []
        for out in self._baseoutputs:
            row = [out.method]
            for c, s in zip(out.mean, out.uncert):
                row.extend([output.formatter.f(c, matchto=s, **kwargs), output.formatter.f(s, **kwargs)])
            rows.append(row)
        r = output.md_table(rows, hdr, **kwargs)
        return r

    def report_summary(self, **kwargs):
        ''' Generate report table with method, mean, stdunc, 95% range, and k.

            Keyword Arguments
            -----------------
            See NumFormatter()
        '''
        # rows are GUM, MC, LSQ, etc. Columns are means, uncertainties. Checks if baseoutput is CurveFit, and if so uses that format.
        conf = .95
        hdr = ['Method']
        if self.pnames is None:
            hdr.extend(['Mean', 'Std. Uncertainty', '95% Coverage', 'k', 'Deg. Freedom'])
        else:
            for p in self.pnames:
                hdr.extend(['{}'.format(p), 'u({})'.format(p), '{} Range'.format(p), '{} k'.format(p), '{} degf'.format(p)])
        rows = []
        for out in self._baseoutputs:
            row = [out.method]
            if out._method == 'mc':
                umins, umaxs, ks = out.expanded(conf)
                for c, s, umin, umax, k in zip(out.mean, out.uncert, umins, umaxs, ks):
                    rng95 = '[{}, {}]'.format(output.formatter.f(umin, matchto=s, **kwargs), output.formatter.f(umax, matchto=s, **kwargs))
                    row.extend([output.formatter.f(c, matchto=s, **kwargs), output.formatter.f(s, **kwargs), rng95, format(k, '.3f'), '-'])
            else:
                unc, k = out.expanded(conf)
                for c, s, uc, in zip(out.mean, out.uncert, unc):
                    rng95 = u'{} {}'.format(output.UPLUSMINUS, output.formatter.f(uc, **kwargs))
                    row.extend([output.formatter.f(c, matchto=s, **kwargs), output.formatter.f(s, **kwargs), rng95, format(k, '.3f'), format(out.degf, '.1f')])

            rows.append(row)
        r = output.md_table(rows, hdr, **kwargs)
        return r

    def report_warns(self, **kwargs):
        ''' Report of warnings raised during calculation

            Returns
            -------
            report: MDstring
        '''
        r = output.MDstring()
        for out in self._baseoutputs:
            r += out.report_warns(**kwargs)
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
            See NumFormatter()

            Returns
            -------
            report: MDstring
        '''
        if covlist is None:
            covlist = [.99, .95, .90, .68]
        if covlistgum is None:
            covlistgum = covlist

        r = output.MDstring()
        for out in self._baseoutputs:
            r += '### {}\n\n'.format(out.method)
            r += out.report_expanded(covlist, normal=normal, shortest=shortest, covlistgum=covlistgum, **kwargs)
        return r

    def report_derivation(self, solve=True, **kwargs):
        ''' Generate report of derivation of GUM method.

            Parameters
            ----------
            solve: bool
                Calculate values of each function.

            Keyword Arguments
            -----------------
            See NumFormatter()

            Returns
            -------
            report: MDstring
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
            See NumFormatter()
        '''
        cols = ['Variable', 'Component', 'Standard Uncertainty', 'Deg. Freedom', 'Description']
        rows = []
        for i, inpt in enumerate(self.inputfunc.get_basevars()):
            if len(inpt.uncerts) > 1:
                rows.append([output.sympyeqn(sympy.Symbol(inpt.name)), '',
                             inpt.desc, output.formatter.f(inpt.stdunc(), **kwargs), format(inpt.degf(), '.1f')])
                for u in inpt.uncerts:
                    rows.append(['', output.sympyeqn(sympy.Symbol(u.name)),
                                 u.desc if u.desc != '' else '--', output.formatter.f(u.std(), **kwargs), format(u.degf, '.1f')])
            elif len(inpt.uncerts) == 1:
                u = inpt.uncerts[0]
                rows.append([output.sympyeqn(sympy.Symbol(inpt.name)),
                             output.sympyeqn(sympy.Symbol(u.name)),
                             output.formatter.f(u.std(), **kwargs),
                             format(inpt.degf(), '.1f'),
                             u.desc if u.desc != '' else '--'])
        r = output.md_table(rows, hdr=cols, **kwargs)
        return r

    def report_inputs(self, **kwargs):
        ''' Report of input values.

            Keyword Arguments
            -----------------
            See NumFormatter()
        '''
        rows = []
        cols = ['Variable', 'Mean', 'Std. Uncertainty', 'Deg. Freedom', 'Description']
        for i in self.inputfunc.get_basevars():
            rows.append([output.sympyeqn(sympy.Symbol(i.name)),
                         output.formatter.f(i.mean(), matchto=i.stdunc(), **kwargs),
                         output.formatter.f(i.stdunc(), **kwargs),
                         output.formatter.f(i.degf(), **kwargs),
                         i.desc])
        r = output.md_table(rows, hdr=cols, **kwargs)
        return r

    def report_sens(self, **kwargs):
        ''' Report sensitivity coefficients '''
        r = output.MDstring()
        inames = [i.name for i in self.inputfunc.get_basevars()]

        if (hasattr(self, 'gum') and hasattr(self, 'mc') and self.gum.sensitivity is not None and self.gum.props is not None and self.mc.sensitivity is not None and self.mc.props is not None):
            # Both GUM and MC in same tables
            rows = [[output.sympyeqn(sympy.Symbol(name)), format(sGUM, '.3g'), format(pGUM, '.2f')+'%', format(sMC, '.3g'), format(pMC, '.2f')+'%']
                    for name, sGUM, pGUM, sMC, pMC in zip(inames, self.gum.sensitivity, self.gum.props, self.mc.sensitivity, self.mc.props)]
            if hasattr(self.gum, 'residprops') and self.gum.residprops != 0:
                rows.append(['Correlations', '', format(self.gum.residprops, '.2f')+'%', '', ''])
            r += output.md_table(rows, hdr=['Variable', 'GUM Sensitivity', 'GUM Proportion', 'MC Sensitivity', 'MC Proportion'], **kwargs)

        elif hasattr(self, 'gum') and self.gum.sensitivity is not None and self.gum.props is not None:
            rows = [[name, format(sGUM, '.3g'), format(pGUM, '.2f')+'%']
                    for name, sGUM, pGUM in zip(inames, self.gum.sensitivity, self.gum.props)]
            if hasattr(self, 'residprops') and self.gum.residprops != 0:
                rows.append(['Correlations', '', format(self.residprops, '.2f')+'%', '', ''])
            r += output.md_table(rows, hdr=['Variable', 'GUM Sensitivity', 'GUM Proportion'], **kwargs)

        elif hasattr(self, 'mc') and self.mc.sensitivity is not None and self.mc.props is not None:
            rows = [[name, format(sMC, '.3g'), format(pMC, '.2f')+'%']
                    for name, sMC, pMC in zip(inames, self.mc.sensitivity, self.mc.props)]
            r += output.md_table(rows, hdr=['Variable', 'MC Sensitivity', 'MC Proportion'], **kwargs)
        return r

    def report_validity(self, conf=.95, ndig=1, **kwargs):
        ''' Validate the GUM results by comparing the 95% coverage limits with Monte-Carlo
            results.

            Parameters
            ----------
            conf: float
                Confidence interval to use
            ndig: int
                Number of significant digits for comparison

            References
            ----------
            GUM Supplement 1, Section 8
            NPL Report DEM-ES-011, Chapter 8
        '''
        valid, params = self.validate_gum(ndig=ndig, conf=conf, full=True)
        valid = valid[0]  # NOTE: function currently only works for single-parameter output (ie not curve fits)
        for k in params.keys():
            params[k] = params[k][0]

        deltastr = str(params['delta'])
        r = '### Comparison to Monte Carlo {:.2f}% Coverage\n\n'.format(conf*100)
        r += u'{:d} significant digit{}. {} = {}.\n\n'.format(ndig, 's' if ndig > 1 else '', output.UDELTA, deltastr)

        rows = []
        hdr = ['{:.2f}% Coverage'.format(conf*100), 'Lower Limit', 'Upper Limit']
        rows.append(['GUM', output.formatter.f(params['gumlo'], matchto=params['dlow']), output.formatter.f(params['gumhi'], matchto=params['dhi'])])
        rows.append(['MC', output.formatter.f(params['mclo'], matchto=params['dlow']), output.formatter.f(params['mchi'], matchto=params['dhi'])])
        rows.append(['abs(GUM - MC)', output.formatter.f(params['dlow'], matchto=params['dlow']), output.formatter.f(params['dhi'], matchto=params['dhi'])])
        rows.append([u'abs(GUM - MC) < {}'.format(output.UDELTA), '<font color="green">PASS</font>' if params['dlow'] < params['delta'] else '<font color="red">FAIL</font>',
                                      '<font color="green">PASS</font>' if params['dhi'] < params['delta'] else '<font color="red">FAIL</font>'])
        r += output.md_table(rows, hdr=hdr, **kwargs)
        return output.MDstring(r)

    def plot_pdf(self, ax=None, **kwargs):
        ''' Plot probability density function of all methods

            Parameters
            ----------
            ax: list of matplotlib axis
                Axis to plot on

            Keyword Arguments
            -----------------
            pidx: list of int
                Indexes of output parameters to include
            stddevs: float
                Number of standard deviations to include in x range
            label: string
                Label for curve on plot

            Returns
            -------
            ax: matplotlib axis
        '''
        pidx = kwargs.pop('pidx', list(range(len(self._baseoutputs[0].mean))))
        if ax is None:
            fig = plt.gcf()
            if len(fig.axes) < len(pidx):
                ax = output.axes_grid(len(pidx), fig)
            else:
                ax = fig.axes
        ax = np.atleast_1d(ax)

        for p in pidx:
            for b in range(len(self._baseoutputs)):
                kwargs['label'] = self._baseoutputs[b].method
                self._baseoutputs[b].plot_pdf(pidx=p, ax=ax[p], **kwargs)
            ax[p].legend(loc='best', fontsize=10)


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

    def get_dists(self, name=None):
        ''' Get distributions in this output. If name is none, return a list of
            available distribution names. Only returns first parameter if multi-param
            (i.e. curve fit) output.
        '''
        names = []
        for f in self.foutputs:
            for method in f._baseoutputs:
                names.append('{} ({})'.format(f.name, method._method.upper()))

        if name is None:
            return names

        elif name in names:
            fname, method = name.split()
            method = method[1:-1].lower()
            baseout = self.get_output(fname, method=method)
            if 'mc' in method:
                samples = baseout.properties['samples']
                return samples

            else:  # (GUM), (LSQ)
                mean = baseout.mean[0]
                std = baseout.uncert[0]
                degf = baseout.degf
                return {'mean': mean, 'std': std, 'df': degf}

        else:
            raise ValueError('{} not found in output'.format(name))
        return names

    def report(self, **kwargs):
        ''' Generate report of all mean/uncerts for each function.

            Keyword Arguments
            -----------------
            See NumFormatter()
        '''
        r = output.MDstring()
        for f in self.foutputs:
            if f.inputfunc.show:
                r += output.sympyeqn(f.inputfunc.full_func()) + '\n\n'
                r += f.report(**kwargs)
        return r

    def report_summary(self, **kwargs):
        ''' Generate report of all mean/uncerts for each function.

            Keyword Arguments
            -----------------
            See NumFormatter()
        '''
        r = self.report_summary_table(**kwargs)
        r += self.report_summary_plot(**kwargs)
        return r

    def report_summary_table(self, **kwargs):
        ''' Generate summary table

            Keyword Arguments
            -----------------
            See NumFormatter()
        '''
        r = output.MDstring()
        for f in self.foutputs:
            if f.inputfunc.show:
                r += output.sympyeqn(f.inputfunc.full_func()) + '\n'
                r += f.report_summary(**kwargs)
        return r

    def report_summary_plot(self, **kwargs):
        ''' Generate summary plot

            Keyword Arguments
            -----------------
            See NumFormatter()
        '''
        with mpl.style.context(output.mplcontext):
            plt.ioff()
            fig = plt.figure()
            self.plot_pdf(fig=fig)
        r = output.MDstring()
        r.add_fig(fig)
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
        with mpl.style.context(output.mplcontext):
            plt.ioff()
            r = output.MDstring()
            if kwargs.get('summary', True):
                r += '## Summary\n\n'
                r += self.report_summary_table(**kwargs)
            if kwargs.get('outputs', True):
                fig = plt.figure()
                params = kwargs.get('outplotparams', {})
                joint = params.get('joint', False)
                if joint:
                    self.plot_outputscatter(fig=fig, **params.get('plotargs', {}))
                else:
                    self.plot_pdf(fig=fig, **params.get('plotargs', {}))
                r.add_fig(fig)
            if kwargs.get('inputs', True):
                r += '## Standardized Input Values\n\n'
                r += self.report_inputs(**kwargs)
                r += '---\n\n'
            if kwargs.get('components', True):
                r += '## Uncertainty Components\n\n'
                r += self.report_components(**kwargs)
                r += '---\n\n'
            if kwargs.get('sens', True):
                r += '## Sensitivity Coefficients\n\n'
                r += self.report_sens(**kwargs)
                r += '---\n\n'
            if kwargs.get('expanded', True):
                params = kwargs.get('expandedparams', {'intervalsgum': None, 'intervalsmc': None, 'norm': False, 'shortest': False})
                r += '## Expanded Uncertainties\n\n'
                r += self.report_expanded(covlist=params.get('intervalsmc', [0.95]),
                                          normal=params.get('norm', False), shortest=params.get('shortest', False),
                                          covlistgum=params.get('intervalsgum', None), **kwargs)
            if kwargs.get('gumderv', True):
                solve = kwargs.get('gumvalues', False)
                r += '## GUM Derivation\n\n'
                r += self.report_derivation(solve=solve, **kwargs)
            if kwargs.get('gumvalid', True):
                ndig = kwargs.get('gumvaliddig', 2)
                r += '## GUM Validity\n\n'
                r += self.report_validity(ndig=ndig, **kwargs)
            if kwargs.get('mchist', True):
                params = kwargs.get('mchistparams', {})
                fig = plt.figure()
                plotparams = params.get('plotargs', {})
                if params.get('joint', False):
                    self.plot_xscatter(fig=fig, **plotparams)
                else:
                    self.plot_xhists(fig=fig, **plotparams)
                r += '## Monte Carlo Inputs\n\n'
                r.add_fig(fig)
            if kwargs.get('mcconv', True):
                relative = kwargs.get('mcconvnorm', False)
                fig = plt.figure()
                self.plot_converge(fig, relative=relative)
                r += '## Monte Carlo Convergence\n\n'
                r.add_fig(fig)
        return r

    def report_func(self, **kwargs):
        ''' Report a list of functions in the calculator '''
        r = output.MDstring()
        for f in self.foutputs:
            if f.inputfunc.show:
                r += output.sympyeqn(f.inputfunc.full_func()) + '\n\n'
        return r

    def report_warns(self, **kwargs):
        ''' Report warnings raised during calculation '''
        r = output.MDstring()
        if self.warns:   # Calculator-level warnings
            for w in self.warns:
                r += w

        for f in self.foutputs:
            r += f.report_warns(**kwargs)
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
            See NumFormatter()

            Returns
            -------
            report: MDstring
        '''
        if covlist is None:
            covlist = [.99, .95, .90, .68]
        if covlistgum is None:
            covlistgum = covlist

        r = output.MDstring()
        for f in self.foutputs:
            if f.inputfunc.show:
                if len(self.foutputs) > 1:
                    r += output.sympyeqn(f.inputfunc.full_func()) + '\n\n'
                r += f.report_expanded(covlist, normal=normal, shortest=shortest, covlistgum=covlistgum, **kwargs)
        return r

    def report_components(self, **kwargs):
        ''' Report the uncertainty components

            Keyword Arguments
            -----------------
            See NumFormatter()
        '''
        cols = ['Variable', 'Component', 'Description', 'Standard Uncertainty', 'Deg. Freedom']
        rows = []
        for i, inpt in enumerate(self.ucalc.get_baseinputs()):
            if len(inpt.uncerts) > 1:
                rows.append([output.sympyeqn(sympy.Symbol(inpt.name)), '',
                             inpt.desc, output.formatter.f(inpt.stdunc(), **kwargs), format(inpt.degf(), '.1f')])

                for u in inpt.uncerts:
                    rows.append(['', output.sympyeqn(sympy.Symbol(u.name)),
                                 u.desc if u.desc != '' else '--', output.formatter.f(u.std(), **kwargs), format(u.degf, '.1f')])
            elif len(inpt.uncerts) == 1:
                u = inpt.uncerts[0]
                rows.append([output.sympyeqn(sympy.Symbol(inpt.name)),
                             output.sympyeqn(sympy.Symbol(u.name)),
                             u.desc if u.desc != '' else '--',
                             output.formatter.f(u.std(), **kwargs),
                             format(inpt.degf(), '.1f')])
        r = output.md_table(rows, hdr=cols, **kwargs)
        return r

    def report_derivation(self, solve=True, **kwargs):
        ''' Generate report of derivation of GUM method.

            Parameters
            ----------
            solve: bool
                Calculate values of each function.

            Keyword Arguments
            -----------------
            See NumFormatter()

            Returns
            -------
            report: MDstring
        '''
        r = output.MDstring()
        for f in self.foutputs:
            if f.inputfunc.show:
                r += f.report_derivation(solve=solve, **kwargs)
                r += '---\n\n'
        return r

    def report_validity(self, conf=.95, ndig=1, **kwargs):
        ''' Validate the GUM results by comparing the 95% coverage limits with Monte-Carlo
            results.

            Parameters
            ----------
            conf: float
                Confidence interval to use
            ndig: int
                Number of significant digits for comparison

            References
            ----------
            GUM Supplement 1, Section 8
            NPL Report DEM-ES-011, Chapter 8
        '''
        r = output.MDstring()
        for f in self.foutputs:
            if f.inputfunc.show:
                if len(self.foutputs) > 1:
                    r += output.sympyeqn(f.inputfunc.full_func()) + '\n\n'
                r += f.report_validity(conf=conf, ndig=ndig, **kwargs)
                r += '---\n\n'
        return r

    def report_inputs(self, **kwargs):
        ''' Report of input values.

            Keyword Arguments
            -----------------
            See NumFormatter()
        '''
        rows = []
        cols = ['Variable', 'Mean', 'Std. Uncertainty', 'Deg. Freedom', 'Description']
        for i in self.ucalc.get_baseinputs():  # Merge all functions together
            rows.append([output.sympyeqn(sympy.Symbol(i.name)),
                         output.formatter.f(i.mean(), matchto=i.stdunc(), **kwargs),
                         output.formatter.f(i.stdunc(), **kwargs),
                         output.formatter.f(i.degf(), **kwargs),
                         i.desc if i.desc != '' else '--'])
        r = output.md_table(rows, hdr=cols, **kwargs)
        return r

    def report_allinputs(self, **kwargs):
        ''' Combined report of inputs, uncertainties, and sensitivity coefficients.

            Keyword Arguments
            -----------------
            See NumFormatter()
        '''
        r = output.MDstring('## Inputs\n\n')
        r += self.report_inputs(**kwargs)
        r += '\n---\n\n'
        r += '## Uncertainty Components\n\n'
        r += self.report_components(**kwargs)
        r += '\n---\n\n'
        r += '## Sensitivity Coefficients\n\n'
        r += self.report_sens(**kwargs)
        return r

    def report_sens(self, **kwargs):
        ''' Report sensitivity coefficients.

            Keyword Arguments
            -----------------
            See NumFormatter()
        '''
        r = output.MDstring()
        for f in self.foutputs:
            if f.inputfunc.show:
                if len(self.foutputs) > 1:
                    r += output.sympyeqn(f.inputfunc.full_func()) + '\n\n'
                r += f.report_sens(**kwargs)
        return r

    def plot_pdf(self, fig=None, **kwargs):
        ''' Plot output probability (histogram and/or PDF curve)

            Parameters
            ----------
            fig : matplotlib figure, optional
                Figure to plot on. Existing figure will be cleared. If omitted, a new figure/axis
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
        showmc = kwargs.pop('showmc', True)
        showgum = kwargs.pop('showgum', True)
        bins = max(3, kwargs.pop('bins', 120))
        contour = kwargs.pop('contour', False)
        labelmode = kwargs.pop('labelmode', 'name')  # or 'desc'
        showleg = kwargs.pop('legend', True)
        [kwargs.pop(k, None) for k in ['points', 'inpts', 'overlay', 'cmap', 'cmapmc', 'equal_scale']]  # Ignore these in pdf plot

        if fig is None:
            fig = plt.gcf()

        fig.clf()
        fig.subplots_adjust(**dfltsubplots)
        if len(funcs) == 0: return fig

        rows = int(np.ceil(len(funcs)/3))
        cols = int(np.ceil(len(funcs)/rows))

        for plotid, func in enumerate([self.foutputs[i] for i in funcs]):
            ax = fig.add_subplot(rows, cols, plotid+1)

            try:
                mean = func.mc.mean
                uncert = func.mc.uncert
            except AttributeError:
                mean = func.gum.mean
                uncert = func.gum.uncert

            xlow = mean - stds * uncert
            xhi = mean + stds * uncert
            if not np.isfinite(xlow):  # Can happen if uncert is Nan for some reason, like with no input variables
                xlow = func.mc.mean - 1
                xhi = func.mc.mean + 1

            if showmc:
                kwargs['color'] = mccolor
                func.mc.plot_pdf(ax=ax, hist=not contour, bins=bins, stddevs=stds, **kwargs)

            if showgum:
                kwargs['color'] = gumcolor
                func.gum.plot_pdf(ax=ax, stddevs=stds, **kwargs)

            if np.isfinite(xlow) and np.isfinite(xhi) and xhi != xlow:
                ax.set_xlim(xlow, xhi)

            ax.ticklabel_format(style='sci', axis='x', scilimits=(-4, 4), useOffset=False)
            ax.set_ylabel('Probability Density')
            if labelmode == 'desc' and func.mc.desc is not None:
                ax.set_xlabel(func.mc.desc)
            elif func.name:
                ax.set_xlabel('$' + sympy.latex(sympy.sympify(func.name)) + '$')

            if showleg:
                ax.legend(loc='upper left', fontsize=10)

        fig.tight_layout()

    def plot_outputscatter(self, fig=None, **kwargs):
        ''' Plot scatter plot of output functions (if more than one function in system)

            Parameters
            ----------
            fig: matplotlib figure
                Figure to plot on. Will be cleared.

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
        showgum = kwargs.get('showgum', True)
        showmc = kwargs.get('showmc', True)
        points = kwargs.get('points', -1)
        bins = max(3, kwargs.get('bins', 40))
        labelmode = kwargs.get('labelmode', 'name')
        overlay = kwargs.get('overlay', False)
        equal_scale = kwargs.get('equal_scale', False)
        cmap = kwargs.get('cmap', 'viridis')
        cmapmc = kwargs.get('cmapmc', 'viridis')
        overlay = overlay and showgum and showmc

        if fig is None:
            fig = plt.gcf()
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
                        axgum.contour(x, y, h, 10, cmap=plt.get_cmap(cmap))
                        axgum.locator_params(nbins=5)
                        if labelmode == 'desc':
                            axgum.set_xlabel(self.foutputs[f1].desc)
                            axgum.set_ylabel(self.foutputs[f2].desc)
                        else:
                            axgum.set_xlabel('$' + self.foutputs[f1].inputfunc.get_latex() + '$')
                            axgum.set_ylabel('$' + self.foutputs[f2].inputfunc.get_latex() + '$')
                        axgum.ticklabel_format(style='sci', axis='x', scilimits=(-4, 4), useOffset=False)
                        axgum.ticklabel_format(style='sci', axis='y', scilimits=(-4, 4), useOffset=False)

                if showmc:
                    axmc.locator_params(nbins=5)
                    x = self.foutputs[f1].mc.samples[:, 0]
                    y = self.foutputs[f2].mc.samples[:, 0]
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
                        try:
                            axmc.plot(x[:points], y[:points], marker='.', ls='', markersize=2, color=color, zorder=0)
                        except ValueError:  # Case where len(x) != len(y) when one output is constant
                            pass
                    if labelmode == 'desc':
                        axmc.set_xlabel(self.foutputs[f1].mc.desc)
                        axmc.set_ylabel(self.foutputs[f2].mc.desc)
                    else:
                        axmc.set_xlabel('$' + self.foutputs[f1].inputfunc.get_latex() + '$')
                        axmc.set_ylabel('$' + self.foutputs[f2].inputfunc.get_latex() + '$')
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

    def plot_xhists(self, fig=None, **kwargs):
        ''' Plot histograms for each input variable.

            Parameters
            ----------
            fig: matplotlib figure instance
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

        if fig is None:
            fig = plt.gcf()

        # fig.clf() doesn't reset subplots_adjust parameters that change on tight_layout.
        # Reset them here or we can get strange exceptions about negative width while
        # adding the new axes
        fig.clf()
        fig.subplots_adjust(**dfltsubplots)

        variables = self.ucalc.get_baseinputs()
        if inpts is not None:
            if len(inpts) == 0: return fig
            variables = [variables[i] for i in inpts]

        axs = output.axes_grid(len(variables), fig)
        for ax, inpt in zip(axs, variables):
            ax.hist(inpt.sampledvalues, bins=bins, density=True, color=color, ec=color, histtype='bar', label=inpt.name)
            ax.ticklabel_format(style='sci', axis='x', scilimits=(-4, 4), useOffset=False)
            ax.yaxis.set_visible(False)
            if labelmode == 'desc':
                ax.set_xlabel(inpt.desc)
            else:
                ax.set_xlabel('$' + inpt.get_latex() + '$')
        fig.tight_layout()

    def plot_xscatter(self, fig=None, **kwargs):
        ''' Plot input samples against each other to look for correlation.

            Parameters
            ----------
            fig: matplotlib figure instance
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

        if fig is None:
            fig = plt.gcf()
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

                x = variables[col].sampledvalues
                y = variables[row].sampledvalues
                if contour:
                    counts, ybins, xbins = np.histogram2d(y, x, bins=bins)
                    ax.contour(counts, 10, extent=[xbins.min(), xbins.max(), ybins.min(), ybins.max()], cmap=plt.get_cmap(cmapmc))
                else:
                    ax.plot(x[:points], y[:points], marker='.', ls='', markersize=2,
                            color=color, label='{} vs. {}'.format(variables[col].name, variables[row].name))

                if col == row+1:
                    if labelmode == 'desc' and variables[col].desc != '' and variables[row].desc != '':
                        ax.set_xlabel(variables[col].desc)
                        ax.set_ylabel(variables[row].desc)
                    else:
                        ax.set_xlabel('$' + variables[col].get_latex() + '$')
                        ax.set_ylabel('$' + variables[row].get_latex() + '$')
                else:
                    ax.xaxis.set_visible(False)
                    ax.yaxis.set_visible(False)
                ax.ticklabel_format(style='sci', axis='x', scilimits=(-4, 4), useOffset=False)
                ax.ticklabel_format(style='sci', axis='y', scilimits=(-4, 4), useOffset=False)

    def plot_converge(self, fig=None, **kwargs):
        ''' Plot Monte-Carlo convergence. Using the same sampled values, the mean
            and standard uncertainty will be recomputed at fixed intervals
            throughout the sample array.

            Parameters
            ----------
            fig: matplotlib figure instance, optional
                If omitted, a new figure will be created.

            Keyword Arguments
            -----------------
            div: int
                Number of points to plot
        '''
        rows = len(self.foutputs)
        if fig is None:
            fig = plt.gcf()
        else:
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
