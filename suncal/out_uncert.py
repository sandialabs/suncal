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
from . import unitmgr
from . import plotting

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


dfltsubplots = {'wspace': .1, 'hspace': .1, 'left': .05, 'right': .95, 'top': .95, 'bottom': .05}

_foutputgum = namedtuple('funcoutput', ['nom', 'uncert', 'degf', 'name', 'expr', 'latex', 'description'], defaults=[None]*7)
_foutputmc = namedtuple('funcoutput', ['nom', 'uncert', 'name', 'samples', 'expr', 'latex', 'description'], defaults=[None]*7)


class GUMOutput(output.Output):
    ''' Class to hold output of a GUM uncertainty calculation '''
    def __init__(self, model, inputs, symboliconly=False):
        self.model = model
        gumout = self.model.GUMcovariance(symboliconly=symboliconly)
        self._uncert = gumout.uncert
        self._nom = gumout.nom
        self._units = [n.units if hasattr(n, 'units') else unitmgr.dimensionless for n in self._nom]
        self._degf = gumout.degf
        self.descriptions = self.model.descriptions
        self.Uy = gumout.Uy
        self.Ux = gumout.Ux
        self.Cx = gumout.Cx
        self.warnings = gumout.warnings
        self.symbolic = gumout.symbolic
        self.names = self.model.outnames
        self.exprs = self.model.exprs
        self._symbols = self.model.symbols
        self._symbolslatex = [sympy.latex(s) for s in self._symbols]
        self.inputs = inputs
        self.nouts = len(self.names)
        self._props = None
        self._resids = None

    def _index(self, idx):
        return self.names.index(idx) if isinstance(idx, str) else idx

    def nom(self, idx=0):
        ''' Get nominal value of model function '''
        return self._nom[self._index(idx)]

    def uncert(self, idx=0):
        ''' Get standard uncertainty value of model function '''
        return self._uncert[self._index(idx)]

    def degf(self, idx=0):
        ''' Get degrees of freedom of model function '''
        return self._degf[self._index(idx)]

    def report(self, **kwargs):
        ''' Generate/return default report (same as repr)

            Keyword Arguments
            -----------------
            Arguments for Report object

            Returns
            -------
            report.Report
        '''
        hdr = ['Function', 'Nominal', 'Std. Uncertainty']
        rows = []
        for i in range(self.nouts):
            rows.append((report.Math(self.names[i]),
                         report.Number(self._nom[i], matchto=self._uncert[i]),
                         report.Number(self._uncert[i])))
        r = report.Report(**kwargs)
        r.table(rows, hdr=hdr)
        return r

    def get_pdf(self, x, idx=0, **kwargs):
        ''' Get probability density function for this output.

            Parameters
            ----------
            x: float or array
                X-values at which to compute PDF
        '''
        return stats.norm.pdf(x, loc=self.nom(idx).magnitude, scale=self.uncert(idx).magnitude)

    def get_distdef(self, fidx=0):
        ''' Get dictionary defining probability distribution of the function '''
        return {'mean': self.nom(fidx).magnitude,
                'std': self.uncert(fidx).magnitude,
                'df': self.degf(fidx)}

    def _plot_funcpdf(self, ax, fidx=0, **kwargs):
        ''' Plot PDF of one function in the model '''
        stdevs = kwargs.pop('stddevs', 4)
        kwargs.setdefault('color', 'C1')
        kwargs.setdefault('label', 'GUM Approximation')
        intervals = kwargs.pop('intervals', [])
        intervaltype = kwargs.pop('intervaltype', 't')  # 't' or 'k' - use student-t (vs. k-value) to compute interval
        covcolors = kwargs.pop('covcolors', ['C4', 'C5', 'C6', 'C7', 'C8', 'C9'])

        nom = self.nom(fidx).magnitude
        units = self.nom(fidx).units
        uncert = self.uncert(fidx).magnitude
        x = np.linspace(nom - stdevs*uncert, nom+stdevs*uncert, num=100)
        ax.plot(x, self.get_pdf(x, fidx), **kwargs)

        if self.names[fidx] is not None:
            ax.set_xlabel(report.Math(self.names[fidx]).latex())
        if units:
            unitstr = report.Unit(units).latex(bracket=True)

        if intervals:
            ilines, ilabels = [], []
            for covidx, cov in enumerate(intervals):
                u, k = self.expanded(fidx=fidx, cov=cov, normal=(intervaltype == 'k'))
                iline = ax.axvline(nom + u.magnitude, color=covcolors[covidx], ls='--')
                ax.axvline(nom - u.magnitude, color=covcolors[covidx], ls='--')
                ilines.append(iline)
                ilabels.append(cov)

            intervallegend = ax.legend(ilines, ilabels, fontsize=10, loc='center right', title='GUM Intervals')
            ax.add_artist(intervallegend)

        ax.set_xlabel(ax.get_xlabel() + unitstr)

    def plot_pdf(self, plot=None, funcidx=None, **kwargs):
        ''' Plot probability density function

            Parameters
            ----------
            plot: object
                Matplotlib Figure or Axis to plot on
            funcidx: None or list
                List of functions in the model to plot (each in its own axis)
            stdevs: float
                Number of standard deviations to include in x range
            intervals: list
                List of interval values (.99, .95, etc.) for vertical lines
            intervaltype: string
                Use 'gum' or 't' to compute GUM intervals
            covcolors: list
                Matplotlib color names for interval lines
            label: string
                Whether to label axes with 'name' or 'description'
            color: string
                Matplotlib color for PDF line

            Returns
            -------
            ax: matplotlib axis
        '''
        if funcidx is None:
            funcidx = np.arange(self.nouts)

        fig, _ = plotting.initplot(plot)
        fig.clf()
        fig.subplots_adjust(**dfltsubplots)
        axs = plotting.axes_grid(len(funcidx), fig, len(funcidx))

        for ax, fidx in zip(axs, funcidx):
            self._plot_funcpdf(ax, fidx, **kwargs)

        fig.tight_layout()

    def expanded(self, fidx=0, cov=0.95, **kwargs):
        ''' Calculate expanded uncertainty with coverage interval based on degf.

            Parameters
            ----------
            fidx: int
                Model function index
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
        Expanded = namedtuple('Expanded', ['uncertainty', 'k'])
        if isinstance(cov, str):
            if '%' in cov:
                cov = cov.strip('%')
                cov = float(cov) / 100
            elif 'k' in cov:
                _, cov = cov.split('=')
                k = float(cov.strip())
                return Expanded(self.uncert(fidx) * k, k)

        if kwargs.get('normal', False):
            k = stats.norm.ppf(1-(1-cov)/2)
        else:
            d = max(1, min(self.degf(fidx), 1E6))   # Inf will return garbage. Use big number.
            k = stats.t.ppf(1-(1-cov)/2, d)   # Use half of interval for scipy to get both tails
        return Expanded(k*self.uncert(fidx), k)

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
        hdr = ['Function', 'Interval', 'Min', 'Max', 'k', 'Deg. Freedom', 'Expanded Uncertainty']
        for fidx in range(self.nouts):
            if not self.model.show[fidx]: continue
            for cov in covlist:
                uncert, k = self.expanded(fidx=fidx, cov=cov, normal=normal)
                row = [report.Math.from_sympy(self._symbols[fidx])] if cov == covlist[0] else ['-']
                row.append('{:.2f}%'.format(cov*100) if not isinstance(cov, str) else cov)
                row.append(report.Number(self.nom(fidx)-uncert, **kwargs))
                row.append(report.Number(self.nom(fidx)+uncert, **kwargs))
                row.append(format(k, '.3f'))
                row.append(format(self.degf(fidx), '.2f'))
                row.append(report.Number(uncert, **kwargs))
                rows.append(row)
        r = report.Report(**kwargs)
        r.table(rows, hdr)
        return r

    def _calc_props(self):
        ''' Calculate proportions and residual '''
        inputuncert = [np.sqrt(self.Ux[i][i]) for i in range(len(self.Ux))]
        self._props = []
        self._resids = []
        for fidx in range(self.nouts):
            props = [(cx*u)**2/self.uncert(fidx)**2 for cx, u in zip(self.Cx[fidx], inputuncert)]
            # All units should be dimensionless
            props = np.array([p.to_reduced_units().magnitude for p in props])
            resid = 1 - sum(props)
            if abs(resid) < 1E-7: resid = 0
            self._props.append(props)
            self._resids.append(resid)

    @property
    def proportions(self):
        if self._props is None:
            self._calc_props()
        return self._props

    @property
    def residuals(self):
        if self._resids is None:
            self._calc_props()
        return self._resids

    def report_sens(self, **kwargs):
        ''' Report sensitivity coefficients and proportions '''
        r = report.Report(**kwargs)

        for fidx in range(self.nouts):
            if not self.model.show[fidx]: continue
            props = self.proportions[fidx]
            resid = self.residuals[fidx]

            rows = []
            for i, inpt in enumerate(self.inputs):
                rows.append([report.Math.from_latex(inpt.get_latex()),
                             report.Number(self.Cx[fidx][i], fmin=1),
                             format(props[i]*100, '.2f')+'%'])
                if resid > 0:
                    rows.append(['Correlations', '', format(self.residprops, '.2f')+'%'])

            if self.nouts > 1:
                r.hdr(report.Math(self.names[fidx]), level=3)
            r.table(rows, hdr=['Variable', 'Sensitivity', 'Proportion'])
        return r

    def correlation(self, fidx1, fidx2):
        ''' Get correlation coefficient between function 1 and function '''
        if fidx1 == fidx2:
            return 1.0
        fidx1 = self._index(fidx1)
        fidx2 = self._index(fidx2)
        cov = self.Uy[fidx1][fidx2]
        corr = (cov / (self._uncert[fidx1] * self._uncert[fidx2])).to('dimensionless').magnitude  # dimensionless
        return corr

    def report_correlation(self, **kwargs):
        ''' Report correlation matrix of outputs. Returns blank report
            for single-output calculations

            Keyword Arguments
            -----------------
            Passed to report.Report
        '''
        r = report.Report()
        if self.nouts < 2:
            return r

        hdr = ['-'] + [report.Math(n) for n in self.names]
        rows = []
        for idx1 in range(self.nouts):
            row = [report.Math(self.names[idx1])]
            for idx2 in range(self.nouts):
                try:
                    corr = self.correlation(idx1, idx2)
                except IndexError:  # Constant measurement models can end up here
                    corr = np.nan
                row.append(format(corr, '.3f'))
            rows.append(row)
        r.hdr('Correlation Coefficients (GUM)', level=3)
        r.table(rows, hdr)
        return r

    def report_func(self, **kwargs):
        ''' List all the functions in the model in a report '''
        rpt = report.Report(**kwargs)
        for fidx in range(self.nouts):
            if not self.model.show[fidx]: continue
            rpt.sympy(self.model.expr_symbols[fidx], end='\n\n')
        return rpt

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
        rpt = report.Report(**kwargs)

        if self.symbolic is None:
            rpt.txt('No symbolic solution computed for function.')
            return rpt

        if len(self.Ux) == 0:
            rpt.txt('No input variables defined in model.')
            return rpt

        def combine(expr, val):
            ''' Helper function for combining sympy expression and = value using sympy.Eq, and handling units '''
            units = None
            with suppress(AttributeError):
                val, units = val.magnitude, val.units

            if solve:
                symexp = sympy.Eq(expr, sympy.N(val, N), evaluate=False)
                return report.Math.from_sympy(symexp, units)
            else:
                return report.Math.from_sympy(expr)

        rpt.hdr('Measurement Model:', level=3)
        for fidx in range(self.nouts):
            if not self.model.show[fidx]: continue
            rpt.sympy(self.model.expr_symbols[fidx], end='\n\n')

        multiout = len(self.model.exprs) > 1

        # Inputs
        innames = self.inputs.symbols
        if multiout or len(self.inputs.corr_list) > 0:
            if multiout:
                rpt.hdr('Input Covariance Matrix [Ux]:', level=3)
            else:
                rpt.hdr('Input Covariance Matrix:', level=3)

            hdr = ['-'] + innames
            covsym = self.inputs.covariance_sym()
            cov = self.inputs.covariance()
            rows = []
            for i in range(len(innames)):
                row = [innames[i]]
                for j in range(len(innames)):
                    row.append(report.Math.from_sympy(covsym[i][j]))
                rows.append(row)
            rpt.table(rows, hdr=hdr)

            if solve:
                hdr = ['-'] + innames
                covsym = self.inputs.covariance_sym()
                cov = self.inputs.covariance()
                rows = []
                for i in range(len(innames)):
                    row = [innames[i]]
                    for j in range(len(innames)):
                        row.append(report.Number(cov[i][j]))
                    rows.append(row)
                rpt.table(rows, hdr=hdr)

        else:
            rpt.hdr('Measured Values:', level=3)
            rows = []
            insymbols = self.inputs.symbols
            inuncsymbols = self.inputs.unc_symbols
            for i, inpt in enumerate(self.inputs):
                rows.append([combine(insymbols[i], inpt.nom),
                             combine(inuncsymbols[i], inpt.stdunc()),
                             combine(sympy.Symbol('nu_{}'.format(str(innames[i]))), inpt.degf())])
            rpt.table(rows, hdr=['Variable', 'Std. Uncertainty', 'Deg. Freedom'])

        fnames = self.model.symbols
        if multiout:
            rpt.hdr('Sensitivity Matrix [Cx]:', level=3)
            rows = []
            hdr = ['-'] + [report.Math.from_sympy(n) for n in innames]
            for fidx in range(self.nouts):
                if not self.model.show[fidx]: continue
                row = [report.Math.from_sympy(fnames[fidx])]
                for i in range(len(self.inputs)):
                    row.append(report.Math.from_sympy(sympy.Derivative(fnames[fidx], innames[i])))
                rows.append(row)
            rpt.table(rows, hdr)

            # Substituted sensitivity (symbolic)
            rows = []
            hdr = ['-'] + [report.Math.from_sympy(n) for n in innames]
            for fidx in range(self.nouts):
                if not self.model.show[fidx]: continue
                row = [report.Math.from_sympy(fnames[fidx])]
                for i in range(len(self.inputs)):
                    row.append(self.symbolic.Cx[fidx][i])
                rows.append(row)
            rpt.table(rows, hdr)

            if solve:
                # Substituted sensitivity (numeric)
                rows = []
                hdr = ['-'] + [report.Math.from_sympy(n) for n in innames]
                for fidx in range(self.nouts):
                    if not self.model.show[fidx]: continue
                    row = [report.Math.from_sympy(fnames[fidx])]
                    for i in range(len(self.inputs)):
                        row.append(report.Number(self.Cx[fidx][i]))
                    rows.append(row)
                rpt.table(rows, hdr)
        else:  # Single output function
            rpt.hdr('Sensitivity Coefficients:', level=3)
            for i in range(len(self.inputs)):
                cx = report.Math.from_sympy(sympy.Eq(sympy.Derivative(fnames[0], innames[i]), self.symbolic.Cx[0][i]))
                if not solve:
                    rpt.add(cx, end='\n\n')
                else:
                    rpt.add(cx, end='  ')
                    if hasattr(self.Cx[0][i], 'units'):
                        rpt.add(report.Math.from_latex(' = ' + format(self.Cx[0][i].magnitude, '.{}f'.format(N))), end=' ')
                        rpt.add(report.Unit(self.Cx[0][i].units), end='\n\n')
                    else:
                        rpt.add(report.Math.from_latex(' = ' + format(self.Cx[0][i], '.{}f'.format(N))), end='\n\n')

        # Output Uncertainty
        outuncnames = self.model.unc_symbols
        inuncnames = self.inputs.unc_symbols
        if multiout:
            rpt.hdr('Combined Covariance:', level=3)
            rpt.add(report.Math.from_latex(r'U_y = C_x \cdot U_x \cdot C_x^T'), end='\n\n')

            rpt.hdr('Uncertainties:', level=4)
            for fidx in range(self.nouts):
                if not self.model.show[fidx]: continue
                terms = [sympy.Derivative(fnames[fidx], innames[i])**2 * inuncnames[i]**2 for i in range(len(self.inputs))]

                for i in range(len(self.inputs)):
                    for j in range(i+1, len(self.inputs)):
                        if self.Ux[i][j] != 0:
                            corsymbol = sympy.Symbol('sigma_{}{}'.format(innames[i], innames[j]))
                            terms.append(2 * sympy.Derivative(fnames[fidx], innames[i]) *
                                         sympy.Derivative(fnames[fidx], innames[j]) *
                                         inuncnames[i] * inuncnames[j] * corsymbol)

                uexp = sympy.sqrt(sum(terms))
                rpt.sympy(sympy.Eq(outuncnames[fidx], uexp), end='\n\n')

            # Remove unused sigma=0 terms from Uy
            Uy = self.symbolic.Uy
            corrvals = self.inputs.corr_values()
            corrvals = {k: v for k, v in corrvals.items() if v == 0}
            uy0 = []
            for i in range(len(Uy)):
                uy0.append(sympy.sqrt(Uy[i][i].subs(corrvals)).simplify())

            rpt.hdr('Simplified:', level=4)
            for fidx in range(self.nouts):
                if not self.model.show[fidx]: continue
                rpt.add(combine(sympy.Eq(outuncnames[fidx], uy0[fidx]), self._uncert[fidx]), end='\n\n')
        else:  # Single output
            rpt.hdr('Combined Uncertainty:', level=3)
            terms = [sympy.Derivative(fnames[0], innames[i])**2 * inuncnames[i]**2 for i in range(len(self.inputs))]
            for i in range(len(self.inputs)):
                for j in range(i+1, len(self.inputs)):
                    if self.Ux[0][j] != 0:
                        corsymbol = sympy.Symbol('sigma_{}{}'.format(innames[i], innames[j]))
                        terms.append(2 * sympy.Derivative(fnames[0], innames[i]) *
                                     sympy.Derivative(fnames[0], innames[j]) *
                                     inuncnames[i] * inuncnames[j] * corsymbol)

            uexp = sympy.sqrt(sum(terms))
            rpt.sympy(sympy.Eq(outuncnames[0], uexp), end='\n\n') 
            corrvals = self.inputs.corr_values()
            corrvals = {k: v for k, v in corrvals.items() if v == 0}
            uy0 = sympy.sqrt(self.symbolic.Uy[0][0].subs(corrvals)).simplify()
            rpt.add(combine(sympy.Eq(outuncnames[0], uy0), self._uncert[0]), end='\n\n')

        # Degrees freedom
        rpt.hdr('Effective degrees of freedom:', level=3)
        for fidx in range(self.nouts):
            if not self.model.show[fidx]: continue
            numerator = outuncnames[fidx]**4
            denom = [self.symbolic.Cx[fidx][i]**4 * inuncnames[i]**4 / sympy.Symbol('nu_{}'.format(str(innames[i])))
                     for i in range(len(self.inputs))]

            rpt.add(combine(sympy.Eq(sympy.Symbol('nu_{}'.format(self.names[fidx])), numerator/sum(denom)), self._degf[fidx]), end='\n\n')
        return rpt

    def plot_correlation(self, ax=None, fidx1=0, fidx2=1, **kwargs):
        ''' Plot correlation (uncertainty region) between two model functions '''
        _, ax = plotting.initplot(ax)

        cmap = kwargs.get('cmap', 'viridis')
        labelmode = kwargs.get('labelmode', 'name')
        showleg = kwargs.get('legend', True)
        levels = None

        try:
            x, y, h = _contour(self._nom, self.Uy, fidx1, fidx2)
        except IndexError:  # no variables in model
            h = None
        if h is not None:
            levels = np.linspace(h.min(), h.max(), 11)[1:]
            ax.contour(x, y, h, levels, cmap=plt.get_cmap(cmap))
            ax.locator_params(nbins=5)
            if labelmode == 'desc':
                ax.set_xlabel(self.descriptions[fidx1])
                ax.set_ylabel(self.descriptions[fidx2])
            else:
                ax.set_xlabel('$' + self._symbolslatex[fidx1] + '$' + report.Unit(self._units[fidx1]).latex(bracket=True))
                ax.set_ylabel('$' + self._symbolslatex[fidx2] + '$' + report.Unit(self._units[fidx2]).latex(bracket=True))
            ax.ticklabel_format(style='sci', axis='x', scilimits=(-4, 4), useOffset=False)
            ax.ticklabel_format(style='sci', axis='y', scilimits=(-4, 4), useOffset=False)
            if showleg:
                cax = make_axes_locatable(ax).append_axes('right', size='5%', pad=0.05)
                sm = plt.cm.ScalarMappable(cmap=plt.get_cmap(cmap), norm=plt.Normalize(vmin=levels.min(), vmax=levels.max()))
                ax.figure.colorbar(sm, cax=cax, orientation='vertical')
        return levels


def _contour(means, covariance, fidx1, fidx2):
    ''' Generate x, y, z contours for plotting correlation region.
    '''
    Contour = namedtuple('Contour', ['x', 'y', 'pdf'])

    m0 = means[fidx1].magnitude
    m1 = means[fidx2].magnitude
    u0 = np.sqrt(covariance[fidx1][fidx1]).magnitude
    u1 = np.sqrt(covariance[fidx2][fidx2]).magnitude
    Uy = [[covariance[fidx1][fidx1].magnitude, covariance[fidx1][fidx2].magnitude],
          [covariance[fidx2][fidx1].magnitude, covariance[fidx2][fidx2].magnitude]]

    try:
        rv = stats.multivariate_normal(np.array([m0, m1]), cov=Uy)
    except (ValueError, np.linalg.LinAlgError):
        return Contour(None, None, None)

    x, y = np.meshgrid(np.linspace(m0-3*u0, m0+3*u0), np.linspace(m1-3*u1, m1+3*u1))
    pos = np.dstack((x, y))
    return Contour(x, y, rv.pdf(pos))


class MCOutput(output.Output):
    ''' Class to hold output of a Monte Carlo uncertainty calculation '''
    def __init__(self, model, inputs):
        self.model = model
        self._uncert, self._nom, self._samples, self.warnings = model.MCsample()
        self._units = [n.units for n in self._nom]
        self.names = self.model.outnames
        self.exprs = self.model.exprs
        self._symbols = self.model.symbols
        self._symbolslatex = [sympy.latex(s) for s in self._symbols]
        self.descriptions = self.model.descriptions
        self.inputs = inputs
        self._Cx = None  # Use property generate when asked for since it takes time to calc
        self._props = None
        self.nouts = len(self.names)

    def _index(self, idx):
        return self.names.index(idx) if isinstance(idx, str) else idx

    def nom(self, idx=0):
        ''' Get nominal value of model function '''
        return self._nom[self._index(idx)]

    def uncert(self, idx=0):
        ''' Get standard uncertainty value of model function '''
        return self._uncert[self._index(idx)]

    def samples(self, idx=0):
        ''' Get array of sampled values for one model function '''
        if not isinstance(idx, str):
            idx = self.names[idx]
        return self._samples[idx]

    @property
    def Ux(self):
        ''' Input covariance '''
        return self.inputs.covariance()

    @property
    def Cx(self):
        ''' Sensitivity matrix '''
        # Compute MC sensitivity on-demand since it can add a lot of time
        if self._Cx is None:
            self._Cx, self._props = self.model.MCsensitivity()
        return self._Cx

    @property
    def proportions(self):
        ''' Uncertainty contribution proportions '''
        # Compute sensitivity coefficients and proportions on-demand since it's slow
        if self._props is None:
            self._Cx, self._props = self.model.MCsensitivity()
        return self._props

    def report(self, **kwargs):
        ''' Generate/return default report (same as repr)

            Keyword Arguments
            -----------------
            Arguments for Report object

            Returns
            -------
            report.Report
        '''
        hdr = ['Function', 'Nominal', 'Std. Uncertainty']
        rows = []
        for i in range(self.nouts):
            if self.model.show[i]:
                rows.append((report.Math(self.names[i]),
                             report.Number(self._nom[i], matchto=self._uncert[i]),
                             report.Number(self._uncert[i])))
        r = report.Report(**kwargs)
        r.table(rows, hdr=hdr)
        return r

    def get_pdf(self, x, fidx=0, **kwargs):
        ''' Get probability density function for this output. Based on histogram
            of output samples.

            Parameters
            ----------
            x: float or array
                X-values at which to compute PDF
            bins: int, default=100
                Number of bins for PDF histogram
        '''
        name = self.names[self._index(fidx)]
        bins = max(3, kwargs.get('bins', 200))
        yy, xx = np.histogram(self.samples(name).magnitude, bins=bins, range=(x.min(), x.max()), density=True)
        xx = xx[:-1] + (xx[1]-xx[0])/2
        yy = np.interp(x, xx, yy)
        return yy

    def get_distdef(self, fidx=0):
        ''' Get dictionary defining probability distribution of the function '''
        fname = self.names[self._index(fidx)]
        samples = self.samples(fidx).magnitude
        return {'samples': samples,
                'median': np.median(samples),
                'expected': self.model.eval()[fname].magnitude}

    def _plot_funcpdf(self, ax, fidx=0, hist=True, **kwargs):
        ''' Plot PDF of one function in the model on the axis '''
        stdevs = kwargs.pop('stddevs', 4)
        intervals = kwargs.pop('intervals', [])
        intervaltype = kwargs.pop('intervaltype', 'symmetric')  # symmetric or shortest
        covcolors = kwargs.pop('covcolors', ['C4', 'C5', 'C6', 'C7', 'C8', 'C9'])

        if hist:
            kwargs.setdefault('bins', 120)
            kwargs.setdefault('density', True)
            kwargs.setdefault('histtype', 'bar')
            kwargs.setdefault('ec', kwargs.get('color', 'C0'))
            kwargs.setdefault('fc', kwargs.get('color', 'C0'))
            kwargs.setdefault('linewidth', 2)

        nom = self.nom(fidx).magnitude
        units = self.nom(fidx).units
        uncert = self.uncert(fidx).magnitude
        x = np.linspace(nom-stdevs*uncert, nom+stdevs*uncert, num=100)
        xmin, xmax = x.min(), x.max()
        name = self.names[self._index(fidx)]

        kwargs.setdefault('label', 'Monte Carlo')
        if hist and np.isfinite(self._samples[name].magnitude).any():
            if np.isfinite(xmin) and np.isfinite(xmax) and xmax > xmin:
                kwargs['range'] = xmin, xmax
            ax.hist(self._samples[name].magnitude, **kwargs)
        elif np.isfinite(xmin) and np.isfinite(xmax):
            y = self.get_pdf(x, fidx=fidx, bins=kwargs.pop('bins', 200))
            ax.plot(x, y, **kwargs)

        if name is not None:
            ax.set_xlabel(name)
        if units:
            unitstr = report.Unit(units).latex(bracket=True)
            ax.set_xlabel(ax.get_xlabel() + unitstr)

        if intervals:
            ilines, ilabels = [], []
            for covidx, cov in enumerate(intervals):
                mins, maxs, _ = self.expanded(fidx=fidx, cov=cov, shortest=(intervaltype.lower() == 'shortest'))
                iline = ax.axvline(mins.magnitude, color=covcolors[covidx])
                ax.axvline(maxs.magnitude, color=covcolors[covidx])
                ilines.append(iline)
                ilabels.append(cov)

            intervallegend = ax.legend(ilines, ilabels, fontsize=10, loc='upper right', title='Monte Carlo\nIntervals')
            ax.add_artist(intervallegend)

    def plot_pdf(self, plot=None, funcidx=None, hist=True, **kwargs):
        ''' Plot probability density function

            Parameters
            ----------
            plot: object
                Matplotlib Figure or Axis to plot on
            funcidx: None or list
                List of functions in the model to plot (each in its own axis)
            stdevs: float
                Number of standard deviations to include in x range
            intervals: list
                List of interval values (.99, .95, etc.) for vertical lines
            covcolors: list
                Matplotlib color names for interval lines
            label: string
                Whether to label axes with 'name' or 'description'
            color: string
                Matplotlib color for PDF line

            Returns
            -------
            ax: matplotlib axis
        '''
        if funcidx is None:
            funcidx = [i for i in range(self.nouts) if self.model.show[i]]

        fig, _ = plotting.initplot(plot)
        fig.clf()
        fig.subplots_adjust(**dfltsubplots)
        axs = plotting.axes_grid(len(funcidx), fig, len(funcidx))
        for fidx, ax in zip(funcidx, axs):
            self._plot_funcpdf(ax=ax, fidx=fidx, hist=hist, **kwargs)
        fig.tight_layout()

    def expanded(self, fidx=0, cov=0.95, **kwargs):
        ''' Calculate expanded uncertainty with coverage intervals based on degf.

            Parameters
            ----------
            fidx: int
                Function index in model
            cov: float or string
                Coverage interval fraction, 0-1 or string with percent, e.g. '95%'

            Keyword Arguments
            -----------------
            shortest: boolean
                Use shortest interval instead of symmetric interval.

            Returns
            -------
            umin: Minimum uncertainty value of coverage range
            umax: Maximum uncertainty value of coverage range
            k: k-value associated with this expanded uncertainty
        '''
        Expanded = namedtuple('Expanded', ['minimum', 'maximum', 'k'])

        # Generate percentiles from interval list
        if isinstance(cov, str):
            cov = cov.strip('%')
            cov = float(cov) / 100
        assert cov >= 0 and cov <= 1
        if len(self._samples) == 0:
            return Expanded(np.nan, np.nan, np.nan)

        fidx = self._index(fidx)
        fname = self.names[fidx]
        if kwargs.get('shortest', False):
            y = np.sort(self._samples[fname].magnitude)
            q = int(cov*len(y))  # constant number of points in coverage range
            # Shortest interval by looping
            rmin = np.inf
            ridx = 0
            for r in range(len(y)-q):
                if y[r+q] - y[r] < rmin:
                    rmin = y[r+q] - y[r]
                    ridx = r
            quant = (y[ridx]*self._units[fidx], y[ridx+q]*self._units[fidx])
        else:
            q = [100*(1-cov)/2, 100-100*(1-cov)/2]  # Get cov and 1-cov quantiles
            quant = np.nanpercentile(self._samples[fname].magnitude, q)*self._units[fidx]

        k = ((quant[1]-quant[0])/(2*self._uncert[fidx])).magnitude
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

        hdr = ['Function', 'Interval', 'Min', 'Max', 'k']

        rows = []
        for fidx in range(self.nouts):
            if not self.model.show[fidx]: continue
            for cov in covlist:
                minval, maxval, kval = self.expanded(fidx=fidx, cov=cov, shortest=shortest)
                row = [report.Math.from_sympy(self._symbols[fidx])] if cov == covlist[0] else ['-']

                if isinstance(cov, float):
                    row.append('{:.2f}%'.format(cov*100))
                else:
                    row.append(cov)
                row.append(report.Number(minval, **kwargs))
                row.append(report.Number(maxval, **kwargs))
                row.append(format(kval, '.3f'))
                rows.append(row)
        r = report.Report(**kwargs)
        r.txt('Shortest Coverage Intervals\n' if shortest else 'Symmetric Coverage Intervals\n')
        r.table(rows, hdr)
        return r

    def report_sens(self, **kwargs):
        ''' Report sensitivity coefficients and proportions '''
        r = report.Report(**kwargs)
        for fidx in range(self.nouts):
            if not self.model.show[fidx]: continue
            rows = []
            for i, inpt in enumerate(self.inputs):
                rows.append([report.Math.from_latex(inpt.get_latex()),
                             report.Number(self.Cx[fidx][i], fmin=1),
                             format(self.proportions[fidx][i]*100, '.2f')+'%'])

            if self.nouts > 1:
                r.hdr(report.Math(self.names[fidx]), level=3)
            r.table(rows, hdr=['Variable', 'Sensitivity', 'Proportion'])
        return r

    def correlation(self, fidx1, fidx2):
        ''' Get correlation coefficient between function 1 and 2 '''
        fidx1 = self._index(fidx1)
        fidx2 = self._index(fidx2)
        name1 = self.names[fidx1]
        name2 = self.names[fidx2]
        corr = np.corrcoef(self._samples[name1].magnitude, self._samples[name2].magnitude)[0, 1]
        return corr

    def report_correlation(self, **kwargs):
        ''' Report correlation matrix of outputs. Returns blank report
            for single-output calculations

            Keyword Arguments
            -----------------
            Passed to report.Report
        '''
        r = report.Report()
        if self.nouts < 2:
            return r

        rows = []
        hdr = ['-'] + [report.Math(n) for n in self.names]
        for idx1 in range(self.nouts):
            row = [report.Math(self.names[idx1])]
            for idx2 in range(self.nouts):
                if idx1 == idx2:
                    row.append('1.000')
                else:
                    corr = self.correlation(idx1, idx2)
                    row.append(format(corr, '.3f'))
            rows.append(row)
        r.hdr('Correlation Coefficients (Monte Carlo)', level=3)
        r.table(rows, hdr)
        return r

    def _plot_funcconverge(self, ax1, ax2, fidx=0, **kwargs):
        ''' Plot MC convergence for one function in the model

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
        meancolor = kwargs.pop('meancolor', 'C0')
        unccolor = kwargs.pop('unccolor', 'C1')
        kwargs.setdefault('marker', 'o')

        fidx = self._index(fidx)
        title = '$' + self._symbolslatex[fidx] + '$'
        Y = self.samples(fidx).magnitude
        unitstr = report.Unit(self._units[fidx]).latex(bracket=True)
        step = len(Y)//div
        line = np.empty(div)
        steps = np.empty(div)
        for i in range(div):
            line[i] = Y[:step*(i+1)].mean()
            steps[i] = (i+1)*step
        if relative:
            line = line / line[-1]
        kwargs['color'] = meancolor
        ax1.plot(steps, line, **kwargs)

        ax1.set_title(title)
        if relative:
            ax1.set_ylabel('Value (Relative to final)')
        else:
            ax1.set_ylabel('Value' + unitstr)
        ax1.set_xlabel('Samples')

        line = np.empty(div)
        steps = np.empty(div)
        for i in range(div):
            line[i] = Y[:step*(i+1)].std()
            steps[i] = (i+1)*step
        if relative:
            line = line / line[-1]
        kwargs['color'] = unccolor
        ax2.plot(steps, line, **kwargs)
        ax2.set_title(title)
        ax2.set_xlabel('Samples')
        if relative:
            ax2.set_ylabel('Uncertainty (Relative to final)')
        else:
            ax2.set_ylabel('Uncertainty' + unitstr)

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
        rows = [i for i in range(self.nouts) if self.model.show[i]]
        fig, ax = plotting.initplot(plot)
        fig.clf()
        fig.subplots_adjust(**dfltsubplots)
        cols = 2
        ax = []
        for i in range(len(rows)):
            ax.append([fig.add_subplot(len(rows), cols, i*2+1), fig.add_subplot(len(rows), cols, i*2+2)])

        for i in rows:
            self._plot_funcconverge(ax1=ax[i][0], ax2=ax[i][1], fidx=i, **kwargs)
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

        variables = self.inputs
        if inpts is not None:
            if len(inpts) == 0: return fig
            variables = [self.inputs[i] for i in inpts]

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

        variables = self.inputs
        if inpts is not None and len(inpts) > 1:
            variables = [self.inputs[i] for i in inpts]
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
                    counts, ybins, xbins = np.histogram2d(y, x, bins=bins, density=True)
                    levels = np.linspace(counts.min(), counts.max(), 11)[1:]
                    ax.contour(counts, levels, extent=[xbins.min(), xbins.max(), ybins.min(), ybins.max()], cmap=plt.get_cmap(cmapmc))
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
        return ax

    def plot_correlation(self, ax=None, fidx1=0, fidx2=1, **kwargs):
        ''' Plot correlation contour plot between two outputs. Like scatter plot but using contour lines '''
        _, ax = plotting.initplot(ax)
        cmap = kwargs.get('cmap', 'viridis')
        labelmode = kwargs.get('labelmode', 'name')
        bins = max(3, kwargs.get('bins', 40))
        levels = kwargs.get('levels', None)  # To share contour levels with GUM plot
        fill = kwargs.get('fill', False)
        showleg = kwargs.pop('legend', True)

        x = self.samples(fidx1).magnitude
        y = self.samples(fidx2).magnitude

        counts, ybins, xbins = np.histogram2d(y, x, bins=bins, density=True)
        if levels is None:
            levels = np.linspace(counts.min(), counts.max(), 11)[1:]

        if fill:  # Use filled contour plot when showing both
            ax.contourf(counts, levels, extent=[xbins.min(), xbins.max(), ybins.min(), ybins.max()], cmap=plt.get_cmap(cmap))
        else:
            ax.contour(counts, levels, extent=[xbins.min(), xbins.max(), ybins.min(), ybins.max()], cmap=plt.get_cmap(cmap))
        if showleg:
            cax = make_axes_locatable(ax).append_axes('right', size='5%', pad=0.05)
            sm = plt.cm.ScalarMappable(cmap=plt.get_cmap(cmap), norm=plt.Normalize(vmin=levels.min(), vmax=levels.max()))
            ax.figure.colorbar(sm, cax=cax, orientation='vertical')

        if labelmode == 'desc':
            ax.set_xlabel(self.descriptions[fidx1])
            ax.set_ylabel(self.descriptions[fidx2])
        else:
            fidx1, fidx2 = self._index(fidx1), self._index(fidx2)
            ax.set_xlabel('$' + self._symbolslatex[fidx1] + '$' + report.Unit(self._units[fidx1]).latex(bracket=True))
            ax.set_ylabel('$' + self._symbolslatex[fidx2] + '$' + report.Unit(self._units[fidx2]).latex(bracket=True))
        ax.ticklabel_format(style='sci', axis='x', scilimits=(-4, 4), useOffset=False)
        ax.ticklabel_format(style='sci', axis='y', scilimits=(-4, 4), useOffset=False)

    def plot_scatter(self, ax=None, fidx1=0, fidx2=1, **kwargs):
        ''' Plot scatter between two outputs '''
        _, ax = plotting.initplot(ax)
        points = kwargs.get('points', 10000)
        color = kwargs.get('color', 'C3')
        labelmode = kwargs.get('labelmode', 'name')

        name1 = self.names[self._index(fidx1)]
        name2 = self.names[self._index(fidx2)]
        x = self._samples[name1][:points].magnitude
        y = self._samples[name2][:points].magnitude

        with suppress(ValueError):  # Raises in case where len(x) != len(y) when one output is constant
            ax.plot(x, y, marker='.', ls='', markersize=2, color=color, zorder=0)

        if labelmode == 'desc':
            ax.set_xlabel(self.descriptions[fidx1])
            ax.set_ylabel(self.descriptions[fidx2])
        else:
            fidx1, fidx2 = self._index(fidx1), self._index(fidx2)
            ax.set_xlabel('$' + self._symbolslatex[fidx1] + '$' + report.Unit(self._units[fidx1]).latex(bracket=True))
            ax.set_ylabel('$' + self._symbolslatex[fidx2] + '$' + report.Unit(self._units[fidx2]).latex(bracket=True))
        ax.ticklabel_format(style='sci', axis='x', scilimits=(-4, 4), useOffset=False)
        ax.ticklabel_format(style='sci', axis='y', scilimits=(-4, 4), useOffset=False)

    def plot_normprob(self, plot=None, fidx=0, points=None, **kwargs):
        '''
        Plot a normal probability plot of the Monte-Carlo samples vs.
        expected quantile. If data falls on straight line, data is
        normally distributed.

        Parameters
        ----------
        plot: matplotlib figure or axis
            Axis to plot on
        fidx: int
            Index of function in model to plot
        points: int
            Number of sample points to include for speed. Default is 100.
        '''
        fig, ax = plotting.initplot(plot)
        if points is None:
            points = min(100, self.inputs.nsamples)
        thin = len(self.samples(fidx).magnitude)//points
        plotting.probplot(self.samples(fidx)[::thin].magnitude, ax=ax)


class UncertOutput(output.Output):
    ''' Class to hold output of a full uncertainty calculation (GUM and MC) '''
    def __init__(self, model, inputs, gum=True, symboliconly=False, mc=True):
        self.model = model
        self.inputs = inputs
        self.names = list(self.model.outnames).copy()
        self.nouts = len(self.names)
        self.gum = None
        self.mc = None
        if gum:
            self.gum = GUMOutput(model, inputs, symboliconly=symboliconly)
        if mc:
            self.mc = MCOutput(model, inputs)
        self.show = model.show

    def get_dists(self):
        ''' Get distributions in this output. If name is none, return a list of
            available distribution names.
        '''
        dists = {}
        for fidx, f in enumerate(self.names):
            if self.mc is not None:
                name = f'{f} (MC)'
                dists[name] = self.mc.get_distdef(fidx)
            if self.gum is not None:
                name = f'{f} (GUM)'
                dists[name] = self.gum.get_distdef(fidx)
        return dists

    def report(self, **kwargs):
        ''' Generate report of mean/uncertainty values.

            Keyword Arguments
            -----------------
            Passed to report.Report
        '''
        hdr = ['Function', 'Method', 'Nominal', 'Std. Uncertainty', '95% Coverage', 'k', 'Deg. Freedom']
        conf = 0.95

        rows = []
        for fidx in [i for i in range(self.nouts) if self.model.show[i]]:
            if self.gum is not None:
                row = [report.Math(self.names[fidx]), 'GUM']
                unc, k = self.gum.expanded(fidx=fidx, cov=conf)
                row.extend([report.Number(self.gum._nom[fidx], matchto=self.gum._uncert[fidx]),
                            report.Number(self.gum._uncert[fidx]),
                            (u'± ', report.Number(unc)),
                            format(k, '.3f'),
                            format(self.gum._degf[fidx], '.1f')])
                rows.append(row)
            if self.mc is not None:
                row = [report.Math(self.names[fidx]), 'Monte Carlo']
                umin, umax, k = self.mc.expanded(fidx=fidx, cov=conf)
                rng95 = ('(', report.Number(umin, matchto=self.mc._uncert[fidx]), ', ', report.Number(umax, matchto=self.mc._uncert[fidx]), ')')
                row.extend([report.Number(self.mc._nom[fidx], matchto=self.mc._uncert[fidx]),
                            report.Number(self.mc._uncert[fidx]),
                            rng95,
                            format(k, '.3f'), '-'])
                rows.append(row)
        r = report.Report(**kwargs)
        r.table(rows, hdr)
        return r

    def plot_pdf(self, plot=None, **kwargs):
        ''' Plot PDF of both of GUM and MC results for each function in the model '''
        funcs = kwargs.pop('funcs', [i for i in range(self.nouts) if self.model.show[i]])
        stds = kwargs.pop('stddevs', 4)
        mccolor = kwargs.pop('mccolor', 'C0')
        gumcolor = kwargs.pop('gumcolor', 'C1')
        showmc = kwargs.pop('showmc', self.mc is not None)
        showgum = kwargs.pop('showgum', self.gum is not None)
        bins = max(3, kwargs.pop('bins', 120))
        contour = kwargs.pop('contour', False)
        labelmode = kwargs.pop('labelmode', 'name')  # or 'desc'
        showleg = kwargs.pop('legend', True)
        intervalgum = kwargs.pop('intervalsgum', kwargs.pop('intervals', []))
        intervalmc = kwargs.pop('intervalsmc', [])
        intervaltypegum = kwargs.pop('intervaltypegum', 't')
        intervaltypemc = kwargs.pop('intervaltypemc', 'symmetric')

        [kwargs.pop(k, None) for k in ['points', 'inpts', 'overlay', 'cmap', 'cmapmc', 'equal_scale']]  # Ignore these in pdf plot

        fig, ax = plotting.initplot(plot)
        fig.clf()
        fig.subplots_adjust(**dfltsubplots)
        if len(funcs) == 0: return fig

        rows = int(np.ceil(len(funcs)/3))
        cols = int(np.ceil(len(funcs)/rows))

        for plotid, fidx in enumerate(funcs):
            ax = fig.add_subplot(rows, cols, plotid+1)

            try:
                mean = self.mc._nom[fidx].magnitude
                uncert = self.mc._uncert[fidx].magnitude
                units = self.mc._units[fidx]
            except AttributeError:
                mean = self.gum._nom[fidx].magnitude
                uncert = self.gum._uncert[fidx].magnitude
                units = self.gum._units[fidx]

            xlow = mean - stds * uncert
            xhi = mean + stds * uncert
            if not np.isfinite(xlow):  # Can happen if uncert is Nan for some reason, like with no input variables
                xlow = mean - 1
                xhi = mean + 1

            if showmc:
                kwargs['color'] = mccolor
                self.mc._plot_funcpdf(ax, fidx=fidx, hist=not contour, bins=bins, stddevs=stds,
                                      intervals=intervalmc, intervaltype=intervaltypemc, **kwargs)

            if showgum:
                kwargs['color'] = gumcolor
                self.gum._plot_funcpdf(ax, fidx=fidx, stddevs=stds,
                                       intervals=intervalgum, intervaltype=intervaltypegum, **kwargs)

            if np.isfinite(xlow) and np.isfinite(xhi) and xhi != xlow:
                ax.set_xlim(xlow, xhi)

            ax.ticklabel_format(style='sci', axis='x', scilimits=(-4, 4), useOffset=False)
            ax.set_ylabel('Probability Density')
            unitstr = report.Unit(units).latex(bracket=True)
            if labelmode == 'desc' and self.mc.descriptions[fidx] is not None:
                ax.set_xlabel(self.mc.descriptions[fidx] + unitstr)
            elif self.names[fidx]:
                ax.set_xlabel(report.Math(self.names[fidx]).latex() + unitstr)
            else:
                ax.set_xlabel(unitstr)

            if showleg:
                ax.legend(loc='upper left', fontsize=10)

        fig.tight_layout()

    def plot_correlation(self, plot=None, funcs=None, **kwargs):
        ''' Plot correlation between outputs '''
        showmc = kwargs.pop('showmc', self.mc is not None)
        showgum = kwargs.pop('showgum', self.gum is not None)
        equal_scale = kwargs.get('equal_scale', False)
        overlay = kwargs.get('overlay', False)
        contour = kwargs.get('contour', False)

        fig, ax = plotting.initplot(plot)
        fig.clf()
        fig.subplots_adjust(**dfltsubplots)

        if funcs is None:
            funcs = list(range(self.nouts))

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
                levels = None  # Contour levels/heights. Can be shared bw gum and mc if "equalize scales" checked.

                if showgum:
                    levels = self.gum.plot_correlation(axgum, f1, f2, **kwargs)

                if showmc:
                    if contour:
                        kwargs['fill'] = overlay
                        if levels is not None and equal_scale:
                            kwargs['levels'] = levels
                        if overlay:
                            kwargs['legend'] = False  # Already showing a legend
                        self.mc.plot_correlation(axmc, f1, f2, **kwargs)
                    else:
                        self.mc.plot_scatter(axmc, f1, f2, **kwargs)

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

    def report_summary(self, **kwargs):
        ''' Generate report of all mean/uncerts for each function.

            Keyword Arguments
            -----------------
            Passed to report.Report
        '''
        r = self.report(**kwargs)
        with plt.style.context(plotting.plotstyle):
            fig = plt.figure()
            self.plot_pdf(plot=fig)
            plt.close(fig)  # Prevent display twice in Jupyter, and allow figs to be garbage collected
        r.plot(fig)
        r.append(self.report_warns(**kwargs))
        if self.nouts > 1:
            if self.gum:
                r.append(self.gum.report_correlation(**kwargs))
            if self.mc:
                r.append(self.mc.report_correlation(**kwargs))
        return r

    def report_sens(self, **kwargs):
        ''' Tables of GUM vs MC sensitivity coefficients and proportions '''
        r = report.Report(**kwargs)

        if self.gum and self.mc:
            # Both GUM and MC in same tables
            gumsens = self.gum.Cx
            mcsens = self.mc.Cx
            gumprop = self.gum.proportions
            mcprop = self.mc.proportions
            gumresid = self.gum.residuals

            outids = [i for i in range(self.nouts) if self.model.show[i]]
            for fidx in outids:
                rows = []
                for i, inpt in enumerate(self.inputs):
                    row = [report.Math.from_latex(inpt.get_latex()),
                           report.Number(gumsens[fidx][i], fmin=1),
                           format(gumprop[fidx][i]*100, '.2f')+'%',
                           report.Number(mcsens[fidx][i], fmin=1),
                           format(mcprop[fidx][i]*100, '.2f')+'%']
                    rows.append(row)
                if gumresid[fidx] > 0:
                    rows.append(['Correlations', '', format(gumresid[fidx]*100, '.2f')+'%', '', ''])

                if len(outids) > 1:
                    r.hdr(report.Math(self.names[fidx]), level=3)
                r.table(rows, hdr=['Variable', 'GUM Sensitivity', 'GUM Proportion', 'MC Sensitivity', 'MC Proportion'])

        elif self.gum:
            r = self.gum.report_sens()
        elif self.mc:
            r = self.mc.report_sens()
        return r

    def report_inputs(self, **kwargs):
        ''' Report of input values.

            Keyword Arguments
            -----------------
            Passed to report.Report
        '''
        rows = []
        cols = ['Variable', 'Mean', 'Std. Uncertainty', 'Deg. Freedom', 'Description']
        for inpt in self.inputs:
            rows.append([report.Math(inpt.name),
                         report.Number(inpt.nom, matchto=inpt.stdunc()),
                         report.Number(inpt.stdunc()),
                         report.Number(inpt.degf()),
                         inpt.desc])
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
        r.hdr('GUM Approximation', level=3)
        r.append(self.gum.report_expanded(covlistgum, normal=normal, **kwargs))
        r.hdr('Monte Carlo', level=3)
        r.append(self.mc.report_expanded(covlist, shortest=shortest, **kwargs))
        return r

    def report_components(self, **kwargs):
        ''' Report the uncertainty components

            Keyword Arguments
            -----------------
            Passed to report.Report
        '''
        cols = ['Variable', 'Component', 'Description', 'Standard Uncertainty', 'Deg. Freedom']
        rows = []
        for i, inpt in enumerate(self.inputs):
            rows.append([report.Math(inpt.name), '-',
                         inpt.desc if inpt.desc else '--', report.Number(inpt.stdunc()), format(inpt.degf(), '.1f')])

            for u in inpt.uncerts:
                rows.append(['-', report.Math(u.name),
                             u.desc if u.desc else '--', report.Number(u.std()), format(u.degf, '.1f')])
        r = report.Report(**kwargs)
        r.table(rows, hdr=cols)
        return r

    def validate_gum(self, fidx=0, ndig=2, conf=0.95, full=False):
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
        uc, _ = self.gum.expanded(fidx=fidx, cov=conf)
        ucmin, ucmax, _ = self.mc.expanded(fidx=fidx, cov=conf, shortest=True)
        r = np.floor(np.log10(np.abs(self.gum.uncert(fidx).magnitude))).astype(int) - (ndig-1)
        delta = 0.5 * 10.0**r * self.gum._units[self.gum._index(fidx)]
        dlow = abs((self.gum.nom(fidx) - uc) - ucmin)
        dhi = abs((self.gum.nom(fidx) + uc) - ucmax)
        if not full:
            return (dlow < delta) & (dhi < delta)
        else:
            a = (self.gum.uncert(fidx)/10.**r).astype(int)
            fullparams = {'delta': delta,
                          'dlow': dlow,
                          'dhi': dhi,
                          'r': r,
                          'a': a,
                          'gumlo': self.gum.nom(fidx)-uc,
                          'gumhi': self.gum.nom(fidx)+uc,
                          'mclo': ucmin,
                          'mchi': ucmax}
            return (dlow < delta) & (dhi < delta), fullparams

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
        if self.gum is None or self.mc is None:
            r.txt('GUM and Monte Carlo result not run. No validity check can be made.')
        else:
            for fidx in [i for i in range(self.nouts) if self.model.show[i]]:
                valid, params = self.validate_gum(fidx=fidx, ndig=ndig, conf=conf, full=True)
                deltastr = report.Number(params['delta'], fmin=1)
                r.txt(u'{:d} significant digit{}. δ = {}.\n\n'.format(ndig, 's' if ndig > 1 else '', deltastr))

                rows = []
                hdr = ['{:.2f}% Coverage'.format(conf*100), 'Lower Limit', 'Upper Limit']
                rows.append(['GUM', report.Number(params['gumlo'], matchto=params['dlow']), report.Number(params['gumhi'], matchto=params['dhi'])])
                rows.append(['MC', report.Number(params['mclo'], matchto=params['dlow']), report.Number(params['mchi'], matchto=params['dhi'])])
                rows.append(['abs(GUM - MC)', report.Number(params['dlow'], matchto=params['dlow']), report.Number(params['dhi'], matchto=params['dhi'])])
                rows.append([u'abs(GUM - MC) < δ', '<font color="green">PASS</font>' if params['dlow'] < params['delta'] else '<font color="red">FAIL</font>',
                             '<font color="green">PASS</font>' if params['dhi'] < params['delta'] else '<font color="red">FAIL</font>'])
                if self.nouts > 1:
                    r.hdr(report.Math(self.names[fidx]), level=3)
                r.table(rows, hdr=hdr)
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
        with plt.style.context(plotting.plotstyle):
            r = report.Report(**kwargs)
            if kwargs.get('summary', True):
                r.hdr('Summary', level=2)
                r.append(self.report(**kwargs))
            if kwargs.get('outputs', True):
                fig = plt.figure()
                params = kwargs.get('outplotparams', {})
                joint = params.get('joint', False)
                if joint:
                    self.plot_correlation(plot=fig, **params.get('plotargs', {}))
                else:
                    self.plot_pdf(plot=fig, **params.get('plotargs', {}))
                r.plot(fig)
                plt.close(fig)
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
            if kwargs.get('gumderv', True) and self.gum is not None:
                solve = kwargs.get('gumvalues', False)
                r.hdr('GUM Derivation', level=2)
                r.append(self.gum.report_derivation(solve=solve, **kwargs))
            if kwargs.get('gumvalid', True) and self.gum is not None and self.mc is not None:
                ndig = kwargs.get('gumvaliddig', 2)
                r.hdr('GUM Validity', level=2)
                r.append(self.report_validity(ndig=ndig, **kwargs))
            if kwargs.get('mchist', True) and self.mc is not None:
                params = kwargs.get('mchistparams', {})
                fig = plt.figure()
                plotparams = params.get('plotargs', {})
                if params.get('joint', False):
                    self.mc.plot_xscatter(plot=fig, **plotparams)
                else:
                    self.mc.plot_xhists(plot=fig, **plotparams)
                r.hdr('Monte Carlo Inputs', level=2)
                r.plot(fig)
                plt.close(fig)
            if kwargs.get('mcconv', True) and self.mc is not None:
                relative = kwargs.get('mcconvnorm', False)
                fig = plt.figure()
                self.mc.plot_converge(fig, relative=relative)
                r.hdr('Monte Carlo Convergence', level=2)
                r.plot(fig)
                plt.close(fig)
        return r

    def report_warns(self, **kwargs):
        ''' Report of warnings raised during calculation

            Returns
            -------
            report.Report
        '''
        r = report.Report(**kwargs)
        if self.gum and self.gum.warnings:
            for w in self.gum.warnings:
                r.txt('- ' + w + '\n')
        if self.mc and self.mc.warnings:
            for w in self.mc.warnings:
                r.txt('- ' + w + '\n')
        return r
