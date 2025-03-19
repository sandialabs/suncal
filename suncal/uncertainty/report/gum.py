''' Generate reports and plots of GUM calculation '''

from collections import namedtuple

import numpy as np
import sympy
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from mpl_toolkits.axes_grid1 import make_axes_locatable

from ...common import unitmgr, report, plotting


class ReportGum:
    ''' Reports and plots of GUM calculation results

        Args:
            gumresults: McResults instance
            units: dictionary of functionname: Pint units to convert
    '''
    def __init__(self, gumresults):
        self._results = gumresults
        self._noutputs = len(self._results.functionnames)
        self.plot = GumPlot(self._results)

    def _repr_markdown_(self):
        return self.summary().get_md()

    def summary(self, **kwargs):
        ''' Generate table of expected value and standard uncertainty '''
        hdr = ['Function', 'Nominal', 'Std. Uncertainty']
        rows = []
        for name in self._results.functionnames:
            rows.append((report.Math(name),
                         report.Number(self._results.expected[name], matchto=self._results.uncertainty[name]),
                         report.Number(self._results.uncertainty[name])))
        rpt = report.Report(**kwargs)
        rpt.table(rows, hdr=hdr)
        rpt.append(self.tolerances(**kwargs))
        return rpt

    def expanded(self, conf=0.95, k=None, **kwargs):
        ''' Generate table of expanded uncertainties, min/max, from level of confidence

            Args:
                conf (float): Level of confidence in the interval
                k (float): Coverage factor for interval, overrides conf
        '''
        rows = []
        hdr = ['Function', 'Level of Confidence', 'Minimum', 'Maximum', 'k', 'Deg. Freedom', 'Expanded Uncertainty']
        expanded = self._results.expanded(conf=conf, k=k)

        for funcname in self._results.functionnames:
            uncert = expanded[funcname].uncertainty
            conf = expanded[funcname].confidence
            row = [report.Math(funcname),
                   f'{conf*100:.2f}%',
                   report.Number(self._results.expected[funcname]-uncert, matchto=uncert, **kwargs),
                   report.Number(self._results.expected[funcname]+uncert, matchto=uncert, **kwargs),
                   f'{expanded[funcname].k:.3f}',
                   f'{self._results.degf[funcname]:.2f}',
                   report.Number(uncert, **kwargs)]
            rows.append(row)
        rpt = report.Report(**kwargs)
        rpt.table(rows, hdr=hdr)
        return rpt

    def tolerances(self, **kwargs):
        ''' Generate report of tolerance and probability of conformance for each function '''
        hdr = ['Function', 'Tolerance', 'Probability of Conformance']
        rows = []
        for fname, poc in self._results.prob_conform().items():
            rows.append((
                report.Math(fname),
                str(self._results.tolerances.get(fname, 'NA')),
                report.Number(poc*100, fmin=1, suffix=' %')
            ))
        rpt = report.Report(**kwargs)
        rpt.table(rows, hdr=hdr)
        return rpt

    def sensitivity(self, **kwargs):
        ''' Report sensitivity coefficients and proportions '''
        rpt = report.Report(**kwargs)
        props = self._results.proportions()
        Cx = self._results.sensitivity()

        for funcname in self._results.functionnames:
            rows = []
            for varname in self._results.variablenames:
                rows.append([report.Math(varname),
                             report.Number(Cx[funcname][varname], fmin=1),
                             f'{props[funcname][varname]*100:.2f}%'])

                if props.get('residual', 0) > 0:
                    rows.append(['Correlations', '', f'{props["residual"]:.2f}%'])

            if self._noutputs > 1:
                rpt.hdrmath(funcname, level=2)
            rpt.table(rows, hdr=['Variable', 'Sensitivity', 'Proportion'])
        return rpt

    def correlation(self, **kwargs):
        ''' Report correlation matrix of outputs. Returns blank report
            for single-output calculations
        '''
        rpt = report.Report(**kwargs)
        if self._noutputs < 2:
            return rpt

        correlation = self._results.correlation()

        hdr = ['&nbsp;'] + [report.Math(n) for n in self._results.functionnames]
        rows = []
        for func1 in self._results.functionnames:
            row = [report.Math(func1)]
            for func2 in self._results.functionnames:
                row.append(f'{correlation[func1][func2]:.3f}')
            rows.append(row)
        rpt.hdr('Correlation Coefficients', level=3)
        rpt.table(rows, hdr)
        return rpt

    def model(self, **kwargs):
        ''' Report the measurement model functions '''
        rpt = report.Report(**kwargs)
        for funcname, expr in zip(self._results.functionnames, self._results.symbolic.functions):
            expr = sympy.Eq(sympy.Symbol(funcname), expr)
            rpt.mathtex(self._results.latexify(expr), end='\n\n')
        return rpt

    def input_covariance(self, solve=False, **kwargs):
        ''' Report the input covariance matrix Ux '''
        rpt = report.Report(**kwargs)
        hdr = ['&nbsp;'] + self._results.variablenames
        Ux = self._results.covariance_inputs(symbolic=not solve)
        rows = []
        for varname1 in self._results.variablenames:
            row = [varname1]
            for varname2 in self._results.variablenames:
                if not solve:
                    row.append(report.Math.from_sympy(Ux[varname1][varname2]))
                else:
                    row.append(report.Number(Ux[varname1][varname2]))
            rows.append(row)
        rpt.table(rows, hdr=hdr)
        return rpt

    def measured_values(self, solve=False, **kwargs):
        ''' Report of measured values and their uncertainties

            Args:
                solve (bool): Show numeric values instead of symbols
        '''
        rpt = report.Report(**kwargs)

        def combine_qty(expr, val):
            ''' Build a report.Math from "expr = value" if solve and "expr" if not. '''
            val, units = unitmgr.split_units(val)
            if not solve:
                return report.Math.from_sympy(expr)
            symexp = sympy.Eq(expr, sympy.N(val, kwargs.get('n', report.default_sigfigs)), evaluate=False)
            return report.Math.from_sympy(symexp, units)

        rows = []
        for varname in self._results.variablenames:
            rows.append([
                combine_qty(sympy.Symbol(varname), self._results.variables.expected[varname]),
                combine_qty(sympy.Symbol(f'u_{varname}'), self._results.variables.uncertainty[varname]),
                combine_qty(sympy.Symbol(f'nu_{varname}'), self._results.variables.degf[varname])])
        rpt.table(rows, hdr=['Variable', 'Std. Uncertainty', 'Deg. Freedom'])
        return rpt

    def input_sensitivity(self, solve=False, as_partials=False, **kwargs):
        ''' Report sensitivity coefficients/matrix Cx

            Args:
                solve (bool): Show numeric values instead of symbols
                as_partials (bool): Show as partial derivatives
        '''
        # as_partials only applies to multi-output, matrix form of report.
        rpt = report.Report(**kwargs)
        Cx_symbolic = self._results.sensitivity(True)
        Cx_numeric = self._results.sensitivity(False)

        # Single output, report list of coefficients
        if self._noutputs <= 1:
            Cx_numeric = self._results.sensitivity()
            fname = self._results.functionnames[0]
            for varname in self._results.variablenames:
                expr = sympy.Eq(sympy.Derivative(sympy.Symbol(fname), sympy.Symbol(varname)), Cx_symbolic[fname][varname])
                latex = self._results.latexify(expr)
                if solve:
                    latex += ' = '
                    kwargs.update({'unitfmt': 'latex'})
                    latex += report.Number(Cx_numeric[fname][varname], **kwargs).string()
                rpt.mathtex(latex, end='\n\n')
            return rpt

        # Multiple outputs, return matrix
        # Three choices: matrix of d/dx[y];  matrix of solved derivatives;  matrix of numeric values
        rows = []
        hdr = ['&nbsp;'] + self._results.variablenames
        for funcname in self._results.functionnames:
            row = [report.Math(funcname)]
            for varname in self._results.variablenames:
                if as_partials:
                    row.append(report.Math.from_sympy(sympy.Derivative(sympy.Symbol(funcname), sympy.Symbol(varname))))
                elif not solve:
                    tex = self._results.latexify(Cx_symbolic[funcname][varname])
                    row.append(report.Math.from_latex(tex))
                else:
                    row.append(report.Number(Cx_numeric[funcname][varname]))
            rows.append(row)
        rpt.table(rows, hdr)
        return rpt

    def model_uncert_equations(self, solve=False, **kwargs):
        ''' Report the combined covariance and simplified expressions for uncertainty of model

            Args:
                solve (bool): Show numeric values instead of symbols
        '''
        rpt = report.Report(**kwargs)
        if self._noutputs > 1:
            rpt.mathtex(r'U_y = C_x \cdot U_x \cdot C_x^T', end='\n\n')
            rpt.txt('Uncertainties')
            rpt.mathtex(r'\left(\sqrt{\mathrm{diag}(U_y)}\right):', end='\n\n')
        for funcname in self._results.functionnames:
            ufname = f'u_{funcname}'
            expr = sympy.Eq(sympy.Symbol(ufname), self._results.symbolic.uncertainty[ufname])
            if not solve:
                rpt.mathtex(self._results.latexify(expr), end='\n\n')
            else:
                latex = self._results.latexify(expr) + ' = '
                kwargs.update({'unitfmt': 'latex'})
                latex += report.Number(self._results.uncertainty[funcname], **kwargs).string()
                rpt.mathtex(latex, end='\n\n')
        return rpt

    def degrees_freedom(self, solve=False, **kwargs):
        ''' Report the welch-satterthwaite formula for degrees of freedom

            Args:
                solve (bool): Show numeric values instead of symbols
        '''
        rpt = report.Report(**kwargs)
        for funcname in self._results.functionnames:
            degf_name = f'nu_{funcname}'
            expr = sympy.Eq(sympy.Symbol(degf_name), self._results.symbolic.degf[degf_name])
            latex = self._results.latexify(expr)
            if solve:
                latex += ' = '
                kwargs.update({'unitfmt': 'latex'})
                latex += report.Number(self._results.degf[funcname], **kwargs).string()
            rpt.mathtex(latex, end='\n\n')
        return rpt

    def derivation(self, solve=False, **kwargs):
        ''' Generate report of derivation of GUM method.

            Args:
                solve (bool): Show numeric values instead of symbols
        '''
        rpt = report.Report(**kwargs)
        multiout = self._noutputs > 1
        symbolic = self._results.symbolic  # Symbolic solution to GUM

        if symbolic is None:
            rpt.txt('No symbolic solution computed for function.')
            return rpt

        if len(symbolic.Ux) == 0:
            rpt.txt('No input variables defined in model.')
            return rpt

        rpt.hdr('Measurement Model:', level=3)
        rpt.append(self.model(**kwargs))

        if multiout or self._results.has_correlated_inputs():
            rpt.hdr('Input Covariance Matrix [Ux]:', level=3)
            rpt.append(self.input_covariance(solve=False, **kwargs))
            if solve:  # Include both tables
                rpt.append(self.input_covariance(solve=True, **kwargs))
        else:
            rpt.hdr('Measured Values:', level=3)
            rpt.append(self.measured_values(solve=solve, **kwargs))

        if multiout:
            rpt.hdr('Sensitivity Matrix [Cx]:', level=3)
            rpt.hdr('Partial Derivatives:', level=5)
            rpt.append(self.input_sensitivity(as_partials=True))
            rpt.hdr('Computed Partial Derivatives:', level=5)
            rpt.append(self.input_sensitivity(solve=False))
            if solve:
                rpt.hdr('Values:', level=5)
                rpt.append(self.input_sensitivity(solve=True))
        else:
            rpt.hdr('Sensitivity Coefficients', level=3)
            rpt.append(self.input_sensitivity(solve=solve))

        if multiout:
            rpt.hdr('Combined Covariance [Uy]:', level=3)
        else:
            rpt.hdr('Combined Uncertainty:', level=3)
        rpt.append(self.model_uncert_equations(solve=solve, **kwargs))

        rpt.hdr('Effective degrees of freedom:', level=3)
        rpt.append(self.degrees_freedom(solve=solve, **kwargs))
        return rpt


class GumPlot:
    ''' Functions for plotting GUM results '''
    def __init__(self, gumresults):
        self._results = gumresults
        self._noutputs = len(self._results.functionnames)
        self.axis = GumAxisPlot(self._results)

    def pdf(self, functions=None, fig=None, interval=None, k=None, labeldesc=False, **kwargs):
        ''' Plot probability density function for the outputs. Clears the figure.

            Args:
                functions (list of str): Function names to include
                fig (plt.Figure): Matplotlib Figure to plot on
                interval (float): Show expanded interval to this level of confidence
                k (float): Show expanded interval to this coverage factor (overrides interval argument)
                labeldesc (bool): Use "description" as axis label instead of variable name
        '''
        if functions is None:
            functions = self._results.functionnames

        fig, _ = plotting.initplot(fig)
        fig.clf()
        fig.subplots_adjust(**plotting.dfltsubplots)
        axs = plotting.axes_grid(len(functions), fig, len(functions))
        for fname, ax in zip(functions, axs):
            self.axis.pdf(funcname=fname, ax=ax, interval=interval, k=k, labeldesc=labeldesc, **kwargs)
        fig.tight_layout()

    def joint_pdf(self, functions=None, fig=None, cmap='viridis', legend=True,
                  labeldesc=False, conf=None, **kwargs):
        ''' Plot joint PDF between functions

            Args:
                functions (list of str): Function names to include
                fig (plt.Figure): Matplotlib Figure to plot on
                cmap (str): Matplotlib colormap name
                legend (bool): Show legend
                labeldesc (bool): Use "description" as axis label instead of variable name
                conf (float): Level of confidence for displaying confidence region
        '''
        fig, _ = plotting.initplot(fig)
        fig.clf()
        if functions is None:
            functions = self._results.functionnames

        noutputs = len(functions)
        if noutputs < 2:
            return   # Nothing to plot

        for row, func1 in enumerate(functions):
            for col, func2 in enumerate(functions):
                if col <= row:
                    continue
                ax = fig.add_subplot(noutputs-1, noutputs-1, row*(noutputs-1)+col)
                self.axis.joint_pdf(func1, func2, ax=ax, cmap=cmap, legend=legend,
                                    labeldesc=labeldesc, conf=conf)


class GumAxisPlot:
    ''' Functions for plotting GUM results on a single axis '''
    def __init__(self, gumresults):
        self._results = gumresults
        self._noutputs = len(self._results.functionnames)

    def pdf(self, funcname, ax=None, interval=None, k=None, labeldesc=False, **kwargs):
        ''' Plot probability density for the function

            Args:
                funcname (str): Function to plot
                ax (plt.axes): Matplotlib Axis to plot on
                interval (float): Show expanded interval to this level of confidence
                k (float): Show expanded interval to this coverage factor (overrides interval argument)
                labeldesc (bool): Use "description" as axis label instead of variable name
        '''
        _, ax = plotting.initplot(ax)
        stdevs = 4
        mean, units = unitmgr.split_units(self._results.expected[funcname])
        uncert = unitmgr.strip_units(self._results.uncertainty[funcname])   # Assuming same units?
        x = np.linspace(mean - stdevs*uncert, mean + stdevs*uncert, num=100)
        pdf = stats.norm.pdf(x, loc=unitmgr.strip_units(self._results.expected[funcname]),
                             scale=unitmgr.strip_units(self._results.uncertainty[funcname]))

        line, *_ = ax.plot(x, pdf, **kwargs)

        if k is not None or interval is not None:
            if k is not None:
                expand = unitmgr.strip_units(self._results.expand(funcname, k=k))
                ax.legend(loc='lower left', title=f'k = {k:.2f}')
            else:
                expand = unitmgr.strip_units(self._results.expand(funcname, conf=interval))
                ax.legend(loc='lower left', title=f'{interval*100:.2f}% Coverage')
            ax.axvline(mean + expand, ls='--', color='C3')
            ax.axvline(mean - expand, ls='--', color='C3', label='GUM')

        if labeldesc:
            label = self._results.descriptions.get(funcname, '')
        else:
            label = report.Math(funcname).latex()

        if units:
            label += report.Unit(units).latex(bracket=True)

        ax.set_xlabel(label)
        return line

    def joint_pdf(self, function1, function2, ax=None, cmap='viridis',
                  legend=True, labeldesc=False, conf=None, **kwargs):
        ''' Plot joint PDF (uncertainty region) between two model functions

            Args:
                function1 (str): Function to plot
                function2 (str): Function to plot
                ax (plt.axes): Matplotlib Axis to plot on
                cmap (str): Matplotlib colormap name
                legend (bool): Show legend
                labeldesc (bool): Use "description" as axis label instead of variable name
                conf (float): Level of confidence for displaying confidence region
        '''
        _, ax = plotting.initplot(ax)

        contours = None
        x, y, z = _contour(self._results.expected, self._results.covariance(), function1, function2)

        if z is not None:
            contours = np.linspace(z.min(), z.max(), 11)[1:]
            ax.contour(x, y, z, contours, cmap=cmap, **kwargs)
            ax.locator_params(nbins=5)

            if labeldesc:
                xlabel = self._results.descriptions.get(function1, '')
                ylabel = self._results.descriptions.get(function2, '')
            else:
                xlabel = report.mathstr_to_latex(function1)
                ylabel = report.mathstr_to_latex(function2)

            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)

            ax.ticklabel_format(style='sci', axis='x', scilimits=(-4, 4), useOffset=False)
            ax.ticklabel_format(style='sci', axis='y', scilimits=(-4, 4), useOffset=False)
            if legend:
                cax = make_axes_locatable(ax).append_axes('right', size='5%', pad=0.05)
                ax.figure.colorbar(
                    plt.cm.ScalarMappable(
                        cmap=plt.get_cmap(cmap), norm=plt.Normalize(vmin=contours.min(), vmax=contours.max())),
                    cax=cax, orientation='vertical')

        if conf:
            # Draw confidence ellipse per JCGM102:2011 Section 6.5
            center, width, height, angle = _coverage_ellipse(
                self._results.expected, self._results.covariance(), function1, function2, conf=conf)
            ax.add_patch(Ellipse(center, width, height, angle=angle,
                                 color='C1', fill='none', zorder=5, alpha=.3))

        return contours


def _get_jointdist(means, covariance, function1, function2):
    ''' Get joint distribution means and Covariance matrix '''
    mean0 = unitmgr.strip_units(means[function1])
    mean1 = unitmgr.strip_units(means[function2])
    Uy = [[unitmgr.strip_units(covariance[function1][function1]),
           unitmgr.strip_units(covariance[function1][function2])],
          [unitmgr.strip_units(covariance[function2][function1]),
           unitmgr.strip_units(covariance[function2][function2])]]
    return (mean0, mean1), Uy


def _coverage_ellipse(means, covariance, function1, function2, conf=.95):
    ''' Calculate ellipse containing 95% coverage probability '''
    # See JCGM102:2011 Section 6.5
    CoverageEllipse = namedtuple('CoverageEllipse', 'center width height angle')
    means, Uy = _get_jointdist(means, covariance, function1, function2)

    eigval, eigvec = np.linalg.eigh(Uy)
    chi2 = stats.chi2(df=2).isf(1-conf)
    width = 2*np.sqrt(chi2*eigval[0])
    height = 2*np.sqrt(chi2*eigval[1])
    angle = np.degrees(np.arctan2(*eigvec[::-1, 0]))
    return CoverageEllipse(means, width, height, angle)


def _contour(means, covariance, function1, function2):
    ''' Generate x, y, z contours for plotting correlation region. '''
    Contour = namedtuple('Contour', ['x', 'y', 'pdf'])
    means, Uy = _get_jointdist(means, covariance, function1, function2)
    unc0 = np.sqrt(Uy[0][0])
    unc1 = np.sqrt(Uy[1][1])
    try:
        randvar = stats.multivariate_normal(means, cov=Uy)
    except (ValueError, np.linalg.LinAlgError):
        return Contour(None, None, None)

    x, y = np.meshgrid(np.linspace(means[0]-3*unc0, means[0]+3*unc0), np.linspace(means[1]-3*unc1, means[1]+3*unc1))
    pos = np.dstack((x, y))

    return Contour(x, y, randvar.pdf(pos))
