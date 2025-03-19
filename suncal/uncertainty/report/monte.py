''' Generate reports and plots from Monte Carlo calculation '''

from contextlib import suppress

import numpy as np
from scipy import linalg
from matplotlib.patches import Ellipse, Rectangle

from ...common import unitmgr, report, plotting


class ReportMonteCarlo:
    ''' Monte Carlo calculation report

        Args:
            mcresults: McResults instance
    '''
    def __init__(self, mcresults):
        self._results = mcresults
        self._noutputs = len(self._results.functionnames)
        self.plot = McPlot(self._results)

    def _repr_markdown_(self):
        return self.summary().get_md()

    def summary(self, **kwargs):
        ''' Generate table of expected values '''
        hdr = ['Function', 'Nominal', 'Std. Uncertainty']
        rows = []
        for name in self._results.functionnames:
            rows.append((report.Math(name),
                         report.Number(self._results.expected[name], matchto=self._results.uncertainty[name]),
                         report.Number(self._results.uncertainty[name])))
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

    def expanded(self, conf=0.95, k=None, shortest=False, **kwargs):
        ''' Generate table of expanded uncertainties, min/max, from level of confidence

            Args:
                conf (float): Level of confidence in interval
                k: Coverage factor (overrides confidence)
                shortest (bool): Use shortest instead of symmetric interval
        '''
        rows = []
        hdr = ['Function', 'Level of Confidence', 'Minimum', 'Maximum', 'Coverage Factor']
        expanded = self._results.expanded(conf, k=k, shortest=False)

        for funcname in self._results.functionnames:
            uncert = self._results.uncertainty[funcname]
            low = expanded[funcname].low
            high = expanded[funcname].high
            conf = expanded[funcname].confidence
            row = [report.Math(funcname),
                   f'{conf*100:.2f}%',
                   report.Number(low, matchto=uncert, **kwargs),
                   report.Number(high, matchto=uncert, **kwargs),
                   f'{expanded[funcname].k:.3f}']
            rows.append(row)
        rpt = report.Report(**kwargs)
        rpt.txt('Shortest Coverage Intervals\n' if shortest else 'Symmetric Coverage Intervals\n')
        rpt.table(rows, hdr)
        return rpt

    def sensitivity(self, **kwargs):
        ''' Report sensitivity coefficients and proportions '''
        rpt = report.Report(**kwargs)
        sensitivities, proportions = self._results.sensitivity()

        for funcname in self._results.functionnames:
            rows = []
            for varname in self._results.variablenames:
                rows.append([report.Math(varname),
                             report.Number(sensitivities[funcname][varname], fmin=1),
                             f'{proportions[funcname][varname]*100:.2f}%'])

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
        rpt.hdr('Correlation Coefficients (Monte Carlo)', level=3)
        rpt.table(rows, hdr)
        return rpt


class McPlot:
    ''' Functions for plotting Monte Carlo results '''
    def __init__(self, mcresults):
        self._results = mcresults
        self._noutputs = len(self._results.functionnames)
        self.axis = McAxisPlot(self._results)

    def hist(self, functions=None, fig=None, bins=100, labeldesc=False, **kwargs):
        ''' Plot histogram of Monte Carlo distribution. Clears the figure.

            Args:
                functions (list of str): Function names to include
                fig (plt.Figure): Matplotlib Figure to plot on
                bins (int): Number of histogram bins
                labeldesc (bool): Use "description" as axis label instead of variable name
                **kwargs: passed to hist()
        '''
        if functions is None:
            functions = self._results.functionnames

        fig, _ = plotting.initplot(fig)
        fig.clf()
        fig.subplots_adjust(**plotting.dfltsubplots)
        axs = plotting.axes_grid(len(functions), fig, len(functions))
        for fname, ax in zip(functions, axs):
            self.axis.hist(funcname=fname, ax=ax, bins=bins, labeldesc=labeldesc, **kwargs)
        if len(axs) > 1:
            fig.tight_layout()

    def pdf(self, fig=None, functions=None, bins=100, interval=None, k=None, shortest=False,
            labeldesc=False, **kwargs):
        ''' Plot piecewise PDF of Monte Carlo distribution,

            Args:
                fig (plt.Figure): Matplotlib Figure to plot on
                functions (list of str): Function names to include
                bins (int): Number of histogram bins
                interval (float): Show expanded interval to this level of confidence
                k (float): Show expanded interval to this coverage factor (overrides interval argument)
                shortest (bool): Use shortest instead of symmetric interval
                labeldesc (bool): Use "description" as axis label instead of variable name
                **kwargs: passed to hist()
        '''
        if functions is None:
            functions = self._results.functionnames

        fig, _ = plotting.initplot(fig)
        fig.clf()
        fig.subplots_adjust(**plotting.dfltsubplots)
        axs = plotting.axes_grid(len(functions), fig, len(functions))
        for fname, ax in zip(functions, axs):
            self.axis.pdf(funcname=fname, ax=ax, bins=bins, interval=interval,
                          k=k, shortest=shortest, labeldesc=labeldesc, **kwargs)
        if len(axs) > 1:
            fig.tight_layout()

    def scatter(self, functions=None, fig=None, points=10000, labeldesc=False, conf=None, **kwargs):
        ''' Plot correlations

            Args:
                fig (plt.Figure): Matplotlib Figure to plot on
                functions (list of str): Function names to include
                points (int): Number of scatter points to plot
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
                self.axis.scatter(func1, func2, ax=ax, points=points,
                                  labeldesc=labeldesc, conf=conf, **kwargs)

    def joint_pdf(self, functions=None, fig=None, bins=40, cmap='viridis', labeldesc=False, conf=None, **kwargs):
        ''' Plot correlations

            Args:
                functions (list of str): Function names to include
                fig (plt.Figure): Matplotlib Figure to plot on
                bins (int): Number of histogram bins
                cmap (str): Matplotlib colormap name
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
                self.axis.joint_pdf(func1, func2, ax=ax, bins=bins, cmap=cmap,
                                    labeldesc=labeldesc, conf=conf, **kwargs)

    def probplot(self, function, distname, fig=None, bins=100, points=200):
        ''' Plot Probability (Q-Q) Plot fitting distname to samples

            Args:
                function (str): Function to plot
                distname (str): Name of probability distribution to compare
                fig (plt.Figure): Matplotlib Figure to plot on
                bins (int): Number of histogram bins
                points (int): Number of data points to show in Q-Q
        '''
        y = unitmgr.strip_units(self._results.samples[function])
        fitparams = plotting.fitdist(y, distname=distname, fig=fig, bins=bins, points=points, qqplot=True)
        with suppress(IndexError):  # Raises if axes weren't added (maybe invalid samples)
            fig.axes[0].set_title('Distribution Fit')
            fig.axes[1].set_title('Probability Plot')
        fig.tight_layout()
        return fitparams

    def converge(self, fig=None, div=25, relative=False, **kwargs):
        ''' Plot Monte Carlo convergence for all outputs. Clears figure.

            Args:
                fig (plt.Figure): Matplotlib Figure to plot on
                div (int): Number of divisions to plot
                relative (bool): Plot relative to final value
        '''
        fig, _ = plotting.initplot(fig)
        fig.clf()
        for i, funcname in enumerate(self._results.functionnames):
            ax1 = fig.add_subplot(self._noutputs, 2, i*2+1)
            ax2 = fig.add_subplot(self._noutputs, 2, i*2+2)
            self.axis.converge_mean(ax=ax1, funcname=funcname, div=div, relative=relative, **kwargs)
            self.axis.converge_uncert(ax=ax2, funcname=funcname, div=div, relative=relative, **kwargs)
        fig.tight_layout()

    def variable_hist(self, fig=None, variables=None, bins=100, labeldesc=False, **kwargs):
        ''' Plot histograms of sampled variables. Clears the figure.

            Args:
                fig (plt.Figure): Matplotlib Figure to plot on
                variables (list of str): Names of variables to include
                bins (int): Number of histogram bins
                labeldesc (bool): Use "description" as axis label instead of variable name
        '''
        fig, _ = plotting.initplot(fig)
        fig.clf()
        if variables is None:
            variables = self._results.variablenames

        axs = plotting.axes_grid(len(variables), fig)
        for ax, varname in zip(axs, variables):
            self.axis.variable_hist(varname, ax=ax, bins=bins, labeldesc=labeldesc, **kwargs)
        fig.tight_layout()

    def variable_scatter(self, fig=None, variables=None, points=10000, labeldesc=False, **kwargs):
        ''' Scatter plot between all variables in the list

            Args:
                fig (plt.Figure): Matplotlib Figure to plot on
                variables (list of str): Names of variables to include
                points (int): Number of points to plot
                labeldesc (bool): Use "description" as axis label instead of variable name
        '''
        fig, ax = plotting.initplot(fig)
        fig.clf()

        if variables is None:
            variables = self._results.variablenames

        if len(variables) == 1:
            return  # Only one input, can't plot scatter

        lpad = .1
        rpad = .02
        w = (1-lpad-rpad)/(len(variables)-1)
        for row, varname1 in enumerate(variables):
            for col, varname2 in enumerate(variables):
                if col <= row:
                    continue
                # Use add_axes instead of add_subplot because add_subplot tries to be too smart
                # and adjusts for axes labels etc. and will crash with lots of axes having long labels.
                ax = fig.add_axes([lpad+w*(col-1), 1-w*(row+1), w, w])
                ax.locator_params(nbins=5)
                self.axis.variable_scatter(varname1, varname2, ax=ax, points=points, labeldesc=labeldesc, **kwargs)

    def variable_contour(self, fig=None, variables=None, bins=200, labeldesc=False, **kwargs):
        ''' Contour plot of all variables in the list

            Args:
                fig (plt.Figure): Matplotlib Figure to plot on
                variables (list of str): Names of variables to include
                bins (int): Number of histogram bins
                labeldesc (bool): Use "description" as axis label instead of variable name
        '''
        fig, ax = plotting.initplot(fig)
        fig.clf()

        if variables is None:
            variables = self._results.variablenames

        if len(variables) == 1:
            return  # Only one input, can't plot scatter

        lpad = .1
        rpad = .02
        w = (1-lpad-rpad)/(len(variables)-1)
        for row, varx in enumerate(variables):
            for col, vary in enumerate(variables):
                if col <= row:
                    continue
                # Use add_axes instead of add_subplot because add_subplot tries to be too smart
                # and adjusts for axes labels etc. and will crash with lots of axes having long labels.
                ax = fig.add_axes([lpad+w*(col-1), 1-w*(row+1), w, w])
                ax.locator_params(nbins=5)

                xsamples, xunits = unitmgr.split_units(self._results.varsamples[varx])
                ysamples, yunits = unitmgr.split_units(self._results.varsamples[vary])
                counts, ybins, xbins = np.histogram2d(ysamples, xsamples, bins=bins, density=True)
                levels = np.linspace(counts.min(), counts.max(), 11)[1:]
                ax.contour(counts,
                           levels,
                           extent=[xbins.min(), xbins.max(), ybins.min(), ybins.max()],
                           **kwargs)

                if labeldesc:
                    xlabel = self._results.variables.descriptions.get(varx, '')
                    ylabel = self._results.variables.descriptions.get(vary, '')
                else:
                    xlabel = report.mathstr_to_latex(varx) + report.Unit(xunits).latex(bracket=True)
                    ylabel = report.mathstr_to_latex(vary) + report.Unit(yunits).latex(bracket=True)

                ax.set_xlabel(xlabel)
                ax.set_ylabel(ylabel)
                ax.ticklabel_format(style='sci', axis='x', scilimits=(-4, 4), useOffset=False)
                ax.ticklabel_format(style='sci', axis='y', scilimits=(-4, 4), useOffset=False)


class McAxisPlot:
    ''' Functions for plotting Monte Carlo results on a single axis '''
    def __init__(self, mcresults):
        self._results = mcresults
        self._noutputs = len(self._results.functionnames)

    def _approx_pdf(self, x, funcname, bins=200):
        ''' Approximate probability density function for this output based on histogram
            of output samples.
        '''
        samples = unitmgr.strip_units(self._results.samples[funcname])
        samples = samples[np.isfinite(samples)]
        if len(samples):
            pdfy, pdfx = np.histogram(samples,
                                      bins=bins,
                                      range=(samples.min(), samples.max()),
                                      density=True)
            pdfx = pdfx[:-1] + (pdfx[1]-pdfx[0])/2  # Shift to center of bin
            y = np.interp(x, pdfx, pdfy)
            return y
        return [np.nan] * x

    def hist(self, funcname=None, ax=None, bins=100, interval=None, k=None, shortest=False,
             labeldesc=False, **kwargs):
        ''' Plot histogram of one function

            Args:
                funcname (str): Name of function to plot
                ax (plt.axes): Matplotlib Axis to plot on
                bins (int): Number of histogram bins
                interval (float): Expanded interval to plot at this confidence
                k (float): Show expanded interval to this coverage factor (overrides interval argument)
                shortest (bool): Use shortest instead of symmetric interval
                labeldesc (bool): Use "description" as axis label instead of variable name
        '''
        _, ax = plotting.initplot(ax)
        samples, units = unitmgr.split_units(self._results.samples[funcname])
        line = None
        if np.isfinite(samples).any():
            _, _, line = ax.hist(samples, bins=bins, density=True, **kwargs)

        if k is not None or interval is not None:
            if k is not None:
                expand = self._results.expand(funcname, k=k)
                ax.legend(loc='lower left', title=f'k = {k:.2f}')
            else:
                expand = self._results.expand(funcname, conf=interval, shortest=shortest)
                ax.legend(loc='lower left', title=f'{interval*100:.2f}% Coverage')
            low, high = unitmgr.strip_units(expand.low), unitmgr.strip_units(expand.high)
            ax.axvline(low, ls='--', color='C4')
            ax.axvline(high, ls='--', color='C4', label='Monte Carlo')

        if labeldesc:
            label = self._results.descriptions.get(funcname)
        else:
            label = report.Math(funcname).latex()
            if units:
                label += report.Unit(units).latex(bracket=True)
        ax.set_xlabel(label)
        return line

    def pdf(self, funcname=None, ax=None, bins=100, interval=None, k=None,
            shortest=False, labeldesc=False, **kwargs):
        ''' Plot piecewise PDF of one function

            Args:
                funcname (str): Name of function to plot
                ax (plt.axes): Matplotlib Axis to plot on
                bins (int): Number of histogram bins
                interval (float): Show expanded interval to this level of confidence
                k (float): Show expanded interval to this coverage factor (overrides interval argument)
                shortest (bool): Use shortest instead of symmetric interval
                labeldesc (bool): Use "description" as axis label instead of variable name
                **kwargs: passed to hist()
        '''
        _, ax = plotting.initplot(ax)
        stdevs = 4
        mean, units = unitmgr.split_units(self._results.expected[funcname])
        uncert = unitmgr.strip_units(self._results.uncertainty[funcname])
        x = np.linspace(mean - stdevs*uncert, mean + stdevs*uncert, num=100)
        y = self._approx_pdf(x, funcname=funcname, bins=bins)
        line, *_ = ax.plot(x, y, **kwargs)

        if k is not None or interval is not None:
            if k is not None:
                low, high, _, _ = self._results.expand(funcname, k=k)
                ax.legend(loc='lower left', title=f'k = {k:.2f}')
            else:
                low, high, _, _ = self._results.expand(funcname, conf=interval, shortest=shortest)
                ax.legend(loc='lower left', title=f'{interval*100:.2f}% Coverage')
            low, high = unitmgr.strip_units(low), unitmgr.strip_units(high)
            ax.axvline(low, ls='--', color='C4')
            ax.axvline(high, ls='--', color='C4', label='Monte Carlo')

        if labeldesc:
            label = self._results.descriptions.get(funcname)
        else:
            label = report.Math(funcname).latex()
            if units:
                label += report.Unit(units).latex(bracket=True)
        ax.set_xlabel(label)
        return line

    def variable_hist(self, varname, ax=None, bins=100, labeldesc=False, **kwargs):
        ''' Plot histogram of one sampled variable

            Args:
                varname (str): Name of variable to plot
                ax (plt.axes): Matplotlib Axis to plot on
                bins (int): Number of histogram bins
                labeldesc (bool): Use "description" as axis label instead of variable name
        '''
        _, ax = plotting.initplot(ax)
        samples, units = unitmgr.split_units(self._results.varsamples[varname])
        ax.hist(samples, bins=bins, density=True, **kwargs)
        ax.ticklabel_format(style='sci', axis='x', scilimits=(-4, 4), useOffset=False)
        ax.yaxis.set_visible(False)

        if labeldesc:
            label = self._results.variables.descriptions.get(varname)
        else:
            unitstr = report.Unit(units).latex(bracket=True)
            label = f'{report.mathstr_to_latex(varname)}' + unitstr
        ax.set_xlabel(label)

    def variable_scatter(self, varname1, varname2, ax=None, points=10000, labeldesc=False, **kwargs):
        ''' Plot scatter of variable1 and variable2

            Args:
                varname1 (str): Name of first variable to plot
                varname2 (str): Name of second variable to plot
                ax (plt.axes): Matplotlib Axis to plot on
                points (int): Number of scatter points to plot
                labeldesc (bool): Use "description" as axis label instead of variable name
                **kwargs: passed to plot()
        '''
        _, ax = plotting.initplot(ax)
        kwargs.setdefault('marker', '.')
        kwargs.setdefault('ls', '')
        kwargs.setdefault('markersize', 2)
        kwargs.setdefault('color', 'C2')
        xsamples, xunits = unitmgr.split_units(self._results.varsamples[varname1])
        ysamples, yunits = unitmgr.split_units(self._results.varsamples[varname2])
        ax.plot(xsamples[:points], ysamples[:points], **kwargs)

        if labeldesc:
            xlabel = self._results.variables.descriptions.get(varname1, '')
            ylabel = self._results.variables.descriptions.get(varname2, '')
        else:
            xlabel = report.mathstr_to_latex(varname1) + report.Unit(xunits).latex(bracket=True)
            ylabel = report.mathstr_to_latex(varname2) + report.Unit(yunits).latex(bracket=True)

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.ticklabel_format(style='sci', axis='x', scilimits=(-4, 4), useOffset=False)
        ax.ticklabel_format(style='sci', axis='y', scilimits=(-4, 4), useOffset=False)

    def normprob(self, funcname, ax=None, points=100):
        ''' Normal probability plot of the model function results

            Args:
                funcname (str): Name of function
                ax (plt.axes): Matplotlib Axis to plot on
                points (int): Number of points to include
        '''
        _, ax = plotting.initplot(ax)
        samples = unitmgr.strip_units(self._results.samples[funcname])
        thin = len(samples) // points
        plotting.probplot(samples[::thin], ax=ax)

    def joint_pdf(self, funcname1, funcname2, ax=None, bins=40, fill=False, levels=None,
                  cmap='viridis', labeldesc=False, conf=None, shortest=False, **kwargs):
        ''' Joint probability density between two model functions

            Args:
                funcname1 (str): Name of first function
                funcname2 (str): Name of second function
                ax (plt.axes): Matplotlib Axis to plot on
                bins (int): Number of bins for defining contour
                fill (bool): Fill the contours
                levels (list): Levels of contour lines
                cmap (str): Matplotlib colormap name
                labeldesc (bool): Use "description" as axis label instead of variable name
                conf (float): Level of confidence for displaying coverage region
                shortest (bool): Use shortest coverage region
        '''
        _, ax = plotting.initplot(ax)

        xsamples, xunits = unitmgr.split_units(self._results.samples[funcname1])
        ysamples, yunits = unitmgr.split_units(self._results.samples[funcname2])
        counts, ybins, xbins = np.histogram2d(ysamples, xsamples, bins=bins, density=True)

        if levels is None:  # To share contour levels with a GUM plot
            levels = np.linspace(counts.min(), counts.max(), 11)[1:]

        if fill:
            ax.contourf(counts,
                        levels,
                        extent=[xbins.min(), xbins.max(), ybins.min(), ybins.max()],
                        cmap=cmap,
                        **kwargs)
        else:
            ax.contour(counts,
                       levels,
                       extent=[xbins.min(), xbins.max(), ybins.min(), ybins.max()],
                       cmap=cmap,
                       **kwargs)

        if conf and not shortest:
            coverage_ellipse(ax, funcname1, funcname2, self._results, conf=conf)
        elif conf and shortest:
            coverage_blocks(ax, funcname1, funcname2, self._results, conf=conf, bins=kwargs.get('bins', 100))

        if labeldesc:
            xlabel = self._results.descriptions.get(funcname1, '')
            ylabel = self._results.descriptions.get(funcname2, '')
        else:
            xlabel = report.mathstr_to_latex(funcname1) + report.Unit(xunits).latex(bracket=True)
            ylabel = report.mathstr_to_latex(funcname2) + report.Unit(yunits).latex(bracket=True)

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.ticklabel_format(style='sci', axis='x', scilimits=(-4, 4), useOffset=False)
        ax.ticklabel_format(style='sci', axis='y', scilimits=(-4, 4), useOffset=False)

    def scatter(self, funcname1, funcname2, ax=None, points=10000,
                labeldesc=False, conf=None, shortest=False, **kwargs):
        ''' Scatter plot of samples from two model functions

            Args:
                funcname1 (str): Name of first function
                funcname2 (str): Name of second function
                ax (plt.axes): Matplotlib Axis to plot on
                points (int): Number of points to plot
                labeldesc (bool): Use "description" as axis label instead of variable name
                conf (float): Level of confidence for displaying confidence region
        '''
        _, ax = plotting.initplot(ax)
        xsamples, xunits = unitmgr.split_units(self._results.samples[funcname1])
        ysamples, yunits = unitmgr.split_units(self._results.samples[funcname2])

        kwargs.setdefault('marker', '.')
        kwargs.setdefault('ls', '')
        kwargs.setdefault('markersize', 2)
        kwargs.setdefault('color', 'C2')

        with suppress(ValueError):  # Raises in case where len(x) != len(y) when one output is constant
            ax.plot(xsamples[:points], ysamples[:points], **kwargs)

        if conf and not shortest:
            coverage_ellipse(ax, funcname1, funcname2, self._results, conf=conf)
        elif conf and shortest:
            coverage_blocks(ax, funcname1, funcname2, self._results, conf=conf, bins=kwargs.get('bins', 100))

        if labeldesc:
            xlabel = self._results.descriptions.get(funcname1, '')
            ylabel = self._results.descriptions.get(funcname2, '')
        else:
            xlabel = report.mathstr_to_latex(funcname1) + report.Unit(xunits).latex(bracket=True)
            ylabel = report.mathstr_to_latex(funcname2) + report.Unit(yunits).latex(bracket=True)

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.ticklabel_format(style='sci', axis='x', scilimits=(-4, 4), useOffset=False)
        ax.ticklabel_format(style='sci', axis='y', scilimits=(-4, 4), useOffset=False)

    def converge_mean(self, ax, funcname, div=25, relative=False, **kwargs):
        ''' Convergance plot of average value

            Args:
                ax (plt.axes): Matplotlib Axis to plot on
                funcname (str): Function to plot
                div (int): Number of divisions to plot
                relative (bool): Plot relative to final value
        '''
        kwargs.setdefault('color', 'C0')
        kwargs.setdefault('marker', 'o')
        samples, units = unitmgr.split_units(self._results.samples[funcname])
        step = len(samples) // div
        line = np.empty(div)
        steps = np.empty(div)
        for i in range(div):
            line[i] = samples[:step*(i+1)].mean()
            steps[i] = (i+1)*step
        if relative:
            line = line / line[-1]
        ax.plot(steps, line, **kwargs)

        ax.set_xlabel('Samples')
        if relative:
            ax.set_ylabel(f'{report.mathstr_to_latex(funcname)} (Relative to final)')
        else:
            ax.set_ylabel(f'{report.mathstr_to_latex(funcname)} {report.Unit(units).latex(bracket=True)}')

    def converge_uncert(self, ax, funcname, div=25, relative=False, **kwargs):
        ''' Convergance plot of uncertianty

            Args:
                ax (plt.axes): Matplotlib Axis to plot on
                funcname (str): Function to plot
                div (int): Number of divisions to plot
                relative (bool): Plot relative to final value
        '''
        kwargs.setdefault('color', 'C1')
        kwargs.setdefault('marker', 'o')
        samples, units = unitmgr.split_units(self._results.samples[funcname])
        step = len(samples) // div
        line = np.empty(div)
        steps = np.empty(div)
        for i in range(div):
            line[i] = samples[:step*(i+1)].std(ddof=1)
            steps[i] = (i+1)*step
        if relative:
            line = line / line[-1]
        ax.plot(steps, line, **kwargs)

        ax.set_xlabel('Samples')
        if relative:
            ax.set_ylabel(f'{report.mathstr_to_latex(f"u({funcname})")} (Relative to final)')
        else:
            ax.set_ylabel(f'{report.mathstr_to_latex(f"u({funcname})")} {report.Unit(units).latex(bracket=True)}')


def _get_jointdist(means, covariance, function1, function2):
    ''' Get joint distribution means and Covariance matrix '''
    mean0 = unitmgr.strip_units(means[function1])
    mean1 = unitmgr.strip_units(means[function2])
    Uy = [[unitmgr.strip_units(covariance[function1][function1]),
           unitmgr.strip_units(covariance[function1][function2])],
          [unitmgr.strip_units(covariance[function2][function1]),
           unitmgr.strip_units(covariance[function2][function2])]]
    return (mean0, mean1), Uy


def coverage_ellipse(ax, fname1, fname2, result, conf=.95, N=1000):
    ''' Add 95% coverage ellipse to the plot using JCGM102:2011 Section 7.7.2 '''
    center, Uy = _get_jointdist(result.expected, result.covariance(), fname1, fname2)
    Uy = np.asarray(Uy)
    Linv = linalg.inv(linalg.cholesky(Uy, lower=True))

    # Because this is slow, don't use full MC samples
    yr = np.array(list(result.samples.values())).T[:N]   # Samples
    y = np.array(list(result.expected.values()))   # Expected

    dr2 = np.zeros(N)
    for i, yy in enumerate(yr):
        yr_circle = (Linv @ (yy-y))
        dr2[i] = yr_circle.T @ yr_circle
    dr = np.sqrt(dr2)

    kp = np.quantile(dr, conf)  # Coverage factor

    eigval, eigvec = np.linalg.eigh(Uy)
    width = 2*np.sqrt(eigval[0])*kp
    height = 2*np.sqrt(eigval[1])*kp
    angle = np.degrees(np.arctan2(*eigvec[::-1, 0]))

    ax.add_patch(
        Ellipse(center, width, height, angle=angle,
                color='C1', fill='none', zorder=5, alpha=.3))


def coverage_blocks(ax, fname1, fname2, result, conf=.95, bins=100):
    ''' Add 95% "smallest" coverage region using JCGM102:2011 Section 7.7.4 '''
    xsamples = unitmgr.strip_units(result.samples[fname1])
    ysamples = unitmgr.strip_units(result.samples[fname2])
    counts, ybins, xbins = np.histogram2d(ysamples, xsamples, bins=bins, density=True)
    dx = np.diff(xbins)[0]
    dy = np.diff(ybins)[0]
    area = dx * dy
    sort_idx = np.unravel_index(np.argsort(counts, axis=None), counts.shape)
    iconf = np.where(np.cumsum((counts[sort_idx]*area)[::-1]) > conf)[0][0]
    yidx = sort_idx[0][::-1][:iconf]
    xidx = sort_idx[1][::-1][:iconf]

    for xi, yi in zip(xidx, yidx):
        ax.add_patch(
            Rectangle((xbins[xi], ybins[yi]), dx, dy,
                      color='C1', lw=0, alpha=.3, zorder=5))
