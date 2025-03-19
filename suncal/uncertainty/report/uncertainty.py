''' Generate reports combining GUM and Monte Carlo results '''

import numpy as np

from ...common import unitmgr, report, plotting


class VariablesReport:
    ''' Report echoing input variables only. Applies to either GUM or MC

        Args:
            variables (Variables): Variables instance to report
    '''
    def __init__(self, variables):
        self._variables = variables
        self._variablenames = list(self._variables.expected.keys())

    def _repr_markdown_(self):
        return self.summary().get_md()

    def summary(self, **kwargs):
        ''' Report of input variables and their values/uncertainties '''
        rows = []
        cols = ['Variable', 'Mean', 'Std. Uncertainty', 'Deg. Freedom', 'Description']
        for varname in self._variablenames:
            rows.append([report.Math(varname),
                         report.Number(self._variables.expected[varname],
                                       matchto=self._variables.uncertainty[varname]),
                         report.Number(self._variables.uncertainty[varname]),
                         report.Number(self._variables.degf[varname], n=2),
                         self._variables.descriptions[varname],
                         ])
        rpt = report.Report(**kwargs)
        rpt.table(rows, hdr=cols)
        return rpt

    def components(self, **kwargs):
        ''' Report the uncertainty components '''
        cols = ['Variable', 'Component', 'Standard Uncertainty', 'Deg. Freedom', 'Description']
        rows = []
        for varname in self._variablenames:
            rows.append([report.Math(varname), '&nbsp;', '&nbsp;', '&nbsp;', self._variables.descriptions[varname]])
            for component in self._variables.components[varname]:
                rows.append([
                    '&nbsp;',
                    component['name'],
                    report.Number(component['uncertainty']),
                    report.Number(component['degf']),
                    component['description'],
                ])
        rpt = report.Report(**kwargs)
        rpt.table(rows, hdr=cols)
        return rpt


class ReportUncertainty:
    ''' Report combined GUM and Monte Carlo Uncertainties

        Args:
            results (UncertaintyResults): Results of GUM+MC calculation
    '''
    def __init__(self, results):
        # Both GUM and MC are required
        self._results = results
        self.gum = self._results.gum.report
        self.montecarlo = self._results.montecarlo.report
        self._functionnames = self._results.gum.functionnames
        self._variablenames = self._results.gum.variablenames
        self._noutputs = len(self._functionnames)
        self.variables = VariablesReport(self._results.gum.variables)
        self.plot = UncertaintyPlot(self._results)

    def _repr_markdown_(self):
        return self.summary().get_md()

    def summary(self, conf=.95, **kwargs):
        ''' Summary table. Maybe with plot? '''
        hdr = ['Function', 'Method', 'Nominal', 'Std. Uncertainty', f'{conf*100:.0f}% Coverage', 'k', 'Deg. Freedom']

        rows = []
        gum_expanded = self._results.gum.expanded(conf)
        mc_expanded = self._results.montecarlo.expanded(conf)
        for funcname in self._functionnames:
            rows.append([
                report.Math(funcname),
                'GUM',
                report.Number(self._results.gum.expected[funcname],
                              matchto=self._results.gum.uncertainty[funcname]),
                report.Number(self._results.gum.uncertainty[funcname]),
                ('± ', report.Number(gum_expanded[funcname].uncertainty)),
                f'{gum_expanded[funcname].k:.3f}',
                f'{self._results.gum.degf[funcname]:.1f}'
                ]
            )

            low = mc_expanded[funcname].low
            high = mc_expanded[funcname].high
            uncert = self._results.montecarlo.uncertainty[funcname]
            expanded_range = ('(', report.Number(low, matchto=uncert),
                              ', ', report.Number(high, matchto=uncert), ')')
            rows.append([
                report.Math(funcname),
                'Monte Carlo',
                report.Number(self._results.montecarlo.expected[funcname],
                              matchto=uncert),
                report.Number(uncert),
                expanded_range,
                f'{mc_expanded[funcname].k:.3f}',
                '&nbsp;'
                ]
            )

        rpt = report.Report(**kwargs)
        rpt.table(rows, hdr)

        gumpoc = self._results.gum.prob_conform()
        mcpoc = self._results.montecarlo.prob_conform()
        hdr = ['Function', 'Method', 'Tolerance', 'Probability of Conformance']
        rows = []
        for fname in self._functionnames:
            if fname in gumpoc:
                rows.append((
                    report.Math(fname),
                    'GUM',
                    str(self._results.gum.tolerances.get(fname, 'NA')),
                    report.Number(gumpoc.get(fname)*100, fmin=1, postfix=' %')
                ))
            if fname in mcpoc:
                rows.append((
                    report.Math(fname),
                    'Monte Carlo',
                    str(self._results.montecarlo.tolerances.get(fname, 'NA')),
                    report.Number(mcpoc.get(fname)*100, fmin=1, postfix=' %')
                ))
        if rows:
            rpt.hdr('Tolerances', level=3)
            rpt.table(rows, hdr)

        return rpt

    def summary_withplots(self, **kwargs):
        ''' Summary including plots '''
        rpt = self.summary(**kwargs)
        with plotting.plot_figure() as fig:
            self.plot.pdf(fig=fig)

        rpt.plot(fig)
        rpt.append(self.warnings(**kwargs))
        rpt.append(self.gum.correlation(**kwargs))
        rpt.append(self.montecarlo.correlation(**kwargs))
        return rpt

    def sensitivity(self, **kwargs):
        ''' Tables of GUM vs MC sensitivity coefficients and proportions '''
        gumsens = self._results.gum.sensitivity()
        gumprop = self._results.gum.proportions()
        mcsens = self._results.montecarlo.sensitivity().sensitivity
        mcprop = self._results.montecarlo.sensitivity().proportions

        rpt = report.Report(**kwargs)

        for funcname in self._functionnames:
            rows = []
            for varname in self._variablenames:
                rows.append([
                    report.Math(varname),
                    report.Number(gumsens[funcname][varname], fmin=1),
                    f'{gumprop[funcname][varname]*100:.2f}%',
                    report.Number(mcsens[funcname][varname], fmin=1),
                    f'{mcprop[funcname][varname]*100:.2f}%'
                    ]
                )

            if abs(gumprop[funcname]['residual']) > 1E-4:
                rows.append(['Correlations', '', f'{gumprop[funcname]["residual"]*100:.2f}%', '', ''])

            if self._noutputs > 1:
                rpt.hdr(report.Math(funcname), level=3)
            rpt.table(rows, hdr=['Variable', 'GUM Sensitivity', 'GUM Proportion', 'MC Sensitivity', 'MC Proportion'])
        return rpt

    def expanded(self, conf=.95, k=None, shortest=False, **kwargs):
        ''' Expanded uncertainties of GUM and MC

            Args:
                conf (float): Level of confidence in interval
                k (float): Coverage factor for interval, overrides conf
                shortest (bool): Use shortest instead of symmetric interval for
                  Monte Carlo interval
        '''
        rpt = report.Report(**kwargs)
        rpt.hdr('GUM Approximation', level=3)
        rpt.append(self.gum.expanded(conf=conf, k=k))
        rpt.hdr('Monte Carlo', level=3)
        rpt.append(self.montecarlo.expanded(conf=conf, k=k, shortest=shortest))
        return rpt

    def allinputs(self, **kwargs):
        ''' Report the inputs (Uncertainty Budget page) '''
        rpt = report.Report(**kwargs)
        rpt.hdr('Input Measurements', level=2)
        rpt.append(self.variables.summary())
        rpt.div()
        rpt.hdr('Uncertainty Budget', level=2)
        rpt.append(self.variables.components())
        rpt.div()
        rpt.hdr('Sensitivity Coefficients and Proportions', level=2)
        rpt.append(self.sensitivity())
        return rpt

    def check_validity(self, funcname, ndig=2, conf=.95, full=False):
        ''' Validate GUM by comparing endpoints of 95% coverage interval.

            1. Express u(y) as "a x 10^r" where a has ndig digits and r is integer.
            2. delta = .5 * 10^r
            3. dlow = abs(ymean - uy_gum - ylow_mc); dhi = abs(ymean + uy_gum - yhi_mc)
            4. PASS if dlow < delta and dhi < delta

            Args:
                ndig (int): Number of significant figures for comparison
                conf (float): Level of confidence for comparison (0-1 range)
                full (boolean): Return full set of values, including delta, dlow and dhigh

            Returns:
                valid (boolean): Validity of the GUM approximation compared to Monte-Carlo
                delta (float): Allowable delta between GUM and MC
                dlow (float): Low value abs(ymean - uy_gum - ylow_mc)
                dhi (float): High value abs(ymean + uy_gum - yhi_mc)
                r (int): r value used to find delta
                a (int): a value. u(y) = a x 10^r

            References:
                GUM-S2, Section 8, also NPL Report DEM-ES-011, Chapter 8
        '''
        assert ndig > 0
        gumexpected = self._results.gum.expected[funcname]
        gumexpanded = self._results.gum.expand(funcname, conf=conf)
        mclow, mchigh, _, _ = self._results.montecarlo.expand(funcname, conf=conf)
        gumstandard, units = unitmgr.split_units(self._results.gum.uncertainty[funcname])
        units = 1 if units is None else units

        try:
            r = int(np.floor(np.log10(np.abs(gumstandard))) - (ndig-1))
        except OverflowError:
            delta = r = dlow = dhi = np.nan
        else:
            delta = 0.5 * 10.0**r * units
            dlow = abs((gumexpected - gumexpanded) - mclow)
            dhi = abs((gumexpected + gumexpanded) - mchigh)

        if not full:
            return (dlow < delta) & (dhi < delta)

        fullparams = {'delta': delta,
                      'dlow': dlow,
                      'dhi': dhi,
                      'r': r,
                      'a': (gumstandard/10.**r).astype(int),
                      'gumlo': gumexpected - gumexpanded,
                      'gumhi': gumexpected + gumexpanded,
                      'mclo': mclow,
                      'mchi': mchigh}
        return (dlow < delta) & (dhi < delta), fullparams

    def validity(self, ndig=2, conf=.95, **kwargs):
        ''' Validate the GUM by comparing endpoints

            Args:
                ndig (int): Number of significant figures for comparison
                conf (float): Level of confidence for comparison (0-1 range)
                **kwargs: passed to Report
        '''
        rpt = report.Report(**kwargs)
        rpt.hdr(f'Comparison to Monte Carlo {conf*100:.2f}% Coverage', level=3)
        for funcname in self._functionnames:
            _, params = self.check_validity(funcname=funcname, ndig=ndig, conf=conf, full=True)
            deltastr = report.Number(params['delta'], fmin=1)
            rpt.txt(f'{ndig:d} significant digit{"s" if ndig > 1 else ""}. δ = {deltastr}.\n\n')

            rows = []
            hdr = [f'{conf*100:.2f}% Coverage', 'Lower Limit', 'Upper Limit']
            rows.append(['GUM', report.Number(params['gumlo'], matchto=params['dlow']),
                         report.Number(params['gumhi'], matchto=params['dhi'])])
            rows.append(['MC', report.Number(params['mclo'], matchto=params['dlow']),
                         report.Number(params['mchi'], matchto=params['dhi'])])
            rows.append(['abs(GUM - MC)', report.Number(params['dlow'], matchto=params['dlow']),
                         report.Number(params['dhi'], matchto=params['dhi'])])
            rows.append(['abs(GUM - MC) < δ', '<font color="green">PASS</font>'
                         if params['dlow'] < params['delta'] else '<font color="red">FAIL</font>',
                         '<font color="green">PASS</font>'
                         if params['dhi'] < params['delta'] else '<font color="red">FAIL</font>'])

            if self._noutputs > 1:
                rpt.hdr(report.Math(funcname), level=3)
            rpt.table(rows, hdr=hdr)
        return rpt

    def all(self, setup=None, **kwargs):
        ''' Comprehensive report '''
        rpt = report.Report(**kwargs)
        if setup is None:
            setup = {}  # Use all defaults

        if setup.get('summary', True):
            rpt.hdr('Summary', level=2)
            rpt.append(self.summary(**kwargs))

        if setup.get('outputs', True):
            params = setup.get('outplotparams', {})
            joint = params.get('joint', False)

            with plotting.plot_figure() as fig:
                if joint:
                    self.plot.joint_pdf(fig=fig, overlay=params.get('overlay', False),
                                        cmap=params.get('cmap', 'viridis'),
                                        cmapmc=params.get('cmapmc', 'viridis'),
                                        legend=params.get('legend'))
                else:
                    self.plot.pdf(fig=fig, mchist=not params.get('contour'),
                                  legend=params.get('legend'),
                                  interval=params.get('interval'),
                                  k=params.get('k'),
                                  shortest=params.get('shortest'),
                                  bins=params.get('bins', 100))
                rpt.plot(fig)

        if setup.get('inputs', True):
            rpt.hdr('Standardized Input Values', level=2)
            rpt.append(self.variables.summary(**kwargs))
            rpt.div()

        if setup.get('components', True):
            rpt.hdr('Uncertainty Budget', level=2)
            rpt.append(self.variables.components(**kwargs))
            rpt.div()

        if setup.get('sens', True):
            rpt.hdr('Sensitivity Coefficients', level=2)
            rpt.append(self.sensitivity(**kwargs))
            rpt.div()

        if setup.get('expanded', True):
            rpt.hdr('Expanded Uncertainties', level=2)
            conf = setup.get('conf', .95)
            k = setup.get('k')
            shortest = setup.get('shortest', False)
            rpt.append(self.expanded(conf=conf, k=k, shortest=shortest, **kwargs))

        if setup.get('gumderv', True):
            solve = setup.get('gumvalues', False)
            rpt.hdr('GUM Derivation', level=2)
            rpt.append(self.gum.derivation(solve=solve, **kwargs))

        if setup.get('gumvalid', True):
            ndig = setup.get('gumvaliddig', 2)
            rpt.hdr('GUM Validity', level=2)
            rpt.append(self.validity(ndig=ndig, **kwargs))

        if setup.get('mchist', True):
            params = setup.get('mchistparams', {})
            with plotting.plot_figure() as fig:
                if params.get('joint', False):
                    self.montecarlo.plot.variable_scatter(fig=fig, variables=params.get('inpts'),
                                                          points=params.get('points', 10000))
                else:
                    self.montecarlo.plot.variable_hist(fig=fig, variables=params.get('inpts'),
                                                       bins=params.get('bins', 100))
                rpt.hdr('Monte Carlo Inputs', level=2)
                rpt.plot(fig)

        if setup.get('mcconv', True):
            relative = setup.get('mcconvnorm', False)
            with plotting.plot_figure() as fig:
                self.montecarlo.plot.converge(fig, relative=relative)
                rpt.hdr('Monte Carlo Convergence', level=2)
                rpt.plot(fig)
        return rpt

    def warnings(self, **kwargs):
        ''' Report of warnings raised during calculation '''
        rpt = report.Report(**kwargs)
        if self._results.gum.warns:
            for w in self._results.gum.warns:
                rpt.txt('- ' + w + '\n')
        if self._results.montecarlo.warns:
            for w in self._results.montecarlo.warns:
                rpt.txt('- ' + w + '\n')
        return rpt


class UncertaintyPlot:
    ''' Plots of combined GUM and MC results '''
    def __init__(self, results):
        self._results = results
        self.gum = self._results.gum.report
        self.montecarlo = self._results.montecarlo.report
        self._functionnames = self._results.gum.functionnames
        self._variablenames = self._results.gum.variablenames
        self._noutputs = len(self._functionnames)

    def pdf(self, functions=None, fig=None, mchist=True, legend=True,
            interval=None, k=None, shortest=False, labeldesc=False, bins=100, **kwargs):
        ''' Plot PDF/Histogram of results

            Args:
                functions (list of str): list of functionnames to include
                fig (plt.Figure): Matplotlib Figure
                mchist (bool): Show MC as histogram
                legend (bool): Show legend
                interval (float): Show expanded interval to this level of confidence
                k (float): Show expanded interval to this coverage factor (overrides interval argument)
                shortest (bool): Use shortest interval for MC
                labeldesc (bool): Use "description" as axis label instead of variable name
                bins (int): Number of bins for histogram
                **kwargs: passed to plot
        '''
        fig, _ = plotting.initplot(fig)
        fig.clf()
        if functions is None:
            functions = self._functionnames

        if len(functions) < 1:
            return

        rows = int(np.ceil(len(functions)/3))
        cols = int(np.ceil(len(functions)/rows))

        for plotid, funcname in enumerate(functions):
            ax = fig.add_subplot(rows, cols, plotid+1)
            if mchist:
                mcline = self.montecarlo.plot.axis.hist(funcname, ax=ax, interval=interval, k=k, shortest=shortest, bins=bins, **kwargs)
            else:
                mcline = self.montecarlo.plot.axis.pdf(funcname, ax=ax, interval=interval, k=k, shortest=shortest, **kwargs)
            gumline = self.gum.plot.axis.pdf(funcname, ax=ax, interval=interval, k=k, **kwargs)

            if (tol := self._results.gum.tolerances.get(funcname)):
                if np.isfinite(tol.flow):
                    ax.axvline(tol.flow, color='C3', ls='--', label='Tolerance')
                if np.isfinite(tol.fhigh):
                    ax.axvline(tol.fhigh, color='C3', ls='--')

            if legend:
                # Intervals will take the default legend. Add this as secondary legend.
                ax.add_artist(ax.legend((gumline, mcline), ('GUM Approximation', 'Monte Carlo'),
                                        fontsize=10, loc='upper right'))

                # Put the interval legend back
                if k:
                    ax.legend(loc='lower left', title=f'k = {k:.2f}')
                elif interval:
                    ax.legend(loc='lower left', title=f'{interval*100:.2f}% Coverage')

            ax.ticklabel_format(style='sci', axis='x', scilimits=(-4, 4), useOffset=False)
            ax.set_ylabel('Probability Density')

            if labeldesc:
                ax.set_xlabel(self._results.descriptions.get(funcname, ''))
            else:
                units = unitmgr.get_units(self._results.gum.expected[funcname])
                ax.set_xlabel(report.mathstr_to_latex(funcname) + report.Unit(units).latex(bracket=True))

    def joint_pdf(self, functions=None, fig=None, overlay=False, cmap='viridis', cmapmc='viridis',
                  legend=True, labeldesc=False, **kwargs):
        ''' Plot joint PDF

            Args:
                functions (list of str): list of functionnames to include
                fig (plt.Figure): Matplotlib Figure
                overlay (bool): Plot GUM and MC results on the same axis
                cmap (str): Matplotlib colormap name for GUM results
                cmapmc (str): Matplotlib colormap name for MC results
                legend (bool): Show legend
                labeldesc (bool): Use "description" as axis label instead of variable name
                **kwargs: passed to plot
        '''
        fig, _ = plotting.initplot(fig)
        fig.clf()
        if functions is None:
            functions = self._functionnames

        noutputs = len(functions)
        if noutputs < 2:
            return   # Nothing to plot

        for row, func1 in enumerate(functions):
            for col, func2 in enumerate(functions):
                if col <= row:
                    continue

                if overlay:
                    axgum = fig.add_subplot(noutputs-1, noutputs-1, row*(noutputs-1)+col)
                    axmc = axgum
                else:
                    axgum = fig.add_subplot(noutputs-1, (noutputs-1)*2, row*(noutputs-1)*2+col*2-1)
                    axmc = fig.add_subplot(noutputs-1, (noutputs-1)*2, row*(noutputs-1)*2+col*2)

                levels = self.gum.plot.axis.joint_pdf(
                    func1,
                    func2,
                    ax=axgum,
                    cmap=cmap,
                    legend=legend,
                    labeldesc=labeldesc,
                    **kwargs)
                self.montecarlo.plot.axis.joint_pdf(
                    func1, func2, ax=axmc,
                    fill=overlay,
                    levels=levels,
                    cmap=cmapmc,
                    labeldesc=labeldesc,
                    **kwargs)

                if not overlay:
                    plotting.equalize_scales(axmc, axgum)
                    axmc.set_title('Monte-Carlo')
                    axgum.set_title('GUM Approximation')
        fig.tight_layout()

    def joint_scatter(self, functions=None, fig=None, overlay=False, labeldesc=False,
                      conf=None, **kwargs):
        ''' Plot joint PDF, with Monte Carlo as scatter plot

            Args:
                functions (list of str): list of functionnames to include
                fig (plt.Figure): Matplotlib Figure
                overlay (bool): Plot GUM and MC results on the same axis
                labeldesc (bool): Use "description" as axis label instead of variable name
                conf (float): Level of confidence for displaying confidence region
         '''
        fig, _ = plotting.initplot(fig)
        fig.clf()
        if functions is None:
            functions = self._functionnames

        noutputs = len(functions)
        if noutputs < 2:
            return   # Nothing to plot

        for row, func1 in enumerate(functions):
            for col, func2 in enumerate(functions):
                if col <= row:
                    continue

                if overlay:
                    axgum = fig.add_subplot(noutputs-1, noutputs-1, row*(noutputs-1)+col)
                    axmc = axgum
                else:
                    axgum = fig.add_subplot(noutputs-1, (noutputs-1)*2, row*(noutputs-1)*2+col*2-1)
                    axmc = fig.add_subplot(noutputs-1, (noutputs-1)*2, row*(noutputs-1)*2+col*2)

                self.gum.plot.axis.joint_pdf(func1, func2, ax=axgum, cmap='viridis', legend=True,
                                             labeldesc=labeldesc, conf=conf, **kwargs)
                self.montecarlo.plot.axis.scatter(func1, func2, ax=axmc, labeldesc=labeldesc,
                                                  conf=conf, **kwargs)

                if not overlay:
                    plotting.equalize_scales(axmc, axgum)
                    axmc.set_title('Monte-Carlo')
                    axgum.set_title('GUM Approximation')
        fig.tight_layout()
