''' Report results of a reverse uncertainty sweep '''

from ...common import unitmgr, report, plotting


def make_sweepheader(sweeplist):
    ''' Generate column headers for input values that are changing '''
    sweepheader = []
    for sweepparams in sweeplist:
        inptname = sweepparams.get('var', None)
        comp = sweepparams.get('comp', 'nom')
        param = sweepparams.get('param', None)
        if comp == 'nom':
            r = report.Math(inptname)
        elif param == 'df':
            r = (report.Math(inptname), ' deg.f')
        elif param in ['unc', 'std']:
            r = report.Math(comp.replace('(', '_').replace(')', ''))
        else:
            r = (report.Math(comp.replace('(', '_').replace(')', '')), ', ', param)
        sweepheader.append(r)

    sweepheader_strs = []
    for hdr in sweepheader:
        r = report.Report()
        try:
            r.add(*hdr, end='')
        except TypeError:
            r.add(hdr, end='')
        sweepheader_strs.append(r.get_md(mathfmt='ascii'))
    return sweepheader, sweepheader_strs


def plotsweep(fig, rpt, xindex=0, label=None):
    ''' Plot a sweep

        Args:
            fig: Figure to plot on
            rpt: ReportSweep(Gum/Mc) object to plot
            xindex: Index of sweep variable to use as x axis
            label: Label for plot legend
    '''
    fig, ax = plotting.initplot(fig)
    xvals, xunits = unitmgr.split_units(rpt._sweepvals[xindex])
    yvals = [unitmgr.strip_units(f.u_solvefor_value) for f in rpt._results.resultlist]
    ax.plot(xvals, yvals, marker='o', label=label)
    _, yunits = unitmgr.split_units(rpt._results.resultlist[0].u_solvefor_value)

    xunitstr = report.Unit(xunits).latex(bracket=True)
    yunitstr = report.Unit(yunits).latex(bracket=True)
    ax.set_xlabel(rpt._header_strs[xindex] + xunitstr)
    ax.set_ylabel(f'Required $u_{{{rpt._results.resultlist[0].solvefor}}}$ {yunitstr}')
    ax.legend(loc='best')


class ReportReverseSweepGum:
    ''' Report results of a reverse GUM uncertainty sweep '''
    def __init__(self, results):
        self._results = results
        self._header, self._header_strs = make_sweepheader(self._results.sweeplist)
        self._sweepvals = [v['values']*unitmgr.parse_units(v.get('units', '')) for v in self._results.sweeplist]
        self.N = len(self._sweepvals[0])

    def _repr_markdown_(self):
        return self.summary().get_md()

    def summary(self, **kwargs):
        ''' Summary Report '''
        solveforvar = self._results.resultlist[0].solvefor
        solvefor = report.Math(solveforvar)
        usolvefor = report.Math(f'u_{solveforvar}')
        hdr = self._header + [solvefor, usolvefor]
        sweepvalstrs = [report.Number.number_array(a) for a in self._sweepvals]
        rows = []
        for i in range(self.N):
            row = [swpvals[i] for swpvals in sweepvalstrs]
            row.append(report.Number(self._results.resultlist[i].solvefor_value))
            row.append(report.Number(self._results.resultlist[i].u_solvefor_value))
            rows.append(row)
        rpt = report.Report(**kwargs)
        rpt.table(rows, hdr)
        return rpt

    def plot(self, fig=None, xindex=0):
        ''' Plot results of the reverse sweep

            Args:
                fig: Figure to plot on
                xindex: Index of sweep variable to use for x axis
        '''
        plotsweep(fig, self, xindex=xindex, label='GUM')


class ReportReverseSweepMc:
    ''' Report results of a reverse Monte Carlo uncertainty sweep '''
    def __init__(self, results):
        self._results = results
        self._header, self._header_strs = make_sweepheader(self._results.sweeplist)
        self._sweepvals = [v['values']*unitmgr.parse_units(v.get('units', '')) for v in self._results.sweeplist]
        self.N = len(self._sweepvals[0])

    def _repr_markdown_(self):
        return self.summary().get_md()

    def summary(self, **kwargs):
        ''' Summary Report '''
        solveforvar = self._results.resultlist[0].solvefor
        solvefor = report.Math(solveforvar)
        usolvefor = report.Math(f'u_{solveforvar}')
        hdr = self._header + [solvefor, usolvefor]
        sweepvalstrs = [report.Number.number_array(a) for a in self._sweepvals]
        rows = []
        for i in range(self.N):
            row = [swpvals[i] for swpvals in sweepvalstrs]
            row.append(report.Number(self._results.resultlist[i].solvefor_value))
            row.append(report.Number(self._results.resultlist[i].u_solvefor_value))
            rows.append(row)
        rpt = report.Report(**kwargs)
        rpt.table(rows, hdr)
        return rpt

    def plot(self, fig=None, xindex=0):
        ''' Plot results of the reverse sweep

            Args:
                fig: Figure to plot on
                xindex: Index of sweep variable to use for x axis
        '''
        plotsweep(fig, self, xindex=xindex, label='Monte Carlo')


class ReportReverseSweep:
    ''' Report the results of a multi-point reverse sweep uncertainty calculation. '''
    def __init__(self, results):
        self._results = results
        self._header, self._header_strs = make_sweepheader(self._results.sweeplist)
        self._sweepvals = [v['values']*unitmgr.parse_units(v.get('units', '')) for v in self._results.sweeplist]
        self.N = len(self._sweepvals[0])

    def summary(self, **kwargs):
        ''' Report table of results of reverse-sweep '''
        r = report.Report(**kwargs)
        if self._results.gum is not None:
            r.hdr('GUM', level=3)
            r.append(self._results.gum.report.summary(**kwargs))

        if self._results.montecarlo is not None:
            r.hdr('Monte Carlo', level=3)
            r.append(self._results.montecarlo.report.summary(**kwargs))
        return r

    def summary_withplots(self, **kwargs):
        ''' Report a summary with plots '''
        rpt = self.summary(**kwargs)
        with plotting.plot_figure() as fig:
            self.plot(fig=fig)
            rpt.plot(fig)
        return rpt

    def warnings(self, **kwargs):
        ''' Report any warnings generated during the calculation '''
        return report.Report(**kwargs)

    def describe(self, idx):
        ''' Get description for a single index in the sweep '''
        slist = []
        for i in range(len(self.sweepheader)):
            valstrs = report.Number.number_array(self.sweepvals[i])
            slist.append(f'{self.sweepheader_strs[i]} = {valstrs[idx].string()}')
        return '; '.join(slist).strip()

    def plot(self, fig=None, xindex=0):
        ''' Plot results of the reverse sweep

            Args:
                fig: Figure to plot on
                xindex: Index of sweep variable to use for x axis
        '''
        plotsweep(fig, self._results.gum.report, xindex=xindex, label='GUM')
        plotsweep(fig, self._results.montecarlo.report, xindex=xindex, label='Monte Carlo')
