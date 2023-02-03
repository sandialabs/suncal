''' Report results of an uncertainty sweep '''

import matplotlib.pyplot as plt

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


def plotsweep(ax, rpt, xindex=0, funcname=None, label=None):
    ''' Plot the results of a sweep

        Args:
            rpt: The ReportSweep(Gum)(Mc) instance to plot
            xindex: Index of sweep variable to use for x axis
            funcname: Name of output function to plot
            label: Label for plot legend
    '''
    if funcname is None:
        funcname = rpt._results.functionnames[0]

    xvals, xunits = unitmgr.split_units(rpt._sweepvals[xindex])
    yvals, yunits = unitmgr.split_units(rpt._results.expected()[funcname])
    uyvals = unitmgr.strip_units(rpt._results.uncertainties()[funcname])
    ax.errorbar(xvals, yvals, yerr=uyvals, marker='o', label=label)
    ax.set_xlabel(rpt._header_strs[xindex])
    ax.set_ylabel(funcname)
    if xunits:
        ax.set_xlabel(ax.get_xlabel() + report.Unit(xunits).latex(bracket=True))
    if yunits:
        ax.set_ylabel(ax.get_ylabel() + report.Unit(yunits).latex(bracket=True))


class ReportSweepGum:
    ''' Report the results of a GUM sweep '''
    def __init__(self, gumresults):
        self._results = gumresults
        self._header, self._header_strs = make_sweepheader(self._results.sweeplist)
        self._sweepvals = [v['values']*unitmgr.parse_units(v.get('units', '')) for v in self._results.sweeplist]
        self.N = len(self._sweepvals[0])

    def _repr_markdown_(self):
        return self.summary().get_md()

    def summary(self, **kwargs):
        ''' Summary Report '''
        rpt = report.Report(**kwargs)
        sweepvalstrs = [report.Number.number_array(a) for a in self._sweepvals]
        expected = {name: report.Number.number_array(a) for name, a in self._results.expected().items()}
        uncerts = {name: [report.Number(a) for a in x] for name, x in self._results.uncertainties().items()}

        hdr = self._header.copy()
        for funcname in self._results.functionnames:
            hdr.append(report.Math(funcname))
            hdr.append(report.Math(f'u_{funcname}'))

        rows = []
        for i in range(self.N):
            row = [swpvals[i] for swpvals in sweepvalstrs]
            for funcname in self._results.functionnames:
                row.append(expected[funcname][i])
                row.append(uncerts[funcname][i])
            rows.append(row)
        rpt.table(rows, hdr=hdr)
        return rpt

    def plot(self, ax=None, funcname=None, xindex=0):
        ''' Plot the results of the sweep

            Args:
                ax: Matplotlib axis to plot on
                xindex: Index of sweep variable to use for x axis
                funcname: Name of output function to plot
        '''
        fig, ax = plotting.initplot(ax)
        plotsweep(ax, self, xindex=xindex, funcname=funcname)


class ReportSweepMc:
    ''' Report the results of a Monte Carlo uncertainty sweep '''
    def __init__(self, mcresults):
        self._results = mcresults
        self._header, self._header_strs = make_sweepheader(self._results.sweeplist)
        self._sweepvals = [v['values']*unitmgr.parse_units(v.get('units', '')) for v in self._results.sweeplist]
        self.N = len(self._sweepvals[0])

    def _repr_markdown_(self):
        return self.summary().get_md()

    def summary(self, **kwargs):
        ''' Summary Report '''
        rpt = report.Report(**kwargs)
        sweepvalstrs = [report.Number.number_array(a) for a in self._sweepvals]
        expected = {name: report.Number.number_array(a) for name, a in self._results.expected().items()}
        uncerts = {name: [report.Number(a) for a in x] for name, x in self._results.uncertainties().items()}

        hdr = self._header.copy()
        for funcname in self._results.functionnames:
            hdr.append(report.Math(funcname))
            hdr.append(report.Math(f'u_{funcname}'))

        rows = []
        for i in range(self.N):
            row = [swpvals[i] for swpvals in sweepvalstrs]
            for funcname in self._results.functionnames:
                row.append(expected[funcname][i])
                row.append(uncerts[funcname][i])
            rows.append(row)
        rpt.table(rows, hdr=hdr)
        return rpt

    def plot(self, ax=None, funcname=None, xindex=0):
        ''' Plot the results of the sweep

            Args:
                ax: Matplotlib axis to plot on
                xindex: Index of sweep variable to use for x axis
                funcname: Name of output function to plot
        '''
        fig, ax = plotting.initplot(ax)
        plotsweep(ax, self, xindex=xindex, funcname=funcname)


class ReportSweep:
    ''' Report the results of a multi-point sweep uncertainty calculation. '''
    def __init__(self, results):
        self._gumresults = results.gum
        self._mcresults = results.montecarlo
        self._functionnames = self._gumresults.functionnames
        self._variablenames = self._gumresults.variablenames
        self._header, self._header_strs = make_sweepheader(self._gumresults.sweeplist)
        self._sweepvals = [v['values']*unitmgr.parse_units(v.get('units', '')) for v in self._gumresults.sweeplist]
        self.N = len(self._sweepvals[0])

    def _repr_markdown_(self):
        return self.summary().get_md()

    def summary(self, **kwargs):
        ''' Show results (table) of sweep calculation '''
        rpt = report.Report(**kwargs)
        if self._gumresults is not None:
            rpt.hdr('GUM Results', level=2)
            rpt.append(self._gumresults.report.summary(**kwargs))
        if self._mcresults is not None:
            rpt.hdr('Monte Carlo results', level=2)
            rpt.append(self._mcresults.report.summary(**kwargs))
        return rpt

    def summary_withplots(self, **kwargs):
        ''' Report summary, including table AND plot '''
        rpt = report.Report(**kwargs)
        with plt.style.context(plotting.plotstyle):
            rpt.hdr('Sweep Results', level=2)
            rpt.append(self.summary(**kwargs))
            for funcname in self._functionnames:
                fig = plt.figure()
                self.plot(ax=fig, xindex=0, funcname=funcname)
                rpt.plot(fig)
                plt.close(fig)
        return rpt

    all = summary_withplots

    def plot(self, ax=None, funcname=None, xindex=0):
        fig, ax = plotting.initplot(ax)
        if self._gumresults is not None:
            plotsweep(ax, self._gumresults.report, xindex=xindex, funcname=funcname, label='GUM')

        if self._mcresults is not None:
            plotsweep(ax, self._mcresults.report, xindex=xindex, funcname=funcname, label='Monte Carlo')
        ax.legend(loc='best')

    def warnings(self, **kwargs):
        ''' Report any warnings generated during the calculation '''
        return report.Report(**kwargs)

    def describe(self, idx):
        ''' Get description for a single index in the sweep '''
        slist = []
        for i in range(len(self._header)):
            valstrs = report.Number.number_array(self._sweepvals[i])
            slist.append(f'{self._header_strs[i]} = {valstrs[idx].string()}')
        return '; '.join(slist).strip()
