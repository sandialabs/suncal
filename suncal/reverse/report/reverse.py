''' Report results of a reverse uncertainty calculation '''

import numpy as np
from scipy import stats
import sympy

from ...common import unitmgr, report, plotting


class ReportReverseGum:
    ''' Report a Reverse GUM uncertainty calculation '''
    def __init__(self, gumresults):
        self._results = gumresults

    def _repr_markdown_(self):
        return self.summary().get_md()

    def summary(self, **kwargs):
        ''' Summarize GUM results '''
        rpt = report.Report(**kwargs)
        solvefor_unc = self._results.u_solvefor_value
        solvefor_val = self._results.solvefor_value

        rpt.hdr('GUM reverse uncertainty', level=2)
        rpt.sympy(sympy.Eq(self._results.funcname, self._results.function))
        rpt.txt('\n\n Combined uncertainty:\n\n')
        rpt.sympy(sympy.Eq(self._results.u_fname, self._results.u_forward_expr))
        rpt.txt('\n\nsolved for uncertainty of input:\n\n')
        rpt.sympy(sympy.Eq(self._results.u_solvefor, self._results.u_solvefor_expr))
        rpt.add('\n\n For output value of ',
                report.Number(self._results.f_required, matchto=self._results.uf_required),
                ' ± ',
                report.Number(self._results.uf_required),
                ' (k=1),\n')
        if solvefor_val is None or solvefor_unc is None:
            rpt.txt('No real solution found\n\n')
        else:
            rpt.add('required input value is ',
                    report.Number(solvefor_val, matchto=solvefor_unc),
                    ' ± ',
                    report.Number(solvefor_unc),
                    ' (k=1).\n\n')
        return rpt

    def plot(self, fig=None, **kwargs):
        fig, ax = plotting.initplot(fig)
        mean = unitmgr.strip_units(self._results.solvefor_value)
        std = unitmgr.strip_units(self._results.u_solvefor_value)
        xx = np.linspace(mean-4*std, mean+4*std, num=200)
        yy = stats.norm.pdf(xx, loc=mean, scale=std)
        kwargs.setdefault('label', 'GUM PDF')
        ax.plot(xx, yy, **kwargs)
        ax.set_xlabel(self._results.solvefor)


class ReportReverseMc:
    ''' Report a Reverse Monte Carlo uncertainty calculation '''
    def __init__(self, mcresults):
        self._results = mcresults

    def _repr_markdown_(self):
        return self.summary().get_md()

    def summary(self, **kwargs):
        ''' Summary of Monte Carlo results '''
        rpt = report.Report(**kwargs)
        rpt.hdr('Monte Carlo reverse uncertainty', level=2)
        solvefor_value = self._results.solvefor_value
        solvefor_unc = self._results.u_solvefor_value
        if solvefor_value is None or solvefor_unc is None:
            rpt.txt('No real solution found\n\n')
        else:
            rpt.add('For output value of ',
                    report.Number(self._results.f_required, matchto=self._results.uf_required),
                    ' ± ',
                    report.Number(self._results.uf_required),
                    ' (k=1), required input value is: ',
                    report.Number(solvefor_value, matchto=solvefor_unc),
                    ' ± ',
                    report.Number(solvefor_unc),
                    ' (k=1).\n\n')
        return rpt

    def plot(self, fig=None, bins=100, **kwargs):
        fig, ax = plotting.initplot(fig)
        samples = self._results.mcresults.samples[self._results.solvefor]
        kwargs.setdefault('label', 'GUM PDF')
        ax.hist(unitmgr.strip_units(samples), bins=bins, density=True, **kwargs)
        ax.set_xlabel(self._results.solvefor)


class ReportReverse:
    ''' Class for reporting results of reverse uncertainty calculation
    '''
    def __init__(self, results):
        self._gumresults = results.gum
        self._mcresults = results.montecarlo

    def _repr_markdown_(self):
        return self.summary().get_md()

    def summary(self, **kwargs):
        ''' Summary of both GUM and Monte Carlo results '''
        rpt = self._gumresults.report.summary(**kwargs)
        rpt.div()
        rpt.append(self._mcresults.report.summary(**kwargs))
        return rpt

    def all(self, **kwargs):
        rpt = self.summary(**kwargs)
        with plotting.plot_figure() as fig:
            self.plot(fig)
            rpt.plot(fig)
        return rpt

    def plot(self, fig=None, **kwargs):
        fig, ax = plotting.initplot(fig)
        if self._mcresults is not None:
            self._mcresults.report.plot(fig=fig, label='Monte Carlo', **kwargs)
        if self._gumresults is not None:
            self._gumresults.report.plot(fig=fig, label='GUM', **kwargs)
        ax.legend()
