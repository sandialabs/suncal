''' Report Attribute interval methods A3 and S2 '''

import numpy as np

from ...common import report, plotting
from .. import s2models


class ReportIntervalA3:
    ''' Output report for Test Interval (A3) calculation

        Args:
            results (dict): Output from TestInterval.calculate()
    '''
    def __init__(self, results):
        self._results = results

    def _repr_markdown_(self):
        return self.summary().get_md()

    def summary(self, **kwargs):
        ''' Report the results '''
        rows = [['Suggested Interval', report.Number(self._results.interval, fmin=0)],
                ['Calculated Interval', report.Number(self._results.calculated, fmin=0)],
                ['Current Interval Rejection Confidence', f'{self._results.rejection*100:.2f}%'],
                ['True reliability range', f'{self._results.RL*100:.2f}% - {self._results.RU*100:.2f}%'],
                ['Observed Reliability', f'{self._results.Robserved*100:.2f}% '
                                         f'({self._results.intol} / {self._results.n})'],
                ['Number of calibrations used', f'{self._results.n:.0f}']
                ]
        if self._results.unused is not None:
            rows.append(['Rejected calibrations (wrong interval)', f'{self._results.unused:d}'])
        rpt = report.Report(**kwargs)
        rpt.table(rows, hdr=['Parameter', 'Value'])
        return rpt


class ReportIntervalS2:
    ''' Output report for Binomial interval calculation

        Args:
            results (dict): Output from BinomialInterval.calculate()
    '''
    def __init__(self, results):
        self._allresults = results
        self._results = self._allresults._methodresults
        self._best = self._allresults.best
        self._bestmodel = self._allresults.method(self._best)
        self.interval = self._bestmodel.interval
        self.plot = PlotIntervalS2(self._results, self._best)

    def _repr_markdown_(self):
        return self.summary().get_md()

    def summary(self, **kwargs):
        ''' Report the results '''
        rpt = self.allmodels(**kwargs)
        with plotting.plot_figure() as fig:
            fig = self.plot.allmodels(**kwargs)
            rpt.plot(fig)
        rpt.append(self.bins(**kwargs))
        return rpt

    def bins(self, **kwargs):
        ''' Report table of binned data '''
        hdr = ['Range', 'Reliability', 'Number of measurements']
        rows = []

        ti = self._bestmodel.binned.interval
        ri = self._bestmodel.binned.reliability
        ni = self._bestmodel.binned.number
        binleft = self._bestmodel.binned.binleft
        if binleft is None:
            binleft = ti - ti[0]

        for bleft, t, r, n in zip(binleft, ti, ri, ni):
            rows.append([f'{bleft:.0f} - {t:.0f}', f'{r:.3f}', f'{n:.0f}'])
        rpt = report.Report(**kwargs)
        rpt.hdr('Binned reliability data', level=2)
        rpt.table(rows, hdr)
        return rpt

    def model(self, model, **kwargs):
        ''' Report of one model '''
        hdr = ['Interval', 'Model', 'Rejection Confidence',
               f'{self._results[model].conf*100:.1f}% Confidence Interval Range']
        rows = [[f'{self._results[model].interval:.1f}',
                 model,
                 f'{self._results[model].C:.1f}%',
                 '{:.1f} - {:.1f}'.format(*self._results[model].interval_range)]]
        rpt = report.Report(**kwargs)
        rpt.hdr('Best Fit Model', level=2)
        rpt.table(rows, hdr)
        return rpt

    def allmodels(self, **kwargs):
        ''' Report a table of all reliability models for comparison '''
        hdr = ['Reliability Model', 'Interval', 'Rejection Confidence', 'F-Test', 'Figure of Merit']
        rows = []
        for name, r in self._results.items():
            rows.append([name,
                         f'{r.interval:.0f}',
                         f'{r.C:.2f}%',
                         f'{r.accept}',
                         f'{r.G:.2f}'])
        rpt = report.Report(**kwargs)
        rpt.table(rows, hdr)
        return rpt

    def bestmodel(self, **kwargs):
        ''' Report of best model '''
        return self.model(self._best)


class PlotIntervalS2:
    ''' Plot the S2 reliability model fits '''
    def __init__(self, results, best):
        self._results = results
        self._best = best
        self.interval = results[self._best].interval

    def bestmodel(self, ax=None, **kwargs):
        ''' Plot of best model '''
        self.model(self._best, ax=ax, **kwargs)

    def model(self, model, ax=None, axlabels=True, **kwargs):
        ''' Plot individual model '''
        _, ax = plotting.initplot(ax)
        ax.plot(self._results[model].arr.x, self._results[model].arr.y*100, ls='', marker='o')
        if self._results[model].theta is not None:
            xx = np.linspace(0, self._results[model].arr.x.max(), num=100)
            yy = s2models.models[model](xx, *self._results[model].theta)
            ax.plot(xx, yy*100, ls='-')
        if self._results[model].interval > 0:
            ax.axvline(self._results[model].interval, ls=':', color='black')
        ax.axhline(self._results[model].target*100, ls=':', color='black')
        if axlabels:
            ax.set_xlabel('Interval Days')
            ax.set_ylabel('Reliability %')
            ax.set_title(self._best)

    def allmodels(self, fig=None, **kwargs):
        ''' Plot all the models '''
        fig, _ = plotting.initplot(fig)
        fig.clf()
        axs = plotting.axes_grid(len(s2models.models), fig=fig, maxcols=3)
        for ax, modelname in zip(axs, self._results.keys()):
            self.model(modelname, ax=ax, axlabels=False, **kwargs)
            ax.set_title(modelname)
        fig.tight_layout()
        return fig
