''' Report variables interval calculation '''

import numpy as np

from ...common import report, plotting, ttable
from ..fit import y_pred, u_pred


class ReportIntervalVariables:
    ''' Report for both methods of interval

        Args:
            uncertainty: Results of uncertainty target method
            reliability: Results of reliability target method
    '''
    def __init__(self, results):
        self.uncertainty = ReportIntervalVariablesUncertainty(results.uncertaintytarget)
        self.reliability = ReportIntervalVariablesReliability(results.reliabilitytarget)
        self.fit = ReportFit(results.uncertaintytarget, results.uncertaintytarget)

    def table(self, **kwargs):
        ''' Generate summary report '''
        hdr = ['Method', 'Interval', 'Predicted value at end of interval']
        rows = []
        if self.uncertainty is not None:
            eop_val, eop_unc = self.uncertainty.eop()
            rows.append(['Uncertainty Target',
                         f'{self.uncertainty.interval:.2f}' if self.uncertainty.interval else 'N/A',
                         ((report.Number(eop_val, matchto=eop_unc),
                           ' ± ',
                           report.Number(eop_unc),
                           f' (k = {self.uncertainty._results.k:.2f})'))
                         ])
        if self.reliability._results is not None:
            eop_val, eop_unc = self.reliability.eop()
            rows.append(['Reliability Target',
                         f'{self.reliability.interval:.2f}' if self.reliability.interval else 'N/A',
                         ((report.Number(eop_val, matchto=eop_unc),
                           ' ± ',
                           report.Number(eop_unc),
                           f' (k = {self.reliability._results.k:.2f})'))
                         ])
        rpt = report.Report(**kwargs)
        rpt.table(rows, hdr)
        return rpt

    def summary(self, **kwargs):
        ''' Generate formatted report including plots '''
        rpt = self.table(**kwargs)
        if self.uncertainty._results is not None and self.uncertainty._results.dt is not None:
            with plotting.plot_figure() as fig:
                self.uncertainty.plot(fig=fig)
                rpt.plot(fig)

        if self.reliability._results is not None and self.reliability._results.dt is not None:
            with plotting.plot_figure() as fig:
                self.reliability.plot(fig=fig)
                rpt.plot(fig)

        rpt.div()
        rpt.append(self.params(**kwargs))
        with plotting.plot_figure() as fig:
            self.fit.plot(fig=fig)
            rpt.plot(fig)
        rpt.append(self.fit.summary(**kwargs))
        return rpt

    def params(self, **kwargs):
        ''' Report parameters used in the calculation, including
            table of dt, dy values
        '''
        x = deltas = None
        if self.reliability._results is not None:
            x = self.reliability._results.dt
            deltas = self.reliability._results.deltas
        elif self.uncertainty._results is not None:
            x = self.uncertainty._results.dt
            deltas = self.uncertainty._results.deltas

        rpt = report.Report(**kwargs)
        if x is not None:
            rpt.hdr('Deviation Values', level=3)
            rows = []
            idx = np.argsort(x)
            for x, d in zip(x[idx], deltas[idx]):
                rows.append([report.Number(x, fmin=0), report.Number(d, fmin=0)])
            rpt.table(rows, ['Time since calibration', 'Deviation from prior'])
        return rpt


class ReportFit:
    ''' Report for curve fit to deviation vs time since last cal

        Args:
            uncertainty (ReportIntervalVariablesUncertainty):
                Results of uncertainty target method
            reliability (ReportIntervalVariablesReliability):
                Results of reliability target method
    '''
    def __init__(self, uncertainty, reliability):
        results = None
        if uncertainty is not None:
            results = uncertainty
        elif reliability is not None:
            results = reliability

        if results is not None:
            self.t = results.dt
            self.deltas = np.asarray(results.deltas)
            self.y0 = results.y0
            self.b = results.b
            self.cov = results.cov
            self.syx = results.syx
            self.u0 = results.u0

    def summary(self, **kwargs):
        ''' Generate formatted report '''
        rpt = report.Report(**kwargs)
        rpt.hdr('Fit line', level=3)
        unc = np.sqrt(np.diag(self.cov))
        rows = [[chr(ord('a')+i), report.Number(v, matchto=u), report.Number(u)]
                for i, (v, u) in enumerate(zip(self.b, unc))]
        hdr = ['Parameter', 'Value', 'Std. Uncertainty']
        rpt.table(rows, hdr=hdr)
        rpt.txt('Standard Error:   ')
        rpt.num(self.syx)
        return rpt

    def plot(self, fig=None, conf=.95):
        ''' Plot fit line '''
        xx = np.linspace(0, self.t.max())
        fit = y_pred(xx, self.b)
        k = ttable.k_factor(conf, len(self.t)-len(self.b))
        upred = k*np.sqrt(u_pred(xx, self.b, self.cov, self.syx)**2 + self.u0**2)

        fig, ax = plotting.initplot(fig)
        ax.plot(xx, fit, color='C1', label='Fit')
        ax.plot(xx, fit+upred, color='C4', ls='--', label=f'{conf*100:.0f}% Uncertainty (k={k:.2f})')
        ax.plot(xx, fit-upred, color='C4', ls='--')
        ax.plot(self.t, self.deltas, marker='o', ls='')
        ax.set_xlabel('Time Since Calibration')
        ax.set_ylabel('Deviation from Prior')
        ax.legend(fontsize=12, bbox_to_anchor=(1, 1))
        return fig


class ReportIntervalVariablesUncertainty:
    ''' Report for Variables Uncertainty Target method

        Args:
            results (dict): Dictionary of results from
              VariablesInterval.calc_uncertainty_target()
    '''
    def __init__(self, results):
        self._results = results
        if results is not None:
            self.interval = self._results.interval

    def eop(self):
        ''' Get end-of-period value and uncertainty '''
        u0 = self._results.u0 / self._results.k
        y0 = self._results.y0
        b = self._results.b
        cov = self._results.cov
        syx = self._results.syx
        kvalue = self._results.k
        upred = kvalue * np.sqrt(u_pred(self.interval, b, cov, syx)**2 + u0**2)
        ypred = y0 + y_pred(self.interval, b)
        return ypred, upred

    def summary(self, **kwargs):
        ''' Report the interval and fit parameters '''
        rpt = report.Report(**kwargs)
        if self.interval is not None:
            rpt.hdr(f'Interval: {self.interval:.2f}\n\n', level=3)
        else:
            rpt.hdr('Interval: N/A', level=3)
        if self._results.dt is not None:
            with plotting.plot_figure() as fig:
                self.plot(fig)
                rpt.plot(fig)
        return rpt

    def plot(self, fig=None, **kwargs):
        ''' Plot the interval, fit line, limits, etc. '''
        t = self._results.dt
        b = self._results.b
        u0 = self._results.u0 / self._results.k
        kvalue = self._results.k
        cov = self._results.cov
        syx = self._results.syx
        y0 = self._results.y0
        target = self._results.target
        deltas = self._results.deltas
        xx = np.linspace(0, max(self.interval, t.max()))
        fit = y_pred(xx, b)
        upred = kvalue * np.sqrt(u_pred(xx, b, cov, syx)**2 + u0**2)

        fig, ax = plotting.initplot(fig)
        fig.clf()
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(xx, fit+y0, color='C1', label='Fit')
        ax.plot(xx, fit+upred+y0, color='C4', ls='--', label=f'Uncertainty (k={kvalue:.2f})')
        ax.plot(xx, fit-upred+y0, color='C4', ls='--')
        ax.plot(xx, fit+target+y0, color='C2', label='Uncertainty Limit')
        ax.plot(xx, fit-target+y0, color='C2')
        ax.plot(t, deltas+y0, marker='o', ls='')
        ax.axvline(self.interval, color='C3', label='Interval')
        ax.set_xlabel('Time Since Calibration')
        ax.set_ylabel('Predicted Value')
        ax.set_title('Uncertainty Target')
        ax.legend(fontsize=12, bbox_to_anchor=(1, 1))
        return fig


class ReportIntervalVariablesReliability:
    ''' Report for Interval Reliability Target method

        Args:
            results (dict): Dictionary of results from
              VariablesInterval.calc_reliability_target()
    '''
    def __init__(self, results):
        self._results = results
        if results is not None:
            self.interval = results.interval

    def eop(self):
        ''' Get end-of-period value and uncertainty '''
        k = self._results.k
        u0 = self._results.u0 / k
        b = self._results.b
        cov = self._results.cov
        syx = self._results.syx
        y0 = self._results.y0
        upred = k * np.sqrt(u_pred(self.interval, b, cov, syx)**2 + u0**2)
        ypred = y0 + y_pred(self.interval, b)
        return ypred, upred

    def summary(self, **kwargs):
        ''' Report the interval and fit parameters '''
        rpt = report.Report(**kwargs)
        if self.interval is not None:
            rpt.hdr(f'Interval: {self.interval:.2f}\n\n', level=3)
        else:
            rpt.hdr('Interval: N/A', level=3)
        if self._results.dt is not None:
            with plotting.plot_figure() as fig:
                self.plot(fig)
                rpt.plot(fig)
        return rpt

    def plot(self, fig=None, **kwargs):
        ''' Plot the variables fit and suggested interval '''
        t = self._results.dt
        k = self._results.k
        u0 = self._results.u0 / k
        b = self._results.b
        cov = self._results.cov
        syx = self._results.syx
        y0 = self._results.y0
        deltas = self._results.deltas
        LL = self._results.LL
        UL = self._results.UL
        conf = self._results.conf

        tmax = max(self.interval, t.max())
        xx = np.linspace(0, tmax)
        fit = y_pred(xx, b, y0=y0)
        upred = np.sqrt(u_pred(xx, b, cov, syx)**2 + u0**2)

        fig, _ = plotting.initplot(fig)
        fig.clf()
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(t, deltas+y0, marker='o', ls='')
        ax.plot(xx, fit, color='C1', ls='-', label='Fit')
        ax.plot(xx, fit+k*upred, color='C4', ls='--',
                label=f'{conf*100:.0f}% Uncertainty (k={k:.2f})')
        ax.plot(xx, fit-k*upred, color='C4', ls='--')
        ax.set_title('Reliability Target')

        if LL is not None:
            ax.axhline(LL, color='C0', label='Tolerance Limit')
        if UL is not None:
            ax.axhline(UL, color='C0')
        ax.axvline(self.interval, color='C3', label='Interval')
        ax.set_xlabel('Time Since Calibration')
        ax.set_ylabel('Predicted Value')
        ax.legend(fontsize=12, bbox_to_anchor=(1, 1))
        return fig
