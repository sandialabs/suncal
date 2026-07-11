''' Report variables interval calculation '''
import numpy as np

from ...common import report, plotting, ttable
from ..fit import y_pred, u_pred


class ReportFit:
    ''' Report for curve fit to deviation vs time since last cal

        Args:
            fitresult: Results of the curve fit
            model: The VariablesData input
    '''
    def __init__(self, fitresult, model):
        self.fitresult = fitresult
        self.model = model

    def summary(self, **kwargs):
        ''' Generate formatted report '''
        rpt = report.Report(**kwargs)
        unc = np.sqrt(np.diag(self.fitresult.cov))
        rows = [[chr(ord('a')+i), (report.Number(v, matchto=u), ' ± ', report.Number(u))]
                for i, (v, u) in enumerate(zip(self.fitresult.b, unc))]
        rows.append(['Standard Error', report.Number(self.fitresult.syx)])
        rows.append(['Order', str(self.fitresult.order)])
        rows.append(['Degrees Freedom', report.Number(self.fitresult.degf, fmin=1)])
        hdr = ['Parameter', 'Value']
        rpt.table(rows, hdr)
        return rpt


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
        self.fit = ReportFit(results.fit, results.data)

    def summary(self, **kwargs):
        ''' Report the interval and fit parameters '''
        rpt = report.Report(**kwargs)
        if self.interval is not None:
            rpt.hdr(f'Interval: {self.interval:.2f}\n\n', level=3)
        else:
            rpt.hdr('Interval: N/A', level=3)
        if self._results.data.dt is not None:
            with plotting.plot_figure() as fig:
                self.plot(fig)
                rpt.plot(fig)

            with plotting.plot_figure() as fig:
                self.plot_uncertainty(fig)
                rpt.plot(fig)

        rpt.append(self.output(**kwargs))
        return rpt

    def output(self, **kwargs):
        ''' Report output parameters in a table '''
        rpt = report.Report(**kwargs)
        hdr = ['Parameter', 'Value']
        rows = [
            ['Computed Interval', report.Number(self._results.interval, fmin=1)],
            ['Uncertainty Target', str(self._results.target)],
            ['Initial Value', str(self._results.data.y0)],
            ['Initial Uncertainty', f'{self._results.data.u0} (ν = {self._results.data.u0_degf})'],
        ]
        rpt.table(rows, hdr)
        return rpt

    def plot(self, fig=None, **kwargs):
        ''' Plot the interval, fit line, limits, etc. '''
        t = self._results.data.dt
        b = self._results.fit.b
        u0 = self._results.data.u0
        cov = self._results.fit.cov
        syx = self._results.fit.syx
        y0 = self._results.data.y0
        target = self._results.target
        deltas = self._results.data.deltas
        xx = np.linspace(0, max(self.interval, t.max()))
        fit = y_pred(xx, b)
        upred = np.sqrt(u_pred(xx, b, cov, syx)**2 + u0**2)

        fig, ax = plotting.initplot(fig)
        fig.clf()
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(xx, fit+y0, color='C1', label='Fit')
        ax.plot(xx, fit+upred+y0, color='C4', ls='--', label='Standard Uncertainty')
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

    def plot_uncertainty(self, fig=None, **kwargs):
        ''' Plot the uncertainty growth '''
        t = self._results.data.dt
        b = self._results.fit.b
        u0 = self._results.data.u0
        cov = self._results.fit.cov
        syx = self._results.fit.syx
        target = self._results.target
        xx = np.linspace(0, max(self.interval, t.max()))
        upred = np.sqrt(u_pred(xx, b, cov, syx)**2 + u0**2)

        fig, ax = plotting.initplot(fig)
        fig.clf()
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(xx, upred, color='C4', label='Standard Uncertainty')
        ax.axhline(target, color='C2', ls='--', label='Uncertainty Limit')
        ax.axvline(self.interval, color='C3', label='Interval')
        ax.set_xlabel('Time Since Calibration')
        ax.set_ylabel('Uncertainty')
        ax.set_title('Uncertainty Growth')
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
        self.fit = ReportFit(results.fit, results.data)

    def summary(self, **kwargs):
        ''' Report the interval and fit parameters '''
        rpt = report.Report(**kwargs)
        rpt.hdr('Variable Interval Results', level=3)
        rpt.append(self.output(**kwargs))

        if self._results.data.dt is not None:
            with plotting.plot_figure() as fig:
                self.plot(fig)
                rpt.plot(fig)

            with plotting.plot_figure() as fig:
                self.plot_reliability(fig)
                rpt.plot(fig)

            with plotting.plot_figure() as fig:
                self.plot_uncertainty(fig)
                rpt.plot(fig)

        rpt.hdr('Inputs', level=3)
        rpt.append(self.model(**kwargs))
        return rpt

    def model(self, **kwargs):
        ''' Report input parameters to model '''
        rpt = report.Report(**kwargs)
        hdr = ['Parameter', 'Value']
        rows = [
            ['Lower Tolerance', str(self._results.LL)],
            ['Upper Tolerance', str(self._results.UL)],
            ['Initial Value', str(self._results.data.y0)],
            ['Initial Uncertainty', f'{self._results.data.u0} (ν = {self._results.data.u0_degf})'],
            ['Reliability Target', f'{self._results.target*100:.2f} %'],
        ]
        rpt.table(rows, hdr)
        return rpt

    def output(self, **kwargs):
        ''' Report output parameters in a table '''
        rpt = report.Report(**kwargs)
        hdr = ['Parameter', 'Value']
        rows = [
            ['Computed Interval', report.Number(self._results.interval, fmin=1)],
            ['Projected Result', report.Number(self._results.final_value, fmin=1)],
            ['Projected Uncertainty', report.Number(self._results.projected_uncert, fmin=1)],
            ['Projected Degrees Freedom', report.Number(self._results.projected_degf, fmin=1)],
            ['Forecast Uncertainty', report.Number(self._results.forecast_uncert, fmin=1)],
            ['Reliability Target', f'{self._results.target*100:.1f} %'],
        ]
        rpt.table(rows, hdr)
        return rpt

    def plot(self, fig=None, **kwargs):
        ''' Plot the variables fit and suggested interval '''
        t = self._results.data.dt
        deltas = self._results.data.deltas
        y0 = self._results.data.y0
        LL = self._results.LL
        UL = self._results.UL
        target_reliability = self._results.target
        b = self._results.fit.b

        tmax = max(self.interval*1.1, t.max())
        xx = np.linspace(0, tmax)
        fit = y_pred(xx, b, y0=y0)
        sigb, degf = self._results.data.uncertainty(xx, self._results.fit)
        k = np.array([ttable.k_factor(target_reliability, d) for d in degf])
        utotal = k * sigb

        fig, _ = plotting.initplot(fig)
        fig.clf()
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(t, deltas+y0, marker='o', ls='')
        ax.plot(xx, fit, color='C1', ls='-', label='Fit')
        ax.plot(xx, fit+utotal, color='C4', ls='--',
                label=f'{target_reliability*100:.0f}% Uncertainty')
        ax.plot(xx, fit-utotal, color='C4', ls='--')
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

    def plot_reliability(self, fig=None, **kwargs):
        ''' Plot predicted reliability over time '''
        tmax = max(self.interval*1.1, self._results.data.dt.max())
        xx = np.linspace(0, tmax)
        rel = self._results.data.reliability(xx, self._results.LL, self._results.UL, self._results.fit)
        fig, _ = plotting.initplot(fig)
        fig.clf()
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(xx, rel*100)
        ax.axhline(self._results.target*100, ls='--', color='black')
        ax.axvline(self._results.interval, ls='--', color='black')
        ax.set_xlabel('Time Since Calibration')
        ax.set_ylabel('Reliability %')
        return fig

    def plot_uncertainty(self, fig=None, **kwargs):
        ''' Plot uncertainty growth over time '''
        tmax = max(self.interval*1.1, self._results.data.dt.max())
        xx = np.linspace(0, tmax)
        sigb, _ = self._results.data.uncertainty(xx, self._results.fit)
        fig, _ = plotting.initplot(fig)
        fig.clf()
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(xx, sigb)
        ax.set_xlabel('Time Since Calibration')
        ax.set_ylabel('Standard Uncertainty')
        return fig
