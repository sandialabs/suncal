''' Reports for MqaQuantities '''
import numpy as np

from ...common.report import Report, Number
from ...common import plotting


class MqaUncertaintyReport:
    ''' Report of uncertainty components '''
    def __init__(self, quantity: 'MqaUncertaintyResult'):
        self.result = quantity

    def summary(self, **kwargs):
        ''' Generate summary report '''
        return self.report(**kwargs)

    def report(self, **kwargs) -> Report:
        ''' Generate a detailed report of uncertainty contributors '''
        rpt = Report(**kwargs)
        hdr = ['Parameter', 'Value']
        rows = [
            ['Standard Deviation', Number(self.result.stdev)],
            ['Degrees of Freedom', Number(self.result.degrees_freedom)],
            ['Expanded Uncertainty', Number(self.result.expanded())],
        ]
        rpt.table(rows, hdr)

        if self.result.gum is not None:
            rpt.append(self.result.gum.report.summary(**kwargs))
            rpt.append(self.result.gum.report.derivation(**kwargs))
        return rpt


class MqaQuantityReport:
    '''  Report for a direct quantity '''
    def __init__(self, quantity: 'MqaQuantityResult'):
        self.result = quantity

    def summary(self, **kwargs) -> Report:
        ''' Generate summary report '''
        return self.report(**kwargs)

    def _rows(self):
        if self.result.uncertainty is None:
            uncert = 0
            expanded = 0

        else:
            uncert = self.result.uncertainty.stdev
            expanded = self.result.uncertainty.expanded()

        rows = []
        if self.result.item.measurand.name:
            rows.append(['Name', self.result.item.measurand.name])

        rows.append(['Testpoint', Number(self.result.item.measurand.testpoint, matchto=uncert)])

        if self.result.item.measurand.units:
            rows.append(['Units', self.result.item.measurand.units])
        rows.extend([
            ['Standard Uncertainty (k=1)', Number(uncert)],
            ['Expanded Uncertainty (k=2)', Number(expanded)],
            ['Tolerance', str(self.result.tolerance)],
            ['Acceptance Limit', str(self.result.guardband)],
            ['Test Uncertainty Ratio', Number(self.result.capability.tur, fmin=1)],
            ['Test Accuracy Ratio', Number(self.result.capability.tar, fmin=1)],])
        if self.result.item.mqa_mode > self.result.item.Mode.BASIC:
            rows.extend([
                ['Beginning of Period Reliability', Number(self.result.reliability.bop.pct*100, fmin=2, postfix=' %')],
                ['Average over Period Reliability', Number(self.result.reliability.aop.pct*100, fmin=2, postfix=' %')],
            ])
        rows.extend([
            ['End-of-period Reliability (Observed)', Number(self.result.eopr.observed.pct*100, fmin=2, postfix=' %')],
            ['End-of-period Reliability (True)', Number(self.result.eopr.true.pct*100, fmin=2, postfix=' %')],
            ['Conditional Probability of False Accept', Number(self.result.risk.cpfa_true*100, fmin=2, postfix=' %')],
            ['Probability of False Accept', Number(self.result.risk.pfa_true*100, fmin=2, postfix=' %')],
            ['Probability of False Reject', Number(self.result.risk.pfr_true*100, fmin=2, postfix=' %')],
        ])
        if self.result.item.utility_enabled:
            rows.append(
                ['Probability of Success', Number(self.result.reliability.success*100, fmin=2, postfix=' %')]
            )
        return rows

    def report(self, chain: bool = True, **kwargs) -> Report:
        ''' Generate a detailed report of uncertainty and reliability '''
        rpt = Report(**kwargs)
        hdr = ['Parameter', 'Value']
        rpt.table(self._rows(), hdr)
        if (chain
                and self.result.uncertainty is not None
                and self.result.uncertainty.parent is not None):
            rpt.append(self.result.uncertainty.parent.report.summary())
        return rpt

    def _cost_rows(self, result, enditem: bool = True, chain: bool = False):
        rows = []
        if result.cost_annual is None:
            return rows

        rows.append(['Annual Test and Calibration', format(result.cost_annual.cal, '.0f')])
        rows.append(['Annual Adjustment', format(result.cost_annual.adj, '.0f')])
        rows.append(['Annual Repair', format(result.cost_annual.rep, '.0f')])
        rows.append(['Total Annual Support', format(result.cost_annual.support, '.0f')])
        if enditem:
            rows.append(['Total Annual Cost', format(result.cost_annual.total, '.0f')])
            rows.append(['Annual Performance', format(result.cost_annual.performance, '.0f')])
        else:
            rows.append(['Total Annual Cost', format(result.cost_annual.total, '.0f')])
        rows.append(['Spares Acquisition', format(result.cost_annual.spare_cost, '.0f')])

        parent = result.uncertainty.parent
        if chain and parent is not None:
            rows.append(['Item', parent.item.measurand.name])
            rows.extend(self._cost_rows(parent, enditem=False, chain=chain))
        return rows

    def report_cost(self, enditem: bool = True, chain: bool = True, **kwargs) -> Report:
        ''' Report costs for one item in the MQA chain '''
        rpt = Report(**kwargs)
        hdr = ['Parameter', 'Value']
        rows = self._cost_rows(self.result, enditem, chain)
        rpt.table(rows, hdr)
        return rpt

    def report_cost_total(self, enditem: bool = True, **kwargs) -> Report:
        ''' Report total costs '''
        costs = self.result.total_costs()
        rows = []
        rows.append(['Annual Test and Calibration', format(costs.cal, '.0f')])
        rows.append(['Annual Adjustment', format(costs.adj, '.0f')])
        rows.append(['Annual Repair', format(costs.rep, '.0f')])
        rows.append(['Total Annual Support', format(costs.support, '.0f')])
        rows.append(['Total Annual Cost', format(costs.total, '.0f')])
        if enditem:
            rows.append(['Annual Performance', format(costs.performance, '.0f')])
        rows.append(['Spares Acquisition', format(costs.spare_cost, '.0f')])
        rpt = Report(**kwargs)
        hdr = ['Parameter', 'Value']
        rpt.table(rows, hdr)
        return rpt

    def report_reliability_plots(self, **kwargs) -> Report:
        ''' Show plot of uncertainty and tolerance '''
        bop = self.result.reliability.bop.pdf
        eop = self.result.eopr.observed.pdf
        fig, ax = plotting.initplot()
        rpt = Report(**kwargs)
        with plotting.plot_figure() as fig:
            ax = fig.add_subplot(1, 1, 1)
            ax.axvline(self.result.testpoint, ls=':', color='gray', label='Testpoint')
            ax.plot(bop._x, bop._y, label='Beginning of Period PDF')
            ax.plot(eop._x, eop._y, label='End of Period PDF')
            ax.axvline(self.result.tolerance.flow, ls='--', color='black')
            ax.axvline(self.result.tolerance.fhigh, ls='--', color='black')
            ax.set_xlabel('Indicated Value')
            ax.set_ylabel('Probability Density')
            ax.legend(loc='best')
        rpt.plot(fig)
        return rpt

    def report_prepost_pdf(self, **kwargs) -> Report:
        ''' Show plot of uncertainty and tolerance '''
        bt = self.result.reliability.pdfs.bt
        pt = self.result.reliability.pdfs.pt
        fig, ax = plotting.initplot()
        rpt = Report(**kwargs)
        with plotting.plot_figure() as fig:
            ax = fig.add_subplot(1, 1, 1)
            ax.axvline(self.result.testpoint, ls=':', color='gray', label='Testpoint')
            ax.plot(bt._x, bt._y, label='Pre-Test PDF')
            ax.plot(pt._x, pt._y, label='Post-Test PDF')
            ax.axvline(self.result.tolerance.flow, ls='--', color='black')
            ax.axvline(self.result.tolerance.fhigh, ls='--', color='black')
            ax.set_xlabel('Indicated Value')
            ax.set_xlabel('Probability Density')
            ax.legend(loc='best')
        rpt.plot(fig)
        return rpt

    def report_utility(self, **kwargs) -> Report:
        ''' Plot the utility function curve '''
        rpt = Report(**kwargs)
        utility = self.result.item.measurand.utility()
        y = utility._y * self.result.item.measurand.psr
        rpt = Report(**kwargs)
        with plotting.plot_figure() as fig:
            ax = fig.add_subplot(1, 1, 1)
            ax.axvline(self.result.testpoint, ls=':', color='gray', label='Testpoint')
            ax.plot(utility._x, y, label='End-item Utility Function')
            ax.axvline(self.result.tolerance.flow, ls='--', color='black')
            ax.axvline(self.result.tolerance.fhigh, ls='--', color='black')
            ax.set_xlabel('Indicated Value')
            ax.set_ylabel('Probability of Successful Outcome')
            ax.legend(loc='best')
        rpt.plot(fig)
        return rpt

    def report_decay(self, **kwargs) -> Report:
        ''' Plot reliability decay over the interval '''
        rpt = Report(**kwargs)
        years = self.result.item.measurement.interval.years
        tt = np.linspace(0, years*2, num=100)

        reliability = [self.result.item.reliability_t(t) for t in tt]
        decay = [r.itp(self.result.tolerance)*100 for r in reliability]
        with plotting.plot_figure() as fig:
            ax = fig.add_subplot(1, 1, 1)
            ax.plot(tt, decay)
            ax.axvline(years, ls=':')
            ax.set_xlabel('Years Since Calibration')
            ax.set_ylabel('True Reliability %')
        rpt.plot(fig)
        return rpt
