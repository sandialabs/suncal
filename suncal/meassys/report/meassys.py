''' Specific Measurement Result Report '''
import numpy as np
from scipy import stats

from ...common import unitmgr, plotting, ttable
from ...common.report import Report, Number, Math
from ...datasets.dataset_model import DataSet


class SystemQuantityReport:
    ''' Report for one Quantity in a Measurement System '''
    def __init__(self, results: 'SystemQuantityResult'):
        self.result = results

    def _repr_markdown_(self):
        return self.summary().get_md()

    def summary(self, **kwargs) -> Report:
        ''' Summary report '''
        rpt = Report(**kwargs)
        rpt.hdr(f'Quantity {self.result.symbol}', level=2)
        if self.result.qty.description:
            rpt.txt(self.result.qty.description)
            rpt.newline()

        expanded = self.result.expanded()
        conf = self.result.meta.get('confidence', .95)
        hdr = ['Parameter', 'Value']
        rows = [
            ['Measured Value', Number(self.result.value, matchto=expanded)],
            ['Standard Uncertainty', Number(self.result.uncertainty)],
            ['Degrees of Freedom', Number(self.result.degrees_freedom, fmin=1)],
            [f'Expanded Uncertainty ({conf*100:.2f} %)', Number(expanded)],
            ['Coverage Factor k', Number(ttable.k_factor(conf, self.result.degrees_freedom))]
        ]
        if (tol := self.result.tolerance) is not None:
            rows.append(['Tolerance', str(tol)])
            rows.append(['Probability of Conformance', Number(self.result.p_conformance*100, fmin=1, postfix=' %')])
        rpt.table(rows, hdr)
        return rpt

    def report_all(self, **kwargs) -> Report:
        ''' Report the summary and uncertainty components '''
        rpt = self.summary()

        if hasattr(self.result.qty, 'typea'):
            # SystemQuantity
            rpt.hdr('Uncertainty Components', level=3)
            rpt.append(self.report_uncertainties(**kwargs))

            if self.result.qty.typea is not None:
                rpt.append(self.report_typea(**kwargs))
            else:
                rpt.append(self.report_plot(**kwargs))
        return rpt

    def _typeb_rows(self) -> list[list[str]]:
        ''' Get rows of Type B table '''
        rows = []
        qty = self.result.qty
        for typeb in qty.typebs:
            uncert = typeb.uncertainty
            conf = self.result.meta.get('confidence', .95)
            rows.append([
                typeb.name,
                typeb.distname.title(),
                Number(uncert),
                Number(uncert * ttable.k_factor(conf, typeb.degf)),
                Number(typeb.degf, fmin=1),
                typeb.description,
            ])
        return rows

    def _typea_row(self) -> list[list[str]]:
        ''' One row of the Type A table '''
        qty = self.result.qty
        if qty.typea is None or qty.typea.size == 0:
            return []

        dset = DataSet(unitmgr.strip_units(qty.typea))
        dset.calculate()
        result = dset.result
        conf = self.result.meta.get('confidence', .95)
        if result.ncolumns > 1:
            uncert = result.uncertainty.stderr
            degf = result.uncertainty.stderr_degf
        else:
            group = result.group(1)
            uncert = group.std_err
            degf = group.degf

        rows = [[
            f'Type A - {result.totalN} Measurements',
            'Normal',
            Number(uncert, fmin=1),
            Number(ttable.k_factor(conf, degf) * uncert, fmin=1),
            Number(degf, fmin=1),
            qty.typea_description,
        ]]
        return rows

    def report_uncertainties(self, **kwargs):
        ''' Report table of uncertainty components '''
        rows = self._typea_row()
        rows.extend(self._typeb_rows())
        hdr = ['Name', 'Distribution', 'Standard Uncertainty', 'Expanded Uncertainty', 'Deg. Freedom', 'Description']
        rpt = Report(**kwargs)
        rpt.table(rows, hdr)
        return rpt

    def report_typeb(self) -> Report:
        ''' Report of Type B uncertainties '''
        rpt = Report()

        hdr = ['Name', 'Standard Uncertainty', 'Expanded Uncertainty', 'Deg. Freedom', 'Description']
        rows = self._typeb_rows()
        if rows:
            rpt.hdr('Type B Uncertainties', level=2)
            rpt.table(rows, hdr)
        return rpt

    def report_typea(self, **kwargs) -> Report:
        ''' Report of details Type A uncertainties '''
        rpt = Report(**kwargs)
        qty = self.result.qty
        if qty.typea is None:
            rpt.txt('No Type A data measured for this quantity.')
            return rpt

        dset = DataSet(unitmgr.strip_units(qty.typea))
        dset.calculate()
        result = dset.result

        hdr = ['Parameter', 'Value']
        if result.ncolumns == 1:
            rpt.hdr('Type A Uncertainty - Repeatability', level=2)
            group = result.group(1)
            q025, q25, q75, q975 = np.quantile(qty.typea, (.025, .25, .75, .975))
            rows = [
                ['Standard Deviation', Number(group.std_dev, fmin=3)],
                ['Standard Error of the Mean', Number(group.std_err, fmin=3)],
                ['Number of Measurements', str(group.degf + 1)],
                ['Degrees of Freedom', Number(group.degf)],
                ['Minimum', Number(qty.typea.min(), fmin=3)],
                ['First Quartile', Number(q25, fmin=3)],
                ['Median', Number(np.median(qty.typea), fmin=3)],
                ['Third Quartile', Number(q75, fmin=3)],
                ['Maximum', Number(qty.typea.max(), fmin=3)],
                ['95% Coverage Interval', f'{Number(q025, fmin=3)}, {Number(q975, fmin=3)}']
            ]
            if qty.autocorrelation:
                acorr = result.group_acorr(1)
                if acorr.r_unc > 1.2:
                    rows.append(['Autocorrelated Uncertainty', Number(acorr.uncert, fmin=3)])
                    rows.append(['Autocorrelation Multiplier', Number(acorr.r_unc, fmin=2)])
                    rows.append(['Autocorrelation Cut-off Lag', str(acorr.nc)])

            rpt.table(rows, hdr)
            fig, _ = plotting.initplot()
            result.report.plot.histogram(fig=fig)
            rpt.plot(fig)

        else:
            rpt.hdr('Type A Uncertainty - Reproducibility', level=2)
            rpt.append(result.report.pooled())
            rpt.append(result.report.summary())
            fig, _ = plotting.initplot()
            result.report.plot.groups(fig=fig)
            rpt.plot(fig)
        return rpt

    def report_plot(self, **kwargs) -> Report:
        ''' Show plot of uncertainty and tolerance '''
        fig, ax = plotting.initplot()
        rpt = Report(**kwargs)
        with plotting.plot_figure() as fig:
            ax = fig.add_subplot(1, 1, 1)
            qty = self.result.qty
            mean = unitmgr.strip_units(self.result.value)
            unc = unitmgr.strip_units(self.result.uncertainty)
            degf = self.result.degrees_freedom
            xx = np.linspace(mean - unc*6, mean + unc*6, num=200)
            pdf = stats.t.pdf(xx, loc=mean, scale=unc, df=degf)
            ax.plot(xx, pdf)
            if qty.tolerance:
                ax.axvline(qty.tolerance.flow, ls='--', color='black')
                ax.axvline(qty.tolerance.fhigh, ls='--', color='black')
            rpt.plot(fig)
        return rpt


class SystemReport:
    ''' Report the full set of specific measurements '''
    def __init__(self, result: 'SystemResult'):
        self.result = result

    def summary(self, **kwargs):
        ''' Get summary report '''
        conf = self.result.confidence
        rpt = Report(**kwargs)
        hdr = ['Symbol', 'Value', 'Standard Unc.', f'Expanded Unc. ({conf*100:.2f} %)',
               'Deg. Freedom', 'Tolerance', 'Probability of Conformance', 'Description']
        rows = []
        for qty in self.result.quantities:
            rows.append([
                Math(qty.symbol),
                Number(qty.value, matchto=qty.uncertainty),
                Number(qty.uncertainty),
                Number(qty.expanded()),
                Number(qty.degrees_freedom, fmin=1),
                str(qty.tolerance) if qty.tolerance else '-',
                Number(qty.p_conformance*100, fmin=1, postfix=' %') if qty.tolerance else '-',
                qty.qty.description
            ])
        rpt.table(rows, hdr)
        return rpt

    def all_summaries(self, **kwargs) -> Report:
        ''' Combine the summary reports for all quantities '''
        rpt = Report(**kwargs)
        for result in self.result.quantities:
            rpt.append(result.report.summary(**kwargs))
        return rpt
