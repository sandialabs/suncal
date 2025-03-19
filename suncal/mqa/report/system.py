''' Report for Measured Data Evaluation '''
from ...common.report import Report, Number


class MqaSystemReport:
    ''' Report of Mqa System '''
    def __init__(self, results: 'MeasSysResults'):
        self.result = results

    def _repr_markdown_(self):
        return self.summary().get_md()

    def report_details(self, **kwargs) -> Report:
        ''' Report table of reliability details '''
        rpt = Report(**kwargs)
        for qty in self.result.quantities:
            name = qty.item.measurand.name
            rpt.hdr(name, level=3)
            rpt.append(qty.report.summary(**kwargs))

            if hasattr(qty, 'cost_item'):
                rpt.hdr('Calibration Costs:', level=4)
                rpt.append(qty.report.report_cost(**kwargs))
                rpt.hdr('Total Cost:', level=4)
                rpt.append(qty.report.report_cost_total(**kwargs))
        return rpt

    def summary(self, **kwargs) -> Report:
        ''' Generate summary table '''
        rpt = Report(**kwargs)
        hdr = [
            'Quantity',
            'Testpoint',
            'Tolerance',
            'Equipment',
            'Meas. Uncertainty',
            'TAR',
            'TUR',
            'PFA %']

        rows = []

        def add_qty(name, qty):
            name = name if name else qty.item.measurand.name
            rows.append([
                name,
                Number(qty.testpoint),
                str(qty.tolerance),
                qty.item.measurement.mte.equip_name(),
                ('±', Number(qty.uncertainty.accuracy, fmin=0)),
                Number(qty.capability.tar, fmin=0),
                Number(qty.capability.tur, fmin=0),
                Number(qty.risk.pfa_true*100, fmin=0, postfix=' %')
            ])
            if qty.item.measurement.mte.quantity is not None:
                add_qty('→ ' + qty.item.measurement.mte.quantity.measurand.name,
                        qty.item.measurement.mte.quantity.calculate())
            elif qty.item.measurement.equation is not None:
                for mtename, mte in qty.item.measurement.mteindirect.items():
                    if mte.quantity is not None:
                        add_qty(f'→ {mtename}', mte.quantity)
                    else:
                        rows.append([
                            f'→ {mtename}',
                            qty.item.measurement.testpoints[mtename],
                            '-',
                            str(mte.equipment) if mte.equipment else mte.name,
                            str(mte.accuracy_plusminus),
                            '-',
                            '-',
                            '-'
                        ])

        for qty in self.result.quantities:
            add_qty(None, qty)

        rpt.table(rows, hdr)
        return rpt
