''' Tool/Dialog for computing Risk given TUR and ITP '''
import sys
from PyQt6 import QtWidgets

from . import gui_common
from .widgets import MarkdownTextEdit
from .page_guardband import GuardBandFinderWidget
from ..risk.risk_model import RiskModelSimple


class SimpleRiskWidget(QtWidgets.QDialog):
    ''' Controls for simple-mode risk calculations (TUR, ITP, and GBF) '''
    def __init__(self, parent=None):
        super().__init__(parent=parent)
        gui_common.centerWindow(self, 1000, 600)
        self.setWindowTitle('Simple Risk Calculator')
        self.tur = QtWidgets.QDoubleSpinBox()
        self.tur.setRange(0.1, 1000)
        self.tur.setValue(4.0)
        self.tur.setSingleStep(0.1)
        self.tur.setDecimals(2)
        self.gbfactor = QtWidgets.QDoubleSpinBox()
        self.gbfactor.setRange(-100, 100)
        self.gbfactor.setValue(1.0)
        self.gbfactor.setSingleStep(0.01)
        self.gbfactor.setDecimals(2)
        self.itp = QtWidgets.QDoubleSpinBox()
        self.itp.setRange(0.01, 99.99)
        self.itp.setValue(95.0)
        self.itp.setSingleStep(1)
        self.itp.setDecimals(2)
        self.chkConditional = QtWidgets.QCheckBox('Conditional PFA')
        self.btngb = QtWidgets.QPushButton('Calculate Guardband...')

        self.tur.valueChanged.connect(self.update)
        self.gbfactor.valueChanged.connect(self.update)
        self.itp.valueChanged.connect(self.update)
        self.chkConditional.stateChanged.connect(self.update)
        self.btngb.clicked.connect(self.guardband)

        self.report = MarkdownTextEdit()

        flayout = QtWidgets.QFormLayout()
        flayout.addRow('Test Uncertainty Ratio:', self.tur)
        hlayout = QtWidgets.QHBoxLayout()
        hlayout.addWidget(self.itp)
        flayout.addRow('In-tolerance probability:', hlayout)
        flayout.addRow('Guardband factor:', self.gbfactor)
        rlayout = QtWidgets.QVBoxLayout()
        rlayout.addLayout(flayout)
        rlayout.addWidget(self.chkConditional)
        rlayout.addWidget(self.btngb)
        rlayout.addStretch()

        layout = QtWidgets.QHBoxLayout()
        layout.addLayout(rlayout, stretch=1)
        layout.addWidget(self.report, stretch=3)
        self.setLayout(layout)
        self.update()

    def riskmodel(self):
        return RiskModelSimple(tur=self.tur.value(),
                               itp=self.itp.value()/100,
                               gbf=self.gbfactor.value())

    def update(self):
        ''' Widget was changed, update results '''
        model = self.riskmodel()
        report = model.calculate().report.summary(conditional=self.chkConditional.isChecked())
        self.report.setReport(report)

    def guardband(self):
        ''' Show guardband dialog '''
        dlg = GuardBandFinderWidget()
        ok = dlg.exec()
        if not ok:
            return

        with gui_common.BlockedSignals(self):
            methodargs = dlg.get_method()
            method = methodargs['method']
            model = self.riskmodel()
            if method in ['pfa', 'cpfa']:
                model.guardband_pfa(
                    methodargs.get('pfa', .08),
                    conditional='c' in method)

            elif method == 'pfr':
                self.component.model.guardband_pfr(
                    methodargs.get('pfr', 2)
                )

            elif method in ['mincost', 'minimax']:
                model.guardband_cost(
                    method=method,
                    costfa=methodargs.get('costfa', 100),
                    costfr=methodargs.get('costfr', 10)
                )

            elif method == 'specific':
                model.guardband_specific(methodargs.get('pfa', 0.08))

            else:  # TUR-based
                model.guardband_tur(method=method)

            # With TUR-guardbands, the lower and upper guardbands should
            # be the same. Convert to guardband factor.
            self.gbfactor.setValue(model.gbf)
        self.update()


if __name__ == '__main__':
    # run using: "python -m suncal.gui.tool_risk"
    app = QtWidgets.QApplication(sys.argv)
    main = SimpleRiskWidget()
    main.show()
    app.exec()
