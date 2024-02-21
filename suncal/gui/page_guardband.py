''' Guardband Finder Widgets '''
from PyQt6 import QtWidgets

from .widgets import SpinBoxLabelWidget


class TabTur(QtWidgets.QWidget):
    ''' TUR-based guardband options tab '''
    def __init__(self, parent=None):
        super().__init__(parent)
        self.dobbert = QtWidgets.QRadioButton('Dobbert Managed 2% PFA')
        self.rss = QtWidgets.QRadioButton('RDS')
        self.rp10 = QtWidgets.QRadioButton('NCSL RP10')
        self.test = QtWidgets.QRadioButton('95% Test Uncertainty')
        self.fourtoone = QtWidgets.QRadioButton('Same as 4:1')
        self.dobbert.setChecked(True)
        layout = QtWidgets.QGridLayout()
        layout.addWidget(QtWidgets.QLabel('Select Guardband Method'), 0, 0, 1, 2)
        layout.addWidget(self.dobbert, 1, 0)
        layout.addWidget(QtWidgets.QLabel('<b>k = 1 - M<sub>2%</sub>/TUR</b>'), 1, 1)
        layout.addWidget(self.rss, 2, 0)
        layout.addWidget(QtWidgets.QLabel('<b>k = √(1-1/TUR<sup>2</sup>)</b>'), 2, 1)
        layout.addWidget(self.rp10, 3, 0)
        layout.addWidget(QtWidgets.QLabel('<b>k = 1.25 - 1/TUR</b>'), 3, 1)
        layout.addWidget(self.test, 4, 0)
        layout.addWidget(QtWidgets.QLabel('<b>k = 1 - 1/TUR</b>'), 4, 1)
        layout.addWidget(self.fourtoone, 5, 0)
        self.setLayout(layout)

    def get_method(self):
        ''' Get guardbanding method '''
        kargs = {}
        if self.dobbert.isChecked():
            kargs['method'] = 'dobbert'
        elif self.rss.isChecked():
            kargs['method'] = 'rss'
        elif self.rp10.isChecked():
            kargs['method'] = 'rp10'
        elif self.test.isChecked():
            kargs['method'] = 'test'
        elif self.fourtoone.isChecked():
            kargs['method'] = '4:1'
        return kargs


class TabPfa(QtWidgets.QWidget):
    ''' PFA-based guardband options tab '''
    def __init__(self, parent=None):
        super().__init__(parent)
        self.pfaval = SpinBoxLabelWidget(label='Target PFA', value=0.8)
        self.pfaval.setValue(0.8)  # in percent
        self.pfaval.setRange(.01, 99.99)
        self.pfaval.setDecimals(2)
        self.pfaval.setSingleStep(0.1)
        self.optUnconditional = QtWidgets.QRadioButton('Unconditional PFA')
        self.optConditional = QtWidgets.QRadioButton('Conditional PFA')
        self.chkOptimize = QtWidgets.QCheckBox('Minimize PFR (may result in asymmetric guardband)')
        self.chkNegative = QtWidgets.QCheckBox('Allow negative guardbands')
        self.optUnconditional.setChecked(True)
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.pfaval)
        layout.addWidget(self.optUnconditional)
        layout.addWidget(self.optConditional)
        layout.addWidget(self.chkOptimize)
        layout.addWidget(self.chkNegative)
        layout.addStretch()
        self.setLayout(layout)

    def get_method(self):
        ''' Get guardbanding method '''
        kargs = {}
        kargs['pfa'] = self.pfaval.value() / 100
        if self.optUnconditional.isChecked():
            kargs['method'] = 'pfa'
        else:
            kargs['method'] = 'cpfa'
        kargs['optimize'] = self.chkOptimize.isChecked()
        kargs['allownegative'] = self.chkNegative.isChecked()
        return kargs


class TabPfr(QtWidgets.QWidget):
    ''' PFR-based guardband options tab '''
    def __init__(self, parent=None):
        super().__init__(parent)
        self.pfrval = SpinBoxLabelWidget(label='Target PFR', value=0.8)
        self.pfrval.setValue(2.0)  # in percent
        self.pfrval.setRange(.01, 99.99)
        self.pfrval.setDecimals(2)
        self.pfrval.setSingleStep(0.1)
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.pfrval)
        layout.addStretch()
        self.setLayout(layout)

    def get_method(self):
        ''' Get guardbanding method '''
        kargs = {}
        kargs['pfr'] = self.pfrval.value() / 100
        kargs['method'] = 'pfr'
        return kargs


class TabCost(QtWidgets.QWidget):
    ''' Cost-based guardband options tab '''
    def __init__(self, parent=None):
        super().__init__(parent)
        self.facost = SpinBoxLabelWidget(label='Cost of False Accept', value=100)
        self.frcost = SpinBoxLabelWidget(label='Cost of False Reject', value=10)
        self.facost.setRange(.01, 1000000)
        self.frcost.setRange(.01, 1000000)
        self.facost.setSingleStep(1)
        self.frcost.setSingleStep(1)
        self.facost.setValue(100)
        self.frcost.setValue(10)
        self.optMincost = QtWidgets.QRadioButton('Minimize Expected/Average Cost')
        self.optMinimax = QtWidgets.QRadioButton('Minimize Maximum Cost')
        self.optMincost.setChecked(True)
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.facost)
        layout.addWidget(self.frcost)
        layout.addWidget(self.optMincost)
        layout.addWidget(self.optMinimax)
        layout.addStretch()
        self.setLayout(layout)

    def get_method(self):
        ''' Get guardbanding method '''
        kargs = {}
        if self.optMincost.isChecked():
            kargs['method'] = 'mincost'
        else:
            kargs['method'] = 'minimax'
        kargs['facost'] = self.facost.value()
        kargs['frcost'] = self.frcost.value()
        return kargs


class TabSpecific(QtWidgets.QWidget):
    ''' Specific-Risk guardband options tab '''
    def __init__(self, parent=None):
        super().__init__(parent)
        self.pfaval = SpinBoxLabelWidget(label='Worst-case specific risk', value=2)
        self.pfaval.setValue(2)  # in percent
        self.pfaval.setRange(.01, 99.99)
        self.pfaval.setDecimals(2)
        self.pfaval.setSingleStep(0.1)
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.pfaval)
        layout.addStretch()
        self.setLayout(layout)

    def get_method(self):
        ''' Get guardbanding method '''
        return {'method': 'specific',
                'pfa': self.pfaval.value() / 100}


class GuardBandFinderWidget(QtWidgets.QDialog):
    ''' Widget providing options for calculating a guardband '''
    def __init__(self, parent=None):
        super().__init__(parent)
        font = self.font()
        font.setPointSize(10)
        self.setFont(font)
        self.setWindowTitle('Calculate Guardband')
        self.tabPfa = TabPfa()
        self.tabPfr = TabPfr()
        self.tabTur = TabTur()
        self.tabCost = TabCost()
        self.tabSpecific = TabSpecific()
        self.tab = QtWidgets.QTabWidget()
        self.tab.addTab(self.tabPfa, 'PFA')
        self.tab.addTab(self.tabPfr, 'PFR')
        self.tab.addTab(self.tabTur, 'TUR')
        self.tab.addTab(self.tabCost, 'Cost')
        self.tab.addTab(self.tabSpecific, 'Specific')
        self.buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Ok |
                                                  QtWidgets.QDialogButtonBox.StandardButton.Cancel)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(QtWidgets.QLabel('Select Guardband Type:'))
        layout.addWidget(self.tab)
        layout.addStretch()
        layout.addWidget(self.buttons)
        self.setLayout(layout)

    def get_method(self):
        return self.tab.currentWidget().get_method()
