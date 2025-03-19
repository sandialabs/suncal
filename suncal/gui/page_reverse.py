'''
GUI page for calculating reverse uncertainty propagation.
'''
from PyQt6 import QtWidgets, QtGui, QtCore

from . import widgets
from . import gui_styles
from . import page_uncert
from .help_strings import UncertHelp


class TargetSetupWidget(QtWidgets.QTableWidget):
    ''' Widget for entering target value, uncertainty for reverse calculations '''
    COL_NAME = 0
    COL_VALUE = 1
    COL_CNT = 2

    ROW_FUNC = 0
    ROW_TARG = 1
    ROW_TUNC = 2
    ROW_SOLVEFOR = 3
    ROW_CNT = 4

    def __init__(self, component, parent=None):
        super().__init__(parent=parent)
        self.component = component

        font = self.font()
        font.setPointSize(12)
        self.setFont(font)
        self.setColumnCount(self.COL_CNT)
        self.setRowCount(self.ROW_CNT)
        self.setHorizontalHeaderLabels(['Parameter', 'Value'])
        self.verticalHeader().hide()
        self.setSizePolicy(QtWidgets.QSizePolicy.Policy.Minimum, QtWidgets.QSizePolicy.Policy.Minimum)
        self.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        self.setItem(self.ROW_FUNC, self.COL_NAME, widgets.ReadOnlyTableItem('Function'))
        self.setItem(self.ROW_TARG, self.COL_NAME, widgets.ReadOnlyTableItem('Target Value'))
        self.setItem(self.ROW_TUNC, self.COL_NAME, widgets.ReadOnlyTableItem('Target Uncertainty'))
        self.setItem(self.ROW_SOLVEFOR, self.COL_NAME, widgets.ReadOnlyTableItem('Solve For'))

        self.cmbFunction = QtWidgets.QComboBox()
        self.txtTarget = QtWidgets.QLineEdit('0.0')
        self.txtTargetUnc = QtWidgets.QLineEdit('0.0')
        self.cmbSolveFor = QtWidgets.QComboBox()

        # I don't know why these widgets don't pick up the parent font size...
        font = self.cmbFunction.font()
        font.setPointSize(10)
        self.cmbFunction.setFont(font)
        self.txtTarget.setFont(font)
        self.txtTargetUnc.setFont(font)
        self.cmbSolveFor.setFont(font)

        self.setCellWidget(self.ROW_FUNC, self.COL_VALUE, self.cmbFunction)
        self.setCellWidget(self.ROW_SOLVEFOR, self.COL_VALUE, self.cmbSolveFor)
        self.setCellWidget(self.ROW_TARG, self.COL_VALUE, self.txtTarget)
        self.setCellWidget(self.ROW_TUNC, self.COL_VALUE, self.txtTargetUnc)
        self.update_names()
        self.resizeColumnsToContents()

        validator = QtGui.QDoubleValidator(-1E99, 1E99, 4)
        validator.setNotation(QtGui.QDoubleValidator.Notation.ScientificNotation)
        self.txtTarget.setValidator(validator)
        self.txtTargetUnc.setValidator(validator)

        fidx = max(self.component.model.reverseparams.get('func', 0), 0)
        self.cmbFunction.setCurrentIndex(fidx)
        self.txtTarget.setText(str(self.component.model.reverseparams.get('targetnom', 1)))
        self.txtTargetUnc.setText(str(self.component.model.reverseparams.get('targetunc', 1)))
        self.cmbSolveFor.setCurrentIndex(
            self.cmbSolveFor.findText(self.component.model.reverseparams.get('solvefor', '')))
        self.fixSize()

    def fixSize(self):
        ''' Adjust size of widget '''
        height = max(self.horizontalHeader().height()+20,
                     self.verticalHeader().length() + self.horizontalHeader().height())
        self.setFixedHeight(height)

    def update_names(self, fnames=None):
        ''' Function/variable names have changed. Change entries in comboboxes to match. '''
        if fnames is None:
            fnames = self.component.model.model.functionnames
        current_f = self.cmbFunction.currentText()
        current_v = self.cmbSolveFor.currentText()
        self.cmbFunction.clear()
        self.cmbFunction.addItems(fnames)
        if current_f in fnames:
            self.cmbFunction.setCurrentIndex(self.cmbFunction.findText(current_f))
        else:
            self.cmbFunction.setCurrentIndex(0)
        self.cmbSolveFor.clear()
        self.cmbSolveFor.addItems(self.component.model.model.variables.names)
        self.cmbSolveFor.setCurrentIndex(self.cmbSolveFor.findText(current_v))

    def get_target(self):
        try:
            nom = float(self.txtTarget.text())
        except (ValueError, TypeError):
            nom = 0
        try:
            unc = float(self.txtTargetUnc.text())
        except (ValueError, TypeError):
            unc = 0

        return {'solvefor': self.cmbSolveFor.currentText(),
                'targetnom': nom,
                'targetunc': unc,
                'funcname': self.cmbFunction.currentText()}


class PageReverseOutput(QtWidgets.QWidget):
    ''' Page for viewing output of reverse calculation '''
    back = QtCore.pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.txtOutput = widgets.MarkdownTextEdit()
        self.btnBack = QtWidgets.QPushButton('Back')
        self.btnBack.clicked.connect(self.goback)

        llayout = QtWidgets.QVBoxLayout()
        llayout.addStretch()
        llayout.addWidget(self.btnBack)
        rlayout = QtWidgets.QHBoxLayout()
        rlayout.addLayout(llayout)
        rlayout.addWidget(self.txtOutput)
        self.setLayout(rlayout)

    def goback(self):
        ''' Back button pressed '''
        self.back.emit()

    def update(self, result, **kwargs):
        ''' Update the report '''
        self.txtOutput.setReport(result.report.all())

    def outputupdate(self):
        pass


class UncertReverseWidget(page_uncert.UncertPropWidget):
    ''' Uncertainty Propagation in Reverse. Adds Target tab to tabwidget '''
    def __init__(self, component, parent=None):
        self.component = component
        self.revsetup = TargetSetupWidget(self.component)
        super().__init__(component, parent)

        self.menu.removeAction(self.mnuSaveSamples.menuAction())
        self.actNewUnc = QtGui.QAction('New &forward calculation from model', self)
        self.actNewUnc.triggered.connect(lambda event, x=component: self.newtype.emit(x.get_config(), 'uncertainty'))
        self.actSweep.disconnect()
        self.actSweep.triggered.connect(lambda event, x=component: self.newtype.emit(x.get_config(), 'reversesweep'))
        self.menu.insertAction(self.actReverse, self.actNewUnc)
        self.menu.removeAction(self.actReverse)
        self.stack.removeWidget(self.pgoutput)
        self.pgoutput = PageReverseOutput(None)
        self.pgoutput.back.connect(self.backbutton)
        self.stack.addWidget(self.pgoutput)
        self.pginput.panel.insert_widget('Reverse Target Value', self.revsetup, 4)
        self.pginput.panel.expand('Reverse Target Value')
        self.pginput.funclist.funcchanged.connect(self.funcchanged)

    def funcchanged(self, config):
        ''' Function has changed '''
        fnames = [f['name'] for f in config]
        self.revsetup.update_names(fnames)  # Change functions in dropdown
        self.update_proj_config()
        self.revsetup.update_names()  # Change variables in dropdown

    def calculate(self):
        ''' Run the calculation '''
        if self.revsetup.cmbSolveFor.currentText() == '':
            QtWidgets.QMessageBox.warning(self, 'Suncal', 'Please define solve-for parameter in the Target tab.')
            self.actSaveReport.setEnabled(False)
            self.mnuSaveSamples.setEnabled(False)
            return
        self.update_proj_config()
        super().calculate()

    def get_config(self):
        ''' Get configuration from page '''
        config = self.pginput.get_config()
        reverseparams = self.revsetup.get_target()
        funcnames = [f['name'] for f in config.get('functions', [])]
        funcconfig = config['functions'][funcnames.index(reverseparams['funcname'])]
        units = funcconfig.get('units', None)
        reverseparams['targetunits'] = units
        config.update({'reverse': reverseparams})
        return config

    def get_report(self):
        ''' Get full report of curve fit, using page settings '''
        return self.component.result.report.all()

    def save_report(self):
        ''' Save full report, asking user for settings/filename '''
        with gui_styles.LightPlotstyle():
            widgets.savereport(self.get_report())

    def help_report(self):
        if self.stack.m_next == 1:
            return UncertHelp.reverse_output()
        return UncertHelp.reverse_input()
