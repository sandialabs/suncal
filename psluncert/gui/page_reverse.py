'''
GUI page for calculating reverse uncertainty propagation.
'''
from PyQt5 import QtWidgets, QtGui, QtCore

from .. import reverse
from . import gui_widgets
from . import gui_common
from . import page_uncertprop


class TargetSetupWidget(QtWidgets.QWidget):
    ''' Widget for entering target value, uncertainty for reverse calculations '''
    def __init__(self, ucalc, parent=None):
        super(TargetSetupWidget, self).__init__(parent=parent)
        self.ucalc = ucalc

        self.cmbFunction = QtWidgets.QComboBox()
        self.txtTarget = QtWidgets.QLineEdit('0.0')
        self.txtTargetUnc = QtWidgets.QLineEdit('0.0')
        self.cmbSolveFor = QtWidgets.QComboBox()
        self.update_names()

        validator = QtGui.QDoubleValidator(-1E99, 1E99, 4)
        validator.setNotation(QtGui.QDoubleValidator.StandardNotation | QtGui.QDoubleValidator.ScientificNotation)
        self.txtTarget.setValidator(validator)
        self.txtTargetUnc.setValidator(validator)

        layout = QtWidgets.QFormLayout()
        layout.addRow('Function:', self.cmbFunction)
        layout.addRow('Target Value:', self.txtTarget)
        layout.addRow('Target Uncertainty:', self.txtTargetUnc)
        layout.addRow('Solve For:', self.cmbSolveFor)
        self.setLayout(layout)

        fidx = max(self.ucalc.reverseparams.get('func', 0), 0)
        self.cmbFunction.setCurrentIndex(fidx)
        self.txtTarget.setText(str(self.ucalc.reverseparams.get('targetnom', 1)))
        self.txtTargetUnc.setText(str(self.ucalc.reverseparams.get('targetunc', 1)))
        self.cmbSolveFor.setCurrentIndex(self.cmbSolveFor.findText(self.ucalc.reverseparams.get('solvefor', '')))

        self.cmbFunction.currentIndexChanged.connect(self.update_values)
        self.cmbSolveFor.currentIndexChanged.connect(self.update_values)
        self.txtTarget.editingFinished.connect(self.update_values)
        self.txtTargetUnc.editingFinished.connect(self.update_values)

    def update_names(self):
        ''' Function/variable names have changed. Change entries in comboboxes to match. '''
        current_f = self.cmbFunction.currentText()
        current_v = self.cmbSolveFor.currentText()
        self.cmbFunction.clear()
        self.cmbFunction.addItems(self.ucalc.get_functionnames())
        self.cmbFunction.setCurrentIndex(self.cmbFunction.findText(current_f))
        self.cmbSolveFor.clear()
        self.cmbSolveFor.addItems(self.ucalc.get_baseinputnames())
        self.cmbSolveFor.setCurrentIndex(self.cmbSolveFor.findText(current_v))

    def update_values(self):
        ''' Store target values to ucalc object '''
        self.ucalc.set_reverse(**self.get_values())

    def get_values(self):
        ''' Get reverse/target values. Keys should be compatibile with set_reverse() function '''
        return {'solvefor': self.cmbSolveFor.currentText(),
                'targetnom': float(self.txtTarget.text()),
                'targetunc': float(self.txtTargetUnc.text()),
                'fidx': self.cmbFunction.currentIndex()}


class PageReverseOutput(QtWidgets.QWidget):
    ''' Page for viewing output of reverse calculation '''
    back = QtCore.pyqtSignal()

    def __init__(self, uncCalc, parent=None):
        super(PageReverseOutput, self).__init__(parent)
        self.txtOutput = gui_widgets.MarkdownTextEdit()
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


class UncertReverseWidget(page_uncertprop.UncertPropWidget):
    ''' Uncertainty Propagation in Reverse. Adds Target tab to tabwidget '''
    def __init__(self, item, parent=None):
        assert isinstance(item, reverse.UncertReverse)
        self.uncReverse = item  # Needed for funcchanged which gets called during super()
        self.revsetup = TargetSetupWidget(self.uncReverse)
        super(UncertReverseWidget, self).__init__(item, parent)

        self.menu.removeAction(self.mnuSaveSamples.menuAction())
        self.actNewUnc = QtWidgets.QAction('New forward calculation from model', self)
        self.actNewUnc.triggered.connect(lambda event, x=item: self.newtype.emit(x.get_config(), 'uncertainty'))  # y is bool from triggered event
        self.actSweep.disconnect()
        self.actSweep.triggered.connect(lambda event, x=item: self.newtype.emit(x.get_config(), 'reversesweep'))
        self.menu.insertAction(self.actReverse, self.actNewUnc)
        self.menu.removeAction(self.actReverse)
        self.pgoutputrev = PageReverseOutput(None)
        self.pgoutputrev.back.connect(self.backbutton)
        self.stack.removeWidget(self.pgoutput)
        self.stack.addWidget(self.pgoutputrev)
        self.pginput.tab.insertTab(0, self.revsetup, 'Target')
        if self.uncReverse.longdescription != '':
            self.pginput.tab.setCurrentIndex([self.pginput.tab.tabText(i) for i in range(self.pginput.tab.count())].index('Notes'))
        else:
            self.pginput.tab.setCurrentIndex(0)

    def funcchanged(self, row, fdict):
        ''' Function has changed '''
        super(UncertReverseWidget, self).funcchanged(row, fdict)
        self.revsetup.update_names()

    def calculate(self):
        ''' Run the calculation '''
        valid = True
        if not (self.pginput.funclist.isValid() and self.pginput.inputtree.isValid()):
            valid = False

        if len(self.uncReverse.functions) < 1:
            valid = False

        if valid:
            try:
                self.uncReverse.calculate()
            except (ValueError, RecursionError):
                valid = False

        if valid:
            self.pgoutputrev.txtOutput.setMarkdown(self.uncReverse.out.report_summary(**gui_common.get_rptargs()))
            self.stack.setCurrentIndex(self.PG_OUTPUT)
            self.actSaveReport.setEnabled(True)
            self.mnuSaveSamples.setEnabled(True)
        else:
            QtWidgets.QMessageBox.warning(self, 'Uncertainty Calculator', 'Invalid Input Parameter!')
            self.actSaveReport.setEnabled(False)
            self.mnuSaveSamples.setEnabled(False)

    def get_report(self):
        ''' Get full report of curve fit, using page settings '''
        return self.uncReverse.get_output().report_all(**gui_common.get_rptargs())

    def save_report(self):
        ''' Save full report, asking user for settings/filename '''
        gui_widgets.savemarkdown(self.get_report())
