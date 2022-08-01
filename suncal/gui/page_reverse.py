'''
GUI page for calculating reverse uncertainty propagation.
'''
import re
from PyQt5 import QtWidgets, QtGui, QtCore
from pint import DimensionalityError, UndefinedUnitError, OffsetUnitCalculusError

from .. import reverse
from . import gui_widgets
from . import gui_common
from . import page_uncertprop


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

    def __init__(self, ucalc, parent=None):
        super().__init__(parent=parent)
        self.ucalc = ucalc

        self.setColumnCount(self.COL_CNT)
        self.setRowCount(self.ROW_CNT)
        self.setHorizontalHeaderLabels(['Parameter', 'Value'])
        self.setStyleSheet(page_uncertprop.TABLESTYLE)
        self.verticalHeader().hide()
        self.setSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum)
        self.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)

        self.setItem(self.ROW_FUNC, self.COL_NAME, gui_widgets.ReadOnlyTableItem('Function'))
        self.setItem(self.ROW_TARG, self.COL_NAME, gui_widgets.ReadOnlyTableItem('Target Value'))
        self.setItem(self.ROW_TUNC, self.COL_NAME, gui_widgets.ReadOnlyTableItem('Target Uncertainty'))
        self.setItem(self.ROW_SOLVEFOR, self.COL_NAME, gui_widgets.ReadOnlyTableItem('Solve For'))

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
        validator.setNotation(QtGui.QDoubleValidator.StandardNotation | QtGui.QDoubleValidator.ScientificNotation)
        self.txtTarget.setValidator(validator)
        self.txtTargetUnc.setValidator(validator)

        fidx = max(self.ucalc.reverseparams.get('func', 0), 0)
        self.cmbFunction.setCurrentIndex(fidx)
        self.txtTarget.setText(str(self.ucalc.reverseparams.get('targetnom', 1)))
        self.txtTargetUnc.setText(str(self.ucalc.reverseparams.get('targetunc', 1)))
        self.cmbSolveFor.setCurrentIndex(self.cmbSolveFor.findText(self.ucalc.reverseparams.get('solvefor', '')))

        self.cmbFunction.currentIndexChanged.connect(self.update_values)
        self.cmbSolveFor.currentIndexChanged.connect(self.update_values)
        self.txtTarget.editingFinished.connect(self.update_values)
        self.txtTargetUnc.editingFinished.connect(self.update_values)
        self.fixSize()

    def fixSize(self):
        height = max(self.horizontalHeader().height()+20, self.verticalHeader().length() + self.horizontalHeader().height())
        self.setFixedHeight(height)

    def update_names(self):
        ''' Function/variable names have changed. Change entries in comboboxes to match. '''
        current_f = self.cmbFunction.currentText()
        current_v = self.cmbSolveFor.currentText()
        self.cmbFunction.clear()
        self.cmbFunction.addItems(self.ucalc.get_functionnames())
        self.cmbFunction.setCurrentIndex(self.cmbFunction.findText(current_f))
        self.cmbSolveFor.clear()
        self.cmbSolveFor.addItems(self.ucalc.model.inputnames)
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
        super().__init__(parent)
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
        super().__init__(item, parent)

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
        self.pginput.panel.insert_widget('Reverse Target Value', self.revsetup, 4)
        self.pginput.panel.expand('Reverse Target Value')

    def funcchanged(self, row, fdict):
        ''' Function has changed '''
        super().funcchanged(row, fdict)
        self.revsetup.update_names()

    def calculate(self):
        ''' Run the calculation '''
        valid = True
        if not self.pginput.isValid():
            QtWidgets.QMessageBox.warning(self, 'Uncertainty Calculator', 'Invalid Input Parameter!')
            valid = False

        elif len(self.uncReverse.model.exprs) < 1:
            QtWidgets.QMessageBox.warning(self, 'Uncertainty Calculator', 'Need at least one measurement function to calculate.')
            valid = False

        elif self.uncReverse.model.check_circular():
            msg = 'Circular reference in function definitions'

        elif self.revsetup.cmbSolveFor.currentText() == '':
            QtWidgets.QMessageBox.warning(self, 'Uncertainty Calculator', 'Please define solve-for parameter in the Target tab.')
            valid = False

        msg = ''
        try:
            self.uncReverse.model.check_dimensionality()
        except (TypeError, DimensionalityError, UndefinedUnitError) as e:
            msg = 'Units Error: {}'.format(e)
        except OffsetUnitCalculusError as e:
            badunit = re.findall(r'\((.+ )', str(e))[0].split()[0].strip(', ')
            msg = 'Ambiguous unit {}. Try "{}".'.format(badunit, 'delta_{}'.format(badunit))
        except RecursionError:
            msg = 'Error - possible circular reference in function definitions'

        if msg:
            QtWidgets.QMessageBox.warning(self, 'Uncertainty Calculator', msg)
            valid = False

        if not valid:
            self.actSaveReport.setEnabled(False)
            self.mnuSaveSamples.setEnabled(False)
            return

        try:
            self.uncReverse.calculate()
        except (ValueError, RecursionError):
            msg = 'Error computing solution!'
            valid = False
        except OffsetUnitCalculusError as e:
            badunit = re.findall(r'\((.+ )', str(e))[0].split()[0].strip(', ')
            msg = 'Ambiguous unit {}. Try "{}".'.format(badunit, 'delta_{}'.format(badunit))
            valid = False
        except RecursionError:
            msg = 'Error - possible circular reference in function definitions'
            valid = False

        if valid:
            self.pgoutputrev.txtOutput.setReport(self.uncReverse.out.report_summary())
            self.stack.setCurrentIndex(self.PG_OUTPUT)
            self.actSaveReport.setEnabled(True)
            self.mnuSaveSamples.setEnabled(True)
        else:
            QtWidgets.QMessageBox.warning(self, 'Uncertainty Calculator', msg)
            self.actSaveReport.setEnabled(False)
            self.mnuSaveSamples.setEnabled(False)

    def get_report(self):
        ''' Get full report of curve fit, using page settings '''
        return self.uncReverse.get_output().report_all()

    def save_report(self):
        ''' Save full report, asking user for settings/filename '''
        gui_widgets.savereport(self.get_report())
