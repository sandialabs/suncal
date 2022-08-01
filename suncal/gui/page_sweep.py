'''
Pages for sweeping uncertainty propagation and reverse propagations.
'''
from contextlib import suppress
import numpy as np
from PyQt5 import QtWidgets, QtGui, QtCore

from .. import sweeper
from . import gui_common
from . import gui_widgets
from . import page_uncertprop
from . import page_reverse
from . import page_dataimport


class StartStopCountWidget(QtWidgets.QDialog):
    ''' Dialog for defining a sweep from start/stop/step values '''
    def __init__(self, title=None, parent=None):
        super().__init__(parent=parent)
        font = self.font()
        font.setPointSize(10)
        self.setFont(font)
        self.setWindowTitle(title if title else 'Enter Range')
        self.start = QtWidgets.QLineEdit()
        self.stop = QtWidgets.QLineEdit()
        self.num = QtWidgets.QLineEdit()
        self.buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        validator = QtGui.QDoubleValidator(-1E99, 1E99, 4)
        validator.setNotation(QtGui.QDoubleValidator.StandardNotation | QtGui.QDoubleValidator.ScientificNotation)
        self.start.setValidator(validator)
        self.stop.setValidator(validator)
        self.num.setValidator(QtGui.QIntValidator(0, 1E5))
        flayout = QtWidgets.QFormLayout()
        flayout.addRow('Start:', self.start)
        flayout.addRow('Stop:', self.stop)
        flayout.addRow('Count:', self.num)
        layout = QtWidgets.QVBoxLayout()
        layout.addLayout(flayout)
        layout.addWidget(self.buttons)
        self.setLayout(layout)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)

    def get_range(self):
        ''' Get array of values. Blank array if not all params defined. '''
        try:
            return np.linspace(float(self.start.text()), float(self.stop.text()), int(self.num.text()))
        except (TypeError, ValueError):
            return np.array([])


class SweepSetupTable(gui_widgets.FloatTableWidget):
    ''' Table for defining sweep variables '''
    def __init__(self, calc=None, parent=None):
        super().__init__(parent=parent, movebyrows=True, paste_multicol=False)
        self.sweepcalc = calc
        self.inptlist = []
        self.setStyleSheet(page_uncertprop.TABLESTYLE)

        # This is stupid and ugly but otherwise the keyboard navigation doesn't work with a table
        # embedded in a tree. It works fine when table isn't embedded in tree.
        self.setEditTriggers(QtWidgets.QTableWidget.AllEditTriggers)

        # Load initial values from saved file
        self.setColumnCount(len(self.sweepcalc.sweeplist))
        for swpidx, swp in enumerate(self.sweepcalc.sweeplist):
            values = swp.get('values', [])
            if len(values) > self.rowCount():
                self.setRowCount(len(values))

            for i, val in enumerate(values):
                self.setItem(i, swpidx, QtWidgets.QTableWidgetItem(str(val)))
            var = swp.get('var')
            comp = swp.get('comp')
            param = swp.get('param')
            if var == 'corr':
                sweepstr = 'corr({}, {})'.format(swp.get('var1'), swp.get('var2'))
            elif comp == 'nom':
                sweepstr = 'Mean({})'.format(swp.get('var'))
            elif param == 'df':
                sweepstr = 'df({})'.format(comp)
            else:
                sweepstr = 'Unc({}, {}, {})'.format(swp.get('var'), comp, param)
            self.setHorizontalHeaderItem(swpidx, QtWidgets.QTableWidgetItem(sweepstr))

        self.valueChanged.connect(self.update_sweepparams)

    def set_inptlist(self, inptlist):
        ''' Set the list of inputs '''
        self.inptlist = inptlist

    def clear(self):
        ''' Clear the table '''
        super().clear()
        self.setRowCount(1)
        self.setColumnCount(0)

    def contextMenuEvent(self, event):
        ''' Right-click menu '''
        menu = QtWidgets.QMenu(self)
        actAddCol = menu.addAction('Add sweep column')
        actRemCol = menu.addAction('Remove sweep column')
        actFill = menu.addAction('Fill column...')
        actImport = menu.addAction('Import Sweep List...')
        menu.addSeparator()
        actInsert = menu.addAction('Insert Row')
        actRemove = menu.addAction('Remove Row')
        menu.addSeparator()
        actCopy = menu.addAction('Copy')
        actPaste = menu.addAction('Paste')
        actPaste.setEnabled(QtWidgets.QApplication.instance().clipboard().text() != '')
        actAddCol.triggered.connect(self.addcol)
        actRemCol.triggered.connect(self.remcol)
        actFill.triggered.connect(self.filldata)
        actImport.triggered.connect(self.importdata)
        actPaste.triggered.connect(self._paste)
        actCopy.triggered.connect(self._copy)
        actInsert.triggered.connect(self._insertrow)
        actRemove.triggered.connect(self._removerow)
        menu.popup(QtGui.QCursor.pos())

    def remcol(self):
        ''' Remove the selected column '''
        col = self.currentColumn()
        if col >= 0:
            self.blockSignals(True)
            self.removeColumn(col)
            self.blockSignals(False)
            self.update_sweepparams()

    def addcol(self):
        ''' Add a column to the sweep table '''
        w = SweepParamWidget(self.inptlist)
        status = w.exec_()
        if status:
            self.blockSignals(True)
            col = self.columnCount()
            self.setColumnCount(col+1)
            sweepstr = w.get_sweepstr()
            self.setHorizontalHeaderItem(col, QtWidgets.QTableWidgetItem(sweepstr))
            self.setCurrentCell(0, col)
            self.blockSignals(False)
        return status

    def importdata(self):
        ''' Import sweep array from the project into the selected column '''
        col = self.currentColumn()
        if col < 0 and self.columnCount() == 0:
            ok = self.addcol()
            col = 0
            if not ok:
                return
        elif col < 0:
            col = 0

        dlg = page_dataimport.ArraySelectWidget(singlecol=True, project=self.sweepcalc.project)
        ok = dlg.exec_()
        if ok:
            sweep = dlg.get_array().get('y')
            if sweep is not None:
                self.blockSignals(True)
                if len(sweep) > self.rowCount():
                    self.setRowCount(len(sweep))

                # Clear out all rows in column
                for i in range(self.rowCount()):
                    self.setItem(i, col, QtWidgets.QTableWidgetItem(''))

                # Fill rows from range
                for i, val in enumerate(sweep):
                    self.item(i, col).setText(str(val))
                self.blockSignals(False)
                self.update_sweepparams()
                self.resizeColumnsToContents()

    def filldata(self):
        ''' Fill selected column with start/stop/step values '''
        col = self.currentColumn()
        if col < 0 and self.columnCount() == 0:
            # No columns defined, add one
            ok = self.addcol()
            col = 0
            if not ok:
                return
        elif col < 0:
            # Sweep the first column by default
            col = 0

        dlg = StartStopCountWidget(title=self.horizontalHeaderItem(col).text())
        ok = dlg.exec_()
        if ok:
            self.blockSignals(True)
            rng = dlg.get_range()
            if len(rng) > self.rowCount():
                self.setRowCount(len(rng))

            # Clear out all rows in column
            for i in range(self.rowCount()):
                self.setItem(i, col, QtWidgets.QTableWidgetItem(''))

            # Fill rows from range
            for i, val in enumerate(rng):
                self.item(i, col).setText(str(val))
            self.blockSignals(False)
            self.update_sweepparams()


    def update_sweepparams(self):
        ''' Update UncertSweep model with currently displayed parameters '''
        sweeplist = []
        for col in range(self.columnCount()):
            sweepstr = self.horizontalHeaderItem(col).text()
            vals = [self.item(i, col).text() for i in range(self.rowCount()) if self.item(i, col) and self.item(i, col).text() != '']
            try:
                vals = np.array([float(v) for v in vals])
            except ValueError:
                return
            start = sweepstr.find('(')
            params = sweepstr[start+1:-1].split(', ')
            d = {'mode': sweepstr[:start],
                 'values': vals,
                 'varname': params[0]}
            if len(params) > 1:
                d['comp'] = params[1]
            if len(params) > 2:
                d['param'] = params[2]
            sweeplist.append(d)

        self.sweepcalc.clear_sweeps()
        for params in sweeplist:
            if params['mode'] == 'Mean':
                self.sweepcalc.add_sweep_nom(params['varname'], params['values'])
            elif params['mode'] == 'Unc':
                self.sweepcalc.add_sweep_unc(params['varname'], params['values'], params['comp'], params['param'])
            elif params['mode'] == 'df':
                self.sweepcalc.add_sweep_df(params['varname'], params['values'])
            elif params['mode'] == 'corr':
                self.sweepcalc.add_sweep_corr(params['varname'], params['comp'], params['values'])


class SweepParamWidget(QtWidgets.QDialog):
    ''' Widget for selecting what is being swept in this column '''
    def __init__(self, inptlist, parent=None):
        super().__init__(parent=parent)
        font = self.font()
        font.setPointSize(10)
        self.setFont(font)
        self.inptlist = inptlist
        self.setWindowTitle('Add Parameter Sweep')
        self.mode = QtWidgets.QComboBox()
        self.mode.addItems(['Mean', 'Uncertainty', 'Degrees of Freedom', 'Correlation'])
        self.varname = QtWidgets.QComboBox()
        self.var2 = QtWidgets.QComboBox()
        self.var2.setVisible(False)
        self.unccomp = QtWidgets.QComboBox()
        self.uncparam = QtWidgets.QComboBox()
        self.lblcomp = QtWidgets.QLabel('Component:')
        self.lblparam = QtWidgets.QLabel('Parameter:')
        self.unccomp.setVisible(False)
        self.uncparam.setVisible(False)
        self.lblparam.setVisible(False)
        self.lblcomp.setVisible(False)
        self.buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        self.buttons.rejected.connect(self.reject)
        self.buttons.accepted.connect(self.accept)
        self.varname.addItems([v.name for v in inptlist])
        self.var2.addItems([v.name for v in inptlist])

        unclayout = QtWidgets.QHBoxLayout()
        unclayout.addWidget(self.lblcomp)
        unclayout.addWidget(self.unccomp)
        unclayout.addStretch()
        u2layout = QtWidgets.QHBoxLayout()
        u2layout.addWidget(self.lblparam)
        u2layout.addWidget(self.uncparam)
        u2layout.addStretch()
        layout = QtWidgets.QVBoxLayout()
        llayout = QtWidgets.QHBoxLayout()
        llayout.addWidget(QtWidgets.QLabel('Sweep Parameter:'))
        llayout.addWidget(self.mode)
        layout.addLayout(llayout)
        vlayout = QtWidgets.QHBoxLayout()
        vlayout.addWidget(QtWidgets.QLabel('Variable:'))
        vlayout.addWidget(self.varname)
        vlayout.addWidget(self.var2)
        vlayout.addStretch()
        layout.addLayout(vlayout)
        layout.addLayout(unclayout)
        layout.addLayout(u2layout)
        layout.addWidget(self.buttons)
        self.setLayout(layout)
        self.mode.currentIndexChanged.connect(self.modechange)
        self.varname.currentIndexChanged.connect(self.varchanged)
        self.unccomp.currentIndexChanged.connect(self.compchanged)
        self.varchanged()
        self.compchanged()

    def varchanged(self):
        ''' Variable selection has changed '''
        self.blockSignals(True)
        name = self.varname.currentText()
        varnames = [v.name for v in self.inptlist]
        idx = varnames.index(name)
        comps = [u.name for u in self.inptlist[idx].uncerts]
        self.unccomp.clear()
        if len(comps) > 0:
            self.unccomp.addItems(comps)
        else:
            self.unccomp.addItems(['u({})'.format(name)])
        self.blockSignals(False)

    def compchanged(self):
        ''' Component selection changed '''
        self.blockSignals(True)
        varname = self.varname.currentText()
        varidx = [v.name for v in self.inptlist].index(varname)
        compname = self.unccomp.currentText()
        if compname:
            try:
                compidx = [u.name for u in self.inptlist[varidx].uncerts].index(compname)
            except ValueError:   # No uncertainty components!
                items = []
            else:
                distname = self.inptlist[varidx].uncerts[compidx].distname
                if distname in ['normal', 't']:
                    items = ['unc', 'k']
                else:
                    items = self.inptlist[varidx].uncerts[compidx].required_args
            self.uncparam.clear()
            self.uncparam.addItems(items)
        self.blockSignals(False)

    def get_sweepstr(self):
        ''' Get string representation of sweep '''
        if self.mode.currentText() == 'Mean':
            return 'Mean({})'.format(self.varname.currentText())
        elif self.mode.currentText() == 'Degrees of Freedom':
            return 'df({})'.format(self.varname.currentText())
        elif self.mode.currentText() == 'Correlation':
            return 'corr({}, {})'.format(self.varname.currentText(), self.var2.currentText())
        elif self.mode.currentText() == 'Uncertainty':
            return 'Unc({}, {}, {})'.format(self.varname.currentText(), self.unccomp.currentText(), self.uncparam.currentText())
        else:
            assert False

    def modechange(self):
        ''' Mode (mean, uncertainty, correlation, etc.) of sweep has changed '''
        mode = self.mode.currentText()
        if mode in ['Mean', 'Degrees of Freedom']:
            self.var2.setVisible(False)
            self.unccomp.setVisible(False)
            self.uncparam.setVisible(False)
            self.lblparam.setVisible(False)
            self.lblcomp.setVisible(False)
        elif mode == 'Uncertainty':
            self.var2.setVisible(False)
            self.unccomp.setVisible(True)
            self.uncparam.setVisible(True)
            self.lblparam.setVisible(True)
            self.lblcomp.setVisible(True)
        elif mode == 'Correlation':
            self.var2.setVisible(True)
            self.unccomp.setVisible(False)
            self.uncparam.setVisible(False)
            self.lblparam.setVisible(False)
            self.lblcomp.setVisible(False)
        else:
            assert False


class SweepSlider(QtWidgets.QWidget):
    ''' Widget for showing sweep index slider '''
    valueChanged = QtCore.pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider.valueChanged.connect(self.valueChanged)
        self.lblValue = QtWidgets.QLabel()
        div = gui_widgets.QHLine()
        layout = QtWidgets.QVBoxLayout()
        slayout = QtWidgets.QHBoxLayout()
        slayout.addWidget(QtWidgets.QLabel('Sweep Index:'))
        slayout.addWidget(self.slider)
        layout.addLayout(slayout)
        layout.addWidget(self.lblValue)
        layout.addWidget(div)
        self.setLayout(layout)

    def setLblTxt(self, txt):
        ''' Set the label indicating the current sweep value '''
        self.lblValue.setText(txt)

    def __getattr__(self, name):
        ''' Get all other attributes from the table widget '''
        return getattr(self.slider, name)


class PageOutputSweep(page_uncertprop.PageOutput):
    ''' Page for viewing output of sweep. Adds a slider to select
        individual runs in the sweep
    '''
    def __init__(self, sweepReport=None, parent=None):
        super().__init__(uncCalc=None, parent=parent)
        self.sweepReport = sweepReport
        self.slider = SweepSlider()
        self.slider.valueChanged.connect(self.sliderchange)
        self.oplayout.insertWidget(1, self.slider)  # Add just below combobox

    def set_sweepreport(self, sweepreport):
        ''' Set the report to show '''
        self.sweepReport = sweepreport
        self.slider.setRange(0, self.sweepReport.N-1)
        self.slider.setValue(0)
        self.update(self.sweepReport.outputlist[0])
        self.slider.setLblTxt(self.sweepReport.get_single_desc(0))

    def sliderchange(self):
        ''' Slider was changed, update the unccalc report '''
        i = self.slider.value()
        self.update(self.sweepReport.outputlist[i])
        # TODO: update slider to show math image instead of text only, get_single_desc would return latex
        self.slider.setLblTxt(self.sweepReport.get_single_desc(i))
        self.outputupdate()

    def outputupdate(self):
        ''' Override PageOutput update to show sweep report in summary. Other pages
            stay the same
        '''
        option = self.outputSelect.currentText()
        if option == 'Summary':
            r = self.sweepReport.report_summary(uy=':')
            self.txtOutput.setReport(r)
            self.outputStack.setCurrentIndex(1)
            self.slider.setVisible(False)
        else:
            self.slider.setVisible(True)
            super().outputupdate()


class UncertSweepWidget(page_uncertprop.UncertPropWidget):
    ''' Uncertainty Propagation Sweep. Adds sweep tab to tabwidget '''
    def __init__(self, item, parent=None):
        assert isinstance(item, sweeper.UncertSweep)
        self.uncSweep = item
        self.sweepsetup = SweepSetupTable(self.uncSweep)
        super().__init__(item.unccalc, parent)

        self.menu.removeAction(self.mnuSaveSamples.menuAction())
        self.actNewUnc = QtWidgets.QAction('New single calculation from sweep', self)
        self.actNewUnc.triggered.connect(lambda event, x=item: self.newtype.emit(x.get_config(), 'uncertainty'))
        self.actReverse.disconnect()
        self.actReverse.triggered.connect(lambda event, x=item: self.newtype.emit(x.get_config(), 'reversesweep'))
        self.menu.insertAction(self.actSweep, self.actNewUnc)
        self.menu.removeAction(self.actSweep)
        self.pgoutputsweep = PageOutputSweep(None)
        self.pgoutputsweep.back.connect(self.backbutton)
        self.stack.removeWidget(self.pgoutput)
        self.stack.addWidget(self.pgoutputsweep)
        _, buttons = self.pginput.panel.insert_widget('Sweep', self.sweepsetup, 4, buttons=True)
        buttons.plusclicked.connect(self.sweepsetup.addcol)
        buttons.minusclicked.connect(self.sweepsetup.remcol)
        self.pginput.panel.expand('Sweep')

    def funcchanged(self, row, fdict):
        ''' Sweep function has changed '''
        super().funcchanged(row, fdict)
        self.sweepsetup.set_inptlist(self.uncSweep.unccalc.inputs)
        with suppress(AttributeError):  # May not be defined yet
            self.actNewUnc.setEnabled(True)

    def clearinput(self):
        ''' Clear all the input/output values '''
        self.sweepsetup.clear()
        self.actNewUnc.setEnabled(False)
        super().clearinput()

    def calculate(self):
        ''' Run the calculation '''
        valid = True
        if not self.pginput.isValid():
            QtWidgets.QMessageBox.warning(self, 'Uncertainty Calculator', 'Invalid Input Parameter!')
            valid = False

        elif len(self.uncSweep.unccalc.model.exprs) < 1:
            QtWidgets.QMessageBox.warning(self, 'Uncertainty Calculator', 'Need at least one measurement function to calculate.')
            valid = False

        elif len(self.uncSweep.sweeplist) < 1:
            QtWidgets.QMessageBox.warning(self, 'Uncertainty Calculator', 'Please define at least one sweep in the Sweep tab.')
            valid = False

        elif len(set([len(s.get('values', [])) for s in self.uncSweep.sweeplist])) > 1:
            QtWidgets.QMessageBox.warning(self, 'Uncertainty Calculator', 'All sweep columns must be the same length.')
            valid = False

        if not valid:
            self.actSaveReport.setEnabled(False)
            return

        try:
            self.uncSweep.calculate()
        except (ValueError, RecursionError):
            valid = False

        if valid:
            self.pgoutputsweep.set_sweepreport(self.uncSweep.out)
            self.pgoutputsweep.outputupdate()
            self.stack.setCurrentIndex(self.PG_OUTPUT)
            self.actSaveReport.setEnabled(True)
        else:
            QtWidgets.QMessageBox.warning(self, 'Uncertainty Calculator', 'Error computing solution!')
            self.actSaveReport.setEnabled(False)

    def get_report(self):
        ''' Get full report of curve fit, using page settings '''
        return self.uncSweep.get_output().report_all()

    def save_report(self):
        ''' Save full report, asking user for settings/filename '''
        gui_widgets.savereport(self.get_report())


class UncertReverseSweepWidget(page_uncertprop.UncertPropWidget):
    ''' Widget for calculating a reverse uncertainty sweep '''
    def __init__(self, item, parent=None):
        self.uncRevSwp = item
        self.uncCalc = item.unccalc
        self.sweepsetup = SweepSetupTable(self.uncRevSwp)
        self.targetsetup = page_reverse.TargetSetupWidget(self.uncCalc)
        super().__init__(self.uncCalc, parent=parent)

        self.menu.removeAction(self.mnuSaveSamples.menuAction())
        self.actNewRev = QtWidgets.QAction('New single reverse from sweep', self)
        self.actNewRev.triggered.connect(lambda event, x=item: self.newtype.emit(x.get_config(), 'reverse'))
        self.actNewSwp = QtWidgets.QAction('New forward sweep from model', self)
        self.actNewSwp.triggered.connect(lambda event, x=item: self.newtype.emit(x.get_config(), 'sweep'))
        self.menu.insertAction(self.actSweep, self.actNewRev)
        self.menu.insertAction(self.actSweep, self.actNewSwp)
        self.menu.removeAction(self.actSweep)
        self.menu.removeAction(self.actReverse)
        self.pgoutputrevsweep = page_reverse.PageReverseOutput(None)
        self.pgoutputrevsweep.back.connect(self.backbutton)
        self.stack.removeWidget(self.pgoutput)
        self.stack.addWidget(self.pgoutputrevsweep)
        _, buttons = self.pginput.panel.insert_widget('Sweep', self.sweepsetup, 4, buttons=True)
        self.pginput.panel.insert_widget('Reverse Target Value', self.targetsetup, 4)
        buttons.plusclicked.connect(self.sweepsetup.addcol)
        buttons.minusclicked.connect(self.sweepsetup.remcol)

    def funcchanged(self, row, fdict):
        ''' Function has changed '''
        super().funcchanged(row, fdict)
        self.sweepsetup.set_inptlist(self.uncCalc.inputs)
        self.targetsetup.update_names()
        with suppress(AttributeError):
            self.actNewSwp.setEnabled(True)
            self.actNewRev.setEnabled(True)

    def clearinput(self):
        ''' Clear all the input/output values '''
        self.sweepsetup.clear()
        self.actNewSwp.setEnabled(False)
        self.actNewRev.setEnabled(False)
        super().clearinput()

    def calculate(self):
        ''' Run the calculation '''
        valid = True

        if not self.pginput.isValid():
            QtWidgets.QMessageBox.warning(self, 'Uncertainty Calculator', 'Invalid Input Parameter!')
            valid = False

        elif len(self.uncCalc.model.exprs) < 1:
            QtWidgets.QMessageBox.warning(self, 'Uncertainty Calculator', 'Need at least one measurement function to calculate.')
            valid = False

        elif self.targetsetup.cmbSolveFor.currentText() == '':
            QtWidgets.QMessageBox.warning(self, 'Uncertainty Calculator', 'Please define solve-for parameter in the Target tab.')
            valid = False

        elif len(self.uncRevSwp.sweeplist) < 1:
            QtWidgets.QMessageBox.warning(self, 'Uncertainty Calculator', 'Please define at least one sweep in the Sweep tab.')
            valid = False

        elif len(set([len(s.get('values', [])) for s in self.uncRevSwp.sweeplist])) > 1:
            QtWidgets.QMessageBox.warning(self, 'Uncertainty Calculator', 'All sweep columns must be the same length.')
            valid = False

        if not valid:
            self.actSaveReport.setEnabled(False)
            return

        try:
            self.uncRevSwp.calculate()
        except (ValueError, RecursionError):
            valid = False

        if valid:
            self.pgoutputrevsweep.txtOutput.setReport(self.uncRevSwp.out.report_summary())
            self.stack.setCurrentIndex(self.PG_OUTPUT)
            self.actSaveReport.setEnabled(True)
        else:
            QtWidgets.QMessageBox.warning(self, 'Uncertainty Calculator', 'Error computing solution!')
            self.actSaveReport.setEnabled(False)

    def get_report(self):
        ''' Get full report of curve fit, using page settings '''
        return self.uncRevSwp.get_output().report_all()

    def save_report(self):
        ''' Save full report, asking user for settings/filename '''
        gui_widgets.savereport(self.get_report())
