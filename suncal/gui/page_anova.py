''' UI for Analysis of Variance, Pooled Standard Deviation '''

from io import StringIO

import numpy as np
from PyQt5 import QtWidgets
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from dateutil.parser import parse

from .. import anova
from . import gui_common
from . import gui_widgets


class CustomTable(QtWidgets.QTableWidget):
    def itemtext(self, row, col):
        ''' Get text of item, or 'nan' if item is blank '''
        txt = 'nan'
        item = self.item(row, col)
        if item and item.text() != '':
            txt = item.text()
        return txt


class CSVTableSelectWidget(QtWidgets.QDialog):
    ''' Widget for loading a entire table of groups into the ArrayGrouped. '''
    def __init__(self, parent=None):
        super(CSVTableSelectWidget, self).__init__(parent=parent)
        self.setWindowTitle('Load grouped measurement data')
        self.setGeometry(600, 200, 1000, 600)
        self.table = CustomTable()
        self.startrow = gui_widgets.SpinWidget('Start Row:')
        self.startrow.spin.setRange(1, 1E6)
        self.startcol = gui_widgets.SpinWidget('Start Column:')
        self.startcol.spin.setRange(1, 1E6)
        self.startrow.setValue(1)
        self.startcol.setValue(1)
        self.btnGroupRows = QtWidgets.QRadioButton('Rows')
        self.btnGroupCols = QtWidgets.QRadioButton('Columns')
        self.btnGroupCols.setChecked(True)
        self.chkFirstCol = QtWidgets.QCheckBox('First row is group value')
        self.dlgbutton = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        self.dlgbutton.rejected.connect(self.reject)
        self.dlgbutton.accepted.connect(self.accept)
        self.fig = Figure()
        self.canvas = FigureCanvas(self.fig)

        llayout = QtWidgets.QVBoxLayout()
        clayout = QtWidgets.QHBoxLayout()
        clayout.addWidget(self.startrow)
        clayout.addWidget(self.startcol)
        clayout.addStretch()
        llayout.addLayout(clayout)
        hlayout = QtWidgets.QHBoxLayout()
        hlayout.addWidget(QtWidgets.QLabel('Group by:'))
        hlayout.addWidget(self.btnGroupRows)
        hlayout.addWidget(self.btnGroupCols)
        hlayout.addStretch()
        hlayout.addWidget(self.chkFirstCol)
        hlayout.addStretch()
        llayout.addLayout(hlayout)
        llayout.addWidget(self.table)
        rlayout = QtWidgets.QVBoxLayout()
        rlayout.addWidget(self.canvas)
        rlayout.addWidget(self.dlgbutton)
        layout = QtWidgets.QHBoxLayout()
        layout.addLayout(llayout, stretch=2)
        layout.addLayout(rlayout, stretch=2)
        self.setLayout(layout)

        self.startrow.valueChanged.connect(self.replot)
        self.startcol.valueChanged.connect(self.replot)
        self.btnGroupRows.toggled.connect(self.replot)
        self.btnGroupCols.toggled.connect(self.replot)
        self.chkFirstCol.toggled.connect(self.replot)

    def replot(self):
        ''' Replot data in selected column '''
        self.startrow.spin.setMaximum(self.table.rowCount()-(self.chkFirstCol.isChecked()*self.btnGroupCols.isChecked()))
        self.startcol.spin.setMaximum(self.table.columnCount()-(self.chkFirstCol.isChecked()*self.btnGroupRows.isChecked()))

        if self.btnGroupRows.isChecked():
            self.chkFirstCol.setText('First column is group value')
        else:
            self.chkFirstCol.setText('First row is group value')

        for row in range(self.table.rowCount()):
            for col in range(self.table.columnCount()):
                if self.table.item(row, col) is not None:
                    if row < self.startrow.value()-1 or col < self.startcol.value() - 1:
                        self.table.item(row, col).setBackground(gui_common.COLOR_UNUSED)
                    elif self.chkFirstCol.isChecked() and ((self.btnGroupCols.isChecked() and row == self.startrow.value()-1) or (self.btnGroupRows.isChecked() and col == self.startcol.value()-1)):
                        self.table.item(row, col).setBackground(gui_common.COLOR_HIGHLIGHT)
                    else:
                        self.table.item(row, col).setBackground(gui_common.COLOR_OK)

        self.fig.clf()
        ax = self.fig.add_subplot(1, 1, 1)
        groups = self.get_groups()
        values = self.get_values()
        n = values.shape[0]
        if n > 0:
            try:
                x = np.array(groups)
            except ValueError:
                # Maybe they're dates?
                try:
                    x = [parse(i) for i in groups]
                except ValueError:
                    x = np.arange(len(groups))
            for i in range(len(x)):
                ax.plot([x[i]]*n, values[:, i], marker='o', ls='')

            ax.set_xlabel('Measurement Number')
            ax.set_ylabel('Value')
            self.fig.tight_layout()
            self.dlgbutton.button(QtWidgets.QDialogButtonBox.Ok).setEnabled(True)
        else:
            self.dlgbutton.button(QtWidgets.QDialogButtonBox.Ok).setEnabled(False)
        self.canvas.draw_idle()

    def get_groups(self):
        ''' Get group values (measx in ArrayGrouped) '''
        if self.chkFirstCol.isChecked():
            if self.btnGroupCols.isChecked():
                groups = [self.table.itemtext(self.startrow.value()-1, i) for i in range(self.startcol.value()-1, self.table.columnCount())]
            else:
                groups = [self.table.itemtext(i, self.startcol.value()-1) for i in range(self.startrow.value()-1, self.table.rowCount())]
            try:
                groups = [float(g) for g in groups]
            except ValueError:
                pass
        else:
            if self.btnGroupCols.isChecked():
                groups = np.arange(1, self.table.columnCount()+2-self.startcol.value())
            else:
                groups = np.arange(1, self.table.rowCount()+2-self.startrow.value())
        return groups

    def get_values(self):
        ''' Get measured values, list of arrays for each group '''
        if self.btnGroupCols.isChecked():
            data = [[self.table.itemtext(i, j) for i in range(self.startrow.value()-1+self.chkFirstCol.isChecked(), self.table.rowCount())] for j in range(self.startcol.value()-1, self.table.columnCount())]
        else:
            data = [[self.table.itemtext(i, j) for j in range(self.startcol.value()-1+self.chkFirstCol.isChecked(), self.table.columnCount())] for i in range(self.startrow.value()-1, self.table.rowCount())]
        datstr = '\n'.join([','.join(row) for row in data])
        try:
            values = np.atleast_2d(np.loadtxt(StringIO(datstr), delimiter=',', comments=None, encoding='utf_8_sig').transpose())
        except ValueError:
            values = np.array([])   # Non-floats in data. Don't return anything.
        return values

    def loaddata(self, fname):
        ''' Load table with data from file '''
        with open(fname, 'r') as f:
            try:
                self.raw = f.read()
            except UnicodeDecodeError:
                QtWidgets.QMessageBox.warning(self, 'Load CSV', 'Cannot decode file {}'.format(fname))
                return False

        if ',' in self.raw:
            self.delim = ','
        elif '\t' in self.raw:
            self.delim = '\t'
        else:
            self.delim = None  # None will group whitespace, ' ' will end up with multiple splits

        self.table.clear()
        lines = self.raw.splitlines()
        self.table.setRowCount(len(lines))
        for row, line in enumerate(lines):
            columns = line.strip(u'\ufeff').split(self.delim)  # uFEFF is junk left in Excel-saved CSV files at start of file
            self.table.setColumnCount(max(self.table.columnCount(), len(columns)))
            if line == '': continue
            for col, val in enumerate(columns):
                self.table.setItem(row, col, gui_widgets.ReadOnlyTableItem(val.strip()))
        self.table.setCurrentCell(0, 0)
        self.replot()
        return True


class CSVColumnSelectWidget(QtWidgets.QDialog):
    ''' Widget for loading a SINGLE group into the ArrayGrouped. '''
    def __init__(self, parent=None):
        super(CSVColumnSelectWidget, self).__init__(parent=parent)
        self.setWindowTitle('Add group to ANOVA calculation')
        self.setGeometry(600, 200, 1000, 800)
        self.table = CustomTable()
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectColumns)
        self.startrow = gui_widgets.SpinWidget('Start Row:')
        self.startrow.spin.setRange(1, 1E6)
        self.startrow.setValue(1)
        self.dlgbutton = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        self.dlgbutton.rejected.connect(self.reject)
        self.dlgbutton.accepted.connect(self.accept)
        self.fig = Figure()
        self.canvas = FigureCanvas(self.fig)

        llayout = QtWidgets.QVBoxLayout()
        llayout.addWidget(self.startrow)
        llayout.addWidget(QtWidgets.QLabel('Select column for new group:'))
        llayout.addWidget(self.table)
        rlayout = QtWidgets.QVBoxLayout()
        rlayout.addWidget(self.canvas)
        rlayout.addWidget(self.dlgbutton)
        layout = QtWidgets.QHBoxLayout()
        layout.addLayout(llayout, stretch=2)
        layout.addLayout(rlayout, stretch=2)
        self.setLayout(layout)

        self.table.currentCellChanged.connect(self.replot)
        self.startrow.valueChanged.connect(self.replot)

    def replot(self):
        ''' Replot data in selected column '''
        for row in range(self.table.rowCount()):
            for col in range(self.table.columnCount()):
                if self.table.item(row, col) is not None:
                    if row < self.startrow.value()-1:
                        self.table.item(row, col).setBackground(gui_common.COLOR_UNUSED)
                    else:
                        self.table.item(row, col).setBackground(gui_common.COLOR_OK)

        self.fig.clf()
        ax = self.fig.add_subplot(1, 1, 1)
        values = self.get_data()
        if values.shape[0] > 0:
            ax.plot(values, marker='o', ls='')
        ax.set_xlabel('Measurement Number')
        ax.set_ylabel('Value')
        self.canvas.draw_idle()

    def get_data(self):
        ''' Get array of values in selected column '''
        col = self.table.currentColumn()
        if col > -1:
            data = [self.table.itemtext(i, col) for i in range(self.startrow.value()-1, self.table.rowCount())]
        else:
            data = []
        try:
            values = np.array(data, dtype=np.float64)
        except ValueError:
            values = np.array([])
        return values

    def loaddata(self, fname):
        ''' Load table with data from file '''
        with open(fname, 'r') as f:
            try:
                self.raw = f.read()
            except UnicodeDecodeError:
                QtWidgets.QMessageBox.warning(self, 'Load CSV', 'Cannot decode file {}'.format(fname))
                return False

        if ',' in self.raw:
            self.delim = ','
        elif '\t' in self.raw:
            self.delim = '\t'
        else:
            self.delim = None  # None will group whitespace, ' ' will end up with multiple splits

        self.table.clear()
        lines = self.raw.splitlines()
        self.table.setRowCount(len(lines))
        for row, line in enumerate(lines):
            columns = line.strip(u'\ufeff').split(self.delim)  # uFEFF is junk left in Excel-saved CSV files at start of file
            self.table.setColumnCount(max(self.table.columnCount(), len(columns)))
            if line == '': continue
            for col, val in enumerate(columns):
                self.table.setItem(row, col, gui_widgets.ReadOnlyTableItem(val.strip()))

        self.table.setCurrentCell(0, 0)
        return True


class AnovaWidget(QtWidgets.QWidget):
    ''' Widget for ANOVA calculations '''
    def __init__(self, item, parent=None):
        super(AnovaWidget, self).__init__(parent=parent)
        self.anova = item
        self.btnEnterAll = QtWidgets.QRadioButton('Enter all measured values')
        self.btnEnterSum = QtWidgets.QRadioButton('Enter group means and standard deviations')
        self.btnEnterAll.setChecked(True)

        # table for entering 2D data
        self.table = gui_widgets.FloatTableWidget(movebyrows=True, headeredit='str')
        self.table.setMinimumWidth(400)
        self.table.setColumnCount(1)
        self.table.setRowCount(1)
        self.table.setHorizontalHeaderLabels(['1'])

        # table2 for entering mean/std/nmeas
        self.table2 = gui_widgets.FloatTableWidget(xstrings=True)  # Allow strings in first col
        self.table2.setMinimumWidth(400)
        self.table2.setColumnCount(4)
        self.table2.setRowCount(1)
        self.table2.setHorizontalHeaderLabels(['Group Name', 'Mean', 'Std. Dev.', '# Measurements'])
        self.table2.setVisible(False)

        self.btnAddCol = QtWidgets.QToolButton()
        self.btnAddCol.setIcon(gui_common.load_icon('add'))
        self.btnAddCol.setToolTip('Add group')
        self.btnRemCol = QtWidgets.QToolButton()
        self.btnRemCol.setIcon(gui_common.load_icon('remove'))
        self.btnRemCol.setToolTip('Remove selected group')
        self.chkDates = QtWidgets.QCheckBox('Group names are dates')
        self.btnCalc = QtWidgets.QPushButton('Calculate')
        self.txtOutput = gui_widgets.MarkdownTextEdit()
        self.grouplabel = QtWidgets.QLabel('Measurements grouped by column:')
        self.txtDescription = QtWidgets.QPlainTextEdit()

        self.fig = Figure()
        self.canvas = FigureCanvas(self.fig)
        self.toolbar = NavigationToolbar(self.canvas, self, coordinates=True)

        self.menu = QtWidgets.QMenu('ANOVA')
        self.actImportCol = QtWidgets.QAction('New group from CSV...', self)
        self.actImport = QtWidgets.QAction('Load table from CSV...', self)
        self.actClear = QtWidgets.QAction('Clear table', self)
        self.actSaveReport = QtWidgets.QAction('Save Report...', self)
        self.menu.addAction(self.actImport)
        self.menu.addAction(self.actImportCol)
        self.menu.addAction(self.actClear)
        self.menu.addSeparator()
        self.menu.addAction(self.actSaveReport)
        self.actSaveReport.setEnabled(False)

        self.actImportCol.triggered.connect(self.importcolumn)
        self.actImport.triggered.connect(self.importtable)
        self.actClear.triggered.connect(self.clear)
        self.actSaveReport.triggered.connect(self.save_report)

        blayout = QtWidgets.QHBoxLayout()
        blayout.addWidget(self.btnEnterAll)
        blayout.addWidget(self.btnEnterSum)
        h2layout = QtWidgets.QHBoxLayout()
        h2layout.addWidget(self.btnAddCol)
        h2layout.addWidget(self.btnRemCol)
        h2layout.addStretch()
        llayout = QtWidgets.QVBoxLayout()
        llayout.addLayout(blayout)
        llayout.addWidget(self.chkDates)
        llayout.addLayout(h2layout)
        llayout.addWidget(self.grouplabel)
        llayout.addWidget(self.table, stretch=10)
        llayout.addWidget(self.table2, stretch=10)
        llayout.addWidget(QtWidgets.QLabel('Notes:'))
        llayout.addWidget(self.txtDescription, stretch=3)
        rlayout = QtWidgets.QVBoxLayout()
        rlayout.addWidget(self.txtOutput, stretch=10)
        rlayout.addWidget(self.canvas, stretch=10)
        rlayout.addWidget(self.toolbar)
        clayout = QtWidgets.QHBoxLayout()
        clayout.addStretch()
        clayout.addWidget(self.btnCalc)
        rlayout.addLayout(clayout)
        layout = QtWidgets.QHBoxLayout()
        layout.addLayout(llayout)
        layout.addLayout(rlayout)
        self.setLayout(layout)
        self.setup_init()

        self.btnCalc.clicked.connect(self.calculate)
        self.btnAddCol.clicked.connect(self.addcol)
        self.btnRemCol.clicked.connect(self.remcol)
        self.table.valueChanged.connect(self.tablechanged)
        self.table2.valueChanged.connect(self.tablechanged)
        self.btnEnterAll.clicked.connect(self.changemode)
        self.btnEnterSum.clicked.connect(self.changemode)
        self.chkDates.stateChanged.connect(self.tablechanged)
        self.txtDescription.textChanged.connect(self.update_description)
        self.fig.add_subplot(1, 1, 1)
        self.canvas.draw_idle()

    def setup_init(self):
        ''' Initialize fields using self.anova settings '''
        if self.anova.ntot() > 0:
            ngroups = self.anova.ngroups()
            self.txtDescription.setPlainText(self.anova.description)
            if hasattr(self.anova, 'ystds'):
                self.btnEnterSum.setChecked(True)
                self.table2.setVisible(True)
                self.table.setVisible(False)
                self.btnAddCol.setVisible(False)
                self.btnRemCol.setVisible(False)
                self.grouplabel.setVisible(False)
                self.table2.setRowCount(self.anova.measy.shape[1])
                for row in range(len(self.anova.measx)):
                    self.table2.setItem(row, 0, QtWidgets.QTableWidgetItem(str(self.anova.measx[row])))
                    self.table2.setItem(row, 1, QtWidgets.QTableWidgetItem(str(self.anova.measy[row, 0])))
                    self.table2.setItem(row, 2, QtWidgets.QTableWidgetItem(str(self.anova.ystds[row])))
                    self.table2.setItem(row, 3, QtWidgets.QTableWidgetItem(str(self.anova.nmeas[row])))
                self.table2.resizeColumnsToContents()

            else:
                self.btnEnterAll.setChecked(True)
                self.table2.setVisible(False)
                self.table.setVisible(True)
                self.btnAddCol.setVisible(True)
                self.btnRemCol.setVisible(True)
                self.grouplabel.setVisible(True)
                self.table.setColumnCount(ngroups)
                self.table.setRowCount(self.anova.measy.shape[1])
                self.table.setHorizontalHeaderLabels(self.anova.group_names())
                for row in range(self.anova.measy.shape[1]):
                    for col in range(ngroups):
                        val = self.anova.measy[col, row]
                        if np.isfinite(val):
                            self.table.setItem(row, col, QtWidgets.QTableWidgetItem(str(val)))

                self.table.resizeColumnsToContents()
            self.chkDates.setChecked(self.anova.is_date())

    def get_menu(self):
        ''' Get the menu for this calculation mode '''
        return self.menu

    def clear(self):
        ''' Clear the table '''
        self.table.blockSignals(True)
        self.table.clear()
        self.table.setRowCount(1)
        self.table.setColumnCount(1)
        self.table.setItem(0, 0, QtWidgets.QTableWidgetItem(''))
        self.table.setHorizontalHeaderLabels(['1'])
        self.table.blockSignals(False)
        self.table2.blockSignals(True)
        self.table2.clear()
        self.table2.setRowCount(1)
        self.table2.setHorizontalHeaderLabels(['Group Name', 'Mean', 'Std. Dev.', '# Measurements'])
        self.table2.blockSignals(False)
        self.anova.measx = np.array([])
        self.anova.measy = np.array([[]])
        self.actSaveReport.setEnabled(False)

    def addcol(self):
        ''' Add a column to the table '''
        col = self.table.columnCount()
        self.table.setColumnCount(col + 1)
        self.table.setHorizontalHeaderItem(col, QtWidgets.QTableWidgetItem(str(col+1)))  # Must do this to make it editable for some reason.

    def remcol(self):
        ''' Remove selected column from table '''
        self.table.removeColumn(self.table.currentColumn())
        self.tablechanged()

    def importcolumn(self):
        ''' Import data for a new group from CSV file '''
        fname, _ = QtWidgets.QFileDialog.getOpenFileName(caption='Select CSV file to load')
        if fname:
            self.btnEnterAll.setChecked(True)
            dlg = CSVColumnSelectWidget()
            ok = dlg.loaddata(fname) and dlg.exec_()
            if ok:
                newgroup = dlg.get_data()
                if newgroup is not None:
                    self.table.blockSignals(True)
                    if self.table.rowCount() > 1 or self.table.columnCount() > 1:
                        self.addcol()
                    col = self.table.columnCount() - 1
                    if len(newgroup) > self.table.rowCount():
                        self.table.setRowCount(len(newgroup))
                    for row, val in enumerate(newgroup):
                        self.table.setItem(row, col, QtWidgets.QTableWidgetItem(str(val)))
                    self.table.blockSignals(False)
                    self.tablechanged()

    def importtable(self):
        ''' Import entire table from CSV file '''
        fname, _ = QtWidgets.QFileDialog.getOpenFileName(caption='Select CSV file to load')
        if fname:
            self.btnEnterAll.setChecked(True)
            dlg = CSVTableSelectWidget()
            ok = dlg.loaddata(fname) and dlg.exec_()
            if ok:
                groups = dlg.get_groups()
                values = dlg.get_values()
                self.table.blockSignals(True)
                self.table.clear()
                self.table.setRowCount(values.shape[0])
                self.table.setColumnCount(values.shape[1])
                self.table.setHorizontalHeaderLabels([str(g) for g in groups])
                for row in range(values.shape[0]):
                    for col in range(values.shape[1]):
                        self.table.setItem(row, col, QtWidgets.QTableWidgetItem(str(values[row, col])))
                self.table.blockSignals(False)
                self.tablechanged()
                self.calculate()

    def changemode(self):
        ''' Mode changed from entering 2D to 1D data '''
        self.table2.setVisible(self.btnEnterSum.isChecked())
        self.table.setVisible(self.btnEnterAll.isChecked())
        self.btnAddCol.setVisible(self.btnEnterAll.isChecked())
        self.btnRemCol.setVisible(self.btnEnterAll.isChecked())
        self.grouplabel.setVisible(self.btnEnterAll.isChecked())
        if self.btnEnterAll.isChecked():
            self.anova.__class__ = anova.ArrayGrouped
            try:
                del self.anova.ystds
                del self.anova.nmeas
            except AttributeError:
                pass
        else:
            self.anova.__class__ = anova.ArrayGroupedSummary
        self.tablechanged()

    def tablechanged(self):
        ''' Table data has changed. Update the model '''
        if self.btnEnterAll.isChecked():
            # Entering full data set
            try:
                self.anova.measy = self.table.get_table()
            except ValueError:
                self.anova.measy = np.array([[]])

            measx = []
            for i in range(self.table.columnCount()):
                if self.table.horizontalHeaderItem(i):
                    measx.append(self.table.horizontalHeaderItem(i).text())
                else:
                    # Needed when pasting multicolumn data
                    measx.append(str(i+1))
                    self.table.setHorizontalHeaderItem(i, QtWidgets.QTableWidgetItem(str(i+1)))
            self.anova.measx = measx

        else:
            # Entering only mean/std/nmeas
            self.anova.measx = list(self.table2.get_column(0))
            n = len(self.anova.measx)
            try:
                self.anova.measy = np.array([self.table2.get_column(1)], dtype=np.float64).transpose()
            except ValueError:
                self.anova.measy = np.array([np.full(n, np.nan)]).transpose()

            try:
                self.anova.ystds = np.array(self.table2.get_column(2), dtype=np.float64)
            except ValueError:
                self.anova.ystds = np.full(n, np.nan)

            try:
                self.anova.nmeas = np.array(self.table2.get_column(3), dtype=np.float64)
            except ValueError:
                self.anova.nmeas = np.full(n, 0)

            # Remove incomplete rows
            invalididx = np.where(~np.isfinite(self.anova.measy.flatten()*self.anova.ystds*self.anova.nmeas))[0]
            valididx = np.where(np.isfinite(self.anova.measy.flatten()*self.anova.ystds*self.anova.nmeas))[0]
            for i in invalididx[::-1]:
                self.anova.measx.pop(i)
            self.anova.measy = self.anova.measy[valididx]
            self.anova.ystds = self.anova.ystds[valididx]
            self.anova.nmeas = self.anova.nmeas[valididx]

        # Convert x values to dates if requested and possible
        if self.chkDates.isChecked():
            for i in range(len(self.anova.measx)):
                try:
                    self.anova.measx[i] = parse(self.anova.measx[i])
                except ValueError:
                    self.anova.measx[i] = np.nan
        else:
            # Convert to floats if possible
            for i in range(len(self.anova.measx)):
                try:
                    self.anova.measx[i] = float(self.anova.measx[i])
                except ValueError:
                    pass
        self.actSaveReport.setEnabled(False)

    def update_description(self):
        ''' Description was updated, save it. '''
        self.anova.description = self.txtDescription.toPlainText()

    def calculate(self):
        ''' Run calculation and update report/plot. '''
        if self.anova.ngroups() > 0 and self.anova.measy.size > 0:
            out = self.anova.calculate()
            self.txtOutput.setMarkdown(out.report(**gui_common.get_rptargs()))
            self.fig.clf()
            ax = self.fig.add_subplot(1, 1, 1)
            out.plot(ax=ax)
            try:
                self.canvas.draw_idle()
            except ValueError:
                pass  # Can raise when invalid dates in table
            self.actSaveReport.setEnabled(True)
        else:
            self.txtOutput.rpt = None
            self.txtOutput.setHtml('No data to calculate')
            self.actSaveReport.setEnabled(False)

    def get_report(self):
        ''' Get full report of curve fit, using page settings '''
        return self.anova.get_output().report_all(**gui_common.get_rptargs())

    def save_report(self):
        ''' Save full report, asking user for settings/filename '''
        gui_widgets.savemarkdown(self.get_report())
