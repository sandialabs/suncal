''' GUI page for loading datasets and computing stats such as analysis of variance '''

from contextlib import suppress
import numpy as np
from PyQt5 import QtWidgets, QtCore
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

from .. import dataset
from .. import report
from .. import distributions
from . import gui_common
from . import gui_widgets
from . import page_csvload
from . import configmgr

settings = configmgr.Settings()

class HistCtrlWidget(QtWidgets.QWidget):
    ''' Controls for Histogram output mode '''
    changed = QtCore.pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.colSelect = QtWidgets.QComboBox()
        self.fit = QtWidgets.QComboBox()
        dists = settings.getDistributions()
        dists = ['None'] + [d for d in dists if distributions.fittable(d)]
        self.fit.addItems(dists)
        self.probplot = QtWidgets.QCheckBox('Probability Plot')

        layout = QtWidgets.QHBoxLayout()
        layout.addWidget(QtWidgets.QLabel('Column:'))
        layout.addWidget(self.colSelect)
        layout.addStretch()
        layout.addWidget(QtWidgets.QLabel('Distribution Fit:'))
        layout.addWidget(self.fit)
        layout.addWidget(self.probplot)
        self.setLayout(layout)
        self.colSelect.currentIndexChanged.connect(self.changed)
        self.fit.currentIndexChanged.connect(self.changed)
        self.probplot.stateChanged.connect(self.changed)

    def update_colnames(self, names):
        self.colSelect.blockSignals(True)
        self.colSelect.clear()
        self.colSelect.addItems(names)
        self.colSelect.blockSignals(False)
        self.changed.emit()


class CorrCtrlWidget(QtWidgets.QWidget):
    ''' Controls for Correlation output mode '''
    changed = QtCore.pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.col1Select = QtWidgets.QComboBox()
        self.col2Select = QtWidgets.QComboBox()
        layout = QtWidgets.QHBoxLayout()
        layout.addWidget(QtWidgets.QLabel('Column 1:'))
        layout.addWidget(self.col1Select)
        layout.addWidget(QtWidgets.QLabel('Column 2:'))
        layout.addWidget(self.col2Select)
        layout.addStretch()
        self.setLayout(layout)
        self.col1Select.currentIndexChanged.connect(self.changed)
        self.col2Select.currentIndexChanged.connect(self.changed)

    def update_colnames(self, names):
        self.col1Select.blockSignals(True)
        self.col1Select.clear()
        self.col1Select.addItems(names)
        self.col1Select.blockSignals(False)
        self.col2Select.blockSignals(True)
        self.col2Select.clear()
        self.col2Select.addItems(names)
        self.col2Select.blockSignals(False)
        self.changed.emit()


class ACorrCtrlWidget(QtWidgets.QWidget):
    ''' Controls for Autocorrelation output mode '''
    changed = QtCore.pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.colSelect = QtWidgets.QComboBox()
        self.mode = QtWidgets.QComboBox()
        self.mode.addItems(['Autocorrelation Plot', 'Lag Plot'])
        self.laglabel = QtWidgets.QLabel('Lag:')
        self.lag = QtWidgets.QSpinBox()
        self.lag.setRange(1, 50)
        layout = QtWidgets.QHBoxLayout()
        layout.addWidget(QtWidgets.QLabel('Column:'))
        layout.addWidget(self.colSelect)
        layout.addWidget(self.mode)
        layout.addWidget(self.laglabel)
        layout.addWidget(self.lag)
        layout.addStretch()
        self.setLayout(layout)
        self.colSelect.currentIndexChanged.connect(self.changed)
        self.mode.currentIndexChanged.connect(self.changemode)
        self.mode.currentIndexChanged.connect(self.changed)
        self.lag.valueChanged.connect(self.changed)
        self.changemode()

    def changemode(self):
        self.lag.setVisible(self.mode.currentText() == 'Lag Plot')
        self.laglabel.setVisible(self.mode.currentText() == 'Lag Plot')

    def update_colnames(self, names):
        self.colSelect.blockSignals(True)
        self.colSelect.clear()
        self.colSelect.addItems(names)
        self.colSelect.blockSignals(False)
        self.changed.emit()


class DataSetWidget(QtWidgets.QWidget):
    ''' Widget for displaying measured data and ANOVA calculations '''
    def __init__(self, item, parent=None):
        super().__init__(parent=parent)
        self.dataset = item

        self.table = gui_widgets.FloatTableWidget(movebyrows=True, headeredit='str')
        self.table.setMinimumWidth(400)
        self.table.setColumnCount(1)
        self.table.setRowCount(1)
        self.table.setHorizontalHeaderLabels(['1'])

        self.btnAddCol = QtWidgets.QToolButton()
        self.btnAddCol.setIcon(gui_common.load_icon('add'))
        self.btnAddCol.setToolTip('Add column')
        self.btnRemCol = QtWidgets.QToolButton()
        self.btnRemCol.setIcon(gui_common.load_icon('remove'))
        self.btnRemCol.setToolTip('Remove selected column')
        self.txtDescription = QtWidgets.QPlainTextEdit()

        self.cmbMode = QtWidgets.QComboBox()
        self.cmbMode.addItems(['Summary', 'Histogram', 'Correlation', 'Autocorrelation', 'Analysis of Variance'])
        self.histctrls = HistCtrlWidget()
        self.corrctrls = CorrCtrlWidget()
        self.acorctrls = ACorrCtrlWidget()

        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self, coordinates=True)
        self.txtOutput = gui_widgets.MarkdownTextEdit()

        self.menu = QtWidgets.QMenu('Data Set')
        self.actImportCol = QtWidgets.QAction('New column from CSV...', self)
        self.actImport = QtWidgets.QAction('Load data from CSV...', self)
        self.actClear = QtWidgets.QAction('Clear data', self)
        self.actSaveReport = QtWidgets.QAction('Save Report...', self)
        self.actSummary = QtWidgets.QAction('Enter Summarized Values', self)
        self.actSummary.setCheckable(True)
        self.menu.addAction(self.actImport)
        self.menu.addAction(self.actImportCol)
        self.menu.addAction(self.actClear)
        self.menu.addSeparator()
        self.menu.addAction(self.actSummary)
        self.menu.addSeparator()
        self.menu.addAction(self.actSaveReport)

        self.actImportCol.triggered.connect(self.importcolumn)
        self.actImport.triggered.connect(self.importtable)
        self.actClear.triggered.connect(self.clear)
        self.actSaveReport.triggered.connect(self.save_report)
        self.actSummary.triggered.connect(self.togglesummary)
        self.btnAddCol.clicked.connect(self.addcol)
        self.btnRemCol.clicked.connect(self.remcol)
        self.table.valueChanged.connect(self.tablechanged)
        self.txtDescription.textChanged.connect(self.update_description)
        self.cmbMode.currentIndexChanged.connect(self.changemode)
        self.histctrls.changed.connect(self.refresh_output)
        self.corrctrls.changed.connect(self.refresh_output)
        self.acorctrls.changed.connect(self.refresh_output)

        hlayout = QtWidgets.QHBoxLayout()
        hlayout.addStretch()
        hlayout.addWidget(self.btnAddCol)
        hlayout.addWidget(self.btnRemCol)
        llayout = QtWidgets.QVBoxLayout()
        llayout.addLayout(hlayout)
        llayout.addWidget(self.table, stretch=10)
        llayout.addWidget(QtWidgets.QLabel('Notes:'))
        llayout.addWidget(self.txtDescription, stretch=3)
        outlayout = QtWidgets.QVBoxLayout()
        ctrllayout = QtWidgets.QHBoxLayout()
        ctrllayout.addWidget(self.cmbMode)
        ctrllayout.addWidget(self.histctrls)
        ctrllayout.addWidget(self.corrctrls)
        ctrllayout.addWidget(self.acorctrls)
        outlayout.addLayout(ctrllayout)
        outlayout.addWidget(self.txtOutput, stretch=5)
        outlayout.addWidget(self.canvas, stretch=5)
        outlayout.addWidget(self.toolbar)
        toplayout = QtWidgets.QHBoxLayout()
        toplayout.addLayout(llayout, stretch=5)
        toplayout.addLayout(outlayout, stretch=5)
        self.setLayout(toplayout)

        self.init_data()
        self.changemode()  # Initialize setVisible on controls
        self.canvas.draw_idle()

    def calculate(self):
        ''' Run calculation. Dataset is calculated automatically, so this does nothing. '''
        pass

    def init_data(self):
        ''' Initialize the table using data in self.dataset '''
        self.table.blockSignals(True)
        self.txtDescription.setPlainText(self.dataset.description)
        ncols = self.dataset.ncolumns()
        nrows = self.dataset.maxrows()

        if isinstance(self.dataset, dataset.DataSetSummary) != self.actSummary.isChecked():
            self.actSummary.setChecked(not self.actSummary.isChecked())
            means = self.dataset._means()
            stds = self.dataset._stds()
            counts = self.dataset._nmeas()
            ncols = len(means)
            self.table.setRowCount(3)
            self.table.setColumnCount(ncols)
            self.table.setVerticalHeaderLabels(['Mean', 'Std. Dev.', 'Count'])
            self.cmbMode.blockSignals(True)
            self.cmbMode.clear()
            self.cmbMode.addItems(['Summary', 'Analysis of Variance'])
            self.cmbMode.blockSignals(False)
            for col in range(ncols):
                self.table.setItem(0, col, QtWidgets.QTableWidgetItem(str(means[col])))
                self.table.setItem(1, col, QtWidgets.QTableWidgetItem(str(stds[col])))
                self.table.setItem(2, col, QtWidgets.QTableWidgetItem(str(counts[col])))

        else:
            if ncols > 0:
                self.table.setColumnCount(ncols)
                self.table.setRowCount(nrows)
                self.table.setHorizontalHeaderLabels([str(s) for s in self.dataset.colnames])
                for col in range(ncols):
                    for row in range(len(self.dataset.data[col])):
                        val = self.dataset.data[col][row]
                        if np.isfinite(val):
                            self.table.setItem(row, col, QtWidgets.QTableWidgetItem(str(val)))
                self.table.resizeColumnsToContents()
    
        self.table.blockSignals(False)
        self.updatecolnames()
        self.refresh_output()

    def get_menu(self):
        ''' Get the menu for this calculation mode '''
        return self.menu

    def togglesummary(self):
        ''' Change from full table to summarized statistics mode '''
        if self.actSummary.isChecked():
            self.table.setRowCount(3)
            self.table.setVerticalHeaderLabels(['Mean', 'Std. Dev.', 'Count'])
            self.dataset.__class__ = dataset.DataSetSummary
            self.cmbMode.blockSignals(True)
            self.cmbMode.clear()
            self.cmbMode.addItems(['Summary', 'Analysis of Variance'])
            self.cmbMode.blockSignals(False)
        else:
            self.table.setVerticalHeaderLabels([str(i) for i in range(self.table.rowCount())])
            self.dataset.__class__ = dataset.DataSet
            self.cmbMode.blockSignals(True)
            self.cmbMode.clear()
            self.cmbMode.addItems(['Summary', 'Histogram', 'Correlation', 'Autocorrelation', 'Analysis of Variance'])
            self.cmbMode.blockSignals(False)

    def changemode(self):
        ''' Change output mode. Show/hide controls as appropriate '''
        mode = self.cmbMode.currentText()
        self.histctrls.setVisible(mode == 'Histogram')
        self.corrctrls.setVisible(mode == 'Correlation')
        self.acorctrls.setVisible(mode == 'Autocorrelation')
        self.refresh_output()

    def tablechanged(self):
        ''' Table data has changed, update the model '''
        data = self.table.get_table()
        if self.actSummary.isChecked() and data.shape[1] < 3:
            # PAD data to three rows
            data = np.pad(data, ((0, 0), (0, 3-data.shape[1])), mode='constant')
            data = np.nan_to_num(data)
        self.dataset.data = data

        colnames = []
        for i in range(self.table.columnCount()):
            if self.table.horizontalHeaderItem(i):
                colnames.append(self.table.horizontalHeaderItem(i).text())
            else:
                # Needed when pasting multicolumn data
                colnames.append(str(i+1))
                self.table.setHorizontalHeaderItem(i, QtWidgets.QTableWidgetItem(str(i+1)))

        self.dataset.colnames = colnames
        self.updatecolnames()
        self.refresh_output()

    def updatecolnames(self):
        ''' Update column names in column-select widgets '''
        names = [str(c) for c in self.dataset.colnames]
        self.histctrls.update_colnames(names)
        self.corrctrls.update_colnames(names)
        self.acorctrls.update_colnames(names)

    def refresh_output(self):
        ''' Refresh the output plot and report '''
        self.figure.clf()

        rpt = report.Report()
        mode = self.cmbMode.currentText()
        if mode == 'Summary':
            rpt.hdr('Columns', level=2)
            rpt.append(self.dataset.out.report())
            if self.dataset.ncolumns() > 1:
                rpt.hdr('Pooled Statistics', level=3)
                rpt.append(self.dataset.out.report_pooled())
                self.dataset.out.plot_groups(plot=self.figure)
            else:
                self.dataset.out.plot_histogram(plot=self.figure)

        elif mode == 'Histogram':
            col = self.histctrls.colSelect.currentText()
            fit = self.histctrls.fit.currentText()
            fit = None if fit == 'None' else fit
            probplot = self.histctrls.probplot.isChecked()
            rpt.hdr(col, level=2)
            rpt.append(self.dataset.out.report_column(col))
            self.dataset.out.plot_histogram(colname=col, plot=self.figure, fit=fit, qqplot=probplot)

        elif mode == 'Correlation':
            rpt.hdr('Correlation Matrix', level=2)
            rpt.append(self.dataset.out.report_correlation())
            col1 = self.corrctrls.col1Select.currentText()
            col2 = self.corrctrls.col2Select.currentText()
            self.dataset.out.plot_scatter(col1, col2, plot=self.figure)

        elif mode == 'Autocorrelation':
            rpt.hdr('Autocorrelation', level=2)
            col = self.acorctrls.colSelect.currentText()
            lag = self.acorctrls.lag.value()
            rpt.append(self.dataset.out.report_autocorrelation())
            if self.acorctrls.mode.currentText() == 'Lag Plot':
                with suppress(ValueError):  # Raises when lag too high
                    self.dataset.out.plot_lag(col, lag=lag, plot=self.figure)
            else:
                self.dataset.out.plot_autocorrelation(col, plot=self.figure)

        elif mode == 'Analysis of Variance':
            rpt.hdr('Analysis of Variance', level=2)
            rpt.append(self.dataset.out.report_anova())
            self.dataset.out.plot_groups(plot=self.figure)

        else:
            raise NotImplementedError

        self.txtOutput.setReport(rpt)
        self.figure.tight_layout()
        self.canvas.draw_idle()

    def clear(self):
        ''' Clear the table '''
        self.table.blockSignals(True)
        self.table.clear()
        self.table.setRowCount(1)
        self.table.setColumnCount(1)
        self.table.setItem(0, 0, QtWidgets.QTableWidgetItem(''))
        self.table.setHorizontalHeaderLabels(['1'])
        self.table.blockSignals(False)
        self.dataset.colnames = []
        self.dataset.data = np.array([[]])

    def addcol(self):
        ''' Add a column to the table '''
        col = self.table.columnCount()
        self.table.setColumnCount(col + 1)
        self.table.setHorizontalHeaderItem(col, QtWidgets.QTableWidgetItem(str(col+1)))  # Must do this to make it editable for some reason.

    def remcol(self):
        ''' Remove selected column from table '''
        self.table.removeColumn(self.table.currentColumn())
        if self.table.columnCount() == 0:
            self.table.setColumnCount(1)
        self.tablechanged()

    def update_description(self):
        ''' Description was updated, save it. '''
        self.dataset.description = self.txtDescription.toPlainText()

    def importcolumn(self):
        ''' Add one column from a CSV file '''
        fname, _ = QtWidgets.QFileDialog.getOpenFileName(caption='Select CSV file to load')
        if fname:
            dlg = page_csvload.SelectCSVData(fname)
            if dlg.exec_():
                self.table.blockSignals(True)
                dset = dlg.dataset()
                data = dset.get_column() # ONLY FIRST COLUMN is included when importing a column
                hdr = dset.colnames[0]   # List of strings
                rowcnt = len(data)

                if self.table.rowCount() > 1 or self.table.columnCount() > 1:
                    self.addcol()
                self.table.setRowCount(max(rowcnt, self.table.rowCount()))
                col = self.table.columnCount() - 1

                if len(hdr) > 0:
                    self.table.setHorizontalHeaderItem(col, QtWidgets.QTableWidgetItem(hdr[0]))

                for row in range(len(data)):
                    self.table.setItem(row, col, QtWidgets.QTableWidgetItem(str(data[row])))
                self.table.blockSignals(False)
                self.tablechanged()

    def importtable(self):
        ''' Import data table from a CSV file '''
        fname, _ = QtWidgets.QFileDialog.getOpenFileName(caption='Select CSV file to load')
        if fname:
            dlg = page_csvload.SelectCSVData(fname)
            if dlg.exec_():
                self.table.blockSignals(True)
                self.table.clear()
                dset = dlg.dataset()
                data = dset.data
                hdr = dset.colnames
                colcnt = max(len(hdr), len(data))
                rowcnt = max([len(c) for c in data])

                self.table.setRowCount(rowcnt)
                self.table.setColumnCount(colcnt)
                if len(hdr) > 0:
                    self.table.setHorizontalHeaderLabels(hdr)
                for col in range(colcnt):
                    for row in range(len(data[col])):
                        self.table.setItem(row, col, QtWidgets.QTableWidgetItem(str(data[col][row])))
                self.table.blockSignals(False)
                self.tablechanged()

    def get_report(self):
        ''' Get full report of dataset, using page settings '''
        return self.dataset.get_output().report_all()

    def save_report(self):
        ''' Save full report, asking user for settings/filename '''
        gui_widgets.savereport(self.get_report())
