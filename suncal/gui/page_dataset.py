''' GUI page for loading datasets and computing stats such as analysis of variance '''

from contextlib import suppress
import numpy as np
from PyQt6 import QtWidgets, QtCore, QtGui
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

from ..datasets.dataset_model import DataSet
from ..common import report, distributions
from . import widgets
from . import gui_styles
from .gui_settings import gui_settings
from .gui_common import BlockedSignals
from . import page_csvload
from .help_strings import AnovaHelp


class HistCtrlWidget(QtWidgets.QWidget):
    ''' Controls for Histogram output mode '''
    changed = QtCore.pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.colSelect = QtWidgets.QComboBox()
        self.fit = QtWidgets.QComboBox()
        dists = gui_settings.distributions
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
        ''' Update column names in the combobox '''
        with BlockedSignals(self.colSelect):
            self.colSelect.clear()
            self.colSelect.addItems(names)
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
        ''' Update column names in the comboboxes '''
        with BlockedSignals(self.col1Select):
            self.col1Select.clear()
            self.col1Select.addItems(names)
        with BlockedSignals(self.col2Select):
            self.col2Select.clear()
            self.col2Select.addItems(names)
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
        ''' Change page mode from autocorrelation plot to lag plot '''
        self.lag.setVisible(self.mode.currentText() == 'Lag Plot')
        self.laglabel.setVisible(self.mode.currentText() == 'Lag Plot')

    def update_colnames(self, names):
        ''' Update the column names in the combobox '''
        with BlockedSignals(self.colSelect):
            self.colSelect.clear()
            self.colSelect.addItems(names)
        self.changed.emit()


class DataSetWidget(QtWidgets.QWidget):
    ''' Widget for displaying measured data and ANOVA calculations '''

    change_help = QtCore.pyqtSignal()

    def __init__(self, component, parent=None):
        super().__init__(parent=parent)
        self.component = component

        self.table = widgets.FloatTableWidget(movebyrows=True, headeredit='str')
        self.table.setMinimumWidth(400)
        self.table.setColumnCount(1)
        self.table.setRowCount(1)
        self.table.setHorizontalHeaderLabels(['1'])
        self.tolerance = widgets.ToleranceCheck()

        self.btnAddRemCol = widgets.PlusMinusButton(stretch=False)
        self.btnAddRemCol.btnplus.setToolTip('Add Column')
        self.btnAddRemCol.btnminus.setToolTip('Remove Selected Column')
        self.txtDescription = QtWidgets.QPlainTextEdit()

        self.cmbMode = QtWidgets.QComboBox()
        self.cmbMode.addItems(['Summary', 'Histogram', 'Correlation', 'Autocorrelation', 'Analysis of Variance'])
        self.histctrls = HistCtrlWidget()
        self.corrctrls = CorrCtrlWidget()
        self.acorctrls = ACorrCtrlWidget()

        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setStyleSheet("background-color:transparent;")
        self.toolbar = NavigationToolbar(self.canvas, self, coordinates=True)
        self.txtOutput = widgets.MarkdownTextEdit()

        self.menu = QtWidgets.QMenu('&Data Set')
        self.actImportCol = QtGui.QAction('&New column from CSV...', self)
        self.actImport = QtGui.QAction('&Load data from CSV...', self)
        self.actClear = QtGui.QAction('&Clear data', self)
        self.actSaveReport = QtGui.QAction('&Save Report...', self)
        self.actSummary = QtGui.QAction('Enter Summarized &Values', self)
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
        self.btnAddRemCol.plusclicked.connect(self.addcol)
        self.btnAddRemCol.minusclicked.connect(self.remcol)
        self.table.valueChanged.connect(self.tablechanged)
        self.txtDescription.textChanged.connect(self.update_description)
        self.cmbMode.currentIndexChanged.connect(self.changemode)
        self.histctrls.changed.connect(self.refresh_output)
        self.corrctrls.changed.connect(self.refresh_output)
        self.acorctrls.changed.connect(self.refresh_output)
        self.tolerance.changed.connect(self.tolchanged)

        hlayout = QtWidgets.QHBoxLayout()
        hlayout.addWidget(QtWidgets.QLabel('Repeatability & Reproducibility Measurements:'))
        hlayout.addStretch()
        hlayout.addWidget(self.btnAddRemCol)
        topleft = QtWidgets.QVBoxLayout()
        topleft.addLayout(hlayout)
        topleft.addWidget(self.table)
        h2layout = QtWidgets.QHBoxLayout()
        h2layout.addWidget(QtWidgets.QLabel('Tolerance:'))
        h2layout.addWidget(self.tolerance)
        topleft.addLayout(h2layout)
        botleft = QtWidgets.QVBoxLayout()
        botleft.addWidget(QtWidgets.QLabel('Notes:'))
        botleft.addWidget(self.txtDescription)
        ctrllayout = QtWidgets.QHBoxLayout()
        ctrllayout.addWidget(self.cmbMode)
        ctrllayout.addWidget(self.histctrls)
        ctrllayout.addWidget(self.corrctrls)
        ctrllayout.addWidget(self.acorctrls)
        toprght = QtWidgets.QVBoxLayout()
        toprght.addLayout(ctrllayout)
        toprght.addWidget(self.txtOutput)
        botrght = QtWidgets.QVBoxLayout()
        botrght.addWidget(self.canvas)
        botrght.addWidget(self.toolbar)
        self.topleft = QtWidgets.QWidget()
        self.topleft.setLayout(topleft)
        self.botleft = QtWidgets.QWidget()
        self.botleft.setLayout(botleft)
        self.toprght = QtWidgets.QWidget()
        self.toprght.setLayout(toprght)
        self.botrght = QtWidgets.QWidget()
        self.botrght.setLayout(botrght)
        self.leftsplitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        self.leftsplitter.addWidget(self.topleft)
        self.leftsplitter.addWidget(self.botleft)
        self.leftsplitter.setCollapsible(0, False)  # Leave "Notes" collapsible
        self.rghtsplitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        self.rghtsplitter.addWidget(self.toprght)
        self.rghtsplitter.addWidget(self.botrght)
        self.rghtsplitter.setCollapsible(0, False)
        self.splitter = QtWidgets.QSplitter()
        self.splitter.addWidget(self.leftsplitter)
        self.splitter.addWidget(self.rghtsplitter)
        self.splitter.setCollapsible(0, False)
        self.splitter.setCollapsible(1, False)
        toplayout = QtWidgets.QVBoxLayout()
        toplayout.addWidget(self.splitter)
        self.setLayout(toplayout)

        self.init_data()
        self.changemode()  # Initialize setVisible on controls
        self.canvas.draw_idle()

    def calculate(self):
        ''' Run calculation. '''
        self.component.calculate()

    def init_data(self):
        ''' Initialize the table using data in dataset model '''
        with BlockedSignals(self.table):
            self.txtDescription.setPlainText(self.component.description)
            ncols = self.component.result.ncolumns
            nrows = self.component.result.maxrows
            if self.component.model.tolerance:
                with BlockedSignals(self.tolerance):
                    self.tolerance.chkbox.setChecked(True)
                    self.tolerance.tolerance.set_limit(self.component.result.tolerance)
            if self.component.issummary():
                self.actSummary.setChecked(True)
                means = self.component.result.groups.means
                stds = self.component.result.groups.std_devs
                counts = self.component.result.groups.counts
                ncols = len(means)
                self.table.setRowCount(3)
                self.table.setColumnCount(ncols)
                self.table.setVerticalHeaderLabels(['Mean', 'Std. Dev.', 'Count'])
                with BlockedSignals(self.cmbMode):
                    self.cmbMode.clear()
                    self.cmbMode.addItems(['Summary', 'Analysis of Variance'])
                for col in range(ncols):
                    self.table.setItem(0, col, QtWidgets.QTableWidgetItem(str(means[col])))
                    self.table.setItem(1, col, QtWidgets.QTableWidgetItem(str(stds[col])))
                    self.table.setItem(2, col, QtWidgets.QTableWidgetItem(str(counts[col])))

            else:
                if self.component.result.data.size > 0:
                    self.table.setColumnCount(ncols)
                    self.table.setRowCount(max(1, nrows))
                    self.table.setHorizontalHeaderLabels([str(s) for s in self.component.result.colnames])
                    for col in range(ncols):
                        for row in range(len(self.component.result.data[col])):
                            val = self.component.result.data[col][row]
                            if np.isfinite(val):
                                self.table.setItem(row, col, QtWidgets.QTableWidgetItem(str(val)))

        self.updatecolnames()
        self.refresh_output()

    def get_menu(self):
        ''' Get the menu for this calculation mode '''
        return self.menu

    def togglesummary(self):
        ''' Change from full table to summarized statistics mode '''
        if self.actSummary.isChecked():
            self.table.setRowCount(3)
            self.component.model = self.component.model.summarize()
            with BlockedSignals(self.cmbMode):
                self.cmbMode.clear()
                self.cmbMode.addItems(['Summary', 'Analysis of Variance'])
        else:
            self.component.model = DataSet()
            with BlockedSignals(self.cmbMode):
                self.cmbMode.clear()
                self.cmbMode.addItems(['Summary', 'Histogram', 'Correlation', 'Autocorrelation', 'Analysis of Variance'])
        self.component.calculate()
        self.table.clear()
        self.init_data()

    def changemode(self):
        ''' Change output mode. Show/hide controls as appropriate '''
        mode = self.cmbMode.currentText()
        self.histctrls.setVisible(mode == 'Histogram')
        self.corrctrls.setVisible(mode == 'Correlation')
        self.acorctrls.setVisible(mode == 'Autocorrelation')
        self.refresh_output()
        self.change_help.emit()

    def tablechanged(self):
        ''' Table data has changed, update the model '''
        data = self.table.get_table()
        if self.actSummary.isChecked():
            data = np.nan_to_num(data)
            self.component.model.means = data[:, 0]
            self.component.model.stdevs = data[:, 1]
            self.component.model.nmeas = data[:, 2]
            self.component.calculate()
        else:
            self.component.setdata(data)
            self.component.calculate()

        colnames = []
        for i in range(self.table.columnCount()):
            if self.table.horizontalHeaderItem(i):
                colnames.append(self.table.horizontalHeaderItem(i).text())
            else:
                # Needed when pasting multicolumn data
                colnames.append(str(i+1))
                self.table.setHorizontalHeaderItem(i, QtWidgets.QTableWidgetItem(str(i+1)))

        self.component.setcolnames(colnames)
        self.updatecolnames()
        self.refresh_output()

    def tolchanged(self):
        ''' Tolerance widget changed '''
        self.component.model.tolerance = self.tolerance.tolerance.limit() if self.tolerance.chkbox.isChecked() else None
        self.component.calculate()
        self.refresh_output()

    def updatecolnames(self):
        ''' Update column names in column-select widgets '''
        names = [str(c) for c in self.component.model.colnames]
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
            rpt.append(self.component.result.report.summary())
            if self.component.result.ncolumns > 1:
                rpt.hdr('Pooled Statistics', level=3)
                rpt.append(self.component.result.report.pooled())
                self.component.result.report.plot.groups(fig=self.figure)
            else:
                self.component.result.report.plot.histogram(fig=self.figure)

        elif mode == 'Histogram':
            col = self.histctrls.colSelect.currentText()
            fit = self.histctrls.fit.currentText()
            fit = None if fit == 'None' else fit
            probplot = self.histctrls.probplot.isChecked()
            rpt.hdr(col, level=2)
            rpt.append(self.component.result.report.column(col))
            self.component.result.report.plot.histogram(colname=col, fig=self.figure, fit=fit, qqplot=probplot)

        elif mode == 'Correlation':
            rpt.hdr('Correlation Matrix', level=2)
            rpt.append(self.component.result.report.correlation())
            col1 = self.corrctrls.col1Select.currentText()
            col2 = self.corrctrls.col2Select.currentText()
            self.component.result.report.plot.scatter(col1, col2, fig=self.figure)

        elif mode == 'Autocorrelation':
            rpt.hdr('Autocorrelation', level=2)
            col = self.acorctrls.colSelect.currentText()
            lag = self.acorctrls.lag.value()
            rpt.append(self.component.result.report.autocorrelation())
            if self.acorctrls.mode.currentText() == 'Lag Plot':
                with suppress(ValueError):  # Raises when lag too high
                    self.component.result.report.plot.lag(col, lag=lag, fig=self.figure)
            else:
                self.component.result.report.plot.autocorrelation(col, fig=self.figure)

        elif mode == 'Analysis of Variance':
            rpt.hdr('Analysis of Variance', level=2)
            rpt.append(self.component.result.report.anova())
            self.component.result.report.plot.groups(fig=self.figure)

        else:
            raise NotImplementedError

        self.txtOutput.setReport(rpt)
        self.canvas.draw_idle()

    def clear(self):
        ''' Clear the table '''
        with BlockedSignals(self.table):
            self.table.clear()
            self.table.setRowCount(1)
            self.table.setColumnCount(1)
            self.table.setItem(0, 0, QtWidgets.QTableWidgetItem(''))
            self.table.setHorizontalHeaderLabels(['1'])
        self.component.clear()

    def addcol(self):
        ''' Add a column to the table '''
        col = self.table.columnCount()
        self.table.setColumnCount(col + 1)
        # Must do this to make it editable for some reason:
        self.table.setHorizontalHeaderItem(col, QtWidgets.QTableWidgetItem(str(col+1)))

    def remcol(self):
        ''' Remove selected column from table '''
        self.table.removeColumn(self.table.currentColumn())
        if self.table.columnCount() == 0:
            self.table.setColumnCount(1)
        self.tablechanged()

    def update_description(self):
        ''' Description was updated, save it. '''
        self.component.description = self.txtDescription.toPlainText()

    def importcolumn(self):
        ''' Add one column from a CSV file '''
        fname, _ = QtWidgets.QFileDialog.getOpenFileName(caption='Select CSV file to load')
        if fname:
            dlg = page_csvload.SelectCSVData(fname)
            if dlg.exec():
                with BlockedSignals(self.table):
                    dset = dlg.dataset().model
                    data = dset.result.data[0]  # Only first column is included when importing a column
                    hdr = dset.colnames   # List of strings
                    rowcnt = len(data)

                    if self.table.rowCount() > 1 or self.table.columnCount() > 1:
                        self.addcol()
                    self.table.setRowCount(max(rowcnt, self.table.rowCount()))
                    col = self.table.columnCount() - 1

                    if len(hdr) > 0:
                        self.table.setHorizontalHeaderItem(col, QtWidgets.QTableWidgetItem(hdr[0]))

                    for row, item in enumerate(data):
                        self.table.setItem(row, col, QtWidgets.QTableWidgetItem(str(item)))
                self.tablechanged()

    def importtable(self):
        ''' Import data table from a CSV file '''
        fname, _ = QtWidgets.QFileDialog.getOpenFileName(caption='Select CSV file to load')
        if fname:
            dlg = page_csvload.SelectCSVData(fname)
            if dlg.exec():
                with BlockedSignals(self.table):
                    self.table.clear()
                    dset = dlg.dataset().model
                    data = dset.data
                    hdr = dset.colnames
                    colcnt = max(len(hdr), len(data))
                    rowcnt = max(len(c) for c in data)

                    self.table.setRowCount(rowcnt)
                    self.table.setColumnCount(colcnt)
                    if len(hdr) > 0:
                        self.table.setHorizontalHeaderLabels(hdr)
                    for col in range(colcnt):
                        for row in range(len(data[col])):
                            self.table.setItem(row, col, QtWidgets.QTableWidgetItem(str(data[col][row])))
                self.tablechanged()

    def update_proj_config(self):
        ''' Save page setup back to project item configuration '''
        self.component.calculate()

    def get_report(self):
        ''' Get full report of dataset, using page settings '''
        return self.component.result.report.all()

    def save_report(self):
        ''' Save full report, asking user for settings/filename '''
        with gui_styles.LightPlotstyle():
            widgets.savereport(self.get_report())

    def help_report(self):
        ''' Get the help report to display the current widget mode '''
        mode = self.cmbMode.currentText()
        if mode == 'Summary':
            return AnovaHelp.summary()
        elif mode == 'Histogram':
            return AnovaHelp.histogram()
        elif mode == 'Correlation':
            return AnovaHelp.correlation()
        elif mode == 'Autocorrelation':
            return AnovaHelp.autocorrelation()
        elif mode == 'Analysis of Variance':
            return AnovaHelp.anova()
        else:
            return AnovaHelp.nohelp()
