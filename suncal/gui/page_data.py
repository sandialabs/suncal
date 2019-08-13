''' User Interface for loading data from CSV '''

import numpy as np
from scipy import stats
from dateutil.parser import parse

from PyQt5 import QtWidgets, QtCore
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from .. import output
from .. import customdists
from . import configmgr
from . import gui_common
from . import gui_widgets

settings = configmgr.Settings()


def load_csvfile(fname):
    ''' Load CSV file into raw string '''
    with open(fname, 'r') as f:
        try:
            rawcsv = f.read()
        except UnicodeDecodeError:
            QtWidgets.QMessageBox.warning(None, 'Load CSV', 'Cannot decode file {}'.format(fname))
            rawcsv = None
        else:
            rawcsv = rawcsv.strip(u'\ufeff')  # uFEFF is junk left in Excel-saved CSV files at start of file
    return rawcsv


class ProjectTreeDelegate(QtWidgets.QItemDelegate):
    ''' Delegate for project tree items '''
    def sizeHint(self, option, index):
        return QtCore.QSize(32, 32)


class ProjectTreeDists(QtWidgets.QTreeWidget):
    ''' Tree Widget showing all project items with usable distributions '''
    ROLE_CALC = QtCore.Qt.UserRole
    ROLE_NAME = QtCore.Qt.UserRole + 1
    loadcsv = QtCore.pyqtSignal(str)            # Emit file name to load
    loadsamples = QtCore.pyqtSignal(object)     # Emit the monte carlo samples for fitting distribution
    loadnorm = QtCore.pyqtSignal(float, float, float)  # Emit the mean, stddev, degf of the normal/t distribution

    def __init__(self, project=None, parent=None):
        super(ProjectTreeDists, self).__init__(parent=parent)
        self.project = project
        self.setMinimumWidth(250)
        self.setIconSize(QtCore.QSize(32, 32))
        self.setItemDelegate(ProjectTreeDelegate())
        self.setColumnCount(1)
        self.setHeaderHidden(True)

        if self.project:
            names = self.project.get_names()
            for i in range(self.project.count()):
                mode = self.project.get_mode(i)
                if mode in ['uncertainty', 'reverse', 'curvefit', 'anova']:
                    try:
                        dists = self.project.items[i].get_output().get_dists()  # No args to get names
                    except AttributeError:
                        pass
                    else:
                        item = QtWidgets.QTreeWidgetItem([names[i]])
                        item.setIcon(0, gui_common.load_icon(gui_common.iconname[mode]))
                        self.addTopLevelItem(item)

                        for dist in dists:
                            distitem = QtWidgets.QTreeWidgetItem([dist])
                            if '(MC)' in dist or '(MCMC)' in dist:
                                distitem.setIcon(0, gui_common.load_icon('histogram'))
                            else:
                                distitem.setIcon(0, gui_common.load_icon('normal'))
                            distitem.setData(0, self.ROLE_CALC, self.project.items[i])
                            distitem.setData(0, self.ROLE_NAME, dist)
                            item.addChild(distitem)
                        item.setExpanded(True)

        item = QtWidgets.QTreeWidgetItem(['Select CSV File...'])
        item.setIcon(0, gui_common.load_icon('table'))
        self.addTopLevelItem(item)
        if QtWidgets.QApplication.instance().clipboard().text() != '':
            item = QtWidgets.QTreeWidgetItem(['Clipboard'])
            item.setIcon(0, gui_common.load_icon('clipboard'))
            self.addTopLevelItem(item)
        self.itemDoubleClicked.connect(self.treeclick)

    def treeclick(self, item, column):
        ''' The tree was clicked. '''
        if item.text(0) == 'Select CSV File...':
            fname, _ = QtWidgets.QFileDialog.getOpenFileName(caption='CSV file to load')
            if fname:
                rawcsv = load_csvfile(fname)
                if rawcsv:
                    self.loadcsv.emit(rawcsv)

        elif item.text(0) == 'Clipboard':
            # Note: clipboard data is string, tab-separated lines. CSV parser should work.
            strdata = QtWidgets.QApplication.instance().clipboard().text()
            self.loadcsv.emit(strdata)

        else:
            calc = item.data(0, self.ROLE_CALC)
            name = item.data(0, self.ROLE_NAME)

            if 'Confidence' in item.text(0) or 'Prediction' in item.text(0):
                out = calc.get_output()
                if not out.xdates:
                    val, ok = QtWidgets.QInputDialog.getDouble(self, 'Enter Value', 'X-Value at which to calculate interval')
                else:
                    val, ok = gui_widgets.DateDialog.getDate()
                    val = val.toPyDate().toordinal()

                data = out.get_dists(name, xval=val)
                self.loadnorm.emit(data['mean'], data['std'], data['df'])

            elif '(MC)' in name or '(MCMC)' in name:
                data = calc.get_output().get_dists(name)
                self.loadsamples.emit(data)
            else:
                data = calc.get_output().get_dists(name)
                self.loadnorm.emit(data['mean'], data['std'], data['df'])


class ProjectTreeArrays(QtWidgets.QTreeWidget):
    ''' Widget for showing project tree with all calculated arrays to choose from.

        Arrays may come from sweeps, reverse-sweeps, or anova grouped results. OR an
        array may be loaded from CSV.
    '''
    ROLE_CALC = QtCore.Qt.UserRole
    ROLE_NAME = QtCore.Qt.UserRole + 1
    loadcsv = QtCore.pyqtSignal(str)  # Emit file name to load
    loaddata = QtCore.pyqtSignal(object)  # Emit the actual data array

    def __init__(self, project, parent=None):
        super(ProjectTreeArrays, self).__init__(parent=parent)
        self.project = project
        self.setMinimumWidth(250)
        self.setIconSize(QtCore.QSize(32, 32))
        self.setItemDelegate(ProjectTreeDelegate())
        self.setColumnCount(1)
        self.setHeaderHidden(True)

        if self.project:
            names = self.project.get_names()
            for i in range(self.project.count()):
                mode = self.project.get_mode(i)
                if mode in ['sweep', 'reversesweep', 'anova']:
                    try:
                        arrays = self.project.items[i].get_output().get_array()  # No args to get names
                    except AttributeError:
                        pass
                    else:
                        item = QtWidgets.QTreeWidgetItem([names[i]])
                        item.setIcon(0, gui_common.load_icon(gui_common.iconname[mode]))
                        self.addTopLevelItem(item)

                        for arr in arrays:
                            arritem = QtWidgets.QTreeWidgetItem([arr])
                            arritem.setIcon(0, gui_common.load_icon('column'))
                            arritem.setData(0, self.ROLE_CALC, self.project.items[i])
                            arritem.setData(0, self.ROLE_NAME, arr)
                            item.addChild(arritem)
                        item.setExpanded(True)

        item = QtWidgets.QTreeWidgetItem(['Select CSV File...'])
        item.setIcon(0, gui_common.load_icon('table'))
        self.addTopLevelItem(item)
        if QtWidgets.QApplication.instance().clipboard().text() != '':
            item = QtWidgets.QTreeWidgetItem(['Clipboard'])
            item.setIcon(0, gui_common.load_icon('clipboard'))
            self.addTopLevelItem(item)

        self.itemDoubleClicked.connect(self.treeclick)

    def treeclick(self, item, column):
        ''' The tree was clicked. '''
        if item.text(0) == 'Select CSV File...':
            fname, _ = QtWidgets.QFileDialog.getOpenFileName(caption='CSV file to load')
            if fname:
                rawcsv = load_csvfile(fname)
                if rawcsv:
                    self.loadcsv.emit(rawcsv)

        elif item.text(0) == 'Clipboard':
            # Note: clipboard data is string, tab-separated lines. CSV parser should work.
            strdata = QtWidgets.QApplication.instance().clipboard().text()
            self.loadcsv.emit(strdata)

        else:
            calc = item.data(0, self.ROLE_CALC)
            try:
                data = calc.get_output().get_array(item.data(0, self.ROLE_NAME))
            except AttributeError:
                pass  # Could be top-level item
            else:
                self.loaddata.emit(data)


class SelectCSVData(QtWidgets.QDialog):
    ''' Widget for displaying raw CSV and letting user select which data ranges
        to import
    '''
    def __init__(self, rawcsv=None, convertdates=False, parent=None):
        super(SelectCSVData, self).__init__(parent=parent)
        self.convertdates = convertdates
        self.setGeometry(600, 200, 900, 600)
        self.table = QtWidgets.QTableWidget()
        self.dlgbutton = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        self.dlgbutton.rejected.connect(self.reject)
        self.dlgbutton.accepted.connect(self.checkdatarange)
        self.dlgbutton.button(QtWidgets.QDialogButtonBox.Ok).setEnabled(False)
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(QtWidgets.QLabel('Select data range(s) to import'))
        layout.addWidget(self.table)
        layout.addWidget(self.dlgbutton)
        self.setLayout(layout)
        self.loadfile(rawcsv)
        self.table.itemSelectionChanged.connect(self.selection_change)

    def loadfile(self, rawcsv):
        ''' Populate the table with CSV values '''
        self.rawcsv = rawcsv
        if ',' in self.rawcsv:
            self.delim = ','
        elif '\t' in self.rawcsv:
            self.delim = '\t'
        else:
            self.delim = None  # None will group whitespace, ' ' will end up with multiple splits

        lines = self.rawcsv.splitlines()
        self.table.setRowCount(len(lines))
        self.table.setColumnCount(1)
        for row, line in enumerate(lines):
            columns = line.split(self.delim)
            self.table.setColumnCount(max(self.table.columnCount(), len(columns)))
            for col, val in enumerate(columns):
                self.table.setItem(row, col, QtWidgets.QTableWidgetItem(val.strip()))
                self.table.item(row, col).setFlags(QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEnabled)  # No editable
        self.table.resizeColumnsToContents()

    def selection_change(self):
        rng = self.table.selectedRanges()
        self.dlgbutton.button(QtWidgets.QDialogButtonBox.Ok).setEnabled(len(rng) > 0)

    def checkdatarange(self):
        columns = []
        # Get selected cell text
        for rng in self.table.selectedRanges():
            for col in range(rng.columnCount()):
                rstart = rng.topRow()
                rcount = rng.rowCount()
                cidx = rng.leftColumn() + col
                columns.append([self.table.item(rstart+i, cidx).text() for i in range(rcount)])

        # Convert each column to a float or a date
        datcolumns = []
        for col in columns:
            try:
                datcol = [float(v) for v in col]
            except ValueError:
                try:
                    datcol = [parse(v) for v in col]
                    if self.convertdates:
                        datcol = [d.toordinal() for d in datcol]
                except ValueError:
                    QtWidgets.QMessageBox.warning(self, 'Import CSV', 'Non-numeric data in selection')
                    return
            datcolumns.append(np.array(datcol))

        # self.columns will be array of floats or of datetimes
        self.columns = datcolumns
        self.accept()


class DistributionSelectWidget(QtWidgets.QDialog):
    ''' Widget for selecting a distribution from the project or a file '''
    def __init__(self, fname=None, project=None, parent=None):
        super(DistributionSelectWidget, self).__init__(parent=parent)
        self.setGeometry(600, 200, 1000, 800)
        self.setWindowTitle('Select Distribution')
        self.data = None
        self.dist = None
        self.sampleddist = True  # Showing sampled (as opposed to normal) distribution?
        self.project = project

        dists = settings.getDistributions()
        dists = [d for d in dists if hasattr(customdists.get_dist(d), 'fit')]
        self.treeSource = ProjectTreeDists(project=project)
        self.fig = Figure()
        self.canvas = FigureCanvas(self.fig)
        self.cmbDistType = QtWidgets.QComboBox()
        self.cmbDistType.addItems(dists + ['histogram'])
        self.chkProbPlot = QtWidgets.QCheckBox('Show Probability Plot')
        self.lblDist = QtWidgets.QLabel('Distribution to Fit:')
        self.dlgbutton = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        self.dlgbutton.rejected.connect(self.reject)
        self.dlgbutton.accepted.connect(self.accept)

        llayout = QtWidgets.QVBoxLayout()
        llayout.addWidget(QtWidgets.QLabel('Data Source:'))
        llayout.addWidget(self.treeSource)
        hlayout = QtWidgets.QHBoxLayout()
        hlayout.addWidget(self.lblDist)
        hlayout.addWidget(self.cmbDistType)
        hlayout.addWidget(self.chkProbPlot)
        rlayout = QtWidgets.QVBoxLayout()
        rlayout.addLayout(hlayout)
        rlayout.addWidget(self.canvas)
        rlayout.addWidget(self.dlgbutton)
        layout = QtWidgets.QHBoxLayout()
        layout.addLayout(llayout)
        layout.addLayout(rlayout)
        self.setLayout(layout)

        if fname is not None:
            rawcsv = load_csvfile(fname)
            if rawcsv:
                self.load_csv(rawcsv)

        self.cmbDistType.currentIndexChanged.connect(self.updateplot)
        self.chkProbPlot.stateChanged.connect(self.updateplot)
        self.treeSource.loadcsv.connect(self.load_csv)
        self.treeSource.loadsamples.connect(self.load_samples)
        self.treeSource.loadnorm.connect(self.load_norm)

    def updateplot(self):
        ''' Update the distribution and plot '''
        self.fig.clf()
        ax = self.fig.add_subplot(1, 1, 1)

        if self.sampleddist:
            self.lblDist.setVisible(True)
            self.chkProbPlot.setVisible(True)
            self.cmbDistType.setVisible(True)

            data = self.data
            self.distname = self.cmbDistType.currentText()
            if self.distname == 'histogram':
                fitdist = customdists.get_dist(self.distname)(self.data)
                self.params = {'hist': fitdist._histogram[0], 'edges': fitdist._histogram[1]}
                self.median = 0

            else:
                rv = customdists.get_dist(self.distname)
                paramnames = rv.shapes
                if paramnames is not None:
                    paramnames = [r.strip() for r in paramnames.split(',')]
                    paramnames += ['loc', 'scale']
                else:
                    paramnames = ['loc', 'scale']

                self.params = dict(zip(paramnames, rv.fit(data)))
                fitdist = rv(**self.params)
                self.median = fitdist.median()
                self.params['std'] = self.params['scale']

            if self.chkProbPlot.isChecked() and self.distname != 'histogram':
                output.probplot(data, ax=ax, sparams=self.params.values(), dist=fitdist.dist)
            else:
                # Plot histogram
                xx = np.linspace(data.min(), data.max(), num=200)
                yy = fitdist.pdf(xx)
                try:
                    ax.hist(data, density=True, bins='sqrt')
                except ValueError:
                    pass
                ax.plot(xx, yy, label='Distribution Fit')
                ax.set_ylabel('Probability Density')

        else:   # not self.sampleddist
            # Show the GUM distribution
            median = self.tdist['mean']
            std = self.tdist['std']
            degf = self.tdist['df']

            self.lblDist.setVisible(False)
            self.chkProbPlot.setVisible(False)
            self.cmbDistType.setVisible(False)
            dist = stats.norm(loc=median, scale=std)

            xx = np.linspace(median - 5*std, median + 5*std, num=200)
            yy = dist.pdf(xx)
            ax.plot(xx, yy, label='Normal Distribution')

            self.params = {'median': median, 'std': std}
            self.median = median
            if degf > 1000:
                self.distname = 'normal'
            else:
                self.distname = 't'
                self.params['df'] = degf
        self.canvas.draw_idle()

    def get_dist(self):
        ''' Get stats distribution '''
        return self.distname, self.params, self.median

    def load_samples(self, data):
        ''' Load a sample set into the widget for fitting a distribution to. '''
        self.sampleddist = True
        self.data = data
        self.updateplot()

    def load_norm(self, mean, stdev, degf):
        ''' Display a normal or t distribution (usually from a GUM calculation) '''
        self.sampleddist = False
        self.tdist = {'mean': mean, 'std': stdev, 'df': degf}
        self.updateplot()

    def load_csv(self, rawcsv):
        ''' Load sampled data from csv. rawcsv is string of csv contents. '''
        csvdlg = SelectCSVData(rawcsv, convertdates=True, parent=self)
        ok = csvdlg.exec_()
        if ok:
            self.data = csvdlg.columns[0]  # Only first column is used for histogram
            self.sampleddist = True
            self.updateplot()


class ArraySelectWidget(QtWidgets.QDialog):
    ''' Widget for selecting an array of values from the project or a file '''
    def __init__(self, rawcsv=None, project=None, name_cols=True, parent=None):
        super(ArraySelectWidget, self).__init__(parent=parent)
        self.project = project
        self.name_cols = name_cols
        self.setGeometry(600, 200, 1000, 800)
        self.setWindowTitle('Select Array')

        self.data = None
        self.treeSource = ProjectTreeArrays(project=project)
        self.table = QtWidgets.QTableWidget()
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectColumns)
        self.cmbColumn = QtWidgets.QComboBox()
        self.cmbColumn.addItems(['Not Used', 'x', 'y', 'u(x)', 'u(y)'])
        self.fig = Figure()
        self.canvas = FigureCanvas(self.fig)
        self.dlgbutton = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        self.dlgbutton.rejected.connect(self.reject)
        self.dlgbutton.accepted.connect(self.accept)
        self.lblColumn = QtWidgets.QLabel('Selected Column:')  # Storing to self so we can hide it later

        llayout = QtWidgets.QVBoxLayout()
        llayout.addWidget(QtWidgets.QLabel('Data Source:'))
        llayout.addWidget(self.treeSource)
        hlayout = QtWidgets.QHBoxLayout()
        hlayout.addWidget(self.lblColumn)
        hlayout.addWidget(self.cmbColumn)
        hlayout.addStretch()
        rlayout = QtWidgets.QVBoxLayout()
        rlayout.addWidget(self.canvas, stretch=5)
        rlayout.addLayout(hlayout)
        rlayout.addWidget(self.table, stretch=5)
        rlayout.addWidget(self.dlgbutton)
        layout = QtWidgets.QHBoxLayout()
        layout.addLayout(llayout)
        layout.addLayout(rlayout)
        self.setLayout(layout)

        if rawcsv:
            self.load_file(rawcsv)

        self.updateplot()
        self.cmbColumn.currentIndexChanged.connect(self.tablecolumnchanged)
        self.table.currentCellChanged.connect(self.columnselected)
        self.treeSource.loadcsv.connect(self.load_file)
        self.treeSource.loaddata.connect(self.load_data)

    def tablecolumnchanged(self):
        ''' Column assignment has changed '''
        varname = self.cmbColumn.currentText()
        varlist = self.getvarlist()
        if varname == 'Not used' or varname == '':
            self.table.setHorizontalHeaderItem(self.table.currentColumn(), QtWidgets.QTableWidgetItem(''))
        else:
            if varname in varlist:
                self.table.setHorizontalHeaderItem(varlist.index(varname), QtWidgets.QTableWidgetItem(''))
            self.table.setHorizontalHeaderItem(self.table.currentColumn(), QtWidgets.QTableWidgetItem(varname))
        self.updateplot()

    def columnselected(self, row, col, prow, pcol):
        ''' Column was highlighted in table. Update stats labels and combobox with variable selection. '''
        self.cmbColumn.blockSignals(True)
        try:
            colvar = self.table.horizontalHeaderItem(col).text()
        except AttributeError:
            colvar = None

        self.cmbColumn.setCurrentIndex(self.cmbColumn.findText(colvar))
        if self.cmbColumn.currentIndex() < 0:
            self.cmbColumn.setCurrentIndex(0)  # Not used
        self.cmbColumn.blockSignals(False)
        self.updateplot()

    def updateplot(self):
        ''' Update the plot '''
        colnames = [self.table.horizontalHeaderItem(i).text() if self.table.horizontalHeaderItem(i) else '' for i in range(self.table.columnCount())]

        self.fig.clf()
        ax = self.fig.add_subplot(1, 1, 1)
        x = self.get_column_data('x')
        y = self.get_column_data('y')
        uy = self.get_column_data('u(y)')
        ux = self.get_column_data('u(x)')
        ux = None if len(ux) == 0 else ux
        uy = None if len(uy) == 0 else uy

        if len(x) == 0 and len(y) > 0:
            x = np.arange(len(y), dtype=float)

        # Can't plot datetime as errorbar
        if len(x) > 0 and hasattr(x[0], 'date'):
            ux = None
        if len(y) > 0 and hasattr(y[0], 'date'):
            uy = None
        if ux is not None and hasattr(ux[0], 'date'):
            ux = None
        if uy is not None and hasattr(uy[0], 'date'):
            uy = None

        if len(y) > 0:
            if len(x) != len(y):
                x = x[:len(y)]
                y = y[:len(x)]
            if uy is not None and len(uy) != len(y):
                newuy = np.zeros(len(y))
                newuy[:len(uy)] = uy[:len(newuy)]
                uy = newuy
            if ux is not None and len(ux) != len(x):
                newux = np.zeros(len(x))
                newux[:len(ux)] = ux[:len(newux)]
                ux = newux

            ax.errorbar(x, y, yerr=uy, xerr=ux, marker='o', ls='', capsize=5)
            ax.set_xlabel('x')
            ax.set_ylabel('y')
        self.canvas.draw_idle()

    def getvarlist(self):
        ''' Get list of variable names for each column '''
        return [self.table.horizontalHeaderItem(i).text() if self.table.horizontalHeaderItem(i) else '' for i in range(self.table.columnCount())]

    def is_datecol(self, colname):
        ''' Return True if column with name === colname is date '''
        isdate = False
        colnames = self.getvarlist()
        if colname in colnames:
            idx = colnames.index(colname)
            isdate = hasattr(self.data[idx][0], 'date')
        return isdate

    def get_column_data(self, colname):
        ''' Get data for column with name == colname '''
        colnames = self.getvarlist()
        if colname in colnames:
            return self.data[colnames.index(colname)]
        else:
            return []

    def load_data(self, arr):
        ''' Use fill table with data array

            Parameters
            ----------
            arr: uarray.Array object
                Array data to use in table
        '''
        self.table.clear()
        self.table.setRowCount(len(arr))
        self.table.setColumnCount(2 + arr.has_ux() + arr.has_uy())
        for row in range(len(arr)):
            item = QtWidgets.QTableWidgetItem(str(arr.x[row]))
            item.setFlags(QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEnabled)
            self.table.setItem(row, 0, item)
            item = QtWidgets.QTableWidgetItem(str(arr.y[row]))
            item.setFlags(QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEnabled)
            self.table.setItem(row, 1, item)
            if arr.has_uy():
                item = QtWidgets.QTableWidgetItem(str(arr.uy[row]))
                item.setFlags(QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEnabled)
                self.table.setItem(row, 2, item)
            if arr.has_ux():
                item = QtWidgets.QTableWidgetItem(str(arr.ux[row]))
                item.setFlags(QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEnabled)
                self.table.setItem(row, 3, item)

        if self.name_cols:
            self.table.setHorizontalHeaderLabels(['x', 'y', 'u(y)' if arr.has_uy() else '', 'u(x)' if arr.has_ux() else ''])

        self.data = arr.get_numpy().transpose()
        self.updateplot()

    def load_file(self, rawcsv):
        ''' Load data from the raw csv string '''
        csvdlg = SelectCSVData(rawcsv, parent=self)
        ok = csvdlg.exec_()
        if ok:
            #    print(csvdlg.columns)
            columns = csvdlg.columns
            self.table.clear()
            self.table.setColumnCount(len(columns))
            self.table.setRowCount(max([len(c) for c in columns]))
            for cidx, col in enumerate(columns):
                for ridx, row in enumerate(col):
                    self.table.setItem(ridx, cidx, QtWidgets.QTableWidgetItem(str(row)))   # TODO: format datetimes?

            self.data = columns
            if self.name_cols:
                self.table.setHorizontalHeaderLabels(['x', 'y'] + [''] * (self.table.columnCount() - 2))
        self.updateplot()


class SweepSelectWidget(ArraySelectWidget):
    ''' Widget for selecting ONE column of sweep data '''
    def __init__(self, rawcsv=None, project=None, parent=None):
        super(SweepSelectWidget, self).__init__(rawcsv=rawcsv, project=project, parent=parent, name_cols=False)
        self.cmbColumn.setVisible(False)
        self.lblColumn.setVisible(False)

    def updateplot(self):
        ''' Update the plot '''
        if len(self.table.selectedRanges()) == 0:
            self.table.selectColumn(0)
        self.fig.clf()
        ax = self.fig.add_subplot(1, 1, 1)
        y = self.get_sweep()
        if y is not None:
            # Plot against range(len(y))
            ax.plot(y, marker='o', ls='')
            ax.set_ylabel('y')
        self.canvas.draw_idle()

    def get_sweep(self):
        ''' Get array of sweep values '''
        ycol = self.table.currentColumn()
        if ycol >= 0:
            y = self.data[ycol]
            return y
        return None


class UncertsFromCSV(QtWidgets.QDialog):
    ''' Widget for loading uncert propagation variable statistics from CSV file '''
    def __init__(self, fname=None, colnames=None, parent=None):
        super(UncertsFromCSV, self).__init__(parent=parent)
        self.colnames = colnames
        self.setGeometry(600, 200, 900, 600)
        self.setWindowTitle('Load Uncertainties from File')
        self.table = QtWidgets.QTableWidget()
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectColumns)
        self.cmbColumn = QtWidgets.QComboBox()
        self.lblmean = QtWidgets.QLabel('Mean: --')
        self.lblstdunc = QtWidgets.QLabel('Standard Deviation of Mean: --')
        self.lbldegf = QtWidgets.QLabel('Degrees of Freedom: --')

        self.bins = QtWidgets.QSpinBox()
        self.bins.setValue(20)
        self.bins.setRange(3, 1000)
        self.chkNorm = QtWidgets.QCheckBox('Show Normal')
        self.chkProbPlot = QtWidgets.QCheckBox('Show Normal Probability Plot')
        self.fig = Figure()
        self.canvas = FigureCanvas(self.fig)
        self.dlgbutton = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        self.dlgbutton.rejected.connect(self.reject)
        self.dlgbutton.accepted.connect(self.accept)
        if colnames is not None:
            self.cmbColumn.addItems(['Not Used'] + colnames)

        hlayout = QtWidgets.QHBoxLayout()
        hlayout.addWidget(QtWidgets.QLabel('Variable for Column:'))
        hlayout.addWidget(self.cmbColumn)
        hlayout.addStretch()
        llayout = QtWidgets.QVBoxLayout()
        llayout.addLayout(hlayout)
        llayout.addWidget(self.table)
        llayout.addWidget(self.lblmean)
        llayout.addWidget(self.lblstdunc)
        llayout.addWidget(self.lbldegf)
        playout = QtWidgets.QHBoxLayout()
        playout.addWidget(QtWidgets.QLabel('Bins:'))
        playout.addWidget(self.bins)
        playout.addWidget(self.chkNorm)
        playout.addWidget(self.chkProbPlot)
        playout.addStretch()
        rlayout = QtWidgets.QVBoxLayout()
        rlayout.addLayout(playout)
        rlayout.addWidget(self.canvas)
        rlayout.addWidget(self.dlgbutton)
        layout = QtWidgets.QHBoxLayout()
        layout.addLayout(llayout, stretch=2)
        layout.addLayout(rlayout, stretch=3)
        self.setLayout(layout)

        self.cmbColumn.currentIndexChanged.connect(self.tablecolumnchanged)
        self.table.currentCellChanged.connect(self.columnselected)
        self.bins.valueChanged.connect(self.updateplot)
        self.chkNorm.stateChanged.connect(self.updateplot)
        self.chkProbPlot.stateChanged.connect(self.updateplot)

        if fname is not None:
            rawcsv = load_csvfile(fname)
            if rawcsv:
                self.load_file(rawcsv)

    def load_file(self, rawcsv):
        ''' Load data from the raw csv string '''
        csvdlg = SelectCSVData(rawcsv, convertdates=True, parent=self)
        ok = csvdlg.exec_()
        if ok:
            columns = csvdlg.columns
            self.table.clear()
            self.table.setColumnCount(len(columns))
            self.table.setRowCount(max([len(c) for c in columns]))
            for cidx, col in enumerate(columns):
                for ridx, row in enumerate(col):
                    self.table.setItem(ridx, cidx, QtWidgets.QTableWidgetItem(str(row)))   # TODO: format datetimes?

            self.data = columns
            maxlen = max([len(c) for c in columns])
            self.table.setHorizontalHeaderLabels(['']*self.table.columnCount())
            self.bins.setValue(min(max(7, int(np.sqrt(maxlen)*3)), 200))

    def getvarlist(self):
        ''' Get list of variable names for each column '''
        return [self.table.horizontalHeaderItem(i).text() if self.table.horizontalHeaderItem(i) else '' for i in range(self.table.columnCount())]

    def tablecolumnchanged(self):
        ''' Column assignment has changed '''
        varname = self.cmbColumn.currentText()
        varlist = self.getvarlist()
        if varname == 'Not used' or varname == '':
            self.table.setHorizontalHeaderItem(self.table.currentColumn(), QtWidgets.QTableWidgetItem(''))
        else:
            if varname in varlist:
                self.table.setHorizontalHeaderItem(varlist.index(varname), QtWidgets.QTableWidgetItem(''))
            self.table.setHorizontalHeaderItem(self.table.currentColumn(), QtWidgets.QTableWidgetItem(varname))
        self.updateplot()

    def updateplot(self):
        ''' Update the plot. '''
        column = self.table.currentColumn()
        try:
            coldata = self.data[column]
        except (ValueError, IndexError, TypeError):
            self.lblmean.setText('Mean: --')
            self.lblstdunc.setText('Standard Deviation of Mean: --')
            self.lbldegf.setText('Degrees of Freedom: --')
            self.fig.clf()
        else:
            mean = coldata.mean()
            n = len(coldata)
            varmean = coldata.var(ddof=1) / n
            stdunc = np.sqrt(varmean)
            degf = n - 1
            self.lblmean.setText('Mean: {:.4g}'.format(mean))
            self.lblstdunc.setText('Standard Deviation of Mean: {:.4g}'.format(stdunc))
            self.lbldegf.setText('Degrees of Freedom: {:.4g}'.format(degf))

            self.fig.clf()
            ax = self.fig.add_subplot(1, 1, 1)
            if not self.chkProbPlot.isChecked():
                try:
                    ax.hist(coldata, bins=self.bins.value(), density=True)
                except ValueError:
                    pass
                else:
                    if self.table.horizontalHeaderItem(column) and self.table.horizontalHeaderItem(column).text() != '':
                        ax.set_xlabel(self.table.horizontalHeaderItem(column).text())
                    else:
                        ax.set_xlabel('Column {}'.format(column+1))
                    ax.set_ylabel('Probability Density')

                    if self.chkNorm.isChecked():
                        xx = np.linspace(coldata.min(), coldata.max(), num=250)
                        yy = stats.norm.pdf(xx, loc=mean, scale=stdunc)
                        ax.plot(xx, yy)
            else:
                (osm, osr), (slope, intercept, r) = stats.probplot(coldata, dist='norm')
                ax.plot(osm, osr, marker='o', ls='', label='Samples')
                xx = np.linspace(osm.min(), osm.max())
                ax.plot(xx, np.poly1d((slope, intercept))(xx), color='C1', label='Line Fit')
                ax.set_xlabel('Theoretical Quantiles')
                ax.set_ylabel('Ordered Sample Values')
                ax.legend(loc='upper left')

            self.fig.tight_layout()
            self.canvas.draw_idle()

    def columnselected(self, row, col, prow, pcol):
        ''' Column was highlighted in table. Update stats labels and combobox with variable selection. '''
        self.cmbColumn.blockSignals(True)
        colvar = self.table.horizontalHeaderItem(col).text() if self.table.horizontalHeaderItem(col) else ''
        self.cmbColumn.setCurrentIndex(self.cmbColumn.findText(colvar))
        if self.cmbColumn.currentIndex() < 0:
            self.cmbColumn.setCurrentIndex(0)  # Not used
        self.cmbColumn.blockSignals(False)
        self.updateplot()

    def getdata(self):
        ''' Get data in numpy array '''
        data = np.array(self.data).transpose()
        return data

    def get_statistics(self):
        ''' Get list of statistics as selected in table '''
        data = self.getdata()
        n = len(data)
        stats = []
        usedcols = []
        for col in range(self.table.columnCount()):
            if self.table.horizontalHeaderItem(col) and self.table.horizontalHeaderItem(col).text() != '':
                usedcols.append(col)
                stats.append({
                    'var': self.table.horizontalHeaderItem(col).text(),
                    'mean': data[:, col].mean(),
                    'stdu': np.sqrt(data[:, col].var(ddof=1)/n),
                    'degf': n-1})

        if len(usedcols) > 1:
            for i, col in enumerate(usedcols):
                for col2 in usedcols[i+1:]:
                    stats.append({'corr': (self.table.horizontalHeaderItem(col).text(), self.table.horizontalHeaderItem(col2).text()),
                                  'coef': np.corrcoef(data[:, col], data[:, col2])[0, 1]})
        return stats
