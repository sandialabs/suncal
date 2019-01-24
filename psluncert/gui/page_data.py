''' User Interface for loading data from CSV '''

from io import BytesIO
import numpy as np
from scipy import stats
from dateutil.parser import parse

from PyQt5 import QtWidgets, QtCore
import matplotlib.dates as mdates
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from .. import output
from .. import customdists
from . import configmgr
from . import gui_common
from . import gui_widgets

settings = configmgr.Settings()


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
                with open(fname, 'r') as f:
                    try:
                        rawcsv = f.read()
                    except UnicodeDecodeError:
                        QtWidgets.QMessageBox.warning(self, 'Load CSV', 'Cannot decode file {}'.format(fname))
                        return
                rawcsv = rawcsv.strip(u'\ufeff')  # uFEFF is junk left in Excel-saved CSV files at start of file
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
                with open(fname, 'r') as f:
                    try:
                        rawcsv = f.read()
                    except UnicodeDecodeError:
                        QtWidgets.QMessageBox.warning(self, 'Load CSV', 'Cannot decode file {}'.format(fname))
                        return
                rawcsv = rawcsv.strip(u'\ufeff')  # uFEFF is junk left in Excel-saved CSV files at start of file
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
        self.table = QtWidgets.QTableWidget()   # Only show if CSV mode. Other calculated options just have one col.
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectColumns)
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
        llayout.addWidget(self.table)
        llayout.addStretch()
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
            with open(fname, 'r') as f:
                try:
                    rawcsv = f.read()
                except UnicodeDecodeError:
                    QtWidgets.QMessageBox.warning(self, 'Load CSV', 'Cannot decode file {}'.format(fname))
                    return
            self.load_csv(rawcsv)

        self.cmbDistType.currentIndexChanged.connect(self.updateplot)
        self.chkProbPlot.stateChanged.connect(self.updateplot)
        self.table.currentCellChanged.connect(self.updateplot)
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
            self.table.setVisible(True)
            self.cmbDistType.setVisible(True)

            column = self.table.currentColumn()
            if column >= 0:
                data = self.data[:, column]
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
                    ax.hist(data, density=True, bins='sqrt')
                    ax.plot(xx, yy, label='Distribution Fit')
                    ax.set_ylabel('Probability Density')

            else:
                self.distname = None
                self.params = None
                self.median = None
        else:   # not self.sampleddist
            # Show the GUM distribution
            median = self.tdist['mean']
            std = self.tdist['std']
            degf = self.tdist['df']

            self.lblDist.setVisible(False)
            self.chkProbPlot.setVisible(False)
            self.table.setVisible(False)
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
        MAXROWS = 100  # Maximum number of rows to show in widget (don't want all million samples or loading will take forever)
        self.sampleddist = True
        self.data = np.atleast_2d(data).transpose()
        self.table.clear()
        self.table.setRowCount(min(len(data), MAXROWS))
        self.table.setColumnCount(1)
        for i in range(self.table.rowCount()):
            item = QtWidgets.QTableWidgetItem(str(data[i]))
            item.setFlags(QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEnabled) # No editable
            self.table.setItem(i, 0, item)
        if self.table.rowCount() >= MAXROWS:
            self.table.setItem(MAXROWS-1, 0, QtWidgets.QTableWidgetItem('...'))

        self.table.setCurrentCell(0, 0)
        self.updateplot()

    def load_norm(self, mean, stdev, degf):
        ''' Display a normal or t distribution (usually from a GUM calculation) '''
        self.sampleddist = False
        self.tdist = {'mean': mean, 'std': stdev, 'df': degf}
        self.table.clear()
        self.updateplot()

    def load_csv(self, rawcsv):
        ''' Load sampled data from csv. rawcsv is string of csv contents. '''
        self.raw = rawcsv.strip(u'\ufeff')  # uFEFF is junk left in Excel-saved CSV files at start of file

        if ',' in self.raw:
            self.delim = ','
        elif '\t' in self.raw:
            self.delim = '\t'
        else:
            self.delim = None  # None will group whitespace, ' ' will end up with multiple splits

        self.table.clear()
        lines = self.raw.splitlines()
        self.table.setRowCount(len(lines))
        startrow = 1
        for row, line in enumerate(lines):
            columns = line.split(self.delim)
            self.table.setColumnCount(max(self.table.columnCount(), len(columns)))
            if line == '': continue
            for col, val in enumerate(columns):
                self.table.setItem(row, col, gui_widgets.ReadOnlyTableItem(val.strip()))
                try:
                    float(val.strip())
                except ValueError:
                    startrow = row + 2
        self.table.setHorizontalHeaderLabels(['']*self.table.columnCount())

        try:
            self.data = np.genfromtxt(BytesIO(self.raw.encode('utf-8')), delimiter=self.delim, skip_header=startrow-1)
            if self.data.size == 0: raise ValueError
        except (IndexError, ValueError):
            self.data = np.full((self.table.rowCount(), self.table.columnCount()), np.nan)

        if self.data.ndim == 1:
            self.data = np.atleast_2d(self.data).transpose()

        self.table.setCurrentCell(0, 0)
        self.table.setVisible(True)

        self.table.setHorizontalHeaderLabels(['x', 'y'])  # Guess the columns
        self.updateplot()


class ArraySelectWidget(QtWidgets.QDialog):
    ''' Widget for selecting an array of values from the project or a file '''
    def __init__(self, fname=None, project=None, name_cols=True, parent=None):
        super(ArraySelectWidget, self).__init__(parent=parent)
        self.project = project
        self.name_cols = name_cols
        self.datecols = []  # Columns to be interpreted as dates
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

        if fname:
            self.load_file(fname)

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
        xcol = colnames.index('x') if 'x' in colnames else None

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

        if xcol in self.datecols:
            x = mdates.num2date(x)
            ux = None

        if len(y) > 0:
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
            isdate = idx in self.datecols
        return isdate

    def get_column_data(self, colname):
        ''' Get data for column with name == colname '''
        colnames = self.getvarlist()
        if colname in colnames:
            return self.data[:, colnames.index(colname)]
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

        if arr.xdate:
            self.datecols.append(0)

        self.data = arr.get_numpy()
        self.updateplot()

    def load_file(self, rawcsv):
        ''' Load data in file fname '''
        self.raw = rawcsv.strip(u'\ufeff')  # uFEFF is junk left in Excel-saved CSV files at start of file
        self.datecols = []

        if ',' in self.raw:
            self.delim = ','
        elif '\t' in self.raw:
            self.delim = '\t'
        else:
            self.delim = None  # None will group whitespace, ' ' will end up with multiple splits

        self.table.clear()
        lines = self.raw.splitlines()
        self.table.setRowCount(len(lines))
        self.table.setColumnCount(1)
        startrow = 1
        converters = {}
        for row, line in enumerate(lines):
            columns = line.split(self.delim)
            self.table.setColumnCount(max(self.table.columnCount(), len(columns)))
            if line == '': continue
            colnans = 0
            for col, val in enumerate(columns):
                self.table.setItem(row, col, QtWidgets.QTableWidgetItem(val.strip()))
                self.table.item(row, col).setFlags(QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEnabled)  # No editable
                try:
                    float(val.strip())
                    converters.pop(col, None)
                except ValueError:
                    try:  # Not a float, see if it's a date
                        parse(val.strip())
                        converters[col] = lambda x: float(parse(x).toordinal())
                        self.datecols.append(col)
                    except ValueError:
                        converters.pop(col, None)
                        colnans += 1
                if colnans == len(columns):
                    startrow = row + 2
        if self.name_cols:
            self.table.setHorizontalHeaderLabels(['x', 'y'] + [''] * (self.table.columnCount() - 2))

        try:
            self.data = np.genfromtxt(BytesIO(self.raw.encode('utf-8')), delimiter=self.delim, skip_header=startrow-1, converters=converters, dtype=float)
            if self.data.size == 0: raise ValueError
        except (IndexError, ValueError):
            self.data = np.full((self.table.rowCount(), self.table.columnCount()), np.nan)

        if self.data.ndim == 1:
            self.data = np.atleast_2d(self.data).transpose()
        self.updateplot()


class SweepSelectWidget(ArraySelectWidget):
    ''' Widget for selecting ONE column of sweep data '''
    def __init__(self, fname=None, project=None, parent=None):
        super(SweepSelectWidget, self).__init__(fname=fname, project=project, parent=parent, name_cols=False)
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
            y = self.data[:, ycol]
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
        self.startrow = QtWidgets.QSpinBox()
        self.startrow.setMinimum(1)
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
        hlayout.addWidget(QtWidgets.QLabel('Start Row:'))
        hlayout.addWidget(self.startrow)
        hlayout.addWidget(QtWidgets.QLabel('Variable for Column:'))
        hlayout.addWidget(self.cmbColumn)
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
        layout.addLayout(llayout)
        layout.addLayout(rlayout)
        self.setLayout(layout)

        self.cmbColumn.currentIndexChanged.connect(self.tablecolumnchanged)
        self.table.currentCellChanged.connect(self.columnselected)
        self.startrow.valueChanged.connect(self.updateplot)
        self.bins.valueChanged.connect(self.updateplot)
        self.chkNorm.stateChanged.connect(self.updateplot)
        self.chkProbPlot.stateChanged.connect(self.updateplot)

        if fname is None:
            fname, _ = QtWidgets.QFileDialog.getOpenFileName(caption='Select CSV file to load')
        self.load_file(fname)

    def load_file(self, fname):
        ''' Load data in file fname '''
        with open(fname, 'r') as f:
            try:
                self.raw = f.read()
            except UnicodeDecodeError:
                QtWidgets.QMessageBox.warning(self, 'Load CSV', 'Cannot decode file {}'.format(fname))
                return

        self.raw = self.raw.strip(u'\ufeff')  # uFEFF is junk left in Excel-saved CSV files at start of file
        if ',' in self.raw:
            self.delim = ','
        elif '\t' in self.raw:
            self.delim = '\t'
        else:
            self.delim = None  # None will group whitespace, ' ' will end up with multiple splits

        self.table.clear()
        lines = self.raw.splitlines()
        self.table.setRowCount(len(lines))
        self.startrow.setMaximum(len(lines))
        startrow = 1
        for row, line in enumerate(lines):
            columns = line.split(self.delim)
            self.table.setColumnCount(max(self.table.columnCount(), len(columns)))
            if line == '': continue
            for col, val in enumerate(columns):
                self.table.setItem(row, col, QtWidgets.QTableWidgetItem(val.strip()))
                self.table.item(row, col).setFlags(QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEnabled)  # No editable
                try:
                    float(val.strip())
                except ValueError:
                    startrow = row + 2
        self.startrow.setValue(startrow)
        self.table.setHorizontalHeaderLabels(['']*self.table.columnCount())
        self.bins.setValue(min(max(7, int(np.sqrt(len(lines))*3)), 200))

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
        for row in range(self.table.rowCount()):
            for col in range(self.table.columnCount()):
                if row < self.startrow.value()-1:
                    self.table.item(row, col).setBackground(gui_common.COLOR_UNUSED)
                else:
                    self.table.item(row, col).setBackground(gui_common.COLOR_OK)

        column = self.table.currentColumn()
        try:
            self.getdata()
            coldata = self.data[:, column]
        except (ValueError, IndexError):
            self.data = None
            self.lblmean.setText('Mean: --')
            self.lblstdunc.setText('Standard Deviation of Mean: --')
            self.lbldegf.setText('Degrees of Freedom: --')
            self.fig.clf()
        else:
            n = len(coldata)
            mean = coldata.mean()
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
        self.data = np.genfromtxt(BytesIO(self.raw.encode('utf-8')), delimiter=self.delim, skip_header=self.startrow.value()-1)
        if self.data.ndim == 1:
            self.data = np.atleast_2d(self.data).transpose()
        return self.data

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
            for col in usedcols:
                for col2 in usedcols[col+1:]:
                    stats.append({'corr': (self.table.horizontalHeaderItem(col).text(), self.table.horizontalHeaderItem(col2).text()),
                                  'coef': np.corrcoef(data[:, col], data[:, col2])[0, 1]})
        return stats
