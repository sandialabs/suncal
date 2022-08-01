''' User Interface for importing data arrays or distributions from
    a CSV or from another calculation in the project.
'''

from contextlib import suppress
import numpy as np

from PyQt5 import QtWidgets, QtCore
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from .. import plotting
from .. import distributions
from .. import dataset
from . import page_csvload
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
    ROLE_DDATA = QtCore.Qt.UserRole
    loaddata = QtCore.pyqtSignal(object)  # Emit the data array or dictionary

    def __init__(self, project=None, parent=None):
        super().__init__(parent=parent)
        self.project = project
        self.setMinimumWidth(250)
        self.setIconSize(QtCore.QSize(32, 32))
        self.setItemDelegate(ProjectTreeDelegate())
        self.setColumnCount(1)
        self.setHeaderHidden(True)
        self.csvitems = []  # List of DataSet items loaded from CSV but not part of project
        if self.project:
            names = self.project.get_names()
            for i in range(self.project.count()):
                mode = self.project.get_mode(i)
                if mode in ['uncertainty', 'reverse', 'curvefit', 'data']:
                    with suppress(AttributeError):
                        distdict = self.project.items[i].get_output().get_dists()  # Pass no args to get names
                        item = QtWidgets.QTreeWidgetItem([names[i]])
                        item.setIcon(0, gui_common.load_icon(gui_common.iconname[mode]))
                        self.addTopLevelItem(item)
                        item.setData(0, self.ROLE_DDATA, distdict)

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
        if item.text(0) in ['Select CSV File...', 'Clipboard']:
            if item.text(0) == 'Select CSV File...':
                fname, _ = QtWidgets.QFileDialog.getOpenFileName(caption='CSV file to load')
            else:
                fname = '_clipboard_'

            if fname:
                dlg = page_csvload.SelectCSVData(fname)
                if dlg.exec_():
                    self.csvitems.append(dlg.dataset())
                    # DataSets loaded from CSV in this dialog
                    datadict = self.csvitems[-1].get_output().get_dists()
                    item = QtWidgets.QTreeWidgetItem([f'CSV Data {len(self.csvitems)+1}'])
                    item.setIcon(0, gui_common.load_icon(gui_common.iconname['data']))
                    item.setData(0, self.ROLE_DDATA, datadict)
                    self.addTopLevelItem(item)
                    self.loaddata.emit(datadict)

        else:
            datadict = item.data(0, self.ROLE_DDATA)
            self.loaddata.emit(datadict)


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
        super().__init__(parent=parent)
        self.project = project
        self.csvitems = []
        self.setMinimumWidth(250)
        self.setIconSize(QtCore.QSize(32, 32))
        self.setItemDelegate(ProjectTreeDelegate())
        self.setColumnCount(1)
        self.setHeaderHidden(True)

        if self.project:
            names = self.project.get_names()
            for i in range(self.project.count()):
                mode = self.project.get_mode(i)
                if mode in ['sweep', 'reversesweep', 'data']:
                    with suppress(AttributeError):
                        dsets = self.project.items[i].get_output().get_dataset()  # No args to get names
                        item = QtWidgets.QTreeWidgetItem([names[i]])
                        item.setIcon(0, gui_common.load_icon(gui_common.iconname[mode]))
                        self.addTopLevelItem(item)

                        for dset in dsets:
                            dsitem = QtWidgets.QTreeWidgetItem([dset])
                            dsitem.setIcon(0, gui_common.load_icon('column'))
                            dsitem.setData(0, self.ROLE_CALC, self.project.items[i])
                            dsitem.setData(0, self.ROLE_NAME, dset)
                            item.addChild(dsitem)
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
        if item.text(0) in ['Select CSV File...', 'Clipboard']:
            if item.text(0) == 'Select CSV File...':
                fname, _ = QtWidgets.QFileDialog.getOpenFileName(caption='CSV file to load')
            else:
                fname = '_clipboard_'

            if fname:
                dlg = page_csvload.SelectCSVData(fname)
                if dlg.exec_():
                    self.csvitems.append(dlg.dataset())
                    # DataSets loaded from CSV in this dialog
                    dists = self.csvitems[-1].get_output().get_dataset()
                    item = QtWidgets.QTreeWidgetItem([f'CSV Data {len(self.csvitems)+1}'])
                    item.setIcon(0, gui_common.load_icon(gui_common.iconname['data']))
                    self.addTopLevelItem(item)
                    for dist in dists:
                        distitem = QtWidgets.QTreeWidgetItem([dist])
                        distitem.setIcon(0, gui_common.load_icon('column'))
                        distitem.setData(0, self.ROLE_CALC, self.csvitems[-1])
                        distitem.setData(0, self.ROLE_NAME, dist)
                        item.addChild(distitem)
                        item.setExpanded(True)
                    self.setCurrentItem(item)
                    self.loaddata.emit(self.csvitems[-1])

        else:
            calc = item.data(0, self.ROLE_CALC)
            with suppress(AttributeError):  # Could be top-level item
                data = calc.get_output().get_dataset(item.data(0, self.ROLE_NAME))
                self.loaddata.emit(data)


class DistributionSelectWidget(QtWidgets.QDialog):
    ''' Widget for selecting a distribution from the project or a file '''
    ROW_NAME = 0
    ROW_MEAN = 1
    ROW_STD = 2
    ROW_SEM = 3
    ROW_DF = 4
    ROW_DIST = 5
    NROWS = 6

    def __init__(self, project=None, singlecol=True, coloptions=None, enablecorr=True, parent=None):
        super().__init__(parent=parent)
        font = self.font()
        font.setPointSize(10)
        self.setFont(font)
        self.project = project
        self.singlecol = singlecol
        gui_widgets.centerWindow(self, 1000, 800)
        self.setWindowTitle('Select Distribution')
        self.coloptions = coloptions
        self.xval_isdate = False

        self.dataset = None  # Either DataSet object or a dictionary defining mean/std/df for each column?

        dists = settings.getDistributions()
        dists = ['normal (standard error)', 'normal (with autocorrelation)'] + [d for d in dists if distributions.fittable(d)]
        self.treeSource = ProjectTreeDists(project=project)
        self.fig = Figure()
        self.canvas = FigureCanvas(self.fig)
        self.cmbDistType = QtWidgets.QComboBox()
        self.cmbDistType.addItems(dists + ['histogram'])
        self.lblCol = QtWidgets.QLabel('Assigned Variable:')
        self.cmbColumn = QtWidgets.QComboBox()
        self.chkCorr = QtWidgets.QCheckBox('Import Correlations')
        self.chkCorr.setChecked(True)
        self.chkProbPlot = QtWidgets.QCheckBox('Probability Plot')
        self.lblDist = QtWidgets.QLabel('Distribution to Fit:')
        self.xval = QtWidgets.QDoubleSpinBox()
        self.xval.setValue(0)
        self.xval.setDecimals(4)
        self.xval.setRange(-1E99, 1E99)
        self.xdate = QtWidgets.QDateEdit(QtCore.QDate.currentDate())
        self.xdate.setDisplayFormat('yyyy-MM-dd')
        self.lblX = QtWidgets.QLabel('X Value')
        self.table = QtWidgets.QTableWidget()
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectColumns)
        self.lblAssign = QtWidgets.QLabel('<font color="red">Please assign at least one variable before continuing.</font>')
        self.dlgbutton = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)

        if coloptions is not None:
            self.cmbColumn.addItems(['Not Used'] + coloptions)

        if not enablecorr:
            self.chkCorr.setChecked(False)
            self.chkCorr.setVisible(False)

        if self.singlecol:
            self.cmbColumn.setVisible(False)
            self.chkCorr.setVisible(False)
            self.lblCol.setVisible(False)
            self.lblAssign.setVisible(False)

        llayout = QtWidgets.QVBoxLayout()
        llayout.addWidget(QtWidgets.QLabel('Data Source:'))
        llayout.addWidget(self.treeSource)
        hlayout = QtWidgets.QHBoxLayout()
        hlayout.addWidget(self.lblCol)
        hlayout.addWidget(self.cmbColumn)
        hlayout.addStretch
        hlayout.addWidget(self.lblDist)
        hlayout.addWidget(self.cmbDistType)
        hlayout.addWidget(self.lblX)
        hlayout.addWidget(self.xval)
        hlayout.addWidget(self.xdate)
        hlayout.addWidget(self.chkProbPlot)
        h2layout = QtWidgets.QHBoxLayout()
        h2layout.addWidget(self.chkCorr)
        rlayout = QtWidgets.QVBoxLayout()
        rlayout.addWidget(self.canvas, stretch=5)
        rlayout.addLayout(hlayout)
        rlayout.addWidget(self.table, stretch=5)
        rlayout.addLayout(h2layout)
        blayout = QtWidgets.QHBoxLayout()
        blayout.addWidget(self.lblAssign)
        blayout.addWidget(self.dlgbutton)
        rlayout.addLayout(blayout)
        layout = QtWidgets.QHBoxLayout()
        layout.addLayout(llayout, stretch=2)
        layout.addLayout(rlayout, stretch=10)
        self.setLayout(layout)

        self.cmbDistType.currentIndexChanged.connect(self.disttypechanged)
        self.chkProbPlot.stateChanged.connect(self.updateplot)
        self.cmbColumn.currentIndexChanged.connect(self.assignmentchanged)
        self.table.currentCellChanged.connect(self.columnselected)
        self.xval.valueChanged.connect(self.xvalchanged)
        self.xdate.dateChanged.connect(self.xvalchanged)
        self.treeSource.loaddata.connect(self.load_data)
        self.dlgbutton.rejected.connect(self.reject)
        self.dlgbutton.accepted.connect(self.accept)

        self.xval.setVisible(False)
        self.xdate.setVisible(False)
        self.lblX.setVisible(False)
        self.dlgbutton.button(QtWidgets.QDialogButtonBox.Ok).setEnabled(False)
        self.cmbDistType.setEnabled(False)

    def get_selcolname(self):
        ''' Get selected column name '''
        col = self.table.currentColumn()
        if col is not None:
            col = self.colnames[col]
        return col

    def get_seldata(self):
        ''' Get data (array or dict) for selected column '''
        data = None
        colname = self.get_selcolname()
        if colname is not None:
            data = self.datadict[colname]
        return data

    def updateplot(self):
        ''' Update the distribution and plot '''
        self.fig.clf()
        ax = self.fig.add_subplot(1, 1, 1)
        data = self.get_seldata()
        colname = self.get_selcolname()
        if data is not None:
            if 'samples' in data:
                samples = data['samples']
                distname = self.cmbDistType.currentText()
                if distname == 'normal (standard error)':
                    params = {'median': np.median(data['samples']), 'std': data['samples'].std(ddof=1)/np.sqrt(len(data['samples']))}
                    fitdist = distributions.get_distribution('normal', **params)

                elif distname == 'normal (with autocorrelation)':
                    params = {'median': np.median(data['samples']), 'std': dataset.uncert_autocorrelated(data['samples']).uncert}
                    fitdist = distributions.get_distribution('normal', **params)

                else:
                    fitdist = distributions.get_distribution(distname)
                    params = fitdist.fit(samples)
                if self.chkProbPlot.isChecked() and distname != 'histogram':
                    plotting.probplot(samples, ax=ax, sparams=params, dist=fitdist.dist.name)
                else:
                    xx = np.linspace(samples.min(), samples.max(), num=200)
                    yy = fitdist.pdf(xx)
                    with suppress(ValueError):
                        ax.hist(samples, density=True, bins='sqrt')
                    ax.plot(xx, yy, label='Distribution Fit')
                    ax.set_ylabel('Probability Density')

            else:
                if 'function' in data:
                    if self.xval_isdate:
                        x = self.xdate.date().toPyDate().toordinal()
                    else:
                        x = self.xval.value()
                    data = data['function'](x)
                median = data.get('mean', 0)
                std = data.get('std', 1)
                df = min(data.get('df', 100), 1E4)  # Inf degf doesn't work
                xx = np.linspace(median-5*std, median+5*std, num=200)
                dist = distributions.get_distribution('t', median=median, unc=std, df=df)
                yy = dist.pdf(xx)
                ax.plot(xx, yy)
                ax.set_xlabel(colname)
                ax.set_ylabel('Probability Density')

        self.canvas.draw_idle()

    def _get_colassignment(self):
        ''' Get list of variable names for each column '''
        return [self.table.horizontalHeaderItem(i).text() if self.table.horizontalHeaderItem(i) else '' for i in range(self.table.columnCount())]

    def disttypechanged(self):
        ''' Distribution assignment has changed '''
        distname = self.cmbDistType.currentText()
        col = self.table.currentColumn()
        col = max(col, 0)
        self.table.setItem(self.ROW_DIST, col, gui_widgets.ReadOnlyTableItem(distname))
        self.updateplot()

    def assignmentchanged(self):
        ''' Column assignment has changed '''
        varname = self.cmbColumn.currentText()
        varlist = self._get_colassignment()
        if varname in ['Not Used', '']:
            self.table.setHorizontalHeaderItem(self.table.currentColumn(), QtWidgets.QTableWidgetItem(''))
        else:
            if varname in varlist:
                self.table.setHorizontalHeaderItem(varlist.index(varname), QtWidgets.QTableWidgetItem(''))
            self.table.setHorizontalHeaderItem(self.table.currentColumn(), QtWidgets.QTableWidgetItem(varname))
        self.updateplot()

        if self.singlecol and len(self._get_colassignment()) > 0:
            self.dlgbutton.button(QtWidgets.QDialogButtonBox.Ok).setEnabled(True)
        elif all([col == 'Not Used' or col == '' for col in self._get_colassignment()]):
            self.lblAssign.setVisible(True)
            self.dlgbutton.button(QtWidgets.QDialogButtonBox.Ok).setEnabled(False)
        else:
            self.lblAssign.setVisible(False)
            self.dlgbutton.button(QtWidgets.QDialogButtonBox.Ok).setEnabled(True)

        for cidx, colname in enumerate(self._get_colassignment()):
            if colname != '' and colname != 'Not Used':
                colcolor = gui_common.COLOR_SELECTED
            else:
                colcolor = gui_common.COLOR_OK
            for row in range(self.table.rowCount()):
                self.table.item(row, cidx).setBackground(colcolor)

    def columnselected(self, row, col, prow, pcol):
        ''' Column was highlighted in table. Update combobox with variable selection. '''
        self.cmbDistType.blockSignals(True)
        self.cmbColumn.blockSignals(True)
        try:
            colvar = self.table.horizontalHeaderItem(col).text()
        except AttributeError:
            colvar = None

        self.cmbColumn.setCurrentIndex(self.cmbColumn.findText(colvar))

        disttype = ''
        item = self.table.item(self.ROW_DIST, col)
        if item:
            disttype = item.text()

        if disttype != '':
            self.cmbDistType.setCurrentIndex(self.cmbDistType.findText(disttype))
            self.cmbDistType.setEnabled(True)
            self.chkProbPlot.setEnabled(True)
        else:
            self.cmbDistType.setEnabled(False)
            self.chkProbPlot.setEnabled(False)

        if ('Confidence' in str(self.colnames[col]) or 'Prediction' in str(self.colnames[col])):
            self.xval.setVisible(not self.xval_isdate)
            self.xdate.setVisible(self.xval_isdate)
            self.lblX.setVisible(True)
        else:
            self.xval.setVisible(False)
            self.xdate.setVisible(False)
            self.lblX.setVisible(False)

        if self.cmbColumn.currentIndex() < 0:
            self.cmbColumn.setCurrentIndex(0)  # Not used
        self.cmbColumn.blockSignals(False)
        self.cmbDistType.blockSignals(False)
        self.updateplot()

    def xvalchanged(self):
        ''' X-value changed. Update confidence and prediction band columns '''
        if self.xval_isdate:
            x = self.xdate.date().toPyDate().toordinal()
        else:
            x = self.xval.value()

        for cidx, colname in enumerate(self.colnames):
            if 'Confidence' in colname or 'Prediction' in colname:
                params = self.datadict[colname]['function'](x)
                mean = params.get('mean')
                std = params.get('std')
                df = params.get('df')
                sem = std / np.sqrt(df+1)
                self.table.setItem(self.ROW_MEAN, cidx, gui_widgets.ReadOnlyTableItem(format(mean, '.4g')))
                self.table.setItem(self.ROW_STD, cidx, gui_widgets.ReadOnlyTableItem(format(std, '.4g')))
                self.table.setItem(self.ROW_SEM, cidx, gui_widgets.ReadOnlyTableItem(format(sem, '.4g')))
                self.table.setItem(self.ROW_DF, cidx, gui_widgets.ReadOnlyTableItem(format(df, '.0f')))
        self.updateplot()

    def get_dist(self):
        ''' Get dictionary of distributions assigned

        '''
        distlist = []
        if self.singlecol:
            colname = self.get_selcolname()
            colidx = self.colnames.index(colname)
            newname = colname
            distlist.append((newname, colidx, self.datadict[colname]))

        else:
            assignedcols = self._get_colassignment()
            for colidx, newname in enumerate(assignedcols):
                if newname != '':
                    colname = self.colnames[colidx]
                    distlist.append((newname, colidx, self.datadict[colname]))

        dists = {}
        for newname, colidx, data in distlist:
            distname = self.table.item(self.ROW_DIST, colidx).text()
            if 'samples' in data:
                if distname == 'normal (standard error)':
                    distname = 'normal'
                    params = {'median': np.mean(data['samples']), 'std': data['samples'].std(ddof=1)/np.sqrt(len(data['samples']))}

                elif distname == 'normal (with autocorrelation)':
                    params = {'median': np.mean(data['samples']), 'std': dataset.uncert_autocorrelated(data['samples']).uncert}
                    distname = 'normal'

                else:
                    fitdist = distributions.get_distribution(distname)
                    params = fitdist.fit(data['samples'])
                    if 'expected' in data:
                        params['expected'] = data['expected']
                    if 'median' in data:
                        params.setdefault('median', data['median'])

                params['dist'] = distname
                if 'df' not in params:
                    params['df'] = len(data['samples'])-1
            elif 'function' in data:
                f = data['function']
                if self.xval_isdate:
                    x = self.xdate.date().toPyDate().toordinal()
                else:
                    x = self.xval.value()
                params = f(x)
            elif 'sem' in data and distname == 'normal (standard error)':
                params = {'dist': 'normal',
                          'median': data.get('median', data.get('mean', 0)),
                          'std': data.get('sem'),
                          'df': data.get('df', np.inf)}
            else:
                params = data
                params.setdefault('dist', 'normal')
            dists[newname] = params

        # Add correlation coefficients between
        if not self.singlecol and self.chkCorr.isChecked():
            corrdict = {}
            for i in range(len(distlist)):
                for j in range(i+1, len(distlist)):
                    name1, name2 = distlist[i][0], distlist[j][0]
                    data1, data2 = distlist[i][2].get('samples'), distlist[j][2].get('samples')
                    with suppress(TypeError, ValueError):  # Could have one of the datas not be sampled values
                        corrdict[(name1, name2)] = np.nan_to_num(np.corrcoef(data1, data2)[0, 1])
            dists['_correlation_'] = corrdict

        if self.singlecol:
            dists = dists[colname]
        return dists

    def load_data(self, datadict):
        ''' Load distribution options into table '''
        # datadict: dict of distributions
        # each item is either array of samples or dict of mean/std/df
        self.datadict = datadict
        self.colnames = list(datadict.keys())
        self.table.clear()
        self.table.setColumnCount(len(datadict))
        self.table.setRowCount(self.NROWS)
        hdr = []

        for col, (name, data) in enumerate(datadict.items()):
            if 'function' in data:
                # Params must be generated as function of x
                f = data['function']
                xdate = data['xdates']
                if not xdate:
                    x = self.xval.value()
                else:
                    x = self.xdate.date().toPyDate().toordinal()
                    self.xval_isdate = True
                params = f(x)
                mean = params.get('mean', 0)
                std = params.get('std', 1)
                df = params.get('df', 100)
                sem = std / np.sqrt(df + 1)
                dist = ''  # Can't change this dist
            elif 'samples' in data:
                mean = data['samples'].mean()
                std = data['samples'].std(ddof=1)
                df = len(data['samples']) - 1
                sem = std / np.sqrt(df + 1)
                dist = 'normal (standard error)'
            elif 'sem' in data:
                mean = data.get('mean', data.get('median', 0))
                std = data.get('std', 1)
                sem = data.get('sem')
                df = data.get('df', 100)
                dist = 'normal (standard error)'
            else:
                mean = data.get('mean', 0)
                std = data.get('std', 1)
                df = data.get('df', 100)
                sem = std / np.sqrt(df + 1)
                dist = ''  # Can't change this dist

            self.table.setItem(self.ROW_NAME, col, gui_widgets.ReadOnlyTableItem(name))
            self.table.setItem(self.ROW_MEAN, col, gui_widgets.ReadOnlyTableItem(format(mean, '.2g')))
            self.table.setItem(self.ROW_STD, col, gui_widgets.ReadOnlyTableItem(format(std, '.2g')))
            self.table.setItem(self.ROW_SEM, col, gui_widgets.ReadOnlyTableItem(format(sem, '.2g')))
            self.table.setItem(self.ROW_DF, col, gui_widgets.ReadOnlyTableItem(format(df, '.0f')))
            self.table.setItem(self.ROW_DIST, col, gui_widgets.ReadOnlyTableItem(dist))
            if self.coloptions is not None and name in self.coloptions:
                hdr.append(name)
            else:
                hdr.append('')

        self.table.setHorizontalHeaderLabels(hdr)
        self.table.setVerticalHeaderLabels(['Name', 'Mean', 'Std. Dev.', 'SEM', 'Deg. Freedom', 'Distribution'])
        self.table.resizeColumnsToContents()
        self.assignmentchanged()
        self.updateplot()


class ArraySelectWidget(QtWidgets.QDialog):
    ''' Widget for selecting an array of values from the project or a file '''
    def __init__(self, project=None, singlecol=False, colnames=None, parent=None):
        super().__init__(parent=parent)
        font = self.font()
        font.setPointSize(10)
        self.setFont(font)
        self.project = project
        self.singlecol = singlecol
        gui_widgets.centerWindow(self, 1000, 800)
        self.setWindowTitle('Select Array')
        self.colnames = colnames  # Column names that can be assigned
        if self.colnames is None:
            self.colnames = ['x', 'y', 'u(y)', 'u(x)']
        self.colassignments = None

        self.dataset = None
        self.treeSource = ProjectTreeArrays(project=project)
        self.table = QtWidgets.QTableWidget()
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectColumns)
        self.cmbColumn = QtWidgets.QComboBox()
        self.cmbColumn.addItems(['Not Used'] + self.colnames)

        self.fig = Figure()
        self.canvas = FigureCanvas(self.fig)
        self.dlgbutton = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        self.dlgbutton.rejected.connect(self.reject)
        self.dlgbutton.accepted.connect(self.accept)
        self.lblColumn = QtWidgets.QLabel('Assigned Column:')

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
        layout.addLayout(llayout, stretch=2)
        layout.addLayout(rlayout, stretch=4)
        self.setLayout(layout)

        self.cmbColumn.setVisible(not self.singlecol)
        self.lblColumn.setVisible(not self.singlecol)

        self.updateplot()
        self.cmbColumn.currentIndexChanged.connect(self.tablecolumnchanged)
        self.table.currentCellChanged.connect(self.columnselected)
        self.treeSource.loaddata.connect(self.load_data)

    def tablecolumnchanged(self):
        ''' Column assignment has changed '''
        varname = self.cmbColumn.currentText()
        colid = self.table.currentColumn()
        if varname in ['Not Used', '']:
            self.colassignments[colid] = None
        else:
            if varname in self.colassignments:
                self.colassignments[self.colassignments.index(varname)] = None
            self.colassignments[colid] = varname
        self.highlight_columns()
        self.updateplot()

    def highlight_columns(self):
        ''' Highlight columns that will be imported '''
        headerlabels = []
        for cidx in range(self.table.columnCount()):
            if self.colassignments[cidx] and self.dataset.colnames:
                headerlabels.append('{} → {}'.format(self.dataset.colnames[cidx], self.colassignments[cidx]))
            elif self.dataset.colnames:
                headerlabels.append(self.dataset.colnames[cidx])
            else:
                headerlabels.append('')
            
            if self.colassignments[cidx]:
                colcolor = gui_common.COLOR_SELECTED
            else:
                colcolor = gui_common.COLOR_OK
            for row in range(self.table.rowCount()):
                self.table.item(row, cidx).setBackground(colcolor)
        self.table.setHorizontalHeaderLabels(headerlabels)

    def columnselected(self, row, col, prow, pcol):
        ''' Column was highlighted in table. Update combobox with variable selection. '''
        self.cmbColumn.blockSignals(True)
        name = self.colassignments[self.table.currentColumn()]
        self.cmbColumn.setCurrentIndex(self.cmbColumn.findText(name))
        self.cmbColumn.blockSignals(False)
        self.updateplot()

    def updateplot(self):
        ''' Update the plot '''
        self.fig.clf()
        ax = self.fig.add_subplot(1, 1, 1)

        arr = self.get_array()
        x, y = arr.get('x'), arr.get('y')
        ux, uy = arr.get('u(x)'), arr.get('u(y)')

        if y is not None:
            if x is None:
                x = np.arange(len(y), dtype=float)

            # Can't plot datetime as errorbar y value
            if len(x) > 0 and hasattr(x[0], 'date'):
                ux = None
            if len(y) > 0 and hasattr(y[0], 'date'):
                uy = None
            if ux is not None and hasattr(ux[0], 'date'):
                ux = None
            if uy is not None and hasattr(uy[0], 'date'):
                uy = None

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

    def is_datecol(self, colname):
        ''' Return True if column with name === colname is date '''
        isdate = False
        if colname in self.colassignments:
            idx = self.colassignmetns.index(colname)
            isdate = hasattr(self.dataset.data[idx][0], 'date')
        return isdate

    def get_array(self):
        ''' Return dictionary with keys for each column name '''
        if self.dataset is None:
            ret = {}
        elif self.singlecol:
            name = self.dataset.colnames[self.table.currentColumn()]
            y = self.dataset.get_column(name)
            ret = {'y': y}
        else:
            ret = {}
            for cidx, c in enumerate(self.colassignments):
                if c is not None:
                    name = self.dataset.colnames[cidx]
                    ret[c] = self.dataset.get_column(name)
        return ret

    def load_data(self, dset):
        ''' Fill table with data array

            Parameters
            ----------
            dset: DataSet object
                Data table to display
        '''
        self.table.clear()
        nrows = dset.maxrows()
        ncols = dset.ncolumns()
        self.table.setRowCount(nrows)
        self.table.setColumnCount(ncols)

        for col in range(ncols):
            for row in range(nrows):
                item = QtWidgets.QTableWidgetItem(str(dset.data[col][row]))
                item.setFlags(QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEnabled)
                self.table.setItem(row, col, item)

        self.colassignments = [None] * dset.ncolumns()
        if not self.singlecol:
            for cname in self.colnames:
                if cname in dset.colnames:
                    idx = dset.colnames.index(cname)
                    self.colassignments[idx] = dset.colnames[idx]

            if all(c is None for c in self.colassignments):
                if 'x' in self.colnames and 'y' in self.colnames and len(self.colassignments) == 2:
                    self.colassignments = ['x', 'y']
                elif ('x' in self.colnames and 'y' in self.colnames and 
                      'u(y)' in self.colnames and len(self.colassignments) == 3):
                    self.colassignments = ['x', 'y', 'u(y)']

        self.dataset = dset
        self.highlight_columns()
        self.table.resizeColumnsToContents()
        self.updateplot()
