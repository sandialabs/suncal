''' User Interface for importing data arrays or distributions from
    another calculation in the project.
'''
from contextlib import suppress
import numpy as np

from PyQt6 import QtWidgets, QtCore
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from ..common import plotting, distributions, unitmgr, report
from ..datasets import dataset
from . import gui_common
from . import icons
from . import widgets
from .gui_settings import gui_settings


# Icon names for each calculation method
iconname = {'uncertainty': 'target',
            'curvefit': 'curvefit',
            'risk': 'risk',
            'sweep': 'targetlist',
            'reverse': 'calipers',
            'reversesweep': 'rulersweep',
            'data': 'boxplot',
            'wizard': 'wizard'}


class ProjectTreeDelegate(QtWidgets.QItemDelegate):
    ''' Delegate for project tree items '''
    def sizeHint(self, option, index):
        return QtCore.QSize(32, 32)


class ProjectTreeDists(QtWidgets.QTreeWidget):
    ''' Tree Widget showing all project items with usable distributions '''
    ROLE_DDATA = QtCore.Qt.ItemDataRole.UserRole
    loaddata = QtCore.pyqtSignal(object)

    def __init__(self, project, parent=None):
        super().__init__(parent=parent)
        self.project = project
        self.setMinimumWidth(250)
        self.setIconSize(QtCore.QSize(32, 32))
        self.setItemDelegate(ProjectTreeDelegate())
        self.setColumnCount(1)
        self.setHeaderHidden(True)
        names = self.project.get_names()
        for i in range(self.project.count()):
            mode = self.project.get_mode(i)
            if mode in ['uncertainty', 'wizard', 'reverse', 'curvefit', 'data']:
                with suppress(AttributeError):  # AttributeError if calculation has not run
                    distdict = self.project.items[i].get_dists()
                    if len(distdict) > 0:
                        item = QtWidgets.QTreeWidgetItem([names[i]])
                        item.setIcon(0, icons.icon(iconname[mode]))
                        self.addTopLevelItem(item)

                    for name, distdata in distdict.items():
                        distitem = QtWidgets.QTreeWidgetItem([name])
                        distitem.setIcon(0, icons.icon(iconname[mode]))
                        distitem.setData(0, self.ROLE_DDATA, (mode, distdata))
                        item.addChild(distitem)
                    item.setExpanded(True)
        self.itemClicked.connect(self.treeclick)

    def treeclick(self, item, column):
        ''' The tree was clicked. '''
        data = item.data(0, self.ROLE_DDATA)
        if data is not None:
            self.loaddata.emit(data)


class ProjectTreeArrays(QtWidgets.QTreeWidget):
    ''' Widget for showing project tree with all calculated arrays to choose from.

        Arrays may come from sweeps, reverse-sweeps, or anova grouped results
    '''
    ROLE_ARRAY = QtCore.Qt.ItemDataRole.UserRole  # Dictionary of x, y, uy
    ROLE_NAME = QtCore.Qt.ItemDataRole.UserRole + 1
    loaddata = QtCore.pyqtSignal(object)  # Emit the actual data array

    def __init__(self, project, parent=None):
        super().__init__(parent=parent)
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
                if mode in ['sweep', 'reversesweep', 'data']:
                    with suppress(AttributeError):
                        arrays = self.project.items[i].get_arrays()
                        item = QtWidgets.QTreeWidgetItem([names[i]])
                        item.setIcon(0, icons.icon(iconname[mode]))
                        self.addTopLevelItem(item)

                        for name, arr in arrays.items():
                            dsitem = QtWidgets.QTreeWidgetItem([name])
                            dsitem.setIcon(0, icons.icon('column'))
                            dsitem.setData(0, self.ROLE_ARRAY, arr)
                            dsitem.setData(0, self.ROLE_NAME, name)
                            item.addChild(dsitem)
                        item.setExpanded(True)
        self.itemDoubleClicked.connect(self.treeclick)

    def treeclick(self, item, column):
        ''' The tree was clicked. '''
        arr = item.data(0, self.ROLE_ARRAY)
        if arr is not None:
            self.loaddata.emit(arr)


class SampledDataOptions(QtWidgets.QWidget):
    ''' Options for importing from a sampled (Monte Carlo) result '''
    def __init__(self):
        super().__init__()
        self.cmbDistType = QtWidgets.QComboBox()
        self.chkprobplot = QtWidgets.QCheckBox('Show Probability Plot')
        layout = QtWidgets.QHBoxLayout()
        layout.addWidget(QtWidgets.QLabel('Distribution to fit'))
        layout.addWidget(self.cmbDistType)
        layout.addStretch()
        layout.addWidget(self.chkprobplot)
        self.setLayout(layout)
        self.show_norm_options()

    def show_norm_options(self, show: bool = False):
        ''' Enable options for standard error and autocorrelation '''
        dists = gui_settings.distributions
        dists = [d for d in dists if distributions.fittable(d)]
        if show:
            dists = ['normal (standard error)',
                     'normal (with autocorrelation)'] + dists
        with gui_common.BlockedSignals(self.cmbDistType):
            self.cmbDistType.clear()
            self.cmbDistType.addItems(dists + ['histogram'])


class DistributionSelectWidget(QtWidgets.QDialog):
    ''' Dialog for selecting a calculated distribution from the project '''
    def __init__(self, project=None, parent=None):
        super().__init__(parent=parent)
        font = self.font()
        font.setPointSize(10)
        self.setFont(font)
        gui_common.centerWindow(self, 1000, 800)
        self.setWindowTitle('Select Distribution')

        self.mode = None  # Mode of distribution to import
        self.distdata = None  # Distribution dictionary selected
        self.xval_isdate = False

        self.tree = ProjectTreeDists(project)
        self.sampled = SampledDataOptions()
        self.fig = Figure()
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setStyleSheet("background-color:transparent;")
        self.stats = widgets.MarkdownTextEdit()
        self.buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Ok |
                                                  QtWidgets.QDialogButtonBox.StandardButton.Cancel)

        llayout = QtWidgets.QVBoxLayout()
        llayout.addWidget(QtWidgets.QLabel('Distribution source:'))
        llayout.addWidget(self.tree)

        rlayout = QtWidgets.QVBoxLayout()
        rlayout.addWidget(self.sampled)
        rlayout.addWidget(self.canvas)
        rlayout.addWidget(self.stats)
        layout = QtWidgets.QHBoxLayout()
        layout.addLayout(llayout)
        layout.addLayout(rlayout)
        tlayout = QtWidgets.QVBoxLayout()
        tlayout.addLayout(layout)
        tlayout.addWidget(self.buttons)
        self.setLayout(tlayout)

        self.sampled.setVisible(False)
        self.tree.loaddata.connect(self.set_distsource)
        self.sampled.cmbDistType.currentIndexChanged.connect(self.replot)
        self.sampled.chkprobplot.stateChanged.connect(self.replot)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        self.buttons.button(QtWidgets.QDialogButtonBox.StandardButton.Ok).setEnabled(False)

    def set_distsource(self, source):
        ''' Distribution source was clicked in tree. Update plot/stats '''
        self.mode, self.distdata = source  # Component type, distribution dictionary
        if self.mode == 'data' and 'samples' in self.distdata:
            # Column of data from a dataset
            self.sampled.setVisible(True)
            self.sampled.show_norm_options(True)
        elif 'samples' in self.distdata:
            # Monte Carlo uncertainty
            self.sampled.setVisible(True)
            self.sampled.show_norm_options(False)
        else:
            # GUM or full dataset
            self.sampled.setVisible(False)
        self.buttons.button(QtWidgets.QDialogButtonBox.StandardButton.Ok).setEnabled(True)
        self.replot()

    def replot(self):
        ''' Replot the data to import '''
        if 'samples' in self.distdata:
            if self.sampled.chkprobplot.isChecked():
                self.plot_prob(self.distdata)
            else:
                self.plot_hist(self.distdata)
        else:
            self.plot_pdf(self.distdata)
        self.update_report()

    def plot_pdf(self, data):
        ''' Plot PDF of the distribution '''
        median = data.get('median', data.get('mean', 0))
        std = data.get('std', 1)
        df = min(data.get('df', 100), 1E4)  # Inf degf doesn't work
        xx = np.linspace(median-5*std, median+5*std, num=200)
        dist = distributions.get_distribution('t', median=median, unc=std, df=df)
        yy = dist.pdf(xx)

        self.fig.clf()
        ax = self.fig.add_subplot(1, 1, 1)
        ax.plot(xx, yy)
        ax.set_xlabel('Value')
        ax.set_ylabel('Probability Density')
        self.canvas.draw_idle()

    def plot_hist(self, data):
        ''' Plot histogram of the sampled distribution '''
        samples = unitmgr.strip_units(data['samples'])
        fitdist, _ = self.fitdist(samples)
        xx = np.linspace(samples.min(), samples.max(), num=200)
        yy = fitdist.pdf(xx)

        self.fig.clf()
        ax = self.fig.add_subplot(1, 1, 1)
        with suppress(ValueError):
            ax.hist(samples, density=True, bins='sqrt')
        ax.plot(xx, yy, label='Distribution Fit')
        ax.set_ylabel('Probability Density')
        self.canvas.draw_idle()

    def fitdist(self, samples):
        ''' Fit a distribution to the sampled data '''
        distname = self.sampled.cmbDistType.currentText()
        if distname == 'normal (standard error)':
            params = {'median': np.median(samples),
                      'std': samples.std(ddof=1)/np.sqrt(len(samples))}
            fitdist = distributions.get_distribution('normal', **params)
        elif distname == 'normal (with autocorrelation)':
            params = {'median': np.median(samples),
                      'std': dataset.uncert_autocorrelated(samples).uncert}
            fitdist = distributions.get_distribution('normal', **params)
        else:
            fitdist = distributions.get_distribution(distname)
            if len(samples) > 10000:
                samples = samples[:10000]
            params = fitdist.fit(samples)
        return fitdist, params

    def plot_prob(self, data):
        ''' Plot P-P plot of sampled values '''
        samples = unitmgr.strip_units(data['samples'])
        thinsamples = samples[:10000]
        fitdist, params = self.fitdist(thinsamples)
        self.fig.clf()
        ax = self.fig.add_subplot(1, 1, 1)
        plotting.probplot(thinsamples[:1000], ax=ax, sparams=params, dist=fitdist.dist.name)
        self.canvas.draw_idle()

    def update_report(self):
        ''' Update report stats '''
        rpt = report.Report()
        params = self.distribution()
        mean = params.get('median', params.get('mean', 0))
        std = params.get('std', 0)
        degf = params.get('df', np.inf)
        rows = [
            ['Mean', report.Number(mean, fmin=1, matchto=std)],
            ['Standard Deviation', report.Number(std, fmin=1)],
            ['Degrees Freedom', format(degf, '.1g')]]
        rpt.table(rows, hdr=['Parameter', 'Value'])
        self.stats.setReport(rpt)

    def distribution(self):
        ''' Get parameters dict for selected distribution '''
        distname = self.sampled.cmbDistType.currentText()
        if 'samples' in self.distdata:
            samples = unitmgr.strip_units(self.distdata['samples'])
            if distname == 'normal (standard error)':
                distname = 'normal'
                params = {'median': np.mean(samples),
                          'std': samples.std(ddof=1)/np.sqrt(len(samples)),
                          'df': len(samples)-1}
            elif distname == 'normal (with autocorrelation)':
                params = {'median': np.mean(samples),
                          'std': dataset.uncert_autocorrelated(samples).uncert,
                          'df': len(samples)-1}
                distname = 'normal'
            else:
                _, params = self.fitdist(samples)
        else:
            params = self.distdata

        params = {k: unitmgr.strip_units(v) for k, v in params.items()}
        params['name'] = distname
        return params


class ArraySelectWidget(QtWidgets.QDialog):
    ''' Widget for selecting an array of values from the project or a file '''
    def __init__(self, project=None, parent=None):
        super().__init__(parent=parent)
        font = self.font()
        font.setPointSize(10)
        self.setFont(font)
        self.project = project
        gui_common.centerWindow(self, 1000, 800)
        self.setWindowTitle('Select Array')
        self.arr = None

        self.treeSource = ProjectTreeArrays(project=project)
        self.table = QtWidgets.QTableWidget()
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectColumns)

        self.fig = Figure()
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setStyleSheet("background-color:transparent;")
        self.dlgbutton = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Ok |
                                                    QtWidgets.QDialogButtonBox.StandardButton.Cancel)
        self.dlgbutton.rejected.connect(self.reject)
        self.dlgbutton.accepted.connect(self.accept)

        llayout = QtWidgets.QVBoxLayout()
        llayout.addWidget(QtWidgets.QLabel('Data Source:'))
        llayout.addWidget(self.treeSource)
        rlayout = QtWidgets.QVBoxLayout()
        rlayout.addWidget(self.canvas, stretch=5)
        rlayout.addWidget(self.table, stretch=5)
        rlayout.addWidget(self.dlgbutton)
        layout = QtWidgets.QHBoxLayout()
        layout.addLayout(llayout, stretch=2)
        layout.addLayout(rlayout, stretch=4)
        self.setLayout(layout)
        self.updateplot()
        self.treeSource.loaddata.connect(self.load_data)
        self.dlgbutton.button(QtWidgets.QDialogButtonBox.StandardButton.Ok).setEnabled(False)

    def updateplot(self):
        ''' Update the plot '''
        self.fig.clf()
        ax = self.fig.add_subplot(1, 1, 1)
        if self.arr is None:
            self.canvas.draw_idle()
            return

        arr = self.arr
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

    def get_array(self):
        ''' Return dictionary with keys for each column name '''
        return self.arr

    def load_data(self, arr):
        ''' Fill table with data array

            Args:
                arr: Dictionary of x, y, uy, ux
        '''
        self.table.clear()
        nrows = len(arr['x'])
        # Don't trust dictionary order when it comes through QT signals..
        if 'u(x)' in arr and 'u(y)' in arr:
            ncols = 4
            colnames = ['x', 'y', 'u(y)', 'u(x)']
        elif 'u(x)' in arr:
            ncols = 3
            colnames = ['x', 'y', 'u(x)']
        elif 'u(y)' in arr:
            ncols = 3
            colnames = ['x', 'y', 'u(y)']
        else:
            ncols = 2
            colnames = ['x', 'y']
        self.table.setRowCount(nrows)
        self.table.setColumnCount(ncols)
        self.table.setHorizontalHeaderLabels(colnames)

        for col, colname in enumerate(colnames):
            values = arr.get(colname)
            for row, value in enumerate(values):
                item = QtWidgets.QTableWidgetItem(str(value))
                item.setFlags(QtCore.Qt.ItemFlag.ItemIsSelectable | QtCore.Qt.ItemFlag.ItemIsEnabled)
                self.table.setItem(row, col, item)
        self.arr = arr
        self.table.resizeColumnsToContents()
        self.updateplot()
        self.dlgbutton.button(QtWidgets.QDialogButtonBox.StandardButton.Ok).setEnabled(True)
