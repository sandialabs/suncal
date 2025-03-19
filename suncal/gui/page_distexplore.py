''' GUI for exploring distributions and doing manual Monte Carlo '''

import numpy as np
from PyQt6 import QtWidgets, QtGui, QtCore
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

from ..project import ProjectDistExplore
from ..common import report, distributions, uparser
from . import widgets
from . import gui_styles
from .gui_common import BlockedSignals
from .gui_settings import gui_settings
from .help_strings import DistExploreHelp


class DistEntry(QtWidgets.QWidget):
    ''' Widget showing one line in the distribution setup list

        Parameters
        ----------
        name: string
            Name/Expression for this distribution
        dist: stats.rv_continuous
            Stats distribution
    '''
    addDist = QtCore.pyqtSignal(object)  # emits DistEntry object
    remDist = QtCore.pyqtSignal(object)
    sampleDist = QtCore.pyqtSignal(str)  # emits name/expr of dist
    changeDist = QtCore.pyqtSignal(str)

    def __init__(self, name='', dist=None, parent=None):
        super().__init__(parent)
        if dist is None:
            dist = distributions.get_distribution('normal')
        self.dist = dist
        self.btnAddRem = widgets.PlusMinusButton(stretch=False)
        self.btnCustom = QtWidgets.QToolButton()
        self.btnSample = QtWidgets.QToolButton()
        self.btnCustom.setText('normal...')
        self.btnSample.setText('Sample')
        self.btnAddRem.btnminus.setToolTip('Remove Distribution')
        self.btnAddRem.btnplus.setToolTip('Insert Distribution')
        self.btnCustom.setToolTip('Define distribution parameters')
        self.btnSample.setToolTip('Generate random samples')
        self.txtName = QtWidgets.QLineEdit(name)
        self.btnCustom.setSizePolicy(QtWidgets.QSizePolicy.Policy.Fixed, QtWidgets.QSizePolicy.Policy.Fixed)
        self.btnSample.setSizePolicy(QtWidgets.QSizePolicy.Policy.Fixed, QtWidgets.QSizePolicy.Policy.Fixed)

        self.btnCustom.clicked.connect(self.editdist)
        self.btnAddRem.plusclicked.connect(lambda x=self: self.addDist.emit(x))
        self.btnAddRem.minusclicked.connect(lambda x=self: self.remDist.emit(x))
        self.btnSample.clicked.connect(lambda x: self.sampleDist.emit(self.get_name()))
        self.txtName.editingFinished.connect(self.change_name)

        layout = QtWidgets.QHBoxLayout()
        layout.addWidget(self.btnAddRem)
        layout.addWidget(self.txtName)
        layout.addWidget(self.btnCustom)
        layout.addWidget(self.btnSample)
        self.setLayout(layout)

    def editdist(self):
        ''' Show dialog for editing distribution parameters '''
        name = self.get_name()
        dlg = DistDialog(name, self.dist, parent=self)
        ok = dlg.exec()
        if ok:
            self.dist = dlg.get_dist()
            self.btnCustom.setText(self.dist.dist.name + '...')
            self.changeDist.emit(name)

    def change_name(self):
        ''' Name was changed. Check if it's an expression or base variable '''
        name = self.get_name()
        try:
            expr = uparser.parse_math(name)
        except ValueError:
            with BlockedSignals(self.txtName):
                self.txtName.setText('ERROR')
        else:
            if hasattr(expr, 'is_symbol') and expr.is_symbol:
                self.btnCustom.setEnabled(True)  # Base variable, can customize
                self.btnCustom.setText(self.dist.dist.name + '...')
                self.btnSample.setText('Sample')
            else:
                self.btnCustom.setEnabled(False)
                self.btnCustom.setText('Monte Carlo')
                self.btnSample.setText('Calculate')
        self.changeDist.emit(name)

    def get_name(self):
        ''' Get name/expression for this distribution '''
        return self.txtName.text()


class DistDialog(QtWidgets.QDialog):
    ''' Dialog for editing distribution parameters '''
    def __init__(self, name, dist, parent=None):
        super().__init__(parent=parent)
        font = self.font()
        font.setPointSize(10)
        self.setFont(font)
        self.name = name
        self.dist = dist
        args = self.dist.get_config()
        args.update({'median': self.dist.median()})
        self.table = widgets.DistributionEditTable(args)
        self.fig = Figure()
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setStyleSheet("background-color:transparent;")
        self.btnBox = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Ok |
                                                 QtWidgets.QDialogButtonBox.StandardButton.Cancel)

        self.setWindowTitle('Configure Probability Distribution')
        self.setMaximumSize(600, 600)
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.table, stretch=10)
        layout.addWidget(self.canvas, stretch=10)
        layout.addWidget(self.btnBox)
        self.setLayout(layout)

        self.btnBox.accepted.connect(self.accept)
        self.btnBox.rejected.connect(self.reject)
        self.table.changed.connect(self.replot)
        self.replot()

    def get_dist(self):
        ''' Get the distribution distributions.Distribution object '''
        return self.dist

    def replot(self):
        ''' Update the distribution and replot '''
        self.dist = self.table.statsdist
        median = self.dist.median()
        std = self.dist.std()
        if not np.isfinite(std):  # Some distributions (e.g. alpha) have infinite stdev
            std = self.dist.kwds.get('scale', 1)
        xx = np.linspace(median - 4*std, median + 4*std, num=200)
        try:
            yy = self.dist.pdf(xx)
        except (TypeError, ValueError):
            yy = np.full(len(xx), np.nan)

        self.fig.clf()
        ax = self.fig.add_subplot(1, 1, 1)
        ax.plot(xx, yy)
        ax.set_xlabel(report.Math(self.name).latex())
        self.canvas.draw_idle()


class DistributionListWidget(QtWidgets.QWidget):
    ''' Widget for showing a list of distributions. Wrap this in a ScrollArea. '''
    sample = QtCore.pyqtSignal(str)
    changed = QtCore.pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.pagelayout = QtWidgets.QVBoxLayout()
        self.setLayout(self.pagelayout)
        self.pagelayout.addStretch()  # Stretch is always last item in layout

    def addItem(self, name='', dist=None, after=None):
        ''' Add a distribution to the list '''
        if dist is None:
            dist = distributions.get_distribution('normal')
        item = DistEntry(name, dist)
        item.addDist.connect(lambda x: self.addItem(after=x))
        item.remDist.connect(lambda x: self.remItem(item=x))
        item.changeDist.connect(self.changed)
        item.sampleDist.connect(self.sample)

        if after is not None:
            index = self.pagelayout.indexOf(after) + 1
        else:
            index = self.pagelayout.count() - 1
        self.pagelayout.insertWidget(index, item)
        self.changed.emit(name)

    def remItem(self, item=None):
        ''' Remove distribution '''
        self.pagelayout.removeWidget(item)
        item.setParent(None)
        name = item.txtName.text()
        if self.pagelayout.count() == 1:  # 1 because last item is stretch
            self.addItem()
        self.changed.emit(name)

    def clear(self):
        ''' Remove all items, but leave a blank one to edit. '''
        while self.pagelayout.count() > 1:
            item = self.pagelayout.itemAt(0).widget()
            if item:
                self.pagelayout.removeWidget(item)
                item.setParent(None)
        self.addItem()

    def get_dists(self):
        ''' Return list of distribution name/expressions. '''
        dists = {}
        for i in range(self.pagelayout.count()-1):  # -1 because last layout item is a stretch
            item = self.pagelayout.itemAt(i).widget()
            name = item.get_name()
            if name != 'ERROR':
                dists[name] = item.dist
        return dists


class DistExploreWidget(QtWidgets.QWidget):
    ''' Page widget for distribution explorer '''
    def __init__(self, component, parent=None):
        super().__init__(parent)
        assert isinstance(component, ProjectDistExplore)
        self.component = component

        self.distlist = DistributionListWidget()
        self.scroll = QtWidgets.QScrollArea()
        self.scroll.setWidget(self.distlist)
        self.scroll.setWidgetResizable(True)
        self.scroll.setMinimumWidth(400)
        self.samples = QtWidgets.QLineEdit('10000')
        self.samples.setValidator(QtGui.QIntValidator(1, 10000000))
        self.samples.setMaximumWidth(300)
        self.samples.editingFinished.connect(lambda: self.component.model.set_numsamples(int(self.samples.text())))
        self.seed = QtWidgets.QLineEdit('None')

        self.cmbView = QtWidgets.QComboBox()
        self.cmbView.addItems(self.component.model.samplevalues.keys())
        self.cmbFit = QtWidgets.QComboBox()
        dists = [d for d in gui_settings.distributions if distributions.fittable(d)]
        self.cmbFit.addItems(['None'] + dists)
        self.chkInterval = QtWidgets.QCheckBox('Show 95% Coverage')
        self.chkProbPlot = QtWidgets.QCheckBox('Show Probability Plot')
        self.fig = Figure()
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setStyleSheet("background-color:transparent;")
        self.toolbar = NavigationToolbar(self.canvas, self, coordinates=True)
        self.txtOutput = widgets.MarkdownTextEdit()

        slayout = QtWidgets.QHBoxLayout()
        slayout.addWidget(QtWidgets.QLabel('Samples:'))
        slayout.addWidget(self.samples)
        slayout.addStretch()
        slayout.addWidget(QtWidgets.QLabel('Random Seed:'))
        slayout.addWidget(self.seed)
        llayout = QtWidgets.QVBoxLayout()
        llayout.addWidget(QtWidgets.QLabel('Input distributions and Monte Carlo expressions:'))
        llayout.addWidget(self.scroll)
        llayout.addLayout(slayout)
        clayout = QtWidgets.QHBoxLayout()
        clayout.addWidget(QtWidgets.QLabel('Output:'))
        clayout.addWidget(self.cmbView)
        clayout.addStretch()
        clayout.addWidget(QtWidgets.QLabel('Fit:'))
        clayout.addWidget(self.cmbFit)
        clayout.addWidget(self.chkInterval)
        clayout.addWidget(self.chkProbPlot)
        rlayout = QtWidgets.QVBoxLayout()
        rlayout.addLayout(clayout)
        rlayout.addWidget(self.canvas, stretch=10)
        rlayout.addWidget(self.toolbar)
        self.topwidget = QtWidgets.QWidget()
        self.topwidget.setLayout(rlayout)
        self.rightsplitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        self.rightsplitter.addWidget(self.topwidget)
        self.rightsplitter.addWidget(self.txtOutput)
        self.leftwidget = QtWidgets.QWidget()
        self.leftwidget.setLayout(llayout)
        self.splitter = QtWidgets.QSplitter()
        self.splitter.addWidget(self.leftwidget)
        self.splitter.addWidget(self.rightsplitter)
        self.splitter.setCollapsible(0, False)
        self.splitter.setCollapsible(1, False)
        layout = QtWidgets.QHBoxLayout()
        layout.addWidget(self.splitter)
        self.setLayout(layout)

        # Initialize
        if len(self.component.model.dists) == 0:
            self.distlist.addItem('a')  # Make a new distribution named 'a'
            self.dist_changed()   # Save it
        else:
            for expr, dist in self.component.model.dists.items():
                self.distlist.addItem(str(expr), dist)

        self.menu = QtWidgets.QMenu('&Distributions')
        self.actSave = QtGui.QAction('&Save report...', self)
        self.actClear = QtGui.QAction('&Clear', self)
        self.menu.addAction(self.actClear)
        self.menu.addAction(self.actSave)
        self.actClear.triggered.connect(self.clear)
        self.actSave.triggered.connect(self.save_report)
        self.cmbView.currentIndexChanged.connect(self.changeview)
        self.cmbFit.currentIndexChanged.connect(self.changeview)
        self.chkProbPlot.stateChanged.connect(self.changeview)
        self.chkInterval.stateChanged.connect(self.changeview)
        self.distlist.sample.connect(self.sample)
        self.distlist.changed.connect(self.dist_changed)

    def get_menu(self):
        ''' Get the menu for this widget '''
        return self.menu

    def calculate(self):
        ''' Run the calculation. It runs automatically, so this does nothing '''

    def update_proj_config(self):
        ''' Update model with values entered on page.
            Happens on entry, nothing to do here.
        '''

    def clear(self):
        ''' Clear the distribution list '''
        self.component.model.samplevalues = {}
        self.distlist.clear()
        self.cmbView.clear()

    def changeview(self):
        ''' Output view changed. Update plot and text report. '''
        name = self.cmbView.currentText()
        if name in self.component.model.samplevalues:
            fitdist = self.cmbFit.currentText()
            fitdist = None if fitdist == 'None' else fitdist
            qq = self.chkProbPlot.isChecked()
            if fitdist is None and qq:
                fitdist = 'normal'

            fitparams = self.component.result.report.plot.hist(
                name, fig=self.fig, fitdist=fitdist, qqplot=qq, interval=self.chkInterval.isChecked())
            self.txtOutput.setReport(self.component.result.report.single(name, fitparams))
            self.fig.suptitle(report.Math(name).latex())
            self.canvas.draw_idle()
        else:
            self.txtOutput.setText('Sample a variable to see statistics.')

    def dist_changed(self, name=None):
        ''' Distribution changed, store to DistExplore object '''
        dists = self.distlist.get_dists()
        self.component.model.dists = dists
        if name is not None:
            self.component.model.samplevalues.pop(name, None)
        self.updatecmbView()
        self.changeview()

    def updatecmbView(self, name=None):
        ''' Add available (sampled) items to combo box and select name, if given '''
        self.cmbView.blockSignals(True)
        self.cmbView.clear()
        self.cmbView.addItems(list(self.component.model.samplevalues.keys()))
        if name is not None:
            self.cmbView.setCurrentIndex(self.cmbView.findText(name))
        else:
            self.cmbView.setCurrentIndex(0)
        self.cmbView.blockSignals(False)

    def sample(self, name):
        ''' Sample the distribution '''
        if name in [None, '', 'ERROR']:
            return
        try:
            seed = min(abs(int(self.seed.text())), 2**32-1)
        except (ValueError, TypeError):
            seed = None
        if seed is not None:
            np.random.seed(seed)

        try:
            self.component.model.sample(name)
        except ValueError:
            QtWidgets.QMessageBox.warning(self, 'Sampling', f'Undefined variable in expression {name}')
        self.updatecmbView(name=name)
        self.changeview()

    def get_report(self):
        ''' Get full report '''
        fitdist = self.cmbFit.currentText()
        fitdist = None if fitdist == 'None' else fitdist
        qq = self.chkProbPlot.isChecked()
        cov = self.chkInterval.isChecked()
        return self.component.result.report.all(fitdist=fitdist, coverage=cov, qqplot=qq)

    def save_report(self):
        ''' Save full report, asking user for settings/filename '''
        with gui_styles.LightPlotstyle():
            widgets.savereport(self.get_report())

    def help_report(self):
        ''' Get the help report to display the current widget mode '''
        return DistExploreHelp.disthelp()
