''' GUI for exploring distributions and doing manual Monte Carlo '''

import numpy as np
from PyQt5 import QtWidgets, QtGui, QtCore
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from .. import dist_explore
from .. import customdists
from .. import output
from .. import uparser
from . import gui_common
from . import gui_widgets


class DistInputPlot(QtWidgets.QWidget):
    ''' Widget for showing PDF plot and stats for a distribution as it's being entered '''
    def __init__(self, parent=None):
        super(DistInputPlot, self).__init__(parent=parent)
        self.fig = Figure()
        self.canvas = FigureCanvas(self.fig)
        self.lblMean = QtWidgets.QLabel('Mean: ---')
        self.lblMedian = QtWidgets.QLabel('Median: ---')
        self.lblStd = QtWidgets.QLabel('Std. Dev.: ---')

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.canvas)
        hlayout = QtWidgets.QHBoxLayout()
        hlayout.addWidget(self.lblMean)
        hlayout.addWidget(self.lblMedian)
        hlayout.addWidget(self.lblStd)
        hlayout.addStretch()
        layout.addLayout(hlayout)
        layout.addStretch()
        self.setLayout(layout)

    def update(self, dist, name):
        ''' Update the display with the distribution '''
        self.fig.clf()
        ax = self.fig.add_subplot(1, 1, 1)
        if dist:
            mean = dist.mean()
            median = dist.median()
            std = dist.std()
            xx = np.linspace(mean + 4*std, mean - 4*std, num=200)
            yy = dist.pdf(xx)
            ax.plot(xx, yy)
            ax.set_xlabel(name)
            self.fig.tight_layout()
            self.lblMean.setText('Mean: {:.4g}'.format(mean))
            self.lblMedian.setText('Median: {:.4g}'.format(median))
            self.lblStd.setText('Std. Dev.: {:.4g}'.format(std))
            self.canvas.setVisible(True)
        else:
            self.lblMean.setText('Run calculation to get statistics on Monte Carlo')
            self.lblMedian.setText('')
            self.canvas.setVisible(False)
            self.lblStd.setText('')
        self.canvas.draw_idle()


class DistOutputPlot(QtWidgets.QWidget):
    ''' Widget for showing histogram, sample stats, and fitting distributions '''
    def __init__(self, parent=None):
        super(DistOutputPlot, self).__init__(parent=parent)
        self.fig = Figure()
        self.canvas = FigureCanvas(self.fig)
        self.lblMean = QtWidgets.QLabel('Mean: ---')
        self.lblMedian = QtWidgets.QLabel('Median: ---')
        self.lblStd = QtWidgets.QLabel('Std. Dev.: ---')

        self.cmbFit = QtWidgets.QComboBox()
        self.cmbFit.addItems(['None'])
        dists = gui_common.settings.getDistributions()
        dists = [d for d in dists if hasattr(customdists.get_dist(d), 'fit')]
        self.cmbFit.addItems(dists)
        self.lblFitParams = QtWidgets.QLabel('')
        self.chkNormProb = QtWidgets.QCheckBox('Show Probability Plot')
        self.chkNormProb.setEnabled(False)

        llayout = QtWidgets.QVBoxLayout()
        llayout.addWidget(self.canvas)
        hlayout = QtWidgets.QHBoxLayout()
        hlayout.addWidget(QtWidgets.QLabel('Sample'))
        hlayout.addWidget(self.lblMean)
        hlayout.addWidget(self.lblMedian)
        hlayout.addWidget(self.lblStd)
        hlayout.addStretch()
        llayout.addLayout(hlayout)
        llayout.addStretch()

        rlayout = QtWidgets.QVBoxLayout()
        rlayout.addWidget(QtWidgets.QLabel('Fit Distribution:'))
        rlayout.addWidget(self.cmbFit)
        rlayout.addWidget(self.chkNormProb)
        rlayout.addSpacing(10)
        rlayout.addWidget(self.lblFitParams)
        rlayout.addStretch()
        layout = QtWidgets.QHBoxLayout()
        layout.addLayout(llayout)
        layout.addLayout(rlayout)
        self.setLayout(layout)
        self.cmbFit.currentIndexChanged.connect(self.replot)
        self.chkNormProb.stateChanged.connect(self.replot)

    def replot(self):
        ''' Refresh the plot based on control settings '''
        if self.samples is None:
            return

        mean = np.mean(self.samples)
        median = np.median(self.samples)
        std = np.std(self.samples, ddof=1)
        self.lblMean.setText('Mean: {:.4g}'.format(mean))
        self.lblMedian.setText('Median: {:.4g}'.format(median))
        self.lblStd.setText('Std. Dev.: {:.4g}'.format(std))

        fitname = self.cmbFit.currentText()
        if fitname == 'None':
            fitname = None
            self.chkNormProb.setEnabled(False)
            self.chkNormProb.setChecked(False)
        else:
            self.chkNormProb.setEnabled(True)

        self.fig.clf()
        try:
            params = output.fitdist(self.samples, dist=fitname, fig=self.fig, qqplot=self.chkNormProb.isChecked())
        except TypeError:  # samples is single-valued
            params = None

        if params is not None:
            plines = ['{}: {:.4g}'.format(p, v) for p, v in params.items()]
            self.lblFitParams.setText('Fit Parameters:\n\n  ' + '\n  '.join(plines))
        else:
            self.lblFitParams.setText('')

        self.fig.gca().set_xlabel(self.name)
        self.canvas.draw_idle()

    def update(self, samples, name):
        ''' Update the histogram with the sampled values '''
        self.samples = samples
        self.name = name
        if self.samples is not None:
            self.replot()
        else:
            self.lblMean.setText('Mean: ---')
            self.lblMedian.setText('Median: ---')
            self.lblStd.setText('Std. Dev.: ---')


class DistributionWidget(QtWidgets.QWidget):
    ''' Class for entering and displaying output of a single distribution '''
    insertdist = QtCore.pyqtSignal(object)
    remdist = QtCore.pyqtSignal(object)
    changed = QtCore.pyqtSignal()

    MINHEIGHT = 275

    def __init__(self, name='', dist=None, parent=None):
        super(DistributionWidget, self).__init__(parent=parent)
        self.setMinimumHeight(self.MINHEIGHT)
        distargs = customdists.get_config(dist) if dist is not None else None
        self.disttable = gui_widgets.DistributionEditTable(distargs)
        self.name = QtWidgets.QLineEdit(name)
        self.inputplot = DistInputPlot()
        self.outputplot = DistOutputPlot()
        self.plotstack = QtWidgets.QStackedWidget()
        self.plotstack.addWidget(self.inputplot)
        self.plotstack.addWidget(self.outputplot)
        self.btnadd = QtWidgets.QToolButton()
        self.btnrem = QtWidgets.QToolButton()
        self.btnadd.setText('+')
        self.btnadd.setToolTip('Insert distribution')
        self.btnrem.setText(gui_common.CHR_ENDASH)
        self.btnrem.setToolTip('Remove distribution')
        self.btnadd.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        self.btnrem.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)

        nlayout = QtWidgets.QHBoxLayout()
        nlayout.addWidget(self.btnadd)
        nlayout.addWidget(self.btnrem)
        nlayout.addWidget(QtWidgets.QLabel('Expression:'))
        nlayout.addWidget(self.name)
        llayout = QtWidgets.QVBoxLayout()
        llayout.addLayout(nlayout)
        llayout.addWidget(self.disttable)
        llayout.addStretch()
        layout = QtWidgets.QHBoxLayout()
        layout.addLayout(llayout, stretch=3)
        layout.addWidget(self.plotstack, stretch=8)
        self.setLayout(layout)

        self.disttable.changed.connect(self.distchanged)
        self.name.editingFinished.connect(self.distchanged)
        self.btnadd.clicked.connect(lambda y, x=self: self.insertdist.emit(x))
        self.btnrem.clicked.connect(lambda y, x=self: self.remdist.emit(x))
        self.distchanged()

    def distchanged(self):
        ''' Distribution input was changed. '''
        name = self.name.text()
        try:
            expr = uparser.get_expr(name)
        except ValueError:
            expr = ''
        if expr is None:
            self.name.setText('---')
            self.disttable.setVisible(True)
        elif hasattr(expr, 'is_symbol') and expr.is_symbol:
            self.inputplot.update(self.disttable.statsdist, self.name.text())
            self.disttable.setVisible(True)
            self.setMinimumHeight(self.MINHEIGHT)
        else:  # Expression for MC
            self.inputplot.update(None, None)
            self.disttable.setVisible(False)
            self.setMinimumHeight(70)
        self.changed.emit()

    def get_dist(self):
        ''' Get distribution and name '''
        return self.disttable.statsdist, self.name.text()

    def calculate(self, sampledvalues):
        ''' Switch to output view, showing histogram. sampledvalues is array of samples '''
        self.outputplot.update(sampledvalues, self.name.text())
        self.plotstack.setCurrentIndex(1)
        self.setMinimumHeight(self.MINHEIGHT)
        if QtWidgets.QApplication.focusWidget() is not None:
            QtWidgets.QApplication.focusWidget().clearFocus()  # Leaving focus can cause page to flip when it shouldn't

    def back(self):
        ''' Go back to input mode '''
        self.plotstack.setCurrentIndex(0)


class DistributionListWidget(QtWidgets.QWidget):
    ''' Class for showing a list of distributions. Wrap this in a scrollarea. '''
    changed = QtCore.pyqtSignal()

    def __init__(self, parent=None):
        super(DistributionListWidget, self).__init__(parent=parent)
        self.pagelayout = QtWidgets.QVBoxLayout()
        self.setLayout(self.pagelayout)

    def addItem(self, name='a', dist=None, after=None):
        ''' Add a distribution to the list '''
        item = DistributionWidget(name, dist)
        item.insertdist.connect(lambda x: self.addItem(after=x))
        item.remdist.connect(lambda x: self.remItem(item=x))
        item.changed.connect(self.changed)

        if after is not None:
            index = self.pagelayout.indexOf(after) + 1
        else:
            index = self.pagelayout.count()
        self.pagelayout.insertWidget(index, item)
        self.changed.emit()

    def remItem(self, item=None):
        ''' Remove distribution '''
        self.pagelayout.removeWidget(item)
        item.setParent(None)
        if self.pagelayout.count() == 0:
            self.addItem()
        self.changed.emit()

    def clear(self):
        ''' Remove all items '''
        while self.pagelayout.count() > 0:
            item = self.pagelayout.itemAt(0).widget()
            self.pagelayout.removeWidget(item)
            item.setParent(None)
        self.addItem()

    def get_distlist(self):
        ''' Get list of names, list of distributions defined in the widget '''
        dists = []
        names = []
        for i in range(self.pagelayout.count()):
            item = self.pagelayout.itemAt(i).widget()
            names.append(item.name.text())
            dists.append(item.disttable.statsdist)
        return names, dists

    def calculate(self, sampledvalues):
        ''' Switch all distributions to calculated view '''
        for i in range(self.pagelayout.count()):
            item = self.pagelayout.itemAt(i).widget()
            item.calculate(sampledvalues[i])

    def back(self):
        ''' Switch all distributions back to input view '''
        for i in range(self.pagelayout.count()):
            item = self.pagelayout.itemAt(i).widget()
            item.back()

    def get_lastfit_qq(self):
        ''' Get fit distribution and qq setting of last entry in the list (for report) '''
        i = self.pagelayout.count() - 1
        item = self.pagelayout.itemAt(i).widget()
        fit = item.outputplot.cmbFit.currentText()
        fit = None if fit == 'None' else fit
        qq = item.outputplot.chkNormProb.isChecked()
        return fit, qq


class DistWidget(QtWidgets.QWidget):
    ''' Page widget for distribution explorer '''
    def __init__(self, item, parent=None):
        super(DistWidget, self).__init__(parent)
        assert isinstance(item, dist_explore.DistExplore)
        self.distexplore = item

        self.distlist = DistributionListWidget()
        self.scroll = QtWidgets.QScrollArea()
        self.scroll.setWidget(self.distlist)
        self.scroll.setWidgetResizable(True)
        self.samples = QtWidgets.QLineEdit('100000')
        self.samples.setValidator(QtGui.QIntValidator(1, 10000000))
        self.samples.setMaximumWidth(300)
        self.btnCalc = QtWidgets.QPushButton('Sample')
        self.btnCalc.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)

        clayout = QtWidgets.QHBoxLayout()
        clayout.addWidget(QtWidgets.QLabel('Samples:'))
        clayout.addWidget(self.samples)
        clayout.addStretch()
        clayout.addWidget(self.btnCalc)
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.scroll)
        layout.addLayout(clayout)
        self.setLayout(layout)

        self.menu = QtWidgets.QMenu('Distributions')
        self.actSave = QtWidgets.QAction('Save report...', self)
        self.actClear = QtWidgets.QAction('Clear', self)
        self.menu.addAction(self.actClear)
        self.menu.addAction(self.actSave)
        self.actClear.triggered.connect(self.distlist.clear)
        self.actSave.triggered.connect(self.save_report)

        # Initialize
        if len(self.distexplore.distexpr) == 0:
            self.distlist.addItem('a', None)  # Make a new distribution named 'a'
            self.dist_changed()   # Save it
        else:
            for expr, dist in zip(self.distexplore.distexpr, self.distexplore.distlist):
                self.distlist.addItem(str(expr), dist)

        self.btnCalc.clicked.connect(self.calculate)
        self.distlist.changed.connect(self.dist_changed)

    def get_menu(self):
        ''' Get the menu for this widget '''
        return self.menu

    def dist_changed(self):
        ''' Distribution changed, store to DistExplore object '''
        names, dists = self.distlist.get_distlist()
        self.distexplore.distexpr = [uparser.get_expr(name) for name in names]
        self.distexplore.distlist = dists
        if self.btnCalc.text() == 'Back':
            self.distlist.back()
            self.btnCalc.setText('Sample')

    def calculate(self):
        ''' Run Monte-Carlo, sampling each distribution '''
        if self.btnCalc.text() == 'Sample':
            self.distexplore.set_numsamples(int(self.samples.text()))
            self.distexplore.calculate()
            self.distlist.calculate(self.distexplore.out.samples)  # List of arrays of sampled values
            self.btnCalc.setText('Back')
        else:
            self.distlist.back()
            self.btnCalc.setText('Sample')

    def get_report(self):
        ''' Get full report '''
        fitdist, qq = self.distlist.get_lastfit_qq()
        return self.distexplore.get_output().report_all(fitdist=fitdist, qqplot=qq, **gui_common.get_rptargs())

    def save_report(self):
        ''' Save full report, asking user for settings/filename '''
        gui_widgets.savemarkdown(self.get_report())
