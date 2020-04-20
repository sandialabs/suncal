''' User interface for Risk Analysis '''
import numpy as np
from PyQt5 import QtWidgets, QtGui, QtCore
from matplotlib.figure import Figure
from matplotlib.ticker import FormatStrFormatter, ScalarFormatter
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

from .. import risk
from .. import distributions
from . import gui_common
from . import gui_widgets
from . import page_dataimport


class DoubleLineEdit(QtWidgets.QWidget):
    ''' Widget with two line edits for Doubles '''
    editingFinished = QtCore.pyqtSignal()

    def __init__(self, value1=0, value2=0, label1='', label2=''):
        super().__init__()
        self.line1 = QtWidgets.QLineEdit(str(value1))
        self.line2 = QtWidgets.QLineEdit(str(value2))
        self.line1.setValidator(QtGui.QDoubleValidator(-1E99, 1E99, 15))
        self.line2.setValidator(QtGui.QDoubleValidator(-1E99, 1E99, 15))
        layout = QtWidgets.QFormLayout()
        layout.addRow(label1, self.line1)
        layout.addRow(label2, self.line2)
        self.setLayout(layout)

        self.line1.editingFinished.connect(self.editingFinished)
        self.line2.editingFinished.connect(self.editingFinished)

    def getValue(self):
        ''' Return tuple value of two lines '''
        return float(self.line1.text()), float(self.line2.text())

    def setValue(self, value1, value2):
        ''' Set value of both lines '''
        self.line1.setText(str(value1))
        self.line2.setText(str(value2))


class SimpleRiskWidget(QtWidgets.QWidget):
    ''' Controls for simple-mode risk calculations (TUR, SL, and GBF) '''
    editingFinished = QtCore.pyqtSignal()

    def __init__(self):
        super().__init__()
        self.tur = QtWidgets.QDoubleSpinBox()
        self.tur.setRange(0.1, 1000)
        self.tur.setValue(4.0)
        self.tur.setSingleStep(0.1)
        self.tur.setDecimals(2)
        self.gbfactor = QtWidgets.QDoubleSpinBox()
        self.gbfactor.setRange(-100, 100)
        self.gbfactor.setValue(1.0)
        self.gbfactor.setSingleStep(0.01)
        self.gbfactor.setDecimals(2)
        self.itp = QtWidgets.QDoubleSpinBox()
        self.itp.setRange(0.01, 99.99)
        self.itp.setValue(95.0)
        self.itp.setSingleStep(1)
        self.itp.setDecimals(2)
        self.measured = QtWidgets.QSlider(orientation=1)
        self.measured.setRange(0, 200)  # Slider must be an integer
        self.measured.setValue(100)

        self.tur.valueChanged.connect(self.editingFinished)
        self.gbfactor.valueChanged.connect(self.editingFinished)
        self.itp.valueChanged.connect(self.editingFinished)
        self.measured.valueChanged.connect(self.editingFinished)

        layout = QtWidgets.QFormLayout()
        layout.addRow('Test Uncertainty Ratio:', self.tur)
        hlayout = QtWidgets.QHBoxLayout()
        hlayout.addWidget(self.itp)
        layout.addRow('In-tolerance probability:', hlayout)
        layout.addRow('Guard band factor:', self.gbfactor)
        layout.addRow('Test Measurement:', self.measured)
        self.setLayout(layout)

    def get_measured_fraction(self):
        ''' Get measured value from slider, converting to fraction of spec limits '''
        # Scale the 0-200 slider integer to some fraction of spec limits (-1 to 1)
        return (self.measured.value() / self.measured.maximum())*2 - 1

    def get_tur(self):
        ''' Get TUR value '''
        return self.tur.value()

    def get_itp(self):
        ''' Get itp value '''
        return self.itp.value() / 100

    def get_gbf(self):
        ''' Get gbf value '''
        return self.gbfactor.value()


class GuardBandFinderWidget(QtWidgets.QDialog):
    ''' Widget providing options for calculating a guard band '''
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Calculate Guard Band')
        self.pfa = QtWidgets.QRadioButton('Target PFA %')
        self.pfaval = QtWidgets.QDoubleSpinBox()
        self.pfaval.setValue(0.8)  # in percent
        self.pfaval.setRange(.01, 99.99)
        self.pfaval.setDecimals(2)
        self.pfaval.setSingleStep(0.1)
        self.dobbert = QtWidgets.QRadioButton('Dobbert Managed PFA')
        self.rss = QtWidgets.QRadioButton('RSS')
        self.rp10 = QtWidgets.QRadioButton('NCSL RP10')
        self.test = QtWidgets.QRadioButton('95% Test Uncertainty')
        self.fourtoone = QtWidgets.QRadioButton('Same as 4:1')
        self.mincost = QtWidgets.QRadioButton('Minimum Cost')
        self.minimax = QtWidgets.QRadioButton('Minimax Cost')
        self.buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        self.pfa.setChecked(True)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)

        layout = QtWidgets.QGridLayout()
        layout.addWidget(QtWidgets.QLabel('Select Guard Band Method'), 0, 0, 1, 2)
        layout.addWidget(self.pfa, 1, 0)
        layout.addWidget(self.pfaval, 1, 1)
        layout.addWidget(self.dobbert, 2, 0)
        layout.addWidget(QtWidgets.QLabel(u'<b>k = 1 - M<sub>2{}</sub>/TUR</b>'.format(gui_common.CHR_PERCENT)), 2, 1)
        layout.addWidget(self.rss, 3, 0)
        layout.addWidget(QtWidgets.QLabel(u'<b>k = {}(1-1/TUR<sup>2</sup>)</b>'.format(gui_common.CHR_SQRT)), 3, 1)
        layout.addWidget(self.rp10, 4, 0)
        layout.addWidget(QtWidgets.QLabel('<b>k = 1.25 - 1/TUR</b>'), 4, 1)
        layout.addWidget(self.test, 5, 0)
        layout.addWidget(QtWidgets.QLabel('<b>k = 1 - 1/TUR</b>'), 5, 1)
        layout.addWidget(self.fourtoone, 6, 0)
        layout.addWidget(self.mincost, 7, 0)
        layout.addWidget(self.minimax, 8, 0)

        mainlayout = QtWidgets.QVBoxLayout()
        mainlayout.addLayout(layout)
        mainlayout.addWidget(self.buttons)
        self.setLayout(mainlayout)

    def get_method(self):
        ''' Get selected guard band method, as dictionary of arguments for risk.get_guardband() '''
        kargs = {}
        if self.pfa.isChecked():
            kargs['method'] = 'pfa'
            kargs['pfa'] = self.pfaval.value() / 100
        elif self.dobbert.isChecked():
            kargs['method'] = 'dobbert'
        elif self.rss.isChecked():
            kargs['method'] = 'rss'
        elif self.rp10.isChecked():
            kargs['method'] = 'rp10'
        elif self.test.isChecked():
            kargs['method'] = 'test'
        elif self.fourtoone.isChecked():
            kargs['method'] = '4:1'
        elif self.mincost.isChecked():
            kargs['method'] = 'mincost'
        elif self.minimax.isChecked():
            kargs['method'] = 'minimax'
        return kargs


class CostEntryWidget(QtWidgets.QDialog):
    ''' Widget for entering cost of FA and FR '''
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Expected costs')
        self.costfa = QtWidgets.QDoubleSpinBox()
        self.costfr = QtWidgets.QDoubleSpinBox()
        self.costfa.setRange(0, 1000000000)
        self.costfr.setRange(0, 1000000000)
        self.costfa.setValue(100)
        self.costfr.setValue(1)
        self.buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        flayout = QtWidgets.QFormLayout()
        flayout.addRow('Cost of False Accept', self.costfa)
        flayout.addRow('Cost of False Reject', self.costfr)
        layout = QtWidgets.QVBoxLayout()
        layout.addLayout(flayout)
        layout.addWidget(self.buttons)
        self.setLayout(layout)

    def getCost(self):
        return self.costfa.value(), self.costfr.value()


class RiskWidget(QtWidgets.QWidget):
    ''' Widget for risk calculations '''
    def __init__(self, item, parent=None):
        super().__init__(parent)
        assert isinstance(item, risk.Risk)
        self.urisk = item
        self.urisk.calculate()  # With risk, calculate just creates an output object
        self.plotlines = {}  # Saved lines in plot
        self.mode = QtWidgets.QComboBox()
        self.mode.addItems(['Simple', 'Full'])
        self.montecarlo = QtWidgets.QComboBox()
        self.montecarlo.addItems(['Integral', 'Monte-Carlo'])

        self.simple = SimpleRiskWidget()
        self.limits = DoubleLineEdit(-2, 2, 'Lower Specification Limit:', 'Upper Specification Limit:')
        self.chkProc = QtWidgets.QCheckBox('Process Distribution:')
        self.chkTest = QtWidgets.QCheckBox('Test Measurement:')
        self.guardband = DoubleLineEdit(0, 0, 'Lower Guard Band (relative):', 'Upper Guard Band (relative):')
        self.chkGB = QtWidgets.QCheckBox('Guard Band')

        self.limits.setVisible(False)
        self.chkProc.setVisible(False)
        self.chkTest.setVisible(False)
        self.guardband.setVisible(False)
        self.chkGB.setVisible(False)

        self.txtNotes = QtWidgets.QPlainTextEdit()
        self.fig = Figure()
        self.canvas = FigureCanvas(self.fig)
        self.toolbar = NavigationToolbar(self.canvas, self, coordinates=True)
        self.txtOutput = gui_widgets.MarkdownTextEdit()

        if self.urisk.is_simple():
            self.mode.setCurrentIndex(0)
            self.simple.tur.setValue(self.urisk.get_tur())
            self.simple.gbfactor.setValue(self.urisk.get_gbf())
            self.simple.itp.setValue(self.urisk.get_itp()*100)
            procargs = self.urisk.get_procdist_args()
            testargs = self.urisk.get_testdist_args()
            testargs.update({'bias': self.urisk.testbias})
            self.dproc_table = gui_widgets.DistributionEditTable(initargs=procargs)
            self.dtest_table = gui_widgets.DistributionEditTable(initargs=testargs, locslider=True)
            self.chkProc.setChecked(True)
            self.chkTest.setChecked(True)

        else:  # Full Risk
            self.mode.setCurrentIndex(1)
            if self.urisk.get_procdist() is None:
                procargs = {'dist': 'normal', 'median': 0, 'std': 1}
            else:
                procargs = self.urisk.get_procdist_args()

            if self.urisk.get_testdist() is None:
                testargs = {'dist': 'normal', 'std': .25, 'bias': 0}
            else:
                testargs = self.urisk.get_testdist_args()
                testargs.update({'bias': self.urisk.testbias})
            self.dproc_table = gui_widgets.DistributionEditTable(initargs=procargs)
            self.dtest_table = gui_widgets.DistributionEditTable(initargs=testargs, locslider=True)
            self.limits.setValue(*self.urisk.get_speclimits())
            self.guardband.setValue(*self.urisk.get_guardband())

            if self.urisk.testdist is None and self.urisk.procdist is None:
                self.urisk.set_testdist(distributions.get_distribution('normal', std=0.125))
                self.urisk.set_procdist(distributions.get_distribution('normal', std=1))

            elif self.urisk.testdist is None:
                self.urisk.set_testdist(distributions.get_distribution('normal', std=0.125))
                self.urisk.testdist_saved = self.urisk.testdist
                self.urisk.testdist = None
                self.dtest_table.setEnabled(False)
                self.chkGB.setEnabled(False)
                self.guardband.setEnabled(False)

            elif self.urisk.procdist is None:
                self.urisk.set_procdist(distributions.get_distribution('normal', std=1))
                self.urisk.procdist_saved = self.urisk.procdist
                self.urisk.procdist = None
                self.dproc_table.setEnabled(False)

            if self.urisk.testdist is not None:
                self.chkTest.setChecked(True)
                if self.urisk.guardband[0] > 0 or self.urisk.guardband[1] > 0:
                    self.chkGB.setChecked(True)

        self.chkProc.setChecked(self.urisk.get_procdist() is not None)
        self.txtNotes.setPlainText(self.urisk.description)

        layout = QtWidgets.QHBoxLayout()
        vlayout = QtWidgets.QVBoxLayout()
        flayout = QtWidgets.QFormLayout()
        flayout.addRow('Mode:', self.mode)
        flayout.addRow('Calculation:', self.montecarlo)
        vlayout.addLayout(flayout)
        vlayout.addWidget(self.simple)
        vlayout.addWidget(self.limits)
        vlayout.addWidget(self.chkProc)
        vlayout.addWidget(self.dproc_table)
        vlayout.addWidget(self.chkTest)
        vlayout.addWidget(self.dtest_table)
        vlayout.addWidget(self.chkGB)
        vlayout.addWidget(self.guardband)
        vlayout.addWidget(QtWidgets.QLabel('Notes:'))
        vlayout.addWidget(self.txtNotes)
        rlayout = QtWidgets.QVBoxLayout()
        rlayout.addWidget(self.canvas, stretch=20)
        rlayout.addWidget(self.toolbar)
        rlayout.addWidget(self.txtOutput, stretch=8)
        layout.addLayout(vlayout, stretch=1)
        layout.addLayout(rlayout, stretch=2.5)
        self.setLayout(layout)

        self.mode.currentIndexChanged.connect(self.changemode)
        self.simple.editingFinished.connect(self.replot_and_update)
        self.chkTest.stateChanged.connect(self.testprocclick)
        self.chkProc.stateChanged.connect(self.testprocclick)
        self.chkGB.stateChanged.connect(self.gbclick)
        self.dproc_table.changed.connect(self.replot_and_update)
        self.dtest_table.changed.connect(self.replot_and_update)
        self.limits.editingFinished.connect(self.replot_and_update)
        self.guardband.editingFinished.connect(self.replot_and_update)
        self.txtNotes.textChanged.connect(self.update_description)
        self.montecarlo.currentIndexChanged.connect(self.changecalc)

        self.menu = QtWidgets.QMenu('Risk')
        self.actImportDist = QtWidgets.QAction('Import distribution...', self)
        self.actCalcGB = QtWidgets.QAction('Calculate guardband...', self)
        self.actCost = QtWidgets.QAction('Set expected costs...', self)
        self.actSaveReport = QtWidgets.QAction('Save Report...', self)
        self.menu.addAction(self.actImportDist)
        self.menu.addSeparator()
        self.menu.addAction(self.actCalcGB)
        self.menu.addAction(self.actCost)
        self.menu.addSeparator()
        self.menu.addAction(self.actSaveReport)
        self.actImportDist.triggered.connect(self.importdist)
        self.actCalcGB.triggered.connect(self.calc_guardband)
        self.actCost.triggered.connect(self.set_costs)
        self.actSaveReport.triggered.connect(self.save_report)
        self.initplot()
        self.changemode()  # Show/hide controls

    def calculate(self):
        ''' Run calculation. Risk is calculated automatically, so this does nothing. '''
        pass

    def get_menu(self):
        ''' Get the menu for this widget '''
        return self.menu

    def update_description(self):
        ''' Description was updated, save it. '''
        self.urisk.description = self.txtNotes.toPlainText()

    def block(self, block):
        ''' Block all signals in Risk widget '''
        self.blockSignals(block)
        self.simple.blockSignals(block)
        self.simple.tur.blockSignals(block)
        self.simple.gbfactor.blockSignals(block)
        self.simple.itp.blockSignals(block)
        self.limits.blockSignals(block)
        self.guardband.blockSignals(block)
        self.chkProc.blockSignals(block)
        self.chkTest.blockSignals(block)
        self.chkGB.blockSignals(block)
        self.dproc_table.blockSignals(block)
        self.dtest_table.blockSignals(block)
        self.mode.blockSignals(block)

    def changemode(self, replot=True):
        ''' Mode changed (simple to full) '''
        self.block(True)
        simple = self.mode.currentText() == 'Simple'
        if simple and not self.urisk.is_simple():
            self.urisk.to_simple()
            self.simple.tur.setValue(self.urisk.get_tur())
            self.simple.gbfactor.setValue(self.urisk.get_gbf())
            self.simple.itp.setValue(self.urisk.get_itp()*100)
            self.chkProc.setChecked(True)
            self.chkTest.setChecked(True)
            self.chkProc.setEnabled(True)
            self.chkTest.setEnabled(True)

        elif not simple and self.urisk.is_simple():
            self.limits.setValue(*[np.round(x, 3) for x in self.urisk.get_speclimits()])
            self.guardband.setValue(*[np.round(x, 3) for x in self.urisk.get_guardband()])
            self.dproc_table.set_disttype(self.urisk.get_procdist_args())
            self.dtest_table.set_disttype(self.urisk.get_testdist_args())
            self.dproc_table.valuechanged()
            self.dtest_table.valuechanged()

        self.simple.setVisible(simple)
        self.limits.setVisible(not simple)
        self.chkProc.setVisible(not simple)
        self.chkTest.setVisible(not simple)
        self.dproc_table.setVisible(not simple)
        self.dtest_table.setVisible(not simple)
        self.chkGB.setVisible(not simple)
        self.guardband.setVisible(not simple)
        self.block(False)
        if replot:
            self.replot_and_update()

    def changecalc(self):
        ''' Change calculation mode (integration vs monte carlo) '''
        self.block(True)
        if self.montecarlo.currentText() == 'Monte-Carlo':
            self.chkProc.setChecked(True)
            self.chkTest.setChecked(True)
            self.chkProc.setEnabled(False)
            self.chkTest.setEnabled(False)
        else:
            self.chkTest.setEnabled(True)
            self.chkProc.setEnabled(True)
            self.testprocclick()
        self.block(False)
        self.initplot()
        self.replot_and_update()

    def replot_and_update(self):
        ''' Replot and update the text fields '''
        if self.mode.currentText() == 'Simple':
            if ((self.simple.get_gbf() != 1.0 and self.urisk.get_gbf() == 1.0) or
                (self.simple.get_gbf() == 1.0 and self.urisk.get_gbf() != 1.0)):
                self.chkGB.setChecked(self.simple.get_gbf() != 1.0)
            self.urisk.set_itp(self.simple.get_itp())
            self.urisk.set_tur(self.simple.get_tur())
            self.urisk.set_gbf(self.simple.get_gbf())
            self.urisk.set_testmedian(self.simple.get_measured_fraction() * self.get_range()[1])

        else:
            if self.chkProc.isChecked():
                self.urisk.set_procdist(self.dproc_table.statsdist)
            if self.chkTest.isChecked():
                self.urisk.set_testdist(self.dtest_table.statsdist, self.dtest_table.distbias)
            if self.chkGB.isChecked():
                self.urisk.set_guardband(*self.guardband.getValue())
            else:
                self.urisk.set_guardband(0, 0)
            self.urisk.set_speclimits(*self.limits.getValue())

        if self.montecarlo.currentText() == 'Monte-Carlo':
            self.replot_mc()
        else:
            self.update_range()
            self.replot()
            self.update_report()

    def initplot(self):
        ''' (Re)initialize the plot when the mode or options change '''
        self.fig.clf()
        self.axes = []  # List of axes (1, 2, or 3)
        self.plotlines = {}
        if self.montecarlo.currentText() == 'Monte-Carlo':
            # Set up monte carlo plot
            pass

        else:
            # Set up distributions plot(s)
            nrows = self.chkTest.isChecked() + self.chkProc.isChecked()
            plotnum = 0
            LL, UL = self.limits.getValue()
            GBL, GBU = self.guardband.getValue()
            if self.chkProc.isChecked():
                ax = self.fig.add_subplot(nrows, 1, plotnum+1)
                self.axes.append(ax)
                self.plotlines['procdist'], = ax.plot(0, 0, label='Process Distribution')
                self.plotlines['LL'] = ax.axvline(LL, ls='--', color='C2', label='Specification Limits')
                self.plotlines['UL'] = ax.axvline(UL, ls='--', color='C2')
                ax.set_ylabel('Probability Density')
                ax.set_xlabel('Value')
                ax.legend(loc='upper left')
                plotnum += 1

            if self.chkTest.isChecked():
                ax = self.fig.add_subplot(nrows, 1, plotnum+1)
                self.axes.append(ax)
                self.plotlines['testdist'], = ax.plot(0, 0, color='C1', label='Test Distribution')
                self.plotlines['testmed'] = ax.axvline(0, ls='--', color='lightgray', lw=1)
                self.plotlines['testexpected'] = ax.axvline(0, ls='--', color='C1')
                if self.chkProc.isChecked():
                    self.plotlines['product'], = ax.plot(0, 0, color='C4', label='Combined Distribution')
                self.plotlines['LL2'] = ax.axvline(LL, ls='--', color='C2', label='Specification Limits')
                self.plotlines['UL2'] = ax.axvline(UL, ls='--', color='C2')
                if self.chkGB.isChecked():
                    self.plotlines['GBL'] = ax.axvline(LL + GBL, ls='-.', color='C3', label='Guard Band Limit')
                    self.plotlines['GBU'] = ax.axvline(UL - GBU, ls='-.', color='C3')
                ax.set_ylabel('Probability Density')
                ax.set_xlabel('Value')
                ax.legend(loc='upper left')

    def replot(self):
        ''' Update the plot (without clearing completely) '''
        # Note GUI is not using urisk.output.plot_dists() so that it can work interactively without replotting full figure.
        LL, UL = self.urisk.get_speclimits()
        LL, UL = min(LL, UL), max(LL, UL)
        GBL = 0
        GBU = 0
        if ((self.mode.currentText() == 'Simple' and self.simple.get_gbf() > 0) or
            (self.mode.currentText() != 'Simple' and self.chkGB.isChecked())):
            GBL, GBU = self.urisk.get_guardband()

        xmin, xmax = self.get_range()
        x = np.linspace(xmin, xmax, num=300)
        [ax.collections.clear() for ax in self.axes]  # Remove old fill-betweens

        plotnum = 0
        if self.chkProc.isChecked() or self.mode.currentText() == 'Simple':
            yproc = self.urisk.get_procdist().pdf(x)
            self.plotlines['procdist'].set_data(x, yproc)
            self.axes[plotnum].fill_between(x, yproc, where=((x <= LL) | (x >= UL)), label='Nonconforming', alpha=.5, color='C0')
            self.plotlines['LL'].set_xdata([LL, LL])
            self.plotlines['UL'].set_xdata([UL, UL])
            if self.mode.currentText() == 'Simple':
                self.axes[plotnum].xaxis.set_major_formatter(FormatStrFormatter(r'%dSL'))
            else:
                self.axes[plotnum].xaxis.set_major_formatter(ScalarFormatter())
            plotnum += 1

        if self.chkTest.isChecked() or self.mode.currentText() == 'Simple':
            ytest = self.urisk.get_testdist().pdf(x)
            median = self.urisk.get_testmedian()
            measured = median + self.urisk.get_testbias()
            self.plotlines['testdist'].set_data(x, ytest)
            self.plotlines['testmed'].set_xdata([median]*2)
            self.plotlines['testexpected'].set_xdata([measured]*2)
            self.plotlines['LL2'].set_xdata([LL, LL])
            self.plotlines['UL2'].set_xdata([UL, UL])
            if self.mode.currentText() == 'Simple':
                self.axes[plotnum].xaxis.set_major_formatter(FormatStrFormatter(r'%dSL'))
            else:
                self.axes[plotnum].xaxis.set_major_formatter(ScalarFormatter())

            if measured > UL-GBU or measured < LL+GBL:  # Shade PFR
                self.axes[plotnum].fill_between(x, ytest, where=((x >= LL) & (x <= UL)), label='False Reject', alpha=.5, color='C1')
            else:  # Shade PFA
                self.axes[plotnum].fill_between(x, ytest, where=((x <= LL) | (x >= UL)), label='False Accept', alpha=.5, color='C1')

            if self.chkGB.isChecked():
                self.plotlines['GBL'].set_xdata([LL+GBL, LL+GBL])
                self.plotlines['GBU'].set_xdata([UL-GBU, UL-GBU])

            if self.chkProc.isChecked():
                self.plotlines['product'].set_data(x, ytest*yproc)

        [ax.relim() for ax in self.axes]
        [ax.autoscale_view(True, True, True) for ax in self.axes]
        self.fig.tight_layout()
        self.canvas.draw_idle()

    def update_report(self):
        ''' Update label fields, recalculating risk values '''
        self.txtOutput.setReport(self.urisk.out.report())

    def replot_mc(self):
        ''' Replot/report monte carlo method '''
        rpt = self.urisk.out.report_montecarlo(fig=self.fig)
        self.fig.tight_layout()
        self.canvas.draw_idle()
        self.txtOutput.setReport(rpt)

    def testprocclick(self):
        ''' Test Measurement or Process Distribution checkbox was clicked '''
        self.dtest_table.setEnabled(self.chkTest.isChecked())
        self.dproc_table.setEnabled(self.chkProc.isChecked())
        if not self.chkTest.isChecked():
            self.chkGB.setChecked(False)

        # Save off distributions to restore later
        if not self.chkTest.isChecked() and self.urisk.testdist is not None:
            self.urisk.testdist_saved = self.urisk.testdist
            self.urisk.testdist = None
        elif self.chkTest.isChecked() and self.urisk.testdist is None:
            self.urisk.testdist = self.urisk.testdist_saved

        if not self.chkProc.isChecked() and self.urisk.procdist is not None:
            self.urisk.procdist_saved = self.urisk.procdist
            self.urisk.procdist = None
        elif self.chkProc.isChecked() and self.urisk.procdist is None:
            self.urisk.procdist = self.urisk.procdist_saved

        self.chkGB.setEnabled(self.chkTest.isChecked())
        self.guardband.setEnabled(self.chkTest.isChecked() and self.chkGB.isChecked())
        self.initplot()
        self.replot_and_update()

    def gbclick(self):
        ''' Guardband checkbox was clicked '''
        self.guardband.setEnabled(self.chkGB.isChecked())
        self.initplot()
        self.replot_and_update()

    def get_range(self):
        ''' Returns lower, upper limit to plot range '''
        LL, UL = self.limits.getValue()
        LL, UL = min(LL, UL), max(LL, UL)
        LLpad = self.dproc_table.statsdist.std()*3
        ULpad = self.dproc_table.statsdist.std()*3
        if self.chkTest.isChecked():
            LLpad = np.nanmax([LLpad, self.dtest_table.statsdist.std()*5])
            ULpad = np.nanmax([LLpad, self.dtest_table.statsdist.std()*5])

        LL -= LLpad
        UL += ULpad
        return LL, UL

    def update_range(self):
        ''' Update the range in the distribution table widget '''
        LL, UL = self.get_range()
        self.dtest_table.set_locrange(LL, UL)

    def calc_guardband(self):
        ''' Determine guardband to hit specified PFA '''
        simple = self.mode.currentText() == 'Simple'
        if self.urisk.get_testdist() is None:
            QtWidgets.QMessageBox.information(self, 'Uncertainty Calculator', 'Please enable test distribution before finding guard band.')
            return

        dlg = GuardBandFinderWidget()
        ok = dlg.exec()
        if ok:
            self.block(True)
            methodargs = dlg.get_method()
            if methodargs['method'] in ['mincost', 'minimax'] and (self.urisk.cost_FA is None or self.urisk.cost_FR is None):
                QtWidgets.QMessageBox.information(self, 'Uncertainty Calculator', 'Please specify expected costs using Risk menu before using mincost or minimax guardbanding.')
                return

            self.urisk.calc_guardband(**methodargs)
            if simple:
                self.simple.gbfactor.setValue(self.urisk.get_gbf())
                self.chkGB.setChecked(True)
                self.chkTest.setChecked(True)
                self.chkProc.setChecked(True)
            else:
                self.guardband.setValue(*self.urisk.get_guardband())
                self.chkGB.setChecked(True)
                self.chkTest.setChecked(True)
                self.chkProc.setChecked(True)

            self.block(False)
            self.initplot()
            self.replot_and_update()

    def set_costs(self):
        ''' Set expected cost of FA and FR '''
        dlg = CostEntryWidget()
        ok = dlg.exec()
        if ok:
            self.urisk.set_costs(*dlg.getCost())
        self.replot_and_update()

    def get_report(self):
        ''' Get full report of curve fit, using page settings '''
        mc = self.montecarlo.currentText() == 'Monte-Carlo'
        return self.urisk.get_output().report_all(mc=mc)

    def save_report(self):
        ''' Save full report, asking user for settings/filename '''
        gui_widgets.savereport(self.get_report())

    def importdist(self):
        ''' Use process distribution from the project or a file '''
        dlg = page_dataimport.DistributionSelectWidget(singlecol=False, enablecorr=False, project=self.urisk.project,
                    coloptions=['Process Distribution', 'Test Distribution'])

        ok = dlg.exec_()
        if ok:
            self.block(True)
            self.mode.setCurrentIndex(self.mode.findText('Full'))
            self.changemode(replot=False)
            dists = dlg.get_dist()

            params = dists.get('Process Distribution', None)
            if params is not None:
                params.setdefault('median', params.get('mean', 0))
                self.dproc_table.set_disttype(initargs=params)
                self.dproc_table.valuechanged()
                self.chkProc.setChecked(True)

            params = dists.get('Test Distribution', None)
            if params is not None:
                params.setdefault('median', params.get('mean', 0))
                bias = params['median'] - params.get('expected', params['median'])
                params.setdefault('bias', bias)
                self.dtest_table.set_disttype(initargs=params)
                self.dtest_table.valuechanged()
                self.chkTest.setChecked(True)
            self.block(False)
            self.replot_and_update()
