''' User interface for Risk Analysis '''
import numpy as np
from PyQt5 import QtWidgets, QtGui
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from .. import risk
from . import gui_common
from . import gui_widgets
from . import page_data


class RiskWidget(QtWidgets.QWidget):
    ''' Widget for risk calculations '''
    def __init__(self, item, parent=None):
        super(RiskWidget, self).__init__(parent)
        assert isinstance(item, risk.UncertRisk)
        self.urisk = item
        self.urisk.calculate()  # With risk, calculate just creates an output object
        self.plotlines = {}  # Saved lines in plot
        self.limit_low = QtWidgets.QLineEdit(str(self.urisk.speclimits[0]))
        self.limit_hi = QtWidgets.QLineEdit(str(self.urisk.speclimits[1]))
        self.limit_low.setValidator(QtGui.QDoubleValidator(-1E99, 1E99, 15))
        self.limit_hi.setValidator(QtGui.QDoubleValidator(-1E99, 1E99, 15))
        self.chkProc = QtWidgets.QCheckBox('Process Distribution:')
        self.chkTest = QtWidgets.QCheckBox('Test Measurement:')
        self.gbl = QtWidgets.QLineEdit(str(self.urisk.guardband[0]))
        self.gbu = QtWidgets.QLineEdit(str(self.urisk.guardband[1]))
        self.gbl.setValidator(QtGui.QDoubleValidator(-1E99, 1E99, 15))
        self.gbu.setValidator(QtGui.QDoubleValidator(-1E99, 1E99, 15))
        self.chkGB = QtWidgets.QCheckBox('Guard Band')
        self.txtNotes = QtWidgets.QPlainTextEdit()
        self.fig = Figure()
        self.ax = None
        self.ax2 = None
        self.canvas = FigureCanvas(self.fig)
        self.txtOutput = gui_widgets.MarkdownTextEdit()

        if self.urisk.dist_proc is None:
            args = {'dist': 'normal', 'median': 0, 'std': 1}
        else:
            args = self.urisk.get_procdist_args()
        self.dist_proc = gui_widgets.DistributionEditTable(initargs=args)

        if self.urisk.dist_test is None:
            args = {'dist': 'normal', 'median': 0, 'std': .25}
        else:
            args = self.urisk.get_testdist_args()
        self.dist_test = gui_widgets.DistributionEditTable(initargs=args, locslider=True)

        if self.urisk.dist_test is None and self.urisk.dist_proc is None:
            self.urisk.set_test_dist(self.dist_test.statsdist)
            self.urisk.set_process_dist(self.dist_proc.statsdist)

        elif self.urisk.dist_test is None:
            self.urisk.set_test_dist(self.dist_test.statsdist)
            self.urisk.dist_test_saved = self.urisk.dist_test
            self.urisk.dist_test = None
            self.dist_test.setEnabled(False)
            self.chkGB.setEnabled(False)
            self.gbl.setEnabled(False)
            self.gbu.setEnabled(False)

        elif self.urisk.dist_proc is None:
            self.urisk.set_process_dist(self.dist_proc.statsdist)
            self.urisk.dist_proc_saved = self.urisk.dist_test
            self.urisk.dist_proc = None
            self.dist_proc.setEnabled(False)

        if self.urisk.dist_test is not None:
            self.chkTest.setChecked(True)
            if self.urisk.guardband[0] > 0 or self.urisk.guardband[1] > 0:
                self.chkGB.setChecked(True)
        self.chkProc.setChecked(self.urisk.dist_proc is not None)
        self.txtNotes.setPlainText(self.urisk.description)

        layout = QtWidgets.QHBoxLayout()
        vlayout = QtWidgets.QVBoxLayout()
        flayout = QtWidgets.QFormLayout()
        flayout.addRow('Lower Specification Limit', self.limit_low)
        flayout.addRow('Upper Specification Limit', self.limit_hi)
        vlayout.addLayout(flayout)
        vlayout.addWidget(self.chkProc)
        vlayout.addWidget(self.dist_proc)
        vlayout.addWidget(self.chkTest)
        vlayout.addWidget(self.dist_test)
        gblayout = QtWidgets.QFormLayout()
        gblayout.addRow('Lower Guard Band (relative):', self.gbl)
        gblayout.addRow('Upper Guard Band (relative):', self.gbu)
        vlayout.addWidget(self.chkGB)
        vlayout.addLayout(gblayout)
        vlayout.addWidget(QtWidgets.QLabel('Notes:'))
        vlayout.addWidget(self.txtNotes)
        rlayout = QtWidgets.QVBoxLayout()
        rlayout.addWidget(self.canvas, stretch=20)
        rlayout.addWidget(self.txtOutput, stretch=8)

        layout.addLayout(vlayout, stretch=1)
        layout.addLayout(rlayout, stretch=2.5)
        self.setLayout(layout)

        self.chkTest.stateChanged.connect(self.testprocclick)
        self.chkProc.stateChanged.connect(self.testprocclick)
        self.chkGB.stateChanged.connect(self.gbclick)
        self.dist_proc.changed.connect(self.replot_and_update)
        self.dist_test.changed.connect(self.replot_and_update)
        self.limit_low.editingFinished.connect(self.replot_and_update)
        self.limit_hi.editingFinished.connect(self.replot_and_update)
        self.gbl.editingFinished.connect(self.replot_and_update)
        self.gbu.editingFinished.connect(self.replot_and_update)
        self.txtNotes.textChanged.connect(self.update_description)
        self.initplot()
        self.replot_and_update()

        self.menu = QtWidgets.QMenu('Risk')
        self.actProcDist = QtWidgets.QAction('Import process distribution from...', self)
        self.actTestDist = QtWidgets.QAction('Import test distribution from...', self)
        self.actCalcGB = QtWidgets.QAction('Calculate guardband...', self)
        self.actSaveReport = QtWidgets.QAction('Save Report...', self)
        self.menu.addAction(self.actProcDist)
        self.menu.addAction(self.actTestDist)
        self.menu.addSeparator()
        self.menu.addAction(self.actCalcGB)
        self.menu.addSeparator()
        self.menu.addAction(self.actSaveReport)
        self.actProcDist.triggered.connect(self.importprocdist)
        self.actTestDist.triggered.connect(self.importtestdist)
        self.actCalcGB.triggered.connect(self.calc_guardband)
        self.actSaveReport.triggered.connect(self.save_report)

    def calculate(self):
        ''' Run calculation. Risk is calculated automatically, so this does nothing. '''
        pass

    def get_menu(self):
        ''' Get the menu for this widget '''
        return self.menu

    def update_description(self):
        ''' Description was updated, save it. '''
        self.urisk.description = self.txtNotes.toPlainText()

    def replot_and_update(self):
        ''' Replot and update the text fields '''
        if self.chkProc.isChecked():
            self.urisk.set_process_dist(self.dist_proc.statsdist)
        if self.chkTest.isChecked():
            self.urisk.set_test_dist(self.dist_test.statsdist)
        if self.chkGB.isChecked():
            self.urisk.set_guardband(float(self.gbl.text()), float(self.gbu.text()))
        else:
            self.urisk.set_guardband(0, 0)
        self.urisk.set_speclimits(float(self.limit_low.text()), float(self.limit_hi.text()))
        self.update_range()
        self.replot()
        self.update_labels()

    def initplot(self):
        ''' Initialize the plot '''
        self.axes = []  # List of axes (1, 2, or 3)
        nrows = self.chkTest.isChecked() + self.chkProc.isChecked()
        self.fig.clf()

        plotnum = 0
        if self.chkProc.isChecked():
            ax = self.fig.add_subplot(nrows, 1, plotnum+1)
            self.axes.append(ax)
            self.plotlines['procdist'], = ax.plot(0, 0, label='Process Distribution')
            self.plotlines['LL'] = ax.axvline(float(self.limit_low.text()), ls='--', color='C2', label='Specification Limits')
            self.plotlines['UL'] = ax.axvline(float(self.limit_hi.text()), ls='--', color='C2')
            ax.set_ylabel('Probability Density')
            ax.set_xlabel('Value')
            ax.legend(loc='upper left')
            plotnum += 1

        if self.chkTest.isChecked():
            ax = self.fig.add_subplot(nrows, 1, plotnum+1)
            self.axes.append(ax)
            self.plotlines['testdist'], = ax.plot(0, 0, color='C1', label='Test Distribution')
            self.plotlines['testmed'] = ax.axvline(0, ls='--', color='C1')
            if self.chkProc.isChecked():
                self.plotlines['product'], = ax.plot(0, 0, color='C4', label='Combined Distribution')
            self.plotlines['LL2'] = ax.axvline(float(self.limit_low.text()), ls='--', color='C2', label='Specification Limits')
            self.plotlines['UL2'] = ax.axvline(float(self.limit_hi.text()), ls='--', color='C2')
            if self.chkGB.isChecked():
                self.plotlines['GBL'] = ax.axvline(float(self.limit_low.text()) + float(self.gbl.text()), ls='-.', color='C3', label='Guard Band Limit')
                self.plotlines['GBU'] = ax.axvline(float(self.limit_hi.text()) - float(self.gbu.text()), ls='-.', color='C3')
            ax.set_ylabel('Probability Density')
            ax.set_xlabel('Value')
            ax.legend(loc='upper left')

    def replot(self):
        ''' Update the plot (without clearing completely) '''
        # Note GUI is not using urisk.output.plot_dists() so that it can work interactively without replotting full figure.
        LL = float(self.limit_low.text())
        UL = float(self.limit_hi.text())
        LL, UL = min(LL, UL), max(LL, UL)
        if self.chkGB.isChecked():
            GBL = float(self.gbl.text())
            GBU = float(self.gbu.text())
        else:
            GBL = 0
            GBU = 0

        xmin, xmax = self.get_range()
        x = np.linspace(xmax, xmin, num=300)
        yproc = self.dist_proc.statsdist.pdf(x)
        [ax.collections.clear() for ax in self.axes]  # Remove old fill-betweens

        plotnum = 0
        if self.chkProc.isChecked():
            self.plotlines['procdist'].set_data(x, yproc)
            self.axes[plotnum].fill_between(x, yproc, where=((x <= LL) | (x >= UL)), label='Nonconforming', alpha=.5, color='C0')
            self.plotlines['LL'].set_xdata([LL, LL])
            self.plotlines['UL'].set_xdata([UL, UL])
            plotnum += 1

        if self.chkTest.isChecked():
            ytest = self.dist_test.statsdist.pdf(x)
            self.plotlines['testdist'].set_data(x, ytest)
            self.plotlines['testmed'].set_xdata([self.dist_test.statsdist.median()]*2)
            self.plotlines['LL2'].set_xdata([LL, LL])
            self.plotlines['UL2'].set_xdata([UL, UL])
            median = self.dist_test.statsdist.median()

            if median > UL-GBU or median < LL+GBL:  # Shade PFR
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

    def update_labels(self):
        ''' Update label fields, recalculating risk values '''
        self.txtOutput.setMarkdown(self.urisk.out.report(**gui_common.get_rptargs()))

    def testprocclick(self):
        ''' Test Measurement or Process Distribution checkbox was clicked '''
        self.dist_test.setEnabled(self.chkTest.isChecked())
        self.dist_proc.setEnabled(self.chkProc.isChecked())
        if not self.chkTest.isChecked():
            self.chkGB.setChecked(False)

        # Save off distributions to restore later
        if not self.chkTest.isChecked() and self.urisk.dist_test is not None:
            self.urisk.dist_test_saved = self.urisk.dist_test
            self.urisk.dist_test = None
        elif self.chkTest.isChecked() and self.urisk.dist_test is None:
            self.urisk.dist_test = self.urisk.dist_test_saved

        if not self.chkProc.isChecked() and self.urisk.dist_proc is not None:
            self.urisk.dist_proc_saved = self.urisk.dist_proc
            self.urisk.dist_proc = None
        elif self.chkProc.isChecked() and self.urisk.dist_proc is None:
            self.urisk.dist_proc = self.urisk.dist_proc_saved

        self.chkGB.setEnabled(self.chkTest.isChecked())
        self.gbl.setEnabled(self.chkTest.isChecked() and self.chkGB.isChecked())
        self.gbu.setEnabled(self.chkTest.isChecked() and self.chkGB.isChecked())
        self.initplot()
        self.replot_and_update()

    def gbclick(self):
        ''' Guardband checkbox was clicked '''
        self.gbl.setEnabled(self.chkGB.isChecked())
        self.gbu.setEnabled(self.chkGB.isChecked())
        self.initplot()
        self.replot_and_update()

    def get_range(self):
        ''' Returns lower, upper limit to plot range '''
        LL = float(self.limit_low.text())
        UL = float(self.limit_hi.text())
        LL, UL = min(LL, UL), max(LL, UL)
        LLpad = self.dist_proc.statsdist.std()*3
        ULpad = self.dist_proc.statsdist.std()*3
        if self.chkTest.isChecked():
            LLpad = np.nanmax([LLpad, self.dist_test.statsdist.std()*5])
            ULpad = np.nanmax([LLpad, self.dist_test.statsdist.std()*5])

        LL -= LLpad
        UL += ULpad
        return LL, UL

    def update_range(self):
        ''' Update the range in the distribution table widget '''
        LL, UL = self.get_range()
        self.dist_test.set_locrange(LL, UL)

    def calc_guardband(self):
        ''' Determine guardband to hit specified PFA '''
        message = 'Enter Target Probability of False Accept (%)'
        if self.dist_proc.statsdist.dist.name != 'norm' or self.dist_test.statsdist.dist.name != 'norm':
            message += '\nNote this can take some time for non-normal distributions.'

        target, ok = QtWidgets.QInputDialog.getDouble(self, 'Calculate Guard Band', message, value=0.8, min=0, max=100)
        if ok:
            gb = risk.find_guardband(self.dist_proc.statsdist, self.dist_test.statsdist,
                                     float(self.limit_low.text()), float(self.limit_hi.text()),
                                     target/100, approx=True)
            if gb is not None:
                self.blockSignals(True)
                self.gbl.setText(format(gb, '.4g'))
                self.gbu.setText(format(gb, '.4g'))
                self.chkGB.setChecked(True)
                self.chkTest.setChecked(True)
                self.chkProc.setChecked(True)
                self.blockSignals(False)
                self.initplot()
            else:
                QtWidgets.QMessageBox.warning(self, 'Guard band calculation', 'Could not converge on guard band solution!')
            self.replot_and_update()

    def get_report(self):
        ''' Get full report of curve fit, using page settings '''
        return self.urisk.get_output().report_all(**gui_common.get_rptargs())

    def save_report(self):
        ''' Save full report, asking user for settings/filename '''
        gui_widgets.savemarkdown(self.get_report())

    def importprocdist(self):
        ''' Use process distribution from the project or a file '''
        dlg = page_data.DistributionSelectWidget(project=self.urisk.project)
        ok = dlg.exec_()
        if ok:
            distname, distargs, _ = dlg.get_dist()
            params = {'dist': distname}
            params.update(distargs)
            self.dist_proc.set_disttype(initargs=params)
            self.dist_proc.valuechanged()
            self.chkProc.setChecked(True)
            self.replot_and_update()

    def importtestdist(self):
        ''' Use test distribution from the project or a file '''
        dlg = page_data.DistributionSelectWidget(project=self.urisk.project)
        ok = dlg.exec_()
        if ok:
            distname, distargs, _ = dlg.get_dist()
            params = {'dist': distname}
            params.update(distargs)
            self.dist_test.set_disttype(initargs=params)
            self.dist_test.valuechanged()
            self.chkTest.setChecked(True)
            self.replot_and_update()
