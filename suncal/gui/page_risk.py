''' User interface for Risk Analysis '''
import numpy as np
from PyQt6 import QtWidgets, QtCore, QtGui
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

from ..project import ProjectRisk
from ..risk import risk
from . import gui_styles
from .gui_common import BlockedSignals
from . import widgets
from .page_guardband import GuardBandFinderWidget
from .help_strings import RiskHelp
from . import page_dataimport


class RiskWidget(QtWidgets.QWidget):
    ''' Widget for risk calculations '''

    change_help = QtCore.pyqtSignal()  # Tell main window to refresh help display

    def __init__(self, component, parent=None):
        super().__init__(parent)
        assert isinstance(component, ProjectRisk)
        self.component = component
        self.outputpage = QtWidgets.QComboBox()
        self.outputpage.addItems(['Risk',
                                  'Guardband sweep',
                                  'Probability of Conformance'])

        self.limits = widgets.DoubleLineEdit(-2, 2, 'Lower Specification Limit:', 'Upper Specification Limit:')
        self.btnSetItp = QtWidgets.QPushButton('Adjust ITP')
        self.guardband = widgets.DoubleLineEdit(0, 0, 'Lower Guardband (relative):', 'Upper Guardband (relative):')
        self.chkGB = QtWidgets.QCheckBox('Guardband')

        self.txtNotes = QtWidgets.QPlainTextEdit()
        self.fig = Figure()
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setStyleSheet("background-color:transparent;")
        self.toolbar = NavigationToolbar(self.canvas, self, coordinates=True)
        self.txtOutput = widgets.MarkdownTextEdit()

        if self.component.model.process_dist is None:
            procargs = {'dist': 'normal', 'median': 0, 'std': 1}
        else:
            procargs = self.component.model.get_procdist_args()

        if self.component.model.measure_dist is None:
            testargs = {'dist': 'normal', 'std': .25, 'bias': 0}
        else:
            testargs = self.component.model.get_testdist_args()
            testargs.update({'bias': self.component.model.testbias})
        self.dproc_table = widgets.DistributionEditTable(initargs=procargs)
        self.dtest_table = widgets.DistributionEditTable(initargs=testargs, locslider=True)
        self.limits.setValue(*self.component.model.speclimits)
        self.guardband.setValue(*self.component.model.gbofsts)
        self.txtNotes.setPlainText(self.component.description)
        self.tab = QtWidgets.QTabWidget()

        vlayout = QtWidgets.QVBoxLayout()
        flayout = QtWidgets.QFormLayout()
        flayout.addRow('Calculation:', self.outputpage)
        vlayout.addLayout(flayout)
        vlayout.addWidget(self.limits)
        proclayout = QtWidgets.QHBoxLayout()
        proclayout.addWidget(QtWidgets.QLabel('Process Distribution:'))
        proclayout.addStretch()
        proclayout.addWidget(self.btnSetItp)
        vlayout.addLayout(proclayout)
        vlayout.addWidget(self.dproc_table)
        vlayout.addWidget(QtWidgets.QLabel('Measurement Distribution:'))
        vlayout.addWidget(self.dtest_table)
        vlayout.addWidget(self.chkGB)
        vlayout.addWidget(self.guardband)
        vlayout.addStretch()
        rlayout = QtWidgets.QVBoxLayout()
        rlayout.addWidget(self.canvas, stretch=int(20))
        rlayout.addWidget(self.toolbar)
        self.topwidget = QtWidgets.QWidget()
        self.topwidget.setLayout(rlayout)
        self.rightsplitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        self.rightsplitter.addWidget(self.topwidget)
        self.rightsplitter.addWidget(self.txtOutput)
        self.splitter = QtWidgets.QSplitter()
        self.splitter.addWidget(self.tab)
        self.splitter.addWidget(self.rightsplitter)
        self.splitter.setCollapsible(0, False)
        self.splitter.setCollapsible(1, False)
        layout = QtWidgets.QHBoxLayout()
        layout.addWidget(self.splitter)
        self.setLayout(layout)

        setup = QtWidgets.QWidget()
        setup.setLayout(vlayout)
        self.tab.addTab(setup, 'Parameters')
        self.tab.addTab(self.txtNotes, 'Notes')

        self.btnSetItp.clicked.connect(self.setprocitp)
        self.chkGB.stateChanged.connect(self.gbclick)
        self.dproc_table.changed.connect(self.replot)
        self.dtest_table.changed.connect(self.replot)
        self.limits.editingFinished.connect(self.replot)
        self.guardband.editingFinished.connect(self.replot)
        self.txtNotes.textChanged.connect(self.update_description)
        self.outputpage.currentIndexChanged.connect(self.replot)

        self.menu = QtWidgets.QMenu('&Risk')
        self.actConditionalPFA = QtGui.QAction('Conditional PFA', self)
        self.actConditionalPFA.setCheckable(True)
        self.actShowJointPDF = QtGui.QAction('Plot Joint PDFs', self)
        self.actShowJointPDF.setCheckable(True)
        self.actShowJointPDF.setChecked(True)
        self.actImportDistProc = QtGui.QAction('Import process distribution...', self)
        self.actImportDistMeas = QtGui.QAction('Import measurement distribution...', self)
        self.actCalcGB = QtGui.QAction('Calculate guardband...', self)
        self.actSaveReport = QtGui.QAction('Save Report...', self)
        self.menu.addAction(self.actConditionalPFA)
        self.menu.addAction(self.actShowJointPDF)
        self.menu.addSeparator()
        self.menu.addAction(self.actImportDistProc)
        self.menu.addAction(self.actImportDistMeas)
        self.menu.addAction(self.actCalcGB)
        self.menu.addSeparator()
        self.menu.addAction(self.actSaveReport)
        self.actConditionalPFA.triggered.connect(self.update_report)
        self.actImportDistProc.triggered.connect(self.import_distproc)
        self.actImportDistMeas.triggered.connect(self.import_distmeas)
        self.actCalcGB.triggered.connect(self.calc_guardband)
        self.actSaveReport.triggered.connect(self.save_report)
        self.actShowJointPDF.triggered.connect(self.replot)
        self.replot()

    def calculate(self):
        ''' Run calculation. Risk is calculated automatically, so this does nothing. '''

    def get_menu(self):
        ''' Get the menu for this widget '''
        return self.menu

    def update_description(self):
        ''' Description was updated, save it. '''
        self.component.description = self.txtNotes.toPlainText()

    def update_proj_config(self):
        ''' Save page setup back to project item configuration '''
        # GUI updates model in real time - nothing to do

    def replot(self):
        ''' Replot and update the text fields '''
        with BlockedSignals(self):
            self.component.model.process_dist = self.dproc_table.statsdist
            self.component.model.measure_dist = self.dtest_table.statsdist
            self.component.model.testbias = self.dtest_table.distbias
            if self.chkGB.isChecked():
                self.component.model.gbofsts = self.guardband.getValue()
            else:
                self.component.model.gbofsts = (0, 0)
            self.component.model.speclimits = self.limits.getValue()

        result = self.component.calculate()

        if self.outputpage.currentText() == 'Guardband sweep':
            self.replot_gbsweep()
        elif self.outputpage.currentText() == 'Probability of Conformance':
            self.replot_probconform()
        else:
            self.update_range()
            if (self.actShowJointPDF.isChecked()
                    and self.component.model.process_dist is not None
                    and self.component.model.measure_dist is not None):
                result.report.plot.joint(self.fig)
                self.canvas.draw_idle()
            else:
                result.report.plot.distributions(self.fig)
                self.canvas.draw_idle()
        self.update_report()

    def update_report(self):
        ''' Update label fields, recalculating risk values '''
        conditional = self.actConditionalPFA.isChecked()
        self.txtOutput.setReport(self.component.result.report.summary(conditional=conditional))

    def replot_gbsweep(self):
        ''' Plot guardband sweep '''
        rpt = self.component.model.calc_guardband_sweep().report.summary(fig=self.fig)
        self.fig.tight_layout()
        self.canvas.draw_idle()
        self.txtOutput.setReport(rpt)

    def replot_probconform(self):
        ''' Plot probability of conformance given a test measurement result '''
        rpt = self.component.model.calc_probability_conformance().report.summary(fig=self.fig)
        self.canvas.draw_idle()
        self.txtOutput.setReport(rpt)

    def gbclick(self):
        ''' Guardband checkbox was clicked '''
        self.guardband.setEnabled(self.chkGB.isChecked())
        self.replot()

    def get_range(self):
        ''' Returns lower, upper limit to plot range '''
        LL, UL = self.limits.getValue()
        LL, UL = min(LL, UL), max(LL, UL)
        LL = np.nan if not np.isfinite(LL) else LL
        UL = np.nan if not np.isfinite(UL) else UL
        procmean = procstd = testmean = teststd = np.nan
        procmean = self.dproc_table.statsdist.mean()
        procstd = self.dproc_table.statsdist.std()
        testmean = self.dtest_table.statsdist.mean()
        teststd = self.dtest_table.statsdist.std()
        LL = np.nanmin([LL, procmean-procstd*4, testmean-teststd*4])
        UL = np.nanmax([UL, procmean+procstd*4, testmean+teststd*4])
        return LL, UL

    def update_range(self):
        ''' Update the range in the distribution table widget '''
        LL, UL = self.get_range()
        self.dtest_table.set_locrange(LL, UL)

    def calc_guardband(self):
        ''' Determine guardband to hit specified PFA '''
        if self.component.model.measure_dist is None:
            QtWidgets.QMessageBox.information(self, 'Suncal',
                                              'Please enable test distribution before finding guardband.')
            return

        dlg = GuardBandFinderWidget()
        if not np.isfinite(self.component.result.tur):
            dlg.tabTur.dobbert.setEnabled(False)
            dlg.tabTur.rss.setEnabled(False)
            dlg.tabTur.rp10.setEnabled(False)
            dlg.tabTur.test.setEnabled(False)
            dlg.tabTur.fourtoone.setEnabled(False)
            if self.component.model.process_dist is None:
                dlg.tabPfa.optConditional.setEnabled(False)
                dlg.tabPfa.optUnconditional.setEnabled(False)

        elif self.component.model.process_dist is None:
            dlg.tabPfa.optUnconditional.setEnabled(False)
            dlg.tabPfa.optConditional.setEnabled(False)
            dlg.tabTur.fourtoone.setEnabled(False)
            dlg.tabTur.dobbert.setChecked(True)

        ok = dlg.exec()
        if not ok:
            return

        with BlockedSignals(self):
            methodargs = dlg.get_method()
            method = methodargs['method']
            if method in ['pfa', 'cpfa']:
                self.component.model.guardband_pfa(
                    methodargs.get('pfa', .08),
                    conditional='c' in method,
                    optimizepfr=methodargs.get('optimize', False),
                    allow_negative=methodargs.get('allow_negative', False)
                )

            elif method == 'pfr':
                self.component.model.guardband_pfr(
                    methodargs.get('pfr', 2)
                )

            elif method in ['mincost', 'minimax']:
                self.component.model.guardband_cost(
                    method=method,
                    costfa=methodargs.get('facost', 100),
                    costfr=methodargs.get('frcost', 10)
                )

            elif method == 'specific':
                self.component.model.guardband_specific(methodargs.get('pfa', 0.08))

            else:  # TUR-based
                self.component.model.guardband_tur(method=method)

            self.guardband.setValue(*self.component.model.gbofsts)
            self.chkGB.setChecked(True)
            self.guardband.setEnabled(True)
            self.replot()

    def setprocitp(self):
        ''' Set the process distribution standard deviation to reach the
            input ITP value
        '''
        itp, ok = QtWidgets.QInputDialog.getDouble(
            self, 'Enter ITP', 'Adjust process distribution to result in this ITP value (as a percent from 1-99)',
            95, .1, 99.999999, 3)

        params = self.component.model.process_dist.argnames
        param, ok2 = QtWidgets.QInputDialog.getItem(
            self, 'Distribution Parameter', 'Select parameter to adjust',
            params, editable=False)

        if ok and ok2:
            itp /= 100
            pvalue = risk.get_sigmaproc_from_itp_arb(
                self.component.model.process_dist,
                param,
                itp,
                *self.component.model.speclimits)

            if pvalue is None:
                QtWidgets.QMessageBox.warning(self, 'ITP', f'No solution found for itp={itp*100:.2f}%. '
                                                           f'Try again after adjusting {param} manually to get closer to the '
                                                           'target ITP value.')
                return

            args = self.component.model.process_dist.kwds
            args[param] = pvalue
            args['dist'] = self.component.model.process_dist.dist.name
            self.dproc_table.set_disttype(args)
            self.dproc_table.valuechanged()

    def get_report(self):
        ''' Get full report of curve fit, using page settings '''
        return self.component.result.report.all()

    def save_report(self):
        ''' Save full report, asking user for settings/filename '''
        with gui_styles.LightPlotstyle():
            widgets.savereport(self.get_report())

    def import_distproc(self):
        ''' Import Process Distribution from another project component '''
        dlg = page_dataimport.DistributionSelectWidget(project=self.component.project)
        ok = dlg.exec()
        if ok:
            with BlockedSignals(self):
                self.dproc_table.set_disttype(initargs=dlg.distribution())
                self.dproc_table.valuechanged()
        self.replot()

    def import_distmeas(self):
        ''' Import Measurement Distribution from another project component '''
        dlg = page_dataimport.DistributionSelectWidget(project=self.component.project)
        ok = dlg.exec()
        if ok:
            with BlockedSignals(self):
                self.dtest_table.set_disttype(initargs=dlg.distribution())
                self.dtest_table.valuechanged()
        self.replot()

    def help_report(self):
        ''' Get the help report to display the current widget mode '''
        calctype = self.outputpage.currentText()
        if calctype == 'Guardband sweep':
            return RiskHelp.gb_sweep()
        elif calctype == 'Probability of Conformance':
            return RiskHelp.prob_conform()
        return RiskHelp.full()


if __name__ == '__main__':
    import sys
    app = QtWidgets.QApplication(sys.argv)
    main = RiskWidget(ProjectRisk())
    main.show()
    app.exec()
