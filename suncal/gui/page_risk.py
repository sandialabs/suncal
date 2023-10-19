''' User interface for Risk Analysis '''
from collections import namedtuple
import numpy as np
from PyQt5 import QtWidgets, QtCore
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

from ..project import ProjectRisk
from ..common import distributions
from ..risk import risk
from . import gui_common
from . import gui_widgets
from .page_guardband import GuardBandFinderWidget
from .help_strings import RiskHelp
from . import page_dataimport


class DoubleLineEdit(QtWidgets.QWidget):
    ''' Widget with two line edits for Doubles '''
    editingFinished = QtCore.pyqtSignal()

    def __init__(self, value1=0, value2=0, label1='', label2=''):
        super().__init__()
        self.line1 = QtWidgets.QLineEdit(str(value1))
        self.line2 = QtWidgets.QLineEdit(str(value2))
        self.line1.setValidator(gui_common.InfValidator())
        self.line2.setValidator(gui_common.InfValidator())
        layout = QtWidgets.QFormLayout()
        layout.addRow(label1, self.line1)
        layout.addRow(label2, self.line2)
        self.setLayout(layout)

        self.line1.editingFinished.connect(self.editingFinished)
        self.line2.editingFinished.connect(self.editingFinished)

    def getValue(self):
        ''' Return tuple value of two lines '''
        try:
            val1 = float(self.line1.text())
        except ValueError:
            val1 = 0
        try:
            val2 = float(self.line2.text())
        except ValueError:
            val2 = 0
        return val1, val2

    def setValue(self, value1, value2):
        ''' Set value of both lines '''
        self.line1.setText(f'{value1:.5g}')
        self.line2.setText(f'{value2:.5g}')


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
        self.measuredlabel = QtWidgets.QLabel('Test Measurement:')

        self.tur.valueChanged.connect(self.editingFinished)
        self.gbfactor.valueChanged.connect(self.editingFinished)
        self.itp.valueChanged.connect(self.editingFinished)
        self.measured.valueChanged.connect(self.editingFinished)

        layout = QtWidgets.QFormLayout()
        layout.addRow('Test Uncertainty Ratio:', self.tur)
        hlayout = QtWidgets.QHBoxLayout()
        hlayout.addWidget(self.itp)
        layout.addRow('In-tolerance probability:', hlayout)
        layout.addRow('Guardband factor:', self.gbfactor)
        layout.addRow(self.measuredlabel, self.measured)
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


class SweepWidget(QtWidgets.QGroupBox):
    ''' Widget with PFA sweep controls '''
    def __init__(self, parent=None):
        super().__init__('Sweep Setup', parent=parent)
        items = ['Sweep (x) variable', 'Step (z) variable', 'Constant']
        self.itpvssigma = QtWidgets.QComboBox()
        self.itpvssigma.addItems(['In-tol probability %', 'SL/Process Std. Dev.'])
        self.itp = QtWidgets.QComboBox()
        self.tur = QtWidgets.QComboBox()
        self.gbf = QtWidgets.QComboBox()
        self.procbias = QtWidgets.QComboBox()
        self.testbias = QtWidgets.QComboBox()
        self.itp.addItems(items)
        self.tur.addItems(items)
        self.procbias.addItems(items)
        self.testbias.addItems(items)
        self.gbf.addItems(items + ['RDS', 'Dobbert', 'RP10', '95% Test'])
        self.tur.setCurrentIndex(1)  # Z
        self.gbf.setCurrentIndex(2)  # Fixed
        self.procbias.setCurrentIndex(2)  # Fixed
        self.testbias.setCurrentIndex(2)  # Fixed
        self.itpval = gui_widgets.FloatLineEdit('90')
        self.turval = gui_widgets.FloatLineEdit('4')
        self.gbfval = gui_widgets.FloatLineEdit('1')
        self.procbiasval = gui_widgets.FloatLineEdit('0')
        self.testbiasval = gui_widgets.FloatLineEdit('0')

        self.xstart = gui_widgets.FloatLineEdit('50')
        self.xstop = gui_widgets.FloatLineEdit('90')
        self.xpts = gui_widgets.FloatLineEdit('20', low=2)
        self.zvals = QtWidgets.QLineEdit()
        self.zvals.setText('1.5, 2, 3, 4')
        self.itpval.setVisible(False)
        self.turval.setVisible(False)
        self.plot3d = QtWidgets.QCheckBox('3D Plot')
        self.logy = QtWidgets.QCheckBox('Log Scale')
        self.plottype = QtWidgets.QComboBox()
        self.plottype.addItems(['PFA', 'PFR', 'Both'])
        self.btnrefresh = QtWidgets.QPushButton('Replot')
        layout = QtWidgets.QGridLayout()
        layout.addWidget(self.itpvssigma, 0, 0)
        layout.addWidget(self.itp, 0, 1)
        layout.addWidget(self.itpval, 0, 2)
        layout.addWidget(QtWidgets.QLabel('Test uncertainty ratio'), 1, 0)
        layout.addWidget(self.tur, 1, 1)
        layout.addWidget(self.turval, 1, 2)
        layout.addWidget(QtWidgets.QLabel('Guardband factor'), 2, 0)
        layout.addWidget(self.gbf, 2, 1)
        layout.addWidget(self.gbfval, 2, 2)
        layout.addWidget(QtWidgets.QLabel('Process Bias %'), 3, 0)
        layout.addWidget(self.procbias, 3, 1)
        layout.addWidget(self.procbiasval, 3, 2)
        layout.addWidget(QtWidgets.QLabel('Measurement Bias %'), 4, 0)
        layout.addWidget(self.testbias, 4, 1)
        layout.addWidget(self.testbiasval, 4, 2)
        layout.addWidget(gui_widgets.QHLine(), 5, 0, 1, 3)
        layout.addWidget(QtWidgets.QLabel('Sweep (x) Start:'), 6, 0)
        layout.addWidget(self.xstart, 6, 1, 1, 2)
        layout.addWidget(QtWidgets.QLabel('Sweep (x) Stop:'), 7, 0)
        layout.addWidget(self.xstop, 7, 1, 1, 2)
        layout.addWidget(QtWidgets.QLabel('# Points (x):'), 8, 0)
        layout.addWidget(self.xpts, 8, 1, 1, 2)
        layout.addWidget(QtWidgets.QLabel('Step Values (z):'), 9, 0)
        layout.addWidget(self.zvals, 9, 1, 1, 2)
        layout.addWidget(gui_widgets.QHLine(), 10, 0, 1, 3)
        layout.addWidget(QtWidgets.QLabel('Plot'), 11, 0)
        layout.addWidget(self.plottype, 11, 1)
        layout.addWidget(self.plot3d, 12, 1, 1, 2)
        layout.addWidget(self.logy, 13, 1, 1, 2)
        layout.addWidget(self.btnrefresh, 14, 1)
        self.setLayout(layout)

        self.itp.currentIndexChanged.connect(lambda i, x=self.itp: self.cmbchange(x))
        self.tur.currentIndexChanged.connect(lambda i, x=self.tur: self.cmbchange(x))
        self.gbf.currentIndexChanged.connect(lambda i, x=self.gbf: self.cmbchange(x))
        self.procbias.currentIndexChanged.connect(lambda i, x=self.procbias: self.cmbchange(x))
        self.testbias.currentIndexChanged.connect(lambda i, x=self.testbias: self.cmbchange(x))

    def cmbchange(self, boxchanged):
        ''' Combobox changed. Ensure only one sweep/step is selected,
            and enable constant fields
        '''
        # Check the box that was just changed first!
        boxes = [self.itp, self.tur, self.gbf, self.procbias, self.testbias]
        consts = [self.itpval, self.turval, self.gbfval, self.procbiasval, self.testbiasval]
        boxidx = boxes.index(boxchanged)
        boxes.pop(boxidx)
        c = consts.pop(boxidx)
        boxes.insert(0, boxchanged)
        consts.insert(0, c)

        havex = False
        havez = False
        for box, const in zip(boxes, consts):
            if 'Sweep' in box.currentText():
                if havex:
                    box.setCurrentIndex(2)  # Already have a sweep, set this to const
                const.setVisible(havex)  # enabled or visible??
                havex = True
            elif 'Step' in box.currentText():
                if havez:
                    box.setCurrentIndex(2)
                const.setVisible(havez)
                havez = True
            elif 'Constant' not in box.currentText():
                const.setVisible(False)
            else:
                const.setVisible(True)

    def get_sweepvals(self):
        # Convert ALL variables into lists the same length.
        # Some will have all identical values
        xvals = np.linspace(float(self.xstart.text()), float(self.xstop.text()), num=int(self.xpts.text()))
        try:
            zvals = np.array([float(z) for z in self.zvals.text().split(',')])
        except ValueError:
            QtWidgets.QMessageBox.warning(self, 'Risk Sweep', 'Step values must be entered as comma-separated list.')
            return None

        # Defaults if not being swept
        sig0 = None
        itpval = float(self.itpval.text()) / 100  # Percent
        if 'SL' in self.itpvssigma.currentText():
            sig0 = float(self.itpval.text())
        turval = float(self.turval.text())
        pbias = float(self.procbiasval.text()) / 100
        tbias = float(self.testbiasval.text()) / 100
        if self.gbf.currentText() == 'Constant':
            gbfval = float(self.gbfval.text())
        else:
            gbfval = self.gbf.currentText().lower()
            gbfval = 'test' if 'test' in gbfval else gbfval

        if 'Step' in self.itp.currentText() and 'In-' in self.itpvssigma.currentText():
            zvar = 'itp'
        elif 'Step' in self.itp.currentText():
            zvar = 'sig0'
        elif 'Step' in self.tur.currentText():
            zvar = 'tur'
        elif 'Step' in self.gbf.currentText():
            zvar = 'gbf'
        elif 'Step' in self.procbias.currentText():
            zvar = 'pbias'
        elif 'Step' in self.testbias.currentText():
            zvar = 'tbias'
        else:
            zvar = 'none'
            zvals = [None]  # Need one item to loop

        if 'Sweep' in self.itp.currentText() and 'In-' in self.itpvssigma.currentText():
            xvar = 'itp'
        elif 'Sweep' in self.itp.currentText():
            xvar = 'sig0'
        elif 'Sweep' in self.tur.currentText():
            xvar = 'tur'
        elif 'Sweep' in self.gbf.currentText():
            xvar = 'gbf'
        elif 'Sweep' in self.procbias.currentText():
            xvar = 'pbias'
        elif 'Sweep' in self.testbias.currentText():
            xvar = 'tbias'
        else:
            QtWidgets.QMessageBox.warning(self, 'Risk Sweep', 'Please select a variable to sweep.')
            return None

        # Convert percent to decimal 0-1
        if xvar in ['itp', 'tbias', 'pbias']:
            xvals = xvals / 100
        if zvar in ['itp', 'tbias', 'pbias']:
            zvals = zvals / 100

        threed = self.plot3d.isChecked()
        y = self.plottype.currentText()
        logy = self.logy.isChecked()
        SweepSetup = namedtuple('SweepSetup', ['x', 'z', 'xvals', 'zvals', 'itp', 'tur', 'gbf', 'sig0',
                                               'pbias', 'tbias', 'threed', 'y', 'logy'])
        return SweepSetup(xvar, zvar, xvals, zvals, itpval, turval, gbfval, sig0, pbias, tbias, threed, y, logy)


class RiskWidget(QtWidgets.QWidget):
    ''' Widget for risk calculations '''

    change_help = QtCore.pyqtSignal()  # Tell main window to refresh help display

    def __init__(self, projitem, parent=None):
        super().__init__(parent)
        assert isinstance(projitem, ProjectRisk)
        self.projitem = projitem
        self.projitem.calculate()  # With risk, calculate just creates an output object
        self.plotlines = {}  # Saved lines in plot
        self.mode = QtWidgets.QComboBox()
        self.mode.addItems(['Simple', 'Full'])
        self.calctype = QtWidgets.QComboBox()
        self.calctype.addItems(['Integral', 'Monte Carlo', 'Guardband sweep',
                                'Probability of Conformance', 'Risk Curves'])

        self.simple = SimpleRiskWidget()
        self.limits = DoubleLineEdit(-2, 2, 'Lower Specification Limit:', 'Upper Specification Limit:')
        self.sweepsetup = SweepWidget()
        self.chkProc = QtWidgets.QCheckBox('Process Distribution:')
        self.btnSetItp = QtWidgets.QPushButton('Set ITP')
        self.chkTest = QtWidgets.QCheckBox('Test Measurement:')
        self.guardband = DoubleLineEdit(0, 0, 'Lower Guardband (relative):', 'Upper Guardband (relative):')
        self.chkGB = QtWidgets.QCheckBox('Guardband')

        self.limits.setVisible(False)
        self.sweepsetup.setVisible(False)
        self.chkProc.setVisible(False)
        self.btnSetItp.setVisible(False)
        self.chkTest.setVisible(False)
        self.guardband.setVisible(False)
        self.chkGB.setVisible(False)

        self.txtNotes = QtWidgets.QPlainTextEdit()
        self.fig = Figure()
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setStyleSheet("background-color:transparent;")
        self.toolbar = NavigationToolbar(self.canvas, self, coordinates=True)
        self.txtOutput = gui_widgets.MarkdownTextEdit()

        if self.projitem.model.is_simple():
            self.mode.setCurrentIndex(0)
            self.simple.tur.setValue(self.projitem.model.get_tur())
            self.simple.gbfactor.setValue(self.projitem.model.get_gbf())
            self.simple.itp.setValue(self.projitem.model.get_itp()*100)
            procargs = self.projitem.model.get_procdist_args()
            testargs = self.projitem.model.get_testdist_args()
            testargs.update({'bias': self.projitem.model.testbias})
            self.dproc_table = gui_widgets.DistributionEditTable(initargs=procargs)
            self.dtest_table = gui_widgets.DistributionEditTable(initargs=testargs, locslider=True)
            self.chkProc.setChecked(True)
            self.chkTest.setChecked(True)

        else:  # Full Risk
            self.mode.setCurrentIndex(1)
            if self.projitem.model.procdist is None:
                procargs = {'dist': 'normal', 'median': 0, 'std': 1}
            else:
                procargs = self.projitem.model.get_procdist_args()

            if self.projitem.model.testdist is None:
                testargs = {'dist': 'normal', 'std': .25, 'bias': 0}
            else:
                testargs = self.projitem.model.get_testdist_args()
                testargs.update({'bias': self.projitem.model.testbias})
            self.dproc_table = gui_widgets.DistributionEditTable(initargs=procargs)
            self.dtest_table = gui_widgets.DistributionEditTable(initargs=testargs, locslider=True)
            self.limits.setValue(*self.projitem.model.speclimits)
            self.guardband.setValue(*self.projitem.model.gbofsts)

            if self.projitem.model.testdist is None and self.projitem.model.procdist is None:
                self.projitem.model.testdist = distributions.get_distribution('normal', std=0.125)
                self.projitem.model.procdist = distributions.get_distribution('normal', std=1)

            elif self.projitem.model.testdist is None:
                self.projitem.model.testdist = distributions.get_distribution('normal', std=0.125)
                self.projitem.model.testdist_saved = self.projitem.model.testdist
                self.projitem.model.testdist = None
                self.dtest_table.setEnabled(False)
                self.chkGB.setEnabled(False)
                self.guardband.setEnabled(False)

            elif self.projitem.model.procdist is None:
                self.projitem.model.procdist = distributions.get_distribution('normal', std=1)
                self.projitem.model.procdist_saved = self.projitem.model.procdist
                self.projitem.model.procdist = None
                self.dproc_table.setEnabled(False)

            if self.projitem.model.testdist is not None:
                self.chkTest.setChecked(True)
                if self.projitem.model.gbofsts[0] > 0 or self.projitem.model.gbofsts[1] > 0:
                    self.chkGB.setChecked(True)

        self.chkProc.setChecked(self.projitem.model.procdist is not None)
        self.txtNotes.setPlainText(self.projitem.description)
        self.tab = QtWidgets.QTabWidget()

        vlayout = QtWidgets.QVBoxLayout()
        flayout = QtWidgets.QFormLayout()
        flayout.addRow('Mode:', self.mode)
        flayout.addRow('Calculation:', self.calctype)
        vlayout.addLayout(flayout)
        vlayout.addWidget(self.simple)
        vlayout.addWidget(self.limits)
        proclayout = QtWidgets.QHBoxLayout()
        proclayout.addWidget(self.chkProc)
        proclayout.addStretch()
        proclayout.addWidget(self.btnSetItp)
        vlayout.addLayout(proclayout)
        vlayout.addWidget(self.dproc_table)
        vlayout.addWidget(self.chkTest)
        vlayout.addWidget(self.dtest_table)
        vlayout.addWidget(self.chkGB)
        vlayout.addWidget(self.guardband)
        vlayout.addWidget(self.sweepsetup)
        vlayout.addStretch()
        rlayout = QtWidgets.QVBoxLayout()
        rlayout.addWidget(self.canvas, stretch=int(20))
        rlayout.addWidget(self.toolbar)
        self.topwidget = QtWidgets.QWidget()
        self.topwidget.setLayout(rlayout)
        self.rightsplitter = QtWidgets.QSplitter(QtCore.Qt.Vertical)
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

        self.mode.currentIndexChanged.connect(self.changemode)
        self.simple.editingFinished.connect(self.entry_changed)
        self.chkTest.stateChanged.connect(self.testprocclick)
        self.chkProc.stateChanged.connect(self.testprocclick)
        self.btnSetItp.clicked.connect(self.setprocitp)
        self.chkGB.stateChanged.connect(self.gbclick)
        self.dproc_table.changed.connect(self.entry_changed)
        self.dtest_table.changed.connect(self.entry_changed)
        self.limits.editingFinished.connect(self.entry_changed)
        self.guardband.editingFinished.connect(self.entry_changed)
        self.txtNotes.textChanged.connect(self.update_description)
        self.calctype.currentIndexChanged.connect(self.changecalc)
        self.sweepsetup.btnrefresh.clicked.connect(self.replot)

        self.menu = QtWidgets.QMenu('Risk')
        self.actConditionalPFA = QtWidgets.QAction('Conditional PFA', self)
        self.actConditionalPFA.setCheckable(True)
        self.actShowJointPDF = QtWidgets.QAction('Plot Joint PDFs', self)
        self.actShowJointPDF.setCheckable(True)
        self.actShowJointPDF.setChecked(True)
        self.actImportDist = QtWidgets.QAction('Import distribution...', self)
        self.actCalcGB = QtWidgets.QAction('Calculate guardband...', self)
        self.actSaveReport = QtWidgets.QAction('Save Report...', self)
        self.menu.addAction(self.actConditionalPFA)
        self.menu.addAction(self.actShowJointPDF)
        self.menu.addSeparator()
        self.menu.addAction(self.actImportDist)
        self.menu.addAction(self.actCalcGB)
        self.menu.addSeparator()
        self.menu.addAction(self.actSaveReport)
        self.actConditionalPFA.triggered.connect(self.update_report)
        self.actImportDist.triggered.connect(self.importdist)
        self.actCalcGB.triggered.connect(self.calc_guardband)
        self.actSaveReport.triggered.connect(self.save_report)
        self.actShowJointPDF.triggered.connect(self.changemode)
        self.changemode()  # Show/hide controls

    def calculate(self):
        ''' Run calculation. Risk is calculated automatically, so this does nothing. '''

    def get_menu(self):
        ''' Get the menu for this widget '''
        return self.menu

    def update_description(self):
        ''' Description was updated, save it. '''
        self.projitem.description = self.txtNotes.toPlainText()

    def update_proj_config(self):
        ''' Save page setup back to project item configuration '''
        pass  # GUI updates model in real time

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

    def changemode(self, idx=0, replot=True):
        ''' Mode changed (simple to full) '''
        self.block(True)
        simple = self.mode.currentText() == 'Simple'
        if simple and not self.projitem.model.is_simple():
            self.projitem.model.to_simple()
            self.simple.tur.setValue(self.projitem.model.get_tur())
            self.simple.gbfactor.setValue(self.projitem.model.get_gbf())
            self.simple.itp.setValue(self.projitem.model.get_itp()*100)
            self.chkProc.setChecked(True)
            self.chkTest.setChecked(True)
            self.chkProc.setEnabled(True)
            self.chkTest.setEnabled(True)
            self.dproc_table.set_disttype(self.projitem.model.get_procdist_args())
            self.dtest_table.set_disttype(self.projitem.model.get_testdist_args())
            self.dproc_table.valuechanged()
            self.dtest_table.valuechanged()

        elif not simple and self.projitem.model.is_simple():
            self.limits.setValue(*[np.round(x, 3) for x in self.projitem.model.speclimits])
            self.guardband.setValue(*[np.round(x, 3) for x in self.projitem.model.gbofsts])
            self.dproc_table.set_disttype(self.projitem.model.get_procdist_args())
            self.dtest_table.set_disttype(self.projitem.model.get_testdist_args())
            self.dproc_table.valuechanged()
            self.dtest_table.valuechanged()

        self.simple.setVisible(simple)
        self.limits.setVisible(not simple)
        self.chkProc.setVisible(not simple)
        self.btnSetItp.setVisible(not simple)
        self.chkTest.setVisible(not simple)
        self.dproc_table.setVisible(not simple)
        self.dtest_table.setVisible(not simple)
        self.chkGB.setVisible(not simple)
        self.guardband.setVisible(not simple)
        self.block(False)
        if replot:
            self.replot()
        self.change_help.emit()

    def changecalc(self):
        ''' Change calculation mode (integration vs monte carlo vs gbsweep) '''
        self.block(True)
        simple = self.mode.currentText() == 'Simple'
        if self.calctype.currentText() == 'Monte Carlo':
            self.mode.setEnabled(True)
            self.chkProc.setChecked(True)
            self.chkTest.setChecked(True)
            self.chkProc.setEnabled(False)
            self.btnSetItp.setEnabled(False)
            self.chkTest.setEnabled(False)
            self.guardband.setEnabled(True)
            self.chkGB.setEnabled(True)
            self.simple.setVisible(simple)
            self.simple.gbfactor.setEnabled(True)
            self.simple.measured.setVisible(False)
            self.simple.measuredlabel.setVisible(False)
            self.sweepsetup.setVisible(False)
        elif self.calctype.currentText() == 'Guardband sweep':
            self.mode.setEnabled(True)
            self.chkTest.setEnabled(True)
            self.chkProc.setEnabled(True)
            self.btnSetItp.setEnabled(True)
            self.guardband.setEnabled(False)
            self.chkGB.setEnabled(False)
            self.simple.setVisible(simple)
            self.simple.gbfactor.setEnabled(False)
            self.simple.measured.setVisible(False)
            self.simple.measuredlabel.setVisible(False)
            self.sweepsetup.setVisible(False)
        elif self.calctype.currentText() == 'Risk Curves':
            self.mode.setEnabled(False)
            self.mode.setCurrentIndex(0)
            self.changemode(replot=False)
            self.chkTest.setEnabled(True)
            self.chkProc.setEnabled(True)
            self.btnSetItp.setEnabled(True)
            self.guardband.setEnabled(True)
            self.chkGB.setEnabled(False)
            self.simple.setVisible(False)
            self.sweepsetup.setVisible(True)
        else:
            self.mode.setEnabled(True)
            self.chkTest.setEnabled(True)
            self.chkProc.setEnabled(True)
            self.btnSetItp.setEnabled(True)
            self.guardband.setEnabled(True)
            self.chkGB.setEnabled(True)
            self.simple.setVisible(simple)
            self.simple.gbfactor.setEnabled(True)
            self.simple.measured.setVisible(True)
            self.simple.measuredlabel.setVisible(True)
            self.sweepsetup.setVisible(False)
            self.testprocclick()
        self.block(False)
        self.replot()
        self.change_help.emit()

    def entry_changed(self):
        ''' An entry changed. Trigger replot except in curves mode '''
        if self.calctype.currentText() != 'Risk Curves':
            self.replot()
        # Curves mode triggers replot directly with button

    def replot(self):
        ''' Replot and update the text fields '''
        self.block(True)
        if self.mode.currentText() == 'Simple':
            if ((self.simple.get_gbf() != 1.0 and self.projitem.model.get_gbf() == 1.0) or
               (self.simple.get_gbf() == 1.0 and self.projitem.model.get_gbf() != 1.0)):
                self.chkGB.setChecked(self.simple.get_gbf() != 1.0)
            self.projitem.model.set_itp(self.simple.get_itp())
            self.projitem.model.set_tur(self.simple.get_tur())
            self.projitem.model.set_gbf(self.simple.get_gbf())
            self.projitem.model.set_testmedian(self.simple.get_measured_fraction() * self.get_range()[1])

        else:
            if self.chkProc.isChecked():
                self.projitem.model.procdist = self.dproc_table.statsdist
            if self.chkTest.isChecked():
                self.projitem.model.testdist = self.dtest_table.statsdist
                self.projitem.model.testbias = self.dtest_table.distbias
            if self.chkGB.isChecked():
                self.projitem.model.gbofsts = self.guardband.getValue()
            else:
                self.projitem.model.gbofsts = (0, 0)
            self.projitem.model.speclimits = self.limits.getValue()

        if self.calctype.currentText() == 'Monte Carlo':
            self.replot_mc()
        elif self.calctype.currentText() == 'Guardband sweep':
            self.replot_gbsweep()
        elif self.calctype.currentText() == 'Probability of Conformance':
            self.replot_probconform()
        elif self.calctype.currentText() == 'Risk Curves':
            self.replot_sweep()
        else:
            self.update_range()
            if (self.actShowJointPDF.isChecked()
               and self.projitem.model.procdist is not None
               and self.projitem.model.testdist is not None):
                self.projitem.result.report.plot.joint(self.fig)
                self.canvas.draw_idle()
            else:
                self.projitem.result.report.plot.distributions(self.fig)
                self.canvas.draw_idle()
            self.update_report()
        self.block(False)

    def update_report(self):
        ''' Update label fields, recalculating risk values '''
        conditional = self.actConditionalPFA.isChecked()
        self.txtOutput.setReport(self.projitem.result.report.summary(conditional=conditional))

    def replot_mc(self):
        ''' Replot/report monte carlo method '''
        rpt = self.projitem.result.report.montecarlo(fig=self.fig)
        self.fig.tight_layout()
        self.canvas.draw_idle()
        self.txtOutput.setReport(rpt)

    def replot_gbsweep(self):
        ''' Plot guardband sweep '''
        rpt = self.projitem.result.report.guardband_sweep(fig=self.fig)
        self.fig.tight_layout()
        self.canvas.draw_idle()
        self.txtOutput.setReport(rpt)

    def replot_probconform(self):
        ''' Plot probability of conformance given a test measurement result '''
        rpt = self.projitem.result.report.probconform(fig=self.fig)
        self.fig.tight_layout()
        self.canvas.draw_idle()
        self.txtOutput.setReport(rpt)

    def replot_sweep(self):
        ''' Plot generic PFA(R) sweep '''
        setup = self.sweepsetup.get_sweepvals()
        if setup is None:
            return  # No sweep variable
        self.projitem.model.set_itp(setup.itp)  # Store defaults
        self.projitem.model.set_tur(setup.tur)
        rpt = self.projitem.result.report.sweep(fig=self.fig, xvar=setup.x, zvar=setup.z, xvals=setup.xvals,
                                                zvals=setup.zvals, yvar=setup.y, threed=setup.threed,
                                                gbf=setup.gbf, sig0=setup.sig0, tbias=setup.tbias,
                                                pbias=setup.pbias, logy=setup.logy)
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
        if not self.chkTest.isChecked() and self.projitem.model.testdist is not None:
            self.projitem.model.testdist_saved = self.projitem.model.testdist
            self.projitem.model.testdist = None
        elif self.chkTest.isChecked() and self.projitem.model.testdist is None:
            self.projitem.model.testdist = self.projitem.model.testdist_saved

        if not self.chkProc.isChecked() and self.projitem.model.procdist is not None:
            self.projitem.model.procdist_saved = self.projitem.model.procdist
            self.projitem.model.procdist = None
        elif self.chkProc.isChecked() and self.projitem.model.procdist is None:
            self.projitem.model.procdist = self.projitem.model.procdist_saved

        self.chkGB.setEnabled(self.chkTest.isChecked())
        self.guardband.setEnabled(self.chkTest.isChecked() and self.chkGB.isChecked())
        self.entry_changed()

    def gbclick(self):
        ''' Guardband checkbox was clicked '''
        self.guardband.setEnabled(self.chkGB.isChecked())
        self.entry_changed()

    def get_range(self):
        ''' Returns lower, upper limit to plot range '''
        LL, UL = self.limits.getValue()
        LL, UL = min(LL, UL), max(LL, UL)
        LL = np.nan if not np.isfinite(LL) else LL
        UL = np.nan if not np.isfinite(UL) else UL
        procmean = procstd = testmean = teststd = np.nan
        if self.chkProc.isChecked():
            procmean = self.dproc_table.statsdist.mean()
            procstd = self.dproc_table.statsdist.std()
        if self.chkTest.isChecked():
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
        simple = self.mode.currentText() == 'Simple'
        if self.projitem.model.testdist is None:
            QtWidgets.QMessageBox.information(self, 'Suncal',
                                              'Please enable test distribution before finding guardband.')
            return

        dlg = GuardBandFinderWidget()
        if not np.isfinite(self.projitem.model.get_tur()):
            dlg.tabTur.dobbert.setEnabled(False)
            dlg.tabTur.rss.setEnabled(False)
            dlg.tabTur.rp10.setEnabled(False)
            dlg.tabTur.test.setEnabled(False)
            dlg.tabTur.fourtoone.setEnabled(False)
            dlg.tabCost.optMincost.setEnabled(False)
            dlg.tabCost.optMinimax.setEnabled(False)
            if self.projitem.model.procdist is None:
                dlg.tabPfa.optConditional.setEnabled(False)
                dlg.tabPfa.optUnconditional.setEnabled(False)

        elif self.projitem.model.procdist is None:
            dlg.tabPfa.optUnconditional.setEnabled(False)
            dlg.tabPfa.optConditional.setEnabled(False)
            dlg.tabTur.fourtoone.setEnabled(False)
            dlg.tabCost.optMincost.setEnabled(False)
            dlg.tabCost.optMinimax.setEnabled(False)
            dlg.tabTur.dobbert.setChecked(True)

        ok = dlg.exec()
        if ok:
            self.block(True)
            methodargs = dlg.get_method()
            method = methodargs['method']
            if method in ['pfa', 'cpfa']:
                self.projitem.model.guardband_pfa(
                    methodargs.get('pfa', .08),
                    conditional = 'c' in method,
                    optimizepfr = methodargs.get('optimize', False),
                    allow_negative = methodargs.get('allow_negative', False)
                )

            elif method in ['mincost', 'minimax']:
                self.projitem.model.guardband_cost(
                    method=method,
                    costfa=methodargs.get('costfa', 100),
                    costfr=methodargs.get('costfr', 10)
                )

            elif method == 'specific':
                self.projitem.model.guardband_specific(methodargs.get('pfa', 0.08))

            else:  # TUR-based
                self.projitem.model.guardband_tur(method=method)

            if simple:
                self.simple.gbfactor.setValue(self.projitem.model.get_gbf())
                self.chkGB.setChecked(True)
            else:
                self.guardband.setValue(*self.projitem.model.gbofsts)
                self.chkGB.setChecked(True)
            self.guardband.setEnabled(True)

            self.block(False)
            self.replot()

    def setprocitp(self):
        ''' Set the process distribution standard deviation to reach the
            input ITP value
        '''
        itp, ok = QtWidgets.QInputDialog.getDouble(
            self, 'Enter ITP', 'Adjust process distribution to result in this ITP value (as a percent from 1-99)',
            95, .1, 99.999999, 3)

        if self.projitem.model.procdist.dist.name != 'norm':
            params = self.projitem.model.procdist.argnames
            param, ok2 = QtWidgets.QInputDialog.getItem(
                self, 'Distribution Parameter', 'Select parameter to adjust',
                params, editable=False)
            if ok and ok2:
                itp /= 100
                pvalue = risk.get_sigmaproc_from_itp_arb(
                    self.projitem.model.procdist,
                    param,
                    itp,
                    *self.projitem.model.speclimits)

                if pvalue is None:
                    QtWidgets.QMessageBox.warning(self, 'ITP', f'No solution found for itp={itp*100:.2f}%. '
                                                  f'Try again after adjusting {param} manually to get closer to the '
                                                  'target ITP value.')
                    return

                args = self.projitem.model.procdist.kwds
                args[param] = pvalue
                args['dist'] = self.projitem.model.procdist.dist.name
                self.dproc_table.set_disttype(args)
                self.dproc_table.valuechanged()

        else:
            if ok:
                itp /= 100  # percent to decimal
                procmean = self.dproc_table.statsdist.mean()
                UL, LL = self.projitem.model.speclimits
                nominal = (LL + UL)/2
                bias = procmean - nominal
                sigma = risk.get_sigmaproc_from_itp(itp, bias)

                args = {'dist': 'normal',
                        'median': procmean,
                        'std': sigma}
                self.dproc_table.set_disttype(args)
                self.dproc_table.valuechanged()

    def get_report(self):
        ''' Get full report of curve fit, using page settings '''
        mc = self.calctype.currentText() == 'Monte Carlo'
        return self.projitem.result.report.all(mc=mc)

    def save_report(self):
        ''' Save full report, asking user for settings/filename '''
        gui_widgets.savereport(self.get_report())

    def importdist(self):
        ''' Use process distribution from the project or a file '''
        dlg = page_dataimport.DistributionSelectWidget(
            singlecol=False, enablecorr=False, project=self.projitem.project,
            coloptions=['Process Distribution', 'Test Distribution'])

        ok = dlg.exec_()
        if ok:
            self.block(True)
            self.mode.setCurrentIndex(self.mode.findText('Full'))
            self.changemode(replot=False)
            dists = dlg.get_dist()

            params = dists.get('Process Distribution', None)
            if params is not None:
                self.dproc_table.set_disttype(initargs=params)
                self.dproc_table.valuechanged()
                self.chkProc.setChecked(True)

            params = dists.get('Test Distribution', None)
            if params is not None:
                self.dtest_table.set_disttype(initargs=params)
                self.dtest_table.valuechanged()
                self.chkTest.setChecked(True)
            self.block(False)
            self.entry_changed()

    def help_report(self):
        ''' Get the help report to display the current widget mode '''
        mode = self.mode.currentText()  # simple or full
        calctype = self.calctype.currentText()  # integral, MC, etc.

        if calctype == 'Integral' and mode == 'Simple':
            return RiskHelp.simple()
        elif calctype == 'Integral' and mode == 'Full':
            return RiskHelp.full()
        elif calctype == 'Monte Carlo' and mode == 'Simple':
            return RiskHelp.simple_mc()
        elif calctype == 'Monte Carlo' and mode == 'Full':
            return RiskHelp.full_mc()
        elif calctype == 'Guardband sweep':
            return RiskHelp.gb_sweep()
        elif calctype == 'Probability of Conformance':
            return RiskHelp.prob_conform()
        elif calctype == 'Risk Curves':
            return RiskHelp.curves()
        return RiskHelp.nohelp()