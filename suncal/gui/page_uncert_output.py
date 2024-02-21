''' Uncertainty Propagation Output Page '''

from PyQt6 import QtWidgets, QtCore
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from pint import PintError

from .gui_settings import gui_settings
from . import widgets
from ..common import report, distributions, unitmgr
from .help_strings import UncertHelp


class OutputPlotWidget(QtWidgets.QWidget):
    ''' Widget for controlling output plot view '''
    changed = QtCore.pyqtSignal()

    def __init__(self):
        super().__init__()
        self.cmbjoint = QtWidgets.QComboBox()
        self.cmbjoint.addItems(['Individual PDF', 'Joint PDF'])
        self.cmbjoint.setVisible(False)
        self.cmbhist = QtWidgets.QComboBox()
        self.cmbhist.addItems(['Histograms', 'PDFs'])
        self.cmbscat = QtWidgets.QComboBox()
        self.cmbscat.addItems(['Scatter', 'Contours'])
        self.cmbscat.setVisible(False)
        self.bins = widgets.SpinWidget('Bins:')
        self.bins.spin.setRange(int(3), int(1000))
        self.bins.setValue(100)
        self.points = widgets.SpinWidget('Points:')
        self.points.spin.setRange(int(1), int(10000))
        self.points.setValue(10000)
        self.points.setVisible(False)
        self.flist = widgets.ListSelectWidget()

        self.showmc = QtWidgets.QCheckBox('Show Monte-Carlo Result')
        self.showgum = QtWidgets.QCheckBox('Show GUM Result')
        self.overlay = QtWidgets.QCheckBox('Overlay GUM and MC Plots')
        self.overlay.setVisible(False)
        self.showmc.setChecked(True)
        self.showgum.setChecked(True)
        self.showleg = QtWidgets.QCheckBox('Show Legend')
        self.showleg.setChecked(True)
        self.intervals = QtWidgets.QCheckBox('Show Coverage Invervals')
        self.labeldesc = QtWidgets.QCheckBox('Label by Description')
        self.expandedconf = widgets.ExpandedConfidenceWidget()
        self.expandedconf.setVisible(False)

        self.bins.valueChanged.connect(self.changed)
        self.points.valueChanged.connect(self.changed)
        self.showmc.stateChanged.connect(self.changed)
        self.showgum.stateChanged.connect(self.changed)
        self.overlay.stateChanged.connect(self.changed)
        self.showleg.stateChanged.connect(self.changed)
        self.labeldesc.stateChanged.connect(self.changed)
        self.expandedconf.changed.connect(self.changed)
        self.flist.checkChange.connect(self.changed)
        self.cmbhist.currentIndexChanged.connect(self.typechanged)
        self.cmbjoint.currentIndexChanged.connect(self.typechanged)
        self.cmbscat.currentIndexChanged.connect(self.typechanged)
        self.intervals.stateChanged.connect(self.toggleintervals)

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.cmbjoint)
        layout.addWidget(self.cmbhist)
        layout.addWidget(self.cmbscat)
        layout.addWidget(self.bins)
        layout.addWidget(self.points)
        layout.addWidget(self.flist)
        layout.addWidget(self.showmc)
        layout.addWidget(self.showgum)
        layout.addWidget(self.overlay)
        layout.addWidget(self.showleg)
        layout.addWidget(self.labeldesc)
        layout.addWidget(self.intervals)
        layout.addWidget(self.expandedconf)
        layout.addStretch()
        self.setLayout(layout)

    def typechanged(self):
        ''' Plot type has changed '''
        assert self.cmbjoint.findText('Joint PDF') != -1  # Update below if names change
        assert self.cmbscat.findText('Scatter') != -1  # Update below if names change
        joint = self.cmbjoint.currentText() == 'Joint PDF' and len(self.flist) > 1
        self.cmbhist.setVisible(not joint)
        self.points.setVisible(joint and self.cmbscat.currentText() == 'Scatter')
        self.bins.setVisible(not self.points.isVisible())
        self.overlay.setVisible(joint)
        self.cmbscat.setVisible(joint)
        if joint and self.intervals.isChecked():
            self.intervals.setChecked(False)
        self.intervals.setVisible(not joint)
        self.flist.setVisible((not joint and len(self.flist) > 1) or (joint and len(self.flist) > 2))
        self.changed.emit()

    def toggleintervals(self):
        self.expandedconf.setVisible(self.intervals.isChecked())
        self.changed.emit()

    def contour(self):
        ''' Get contour on/off state '''
        assert self.cmbjoint.findText('Joint PDF') != -1  # Update below if names change
        assert self.cmbscat.findText('Contours') != -1  # Update below if names change
        assert self.cmbhist.findText('PDFs') != -1  # Update below if names change
        joint = self.cmbjoint.currentText() == 'Joint PDF' and len(self.flist) > 1
        return ((joint and self.cmbscat.currentText() == 'Contours') or
                (not joint and self.cmbhist.currentText() == 'PDFs'))

    def joint(self):
        ''' Is this a joint probability plot? '''
        return self.cmbjoint.currentText() == 'Joint PDF' and len(self.flist) > 1

    def set_funclist(self, flist):
        ''' Set list of function names '''
        self.flist.clear()
        self.flist.addItems(flist)
        self.flist.selectAll()
        joint = self.cmbjoint.currentText() == 'Joint PDF'
        self.flist.setVisible((not joint and len(self.flist) > 1) or (joint and len(self.flist) > 2))
        self.cmbjoint.setVisible(len(self.flist) > 1)


class OutputMCSampleWidget(QtWidgets.QWidget):
    ''' Widget for controlling MC input plot view '''
    changed = QtCore.pyqtSignal()

    def __init__(self):
        super().__init__()
        self.cmbjoint = QtWidgets.QComboBox()
        self.cmbjoint.addItems(['Individual PDF', 'Joint PDF'])
        self.cmbscat = QtWidgets.QComboBox()
        self.cmbscat.addItems(['Scatter', 'Contours'])
        self.cmbscat.setVisible(False)
        self.bins = widgets.SpinWidget('Bins:')
        self.bins.spin.setRange(int(3), int(1000))
        self.bins.setValue(100)
        self.points = widgets.SpinWidget('Points:')
        self.points.setValue(10000)
        self.points.spin.setRange(int(3), int(10000))
        self.points.setVisible(False)
        self.ilist = widgets.ListSelectWidget()
        self.labeldesc = QtWidgets.QCheckBox('Label by Description')
        self.bins.valueChanged.connect(self.changed)
        self.points.valueChanged.connect(self.changed)
        self.cmbjoint.currentIndexChanged.connect(self.typechanged)
        self.cmbscat.currentIndexChanged.connect(self.typechanged)
        self.ilist.checkChange.connect(self.changed)
        self.labeldesc.stateChanged.connect(self.changed)
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.cmbjoint)
        layout.addWidget(self.cmbscat)
        layout.addWidget(self.bins)
        layout.addWidget(self.points)
        layout.addWidget(self.ilist)
        layout.addWidget(self.labeldesc)
        layout.addStretch()
        self.setLayout(layout)

    def typechanged(self):
        ''' Plot type has changed '''
        assert self.cmbjoint.findText('Joint PDF') != -1   # Update here if names change
        assert self.cmbscat.findText('Scatter') != -1
        joint = self.cmbjoint.currentText() == 'Joint PDF'
        self.cmbscat.setVisible(joint and len(self.ilist) > 1)
        self.points.setVisible(self.cmbjoint.currentText() == 'Joint PDF' and self.cmbscat.currentText() == 'Scatter')
        self.bins.setVisible(not self.points.isVisible())
        self.ilist.setVisible((joint and len(self.ilist) > 2) or (not joint and len(self.ilist) > 1))
        self.changed.emit()

    def set_inptlist(self, ilist):
        ''' Set list of input names '''
        assert self.cmbjoint.findText('Joint PDF') != -1   # Update here if names change
        joint = self.cmbjoint.currentText() == 'Joint PDF'
        self.ilist.clear()
        self.ilist.addItems(ilist)
        self.ilist.selectAll()
        self.cmbjoint.setVisible(len(ilist) > 1)
        self.cmbscat.setVisible(len(self.ilist) > 1 and joint)
        self.ilist.setVisible((joint and len(self.ilist) > 2) or (not joint and len(self.ilist) > 1))


class OutputMCDistributionWidget(QtWidgets.QWidget):
    ''' Widget for controlling MC Distribution Analysis View '''
    changed = QtCore.pyqtSignal()

    def __init__(self):
        super().__init__()
        self.cmbdist = QtWidgets.QComboBox()
        dists = gui_settings.distributions
        dists = [d for d in dists if distributions.fittable(d)]
        self.cmbdist.addItems(dists)
        self.cmbdist.currentIndexChanged.connect(self.changed)
        self.cmbfunc = QtWidgets.QComboBox()
        self.cmbfunc.currentIndexChanged.connect(self.changed)
        self.flabel = QtWidgets.QLabel('Function')
        self.label = QtWidgets.QLabel()
        layout = QtWidgets.QVBoxLayout()
        dlayout = QtWidgets.QHBoxLayout()
        dlayout.addWidget(QtWidgets.QLabel('Distribution'))
        dlayout.addWidget(self.cmbdist)
        dlayout.addStretch()
        layout.addLayout(dlayout)
        self.flayout = QtWidgets.QHBoxLayout()
        self.flayout.addWidget(self.flabel)
        self.flayout.addWidget(self.cmbfunc)
        self.flayout.addStretch()
        layout.addLayout(self.flayout)
        layout.addWidget(QtWidgets.QLabel('Fit Parameters:'))
        layout.addWidget(self.label)
        layout.addStretch()
        self.setLayout(layout)

    def update_label(self, fitparams):
        ''' Update the label with parameter names. fitparams is dictionary of name:value '''
        if fitparams is not None:
            s = ''
            for name, val in fitparams.items():
                s += f'{name} = {val:.4g}\n'
            self.label.setText(s)

    def set_funclist(self, flist):
        self.blockSignals(True)
        self.cmbfunc.clear()
        self.cmbfunc.addItems(flist)
        self.flabel.setVisible(len(flist) > 1)
        self.cmbfunc.setVisible(len(flist) > 1)
        self.blockSignals(False)


class OutputMCConvergeWidget(QtWidgets.QWidget):
    ''' Widget for controlling MC Convergence Analysis View '''
    changed = QtCore.pyqtSignal()

    def __init__(self):
        super().__init__()
        self.relative = QtWidgets.QCheckBox('Relative to final value')
        self.relative.stateChanged.connect(self.changed)
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.relative)
        layout.addStretch()
        self.setLayout(layout)


class OutputGUMValidityWidget(QtWidgets.QWidget):
    ''' Widget for controlling display of GUM validity page '''
    changed = QtCore.pyqtSignal()

    def __init__(self):
        super().__init__()
        self.ndig = widgets.SpinWidget('Significant Digits')
        self.ndig.spin.setRange(1, 5)
        self.ndig.setValue(1)
        self.ndig.valueChanged.connect(self.changed)
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.ndig)
        layout.addStretch()
        self.setLayout(layout)


class OutputGUMDerivationWidget(QtWidgets.QWidget):
    ''' Widget for controlling display of GUM derivation page '''
    changed = QtCore.pyqtSignal()

    def __init__(self):
        super().__init__()
        self.showvalues = QtWidgets.QCheckBox('Show derivation with values')
        self.showvalues.stateChanged.connect(self.changed)
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.showvalues)
        layout.addStretch()
        self.setLayout(layout)


class OutputReportGen(QtWidgets.QWidget):
    ''' Class for controlling full output report '''
    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.report = None  # Cache the report object, only refresh on button press
        self.btnRefresh = QtWidgets.QPushButton('Refresh')
        self.chkSummary = QtWidgets.QCheckBox('Summary')
        self.chkOutputPlot = QtWidgets.QCheckBox('Output Plots')
        self.chkInputs = QtWidgets.QCheckBox('Input Values')
        self.chkComponents = QtWidgets.QCheckBox('Uncertainty Budget')
        self.chkSensitivity = QtWidgets.QCheckBox('Sensitivity Coefficients')
        self.chkExpanded = QtWidgets.QCheckBox('Expanded Uncertainties')
        self.chkGUMderiv = QtWidgets.QCheckBox('GUM Derivation')
        self.chkGUMvalid = QtWidgets.QCheckBox('GUM Validity')
        self.chkMChist = QtWidgets.QCheckBox('MC Input Histograms')
        self.chkMCconverge = QtWidgets.QCheckBox('MC Convergence')
        self.chkSummary.setChecked(True)
        self.chkOutputPlot.setChecked(True)
        self.chkInputs.setChecked(True)
        self.chkComponents.setChecked(True)
        self.chkSensitivity.setChecked(True)
        self.chkExpanded.setChecked(True)
        self.chkGUMderiv.setChecked(True)
        self.chkGUMvalid.setChecked(True)
        self.chkMChist.setChecked(True)
        self.chkMCconverge.setChecked(True)
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.chkSummary)
        layout.addWidget(self.chkOutputPlot)
        layout.addWidget(self.chkInputs)
        layout.addWidget(self.chkExpanded)
        layout.addWidget(self.chkComponents)
        layout.addWidget(self.chkSensitivity)
        layout.addWidget(self.chkGUMderiv)
        layout.addWidget(self.chkGUMvalid)
        layout.addWidget(self.chkMChist)
        layout.addWidget(self.chkMCconverge)
        layout.addWidget(self.btnRefresh)
        self.setLayout(layout)


class OutputUnitsWidget(QtWidgets.QWidget):
    ''' Widget for converting units of output '''
    unitschanged = QtCore.pyqtSignal(object)

    def __init__(self):
        super().__init__()
        self.result = None  # Not set yet
        self.flayout = QtWidgets.QFormLayout()
        self.label = QtWidgets.QLabel('Units:')
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.label)
        layout.addLayout(self.flayout)
        layout.addStretch()
        self.setLayout(layout)
        self.unitwidgets = []
        self.fnames = []

    def setresult(self, result):
        ''' Set the measurment result '''
        self.result = result

        # Reset (for back/recalculate)
        while self.flayout.rowCount() > 0:
            self.flayout.removeRow(0)

        self.unitwidgets = []
        self.fnames = []
        for fname in self.result.functionnames:
            currentunits = self.result.getunits().get(fname)
            if currentunits and not unitmgr.is_dimensionless(currentunits):
                widget = QtWidgets.QLineEdit(str(currentunits))
                widget.editingFinished.connect(self.setunits)
                self.flayout.addRow(f'{fname}:', widget)
                self.unitwidgets.append(widget)
                self.fnames.append(fname)

        self.label.setVisible(len(self.unitwidgets) > 0)

    def setunits(self):
        ''' Units were changed. Update the result and refresh. '''
        changed = False
        oldunits = self.result.getunits()
        for i, fname in enumerate(self.fnames):
            unitstr = self.unitwidgets[i].text()
            try:
                self.result.units(**{fname: unitstr})
            except PintError:
                oldunit = str(oldunits.get(fname, ''))
                self.result.units(**{fname: oldunit})
                self.unitwidgets[i].setText(oldunit)
            else:
                changed = True
        if changed:
            self.unitschanged.emit(self.result)


class PageOutput(QtWidgets.QWidget):
    ''' Page for viewing output values '''
    back = QtCore.pyqtSignal()
    change_help = QtCore.pyqtSignal()

    allitems = ['Summary', 'Comparison Plots', 'Expanded Uncertainties', 'Uncertainty Budget',
                'GUM Derivation', 'GUM Validity', 'Monte Carlo Distribution',
                'Monte Carlo Input Plots', 'Monte Carlo Convergence', 'Full Report']

    def __init__(self, parent=None):
        super().__init__(parent)
        self.outreport = None  # Cached output report
        self.result = None
        self.outputSelect = QtWidgets.QComboBox()
        self.outputSelect.addItems(self.allitems)
        self.outputUnits = OutputUnitsWidget()
        self.outputPlot = OutputPlotWidget()
        self.outputExpanded = widgets.ExpandedConfidenceWidget()
        self.outputMCsample = OutputMCSampleWidget()
        self.outputMCdist = OutputMCDistributionWidget()
        self.outputMCconv = OutputMCConvergeWidget()
        self.outputGUMvalid = OutputGUMValidityWidget()
        self.outputGUMderiv = OutputGUMDerivationWidget()
        self.outputReportSetup = OutputReportGen()
        self.txtOutput = widgets.MarkdownTextEdit()
        self.fig = Figure()
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setStyleSheet("background-color:transparent;")
        self.toolbar = NavigationToolbar(self.canvas, self, coordinates=True)
        self.plotWidget = QtWidgets.QWidget()
        playout = QtWidgets.QVBoxLayout()
        playout.addWidget(self.canvas)
        playout.addWidget(self.toolbar)
        self.plotWidget.setLayout(playout)

        self.ctrlStack = QtWidgets.QStackedWidget()
        self.ctrlStack.addWidget(self.outputUnits)
        self.ctrlStack.addWidget(self.outputPlot)
        self.ctrlStack.addWidget(self.outputExpanded)
        self.ctrlStack.addWidget(QtWidgets.QWidget())
        self.ctrlStack.addWidget(self.outputGUMderiv)
        self.ctrlStack.addWidget(self.outputGUMvalid)
        self.ctrlStack.addWidget(self.outputMCdist)
        self.ctrlStack.addWidget(self.outputMCsample)
        self.ctrlStack.addWidget(self.outputMCconv)
        self.ctrlStack.addWidget(self.outputReportSetup)
        self.ctrlStack.addWidget(QtWidgets.QWidget())
        self.outputStack = QtWidgets.QStackedWidget()
        self.outputStack.addWidget(self.plotWidget)
        self.outputStack.addWidget(self.txtOutput)
        self.btnBack = QtWidgets.QPushButton('Back')
        self.btnBack.clicked.connect(self.goback)

        self.oplayout = QtWidgets.QVBoxLayout()  # Options along left side
        self.oplayout.addWidget(self.outputSelect)
        self.oplayout.addWidget(self.ctrlStack)
        self.oplayout.addStretch()
        self.oplayout.addStretch()
        self.oplayout.addWidget(self.btnBack)
        self.optionswidget = QtWidgets.QWidget()
        self.optionswidget.setLayout(self.oplayout)
        self.splitter = QtWidgets.QSplitter()
        self.splitter.addWidget(self.optionswidget)
        self.splitter.addWidget(self.outputStack)
        self.splitter.setCollapsible(0, False)
        self.splitter.setCollapsible(1, False)
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.splitter)
        self.setLayout(layout)

        self.outputSelect.currentIndexChanged.connect(self.set_outputpage)
        self.outputExpanded.changed.connect(self.outputupdate)
        self.outputPlot.changed.connect(self.outputupdate)
        self.outputMCsample.changed.connect(self.outputupdate)
        self.outputMCconv.changed.connect(self.outputupdate)
        self.outputGUMderiv.changed.connect(self.outputupdate)
        self.outputGUMvalid.changed.connect(self.outputupdate)
        self.outputMCdist.changed.connect(self.outputupdate)
        self.outputReportSetup.btnRefresh.clicked.connect(self.refresh_fullreport)
        self.outputUnits.unitschanged.connect(self.changeunits)

    def set_outputpage(self):
        ''' Change the output page '''
        self.ctrlStack.setCurrentIndex(self.outputSelect.currentIndex())
        self.outputupdate()
        self.change_help.emit()

    def generate_report(self):
        rptsetup = {
            'summary': self.outputReportSetup.chkSummary.isChecked(),
            'outputs': self.outputReportSetup.chkOutputPlot.isChecked(),
            'inputs': self.outputReportSetup.chkInputs.isChecked(),
            'components': self.outputReportSetup.chkComponents.isChecked(),
            'sens': self.outputReportSetup.chkSensitivity.isChecked(),
            'expanded': self.outputReportSetup.chkExpanded.isChecked(),
            'gumderv': self.outputReportSetup.chkGUMderiv.isChecked(),
            'gumvalid': self.outputReportSetup.chkGUMvalid.isChecked(),
            'gumvaliddig': self.outputGUMvalid.ndig.value(),
            'mchist': self.outputReportSetup.chkMChist.isChecked(),
            'mcconv': self.outputReportSetup.chkMCconverge.isChecked(),
            'mcconvnorm': self.outputMCconv.relative.isChecked(),
            'shortest': self.outputExpanded.get_shortest(),
            'gumvalues': self.outputGUMderiv.showvalues.isChecked(),
            'mchistparams': {'joint': (self.outputMCsample.cmbjoint.currentText() == 'Joint PDF' and
                                       len(self.result.variablenames) > 1),
                             'bins': self.outputMCsample.bins.value(),
                             'points': self.outputMCsample.points.value(),
                             'contour': self.outputMCsample.cmbscat.currentText() == 'Contours',
                             'inpts': self.outputMCsample.ilist.getSelectedValues(),
                             'cmap': gui_settings.colormap_contour},
            'outplotparams': {'joint': self.outputPlot.joint(),
                              'showgum': self.outputPlot.showgum.isChecked(),
                              'showmc': self.outputPlot.showmc.isChecked(),
                              'overlay': self.outputPlot.overlay.isChecked(),
                              'bins': self.outputPlot.bins.value(),
                              'points': self.outputPlot.points.value(),
                              'contour': self.outputPlot.contour(),
                              'funcs': self.outputPlot.flist.getSelectedValues(),
                              'legend': self.outputPlot.showleg.isChecked(),
                              'cmap': gui_settings.colormap_contour,
                              'cmapmc': gui_settings.colormap_scatter}
                    }
        rptsetup.update(self.outputExpanded.get_params())
        if self.outputPlot.intervals.isChecked():
            kconf = self.outputPlot.expandedconf.get_params()
            rptsetup['outplotparams'].update(kconf)
            rptsetup['outplotparams']['shortest'] = self.outputPlot.expandedconf.get_shortest()
        return self.result.report.all(rptsetup)

    def refresh_fullreport(self):
        self.outreport = self.generate_report()  # Cache report for displaying/saving
        self.outputupdate()
        return self.outreport

    def show_summary(self):
        r = report.Report()
        r.hdr('Results', level=2)
        r.append(self.result.report.summary_withplots())
        self.txtOutput.setReport(r)

    def show_fullreport(self):
        if self.outreport is None:
            self.refresh_fullreport()
        self.txtOutput.setReport(self.outreport)

    def show_expanded(self):
        kconf = self.outputExpanded.get_params()
        shortest = self.outputExpanded.get_shortest()
        r = report.Report()
        r.hdr('Expanded Uncertainty', level=2)
        r.append(self.result.report.expanded(shortest=shortest, **kconf))
        self.txtOutput.setReport(r)

    def show_budget(self):
        self.txtOutput.setReport(self.result.report.allinputs())

    def show_warnings(self):
        self.txtOutput.setReport(self.result.report.warnings())

    def show_derivation(self):
        solve = self.outputGUMderiv.showvalues.isChecked()
        self.txtOutput.setReport(self.result.gum.report.derivation(solve=solve))

    def show_convergence(self):
        self.result.montecarlo.report.plot.converge(fig=self.fig, relative=self.outputMCconv.relative.isChecked())
        self.canvas.draw_idle()
        self.fig.tight_layout()

    def show_comparison_plot(self):
        functions = self.outputPlot.flist.getSelectedValues()
        legend = self.outputPlot.showleg.isChecked()
        labeldesc = self.outputPlot.labeldesc.isChecked()
        shortest = None
        kconf = {}
        if self.outputPlot.intervals.isChecked():
            kconf = self.outputPlot.expandedconf.get_params()
            if 'conf' in kconf:
                kconf['interval'] = kconf.pop('conf')
            shortest = self.outputPlot.expandedconf.get_shortest()

        if self.outputPlot.joint():
            if self.outputPlot.showgum.isChecked() and self.outputPlot.showmc.isChecked():
                if self.outputPlot.contour():
                    self.result.report.plot.joint_pdf(
                        fig=self.fig, functions=functions, overlay=self.outputPlot.overlay.isChecked(),
                        cmap=gui_settings.colormap_contour,
                        cmapmc=gui_settings.colormap_scatter,
                        labeldesc=labeldesc)
                else:
                    self.result.report.plot.joint_scatter(
                        fig=self.fig, functions=functions, overlay=self.outputPlot.overlay.isChecked(),
                        points=self.outputPlot.points.value(),
                        labeldesc=labeldesc)

            elif self.outputPlot.showgum.isChecked():
                self.result.gum.report.plot.joint_pdf(fig=self.fig, functions=functions, labeldesc=labeldesc)
            elif self.outputPlot.contour():
                self.result.montecarlo.report.plot.joint_pdf(
                    fig=self.fig, bins=self.outputPlot.bins.value(), functions=functions,
                    cmap=gui_settings.colormap_scatter, labeldesc=labeldesc)
            else:
                self.result.montecarlo.report.plot.scatter(
                    fig=self.fig, points=self.outputPlot.points.value(), functions=functions, labeldesc=labeldesc)

        else:
            if self.outputPlot.showgum.isChecked() and self.outputPlot.showmc.isChecked():
                self.result.report.plot.pdf(
                    fig=self.fig, functions=functions, legend=legend,
                    mchist=not self.outputPlot.contour(), **kconf, shortest=shortest, labeldesc=labeldesc,
                    bins=self.outputPlot.bins.value())
            elif self.outputPlot.showgum.isChecked():
                self.result.gum.report.plot.pdf(fig=self.fig, **kconf,  functions=functions, labeldesc=labeldesc)
            elif self.outputPlot.contour():
                self.result.montecarlo.report.plot.pdf(
                    fig=self.fig, functions=functions, bins=self.outputPlot.bins.value(),
                    **kconf, shortest=shortest, labeldesc=labeldesc)
            else:
                self.result.montecarlo.report.plot.hist(
                    fig=self.fig, functions=functions, bins=self.outputPlot.bins.value(),
                    **kconf, shortest=shortest, labeldesc=labeldesc)
        self.canvas.draw_idle()

    def show_mcinputs(self):
        numinpts = len(self.result.variablenames)
        joint = self.outputMCsample.cmbjoint.currentText() == 'Joint PDF' and numinpts > 1
        bins = self.outputMCsample.bins.value()
        points = self.outputMCsample.points.value()
        labeldesc = self.outputMCsample.labeldesc.isChecked()
        variables = self.outputMCsample.ilist.getSelectedValues()

        if joint and self.outputMCsample.cmbscat.currentText() == 'Contours':
            self.result.montecarlo.report.plot.variable_contour(
                fig=self.fig, bins=bins, variables=variables, labeldesc=labeldesc)
        elif joint:
            self.result.montecarlo.report.plot.variable_scatter(
                fig=self.fig, points=points, variables=variables, labeldesc=labeldesc)
        else:
            self.result.montecarlo.report.plot.variable_hist(
                fig=self.fig, bins=bins, variables=variables, labeldesc=labeldesc)
        self.canvas.draw_idle()

    def show_validity(self):
        ndig = self.outputGUMvalid.ndig.value()
        r = self.result.report.validity(ndig=ndig)
        self.txtOutput.setReport(r)

    def show_mcdistribution(self):
        funcname = self.outputMCdist.cmbfunc.currentText()
        dist = self.outputMCdist.cmbdist.currentText()
        fitparams = self.result.montecarlo.report.plot.probplot(function=funcname, distname=dist, fig=self.fig)
        self.outputMCdist.update_label(fitparams)
        self.canvas.draw_idle()

    def outputupdate(self):
        ''' Update the output page based on widget settings. '''
        PLOT = 0
        TEXT = 1
        option = self.outputSelect.currentText()
        if (option in ['Summary', 'Expanded Uncertainties', 'Uncertainty Budget', 'GUM Validity', 'GUM Derivation',
                       'Monte Carlo Components', 'Full Report', 'Warnings']):
            self.outputStack.setCurrentIndex(TEXT)
        else:
            self.outputStack.setCurrentIndex(PLOT)

        {'Summary': self.show_summary,
         'Full Report': self.show_fullreport,
         'Expanded Uncertainties': self.show_expanded,
         'Uncertainty Budget': self.show_budget,
         'Warnings': self.show_warnings,
         'GUM Derivation': self.show_derivation,
         'Monte Carlo Convergence': self.show_convergence,
         'Comparison Plots': self.show_comparison_plot,
         'Monte Carlo Input Plots': self.show_mcinputs,
         'GUM Validity': self.show_validity,
         'Monte Carlo Distribution': self.show_mcdistribution,
         }.get(option)()

    def changeunits(self, result):
        self.result = result
        self.show_summary()

    def update(self, result, symonly=False):
        ''' Calculation run, update the page '''
        self.result = result
        self.outputUnits.setresult(result)
        if not symonly:
            if self.outputSelect.count() < len(self.allitems):
                # Switched from symbolic back to full
                self.outputSelect.blockSignals(True)
                self.outputSelect.clear()
                self.outputSelect.addItems(self.allitems)
                self.outputSelect.blockSignals(False)
            self.outputPlot.set_funclist(self.result.functionnames)
            self.outputMCdist.set_funclist(self.result.functionnames)
            self.outputMCsample.set_inptlist(self.result.variablenames)
            self.outputReportSetup.report = None
            idx = self.outputSelect.findText('Warnings')
            if len(self.result.report.warnings().get_md().strip()) > 0:
                if idx < 0 or idx is None:
                    self.outputSelect.addItem('Warnings')
                self.outputSelect.setCurrentIndex(self.outputSelect.count()-1)
            elif idx >= 0:
                self.outputSelect.removeItem(idx)
            if self.outputSelect.currentText() == 'Full Report':
                self.refresh_fullreport()
                self.outputupdate()
        else:
            self.outputSelect.blockSignals(True)
            self.outputSelect.clear()
            self.outputSelect.addItems(['GUM Derivation'])
            self.outputSelect.blockSignals(False)
            self.ctrlStack.setCurrentIndex(0)
            self.outputUnits.setVisible(False)
            self.outputupdate()

    def goback(self):
        self.back.emit()

    def help_report(self):
        ''' Get the help report to display the current widget mode '''
        option = self.outputSelect.currentText()
        return {'Summary': UncertHelp.summary,
                'Expanded Uncertainties': UncertHelp.expanded,
                'Uncertainty Budget': UncertHelp.budget,
                'GUM Derivation': UncertHelp.derivation,
                'Monte Carlo Convergence': UncertHelp.converge,
                'Comparison Plots': UncertHelp.plots,
                'Monte Carlo Input Plots': UncertHelp.montecarlo,
                'GUM Validity': UncertHelp.validity,
                'Monte Carlo Distribution': UncertHelp.mcdistribution,
                }.get(option, UncertHelp.nohelp)()
