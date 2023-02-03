''' Guided/wizard uncertainty input interface '''

from enum import IntEnum, auto
from contextlib import suppress

import numpy as np
from PyQt5 import QtWidgets, QtGui, QtCore
from pint import PintError, OffsetUnitCalculusError
import sympy
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from . import gui_common  # noqa: F401
from . import gui_widgets
from ..uncertainty.variables import Typeb
from ..project.proj_wizard import ProjectUncertWizard, VariableDataType
from ..common import report, unitmgr, uparser, ttable
from ..datasets.dataset_model import DataSet
from ..datasets.dataset import uncert_autocorrelated
from ..datasets.report.dataset import ReportDataSet
from .page_uncert_output import PageOutput


class Pages(IntEnum):
    NOPAGE = -1
    MODEL = auto()
    DATATYPE = auto()
    SINGLE = auto()
    REPEAT = auto()
    REPROD = auto()
    UNCERTS = auto()
    TYPEB = auto()
    UNITS = auto()
    SUMMARY = auto()
    VAR_MODIFY = auto()
    OUTPUT = auto()


class UncertWizard(QtWidgets.QWidget):
    ''' Guided uncertainty wizard interface '''
    # Can't use Qt's built-in QWizard since it's not good with non-linear flows.
    def __init__(self, projitem=None, parent=None):
        super().__init__(parent)
        if projitem is None:
            self.projitem = ProjectUncertWizard()
        else:
            self.projitem = projitem

        self.currentvar = None  # Current variable being modified
        self.currentunc = None
        font = self.font()
        font.setPointSize(12)
        self.setFont(font)

        self.btnBack = QtWidgets.QPushButton()
        self.btnNext = QtWidgets.QPushButton()
        self.btnHelp = QtWidgets.QPushButton('Help')
        self.btnBack.setIcon(self.style().standardIcon(QtWidgets.QStyle.SP_ArrowLeft))
        self.btnNext.setIcon(self.style().standardIcon(QtWidgets.QStyle.SP_ArrowRight))
        self.btnHelp.setIcon(self.style().standardIcon(QtWidgets.QStyle.SP_TitleBarContextHelpButton))
        self.stack = gui_widgets.SlidingStackedWidget()
        self.helppane = gui_widgets.MarkdownTextEdit()
        self.splitter = QtWidgets.QSplitter()
        self.splitter.addWidget(self.stack)
        self.splitter.addWidget(self.helppane)
        self.splitter.setCollapsible(0, False)

        self.menu = QtWidgets.QMenu('&Uncertainty')
        self.actSaveReport = QtWidgets.QAction('Save Report...', self)
        self.menu.addAction(self.actSaveReport)
        self.actSaveReport.setEnabled(False)
        self.actSaveReport.triggered.connect(self.save_report)

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.splitter)
        blayout = QtWidgets.QHBoxLayout()
        blayout.addWidget(self.btnHelp)
        blayout.addStretch()
        blayout.addWidget(self.btnBack)
        blayout.addWidget(self.btnNext)
        layout.addLayout(blayout)
        self.setLayout(layout)

        self.btnBack.clicked.connect(self.goback)
        self.btnNext.clicked.connect(self.gonext)
        self.btnHelp.clicked.connect(self.showhelp)
        self.btnNext.setDefault(True)

        # Order must align with Pages enum
        self.stack.addWidget(PageMeasModel(self))  # Pages.MODEL
        self.stack.addWidget(PageDatatype(self))
        self.stack.addWidget(PageSingle(self))
        self.stack.addWidget(PageRepeat(self))
        self.stack.addWidget(PageReprod(self))
        self.stack.addWidget(PageUncerts(self))
        self.stack.addWidget(PageTypeB(self))
        self.stack.addWidget(PageUnits(self))
        self.stack.addWidget(PageSummary(self))
        self.stack.addWidget(PageVariableSelect(self))  # Pages.VAR_MODIFY
        self.outpage = PageOutput(None)
        self.stack.addWidget(self.outpage)
        self.btnBack.setEnabled(False)
        self.outpage.back.connect(self.output_back)

        # Navigate by Enter/Return key
        self.shortcut_next = QtWidgets.QShortcut(QtGui.QKeySequence('enter'), self)
        self.shortcut_next.activated.connect(self.gonext)
        self.shortcut_next2 = QtWidgets.QShortcut(QtGui.QKeySequence('return'), self)
        self.shortcut_next2.activated.connect(self.gonext)

        self.stack.currentWidget().initialize()
        self.showhelp()  # Hides help the first time

    def get_menu(self):
        ''' Get menu for this widget '''
        with suppress(AttributeError):
            self.stack.currentWidget().initialize()  # Hack to run setfocus on init
        return self.menu

    def update_proj_config(self):
        ''' Update project with config on page. Nothing to do for wizard. '''

    def output_back(self):
        ''' Go back from output page '''
        self.stack.slideInRight(Pages.SUMMARY)
        self.btnBack.setEnabled(True)
        self.btnNext.setEnabled(True)

    def gonext(self):
        ''' Navigate to the next page '''
        if self.btnNext.isEnabled():
            try:
                pageid = self.stack.currentWidget().gonext()
            except AttributeError:
                pageid = None

            if pageid == Pages.OUTPUT:
                self.outpage.update(self.projitem.result)
                self.outpage.outputupdate()
                self.stack.slideInLeft(Pages.OUTPUT)
                self.btnBack.setEnabled(True)
                self.btnNext.setEnabled(False)
                self.actSaveReport.setEnabled(True)
                self.btnHelp.setVisible(False)
                self.showhelp(hide=True)
            elif pageid == Pages.NOPAGE:
                pageid = self.stack.currentIndex() + 1
                self.loadhelp(pageid)
            elif pageid is not None:
                self.stack.widget(pageid).initialize()
                self.btnBack.setEnabled(self.stack.widget(pageid).backenabled)
                self.stack.slideInLeft(pageid)
                self.loadhelp(pageid)

    def goback(self):
        ''' Navigate to the previous page '''
        if self.btnBack.isEnabled():
            self.btnHelp.setVisible(True)
            pageid = self.stack.currentWidget().goback()
            if pageid is not None:
                self.stack.widget(pageid).initialize()
                self.btnBack.setEnabled(self.stack.widget(pageid).backenabled)
                self.stack.slideInRight(pageid)
                self.loadhelp(pageid)

    def showhelp(self, hide=False):
        ''' Show help popup '''
        sizes = self.splitter.sizes()
        total = sum(sizes)
        if sizes[-1] == 0 and not hide:
            newsizes = (int(total * .65), int(total * .35))
        else:
            newsizes = (total, 0)
        self.splitter.setSizes(newsizes)
        self.loadhelp()

    def loadhelp(self, pageid=None):
        if pageid is None:
            pageid = self.stack.currentIndex()
        rpt = self.stack.widget(pageid).gethelp()
        self.helppane.setReport(rpt)

    def current_randvar(self):
        ''' Get the current input variable being worked on '''
        return self.projitem.model.var(self.currentvar)

    def calculate(self):
        ''' Run the calculation '''
        return self.projitem.calculate()

    def save_report(self):
        ''' Save full output report to file, using user's GUI settings '''
        gui_widgets.savereport(self.get_report())

    def get_report(self):
        ''' Get output report '''
        rpt = self.outpage.outputReportSetup.report
        if rpt is None and hasattr(self.outpage, 'outdata'):
            rpt = self.outpage.refresh_fullreport()
        return rpt


class Page(QtWidgets.QWidget):
    ''' Mimic QWizard interface, but allow non-linear page traversal '''
    def __init__(self, parent):
        super().__init__(parent=parent)
        self.wizard = parent
        self.backenabled = True
        font = self.font()
        font.setPointSize(12)
        self.setFont(font)

    def initialize(self):
        ''' Called when navigating to the page '''
        pass

    def gonext(self):
        ''' Called when next page is requested. Returns next page ID '''
        return Pages.NOPAGE

    def goback(self):
        ''' Called when previous page is requested. Returns previous page ID '''
        return None

    def gethelp(self):
        rpt = report.Report()
        rpt.txt('No help available')
        return rpt


class PageMeasModel(Page):
    ''' Measurement Model Page '''
    def __init__(self, parent=None):
        super().__init__(parent)
        title = QtWidgets.QLabel('Enter a Measurement Model')
        self.backenabled = False
        self.txtFunction = QtWidgets.QLineEdit('x')
        self.validindicator = QtWidgets.QLabel()
        self.eqlabel = QtWidgets.QLabel()
        self.varlabel = QtWidgets.QLabel()

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(title)
        hlayout = QtWidgets.QHBoxLayout()
        hlayout.addWidget(self.txtFunction)
        hlayout.addWidget(self.validindicator)
        layout.addLayout(hlayout)
        layout.addSpacing(20)
        layout.addWidget(QtWidgets.QLabel('Function:'))
        layout.addWidget(self.eqlabel)
        layout.addSpacing(20)
        layout.addWidget(QtWidgets.QLabel('Variables:'))
        layout.addWidget(self.varlabel)
        layout.addStretch()
        self.setLayout(layout)
        self.txtFunction.textChanged.connect(self.updatemath)
        self.updatemath()

    def updatemath(self):
        ''' Update the math displayed '''
        expr = self.txtFunction.text()
        if '=' in expr:
            _, expr = expr.split('=')
            expr = expr.strip()
        expr = uparser.parse_math(expr, raiseonerr=False)
        if expr is None:
            self.validindicator.setPixmap(self.style().standardPixmap(QtWidgets.QStyle.SP_DialogNoButton))
            self.wizard.btnNext.setEnabled(False)
        else:
            self.validindicator.setPixmap(self.style().standardPixmap(QtWidgets.QStyle.SP_DialogYesButton))
            ratio = QtWidgets.QApplication.instance().devicePixelRatio()
            px = QtGui.QPixmap()
            px.loadFromData(report.Math.from_sympy(expr).svg_buf(fontsize=16*ratio).read())
            px.setDevicePixelRatio(ratio)
            self.eqlabel.setPixmap(px)

            varnames = ', '.join(sorted(sympy.latex(s) for s in expr.free_symbols))
            px = QtGui.QPixmap()
            px.loadFromData(report.Math.from_latex(varnames).svg_buf(fontsize=16*ratio).read())
            px.setDevicePixelRatio(ratio)
            self.varlabel.setPixmap(px)
            self.wizard.btnNext.setEnabled(True)

    def initialize(self):
        if len(self.wizard.projitem.model.exprs) > 0:
            self.txtFunction.setText(self.wizard.projitem.model.exprs[0])
        else:
            self.txtFunction.setText('f = x')
        self.txtFunction.setFocus()

    def gonext(self):
        self.wizard.projitem.set_function(self.txtFunction.text())
        missing = self.wizard.projitem.missingvars
        if len(missing) == 0:
            return Pages.SUMMARY
        self.wizard.currentvar = missing[0]
        return Pages.DATATYPE

    def gethelp(self):
        rpt = report.Report()
        rpt.hdr('Measurement Model', level=2)
        rpt.txt('The Measurement Model defines the function to calculate, in the form of\n\n')
        rpt.mathtex('y = f(x_1, x_2, ...)', end='\n\n')
        rpt.txt('Enter an equation, including equal sign, in the form "f = x". ')
        rpt.txt('Use "Excel-like" notation, with operators +, -, *, /, and powers using ^.\n\n')
        rpt.txt('Variables:\n\n- Are case sensitive\n- Use underscores for subscripts (R_1)\n')
        rpt.txt('- May be the name of Greek letters ("theta" = θ, "omega" = ω, "Omega" = Ω)\n')
        rpt.txt('- Some variables are interpreted as constants (e, pi, inf)\n')
        rpt.txt('- Cannot be Python keywords (if, else, def, class, etc.)\n\n')
        rpt.txt('Recognized Functions:\n\n - sin, cos, tan\n- asin, acos, atan, atan2\n')
        rpt.txt('- sinh, cosh, tanh\n- exp, log, log10\n- sqrt, root\n- rad, deg\n\n')
        rpt.txt('If the indicator turns red, Suncal cannot interpret the equation.')
        return rpt


class VarnameTitle(QtWidgets.QWidget):
    ''' Title containing an SVG variable name '''
    def __init__(self, title, varname, title2='', parent=None):
        super().__init__(parent)
        self.title = QtWidgets.QLabel(title)
        self.title2 = QtWidgets.QLabel(title2)
        self.varname = QtWidgets.QLabel()
        layout = QtWidgets.QHBoxLayout()
        layout.addWidget(self.title)
        layout.addWidget(self.varname)
        layout.addWidget(self.title2)
        layout.addStretch()
        self.setLayout(layout)
        self.setvarname(varname)

    def setvarname(self, varname, title2=None):
        ''' Set the variable name to show in title '''
        ratio = QtWidgets.QApplication.instance().devicePixelRatio()
        px = QtGui.QPixmap()
        px.loadFromData(report.Math.from_latex(sympy.latex(sympy.Symbol(varname))).svg_buf(fontsize=16*ratio).read())
        px.setDevicePixelRatio(ratio)
        self.varname.setPixmap(px)
        if title2 is not None:
            self.title2.setText(title2)


class PageDatatype(Page):
    ''' Variable input Page '''
    def __init__(self, parent=None):
        super().__init__(parent)
        self.backenabled = False
        self.title = VarnameTitle('What type of data was measured for variable', varname='x', title2='?')
        self.optSingle = QtWidgets.QRadioButton('Single value')
        self.optRepeat = QtWidgets.QRadioButton('Repeatability (K measurements)')
        self.optReprod = QtWidgets.QRadioButton('Reproducibility (K measurements x J conditions)')
        self.optSkip = QtWidgets.QRadioButton('Skip this one for now')
        self.optSingle.setChecked(True)
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.title)
        layout.addSpacing(20)
        layout.addWidget(self.optSingle)
        layout.addWidget(self.optRepeat)
        layout.addWidget(self.optReprod)
        layout.addWidget(self.optSkip)
        layout.addStretch()
        self.setLayout(layout)

    def initialize(self):
        varname = self.wizard.currentvar
        self.optSkip.setEnabled(len(self.wizard.projitem.missingvars) > 1)
        self.title.setvarname(varname)
        if varname in self.wizard.projitem.model.variables.names:
            inpttype = self.wizard.projitem.inpt_type(varname)
            if inpttype == VariableDataType.REPEATABILITY:
                self.optRepeat.setChecked(True)
            elif inpttype == VariableDataType.REPRODUCIBILITY:
                self.optReprod.setChecked(True)
            else:
                self.optSingle.setChecked(True)

    def gonext(self):
        if self.optSingle.isChecked():
            page = Pages.SINGLE
        elif self.optRepeat.isChecked():
            page = Pages.REPEAT
        elif self.optReprod.isChecked():
            page = Pages.REPROD
        elif self.optSkip.isChecked():
            missingvars = self.wizard.projitem.missingvars
            if len(missingvars) > 1:
                missingvars.remove(self.wizard.currentvar)
            self.wizard.currentvar = missingvars[0]
            page = Pages.DATATYPE
        else:
            page = Pages.SINGLE
        return page

    def gethelp(self):
        rpt = report.Report()
        rpt.hdr('Variable Data Type', level=2)
        rpt.txt('Each variable in the measurement model must be assigned a value. This page provides '
                'options for how that value is determined. Select **Single value** when the variable was '
                'a known constant, or provided as one number. Select **Repeatability** if the variable '
                'was determined from a series of K repeated measurements under the same conditions. '
                '**Reproducibility** may be used if the variable was determined using K measurements over '
                'J different conditions - often K measurements per day over J days.')
        return rpt


class PageSingle(Page):
    ''' Page for single nominal value if no Type A '''
    def __init__(self, parent=None):
        super().__init__(parent)
        self.title = VarnameTitle('What is the expected value of variable', varname='x', title2='?')
        self.txtValue = QtWidgets.QLineEdit()
        self.lblValue = QtWidgets.QLabel()
        self.backenabled = True
        self.txtValue.textChanged.connect(self.parse)
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.title)
        layout.addSpacing(20)
        flayout = QtWidgets.QFormLayout()
        flayout.addRow('Value', self.txtValue)
        layout.addLayout(flayout)
        layout.addSpacing(20)
        layout.addWidget(self.lblValue)
        layout.addStretch()
        self.setLayout(layout)

    def initialize(self):
        varname = self.wizard.currentvar
        self.title.setvarname(varname)
        if varname in self.wizard.projitem.variablesdone:
            var = self.wizard.projitem.model.var(varname)
            self.txtValue.setText(f'{var.value[0]}')
        else:
            self.txtValue.setText('')

    def parse(self):
        ''' Parse the value+units and ensure it's valid '''
        try:
            qty = uparser.parse_value(self.txtValue.text())
        except (ValueError, AttributeError):
            self.lblValue.setText('---')
            qty = None
            self.wizard.btnNext.setEnabled(False)
        else:
            unit = ''
            if hasattr(qty, 'magnitude'):
                val = qty.magnitude
                if not unitmgr.is_dimensionless(qty.units):
                    unit = f'Units: {str(report.Unit(qty.units, abbr=False))}'
            else:
                val = qty
            self.lblValue.setText(f'Value: {val}\n{unit}')
            self.wizard.btnNext.setEnabled(True)
        return qty

    def gonext(self):
        varname = self.wizard.currentvar
        qty = self.parse()
        if qty is not None:
            if isinstance(qty, unitmgr.Quantity):
                value, unit = qty.magnitude, format(qty.units, 'D')
            else:
                value, unit = qty, None
            self.wizard.projitem.set_inputval(varname, value, unit)
            return Pages.UNCERTS
        return None

    def goback(self):
        return Pages.DATATYPE

    def gethelp(self):
        rpt = report.Report()
        rpt.hdr('Single Value Measurement', level=2)
        rpt.txt('Enter the value of the variable, along with measurement units if applicable. '
                'Check the **Value** and **Units** indicator below the entry to make sure '
                'the units are interpreted as desired. If the indicator is dashed out, '
                "Suncal doesn't understand the units.")
        return rpt


class PageUncerts(Page):
    ''' Page to List all loaded uncertainties for a variable '''
    def __init__(self, parent=None):
        super().__init__(parent)
        self.title = VarnameTitle('These are the expected value and standard uncertainties for', varname='x', title2='')
        self.uncsummary = gui_widgets.MarkdownTextEdit()
        self.backenabled = False
        self.optGood = QtWidgets.QRadioButton('These look good, keep going')
        self.optAdd = QtWidgets.QRadioButton('Add a Type B uncertainty')
        self.optMod = QtWidgets.QRadioButton('Modify something')
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.title)
        layout.addSpacing(20)
        layout.addWidget(self.uncsummary, stretch=2)
        layout.addSpacing(20)
        layout.addWidget(self.optAdd)
        layout.addWidget(self.optMod)
        layout.addWidget(self.optGood)
        self.setLayout(layout)

    def initialize(self):
        varname = self.wizard.currentvar
        self.title.setvarname(varname)
        randvar = self.wizard.current_randvar()

        rpt = report.Report()
        hdr = ['Name', 'Standard Uncertainty', 'Distribution', 'Degrees of Freedom']
        rpt.txt(f'Expected value: {randvar.expected}\n\n')
        rpt.txt(f'Combined standard uncertainty: {report.Number(randvar.uncertainty)}\n\n')
        rpt.txt('Uncertainty Components:\n\n')
        typea = np.sqrt(randvar._typea_variance_ofmean())
        rows = []
        if typea == 0 and len(randvar.typeb_names) == 0:
            rpt.txt('• No Uncertainties')
        else:
            if typea > 0:
                rows.append(['Type A', report.Number(typea, fmin=0), 'normal', randvar.value.size-1])
            for typeb in randvar._typeb:
                rows.append([typeb.name, report.Number(typeb.uncertainty, fmin=0),
                             typeb.distname, report.Number(typeb.degf, fmin=0)])
            rpt.table(rows, hdr)

        if len(randvar.typeb_names) == 0:
            self.optAdd.setChecked(True)
            self.optAdd.setFocus()
        else:
            self.optGood.setChecked(True)
            self.optGood.setFocus()
        self.uncsummary.setReport(rpt)

    def gonext(self):
        if self.optGood.isChecked():
            missingvars = self.wizard.projitem.missingvars
            if len(missingvars) == 0:
                # All inputs defined, go to summary
                self.wizard.currentvar = None
                units = self.wizard.projitem.model.variables.units
                if all(u is None or unitmgr.is_dimensionless(u) for u in units.values()):
                    page = Pages.SUMMARY
                else:
                    page = Pages.UNITS
            else:
                self.wizard.currentvar = missingvars[0]
                page = Pages.DATATYPE
        elif self.optAdd.isChecked():
            page = Pages.TYPEB
        else:
            randvar = self.wizard.current_randvar()
            datatype = self.wizard.projitem.inpt_type(self.wizard.currentvar)
            dlg = ModifySelect(randvar, datatype, parent=self)
            ok = dlg.exec_()
            if ok:
                comp = dlg.get_comp()
                if comp == 'measured' and datatype == VariableDataType.SINGLE:
                    page = Pages.SINGLE
                elif comp == 'measured' and datatype == VariableDataType.REPEATABILITY:
                    page = Pages.REPEAT
                elif comp == 'measured' and datatype == VariableDataType.REPRODUCIBILITY:
                    page = Pages.REPROD
                else:
                    page = Pages.TYPEB
                    self.wizard.currentunc = comp.name
            else:
                page = Pages.UNCERTS
        return page

    def gethelp(self):
        rpt = report.Report()
        rpt.hdr('Variable Uncertainty Review', level=2)
        rpt.txt('This screen summarizes the information provided about the variable so far, '
                'including the expected (mean) value, combined standard uncertainty, and any '
                'uncertainty components.\n\nA variable may have one Type A uncertainty (calculated '
                'from repeatability and reproducibility values), and any number of Type B '
                'uncertainties. Select the **Add a Type B uncertainty** option to add more Type B '
                'uncertainty components, or **Modify something** to change the parameters of '
                'an already entered component.')
        return rpt


class ModifySelect(QtWidgets.QDialog):
    ''' List of values that can be modified '''
    def __init__(self, randvar, datatype, parent=None):
        super().__init__(parent=parent)
        self.setWindowTitle('Suncal')
        self.randvar = randvar
        self.datatype = datatype
        self.buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(QtWidgets.QLabel('Select component to modify'))
        self.opts = []
        if self.datatype == VariableDataType.SINGLE:
            self.opts.append(QtWidgets.QRadioButton('Nominal Value'))
            layout.addWidget(self.opts[-1])
        else:
            self.opts.append(QtWidgets.QRadioButton('Measured Data (Type A)'))
            layout.addWidget(self.opts[-1])
        for typebname in self.randvar.typeb_names:
            self.opts.append(QtWidgets.QRadioButton(typebname))
            layout.addWidget(self.opts[-1])
        self.opts[0].setChecked(True)
        layout.addStretch()
        layout.addWidget(self.buttons)
        self.setLayout(layout)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)

    def get_comp(self):
        ''' Get the selected uncertainty component '''
        vals = [w.isChecked() for w in self.opts]
        if vals[0]:
            return 'measured'

        return self.randvar._typeb[vals.index(True)-1]


class DistributionPlotWidget(QtWidgets.QWidget):
    ''' Widget with plot of probability distribution '''
    def __init__(self, typeb):
        super().__init__()
        self.setFixedSize(300, 200)
        self.typeb = typeb
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setStyleSheet("background-color:transparent;")
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.canvas)
        layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)
        self.replot(self.typeb)

    def replot(self, typeb):
        self.typeb = typeb
        if self.typeb is not None:
            with plt.style.context({'font.size': 8}):
                self.figure.clf()
                ax = self.figure.add_subplot(1, 1, 1)
                x, y = self.typeb.pdf()
                x = unitmgr.strip_units(x)
                ax.plot(x, y)
                ax.yaxis.set_ticks([])
                self.figure.tight_layout()
            self.canvas.draw_idle()


class DistributionWidget(QtWidgets.QWidget):
    ''' Widget for editing a Type B uncertainty distribution '''
    changed = QtCore.pyqtSignal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.typeb = Typeb(name='Type B')
        self.name = QtWidgets.QLineEdit('Type B')
        self.cmbdist = QtWidgets.QComboBox()
        self.cmbdist.addItems(['normal', 'uniform', 'triangular', 'arcsine', 'resolution'])
        self.unclabel = QtWidgets.QLabel('± Uncertainty')
        self.covlabel = QtWidgets.QLabel('Coverage Factor')
        self.conflabel = QtWidgets.QLabel('Confidence')
        self.txtvalue = QtWidgets.QLineEdit('1%')
        self.unitsvalue = QtWidgets.QLineEdit('dimensionless')
        self.kvalue = QtWidgets.QLineEdit('1.96')
        self.confvalue = QtWidgets.QLineEdit('95%')
        self.degfvalue = QtWidgets.QLineEdit('inf')
        self.message = QtWidgets.QLabel()
        self.plot = DistributionPlotWidget(self.typeb)

        kvalidator = QtGui.QDoubleValidator()
        kvalidator.setRange(0, 1000, 3)
        self.kvalue.setValidator(kvalidator)
        self.degfvalue.setValidator(kvalidator)
        pctvalidator = QtGui.QRegExpValidator(QtCore.QRegExp(r'\d*(\.)?\d+%?'))
        self.confvalue.setValidator(pctvalidator)

        # TODO: implement G.4.2, degf approximation. Maybe apply a warning if uB > 2uA and infinite?
        # Would need to come later though?

        layout = QtWidgets.QHBoxLayout()
        flayout = QtWidgets.QFormLayout()
        flayout.addRow(self.unclabel, self.txtvalue)
        flayout.addRow('Distribution', self.cmbdist)
        flayout.addRow('Units', self.unitsvalue)
        flayout.addRow('Degrees of Freedom', self.degfvalue)
        flayout.addRow(self.covlabel, self.kvalue)
        flayout.addRow(self.conflabel, self.confvalue)
        flayout.addRow('Name', self.name)
        flayout.addRow('', self.message)
        vlayout = QtWidgets.QVBoxLayout()
        vlayout.addLayout(flayout)
        vlayout.addWidget(self.message)
        layout.addLayout(vlayout)
        layout.addWidget(self.plot)
        self.setLayout(layout)

        self.cmbdist.currentIndexChanged.connect(self.changedist)
        self.name.editingFinished.connect(self.changename)
        self.txtvalue.editingFinished.connect(self.changeparam)
        self.unitsvalue.editingFinished.connect(self.changeparam)
        self.kvalue.editingFinished.connect(self.kchanged)
        self.confvalue.editingFinished.connect(self.confchanged)
        self.degfvalue.editingFinished.connect(self.kchanged)

    def changename(self):
        self.typeb.name = self.name.text()

    def changedist(self, index):
        ''' Distribution changed '''
        if self.cmbdist.currentText() in ['normal', 't']:
            self.covlabel.setVisible(True)
            self.conflabel.setVisible(True)
            self.kvalue.setVisible(True)
            self.confvalue.setVisible(True)
        else:
            self.covlabel.setVisible(False)
            self.conflabel.setVisible(False)
            self.kvalue.setVisible(False)
            self.confvalue.setVisible(False)

        if self.cmbdist.currentText() in ['resolution']:
            self.unclabel.setText('Smallest Increment')
        else:
            self.unclabel.setText('± Uncertainty')
        self.changeparam()

    def kchanged(self):
        ''' Coverage value changed, update confidence '''
        k = float(self.kvalue.text())
        conf = ttable.confidence(k, float(self.degfvalue.text()))
        self.confvalue.setText(f'{conf*100:.2f}%')
        self.changeparam()

    def confchanged(self):
        ''' Coverage value changed, update confidence '''
        conf = float(self.confvalue.text().rstrip('%'))/100
        k = ttable.k_factor(conf, float(self.degfvalue.text()))
        self.kvalue.setText(f'{k:.2f}')
        self.changeparam()

    def changeparam(self):
        ''' Parameter changed, update distribution '''
        params = {'degf': float(self.degfvalue.text())}
        params['dist'] = self.cmbdist.currentText()
        self.typeb.degf = params['degf']
        value = self.txtvalue.text()
        if self.cmbdist.currentText() in ['normal', 't']:
            params['k'] = float(self.kvalue.text())
            params['unc'] = value
        else:
            params['a'] = value
        self.typeb.set_kwargs(**params)

        unitsvalid = True
        units = self.unitsvalue.text()
        if units is not None:
            try:
                units = unitmgr.parse_units(self.unitsvalue.text())
            except (PintError, ValueError):
                unitsvalid = False
                self.message.setText(f'<font color="red">Undefined Units: {self.unitsvalue.text()}</font>')
        else:
            units = unitmgr.dimensionless

        if self.parentunits is None:
            parentunits = unitmgr.dimensionless
        else:
            parentunits = self.parentunits

        if unitsvalid and units.dimensionality != parentunits.dimensionality:
            self.message.setText(f'<font color="red">Units Error: cannot convert '
                                 f'{units} to {parentunits}.</font>')
            unitsvalid = False
        elif unitsvalid:
            try:
                1*units + 1*parentunits  # Will fail on offest units such as temperature
            except OffsetUnitCalculusError:
                self.message.setText(f'<font color="red">Ambiguous unit {units}. Try "delta_{units}"</font>')
                unitsvalid = False

        if unitsvalid:
            self.typeb.units = units
            self.message.setText('')

        valid = self.typeb.isvalid()
        if valid:
            self.plot.replot(self.typeb)
        elif unitsvalid:
            self.message.setText('<font color="red">Error in distribution parameters</font>')
        self.changed.emit(valid and unitsvalid)

    def set_parent_units(self, units):
        self.parentunits = units  # pint units, not str

    def setdefault(self, nom=0):
        ''' Restore fields to defaults '''
        units = str(self.parentunits) if self.parentunits else None
        if self.parentunits and self.parentunits in [unitmgr.ureg.degC,
                                                     unitmgr.ureg.celsius,
                                                     unitmgr.ureg.degree_Celsius]:
            units = 'delta_degree_Celsius'
        if self.parentunits and self.parentunits in [unitmgr.ureg.degF,
                                                     unitmgr.ureg.fahrenheit,
                                                     unitmgr.ureg.degree_Fahrenheit]:
            units = 'delta_degree_Fahrenheit'
        self.typeb = Typeb(name='Type B', nominal=nom, units=units)
        self.txtvalue.setText('1%')
        self.cmbdist.setCurrentIndex(0)
        self.degfvalue.setText('inf')
        self.kvalue.setText('1.96')
        self.confvalue.setText('95%')
        self.message.setText('')
        self.unitsvalue.setText(units)
        self.plot.replot(self.typeb)


class PageTypeB(Page):
    ''' Page to enter Type B uncertainty '''
    def __init__(self, parent=None):
        super().__init__(parent)
        self.title = VarnameTitle('Enter Type B uncertainty for', varname='x', title2='')
        self.dist = DistributionWidget()
        self.backenabled = True
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.title)
        layout.addWidget(self.dist)
        layout.addStretch()
        self.setLayout(layout)
        self.dist.changed.connect(lambda x: self.wizard.btnNext.setEnabled(x))

    def initialize(self):
        varname = self.wizard.currentvar
        randvar = self.wizard.current_randvar()
        self.title.setvarname(varname, str(randvar.expected))
        parentunits = unitmgr.split_units(randvar.expected)[1]
        self.dist.set_parent_units(parentunits)
        if self.wizard.currentunc is None or self.wizard.currentunc not in randvar.typeb_names:
            # New component
            uncertnames = randvar.typeb_names
            if len(uncertnames) == 0:
                uname = 'Type B'
            else:
                uname = f'Type B {len(uncertnames)}'
            self.dist.setdefault(randvar.expected)
            self.dist.name.setText(uname)
        else:
            # Existing component
            typeb = randvar.get_typeb(self.wizard.currentunc)
            self.dist.typeb = typeb
            self.dist.name.setText(typeb.name)
            self.dist.cmbdist.setCurrentIndex(self.dist.cmbdist.findText(typeb.distname))
            self.dist.kvalue.setText(str(typeb.kwargs.get('k', 1.96)))
            self.dist.confvalue.setText(str(typeb.kwargs.get('conf', 95)))
            self.dist.unitsvalue.setText(str(typeb.units))
            self.dist.degfvalue.setText(str(typeb.degf))
            self.dist.txtvalue.setText(str(typeb.kwargs.get('unc', '1%')))
            self.dist.plot.replot(self.dist.typeb)

    def gonext(self):
        self.dist.changeparam()  # Make sure to save any changes
        randvar = self.wizard.current_randvar()
        uname = self.dist.name.text()
        unames = randvar.typeb_names
        idx = len(unames)
        if uname in unames:
            idx = unames.index(uname)
            randvar._typeb.pop(idx)
        randvar._typeb.insert(idx, self.dist.typeb)
        self.wizard.currentunc = None
        return Pages.UNCERTS

    def goback(self):
        return Pages.UNCERTS

    def gethelp(self):
        rpt = report.Report()
        rpt.hdr('Type B Uncertainty', level=2)
        rpt.txt('Type B uncertainties are defined using a probability distribution based '
                'on the information that is given about the uncertainty. '
                'On this screen, a distribution type and its parameters may be entered. '
                'The **Name** field may be used for your own housekeeping, but does not '
                'affect the calculation. Choice of distribution depends on the information '
                'available about the uncertainty:\n\n'
                '- **normal**: The default choice, use when given an uncertainty *with* a '
                'confidence and/or coverage factor\n- **uniform**: Use when given an uncertainty '
                'as a plus-or-minus value with no other information\n- **triangular**: Use when '
                'given an uncertainty as a plus-or-minus value, and the value is more likely '
                'to be near the center of the tolerance range\n- **arcsine**: Use for values '
                'that flucuate sinusoidally\n- **resolution**: Use for entering equipment '
                'resolution. Same shape as uniform.\n\n')
        rpt.txt('The **± Uncertainty** field can do basic calculations. Values may be entered '
                "as a number, as a percent (of the variable's expected value), or as a simple "
                'formula. It also understands "%range()" for entry of manufacturer specifications '
                'given as "%reading + %range" values. For example, enter "1% + 0.01%range(100)" for '
                'an un uncertainty on the 100V range of a meter.\n\n')
        rpt.txt('The **Units** field is automatically populated with the same units as the variable, '
                'however it may be changed to any compatible units.')
        rpt.txt('**Degrees of Freedom** may be entered if known, but is often assumed infinite '
                'for Type B uncertainties')
        return rpt


class DataEntry(QtWidgets.QDialog):
    ''' Dialog for entering Type A measurement data '''
    def __init__(self, title='', data=None, multicol=False, parent=None):
        super().__init__(parent=parent)
        self.setWindowTitle('Suncal - Enter Data')
        gui_widgets.centerWindow(self, 700, 500)
        font = self.font()
        font.setPointSize(12)
        self.setFont(font)
        self.title = QtWidgets.QLabel(title)
        self.table = gui_widgets.FloatTableWidget(paste_multicol=multicol, movebyrows=True)
        self.table.setColumnCount(1)
        self.btnok = QtWidgets.QPushButton()
        self.btnok.setIcon(self.style().standardIcon(QtWidgets.QStyle.SP_DialogApplyButton))
        self.btnaddcol = gui_widgets.TreeButton('+')
        self.btnaddcol.setVisible(multicol)

        layout = QtWidgets.QVBoxLayout()
        tlayout = QtWidgets.QHBoxLayout()
        tlayout.addWidget(self.title)
        tlayout.addStretch()
        tlayout.addWidget(self.btnaddcol)
        layout.addLayout(tlayout)
        layout.addWidget(self.table)
        hlayout = QtWidgets.QHBoxLayout()
        hlayout.addStretch()
        hlayout.addWidget(self.btnok)
        layout.addLayout(hlayout)
        self.setLayout(layout)
        self.btnok.clicked.connect(self.accept)
        self.btnaddcol.clicked.connect(lambda x: self.table.setColumnCount(self.table.columnCount()+1))

        if data is not None and len(data) > 1:
            data = np.atleast_2d(data)
            for i, col in enumerate(data):
                self.table.setColumnCount(i+1)
                for j, row in enumerate(col):
                    self.table.setRowCount(max(self.table.rowCount(), j+1))
                    self.table.setItem(j, i, QtWidgets.QTableWidgetItem(str(row)))


class PageRepeat(Page):
    ''' Page for entering Type A repeatability data '''
    def __init__(self, parent=None):
        super().__init__(parent)
        self.backenabled = True
        self.data = None
        self.title = VarnameTitle('Set parameters of repeatability data for', varname='x', title2='')
        self.btnEditData = QtWidgets.QPushButton('Edit Data')
        self.btnEditData.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)

        self.units = QtWidgets.QLineEdit('')
        self.units.setMaximumWidth(250)
        self.chkNewmeas = QtWidgets.QCheckBox('Use this data to estimate uncertainty in')
        self.chkAutocorr = QtWidgets.QCheckBox('Autocorrelation appears significant. Check here to account for it.')
        self.Nnewmeas = QtWidgets.QSpinBox()
        self.Nnewmeas.setRange(1, 999)
        self.Nlabel = QtWidgets.QLabel('new measurements')
        self.stats = gui_widgets.MarkdownTextEdit()
        self.fig = Figure()
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setStyleSheet("background-color:transparent;")
        self.canvas.setFixedSize(300, 200)

        hlayout = QtWidgets.QHBoxLayout()
        hlayout.addWidget(self.canvas)
        hlayout.addSpacing(25)
        hlayout.addWidget(self.stats)
        flayout = QtWidgets.QFormLayout()
        flayout.addRow('Units', self.units)
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.title)
        layout.addLayout(hlayout)
        layout.addStretch()
        layout.addLayout(flayout)
        nlayout = QtWidgets.QHBoxLayout()
        nlayout.addWidget(self.chkNewmeas)
        nlayout.addWidget(self.Nnewmeas)
        nlayout.addWidget(self.Nlabel)
        nlayout.addStretch()
        layout.addLayout(nlayout)
        layout.addWidget(self.chkAutocorr)
        layout.addSpacing(20)
        layout.addWidget(self.btnEditData)
        layout.addStretch()
        self.chkAutocorr.setVisible(False)  # Only show if significant

        self.setLayout(layout)
        self.units.editingFinished.connect(self.parsedata)
        self.chkNewmeas.stateChanged.connect(self.parsedata)
        self.chkAutocorr.stateChanged.connect(self.parsedata)
        self.Nnewmeas.valueChanged.connect(self.parsedata)
        self.btnEditData.clicked.connect(self.popupdata)

    def initialize(self):
        varname = self.wizard.currentvar
        self.title.setvarname(varname)
        self.data = self.wizard.projitem.model.var(varname).value
        self.Nnewmeas.setValue(1)
        units = unitmgr.split_units(self.data)[1]
        if units is not None and not unitmgr.is_dimensionless(units):
            self.units.setText(str(units))
        else:
            self.units.setText('')
        self.popupdata()

    def popupdata(self):
        ''' Show data entry '''
        dlg = DataEntry('Type or paste repeated measurements', data=self.data)
        dlg.exec_()
        data = dlg.table.get_column(0)
        self.data = data[np.isfinite(data)]

        if len(self.data) > 50:  # useful minimum N suggested by Zhang
            self.autocor = uncert_autocorrelated(self.data)
            if self.autocor.r_unc > 1.3:  # What's the threshold?
                self.chkAutocorr.setVisible(True)
                self.chkAutocorr.setChecked(True)
        else:
            self.autocor = None
            self.chkAutocorr.setVisible(False)
            self.chkAutocorr.setChecked(False)

        self.parsedata()

    def parsedata(self):
        ''' Parse the Type A data '''
        rpt = report.Report()
        if self.data is not None and len(self.data) > 0:
            mean = report.Number(self.data.mean(), fmin=1)
            std = report.Number(self.data.std(ddof=1), fmin=1)
            rows = [['Mean', mean],
                    ['Standard Deviation', std]]

            if self.chkNewmeas.isChecked():
                nmeas = self.Nnewmeas.value()
            else:
                nmeas = len(self.data)

            if self.autocor is not None and self.chkAutocorr.isChecked():
                sem = self.autocor.uncert * np.sqrt(len(self.data)/nmeas)  # r is already SEM
                sem = report.Number(sem, fmin=1)
                semeq = f'rσ/√{nmeas}'
                rows.append(['Autocorrelation Factor', f'r ={self.autocor.r_unc:.3f}'])
            else:
                semeq = f'σ/√{nmeas}'
                sem = report.Number(std.value/np.sqrt(nmeas), fmin=1)
            rows.append(['Uncertainty', f'{semeq} = {sem}'])
            rpt.table(rows, hdr=['Parameter', 'Value'])

            unitvalid = True
            unit = ''
            if self.units.text():
                try:
                    unit = unitmgr.parse_units(self.units.text())
                except (PintError, ValueError):
                    rpt.txt(f'<font color="red">Undefined Units: {self.units.text()}</font>')
                    unitvalid = False
                else:
                    rpt.txt(f'Units: {unit}</font>')

            if unitvalid:
                # TODO: if autocorr, maybe have a button to pop up lag plot?
                self.fig.clf()
                self.fig.gca().hist(self.data)
                self.fig.gca().set_xlabel(str(unit))
                self.fig.tight_layout()
                self.canvas.draw_idle()
                self.wizard.btnNext.setEnabled(True)
            else:
                self.wizard.btnNext.setEnabled(False)
        else:
            rpt.txt('<font color="red">No Data Entered</font>')
            self.wizard.btnNext.setEnabled(False)
        self.stats.setReport(rpt)

    def gonext(self):
        varname = self.wizard.currentvar
        if self.chkNewmeas.isChecked():
            nmeas = self.Nnewmeas.value()
        else:
            nmeas = len(self.data)
        self.wizard.projitem.set_inputval(varname, self.data,
                                          num_newmeas=nmeas,
                                          autocor=self.chkAutocorr.isChecked(),
                                          units=self.units.text())
        return Pages.UNCERTS

    def goback(self):
        return Pages.DATATYPE

    def gethelp(self):
        rpt = report.Report()
        rpt.hdr('Repeatability Data (Type A Uncertainty)', level=2)
        rpt.txt('The expected (mean) value and uncertainty of the variable are estimated '
                'using the repeated measurements. The plot shows a histogram of the repeated '
                'measurement values for reference. The uncertainty is calculated as the '
                'standard error of the mean (standard deviation divided by the square '
                'root of the number of measurements).\n\n'
                'The screen provides a field to enter any measurement units for the measurements. '
                'Check the **Use this data to estimate uncertainty** box if the repeated measurements '
                'were part of a repeatability study to apply the uncertainty to other future identical '
                'measurements. Leave it unchecked to compute the uncertainty in that specific set '
                'of repeated measurements. The **Edit Data** button provides an option to change '
                'the measurement data.\n\n')
        rpt.txt('When there are more than 30 measurements in the data, Suncal will check '
                'for autocorrelation between the measurements and apply an uncertainty '
                'correction if autocorrelation is significant. Uncheck **Autocorrelation appears '
                'significant** to disable this correction.')
        return rpt


class PageReprod(Page):
    ''' Page for entering Type A reproducibility data '''
    def __init__(self, parent=None):
        super().__init__(parent)
        self.backenabled = True
        self.data = None
        self.title = VarnameTitle('Set parameters of reproducibility data for', varname='x', title2='')
        self.btnEditData = QtWidgets.QPushButton('Edit Data')
        self.btnEditData.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)

        self.units = QtWidgets.QLineEdit('')
        self.units.setMaximumWidth(250)
        self.chkNewmeas = QtWidgets.QCheckBox('Use this data to estimate uncertainty in')
        self.Nnewmeas = QtWidgets.QSpinBox()
        self.Nnewmeas.setRange(1, 999)
        self.Nlabel = QtWidgets.QLabel('new measurements')
        self.stats = gui_widgets.MarkdownTextEdit()
        self.fig = Figure()
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setStyleSheet("background-color:transparent;")
        self.canvas.setFixedSize(300, 200)

        hlayout = QtWidgets.QHBoxLayout()
        hlayout.addWidget(self.canvas)
        hlayout.addSpacing(25)
        hlayout.addWidget(self.stats)
        flayout = QtWidgets.QFormLayout()
        flayout.addRow('Units', self.units)
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.title)
        layout.addLayout(hlayout)
        layout.addStretch()
        layout.addLayout(flayout)
        nlayout = QtWidgets.QHBoxLayout()
        nlayout.addWidget(self.chkNewmeas)
        nlayout.addWidget(self.Nnewmeas)
        nlayout.addWidget(self.Nlabel)
        nlayout.addStretch()
        layout.addLayout(nlayout)
        layout.addSpacing(20)
        layout.addWidget(self.btnEditData)
        layout.addStretch()

        self.setLayout(layout)
        self.units.editingFinished.connect(self.parsedata)
        self.chkNewmeas.stateChanged.connect(self.parsedata)
        self.Nnewmeas.valueChanged.connect(self.parsedata)
        self.btnEditData.clicked.connect(self.popupdata)

    def initialize(self):
        varname = self.wizard.currentvar
        self.title.setvarname(varname)
        self.data = self.wizard.projitem.model.var(varname).value
        self.Nnewmeas.setValue(1)
        units = unitmgr.split_units(self.data)[1]
        if units is not None and not unitmgr.is_dimensionless(units):
            self.units.setText(str(units))
        else:
            self.units.setText('')
        self.popupdata()

    def popupdata(self):
        ''' Show data entry '''
        data = self.data.data if self.data is not None else None
        dlg = DataEntry('Type or paste reproducibility measurements', data=data, multicol=True)
        dlg.exec_()
        data = dlg.table.get_table()
        self.data = DataSet(data)
        self.parsedata()

    def parsedata(self):
        ''' Parse the Type A data '''
        rpt = report.Report()
        if self.data is not None and self.data.ncolumns() > 0:
            stderr = self.data.standarderror()
            mean = self.data.mean()
            sem = report.Number(stderr.standarderror, fmin=1)
            if self.chkNewmeas.isChecked():
                nmeas = self.Nnewmeas.value()
                sem = report.Number(stderr.standarddeviation / np.sqrt(nmeas), fmin=1)

            rows = [
                ['Groups', str(self.data.ncolumns())],
                ['Mean', report.Number(mean, fmin=1)],
                ['Degrees of Freedom', report.Number(stderr.degf, fmin=1)],
                ['Uncertainty', sem]]
            rpt.table(rows, ['Parameter', 'Value'])
            if stderr.reprod_significant:
                rpt.txt('\nReproducibility is significant')
            else:
                rpt.txt('\nReproducibility is insignificant')

            unitvalid = True
            if self.units.text():
                try:
                    unit = unitmgr.parse_units(self.units.text())
                except (PintError, ValueError):
                    rpt.txt(f'\n\n<font color="red">Undefined Units: {self.units.text()}</font>')
                    unitvalid = False
                else:
                    rpt.txt(f'\n\nUnits: {unit}</font>')

            if unitvalid:
                self.fig.clf()
                ReportDataSet(self.data).plot.groups(self.fig)
                self.fig.tight_layout()
                self.canvas.draw_idle()
                self.wizard.btnNext.setEnabled(np.isfinite(mean))
            else:
                self.wizard.btnNext.setEnabled(False)
        else:
            rpt.txt('<font color="red">No Data Entered</font>')
            self.wizard.btnNext.setEnabled(False)
        self.stats.setReport(rpt)

    def gonext(self):
        varname = self.wizard.currentvar
        nmeas = None
        if self.chkNewmeas.isChecked():
            nmeas = self.Nnewmeas.value()
        self.wizard.projitem.set_inputval(varname, self.data.data,
                                          num_newmeas=nmeas,
                                          units=self.units.text())
        return Pages.UNCERTS

    def goback(self):
        return Pages.DATATYPE

    def gethelp(self):
        rpt = report.Report()
        rpt.hdr('Reproducibility Data (Type A Uncertainty)', level=2)
        rpt.txt('The expected (mean) value and uncertainty of the variable are estimated '
                'using the reproducibility measurements. The plot shows a the mean and standard '
                'deviation of each measurement group for reference. The uncertainty is calculated '
                'using the reproducibility if significant, or repeatability otherwise. See GUM '
                'Appendix H.2 for an example.\n\n'
                'The screen provides a field to enter any measurement units for the measurements. '
                'Check the **Use this data to estimate uncertainty** box if the measurements '
                'were part of a reproducibility study to apply the uncertainty to other future identical '
                'measurements. Leave it unchecked to compute the uncertainty in that specific set '
                'of repeated measurements. The **Edit Data** button provides an option to change '
                'the measurement data.\n\n')
        return rpt


class PageUnits(Page):
    ''' Page for verifying/entering output units '''
    def __init__(self, parent=None):
        super().__init__(parent)
        self.title = QtWidgets.QLabel('Verify Units')
        self.lblnatural = QtWidgets.QLabel()
        self.outunits = QtWidgets.QLineEdit()
        self.message = QtWidgets.QLabel()
        self.naturalunits = unitmgr.ureg.dimensionless
        self.backenabled = True
        self.lblconvert = QtWidgets.QLabel('Convert output to units:')
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.title)
        layout.addWidget(self.lblnatural)
        flayout = QtWidgets.QHBoxLayout()
        flayout.addWidget(self.lblconvert)
        flayout.addWidget(self.outunits)
        flayout.addStretch()
        layout.addLayout(flayout)
        layout.addWidget(self.message)
        layout.addStretch()
        self.setLayout(layout)
        self.outunits.editingFinished.connect(self.checkunits)

    def initialize(self):
        try:
            value = list(self.wizard.projitem.model.eval().values())[0]  # Only one output
        except PintError as exc:
            self.lblnatural.setText('<font color="red">Units error in inputs:</font>')
            self.message.setText(str(exc))
            self.wizard.btnNext.setEnabled(False)
            self.outunits.setVisible(False)
            self.lblconvert.setVisible(False)
        else:
            self.naturalunits = unitmgr.split_units(value)[1]
            unitstr = format(self.naturalunits, 'D')
            self.lblnatural.setText(f'Natural units of measurement model are {unitstr}')
            self.outunits.setText(unitstr)
            self.wizard.btnNext.setEnabled(True)

    def checkunits(self):
        '''Check the units and display message if incompatible '''
        self.wizard.projitem.outunits = {self.wizard.projitem.model.functionnames[0]: self.outunits.text()}
        try:
            newunits = unitmgr.parse_units(self.outunits.text())
        except PintError:
            self.message.setText(f'<font color="red">Undefined units: {self.outunits.text()}')
            return

        if not unitmgr.is_dimensionless(self.naturalunits):
            unitstr = report.Unit(newunits, abbr=False).prettytext()
            if self.naturalunits.is_compatible_with(newunits):
                self.message.setText(f'Output Units: {unitstr}')
                self.wizard.btnNext.setEnabled(True)
            else:
                self.message.setText(f'<font color="red">Units Incompatible: {unitstr}</font>')
                self.wizard.btnNext.setEnabled(False)
        else:
            self.wizard.btnNext.setEnabled(True)

    def gonext(self):
        self.wizard.projitem.outunits = {self.wizard.projitem.model.functionnames[0]: self.outunits.text()}
        return Pages.SUMMARY

    def goback(self):
        return Pages.VAR_MODIFY

    def gethelp(self):
        rpt = report.Report()
        rpt.hdr('Units Conversion', level=2)
        rpt.txt('Suncal determines the units of the measurment function automatically. '
                'This screen provides an entry to convert the output to different, but compatible, '
                'units. The unit converter has no way to unambiguously determine the units '
                'of the output, so for example, it may suggest units of "V/A" instead of "Ω" when '
                "computing resistance using Ohm's Law. This page allows you to change those units.")
        return rpt


class PageSummary(Page):
    ''' Final page before calculating '''
    def __init__(self, parent=None):
        super().__init__(parent)
        self.title = QtWidgets.QLabel('Ready to calculate! Summary of variables:')
        self.report = gui_widgets.MarkdownTextEdit()
        self.backenabled = True
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.title)
        layout.addWidget(self.report)
        self.setLayout(layout)

    def initialize(self):
        rpt = report.Report()
        hdr = ['Variable', 'Mean', 'Standard Uncertainty', 'Degrees of Freedom']
        rows = []
        for varname in self.wizard.projitem.model.variables.names:
            var = self.wizard.projitem.model.var(varname)
            rows.append([report.Math.from_latex(varname),
                         str(var.expected),
                         report.Number(var.uncertainty, fmin=0),
                         report.Number(var.degrees_freedom, fmin=0)])
        rpt.table(rows, hdr)
        self.report.setReport(rpt)

    def gonext(self):
        self.wizard.projitem.calculate()
        return Pages.OUTPUT

    def goback(self):
        return Pages.VAR_MODIFY

    def gethelp(self):
        rpt = report.Report()
        rpt.hdr('Uncertainty Model Review', level=2)
        rpt.txt('At this point, everything is set up and ready to calculate. A summary of the '
                'measurement model and variables is provided here. Use the **Back** button '
                'if anything needs to be changed, otherwise go foward to run the calculation '
                'and see the results.')
        return rpt


class PageVariableSelect(Page):
    ''' Display all variables in model and pick one to work on '''
    def __init__(self, parent=None):
        super().__init__(parent)
        self.title = QtWidgets.QLabel('Which variable do you want to modify?')
        self.backenabled = True
        self.opts = []
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.title)
        layout.addSpacing(20)
        layout.addStretch()
        self.setLayout(layout)

    def initialize(self):
        inputnames = self.wizard.projitem.model.variables.names
        layout = self.layout()
        for opt in self.opts:
            layout.removeWidget(opt)
            opt.deleteLater()
        self.opts = []

        for name in reversed(inputnames):
            self.opts.append(QtWidgets.QRadioButton(name, parent=self))
            layout.insertWidget(1, self.opts[-1])
        layout.addStretch()
        self.opts[-1].setChecked(True)
        self.opts[-1].setFocus()

    def gonext(self):
        name = [opt.text() for opt in self.opts if opt.isChecked()]
        if len(name) > 0:
            self.wizard.currentvar = name[0]
            if name[0] in self.wizard.projitem.variablesdone:
                return Pages.UNCERTS
            else:
                return Pages.DATATYPE
        return Pages.MODEL

    def goback(self):
        return Pages.SUMMARY

    def gethelp(self):
        rpt = report.Report()
        rpt.hdr('Modify a Value', level=2)
        rpt.txt('Select the variable to modify.')
        return rpt


def main():
    ''' Start the wizard as standalone app '''
    import sys
    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling)
    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps)
    app = QtWidgets.QApplication(sys.argv)

    wiz = UncertWizard()
    wiz.show()
    app.exec_()


if __name__ == '__main__':
    main()
