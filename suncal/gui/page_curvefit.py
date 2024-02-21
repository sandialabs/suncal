''' Page for fitting curve to experimental data '''

from contextlib import suppress
import logging
from PyQt6 import QtWidgets, QtGui, QtCore
import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import matplotlib.dates as mdates

from ..common import report, plotting
from ..curvefit import fitparse
from . import gui_styles
from .gui_common import BlockedSignals
from . import gui_math
from . import widgets
from . import page_dataimport
from .page_csvload import SelectCSVData
from .help_strings import CurveHelp


class ModelWidget(QtWidgets.QWidget):
    ''' Widget for configuring the fit model (line, poly, etc.) '''
    def __init__(self, component, parent=None):
        super().__init__(parent=parent)
        self.component = component
        self.use_ux = False
        self.customfunc = None
        self.customargs = []

        self.cmbModel = QtWidgets.QComboBox()
        self.cmbModel.addItems(['Line', 'Polynomial', 'Exponential', 'Exponential Decay',
                                'Exponential Decay (rate)', 'Log', 'Logistic Growth', 'Custom'])
        self.polyorder = widgets.SpinWidget(label='Order:  ')
        self.polyorder.spin.setRange(2, 12)
        self.polyorder.setVisible(False)
        self.lblEquation = QtWidgets.QLabel()  # To show rendered equation describing model
        self.custom = QtWidgets.QLineEdit('a + b*x')
        self.lblCustom = QtWidgets.QLabel('Expression:')
        self.custom.setVisible(False)
        self.lblCustom.setVisible(False)

        self.btnVert = QtWidgets.QRadioButton('Minimize Vertical Distances')
        self.btnODR = QtWidgets.QRadioButton('Minimize Orthogonal Distances')
        self.btnVert.setChecked(True)
        self.chkGuess = QtWidgets.QCheckBox('Provide initial guess')
        self.chkGuess.setVisible(False)
        self.tblGuess = QtWidgets.QTableWidget()
        self.tblGuess.setColumnCount(2)
        self.tblGuess.setHorizontalHeaderLabels(['Parameter', 'Initial Guess'])
        self.tblGuess.setVisible(False)
        self.tblGuess.cellChanged.connect(self.update_model)

        layout = QtWidgets.QVBoxLayout()
        hlayout = QtWidgets.QHBoxLayout()
        hlayout.addWidget(QtWidgets.QLabel('Function:'))
        hlayout.addWidget(self.cmbModel)
        hlayout.addWidget(self.polyorder)
        hlayout.addWidget(self.lblCustom)
        hlayout.addWidget(self.custom)
        hlayout.addStretch()
        layout.addLayout(hlayout)
        layout.addWidget(self.lblEquation)
        layout.addWidget(self.btnVert)
        layout.addWidget(self.btnODR)
        layout.addWidget(self.chkGuess)
        layout.addWidget(self.tblGuess)
        layout.addStretch()
        self.setLayout(layout)

        self.cmbModel.currentIndexChanged.connect(self.modelchange)
        self.btnODR.toggled.connect(self.modelchange)
        self.btnVert.toggled.connect(self.modelchange)
        self.polyorder.valueChanged.connect(self.modelchange)
        self.chkGuess.toggled.connect(self.modelchange)
        self.custom.editingFinished.connect(self.update_custom)
        self.showhide()

    def modelchange(self):
        ''' Model selection changed by user. Show/hide fields and update
            the Project Item
        '''
        self.update_model()
        self.showhide()

    def showhide(self):
        ''' Show/hide fields as appropriate. '''
        self.polyorder.setVisible(self.cmbModel.currentText() == 'Polynomial')
        self.custom.setVisible(self.cmbModel.currentText() == 'Custom')
        self.lblCustom.setVisible(self.cmbModel.currentText() == 'Custom')

        if self.cmbModel.currentText() != 'Line' and self.use_ux:
            self.btnODR.setChecked(True)
            self.btnODR.setEnabled(False)
            self.btnVert.setEnabled(False)
        else:
            self.btnODR.setEnabled(True)
            self.btnVert.setEnabled(True)

        if self.btnODR.isChecked() or self.cmbModel.currentText() != 'Line':
            self.chkGuess.setVisible(True)
            if self.chkGuess.isChecked():
                self.tblGuess.setVisible(True)
                self.tblGuess.setHorizontalHeaderLabels(['Parameter', 'Initial Guess'])
                self.tblGuess.setRowCount({'Line': 2,
                                           'Exponential': 3,
                                           'Exponential Decay': 2,
                                           'Exponential Decay (rate)': 2,
                                           'Log': 3,
                                           'Logistic Growth': 4,
                                           'Polynomial': self.polyorder.value()+1,
                                           'Custom': len(self.customargs)}[self.cmbModel.currentText()])
                for i in range(self.tblGuess.rowCount()):
                    self.tblGuess.setItem(i, 0, widgets.ReadOnlyTableItem(chr(ord('a')+i)))
            else:
                self.tblGuess.clear()  # For automatic guess
                self.tblGuess.setVisible(False)
        else:
            self.chkGuess.setVisible(False)
            self.tblGuess.setVisible(False)

        expr = self.component.model.expr
        self.lblEquation.setPixmap(gui_math.pixmap_from_sympy(expr))

    def update_custom(self, showhide=True):
        ''' Update the custom curve-fit model expression '''
        try:
            self.customfunc, _, self.customargs = fitparse.parse_fit_expr(self.custom.text())
        except ValueError:
            self.customfunc, _, self.customargs = None, None, []
            self.lblEquation.setText('<font color="red">Invalid Expression. Must contain "x" '
                                     'variable and at least one fit parameter.</font>')
        else:
            self.lblEquation.setText('')
            if showhide:
                self.update_model()
                self.showhide()

    def update_model(self):
        ''' Update the CurveFit object '''
        func = {'Line': 'line',
                'Exponential': 'exp',
                'Exponential Decay': 'decay',
                'Exponential Decay (rate)': 'decay2',
                'Polynomial': 'poly',
                'Log': 'log',
                'Logistic Growth': 'logistic',
                'Custom': self.custom.text() if self.customfunc else 'line'}[self.cmbModel.currentText()]
        numparams = {'Line': 2,
                     'Exponential': 3,
                     'Exponential Decay': 2,
                     'Exponential Decay (rate)': 2,
                     'Log': 3,
                     'Logistic Growth': 4,
                     'Polynomial': self.polyorder.value()+1,
                     'Custom': len(self.customargs)
                     }[self.cmbModel.currentText()]
        p0 = None
        if self.chkGuess.isChecked():
            # User provided the initial guess
            p0 = []
            for i in range(self.tblGuess.rowCount()):
                try:
                    p0.append(float(self.tblGuess.item(i, 1).text()) if self.tblGuess.item(i, 1) else 1)
                except ValueError:
                    p0.append(1)

        elif func in ['exp', 'decay', 'decay2', 'log', 'logistic']:
            # Attempt to come up with a reasonable initial guess based on the model and data
            x = self.component.model.arr.x
            y = self.component.model.arr.y
            if len(x) > 0:
                if func == 'decay':
                    b, a = np.polyfit(x, np.log(abs(y)), deg=1)   # Fit line to (x, log(y))
                    p0 = [np.exp(a), -1/b]
                elif func == 'decay2':
                    b, a = np.polyfit(x, np.log(abs(y)), deg=1)
                    p0 = [np.exp(a), -b]
                elif func == 'exp':
                    b, a = np.polyfit(x, np.log(abs(y)), deg=1)
                    p0 = [np.exp(a), -1/b, 0]
                elif func == 'log':
                    if all(np.sign(x)):
                        b, a = np.polyfit(np.log(x), y, deg=1)
                        p0 = [a, b, 0]
                    else:
                        b, a = np.polyfit(np.log(x-x.min()+1), y, deg=1)
                        p0 = [a, b, x.min()]
                elif func == 'logisitic':
                    p0 = [y.max()-y.min(), (x[-1]-x[0])/2, x.mean(), y.min()]

        elif self.btnODR.isChecked():
            # ODR requires a guess even for basic polynomials
            p0 = np.ones(numparams)

        self.component.set_fitfunc(func, polyorder=self.polyorder.value(), odr=self.btnODR.isChecked(), p0=p0)

    def enable_ux(self, enable):
        ''' Enable/Disable x-uncertainties. '''
        self.use_ux = enable
        self.update_model()
        self.showhide()


class SettingsWidget(QtWidgets.QWidget):
    ''' Widget for configuring fit setup (LSQ, MC, GUM, ODR, etc.) '''
    xdatechange = QtCore.pyqtSignal()
    absolutesigmachange = QtCore.pyqtSignal()

    def __init__(self, component, parent=None):
        super().__init__(parent=parent)
        self.component = component
        self.chkLSQ = QtWidgets.QCheckBox('Least Squares Analytical')
        self.chkMC = QtWidgets.QCheckBox('Monte Carlo')
        self.chkMCMC = QtWidgets.QCheckBox('Markov-Chain Monte Carlo')
        self.chkGUM = QtWidgets.QCheckBox('GUM')
        self.chkLSQ.setChecked(True)
        self.chkAbsoluteSigma = QtWidgets.QCheckBox(
            'Treat uncertainties as relative values. Will be scaled to match the residual variance.')
        self.chkAbsoluteSigma.stateChanged.connect(self.abssigmaupdate)

        self.txtSamples = QtWidgets.QLineEdit('5000')
        self.txtSeed = QtWidgets.QLineEdit('None')
        validator = QtGui.QIntValidator(1, 2147483647)
        self.txtSamples.setValidator(validator)

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(QtWidgets.QLabel('Uncertainty Calculation Method:'))
        layout.addWidget(self.chkLSQ)
        layout.addWidget(self.chkMC)
        layout.addWidget(self.chkMCMC)
        layout.addWidget(self.chkGUM)
        layout.addSpacing(20)
        flayout = QtWidgets.QFormLayout()
        flayout.addRow('Monte Carlo Samples', self.txtSamples)
        flayout.addRow('Random Seed', self.txtSeed)
        layout.addLayout(flayout)
        layout.addSpacing(20)
        layout.addWidget(self.chkAbsoluteSigma)
        layout.addStretch()
        self.setLayout(layout)
        self.txtSamples.editingFinished.connect(self.calcupdate)
        self.txtSeed.editingFinished.connect(self.calcupdate)

    def abssigmaupdate(self):
        ''' Absolute sigma checkbox was changed. Save state and notify for update_model '''
        self.component.model.absolute_sigma = not self.chkAbsoluteSigma.isChecked()
        self.absolutesigmachange.emit()

    def calcupdate(self):
        ''' Save settings to model object '''
        with suppress(ValueError):
            samp = int(self.txtSamples.text())
            self.component.nsamples = samp

        with suppress(ValueError):
            seed = int(self.txtSeed.text())
            self.component.seed = seed


class PageInputCurveFit(QtWidgets.QWidget):
    ''' Input page for curve fit window '''
    COLWIDTH = 75

    def __init__(self, component, parent=None):
        super().__init__(parent)
        self.component = component
        self.useUX = False
        self.btnCalculate = QtWidgets.QPushButton('Calculate')
        self.btnCalculate.setSizePolicy(QtWidgets.QSizePolicy.Policy.Fixed, QtWidgets.QSizePolicy.Policy.Fixed)
        self.table = widgets.FloatTableWidget(headeredit='str')
        self.table.setColumnCount(3)
        self.table.setHorizontalHeaderLabels(['x', 'y', 'u(y)'])
        self.table.setColumnWidth(0, self.COLWIDTH)
        self.table.setColumnWidth(1, self.COLWIDTH)
        self.table.setColumnWidth(2, self.COLWIDTH)
        self.fig = Figure()
        self.ax = self.fig.add_subplot(1, 1, 1)
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setStyleSheet("background-color:transparent;")

        self.settings = SettingsWidget(component)
        self.model = ModelWidget(component)
        self.notes = QtWidgets.QTextEdit()
        self.tab = QtWidgets.QTabWidget()
        self.tab.addTab(self.model, 'Model')
        self.tab.addTab(self.settings, 'Settings')
        self.tab.addTab(self.notes, 'Notes')

        rlayout = QtWidgets.QVBoxLayout()
        rlayout.addWidget(self.tab)
        clayout = QtWidgets.QHBoxLayout()
        clayout.addStretch()
        clayout.addWidget(self.btnCalculate)
        rlayout.addLayout(clayout)
        self.rwidget = QtWidgets.QWidget()
        self.rwidget.setLayout(rlayout)
        self.rsplitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        self.rsplitter.addWidget(self.canvas)
        self.rsplitter.addWidget(self.rwidget)
        self.splitter = QtWidgets.QSplitter()
        self.splitter.addWidget(self.table)
        self.splitter.addWidget(self.rsplitter)
        self.splitter.setCollapsible(0, False)
        self.splitter.setCollapsible(1, False)
        layout = QtWidgets.QHBoxLayout()
        layout.addWidget(self.splitter)
        self.setLayout(layout)

        self.table.valueChanged.connect(self.update_arr)
        self.notes.textChanged.connect(self.savenotes)
        self.settings.xdatechange.connect(self.update_arr)
        self.settings.absolutesigmachange.connect(self.model.update_model)
        self.canvas.draw_idle()

    def toggle_ux(self):
        ''' Enable/disable x-uncertainty column '''
        self.useUX = not self.useUX
        self.model.enable_ux(self.useUX)
        self.update_table()

    def savenotes(self):
        ''' Store the notes field to the component object '''
        self.component.description = self.notes.toPlainText()

    def clear_table(self):
        ''' Clear the data table '''
        self.table.setRowCount(0)
        self.table.setColumnCount(3)
        self.table.insertRow(0)
        self.update_arr()

    def update_table(self):
        ''' Update data table with appropriate columns '''
        if self.useUX:
            self.table.setColumnCount(4)
            self.table.setHorizontalHeaderLabels(['x', 'y', 'u(y)', 'u(x)'])
            self.table.setColumnWidth(3, self.COLWIDTH)
        else:
            self.table.setColumnCount(3)

    def update_arr(self):
        ''' Table edited, update array '''
        x = self.table.get_column(0)
        y = self.table.get_column(1)

        # Remove NaNs
        mask = ~(np.isnan(x) | np.isnan(y))
        x = x[mask]
        y = y[mask]

        self.component.model.arr.xdate = self.table.has_dates()
        self.component.model.arr.x = x
        self.component.model.arr.y = y
        self.component.model.arr.clear_uyestimate()
        self.component.model.arr.clear()  # Clear MC samples

        uy = self.table.get_column(2)[mask]
        if all(~np.isfinite(uy)):
            self.component.model.arr.uy = np.zeros(len(y))
            self.component.model.arr.uy_estimate = None
        else:
            self.component.model.arr.uy = uy
            # Prevent inf/nan values.. use a really small uncertainty
            self.component.model.arr.uy[~np.isfinite(self.component.model.arr.uy)] = 1E-20

        if self.useUX:
            ux = self.table.get_column(3)[mask]
            if all(~np.isfinite(ux)):
                ux = None
                self.component.model.arr.ux = np.zeros(len(x))
            else:
                self.component.model.arr.ux = ux
                self.component.model.arr.ux[~np.isfinite(self.component.model.arr.ux)] = 1E-20
        else:
            ux = None
            self.component.model.arr.ux = np.zeros(len(x))
        self.component.model.xname = self.table.horizontalHeaderItem(0).text()
        self.component.model.yname = self.table.horizontalHeaderItem(1).text()
        self.updateplot()

    def updateplot(self):
        ''' Update the plot '''
        if len(self.component.model.arr.x) == len(self.component.model.arr.y):
            xdates = self.table.has_dates()
            self.ax.cla()
            x = self.component.model.arr.x
            xerr = self.component.model.arr.ux
            if xdates:
                x = mdates.num2date(x)
                xerr = None
            self.ax.errorbar(x, self.component.model.arr.y, yerr=self.component.model.arr.uy,
                             xerr=xerr, marker='o', ls='')
            self.ax.set_xlabel(self.component.model.xname)
            self.ax.set_ylabel(self.component.model.yname)
            if xdates and len(x) > 0:
                # MPL will crash when attempting to autoscale dates if the date range falls below 0
                self.ax.set_xlim(min(x).toordinal(), max(x).toordinal())
        self.canvas.draw_idle()

    def load_data(self):
        ''' Load data from a project component '''
        dlg = page_dataimport.ArraySelectWidget(project=self.component.project)
        ok = dlg.exec()
        if ok:
            arrvals = dlg.get_array()
            self._load_data(arrvals)

    def load_csv(self):
        ''' Load data from a CSV file '''
        fname, _ = QtWidgets.QFileDialog.getOpenFileName(caption='CSV file to load')
        if fname:
            dlg = SelectCSVData(fname, parent=self)
            if dlg.exec():
                dset = dlg.dataset()
                data = dset.model.data
                datahdr = ['x', 'y', 'u(y)']
                dlg2 = widgets.AssignColumnWidget(data, datahdr, ['x', 'y', 'u(y)', 'u(x)'])
                if dlg2.exec():
                    array = dlg2.get_assignments()
                    # Data comes back as np.atleast_2d. Squeeze out the extra dimension.
                    array = {name: np.squeeze(value) for name, value in array.items()}
                    self._load_data(array)

    def _load_data(self, array: dict[str, list[float]]):
        ''' Process the array data to load into table '''
        with BlockedSignals(self.table):
            xvals = array.get('x', None)
            yvals = array.get('y', None)
            uxvals = array.get('u(x)', None)
            uyvals = array.get('u(y)', None)

            def checkrow(i):
                if i >= self.table.rowCount():
                    self.table.setRowCount(i+1)

            if xvals is not None:
                for i, x in enumerate(xvals):
                    checkrow(i)
                    if hasattr(x, 'date'):
                        x = x.strftime('%d-%b-%Y')
                    self.table.setItem(i, 0, QtWidgets.QTableWidgetItem(str(x)))

            if yvals is not None:
                for i, y in enumerate(yvals):
                    checkrow(i)
                    self.table.setItem(i, 1, QtWidgets.QTableWidgetItem(str(y)))

            if uyvals is not None and len(uyvals) > 0:
                for i, uy in enumerate(uyvals):
                    checkrow(i)
                    self.table.setItem(i, 2, QtWidgets.QTableWidgetItem(str(uy)))

            if uxvals is not None and len(uxvals) > 0:
                self.useUX = True
                self.table.setColumnCount(4)
                self.table.setHorizontalHeaderItem(3, QtWidgets.QTableWidgetItem('u(x)'))
                for i, ux in enumerate(uxvals):
                    checkrow(i)
                    self.table.setItem(i, 3, QtWidgets.QTableWidgetItem(str(ux)))
        self.table.resizeColumnsToContents()
        self.update_arr()

    def save_data(self):
        ''' Save the data table to a file '''
        fname, _ = QtWidgets.QFileDialog.getSaveFileName(caption='Select file to save')
        if fname:
            self.component.model.arr.save_file(fname)


class IntervalWidget(QtWidgets.QWidget):
    ''' Widget for selecting an interval, two floats or two dates '''
    changed = QtCore.pyqtSignal()

    def __init__(self, x1=0, x2=1, xdate=False):
        super().__init__()
        self.xdate = xdate
        self.x1 = QtWidgets.QLineEdit(str(x1))
        self.x2 = QtWidgets.QLineEdit(str(x2))
        validator = QtGui.QDoubleValidator(-1E99, 1E99, 4)
        validator.setNotation(QtGui.QDoubleValidator.Notation.ScientificNotation)
        self.x1.setValidator(validator)
        self.x2.setValidator(validator)
        self.xdate1 = QtWidgets.QDateEdit()
        self.xdate2 = QtWidgets.QDateEdit()
        self.xdate1.setCalendarPopup(True)
        self.xdate2.setCalendarPopup(True)
        self.xdate1.setDisplayFormat('dd-MMM-yyyy')
        self.xdate2.setDisplayFormat('dd-MMM-yyyy')

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(QtWidgets.QLabel('Interval:'))
        hlayout = QtWidgets.QHBoxLayout()
        hlayout.addWidget(self.x1)
        hlayout.addWidget(self.xdate1)
        hlayout.addWidget(QtWidgets.QLabel('to'))
        hlayout.addWidget(self.x2)
        hlayout.addWidget(self.xdate2)
        layout.addLayout(hlayout)
        self.setLayout(layout)
        self.set_datemode(xdate)
        self.x1.editingFinished.connect(self.changed)
        self.x2.editingFinished.connect(self.changed)
        self.xdate1.dateChanged.connect(self.changed)
        self.xdate2.dateChanged.connect(self.changed)

    def set_datemode(self, xdate):
        ''' Set whether interval endpoints are dates '''
        self.xdate = xdate
        self.xdate1.setVisible(self.xdate)
        self.xdate2.setVisible(self.xdate)
        self.x1.setVisible(not self.xdate)
        self.x2.setVisible(not self.xdate)


class FullReportSetup(QtWidgets.QWidget):
    changed = QtCore.pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.chkFitPlot = QtWidgets.QCheckBox('Fit Plot')
        self.chkCoeffs = QtWidgets.QCheckBox('Fit Coefficients')
        self.chkGoodness = QtWidgets.QCheckBox('Goodness of Fit')
        self.chkConfEqn = QtWidgets.QCheckBox('Conf. Band Equations')
        self.chkPrediction = QtWidgets.QCheckBox('Prediction')
        self.chkInterval = QtWidgets.QCheckBox('Interval')
        self.chkResid = QtWidgets.QCheckBox('Residuals')
        self.chkCorr = QtWidgets.QCheckBox('Correlations')

        self.chkFitPlot.setChecked(True)
        self.chkCoeffs.setChecked(True)
        self.chkGoodness.setChecked(True)

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.chkCoeffs)
        layout.addWidget(self.chkFitPlot)
        layout.addWidget(self.chkGoodness)
        layout.addWidget(self.chkConfEqn)
        layout.addWidget(self.chkPrediction)
        layout.addWidget(self.chkInterval)
        layout.addWidget(self.chkResid)
        layout.addWidget(self.chkCorr)
        self.setLayout(layout)

        self.chkCoeffs.stateChanged.connect(self.changed)
        self.chkFitPlot.stateChanged.connect(self.changed)
        self.chkGoodness.stateChanged.connect(self.changed)
        self.chkPrediction.stateChanged.connect(self.changed)
        self.chkConfEqn.stateChanged.connect(self.changed)
        self.chkInterval.stateChanged.connect(self.changed)
        self.chkResid.stateChanged.connect(self.changed)
        self.chkCorr.stateChanged.connect(self.changed)


class PageOutputCurveFit(QtWidgets.QWidget):
    ''' Output page for curve fit calculation '''
    namelookup = {'lsq': 'Least Squares',
                  'gum': 'GUM',
                  'montecarlo': 'Monte Carlo',
                  'markov': 'Markov-Chain Monte Carlo'}
    methodlookup = dict((v, k) for k, v in namelookup.items())

    change_help = QtCore.pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.result = None
        self.methodcnt = 0
        self.btnBack = QtWidgets.QPushButton('Back')

        self.outSelect = QtWidgets.QComboBox()
        self.outSelect.addItems(['Fit Plot', 'Prediction', 'Interval', 'Residuals', 'Correlations', 'Full Report'])
        self.cmbMethod = QtWidgets.QComboBox()     # For selecting a single method
        self.cmbMethod.setVisible(False)

        self.chkConfBand = QtWidgets.QCheckBox('Show Confidence Band')
        self.chkPredBand = QtWidgets.QCheckBox('Show Prediction Band')
        self.chkPredBand.setChecked(True)
        self.chkConfBand.setChecked(True)
        self.xvals = widgets.FloatTableWidget()
        self.xvals.setVisible(False)
        self.xvals.setColumnCount(1)
        self.xvals.setHorizontalHeaderLabels(['X Values'])
        self.interval = IntervalWidget()

        self.cmbMCplot = QtWidgets.QComboBox()
        self.cmbMCplot.addItems(['Histograms', 'Samples'])
        self.cmbMCplot.setVisible(False)
        self.paramlist = widgets.ListSelectWidget()
        self.paramlist.setVisible(False)
        self.kconf = widgets.ExpandedConfidenceWidget(showshortest=False)
        self.predictlabel = QtWidgets.QLabel('Prediction Band Uncertainty of New Measurement:')
        self.predictmode = QtWidgets.QComboBox()
        self.predictmode.addItems(['Syx (Residuals)', 'Interpolate u(y)', 'Last u(y)'])
        self.predictmode.setItemData(
            0, 'Use the average of residuals for all x values. Does not consider any user-entered u(y)',
            QtCore.Qt.ItemDataRole.ToolTipRole)
        self.predictmode.setItemData(
            1, 'Extrapolate user-entered u(y) between x data points', QtCore.Qt.ItemDataRole.ToolTipRole)
        self.predictmode.setItemData(
            2, 'Use the last user-entered u(y) for all predictions. Choose this option,\nfor example, '
            'when predicting into the future assuming the most recent\nmeasurement uncertainty '
            'applies to all new measurements.', QtCore.Qt.ItemDataRole.ToolTipRole)
        self.reportoptions = FullReportSetup()

        self.fig = Figure()
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setStyleSheet("background-color:transparent;")
        self.toolbar = NavigationToolbar(self.canvas, self, coordinates=True)
        self.txtOutput = widgets.MarkdownTextEdit()

        llayout = QtWidgets.QVBoxLayout()
        llayout.addWidget(self.outSelect)
        llayout.addWidget(self.cmbMethod)
        llayout.addWidget(self.chkConfBand)
        llayout.addWidget(self.chkPredBand)
        llayout.addWidget(self.kconf)
        llayout.addWidget(self.xvals)
        llayout.addWidget(self.interval)
        llayout.addWidget(self.cmbMCplot)
        llayout.addWidget(self.paramlist)
        llayout.addWidget(self.predictlabel)
        llayout.addWidget(self.predictmode)
        llayout.addWidget(self.reportoptions)
        llayout.addStretch()
        llayout.addWidget(self.btnBack)
        rlayout = QtWidgets.QVBoxLayout()
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

        self.outSelect.currentIndexChanged.connect(self.changeview)
        self.xvals.valueChanged.connect(self.update)
        self.interval.changed.connect(self.update)
        self.chkConfBand.toggled.connect(self.update)
        self.chkPredBand.toggled.connect(self.update)
        self.cmbMethod.currentIndexChanged.connect(self.update)
        self.cmbMCplot.currentIndexChanged.connect(self.update)
        self.paramlist.checkChange.connect(self.update)
        self.kconf.changed.connect(self.update)
        self.predictmode.currentIndexChanged.connect(self.update)
        self.reportoptions.changed.connect(self.update)

    def changeview(self):
        ''' Combobox selection was changed. '''
        showpredict = self.result.setup.points.has_uy()
        self.canvas.setVisible(True)
        self.toolbar.setVisible(True)

        if self.outSelect.currentText() == 'Fit Plot':
            self.cmbMethod.setVisible(self.methodcnt > 1)
            self.xvals.setVisible(False)
            self.interval.setVisible(False)
            self.chkConfBand.setVisible(True)
            self.chkPredBand.setVisible(True)
            self.paramlist.setVisible(False)
            self.cmbMCplot.setVisible(False)
            self.kconf.setVisible(True)
            self.predictlabel.setVisible(showpredict)
            self.predictmode.setVisible(showpredict)
            self.reportoptions.setVisible(False)

        elif self.outSelect.currentText() == 'Prediction':
            self.cmbMethod.setVisible(self.methodcnt > 1)
            self.chkConfBand.setVisible(True)
            self.chkPredBand.setVisible(True)
            self.paramlist.setVisible(False)
            self.xvals.setVisible(True)
            self.interval.setVisible(False)
            self.cmbMCplot.setVisible(False)
            self.kconf.setVisible(True)
            self.predictlabel.setVisible(showpredict)
            self.predictmode.setVisible(showpredict)
            self.reportoptions.setVisible(False)

        elif self.outSelect.currentText() == 'Interval':
            self.cmbMethod.setVisible(self.methodcnt > 1)
            self.chkConfBand.setVisible(False)
            self.chkPredBand.setVisible(False)
            self.paramlist.setVisible(False)
            self.xvals.setVisible(False)
            self.interval.setVisible(True)
            self.cmbMCplot.setVisible(False)
            self.kconf.setVisible(True)
            self.predictlabel.setVisible(showpredict)
            self.predictmode.setVisible(showpredict)
            self.reportoptions.setVisible(False)

        elif self.outSelect.currentText() == 'Residuals':
            self.cmbMethod.setVisible(self.methodcnt > 1)
            self.chkConfBand.setVisible(False)
            self.chkPredBand.setVisible(False)
            self.paramlist.setVisible(False)
            self.xvals.setVisible(False)
            self.interval.setVisible(False)
            self.kconf.setVisible(False)
            self.cmbMCplot.setVisible(False)
            self.predictlabel.setVisible(False)
            self.predictmode.setVisible(False)
            self.reportoptions.setVisible(False)

        elif self.outSelect.currentText() == 'Correlations':
            self.cmbMethod.setVisible(self.methodcnt > 1)
            self.chkConfBand.setVisible(False)
            self.chkPredBand.setVisible(False)
            self.paramlist.setVisible(True)
            self.xvals.setVisible(False)
            self.interval.setVisible(False)
            self.cmbMCplot.setVisible(False)
            self.kconf.setVisible(False)
            self.predictlabel.setVisible(False)
            self.predictmode.setVisible(False)
            self.reportoptions.setVisible(False)

        elif 'Monte Carlo' in self.outSelect.currentText():
            self.cmbMethod.setVisible(False)
            self.chkConfBand.setVisible(False)
            self.chkPredBand.setVisible(False)
            self.paramlist.setVisible(True)
            self.xvals.setVisible(False)
            self.interval.setVisible(False)
            self.cmbMCplot.setVisible(True)
            self.kconf.setVisible(False)
            self.reportoptions.setVisible(False)

        elif 'Full Report' in self.outSelect.currentText():
            self.cmbMethod.setVisible(False)
            self.chkConfBand.setVisible(False)
            self.chkPredBand.setVisible(False)
            self.paramlist.setVisible(False)
            self.xvals.setVisible(False)
            self.interval.setVisible(False)
            self.cmbMCplot.setVisible(False)
            self.kconf.setVisible(False)
            self.reportoptions.setVisible(True)
            self.canvas.setVisible(False)
            self.toolbar.setVisible(False)

        self.update()
        self.change_help.emit()

    def get_predmode(self):
        return {0: 'Syx', 1: 'sigy', 2: 'sigylast'}.get(self.predictmode.currentIndex(), 'Syx')

    def update(self):
        ''' Update the view based on output and controls '''
        r = report.Report()
        self.fig.clf()

        try:
            method = self.methodlookup[self.cmbMethod.currentText()]
        except (KeyError, AttributeError):
            return  # Something weird, nothing in combobox

        out = self.result.method(method)

        predmode = self.get_predmode()
        kconf = self.kconf.get_params()

        if self.outSelect.currentText() == 'Fit Plot':
            r.hdr('Fit Parameters', level=3)
            r.append(out.report.summary())
            r.sympy(out.fit_expr(subs=True), end='\n\n')
            r.hdr('Goodness of Fit', level=3)
            r.append(out.report.goodness_fit(), end='\n\n')

            _, ax = plotting.initplot(self.fig)
            out.report.plot.points(ax=ax, ls='', marker='o')
            out.report.plot.fit(ax=ax, label='Fit')
            if self.chkConfBand.isChecked():
                out.report.plot.conf(ax=ax, ls='--', color='C2', **kconf)
            if self.chkPredBand.isChecked():
                out.report.plot.pred(ax=ax, ls='--', color='C3', mode=predmode, **kconf)
            ax.legend(loc='best')
            ax.set_xlabel(out.setup.xname)
            ax.set_ylabel(out.setup.yname)

        elif self.outSelect.currentText() == 'Prediction':
            ax = self.fig.add_subplot(1, 1, 1)

            # Extend x-range to the manually entered x-point
            xvalues = self.xvals.get_column(0)
            xdata = out.setup.points.x
            if len(xvalues) > 0:
                xmin = min(np.nanmin(xdata), np.nanmin(xvalues))
                xmax = max(np.nanmax(xdata), np.nanmax(xvalues))
                x = np.linspace(xmin, xmax, num=200)
                r.append(out.report.confpred_xval(xval=self.xvals.get_columntext(0), mode=predmode, **kconf))
            else:
                x = np.linspace(min(xdata), max(xdata), num=200)

            out.report.plot.points(ax=ax, ls='', marker='o')
            out.report.plot.fit(ax=ax, x=x, label='Fit')
            if self.chkConfBand.isChecked():
                out.report.plot.conf(ax=ax, x=x, ls='--', color='C2', **kconf)
            if self.chkPredBand.isChecked():
                out.report.plot.pred(ax=ax, x=x, ls='--', color='C3', mode=predmode, **kconf)
            if len(xvalues) > 0:
                out.report.plot.pred_value(xvalues, ax=ax, mode=predmode, **kconf)

            ax.legend(loc='best')
            ax.set_xlabel(out.setup.xname)
            ax.set_ylabel(out.setup.yname)

        elif self.outSelect.currentText() == 'Interval':
            if out.setup.points.xdate:
                t1 = self.interval.xdate1.date().toPyDate().strftime('%d-%b-%Y')
                t2 = self.interval.xdate2.date().toPyDate().strftime('%d-%b-%Y')
            else:
                t1 = float(self.interval.x1.text())
                t2 = float(self.interval.x2.text())
            ax = self.fig.add_subplot(1, 1, 1)
            if t1 != t2:
                out.report.plot.interval_uncert(t1, t2, ax=ax, mode=predmode, **kconf)
                r.append(out.report.interval_uncert(t1, t2, plot=False, mode=predmode, **kconf), end='\n\n')
                r.div()
            r.append(out.report.interval_uncert_eqns())

        elif self.outSelect.currentText() == 'Residuals':
            ax = self.fig.add_subplot(2, 2, 1)
            out.report.plot.points(ax=ax, ls='', marker='o')
            out.report.plot.fit(ax=ax)
            ax.set_title('Fit Line')
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax2 = self.fig.add_subplot(2, 2, 2)
            out.report.plot.residuals(ax=ax2, hist=True)
            ax2.set_title('Residual Histogram')
            ax2.set_xlabel(r'$\Delta$ y')
            ax2.set_ylabel('Probability')
            ax3 = self.fig.add_subplot(2, 2, 3)
            out.report.plot.residuals(ax=ax3, hist=False)
            ax3.axhline(0, color='C1')
            ax3.set_title('Raw Residuals')
            ax3.set_xlabel('x')
            ax3.set_ylabel(r'$\Delta$ y')
            ax4 = self.fig.add_subplot(2, 2, 4)
            out.report.plot.normprob(ax=ax4)
            ax4.set_title('Normal Probability')
            ax4.set_xlabel('Theoretical quantiles')
            ax4.set_ylabel('Ordered sample values')
            self.fig.tight_layout()
            r.append(out.report.residual_table(**kconf))

        elif self.outSelect.currentText() == 'Correlations':
            if len(out.setup.coeffnames) == 2:
                params = [0, 1]
            else:
                params = self.paramlist.getSelectedIndexes()

            if len(params) >= 2:
                out.report.plot.correlation(fig=self.fig, params=params)

            r.hdr('Correlation Matrix:', level=2)
            r.append(out.report.correlation())

        elif 'Monte Carlo' in self.outSelect.currentText():
            mcmc = 'Markov-Chain' in self.outSelect.currentText()
            out = self.result.method('markov') if mcmc else self.result.method('montecarlo')
            params = self.paramlist.getSelectedValues()

            if self.cmbMCplot.currentText() == 'Histograms':
                out.report.plot.xhists(fig=self.fig, coeffnames=params)
            else:
                out.report.plot.samples(fig=self.fig, coeffnames=params)

            r.sympy(out.fit_expr(), end='\n\n')
            r.append(out.report.summary())
            if mcmc:
                r.hdr('Acceptance Rate', level=3)
                r.append(out.report.acceptance())

        elif 'Full Report' in self.outSelect.currentText():
            r = self.get_report()

        self.txtOutput.setReport(r)
        self.canvas.draw_idle()

    def get_report(self):
        ''' Get full report of curve fit, using page settings '''
        kconf = self.kconf.get_params()
        xvals = self.xvals.get_columntext(0)
        x1 = (float(self.interval.x1.text()) if not self.result.setup.points.xdate else
              self.interval.xdate1.date().toPyDate().strftime('%d-%b-%Y'))
        x2 = (float(self.interval.x2.text()) if not self.result.setup.points.xdate else
              self.interval.xdate2.date().toPyDate().strftime('%d-%b-%Y'))
        interval = (x1, x2)

        args = {
            'summary': self.reportoptions.chkCoeffs.isChecked(),
            'fitplot': self.reportoptions.chkFitPlot.isChecked(),
            'goodness': self.reportoptions.chkGoodness.isChecked(),
            'confpred': self.reportoptions.chkConfEqn.isChecked(),
            'prediction': self.reportoptions.chkPrediction.isChecked(),
            'residuals': self.reportoptions.chkResid.isChecked(),
            'correlations': self.reportoptions.chkCorr.isChecked(),
            'xvals': xvals,
            'mode': self.get_predmode(),
            'interval': interval if self.reportoptions.chkInterval.isChecked() else None
            }
        r = self.result.report.all(**kconf, **args)
        return r

    def set_output(self, result):
        ''' Set the CurveFitResultsCombined result '''
        self.result = result
        methods = {'lsq': result.lsq is not None,
                   'montecarlo': result.montecarlo is not None,
                   'gum': result.gum is not None,
                   'markov': result.markov is not None}
        with BlockedSignals(self):
            self.cmbMethod.clear()
            self.cmbMethod.addItems(self.namelookup[i] for i, v in methods.items() if v)
            self.paramlist.addItems(self.result.setup.coeffnames)
            self.paramlist.selectAll()
            with BlockedSignals(self.outSelect):
                self.outSelect.clear()
                self.outSelect.addItems(['Fit Plot', 'Prediction', 'Interval', 'Residuals', 'Correlations', 'Full Report'])
                if methods['montecarlo']:
                    self.outSelect.addItem('Monte Carlo')
                if methods['markov']:
                    self.outSelect.addItem('Markov-Chain Monte Carlo')

                self.interval.set_datemode(self.result.setup.points.xdate)
                self.xvals.clear()
                self.xvals.setHorizontalHeaderLabels(['X Values'])
                xval = self.result.setup.points.x[-1]
                if self.result.setup.points.xdate:
                    xval = mdates.num2date(xval).strftime('%d-%b-%Y')
                self.xvals.setItem(0, 0, QtWidgets.QTableWidgetItem(str(xval)))
                if self.result.setup.points.xdate:
                    # Must add 1721424.5 to account for difference in Julian day (QT) and proleptic Gregorian day (datetime)
                    lastdate1 = QtCore.QDate.fromJulianDay(int(self.result.setup.points.x[-1] + 1721424.5))
                    lastdate2 = QtCore.QDate.fromJulianDay(int(self.result.setup.points.x[-2] + 1721424.5))
                    self.interval.xdate1.setDate(lastdate2)
                    self.interval.xdate2.setDate(lastdate1)
                else:
                    lastx1 = str(self.result.setup.points.x[-1])
                    lastx2 = str(self.result.setup.points.x[-2])
                    self.interval.x1.setText(lastx2)
                    self.interval.x2.setText(lastx1)

        self.methodcnt = list(methods.values()).count(True)
        self.changeview()
        self.update()


class CurveFitWidget(QtWidgets.QWidget):
    ''' Main widget for calculating uncertainty in curve fitting '''

    change_help = QtCore.pyqtSignal()

    def __init__(self, component, parent=None):
        super().__init__(parent)
        self.component = component
        self.pginput = PageInputCurveFit(self.component)
        self.pgoutput = PageOutputCurveFit()
        self.stack = widgets.SlidingStackedWidget()
        self.stack.addWidget(self.pginput)
        self.stack.addWidget(self.pgoutput)
        self.stack.setCurrentIndex(0)
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.stack)
        self.setLayout(layout)
        self.actEnableUX = QtGui.QAction('Enable X Uncertainty', self)
        self.actEnableUX.setCheckable(True)
        self.actClear = QtGui.QAction('Clear Table', self)
        self.actLoadData = QtGui.QAction('Import Data From Project...', self)
        self.actLoadCSV = QtGui.QAction('Import Data From CSV...', self)
        self.actSaveData = QtGui.QAction('Save Data Table...', self)
        self.actSaveReport = QtGui.QAction('Save Report...', self)
        self.actSaveReport.setEnabled(False)
        self.actFillUx = QtGui.QAction('Fill u(x)...', self)
        self.actFillUy = QtGui.QAction('Fill u(y)...', self)

        self.menu = QtWidgets.QMenu('&Curve Fit')
        self.menu.addAction(self.actEnableUX)
        self.menu.addAction(self.actFillUx)
        self.menu.addAction(self.actFillUy)
        self.menu.addSeparator()
        self.menu.addAction(self.actLoadCSV)
        self.menu.addAction(self.actLoadData)
        self.menu.addAction(self.actSaveData)
        self.menu.addAction(self.actClear)
        self.menu.addSeparator()
        self.menu.addAction(self.actSaveReport)
        self.fill_page()

        self.pginput.btnCalculate.clicked.connect(self.calculate)
        self.pgoutput.btnBack.clicked.connect(self.goback)
        self.actEnableUX.triggered.connect(self.pginput.toggle_ux)
        self.actClear.triggered.connect(self.pginput.clear_table)
        self.actLoadCSV.triggered.connect(self.load_csv)
        self.actLoadData.triggered.connect(self.load_data)
        self.actSaveData.triggered.connect(self.pginput.save_data)
        self.actSaveReport.triggered.connect(self.save_report)
        self.actFillUx.triggered.connect(self.fillux)
        self.actFillUy.triggered.connect(self.filluy)
        self.pgoutput.change_help.connect(self.change_help)

    def goback(self):
        ''' Go back to inputs page '''
        self.stack.slideInRight(0)
        self.change_help.emit()

    def fill_page(self):
        ''' Fill the page using values stored in CurveFit object '''
        self.pginput.model.cmbModel.blockSignals(True)
        self.pginput.model.btnODR.blockSignals(True)
        self.pginput.model.btnVert.blockSignals(True)
        self.pginput.model.polyorder.blockSignals(True)
        self.pginput.model.chkGuess.blockSignals(True)
        self.pginput.model.tblGuess.blockSignals(True)
        self.pginput.model.custom.blockSignals(True)
        self.pginput.settings.chkAbsoluteSigma.blockSignals(True)

        self.pginput.notes.setPlainText(self.component.description)
        self.pginput.settings.chkAbsoluteSigma.setChecked(not self.component.model.absolute_sigma)
        if self.component.model.modelname == 'line':
            self.pginput.model.cmbModel.setCurrentIndex(self.pginput.model.cmbModel.findText('Line'))
        elif self.component.model.modelname == 'exp':
            self.pginput.model.cmbModel.setCurrentIndex(self.pginput.model.cmbModel.findText('Exponential'))
        elif self.component.model.modelname == 'decay':
            self.pginput.model.cmbModel.setCurrentIndex(self.pginput.model.cmbModel.findText('Exponential Decay'))
        elif self.component.model.modelname == 'decay2':
            self.pginput.model.cmbModel.setCurrentIndex(self.pginput.model.cmbModel.findText('Exponential Decay (rate)'))
        elif self.component.model.modelname == 'log':
            self.pginput.model.cmbModel.setCurrentIndex(self.pginput.model.cmbModel.findText('Log'))
        elif self.component.model.modelname == 'logistic':
            self.pginput.model.cmbModel.setCurrentIndex(self.pginput.model.cmbModel.findText('Logistic Growth'))
        elif self.component.model.modelname == 'poly':
            self.pginput.model.cmbModel.setCurrentIndex(self.pginput.model.cmbModel.findText('Polynomial'))
            self.pginput.model.polyorder.setValue(self.component.fitoptions.polyorder)
        else:
            self.pginput.model.cmbModel.setCurrentIndex(self.pginput.model.cmbModel.findText('Custom'))
            self.pginput.model.custom.setText(self.component.model.modelname)
            self.pginput.model.update_custom(showhide=False)

        if self.component.fitoptions.odr:
            self.pginput.model.btnODR.setChecked(True)
        else:
            self.pginput.model.btnVert.setChecked(True)
        self.pginput.model.showhide()

        if self.component.fitoptions.p0 is not None:
            self.pginput.model.chkGuess.setChecked(True)
            self.pginput.model.tblGuess.setVisible(True)
            self.pginput.model.showhide()
            for i, val in enumerate(self.component.fitoptions.p0):
                self.pginput.model.tblGuess.setItem(i, 1, QtWidgets.QTableWidgetItem(str(val)))
        else:
            self.pginput.model.update_model()  # Generate reasonable p0

        if self.pginput.model.cmbModel.currentText() == 'Custom':
            self.pginput.model.update_custom()

        self.pginput.table.blockSignals(True)
        self.pginput.table.setRowCount(0)  # Clear stuff first
        self.pginput.table.setRowCount(max(1, len(self.component.model.arr)))
        arr = self.component.model.arr
        uy = any(arr.uy != 0)
        ux = any(arr.ux != 0)
        for i in range(len(arr)):
            if arr.xdate:
                x = mdates.num2date(arr.x[i]).strftime('%d-%b-%Y')
                self.pginput.table.setItem(i, 0, QtWidgets.QTableWidgetItem(x))
            else:
                self.pginput.table.setItem(i, 0, QtWidgets.QTableWidgetItem(str(arr.x[i])))
            self.pginput.table.setItem(i, 1, QtWidgets.QTableWidgetItem(str(arr.y[i])))
            if uy:
                self.pginput.table.setItem(i, 2, QtWidgets.QTableWidgetItem(str(arr.uy[i])))
            if ux:
                self.pginput.table.setItem(i, 3, QtWidgets.QTableWidgetItem(str(arr.ux[i])))

        if self.component.description != '':
            self.pginput.tab.setCurrentIndex(
                [self.pginput.tab.tabText(i) for i in range(self.pginput.tab.count())].index('Notes'))

        self.pginput.table.setHorizontalHeaderLabels(
            [self.component.model.xname, self.component.model.yname, 'u(y)', 'u(x)'])
        self.pginput.table.blockSignals(False)
        self.pginput.model.cmbModel.blockSignals(False)
        self.pginput.model.btnODR.blockSignals(False)
        self.pginput.model.btnVert.blockSignals(False)
        self.pginput.model.polyorder.blockSignals(False)
        self.pginput.model.chkGuess.blockSignals(False)
        self.pginput.model.tblGuess.blockSignals(False)
        self.pginput.model.custom.blockSignals(False)
        self.pginput.settings.chkAbsoluteSigma.blockSignals(False)
        self.pginput.updateplot()

    def get_menu(self):
        ''' Get the page's menu '''
        return self.menu

    def load_data(self):
        ''' Load Data menu item was selected '''
        with BlockedSignals(self):
            self.pginput.load_data()
            self.actEnableUX.setChecked(self.pginput.useUX)

    def load_csv(self):
        ''' Load Data from CSV '''
        self.pginput.load_csv()
        self.actEnableUX.setChecked(self.pginput.useUX)

    def fillux(self):
        ''' Fill all u(x) values '''
        ux, ok = QtWidgets.QInputDialog.getDouble(
            self, 'Enter value', 'Enter u(y) value to apply to all rows', decimals=9, min=0, max=1E99)
        if ok:
            if not self.pginput.useUX:
                self.pginput.toggle_ux()
            for i in range(self.pginput.table.rowCount()):
                self.pginput.table.setItem(i, 3, QtWidgets.QTableWidgetItem(str(ux)))

    def filluy(self):
        ''' Fill all u(y) values '''
        uy, ok = QtWidgets.QInputDialog.getDouble(
            self, 'Enter value', 'Enter u(y) value to apply to all rows', decimals=9, min=0, max=1E99)
        if ok:
            for i in range(self.pginput.table.rowCount()):
                self.pginput.table.setItem(i, 2, QtWidgets.QTableWidgetItem(str(uy)))

    def calculate(self):
        ''' Run the calculation '''
        lsq = self.pginput.settings.chkLSQ.isChecked()
        mc = self.pginput.settings.chkMC.isChecked()
        mcmc = self.pginput.settings.chkMCMC.isChecked()
        gum = self.pginput.settings.chkGUM.isChecked()

        if len(self.component.model.arr.x) < 2 or len(self.component.model.arr.y) < 2:
            QtWidgets.QMessageBox.warning(self, 'Curve Fit', 'Need at least 2 (x,y) points to calculate a curve.')
            return

        try:
            output = self.component.calculate(lsq=lsq, gum=gum, monte=mc, markov=mcmc)
        except ZeroDivisionError:
            QtWidgets.QMessageBox.warning(self, 'Curve Fit', 'Could not compute solution! '
                                          'Ensure x and y data are entered in the table.')
        except RuntimeError as e:
            if 'Optimal parameters' in str(e):
                QtWidgets.QMessageBox.warning(self, 'Curve Fit', 'Curve fit failed to converge. '
                                              'Try updating the initial guess or use a different method.')
            else:
                QtWidgets.QMessageBox.warning(self, 'Curve Fit', 'Could not compute solution!')
                logging.warn(str(e))
        except TypeError as e:
            if 'Improper input' in str(e):
                QtWidgets.QMessageBox.warning(self, 'Curve Fit', 'Polynomial is overfit. Reduce order.')
            else:
                QtWidgets.QMessageBox.warning(self, 'Curve Fit', 'Could not compute solution!')
                logging.warn(str(e))
        except ValueError as e:
            if 'beta0' in str(e):
                QtWidgets.QMessageBox.warning(self, 'Curve Fit', 'Please provide initial guess.')
            else:
                QtWidgets.QMessageBox.warning(self, 'Curve Fit', 'Could not compute solution!')
                logging.warn(str(e))
        else:
            self.pgoutput.set_output(output)
            self.stack.slideInLeft(1)
        self.actSaveReport.setEnabled(True)
        self.change_help.emit()

    def update_proj_config(self):
        self.pginput.model.update_model()

    def get_report(self):
        ''' Get full report of curve fit, using page settings '''
        return self.pgoutput.get_report()

    def save_report(self):
        ''' Save full report of curve fit, asking user for settings/filename '''
        with gui_styles.LightPlotstyle():
            widgets.savereport(self.get_report())

    def help_report(self):
        ''' Get the help report to display the current widget mode '''
        if self.stack.m_next == 0:
            return CurveHelp.inputs()
        else:
            if self.pgoutput.outSelect.currentText() == 'Fit Plot':
                return CurveHelp.fit()
            elif self.pgoutput.outSelect.currentText() == 'Prediction':
                return CurveHelp.prediction()
            elif self.pgoutput.outSelect.currentText() == 'Interval':
                return CurveHelp.interval()
            elif self.pgoutput.outSelect.currentText() == 'Residuals':
                return CurveHelp.residuals()
            elif self.pgoutput.outSelect.currentText() == 'Correlations':
                return CurveHelp.correlations()
            elif self.pgoutput.outSelect.currentText() == 'Monte Carlo':
                return CurveHelp.montecarlo()
            else:
                return CurveHelp.nohelp()
