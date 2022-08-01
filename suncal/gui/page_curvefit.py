''' Page for fitting curve to experimental data '''

from contextlib import suppress
from PyQt5 import QtWidgets, QtGui, QtCore
import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import matplotlib.dates as mdates

from .. import report
from .. import plotting
from . import gui_common
from . import gui_widgets
from . import page_dataimport


class OrderWidget(QtWidgets.QWidget):
    ''' Widget for showing label and spinbox for polynomial order '''
    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.label = QtWidgets.QLabel('Order:')
        self.order = QtWidgets.QSpinBox()
        self.order.setRange(2, 12)  # 12 is arbitrary limit
        layout = QtWidgets.QHBoxLayout()
        layout.addSpacing(15)
        layout.addWidget(self.label)
        layout.addWidget(self.order)
        layout.addStretch()
        self.setLayout(layout)

    def value(self):
        ''' Get spinbox value '''
        return self.order.value()


class ModelWidget(QtWidgets.QWidget):
    ''' Widget for configuring the fit model (line, poly, etc.) '''
    def __init__(self, fitcalc, parent=None):
        super().__init__(parent=parent)
        self.fitcalc = fitcalc
        self.use_ux = False
        self.customfunc = None
        self.customargs = []

        self.cmbModel = QtWidgets.QComboBox()
        self.cmbModel.addItems(['Line', 'Polynomial', 'Exponential', 'Exponential Decay', 'Exponential Decay (rate)', 'Log', 'Logistic Growth', 'Custom'])
        self.polyorder = OrderWidget()
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
        self.polyorder.order.valueChanged.connect(self.modelchange)
        self.chkGuess.toggled.connect(self.modelchange)
        self.custom.editingFinished.connect(self.update_custom)
        self.showhide()

    def modelchange(self):
        ''' Model selection changed by user. Show/hide fields and update
            the fitcalc model
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
                self.tblGuess.setRowCount({'Line': 2, 'Exponential': 3, 'Exponential Decay': 2, 'Exponential Decay (rate)': 2, 'Log': 3, 'Logistic Growth': 4,
                                           'Polynomial': self.polyorder.value()+1, 'Custom': len(self.customargs)}[self.cmbModel.currentText()])
                for i in range(self.tblGuess.rowCount()):
                    self.tblGuess.setItem(i, 0, gui_widgets.ReadOnlyTableItem(chr(ord('a')+i)))
            else:
                self.tblGuess.clear()  # For automatic guess
                self.tblGuess.setVisible(False)
        else:
            self.chkGuess.setVisible(False)
            self.tblGuess.setVisible(False)

        expr = self.fitcalc.expr
        ratio = QtWidgets.QApplication.instance().devicePixelRatio()
        imgbuf = report.Math.from_sympy(expr).svg_buf(fontsize=16*ratio)
        px = QtGui.QPixmap()
        px.loadFromData(imgbuf.getvalue())
        px.setDevicePixelRatio(ratio)
        self.lblEquation.setPixmap(px)

    def update_custom(self, showhide=True):
        ''' Update the custom curve-fit model expression '''
        try:
            self.customfunc, _, self.customargs = self.fitcalc.parse_math(self.custom.text())
        except ValueError:
            self.customfunc, _, self.customargs = None, None, []
            self.lblEquation.setText('<font color="red">Invalid Expression. Must contain "x" variable and at least one fit parameter.</font>')
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
            x = self.fitcalc.arr.x
            y = self.fitcalc.arr.y
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

        self.fitcalc.set_fitfunc(func, polyorder=self.polyorder.value(), odr=self.btnODR.isChecked(), p0=p0)

    def enable_ux(self, enable):
        ''' Enable/Disable x-uncertainties. '''
        self.use_ux = enable
        self.update_model()
        self.showhide()


class SettingsWidget(QtWidgets.QWidget):
    ''' Widget for configuring fit setup (LSQ, MC, GUM, ODR, etc.) '''
    xdatechange = QtCore.pyqtSignal()
    absolutesigmachange = QtCore.pyqtSignal()

    def __init__(self, fitcalc, parent=None):
        super().__init__(parent=parent)
        self.fitcalc = fitcalc
        self.chkLSQ = QtWidgets.QCheckBox('Least Squares Analytical')
        self.chkMC = QtWidgets.QCheckBox('Monte Carlo')
        self.chkMCMC = QtWidgets.QCheckBox('Markov-Chain Monte Carlo')
        self.chkGUM = QtWidgets.QCheckBox('GUM')
        self.chkLSQ.setChecked(True)
        self.chkAbsoluteSigma = QtWidgets.QCheckBox('Treat uncertainties as relative values. Will be scaled to match the residual variance.')
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
        self.fitcalc.absolute_sigma = not self.chkAbsoluteSigma.isChecked()
        self.absolutesigmachange.emit()

    def calcupdate(self):
        ''' Save settings to model object '''
        with suppress(ValueError):
            samp = int(self.txtSamples.text())
            self.fitcalc.samples = samp

        with suppress(ValueError):
            seed = int(self.txtSeed.text())
            self.fitcalc.seed = seed


class PageInputCurveFit(QtWidgets.QWidget):
    ''' Input page for curve fit window '''
    COLWIDTH = 75

    def __init__(self, fitcalc, parent=None):
        super().__init__(parent)
        self.fitcalc = fitcalc
        self.useUX = False
        self.btnCalculate = QtWidgets.QPushButton('Calculate')
        self.btnCalculate.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        self.table = gui_widgets.FloatTableWidget(headeredit='str')
        self.table.setColumnCount(3)
        self.table.setMaximumWidth(400)
        self.table.setHorizontalHeaderLabels(['x', 'y', 'u(y)'])
        self.table.setColumnWidth(0, self.COLWIDTH)
        self.table.setColumnWidth(1, self.COLWIDTH)
        self.table.setColumnWidth(2, self.COLWIDTH)
        self.fig = Figure()
        self.ax = self.fig.add_subplot(1, 1, 1)
        self.canvas = FigureCanvas(self.fig)
        self.plotline = None

        self.settings = SettingsWidget(fitcalc)
        self.model = ModelWidget(fitcalc)
        self.notes = QtWidgets.QTextEdit()
        self.tab = QtWidgets.QTabWidget()
        self.tab.addTab(self.model, 'Model')
        self.tab.addTab(self.settings, 'Settings')
        self.tab.addTab(self.notes, 'Notes')

        layout = QtWidgets.QHBoxLayout()
        llayout = QtWidgets.QVBoxLayout()
        llayout.addWidget(self.table)
        rlayout = QtWidgets.QVBoxLayout()
        rlayout.addWidget(self.canvas)
        rlayout.addWidget(self.tab)
        clayout = QtWidgets.QHBoxLayout()
        clayout.addStretch()
        clayout.addWidget(self.btnCalculate)
        rlayout.addLayout(clayout)
        layout.addLayout(llayout, stretch=1)
        layout.addLayout(rlayout, stretch=3)
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
        ''' Store the notes field to the fitcalc object '''
        self.fitcalc.desc = self.notes.toPlainText()

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

        self.fitcalc.arr.xdate = self.table.has_dates()
        self.fitcalc.arr.x = x
        self.fitcalc.arr.y = y
        self.fitcalc.arr.clear_uyestimate()
        self.fitcalc.arr.clear()  # Clear MC samples

        uy = self.table.get_column(2)[mask]
        if all(~np.isfinite(uy)):
            self.fitcalc.arr.uy = np.zeros(len(y))
            self.fitcalc.arr.uy_estimate = None
        else:
            self.fitcalc.arr.uy = uy
            # Prevent inf/nan values.. use a really small uncertainty
            self.fitcalc.arr.uy[~np.isfinite(self.fitcalc.arr.uy)] = 1E-20

        if self.useUX:
            ux = self.table.get_column(3)[mask]
            if all(~np.isfinite(ux)):
                ux = None
                self.fitcalc.arr.ux = np.zeros(len(x))
            else:
                self.fitcalc.arr.ux = ux
                self.fitcalc.arr.ux[~np.isfinite(self.fitcalc.arr.ux)] = 1E-20
        else:
            ux = None
            self.fitcalc.arr.ux = np.zeros(len(x))
        self.fitcalc.xname = self.table.horizontalHeaderItem(0).text()
        self.fitcalc.yname = self.table.horizontalHeaderItem(1).text()
        self.updateplot()

    def updateplot(self):
        ''' Update the plot '''
        if len(self.fitcalc.arr.x) == len(self.fitcalc.arr.y):
            xdates = self.table.has_dates()
            self.ax.cla()
            x = self.fitcalc.arr.x
            xerr = self.fitcalc.arr.ux
            if xdates:
                x = mdates.num2date(x)
                xerr = None
            self.ax.errorbar(x, self.fitcalc.arr.y, yerr=self.fitcalc.arr.uy, xerr=xerr, marker='o', ls='')
            self.ax.set_xlabel(self.fitcalc.xname)
            self.ax.set_ylabel(self.fitcalc.yname)
            if xdates and len(x) > 0:
                self.ax.set_xlim(min(x).toordinal(), max(x).toordinal())  # MPL will crash when attempting to autoscale dates if the date range falls below 0
        self.canvas.draw_idle()

    def load_data(self):
        ''' Load data from a data set or a file '''
        dlg = page_dataimport.ArraySelectWidget(project=self.fitcalc.project)
        ok = dlg.exec_()
        if ok:
            arrvals = dlg.get_array()
            self.table.blockSignals(True)

            xvals = arrvals.get('x', None)
            yvals = arrvals.get('y', None)
            uxvals = arrvals.get('u(x)', None)
            uyvals = arrvals.get('u(y)', None)

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
            self.table.blockSignals(False)
            self.table.resizeColumnsToContents()
            self.update_arr()

    def save_data(self):
        ''' Save the data table to a file '''
        fname, _ = QtWidgets.QFileDialog.getSaveFileName(caption='Select file to save')
        if fname:
            self.fitcalc.arr.save_file(fname)


class IntervalWidget(QtWidgets.QWidget):
    ''' Widget for selecting an interval, two floats or two dates '''
    changed = QtCore.pyqtSignal()

    def __init__(self, x1=0, x2=1, xdate=False):
        super().__init__()
        self.xdate = xdate
        self.x1 = QtWidgets.QLineEdit(str(x1))
        self.x2 = QtWidgets.QLineEdit(str(x2))
        validator = QtGui.QDoubleValidator(-1E99, 1E99, 4)
        validator.setNotation(QtGui.QDoubleValidator.StandardNotation | QtGui.QDoubleValidator.ScientificNotation)
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
    namelookup = {'lsq': 'Least Squares', 'gum': 'GUM', 'mc': 'Monte Carlo', 'mcmc': 'Markov-Chain Monte Carlo'}
    methodlookup = dict((v, k) for k, v in namelookup.items())

    def __init__(self, parent=None):
        super().__init__(parent)
        self.output = None
        self.btnBack = QtWidgets.QPushButton('Back')

        self.outSelect = QtWidgets.QComboBox()
        self.outSelect.addItems(['Fit Plot', 'Prediction', 'Interval', 'Residuals', 'Correlations', 'Full Report'])
        self.cmbMethod = QtWidgets.QComboBox()     # For selecting a single method
        self.cmbMethod.setVisible(False)

        self.chkConfBand = QtWidgets.QCheckBox('Show Confidence Band')
        self.chkPredBand = QtWidgets.QCheckBox('Show Prediction Band')
        self.chkPredBand.setChecked(True)
        self.chkConfBand.setChecked(True)
        self.xvals = gui_widgets.FloatTableWidget()
        self.xvals.setVisible(False)
        self.xvals.setColumnCount(1)
        self.xvals.setHorizontalHeaderLabels(['X Values'])
        self.interval = IntervalWidget()

        self.cmbMCplot = QtWidgets.QComboBox()
        self.cmbMCplot.addItems(['Histograms', 'Samples'])
        self.cmbMCplot.setVisible(False)
        self.paramlist = gui_widgets.ListSelectWidget()
        self.paramlist.setVisible(False)
        self.kvalue = gui_widgets.GUMExpandedWidget(label='Expanded:', multiselect=False, dflt=[1])
        self.predictlabel = QtWidgets.QLabel('Prediction Band Uncertainty of New Measurement:')
        self.predictmode = QtWidgets.QComboBox()
        self.predictmode.addItems(['Syx (Residuals)', 'Interpolate u(y)', 'Last u(y)'])
        self.predictmode.setItemData(0, 'Use the average of residuals for all x values. Does not consider any user-entered u(y)', QtCore.Qt.ToolTipRole)
        self.predictmode.setItemData(1, 'Extrapolate user-entered u(y) between x data points', QtCore.Qt.ToolTipRole)
        self.predictmode.setItemData(2, 'Use the last user-entered u(y) for all predictions. Choose this option,\nfor example, when predicting into the future assuming the most recent\nmeasurement uncertainty applies to all new measurements.', QtCore.Qt.ToolTipRole)
        self.reportoptions = FullReportSetup()

        self.fig = Figure()
        self.canvas = FigureCanvas(self.fig)
        self.toolbar = NavigationToolbar(self.canvas, self, coordinates=True)
        self.txtOutput = gui_widgets.MarkdownTextEdit()

        layout = QtWidgets.QHBoxLayout()
        llayout = QtWidgets.QVBoxLayout()
        llayout.addWidget(self.outSelect)
        llayout.addWidget(self.cmbMethod)
        llayout.addWidget(self.chkConfBand)
        llayout.addWidget(self.chkPredBand)
        llayout.addWidget(self.kvalue)
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
        rlayout.addWidget(self.txtOutput, stretch=5)
        
        # Left pane ends up too wide for some reason.
        # Stupid hack to put all the widgets in another widget to set maximum size
        lwidget = QtWidgets.QWidget()
        lwidget.setMaximumWidth(300)
        lwidget.setLayout(llayout)

        layout.addWidget(lwidget)
        layout.addLayout(rlayout)
        self.setLayout(layout)

        self.outSelect.currentIndexChanged.connect(self.changeview)
        self.xvals.valueChanged.connect(self.update)
        self.interval.changed.connect(self.update)
        self.chkConfBand.toggled.connect(self.update)
        self.chkPredBand.toggled.connect(self.update)
        self.cmbMethod.currentIndexChanged.connect(self.update)
        self.cmbMCplot.currentIndexChanged.connect(self.update)
        self.paramlist.checkChange.connect(self.update)
        self.kvalue.changed.connect(self.update)
        self.predictmode.currentIndexChanged.connect(self.update)
        self.reportoptions.changed.connect(self.update)

    def changeview(self):
        ''' Combobox selection was changed. '''
        showpredict = self.output.inputs.has_uy()
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
            self.kvalue.setVisible(True)
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
            self.kvalue.setVisible(True)
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
            self.kvalue.setVisible(True)
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
            self.kvalue.setVisible(False)
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
            self.kvalue.setVisible(False)
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
            self.kvalue.setVisible(False)
            self.reportoptions.setVisible(False)
        
        elif 'Full Report' in self.outSelect.currentText():
            self.cmbMethod.setVisible(False)
            self.chkConfBand.setVisible(False)
            self.chkPredBand.setVisible(False)
            self.paramlist.setVisible(False)
            self.xvals.setVisible(False)
            self.interval.setVisible(False)
            self.cmbMCplot.setVisible(False)
            self.kvalue.setVisible(False)
            self.reportoptions.setVisible(True)
            self.canvas.setVisible(False)
            self.toolbar.setVisible(False)

        self.update()

    def update(self):
        ''' Update the view based on output and controls '''
        r = report.Report()
        self.fig.clf()

        try:
            method = self.methodlookup[self.cmbMethod.currentText()]
        except (KeyError, AttributeError):
            return  # Something weird, nothing in combobox

        out = getattr(self.output, method)

        predmode = {0: 'Syx', 1: 'sigy', 2: 'sigylast'}.get(self.predictmode.currentIndex(), 'Syx')
        out.predmode = predmode

        kstr = self.kvalue.get_covlist()[0]
        if 'k' in kstr:
            k = float(kstr.split('=')[1].strip())
            conf = None
        else:
            k = None
            conf = float(kstr[:-1]) / 100  # Without the % sign, as decimal

        if self.outSelect.currentText() == 'Fit Plot':
            r.hdr('Fit Parameters', level=3)
            r.append(out.report())
            r.sympy(out.expr(subs=True), end='\n\n')
            r.hdr('Goodness of Fit', level=3)
            r.append(out.report_fit(), end='\n\n')

            fig, ax = plotting.initplot(self.fig)
            out.plot_points(ax=ax, ls='', marker='o')
            out.plot_fit(ax=ax, label='Fit')
            if self.chkConfBand.isChecked():
                out.plot_conf(ax=ax, k=k, conf=conf, ls='--', color='C2')
            if self.chkPredBand.isChecked():
                out.plot_pred(ax=ax, k=k, conf=conf, ls='--', color='C3')
            ax.legend(loc='best')
            ax.set_xlabel(out.model.xname)
            ax.set_ylabel(out.model.yname)
            self.fig.tight_layout()

        elif self.outSelect.currentText() == 'Prediction':
            ax = self.fig.add_subplot(1, 1, 1)

            # Extend x-range to the manually entered x-point
            xvalues = self.xvals.get_column(0)
            xdata = out.inputs.x
            if len(xvalues) > 0:
                xmin = min(np.nanmin(xdata), np.nanmin(xvalues))
                xmax = max(np.nanmax(xdata), np.nanmax(xvalues))
                x = np.linspace(xmin, xmax, num=200)
                r.append(out.report_confpred_xval(xval=self.xvals.get_columntext(0), k=k, conf=conf))
            else:
                x = np.linspace(min(xdata), max(xdata), num=200)

            out.plot_points(ax=ax, ls='', marker='o')
            out.plot_fit(ax=ax, x=x, label='Fit')
            if self.chkConfBand.isChecked():
                out.plot_conf(ax=ax, x=x, k=k, conf=conf, ls='--', color='C2')
            if self.chkPredBand.isChecked():
                out.plot_pred(ax=ax, x=x, k=k, conf=conf, ls='--', color='C3')
            if len(xvalues) > 0:
                out.plot_pred_value(xvalues, ax=ax, k=k, conf=conf)

            ax.legend(loc='best')
            ax.set_xlabel(out.model.xname)
            ax.set_ylabel(out.model.yname)

        elif self.outSelect.currentText() == 'Interval':
            if out.inputs.xdate:
                t1 = self.interval.xdate1.date().toPyDate().strftime('%d-%b-%Y')
                t2 = self.interval.xdate2.date().toPyDate().strftime('%d-%b-%Y')
            else:
                t1 = float(self.interval.x1.text())
                t2 = float(self.interval.x2.text())
            ax = self.fig.add_subplot(1, 1, 1)
            if t1 != t2:
                out.plot_interval_uncert(t1, t2, ax=ax, k=k, conf=conf)
                r.append(out.report_interval_uncert(t1, t2, k=k, conf=conf, plot=False), end='\n\n')
                r.div()
            r.append(out.report_interval_uncert_eqns())

        elif self.outSelect.currentText() == 'Residuals':
            ax = self.fig.add_subplot(2, 2, 1)
            out.plot_points(ax=ax, ls='', marker='o')
            out.plot_fit(ax=ax)
            ax.set_title('Fit Line')
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax2 = self.fig.add_subplot(2, 2, 2)
            out.plot_residuals(ax=ax2, hist=True)
            ax2.set_title('Residual Histogram')
            ax2.set_xlabel(r'$\Delta$ y')
            ax2.set_ylabel('Probability')
            ax3 = self.fig.add_subplot(2, 2, 3)
            out.plot_residuals(ax=ax3, hist=False)
            ax3.axhline(0, color='C1')
            ax3.set_title('Raw Residuals')
            ax3.set_xlabel('x')
            ax3.set_ylabel(r'$\Delta$ y')
            ax4 = self.fig.add_subplot(2, 2, 4)
            out.plot_normprob(ax=ax4)
            ax4.set_title('Normal Probability')
            ax4.set_xlabel('Theoretical quantiles')
            ax4.set_ylabel('Ordered sample values')
            self.fig.tight_layout()
            r.append(out.report_residtable(k=k, conf=conf))

        elif self.outSelect.currentText() == 'Correlations':
            if len(out.paramnames) == 2:
                params = [0, 1]
            else:
                params = self.paramlist.getSelectedIndexes()

            if len(params) >= 2:
                out.plot_correlation(fig=self.fig, params=params)

            r.hdr('Correlation Matrix:', level=2)
            r.append(out.report_correlation())

        elif 'Monte Carlo' in self.outSelect.currentText():
            mcmc = 'Markov-Chain' in self.outSelect.currentText()
            out = getattr(self.output, 'mcmc') if mcmc else getattr(self.output, 'mc')
            params = self.paramlist.getSelectedIndexes()

            if self.cmbMCplot.currentText() == 'Histograms':
                out.plot_xhists(fig=self.fig, inpts=params)
            else:
                out.plot_samples(fig=self.fig, inpts=params)

            r.sympy(out.expr(), end='\n\n')
            r.append(out.report())
            if mcmc:
                r.hdr('Acceptance Rate', level=3)
                r.append(out.report_acceptance())

        elif 'Full Report' in self.outSelect.currentText():
            r = self.get_report()

        self.txtOutput.setReport(r)
        self.canvas.draw_idle()

    def get_report(self):
        ''' Get full report of curve fit, using page settings '''
        kstr = self.kvalue.get_covlist()[0]
        if 'k' in kstr:
            k = float(kstr.split('=')[1].strip())
            conf = None
        else:
            k = None
            conf = float(kstr[:-1]) / 100  # Without the % sign, as decimal

        xvals=self.xvals.get_columntext(0)
        method = self.methodlookup[self.cmbMethod.currentText()]
        predmode = getattr(self.output, method).predmode
        x1 = float(self.interval.x1.text()) if not self.output.inputs.xdate else self.interval.xdate1.date().toPyDate().strftime('%d-%b-%Y')
        x2 = float(self.interval.x2.text()) if not self.output.inputs.xdate else self.interval.xdate2.date().toPyDate().strftime('%d-%b-%Y')
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
            'mode': predmode,
            'interval': interval if self.reportoptions.chkInterval.isChecked() else None
            }
        r = self.output.report_all(k=k, conf=conf, **args)
        return r

    def set_output(self, output):
        ''' Set the output object

            Parameters
            ----------
            output: out_curvefit.CurveFitOutput
                The output object to display
        '''
        self.output = output
        methods = {'lsq': output.lsq is not None,
                   'mc': output.mc is not None,
                   'gum': output.gum is not None,
                   'mcmc': output.mcmc is not None}
        self.blockSignals(True)
        self.cmbMethod.clear()
        self.cmbMethod.addItems(self.namelookup[i] for i, v in methods.items() if v)
        self.paramlist.addItems(self.output.paramnames)
        self.paramlist.selectAll()
        self.outSelect.blockSignals(True)
        self.outSelect.clear()
        self.outSelect.addItems(['Fit Plot', 'Prediction', 'Interval', 'Residuals', 'Correlations', 'Full Report'])
        if methods['mc']:
            self.outSelect.addItem('Monte Carlo')
        if methods['mcmc']:
            self.outSelect.addItem('Markov-Chain Monte Carlo')

        self.interval.set_datemode(self.output.inputs.xdate)
        self.xvals.clear()
        self.xvals.setHorizontalHeaderLabels(['X Values'])
        xval = self.output.inputs.x[-1]
        if self.output.inputs.xdate:
            xval = mdates.num2date(xval).strftime('%d-%b-%Y')
        self.xvals.setItem(0, 0, QtWidgets.QTableWidgetItem(str(xval)))
        if self.output.inputs.xdate:
            # Must add 1721424.5 to account for difference in Julian day (QT) and proleptic Gregorian day (datetime)
            lastdate1 = QtCore.QDate.fromJulianDay(self.output.inputs.x[-1] + 1721424.5)
            lastdate2 = QtCore.QDate.fromJulianDay(self.output.inputs.x[-2] + 1721424.5)
            self.interval.xdate1.setDate(lastdate2)
            self.interval.xdate2.setDate(lastdate1)
        else:
            lastx1 = str(self.output.inputs.x[-1])
            lastx2 = str(self.output.inputs.x[-2])
            self.interval.x1.setText(lastx2)
            self.interval.x2.setText(lastx1)

        self.outSelect.blockSignals(False)
        self.blockSignals(False)
        self.methodcnt = list(methods.values()).count(True)
        self.changeview()
        self.update()


class CurveFitWidget(QtWidgets.QWidget):
    ''' Main widget for calculating uncertainty in curve fitting '''
    def __init__(self, item, parent=None):
        super().__init__(parent)
        self.fitcalc = item
        self.pginput = PageInputCurveFit(self.fitcalc)
        self.pgoutput = PageOutputCurveFit()
        self.stack = QtWidgets.QStackedWidget()
        self.stack.addWidget(self.pginput)
        self.stack.addWidget(self.pgoutput)
        self.stack.setCurrentIndex(0)
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.stack)
        self.setLayout(layout)
        self.actEnableUX = QtWidgets.QAction('Enable X Uncertainty', self)
        self.actEnableUX.setCheckable(True)
        self.actClear = QtWidgets.QAction('Clear Table', self)
        self.actLoadData = QtWidgets.QAction('Insert Data From...', self)
        self.actSaveData = QtWidgets.QAction('Save Data Table...', self)
        self.actSaveReport = QtWidgets.QAction('Save Report...', self)
        self.actSaveReport.setEnabled(False)
        self.actFillUx = QtWidgets.QAction('Fill u(x)...', self)
        self.actFillUy = QtWidgets.QAction('Fill u(y)...', self)

        self.menu = QtWidgets.QMenu('&Curve Fit')
        self.menu.addAction(self.actEnableUX)
        self.menu.addAction(self.actFillUx)
        self.menu.addAction(self.actFillUy)
        self.menu.addSeparator()
        self.menu.addAction(self.actLoadData)
        self.menu.addAction(self.actSaveData)
        self.menu.addAction(self.actClear)
        self.menu.addSeparator()
        self.menu.addAction(self.actSaveReport)
        self.fill_page()

        self.pginput.btnCalculate.clicked.connect(self.calculate)
        self.pgoutput.btnBack.clicked.connect(lambda x: self.stack.setCurrentIndex(0))
        self.actEnableUX.triggered.connect(self.pginput.toggle_ux)
        self.actClear.triggered.connect(self.pginput.clear_table)
        self.actLoadData.triggered.connect(self.load_data)
        self.actSaveData.triggered.connect(self.pginput.save_data)
        self.actSaveReport.triggered.connect(self.save_report)
        self.actFillUx.triggered.connect(self.fillux)
        self.actFillUy.triggered.connect(self.filluy)

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

        self.pginput.notes.setPlainText(self.fitcalc.desc)
        self.pginput.settings.chkAbsoluteSigma.setChecked(not self.fitcalc.absolute_sigma)
        if self.fitcalc.fitname == 'line':
            self.pginput.model.cmbModel.setCurrentIndex(self.pginput.model.cmbModel.findText('Line'))
        elif self.fitcalc.fitname == 'exp':
            self.pginput.model.cmbModel.setCurrentIndex(self.pginput.model.cmbModel.findText('Exponential'))
        elif self.fitcalc.fitname == 'decay':
            self.pginput.model.cmbModel.setCurrentIndex(self.pginput.model.cmbModel.findText('Exponential Decay'))
        elif self.fitcalc.fitname == 'decay2':
            self.pginput.model.cmbModel.setCurrentIndex(self.pginput.model.cmbModel.findText('Exponential Decay (rate)'))
        elif self.fitcalc.fitname == 'log':
            self.pginput.model.cmbModel.setCurrentIndex(self.pginput.model.cmbModel.findText('Log'))
        elif self.fitcalc.fitname == 'logistic':
            self.pginput.model.cmbModel.setCurrentIndex(self.pginput.model.cmbModel.findText('Logistic Growth'))
        elif self.fitcalc.fitname == 'poly':
            self.pginput.model.cmbModel.setCurrentIndex(self.pginput.model.cmbModel.findText('Polynomial'))
            self.pginput.model.polyorder.order.setValue(self.fitcalc.polyorder)
        else:
            self.pginput.model.cmbModel.setCurrentIndex(self.pginput.model.cmbModel.findText('Custom'))
            self.pginput.model.custom.setText(self.fitcalc.fitname)
            self.pginput.model.update_custom(showhide=False)

        if self.fitcalc.odr:
            self.pginput.model.btnODR.setChecked(True)
        else:
            self.pginput.model.btnVert.setChecked(True)
        self.pginput.model.showhide()

        if self.fitcalc.p0 is not None:
            self.pginput.model.chkGuess.setChecked(True)
            self.pginput.model.tblGuess.setVisible(True)
            self.pginput.model.showhide()
            for i, val in enumerate(self.fitcalc.p0):
                self.pginput.model.tblGuess.setItem(i, 1, QtWidgets.QTableWidgetItem(str(val)))
        else:
            self.pginput.model.update_model()  # Generate reasonable p0

        if self.pginput.model.cmbModel.currentText() == 'Custom':
            self.pginput.model.update_custom()

        self.pginput.table.blockSignals(True)
        self.pginput.table.setRowCount(0)  # Clear stuff first
        self.pginput.table.setRowCount(max(1, len(self.fitcalc.arr)))
        arr = self.fitcalc.arr
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

        if self.fitcalc.desc != '':
            self.pginput.tab.setCurrentIndex([self.pginput.tab.tabText(i) for i in range(self.pginput.tab.count())].index('Notes'))

        self.pginput.table.setHorizontalHeaderLabels([self.fitcalc.xname, self.fitcalc.yname, 'u(y)', 'u(x)'])
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
        self.blockSignals(True)
        self.pginput.load_data()
        self.actEnableUX.setChecked(self.pginput.useUX)
        self.blockSignals(False)

    def fillux(self):
        ''' Fill all u(x) values '''
        ux, ok = QtWidgets.QInputDialog.getDouble(self, 'Enter value', 'Enter u(y) value to apply to all rows', decimals=9, min=0, max=1E99)
        if ok:
            if not self.pginput.useUX:
                self.pginput.toggle_ux()
            for i in range(self.pginput.table.rowCount()):
                self.pginput.table.setItem(i, 3, QtWidgets.QTableWidgetItem(str(ux)))

    def filluy(self):
        ''' Fill all u(y) values '''
        uy, ok = QtWidgets.QInputDialog.getDouble(self, 'Enter value', 'Enter u(y) value to apply to all rows', decimals=9, min=0, max=1E99)
        if ok:
            for i in range(self.pginput.table.rowCount()):
                self.pginput.table.setItem(i, 2, QtWidgets.QTableWidgetItem(str(uy)))

    def calculate(self):
        ''' Run the calculation '''
        lsq = self.pginput.settings.chkLSQ.isChecked()
        mc = self.pginput.settings.chkMC.isChecked()
        mcmc = self.pginput.settings.chkMCMC.isChecked()
        gum = self.pginput.settings.chkGUM.isChecked()

        if len(self.fitcalc.arr.x) < 2 or len(self.fitcalc.arr.y) < 2:
            QtWidgets.QMessageBox.warning(self, 'Curve Fit', 'Need at least 2 (x,y) points to calculate a curve.')
            return

        try:
            output = self.fitcalc.calculate(lsq=lsq, gum=gum, mc=mc, mcmc=mcmc)
        except ZeroDivisionError:
            QtWidgets.QMessageBox.warning(self, 'Curve Fit', 'Could not compute solution! Ensure x and y data are entered in the table.')
        except RuntimeError as e:
            if 'Optimal parameters' in str(e):
                QtWidgets.QMessageBox.warning(self, 'Curve Fit', 'Curve fit failed to converge. Try updating the initial guess or use a different method.')
            else:
                QtWidgets.QMessageBox.warning(self, 'Curve Fit', 'Could not compute solution!')
                print(str(e))
        except TypeError as e:
            if 'Improper input' in str(e):
                QtWidgets.QMessageBox.warning(self, 'Curve Fit', 'Polynomial is overfit. Reduce order.')
            else:
                QtWidgets.QMessageBox.warning(self, 'Curve Fit', 'Could not compute solution!')
                print(str(e))
        except ValueError as e:
            if 'beta0' in str(e):
                QtWidgets.QMessageBox.warning(self, 'Curve Fit', 'Please provide initial guess.')
            else:
                QtWidgets.QMessageBox.warning(self, 'Curve Fit', 'Could not compute solution!')
                print(str(e))
        else:
            self.pgoutput.set_output(output)
            self.stack.setCurrentIndex(1)
        self.actSaveReport.setEnabled(True)

    def get_report(self):
        ''' Get full report of curve fit, using page settings '''
        return self.pgoutput.get_report()

    def save_report(self):
        ''' Save full report of curve fit, asking user for settings/filename '''
        gui_widgets.savereport(self.get_report())
