''' UI for calibration interval calculations '''

from PyQt6 import QtWidgets, QtCore, QtGui
import matplotlib.dates as mdates
import numpy as np

from ..common import report
from . import widgets
from . import gui_styles
from .gui_common import BlockedSignals
from .help_strings import IntervalHelp
from ..intervals import datearray
from .page_csvload import SelectCSVData

from ..project import (ProjectIntervalTest, ProjectIntervalTestAssets, ProjectIntervalBinom,
                       ProjectIntervalBinomAssets, ProjectIntervalVariables, ProjectIntervalVariablesAssets)


def parse_dates(values):
    ''' Convert list of ordinal date numbers (float) into strings to display in the table '''
    parsed = []
    for val in values:
        try:
            val = datearray([val])[0]
            datestr = mdates.num2date(val).strftime('%Y-%m-%d')   # float ordinal
            parsed.append(datestr)
        except ValueError:
            parsed.append(str(val))
    return parsed


def getNewIntervalCalc():
    ''' Prompt user with questions, return an instance of Interval '''
    dlg = NewIntDialog()
    dlg.exec()
    if dlg.optA3.isChecked():
        if dlg.optAssets.isChecked():
            item = ProjectIntervalTestAssets()
        else:
            item = ProjectIntervalTest()
    elif dlg.optS2.isChecked():
        if dlg.optAssets.isChecked():
            item = ProjectIntervalBinomAssets()
        else:
            item = ProjectIntervalBinom()
    else:
        if dlg.optAssets.isChecked():
            item = ProjectIntervalVariablesAssets()
        else:
            item = ProjectIntervalVariables()
    return item


def split_assets(data, current_asset='A'):
    ''' Take CSV data with an asset list {'Asset':..., 'Start':..., 'End'..., etc}
        and split to nested dict {Asset: {'Start': ..., 'End'..., etc}}
    '''
    length = max(len(values) for values in data.values())
    assets = np.asarray(data.get('Asset', [current_asset]*length))
    newdata = {}

    # Convert between table header and config keys
    keylookup = {'Interval Start': 'startdates',
                 'Interval End': 'enddates',
                 'Pass/Fail': 'passfail',
                 'As Found': 'asfound',
                 'As Left': 'asleft'}

    for asset in set(assets):
        indexes = np.where(assets == asset)
        newdata[asset] = {}
        for key in data.keys():
            if key == 'Asset' or len(data[key]) == 0:
                continue
            newdata[asset][keylookup.get(key)] = np.asarray(data.get(key))[indexes]
    return newdata


class NewIntDialog(QtWidgets.QDialog):
    ''' Dialog for choosing type of interval calculation '''
    def __init__(self, parent=None):
        super().__init__(parent)
        font = self.font()
        font.setPointSize(10)
        self.setFont(font)
        self.setWindowTitle('Interval Calculation Setup')
        self.optA3 = QtWidgets.QRadioButton('Attributes (pass/fail only) data, all intervals similar (Method A3)')
        self.optS2 = QtWidgets.QRadioButton('Attributes (pass/fail only) data, many different intervals (Method S2)')
        self.optVar = QtWidgets.QRadioButton('Variables (as-found and/or as-left) data (Variables Method)')
        self.optAssets = QtWidgets.QRadioButton('Results from Individual Calibrations')
        self.optSummary = QtWidgets.QRadioButton('Summarized Reliability Values')
        self.btnok = QtWidgets.QPushButton('Ok')
        self.optA3.setChecked(True)
        self.optAssets.setChecked(True)

        lay1 = QtWidgets.QVBoxLayout()
        lay1.addWidget(self.optA3)
        lay1.addWidget(self.optS2)
        lay1.addWidget(self.optVar)
        lay1.addStretch()
        lay2 = QtWidgets.QVBoxLayout()
        lay2.addWidget(self.optAssets)
        lay2.addWidget(self.optSummary)
        lay2.addStretch()
        self.frame1 = QtWidgets.QFrame()
        self.frame1.setLayout(lay1)
        self.frame2 = QtWidgets.QFrame()
        self.frame2.setLayout(lay2)

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(QtWidgets.QLabel('Select type of historical data for analysis:'))
        layout.addWidget(self.frame1)
        layout.addWidget(QtWidgets.QLabel('Enter data as:'))
        layout.addWidget(self.frame2)
        layout.addStretch()
        layout.addWidget(self.btnok)
        self.setLayout(layout)
        self.btnok.clicked.connect(self.accept)


class A3ParamsWidget(QtWidgets.QGroupBox):
    ''' Widget for entering parameters for method A3 '''
    def __init__(self, showtol=False, parent=None):
        super().__init__('Test Interval Method (A3) Options', parent=parent)
        self.showtol = showtol
        self.I0 = widgets.IntLineEdit(low=1)
        self.Rt = widgets.FloatLineEdit(low=0.001, high=99.999)
        self.maxchange = widgets.IntLineEdit(low=1, high=999)
        self.mindelta = widgets.IntLineEdit(low=1)
        self.minintv = widgets.IntLineEdit(low=1)
        self.maxintv = widgets.IntLineEdit(low=1)
        self.tol = widgets.IntLineEdit(low=0)
        self.I0.setValue(365)
        self.Rt.setValue(95)
        self.mindelta.setValue(5)
        self.maxchange.setValue(2)
        self.minintv.setValue(14)
        self.maxintv.setValue(1865)
        self.tol.setValue(56)
        self.conf = widgets.FloatLineEdit(low=.001, high=99.999)
        self.conf.setValue(50)
        layout = QtWidgets.QFormLayout()
        layout.addRow('Current Assigned Interval (days)', self.I0)
        layout.addRow('Reliability Target %', self.Rt)
        layout.addRow('Maximum Change Factor', self.maxchange)
        layout.addRow('Minimum Change (days)', self.mindelta)
        layout.addRow('Minimum Allowed Interval (days)', self.minintv)
        layout.addRow('Maximum Allowed Interval (days)', self.maxintv)
        layout.addRow('Minimum Rejection Confidence %', self.conf)
        if self.showtol:
            layout.addRow('Asset Interval Tolerance (days)', self.tol)

        self.I0.setToolTip('Currently assigned interval for the calibrations in the table')
        self.Rt.setToolTip('Desired end-of-period reliability as a percent')
        self.maxchange.setToolTip('Maximum allowable change in interval, as a fraction of the current interval.\n'
                                  'The new interval will not be greater than current*maxchange or less than '
                                  'current/maxchange.\nEqual to the "b" parameter of Method A3.')
        self.mindelta.setToolTip('Minimum number of days required to change the interval.\nSuggested interval '
                                 'will remain the same if calculated interval is within this many days.')
        self.minintv.setToolTip('Minimum allowable interval, in days')
        self.maxintv.setToolTip('Maximum allowable interval, in days')
        self.conf.setToolTip('Confidence required before rejecting the current interval in favor of the new interval.')
        self.tol.setToolTip('Actual calibration interval must be within this many days of the assigned '
                            'interval to be used in the calculation.')
        self.setLayout(layout)

    def fill(self, params):
        ''' Fill widgets with param values '''
        self.I0.setValue(params.get('I0', 365))
        self.Rt.setValue(params.get('target', .95) * 100)
        self.maxchange.setValue(params.get('maxchange', 2))
        self.conf.setValue(params.get('conf', .50)*100)
        self.mindelta.setValue(params.get('mindelta', 5))
        self.minintv.setValue(params.get('minint', 14))
        self.maxintv.setValue(params.get('maxint', 1865))
        self.tol.setValue(params.get('tol', 56))

    def params(self):
        ''' Get entered parameters as dictionary '''
        return {'I0': self.I0.value(),
                'target': self.Rt.value()/100,
                'maxchange': self.maxchange.value(),
                'conf': self.conf.value()/100,
                'mindelta': self.mindelta.value(),
                'minint': self.minintv.value(),
                'maxint': self.maxintv.value()}


class S2ParamsWidget(QtWidgets.QGroupBox):
    ''' Widget for entering parameters for method S2 '''
    def __init__(self, component, showbins=False, parent=None):
        super().__init__('Binomial Method (S2) Options', parent=parent)
        self.component = component
        self.binlefts = None
        self.binwidth = None
        self.showbins = showbins
        self.Rt = widgets.FloatLineEdit(low=.001, high=99.999)
        self.Rt.setValue(self.component.params.target*100)
        self.conf = widgets.FloatLineEdit(low=0.001, high=99.999)
        self.conf.setValue(self.component.conf*100)
        self.bins = QtWidgets.QSpinBox()
        self.bins.setRange(3, 999)
        self.bins.setValue(10)
        self.btnbin = QtWidgets.QPushButton('Set bins manually...')
        layout = QtWidgets.QFormLayout()
        layout.addRow('Reliability Target %', self.Rt)
        layout.addRow('Confidence % for Interval Range', self.conf)
        if self.showbins:
            layout.addRow('Bins', self.bins)
            layout.addRow('', self.btnbin)
            self.bins.setValue(self.component.bins)

        self.Rt.setToolTip('Desired end-of-period reliability as a percent')
        self.conf.setToolTip('Confidence for calculating range of intervals')
        self.bins.setToolTip('Number of bins for condensing individual calibrations into summary statistics.')

        self.setLayout(layout)
        self.btnbin.clicked.connect(self.setbins)

    def fill(self, params):
        ''' Fill widgets with param values '''
        self.Rt.setValue(params.get('Rt', .95) * 100)
        self.conf.setValue(params.get('conf', .95)*100)
        self.bins.setValue(params.get('bins', 10))

    def params(self):
        ''' Get parameters for S2 method '''
        p = {'Rt': self.Rt.value()/100,
             'conf': self.conf.value()/100}
        if self.showbins:
            p['bins'] = self.bins.value()
            p['binlefts'] = self.binlefts
            p['binwidth'] = self.binwidth
        return p

    def setbins(self):
        ''' Set bins manually via dialog '''
        dlg = widgets.BinData(self.component)
        ok = dlg.exec()
        if ok:
            self.binlefts, self.binwidth = dlg.getbins()
            self.bins.setEnabled(False)
        else:
            self.bins.setEnabled(True)
            self.binlefts = None
            self.binwidth = None


class VarsParamsWidget(QtWidgets.QGroupBox):
    ''' Widget for entering parameters for Variables method '''
    def __init__(self, parent=None):
        super().__init__('Variables Method Options', parent=parent)
        self.u0 = widgets.FloatLineEdit()
        self.u0.setValue(0)
        self.kvalue = widgets.FloatLineEdit()
        self.kvalue.setValue(2)
        self.y0 = widgets.FloatLineEdit()
        self.y0.setValue(0)
        self.m = QtWidgets.QSpinBox()
        self.m.setRange(1, 3)
        self.utargetbox = QtWidgets.QGroupBox('Uncertainty Target')
        self.utargetbox.setCheckable(True)
        self.utarget = widgets.FloatLineEdit()
        self.utarget.setValue(1)
        self.rtargetbox = QtWidgets.QGroupBox('Reliability Target')
        self.rtargetbox.setCheckable(True)
        self.rlimL = widgets.FloatLineEdit()
        self.rlimU = widgets.FloatLineEdit()
        self.rlimL.setValue(-1)
        self.rlimU.setValue(1)
        self.conf = widgets.FloatLineEdit(low=0, high=99.999)
        self.conf.setValue(95)
        flayout = QtWidgets.QFormLayout()
        flayout.addRow('Measurement Uncertainty', self.u0)
        flayout.addRow('Uncertainty k', self.kvalue)
        flayout.addRow('Next interval as-left value', self.y0)
        flayout.addRow('Fit Polynomial Order', self.m)
        layout = QtWidgets.QVBoxLayout()
        layout.addLayout(flayout)
        ulayout = QtWidgets.QFormLayout()
        ulayout.addRow('Maximum Allowed Uncertainty', self.utarget)
        self.utargetbox.setLayout(ulayout)
        rlayout = QtWidgets.QFormLayout()
        rlayout.addRow('Lower Tolerance Limit', self.rlimL)
        rlayout.addRow('Upper Tolerance Limit', self.rlimU)
        rlayout.addRow('Confidence %', self.conf)
        self.rtargetbox.setLayout(rlayout)
        layout.addWidget(self.utargetbox)
        layout.addWidget(self.rtargetbox)
        self.setLayout(layout)

        self.u0.setToolTip('Measurement uncertainty in new measurements')
        self.y0.setToolTip('Measured/As-Left value at beginning of upcoming interval')
        self.m.setToolTip('Order of polynomial fit to deviation vs time curve')
        self.utarget.setToolTip('Maximum allowed projected uncertainty before ending interval')
        self.rlimL.setToolTip('Lower deviation limit. Interval ends when fit polynomial minus uncertainty '
                              '(at below confidence level) falls below this limit.')
        self.rlimU.setToolTip('Upper deviation limit. Interval ends when fit polynomial plus uncertainty '
                              '(at below confidence level) exceeds this limit.')
        self.conf.setToolTip('Confidence level in predicted uncertainty')

    def params(self):
        ''' Get parameters for variables method '''
        p = {'u0': self.u0.value(),
             'kvalue': self.kvalue.value(),
             'y0': self.y0.value(),
             'm': self.m.value(),
             'utarget': self.utarget.value(),
             'rlimits': (self.rlimL.value(), self.rlimU.value()),
             'rconf': self.conf.value()/100,
             'calcrel': self.rtargetbox.isChecked(),
             'calcunc': self.utargetbox.isChecked()}
        return p

    def fill(self, params):
        ''' Fill widgets with parameters '''
        self.u0.setValue(params.get('u0', 0))
        self.kvalue.setValue(params.get('kvalue', 2))
        self.y0.setValue(params.get('y0', 0))
        self.m.setValue(params.get('m', 1))
        self.utarget.setValue(params.get('utarget', 1))
        limits = params.get('rlimits', (-1, 1))
        self.rlimL.setValue(limits[0])
        self.rlimU.setValue(limits[1])
        self.conf.setValue(params.get('rconf', .95)*100)
        self.rtargetbox.setChecked(params.get('calcrel', True))
        self.utargetbox.setChecked(params.get('calcunc', True))


class IntervalWidget(QtWidgets.QWidget):
    def __init__(self, component, parent=None):
        super().__init__(parent)
        self.component = component

        self.table = self._setup_history_table()
        self.paramswidget = self._setup_params()
        self.calc_button = QtWidgets.QPushButton('Calculate')
        self.notes = QtWidgets.QPlainTextEdit()
        self.output_report = widgets.MarkdownTextEdit()

        self.actLoadData = QtGui.QAction('&Load Data From CSV...', self)
        self.actSaveReport = QtGui.QAction('&Save Report...', self)
        self.actClear = QtGui.QAction('&Clear Table', self)
        self.actSaveReport.setEnabled(False)
        self.menu = QtWidgets.QMenu('&Intervals')
        self.menu.addSeparator()
        self.menu.addAction(self.actLoadData)
        self.menu.addAction(self.actClear)
        self.menu.addSeparator()
        self.menu.addAction(self.actSaveReport)

        groupbox = QtWidgets.QGroupBox('Calibraiton Data')
        boxlayout = QtWidgets.QVBoxLayout()
        boxlayout.addWidget(self.table)
        groupbox.setLayout(boxlayout)
        llayout = QtWidgets.QVBoxLayout()
        llayout.addWidget(groupbox)
        llayout.addWidget(self.calc_button)
        r1layout = QtWidgets.QHBoxLayout()
        r1layout.addWidget(self.paramswidget)
        notesbox = QtWidgets.QGroupBox('Notes')
        nlayout = QtWidgets.QVBoxLayout()
        nlayout.addWidget(self.notes)
        notesbox.setLayout(nlayout)
        r1layout.addWidget(notesbox)
        rlayout = QtWidgets.QVBoxLayout()
        rlayout.addLayout(r1layout, stretch=1)
        outbox = QtWidgets.QGroupBox('Results')
        outlayout = QtWidgets.QVBoxLayout()
        outlayout.addWidget(self.output_report)
        outbox.setLayout(outlayout)
        rlayout.addWidget(outbox, stretch=3)
        self.leftwidget = QtWidgets.QWidget()
        self.leftwidget.setLayout(llayout)
        self.rightwidget = QtWidgets.QWidget()
        self.rightwidget.setLayout(rlayout)
        self.splitter = QtWidgets.QSplitter()
        self.splitter.addWidget(self.leftwidget)
        self.splitter.addWidget(self.rightwidget)
        self.splitter.setCollapsible(0, False)
        self.splitter.setCollapsible(1, False)
        layout = QtWidgets.QHBoxLayout()
        layout.addWidget(self.splitter)
        self.setLayout(layout)

        self.notes.setPlainText(self.component.description)
        self.calc_button.clicked.connect(self.calculate)
        self.actClear.triggered.connect(self.table.clear)
        self.actLoadData.triggered.connect(self.load_data)
        self.actSaveReport.triggered.connect(self.save_report)

    def _setup_history_table(self):
        ''' Make table with appropriate data '''
        raise NotImplementedError  # Subclass Me

    def _setup_params(self):
        ''' Set params widget '''
        raise NotImplementedError  # Subclass Me

    def update_proj_config(self):
        ''' Update the projects configuration with page values '''
        raise NotImplementedError  # Subclass Me

    def fill_column(self, col, values):
        ''' Fill a column of the table with the values '''
        if self.table.rowCount() < len(values):
            self.table.setRowCount(len(values))
        for row, value in enumerate(values):
            self.table.setItem(row, col, QtWidgets.QTableWidgetItem(str(value)))

    def load_data(self):
        ''' Import data from CSV  '''
        fname, _ = QtWidgets.QFileDialog.getOpenFileName(caption='CSV file to load')
        if fname:
            dlg = SelectCSVData(fname, parent=self)
            if dlg.exec():
                variables = self.column_names
                dset = dlg.dataset()
                data = dset.model.data
                datahdr = dset.model.colnames
                dlg2 = widgets.AssignColumnWidget(data, datahdr, variables)
                if dlg2.exec():
                    data = dlg2.get_assignments()
                    # Data comes back as np.atleast_2d. Squeeze out the extra dimension.
                    data = {name: np.squeeze(value) for name, value in data.items()}
                    self.insert_csvdata(data)

    def get_menu(self):
        ''' Get the menu for this widget '''
        return self.menu

    def get_report(self):
        ''' Get full report of curve fit, using page settings '''
        return self.component.result.report.summary()

    def save_report(self):
        ''' Save full report, asking user for settings/filename '''
        with gui_styles.LightPlotstyle():
            widgets.savereport(self.get_report())

    def calculate(self):
        ''' Run the calculation '''
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.CursorShape.WaitCursor)
        self.update_proj_config()

        try:
            self.component.calculate()
        except (TypeError, ValueError, RuntimeError):
            self.output_report.setHtml('Error computing interval.')
        else:
            self.output_report.setReport(self.get_report())
            self.actSaveReport.setEnabled(True)
        QtWidgets.QApplication.restoreOverrideCursor()

    def help_report(self):
        return IntervalHelp.test()


class IntervalA3Widget(IntervalWidget):
    ''' Widget for A3 Interval with summarized history '''
    COL_INTOL = 0
    COL_TOTAL = 1

    def _setup_history_table(self):
        self.column_names = ['Number In-Tolerance', 'Total Calibrations']
        table = widgets.FloatTableWidget()
        table.setColumnCount(2)
        table.setHorizontalHeaderLabels(self.column_names)
        table.maxrows = 1
        intol = self.component.params.intol
        n = self.component.params.n
        table.setItem(0, self.COL_INTOL, QtWidgets.QTableWidgetItem(str(intol)))
        table.setItem(0, self.COL_TOTAL, QtWidgets.QTableWidgetItem(str(n)))
        table.resizeColumnsToContents()
        return table

    def _setup_params(self):
        params = A3ParamsWidget()
        params.fill(self.component.get_config())
        return params

    def update_proj_config(self):
        ''' Save inputs back to model '''
        config = self.component.get_config()
        config['params'].update(self.paramswidget.params())
        config['params']['intol'] = self.table.get_column(self.COL_INTOL)[0]
        config['params']['n'] = self.table.get_column(self.COL_TOTAL)[0]
        self.component.load_config(config)

    def insert_csvdata(self, data):
        ''' Load data from CSV into table '''
        # data is dict with keys equal to self.column_names
        if 'Number In-Tolerance' in data:
            self.table.item(0, self.COL_INTOL).setText(str(data['Number In-Tolerance'][0]))
        if 'Total Calibrations' in data:
            self.table.item(0, self.COL_TOTAL).setText(str(data['Total Calibrations'][0]))


class IntervalS2Widget(IntervalWidget):
    ''' Widget for S2 Interval with summarized history '''
    COL_LENGTH = 0
    COL_OBSERVED = 1
    COL_TOTAL = 2

    def _setup_history_table(self):
        self.column_names = ['Interval Length', 'Observed Reliability', 'Total Calibrations']
        table = widgets.FloatTableWidget()
        table.setColumnCount(3)
        table.setHorizontalHeaderLabels(self.column_names)
        ti = self.component.params.ti
        ri = self.component.params.ri
        ni = self.component.params.ni
        for row, (t, r, n) in enumerate(zip(ti, ri, ni)):
            table.setItem(row, self.COL_LENGTH, QtWidgets.QTableWidgetItem(str(t)))
            table.setItem(row, self.COL_OBSERVED, QtWidgets.QTableWidgetItem(str(r)))
            table.setItem(row, self.COL_TOTAL, QtWidgets.QTableWidgetItem(str(n)))
        table.resizeColumnsToContents()
        return table

    def _setup_params(self):
        params = S2ParamsWidget(self.component)
        return params

    def update_proj_config(self):
        ''' Save inputs back to model '''
        config = self.component.get_config()
        config.update(self.paramswidget.params())
        config['ti'] = self.table.get_column(self.COL_LENGTH, remove_nan=True)
        config['ri'] = self.table.get_column(self.COL_OBSERVED, remove_nan=True)
        config['ni'] = self.table.get_column(self.COL_TOTAL, remove_nan=True)
        self.component.load_config(config)

    def insert_csvdata(self, data):
        ''' Load data from CSV into table '''
        # data is dict with keys equal to self.column_names
        if (values := data.get('Interval Length')) is not None:
            self.fill_column(self.COL_LENGTH, values)
        if (values := data.get('Observed Reliability')) is not None:
            self.fill_column(self.COL_OBSERVED, values)
        if (values := data.get('Total Calibrations')) is not None:
            self.fill_column(self.COL_TOTAL, values)

    def help_report(self):
        return IntervalHelp.binomial()


class IntervalVariablesWidget(IntervalWidget):
    ''' Widget for Variables Interval with summarized history '''
    COL_LENGTH = 0
    COL_DEVIATION = 1

    def __init__(self, component, parent=None):
        super().__init__(component=component, parent=parent)
        self._calc_reliability = True
        self._calc_uncertainty = True

    def _setup_history_table(self):
        self.column_names = ['Interval Length', 'Deviation from Prior']
        table = widgets.FloatTableWidget()
        table.setColumnCount(2)
        table.setHorizontalHeaderLabels(self.column_names)
        dt = self.component.data.dt
        deltas = self.component.data.deltas
        for row, (t, delt) in enumerate(zip(dt, deltas)):
            table.setItem(row, self.COL_LENGTH, QtWidgets.QTableWidgetItem(str(t)))
            table.setItem(row, self.COL_DEVIATION, QtWidgets.QTableWidgetItem(str(delt)))
        table.resizeColumnsToContents()
        return table

    def _setup_params(self):
        params = VarsParamsWidget()
        params.fill(self.component.get_config())
        return params

    def update_proj_config(self):
        ''' Save inputs back to model '''
        config = self.component.get_config()
        params = self.paramswidget.params()
        self._calc_reliability = params.pop('calcrel', True)
        self._calc_uncertainty = params.pop('calcunc', True)
        config['dt'] = self.table.get_column(self.COL_LENGTH, remove_nan=True)
        config['deltas'] = self.table.get_column(self.COL_DEVIATION, remove_nan=True)
        config.update(params)
        self.component.load_config(config)

    def insert_csvdata(self, data):
        ''' Load data from CSV into table '''
        # data is dict with keys equal to self.column_names
        if (values := data.get('Interval Length')) is not None:
            self.fill_column(self.COL_LENGTH, values)
        if (values := data.get('Deviation from Prior')) is not None:
            self.fill_column(self.COL_DEVIATION, values)

    def calculate(self):
        ''' Run the calculation '''
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.CursorShape.WaitCursor)
        self.update_proj_config()
        rpt = report.Report()

        if self._calc_reliability:
            try:
                self.component.calculate_reliability_target()
            except (TypeError, ValueError, RuntimeError):
                rpt.txt('Error computing reliability target\n\n')
            else:
                rpt.append(self.component.result_reliability.report.summary())

        if self._calc_uncertainty:
            try:
                self.component.calculate_uncertainty_target()
            except (TypeError, ValueError, RuntimeError):
                rpt.txt('Error computing uncertainty target\n\n')
            else:
                rpt.append(self.component.result_uncertainty.report.summary())

        self.output_report.setReport(rpt)
        self.actSaveReport.setEnabled(True)
        QtWidgets.QApplication.restoreOverrideCursor()

    def help_report(self):
        return IntervalHelp.variables()


class AssetTablePassFail(QtWidgets.QWidget):
    ''' Widget for entering multiple assets and their pass/fail histories '''
    COL_START = 0
    COL_END = 1
    COL_PASSFAIL = 2

    def __init__(self, assets: dict, parent=None):
        super().__init__(parent=parent)
        self.assets = assets
        self.combo_asset = widgets.ComboLabel('Asset:')
        self.add_rem_button = widgets.PlusMinusButton()
        self.startend = QtWidgets.QCheckBox('Enter start and end dates')
        self.table = widgets.FloatTableWidget()

        hlayout = QtWidgets.QHBoxLayout()
        hlayout.addWidget(self.combo_asset)
        hlayout.addStretch()
        hlayout.addWidget(self.add_rem_button)
        layout = QtWidgets.QVBoxLayout()
        layout.addLayout(hlayout)
        layout.addWidget(self.table)
        layout.addWidget(self.startend)
        self.setLayout(layout)
        self.table.setColumnCount(3)
        self.table.setHorizontalHeaderLabels(['Interval Start',
                                              'Interval End',
                                              'Pass/Fail'])

        self.combo_asset.addItems(self.assets.keys())
        self.show_startend()
        self.filltable()

        self.startend.stateChanged.connect(self.show_startend)
        self.add_rem_button.plusclicked.connect(self.add_asset)
        self.add_rem_button.minusclicked.connect(self.rem_asset)
        self.table.valueChanged.connect(self.update_asset)
        self.combo_asset.currentTextChanged.connect(self.filltable)

    def show_startend(self):
        ''' Show start/end dates when checked '''
        self.table.setColumnHidden(self.COL_START, not self.startend.isChecked())

    def rowCount(self):
        return self.table.rowCount()

    def setRowCount(self, value):
        self.table.setRowCount(value)

    def setItem(self, row, col, item):
        self.table.setItem(row, col, item)

    def filltable(self):
        ''' Fill table with data for the selected asset '''
        with BlockedSignals(self.table):
            assetname = self.combo_asset.currentText()
            asset = self.assets.get(assetname, {})
            self.table.clear()
            startdates = parse_dates(asset.get('startdates', []))
            enddates = parse_dates(asset.get('enddates', []))
            passfails = asset.get('passfail', [])
            self.table.setRowCount(max(1, len(passfails)))
            if len(startdates) == 0:
                startdates = [''] * len(enddates)
            for row, (start, end, pf) in enumerate(zip(startdates, enddates, passfails)):
                self.table.setItem(row, self.COL_START, QtWidgets.QTableWidgetItem(str(start)))
                self.table.setItem(row, self.COL_END, QtWidgets.QTableWidgetItem(str(end)))
                self.table.setItem(row, self.COL_PASSFAIL, QtWidgets.QTableWidgetItem(str(pf)))
            self.table.resizeColumnsToContents()

    def clear(self):
        ''' Clear the table '''
        self.table.clear()

    def add_asset(self):
        ''' Add asset button was pressed '''
        name, ok = QtWidgets.QInputDialog.getText(self, 'New Asset', 'Asset Name:')
        if ok:
            self.combo_asset.addItems([name])
            self.combo_asset.setCurrentIndex(self.combo_asset.count()-1)

    def rem_asset(self):
        ''' Remove asset button was pressed  '''
        if self.combo_asset.count() > 0:
            asset_to_remove = self.combo_asset.currentText()
            asset_idx = self.combo_asset.currentIndex()
            mbox = QtWidgets.QMessageBox()
            mbox.setWindowTitle('Suncal')
            mbox.setText(f'Remove asset {asset_to_remove}?')
            mbox.setStandardButtons(QtWidgets.QMessageBox.StandardButton.Yes |
                                    QtWidgets.QMessageBox.StandardButton.No)
            ok = mbox.exec()
            if ok == QtWidgets.QMessageBox.StandardButton.Yes:
                self.assets.pop(asset_to_remove, None)
                self.combo_asset.removeItem(asset_idx)

    def update_asset(self):
        ''' Table was edited, update self.assets data '''
        asset = self.combo_asset.currentText()
        self.assets[asset] = {}
        self.assets[asset]['startdates'] = self.table.get_column(self.COL_START, remove_nan=True)
        self.assets[asset]['enddates'] = self.table.get_column(self.COL_END, remove_nan=True)
        self.assets[asset]['passfail'] = self.table.get_column(self.COL_PASSFAIL, remove_nan=True)


class IntervalA3WidgetAssets(IntervalWidget):
    ''' Widget for A3 Interval with individual asset history '''

    def _setup_history_table(self):
        self.column_names = ['Asset', 'Interval Start', 'Interval End', 'Pass/Fail']
        assets = self.component.get_config().get('assets', {'A': {}})
        table = AssetTablePassFail(assets)
        if len(assets) > 0 and list(assets.values())[0].get('enddates') is not None:
            table.startend.setChecked(True)
        return table

    def _setup_params(self):
        params = A3ParamsWidget(showtol=True)
        params.fill(self.component.get_config())
        return params

    def update_proj_config(self):
        ''' Save inputs back to model '''
        config = self.component.get_config()
        params = self.paramswidget.params()
        params.pop('intol', None)  # These get filled in by A3Params.from_assets.
        params.pop('n', None)
        config['params'] = params
        config['assets'] = self.table.assets
        config['tolerance'] = self.paramswidget.tol.value()
        self.component.load_config(config)

    def insert_csvdata(self, data):
        ''' Load data from CSV into table '''
        current_asset = self.table.combo_asset.currentText()
        assets = split_assets(data, current_asset)
        self.table.assets.update(assets)

        # Remove any empty assets
        for assetname in list(self.table.assets.keys()):  # List it so we can change dictionary as we go
            values = self.table.assets[assetname]
            if assetname == '' or (len(values['enddates']) == 0 and len(values['passfail']) == 0):
                self.table.assets.pop(assetname, None)

        with BlockedSignals(self.table):
            self.table.combo_asset.clear()
            assetstrs = [str(a) for a in self.table.assets.keys()]
            self.table.combo_asset.addItems(assetstrs)
            self.table.combo_asset.setCurrentIndex(self.table.combo_asset.count()-1)
        self.table.filltable()


class IntervalS2WidgetAssets(IntervalWidget):
    ''' Widget for S2 Interval with individual asset history '''

    def _setup_history_table(self):
        self.column_names = ['Asset', 'Interval Start', 'Interval End', 'Pass/Fail']
        assets = self.component.get_config().get('assets', {'A': {}})
        table = AssetTablePassFail(assets)
        if len(assets) > 0 and list(assets.values())[0].get('enddates') is not None:
            table.startend.setChecked(True)
        return table

    def _setup_params(self):
        params = S2ParamsWidget(self.component, showbins=True)
        params.fill(self.component.get_config().get('params', {}))
        return params

    def update_proj_config(self):
        ''' Save inputs back to model '''
        config = self.component.get_config()
        config['assets'] = self.table.assets
        params = self.paramswidget.params()
        config.update(params)
        self.component.load_config(config)

    def insert_csvdata(self, data):
        ''' Load data from CSV into table '''
        current_asset = self.table.combo_asset.currentText()
        assets = split_assets(data, current_asset)
        self.table.assets.update(assets)

        # Remove any empty assets
        for assetname in list(self.table.assets.keys()):  # List it so we can change dictionary as we go
            values = self.table.assets[assetname]
            if assetname == '' or (len(values['enddates']) == 0 and len(values['passfail']) == 0):
                self.table.assets.pop(assetname, None)

        with BlockedSignals(self.table):
            self.table.combo_asset.clear()
            assetstrs = [str(a) for a in self.table.assets.keys()]
            self.table.combo_asset.addItems(assetstrs)
            self.table.combo_asset.setCurrentIndex(self.table.combo_asset.count()-1)
        self.table.filltable()

    def help_report(self):
        return IntervalHelp.binomial()


class AssetTableFoundLeft(QtWidgets.QWidget):
    ''' Widget for entering multiple assets and their Found/Left values '''
    COL_START = 0
    COL_END = 1
    COL_FOUND = 2
    COL_LEFT = 3

    def __init__(self, assets: dict, parent=None):
        super().__init__(parent=parent)
        self.assets = assets
        self.combo_asset = widgets.ComboLabel('Asset:')
        self.add_rem_button = widgets.PlusMinusButton()
        self.startend = QtWidgets.QCheckBox('Enter start and end dates')
        self.foundleft = QtWidgets.QCheckBox('Enter as-found and as-left values')
        self.table = widgets.FloatTableWidget()

        hlayout = QtWidgets.QHBoxLayout()
        hlayout.addWidget(self.combo_asset)
        hlayout.addStretch()
        hlayout.addWidget(self.add_rem_button)
        layout = QtWidgets.QVBoxLayout()
        layout.addLayout(hlayout)
        layout.addWidget(self.table)
        layout.addWidget(self.startend)
        layout.addWidget(self.foundleft)
        self.setLayout(layout)

        self.table.setColumnCount(4)
        self.table.setHorizontalHeaderLabels(['Interval Start',
                                              'Interval End',
                                              'As-Found Value',
                                              'As-Left Value'])

        self.combo_asset.addItems(self.assets.keys())
        self.show_startend()
        self.show_left()
        self.filltable()

        self.startend.stateChanged.connect(self.show_startend)
        self.foundleft.stateChanged.connect(self.show_left)
        self.add_rem_button.plusclicked.connect(self.add_asset)
        self.add_rem_button.minusclicked.connect(self.rem_asset)
        self.table.valueChanged.connect(self.update_asset)
        self.combo_asset.currentTextChanged.connect(self.filltable)

    def show_startend(self):
        ''' Show start/end dates when checked '''
        self.table.setColumnHidden(self.COL_START, not self.startend.isChecked())

    def show_left(self):
        ''' Show as-left column when checked '''
        self.table.setColumnHidden(self.COL_LEFT, not self.foundleft.isChecked())

    def rowCount(self):
        return self.table.rowCount()

    def setRowCount(self, value):
        self.table.setRowCount(value)

    def setItem(self, row, col, item):
        self.table.setItem(row, col, item)

    def clear(self):
        ''' Clear the table '''
        self.table.clear()
        self.table.setRowCount(1)

    def filltable(self):
        ''' Fill table with data for the selected asset '''
        with BlockedSignals(self.table):
            assetname = self.combo_asset.currentText()
            asset = self.assets.get(assetname, {})
            self.table.clear()
            startdates = parse_dates(asset.get('startdates', []))
            enddates = parse_dates(asset.get('enddates', []))
            found = asset.get('asfound', [])
            left = asset.get('asleft', [])
            if len(left) == 0:
                left = found
            if len(startdates) == 0:
                startdates = enddates

            self.table.setRowCount(max(1, len(enddates)))
            for row, (start, end, asfound, asleft) in enumerate(zip(startdates, enddates, found, left)):
                self.table.setItem(row, self.COL_START, QtWidgets.QTableWidgetItem(str(start)))
                self.table.setItem(row, self.COL_END, QtWidgets.QTableWidgetItem(str(end)))
                self.table.setItem(row, self.COL_FOUND, QtWidgets.QTableWidgetItem(str(asfound)))
                self.table.setItem(row, self.COL_LEFT, QtWidgets.QTableWidgetItem(str(asleft)))
            self.table.resizeColumnsToContents()

    def add_asset(self):
        ''' Add asset button was pressed '''
        name, ok = QtWidgets.QInputDialog.getText(self, 'New Asset', 'Asset Name:')
        if ok:
            self.combo_asset.addItems([name])
            self.combo_asset.setCurrentIndex(self.combo_asset.count()-1)

    def rem_asset(self):
        ''' Remove asset button was pressed  '''
        if self.combo_asset.count() > 0:
            asset_to_remove = self.combo_asset.currentText()
            asset_idx = self.combo_asset.currentIndex()
            mbox = QtWidgets.QMessageBox()
            mbox.setWindowTitle('Suncal')
            mbox.setText(f'Remove asset {asset_to_remove}?')
            mbox.setStandardButtons(QtWidgets.QMessageBox.StandardButton.Yes |
                                    QtWidgets.QMessageBox.StandardButton.No)
            ok = mbox.exec()
            if ok == QtWidgets.QMessageBox.StandardButton.Yes:
                self.assets.pop(asset_to_remove, None)
                self.combo_asset.removeItem(asset_idx)

    def update_asset(self):
        ''' Table was edited, update self.assets data '''
        asset = self.combo_asset.currentText()
        self.assets[asset] = {}
        self.assets[asset]['startdates'] = self.table.get_column(self.COL_START, remove_nan=True)
        self.assets[asset]['enddates'] = self.table.get_column(self.COL_END, remove_nan=True)
        self.assets[asset]['asfound'] = self.table.get_column(self.COL_FOUND, remove_nan=True)
        self.assets[asset]['asleft'] = self.table.get_column(self.COL_LEFT, remove_nan=True)


class IntervalVariablesWidgetAssets(IntervalWidget):
    ''' Widget for S2 Interval with individual asset history '''

    def _setup_history_table(self):
        self.column_names = ['Asset', 'Interval Start', 'Interval End', 'As Found', 'As Left']
        assets = self.component.get_config().get('assets', {'A': {}})
        table = AssetTableFoundLeft(assets)
        if len(assets) > 0 and list(assets.values())[0].get('enddates') is not None:
            table.startend.setChecked(True)
        return table

    def _setup_params(self):
        params = VarsParamsWidget()
        params.fill(self.component.get_config().get('params', {}))
        return params

    def update_proj_config(self):
        ''' Save inputs back to model '''
        config = self.component.get_config()
        config['assets'] = self.table.assets
        params = self.paramswidget.params()
        config.update(params)
        self.component.load_config(config)
        self._calc_reliability = params.get('calcrel', True)
        self._calc_uncertainty = params.get('calcunc', True)

    def insert_csvdata(self, data):
        ''' Load data from CSV into table '''
        current_asset = self.table.combo_asset.currentText()
        assets = split_assets(data, current_asset)
        self.table.assets.update(assets)

        # Remove any empty assets
        for assetname in list(self.table.assets.keys()):  # List it so we can change dictionary as we go
            values = self.table.assets[assetname]
            if assetname == '' or (len(values['enddates']) == 0 and len(values['passfail']) == 0):
                self.table.assets.pop(assetname, None)

        with BlockedSignals(self.table):
            self.table.combo_asset.clear()
            assetstrs = [str(a) for a in self.table.assets.keys()]
            self.table.combo_asset.addItems(assetstrs)
            self.table.combo_asset.setCurrentIndex(self.table.combo_asset.count()-1)
        self.table.filltable()

    def calculate(self):
        ''' Run the calculation '''
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.CursorShape.WaitCursor)
        rpt = report.Report()

        try:
            self.update_proj_config()
        except ValueError:
            # Possibly no data in table
            rpt.txt('Error computing reliability target\n\n')
        else:
            if self._calc_reliability:
                try:
                    self.component.calculate_reliability_target()
                except (TypeError, ValueError, RuntimeError):
                    rpt.txt('Error computing reliability target\n\n')
                else:
                    rpt.append(self.component.result_reliability.report.summary())

            if self._calc_uncertainty:
                try:
                    self.component.calculate_uncertainty_target()
                except (TypeError, ValueError, RuntimeError):
                    rpt.txt('Error computing uncertainty target\n\n')
                else:
                    rpt.append(self.component.result_uncertainty.report.summary())

        self.output_report.setReport(rpt)
        self.actSaveReport.setEnabled(True)
        QtWidgets.QApplication.restoreOverrideCursor()

    def help_report(self):
        return IntervalHelp.variables()
