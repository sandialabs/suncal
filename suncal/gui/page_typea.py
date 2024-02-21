''' Page for loading/entering measurement data for Type A uncertainties '''
from PyQt6 import QtWidgets, QtCore
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from ..common import report, distributions, unitmgr
from ..datasets.dataset_model import DataSet
from ..uncertainty.variables import RandomVariable
from .widgets import FloatTableWidget, MarkdownTextEdit, PlusMinusButton, AssignColumnWidget
from .page_csvload import SelectCSVData
from .page_dataimport import DistributionSelectWidget
from . import gui_common


class TypeADataWidget(QtWidgets.QDialog):
    ''' Dialog for entering/importing Type A measured values '''
    def __init__(self, config, variable=None, project=None, parent=None):
        # config: the 'inputs' item in ProjUncert config
        super().__init__(parent=parent)
        font = self.font()
        font.setPointSize(10)
        self.setFont(font)
        self.setWindowTitle('Type A Measurement Data')
        gui_common.centerWindow(self, 1000, 800)

        self.config = config
        self.project = project
        self.varnames = [v['name'] for v in self.config]
        self.variabledata = {}  # Cached variable data to allow changing
                                # table without modifying the model until "OK" is pressed
        self.variablennew = {}  # Cached #newmeasurements
        self.importeddata = {}  # Cached variables imported from other components

        self.cmbVariable = QtWidgets.QComboBox()
        self.cmbVariable.addItems(self.varnames)
        if variable:
            self.cmbVariable.setCurrentIndex(self.cmbVariable.findText(variable))

        self.plusminus = PlusMinusButton()
        self.table = FloatTableWidget()
        self.fig = Figure()
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setStyleSheet("background-color:transparent;")
        self.stats = MarkdownTextEdit()
        self.chkCorrelate = QtWidgets.QCheckBox('Correlate variables of same length')
        self.chkAutocor = QtWidgets.QCheckBox('Autocorrelation appears significant. Check here to account for it.')
        self.chkNewmeas = QtWidgets.QCheckBox('Use this data to estimate uncertainty in')
        self.Nnewmeas = QtWidgets.QSpinBox()
        self.Nnewmeas.setRange(1, 999)
        self.Nlabel = QtWidgets.QLabel('new measurements')
        self.btnok = QtWidgets.QPushButton('OK')
        self.btncancel = QtWidgets.QPushButton('Cancel')
        self.btnload = QtWidgets.QPushButton('Load CSV...')
        self.btnimport = QtWidgets.QPushButton('Import From...')
        self.btnok.setDefault(True)
        self.btnok.clicked.connect(self.accept)
        self.btncancel.clicked.connect(self.reject)

        self.chkAutocor.setVisible(False)
        self.chkAutocor.setChecked(True)
        self.chkCorrelate.setVisible(len(self.varnames) > 1)
        self.chkCorrelate.setChecked(True)
        self.plusminus.btnplus.setToolTip('Add Column')
        self.plusminus.btnminus.setToolTip('Remove Column')

        layout = QtWidgets.QVBoxLayout()
        vlayout = QtWidgets.QHBoxLayout()
        vlayout.addWidget(QtWidgets.QLabel('Variable:'))
        vlayout.addWidget(self.cmbVariable)
        vlayout.addStretch()
        layout.addLayout(vlayout)

        # Data table and associated controls (put in widget to show/hide)
        self.datawidget = QtWidgets.QWidget()
        tlayout = QtWidgets.QHBoxLayout()
        tlayout.addWidget(QtWidgets.QLabel('Measured Values:'))
        tlayout.addStretch()
        tlayout.addWidget(self.plusminus)
        t2layout = QtWidgets.QVBoxLayout()
        t2layout.addLayout(tlayout)
        t2layout.addWidget(self.table)
        self.datawidget.setLayout(t2layout)

        # Controls for imported distribution
        self.importwidget = QtWidgets.QWidget()
        self.importlabel = QtWidgets.QLabel('Distribution imported from project component')
        self.btnclearimport = QtWidgets.QPushButton('Remove distribution')
        ilayout = QtWidgets.QVBoxLayout()
        ilayout.addWidget(self.importlabel)
        ilayout.addWidget(self.btnclearimport)
        ilayout.addStretch()
        self.importwidget.setLayout(ilayout)

        self.leftwidget = QtWidgets.QStackedWidget()
        self.leftwidget.addWidget(self.datawidget)
        self.leftwidget.addWidget(self.importwidget)
        self.rightsplitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        self.rightsplitter.addWidget(self.canvas)
        self.rightsplitter.addWidget(self.stats)
        self.splitter = QtWidgets.QSplitter()
        self.splitter.addWidget(self.leftwidget)
        self.splitter.addWidget(self.rightsplitter)
        nlayout = QtWidgets.QHBoxLayout()
        nlayout.addWidget(self.chkNewmeas)
        nlayout.addWidget(self.Nnewmeas)
        nlayout.addWidget(self.Nlabel)
        nlayout.addStretch()
        chklayout = QtWidgets.QVBoxLayout()
        chklayout.addWidget(self.chkCorrelate)
        chklayout.addWidget(self.chkAutocor)
        chklayout.addLayout(nlayout)
        mainlayout = QtWidgets.QHBoxLayout()
        mainlayout.addWidget(self.splitter)
        layout.addLayout(mainlayout)
        layout.addLayout(chklayout)
        blayout = QtWidgets.QHBoxLayout()
        blayout.addWidget(self.btnload)
        blayout.addWidget(self.btnimport)
        blayout.addStretch()
        blayout.addWidget(self.btnok)
        blayout.addWidget(self.btncancel)
        layout.addLayout(blayout)
        self.setLayout(layout)

        self.table.valueChanged.connect(self.tableedited)
        self.cmbVariable.currentIndexChanged.connect(self.changevar)  # Maybe could be straight to fill_values
        self.plusminus.plusclicked.connect(self.add_column)
        self.plusminus.minusclicked.connect(self.rem_column)
        self.chkNewmeas.stateChanged.connect(self.tableedited)
        self.Nnewmeas.valueChanged.connect(self.tableedited)
        self.chkAutocor.stateChanged.connect(self.tableedited)
        self.btnload.clicked.connect(self.loadcsv)
        self.btnimport.clicked.connect(self.importfrom)
        self.btnclearimport.clicked.connect(self.clearimport)

        if self.project is None:
            self.btnimport.setVisible(False)

        self.init_imported()
        self.fill_values()

    def init_imported(self):
        ''' Check for values that already have a typea_unc in their config '''
        for varname in self.varnames:
            varidx = self.varnames.index(varname)
            typea = self.config[varidx].get('typea')
            typea = np.atleast_2d(typea)
            typea_unc = self.config[varidx].get('typea_uncert')
            if typea.size == 1 and typea_unc is not None:
                varcfg = self.config[varidx]
                config = {'name': 'norm',
                          'median': varcfg.get('median', varcfg.get('mean', 0)),
                          'std': varcfg.get('typea_uncert', 1),
                          'df': varcfg.get('typea_degf', np.inf)}
                self.importeddata[varname] = config

    def fill_values(self):
        ''' Fill table with values from the selected variable '''
        varname = self.cmbVariable.currentText()

        if varname in self.importeddata:
            self.leftwidget.setCurrentIndex(1)
            self.plot_imported()
            return

        self.leftwidget.setCurrentIndex(0)
        if varname in self.variabledata:
            self.fill_table_from_cache(varname)
            return

        with gui_common.BlockedSignals(self.table):
            varidx = self.varnames.index(varname)
            typea = self.config[varidx].get('typea', self.config[varidx].get('mean', 0))
            typea = np.atleast_2d(typea)
            typea = unitmgr.strip_units(typea)
            ncols, nrows = typea.shape

            self.table.clear()
            self.table.setColumnCount(ncols)
            self.table.setRowCount(nrows)
            for row in range(nrows):
                for col in range(ncols):
                    self.table.setItem(row, col, QtWidgets.QTableWidgetItem(str(typea[col, row])))

            if num_new := self.config[varidx].get('numnewmeas', 0):
                self.Nnewmeas.setValue(num_new)
                self.chkNewmeas.setChecked(True)
            else:
                self.Nnewmeas.setValue(1)
                self.chkNewmeas.setChecked(False)

        self.update_plots()

    def fill_table_from_cache(self, varname):
        ''' Fill table using cached values '''
        with gui_common.BlockedSignals(self.table):
            data = self.variabledata[varname].data
            ncols, nrows = data.shape
            self.table.setColumnCount(ncols)
            self.table.setRowCount(nrows)
            for row in range(nrows):
                for col in range(ncols):
                    self.table.setItem(row, col, QtWidgets.QTableWidgetItem(str(data[col, row])))

            if num_new := self.variablennew.get(varname, 0):
                self.Nnewmeas.setValue(num_new)
                self.chkNewmeas.setChecked(True)
            else:
                self.Nnewmeas.setValue(1)
                self.chkNewmeas.setChecked(False)
        self.update_plots()

    def update_plots(self):
        ''' Update plot, stats, and checkboxes based on table data '''
        data = self.table.get_table()
        dset = DataSet(data)

        # Check for autocorrelation and enable checkbox if significant
        if (dset.result.ncolumns == 1 and
                dset.result.totalN > 50 and
                dset.result.autocorrelation[0].r_unc > 1.3):  # What's the threshold?
            self.chkAutocor.setVisible(True)
            self.chkAutocor.setEnabled(True)
        else:
            self.chkAutocor.setChecked(False)
            self.chkAutocor.setVisible(False)
            self.chkAutocor.setEnabled(False)  # Enabled state persists after window closed
                                               # and is used to check in update_config

        # Update plot/report
        self.fig.clf()
        if dset.result.ncolumns > 1:
            self.stats.setReport(dset.result.report.pooled())
            dset.result.report.plot.groups(fig=self.fig)
        else:
            self.repeat_report(dset)
            dset.result.report.plot.histogram(fig=self.fig)
        self.canvas.draw_idle()

    def tableedited(self):
        ''' Table data was edited, or checkboxes changed. Cache the data to dictionary '''
        varname = self.cmbVariable.currentText()
        data = self.table.get_table()
        self.variabledata[varname] = data
        if self.chkNewmeas.isChecked():
            self.variablennew[varname] = self.Nnewmeas.value()
        else:
            self.variablennew.pop(varname, None)
        self.update_plots()

    def repeat_report(self, dset):
        ''' Display repeatability report '''
        rpt = report.Report()

        if self.chkNewmeas.isChecked():
            nmeas = self.Nnewmeas.value()
        else:
            nmeas = dset.result.totalN

        N = dset.result.totalN
        mean = report.Number(dset.result.groups.means[0], fmin=1)
        std = report.Number(dset.result.groups.std_devs[0], fmin=1)
        rows = [['Mean', mean],
                ['Standard Deviation', std]]

        if self.chkAutocor.isChecked():
            sem = dset.result.autocorrelation[0].uncert * np.sqrt(N/nmeas)  # r is already SEM
            sem = report.Number(sem, fmin=1)
            semeq = f'rσ/√{nmeas}'
            rows.append(['Autocorrelation Factor', f'r ={dset.result.autocorrelation[0].r_unc:.3f}'])
        else:
            semeq = f'σ/√{nmeas}'
            sem = report.Number(std.value/np.sqrt(nmeas), fmin=1)

        rows.append(['Uncertainty', f'{semeq} = {sem}'])
        rpt.table(rows, hdr=['Parameter', 'Value'])
        self.stats.setReport(rpt)

    def plot_imported(self):
        ''' Plot the distribution that was imported from another project component '''
        varname = self.cmbVariable.currentText()
        params = self.importeddata[varname]
        dist = distributions.get_distribution(**params)
        x = np.linspace(*dist.interval(.9999), num=100)
        self.fig.clf()
        ax = self.fig.add_subplot(1, 1, 1)
        ax.plot(x, dist.pdf(x))
        self.canvas.draw_idle()

        rpt = report.Report()
        hdr = ['Parameter', 'Value']
        rows = [[name, val] for name, val in params.items()]
        rpt.table(rows, hdr)
        self.stats.setReport(rpt)

    def changevar(self):
        ''' Variable combobox selection was changed. Cache the table. '''
        self.fill_values()

    def update_config(self, config):
        ''' Update the config ('inputs' list) with entered typea values '''
        for varname, data in self.importeddata.items():
            varidx = self.varnames.index(varname)
            config[varidx]['mean'] = data.get('mean', data.get('median'))
            config[varidx]['typea'] = config[varidx]['mean']
            config[varidx]['typea_uncert'] = data.get('std')
            config[varidx]['typea_degf'] = data.get('df', np.inf)

        for varname, data in self.variabledata.items():
            if varname in self.importeddata:
                continue  # Skip these
            varidx = self.varnames.index(varname)
            assert config[varidx]['name'] == varname
            data = np.squeeze(data)
            units = config[varidx].get('units', None)
            nnew_meas = self.variablennew.get(varname, None)
            autocor = self.chkAutocor.isChecked() and self.chkAutocor.isEnabled()

            rv = RandomVariable()
            rv.measure(data,
                       typea=None,
                       num_new_meas=nnew_meas,
                       autocor=autocor)

            config[varidx]['mean'] = unitmgr.make_quantity(rv.expected, units)
            typea = unitmgr.make_quantity(rv.typea, units)
            if typea > 0:
                config[varidx]['typea'] = unitmgr.make_quantity(rv.value, units)
                config[varidx]['typea_uncert'] = typea
                config[varidx]['typea_degf'] = rv.degrees_freedom
                config[varidx]['autocorrelate'] = autocor
                if nnew_meas is not None:
                    config[varidx]['numnewmeas'] = nnew_meas
            else:
                config[varidx].pop('typea', None)
                config[varidx].pop('typea_uncert', None)
                config[varidx].pop('typea_degf', None)
                config[varidx].pop('autocorrelate', None)
                config[varidx].pop('numnewmeas', None)

        correlations = []
        if len(self.varnames) > 1 and self.chkCorrelate.isChecked():
            for i, var1 in enumerate(self.varnames):
                for j, var2 in enumerate(self.varnames[:i+1]):
                    if i == j: continue
                    values1 = unitmgr.strip_units(config[i].get('typea'))
                    values2 = unitmgr.strip_units(config[j].get('typea'))
                    if (values1 is not None and values2 is not None and
                            values1.ndim == 1 and values2.ndim == 1 and
                            len(values1) == len(values2)):
                        correlations.append({'var1': var1,
                                             'var2': var2,
                                             'cor': np.corrcoef(values1, values2)[0, 1]})
        return config, correlations

    def add_column(self):
        ''' Add column to the table '''
        self.table.setColumnCount(self.table.columnCount() + 1)

    def rem_column(self):
        ''' Remove column from the table '''
        self.table.setColumnCount(self.table.columnCount() - 1)

    def loadcsv(self):
        ''' Load data from a CSV file '''
        fname, _ = QtWidgets.QFileDialog.getOpenFileName(caption='CSV file to load')
        if fname:
            dlg = SelectCSVData(fname, parent=self)
            if dlg.exec():
                dset = dlg.dataset()
                data = dset.model.data
                datahdr = dset.model.colnames
                dlg2 = AssignColumnWidget(data, datahdr, self.varnames, exclusive=False)
                if dlg2.exec():
                    assignments = dlg2.get_assignments()
                    self.variabledata.update(assignments)
                    self.fill_values()

    def importfrom(self):
        ''' Import distribution from another project component '''
        dlg = DistributionSelectWidget(project=self.project)
        if dlg.exec():
            varname = self.cmbVariable.currentText()
            params = dlg.distribution()
            self.leftwidget.setCurrentIndex(1)
            self.importeddata[varname] = params
            self.fill_values()

    def clearimport(self):
        ''' Clear the distribution that was imported from another component '''
        varname = self.cmbVariable.currentText()
        self.importeddata.pop(varname, None)
        self.fill_values()
