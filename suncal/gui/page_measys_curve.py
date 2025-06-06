''' Measurement System (Specific Measured Values) Page '''
from PyQt6 import QtGui, QtWidgets, QtCore

from ..common import uparser
from .gui_common import BlockedSignals
from ..meassys.curve import SystemCurve
from .page_csvload import SelectCSVData
from . import gui_common
from . import gui_styles
from . import widgets
from . import icons
from .help_strings import SystemHelp


def set_background(item: QtWidgets.QTableWidgetItem, valid: bool):
    ''' Set background color of a table item '''
    item.setBackground(gui_styles.color.transparent if valid else gui_styles.color.invalid)


class FillUncertainty(QtWidgets.QDialog):
    def __init__(self, names: list[str]):
        super().__init__()
        self.cmbName = widgets.ComboLabel('Variable:', reversed(names))
        self.value = widgets.LineEditLabelWidget('Uncertainty', '')
        self.buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Ok |
                                                  QtWidgets.QDialogButtonBox.StandardButton.Cancel)
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.cmbName)
        layout.addWidget(self.value)
        layout.addWidget(self.buttons)
        self.setLayout(layout)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)


class TabUnits(QtWidgets.QTableWidget):
    def __init__(self, qty: SystemCurve, parent=None):
        super().__init__(parent=parent)
        self.loading: bool = False
        self.qty = qty
        self.setColumnCount(2)
        self.setHorizontalHeaderLabels(['Variable', 'Units'])
        self.fill_table()
        self.itemChanged.connect(self.edit_item)

    def fill_table(self):
        ''' Fill table with values from quantity '''
        self.loading = True
        names = self.qty.data_names_direct()
        self.clear()
        self.setHorizontalHeaderLabels(['Variable', 'Units'])
        self.setRowCount(len(names))
        for row, name in enumerate(names):
            unit = self.qty.units.get(name)
            self.setItem(row, 0, widgets.ReadOnlyTableItem(name))
            self.setItem(row, 1, widgets.EditableTableItem(str(unit) if unit else ''))
            # COULD DO: add inferred units for coefficients
        self.loading = False

    def edit_item(self, item):
        ''' A table item was edited '''
        if not self.loading:
            units = item.text()
            name = self.item(item.row(), 0).text()
            try:
                uparser.parse_unit(units)
            except ValueError:
                set_background(item, False)
            else:
                self.qty.units[name] = units


class CurveDataWidget(QtWidgets.QWidget):
    ''' Popup widget for editing curve fit data '''
    varchanged = QtCore.pyqtSignal()

    def __init__(self, qty: SystemCurve, parent=None):
        super().__init__(parent=parent)
        self.loading: bool = False
        self.qty = qty
        self.predictor = widgets.ComboLabel('Predictor (x) variable:', ['x', 'y'])
        self.response = widgets.ComboLabel('Response (y) variable:', ['x', 'y'])
        self.response.setCurrentIndex(1)
        self.btn_load = QtWidgets.QPushButton('Load CSV')
        self.btn_fill = QtWidgets.QPushButton('Fill Uncertainty')
        self.btn_addrem = widgets.PlusMinusButton()
        self.table = widgets.FloatTableWidget(headeredit='str')

        hlayout = QtWidgets.QHBoxLayout()
        hlayout.addWidget(self.btn_addrem)
        hlayout.addStretch()
        hlayout.addWidget(self.btn_fill)
        hlayout.addWidget(self.btn_load)
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.predictor)
        layout.addWidget(self.response)
        layout.addLayout(hlayout)
        layout.addWidget(self.table)
        self.setLayout(layout)
        self.table.setColumnCount(2)
        self.table.setRowCount(1)
        self.table.setHorizontalHeaderLabels(['x', 'y'])
        self.btn_load.clicked.connect(self._loadcsv)
        self.table.valueChanged.connect(self.edit_table)
        self.btn_addrem.plusclicked.connect(self.add_column)
        self.btn_addrem.minusclicked.connect(self.rem_column)
        self.table.headerChanged.connect(self.headerchanged)
        self.predictor.currentIndexChanged.connect(self.var_changed)
        self.response.currentIndexChanged.connect(self.var_changed)
        self.btn_fill.clicked.connect(self.fill_uncertainty)
        self.fill_table()

    def fill_uncertainty(self):
        ''' Fill a column with values '''
        varnames = []
        for name, data in self.qty.data:
            if not isinstance(data, str) and not name.startswith('u('):
                varnames.append(name)

        dlg = FillUncertainty(varnames)
        if dlg.exec():
            variable = dlg.cmbName.currentText()
            name = f'u({variable})'
            value = dlg.value.text().strip()
            nominal = self.qty.get_column(variable)

            if value.endswith('%'):
                value = value.strip('%')
                try:
                    value = float(value) / 100
                except (ValueError, TypeError):
                    value = 0
                value = nominal * value
            else:
                try:
                    value = float(value)
                except (ValueError, TypeError):
                    value = 0
            self.qty.fill_column(name, value)
            self.fill_table()

    def fill_table(self):
        ''' Fill the table with values from the quantity '''
        if self.qty.data:
            self.loading = True
            varnames = []
            for name, data in self.qty.data:
                if isinstance(data, str):
                    varnames.append(f'{name} = {data}')
                else:
                    varnames.append(name)
            self.table.setColumnCount(len(varnames))
            self.table.setRowCount(1)
            self.table.setHorizontalHeaderLabels(varnames)
            for col, (name, values) in enumerate(self.qty.data):
                self.table.insert_data(values, col, 0)
            self.headerchanged()
            self.loading = False

    def var_changed(self):
        ''' Predictor/Reponse assignment changed '''
        if not self.loading:
            self.qty.predictor_var = self.predictor.currentText()
            self.qty.response_var = self.response.currentText()

    def _loadcsv(self):
        ''' Load table from CSV file '''
        fname, _ = QtWidgets.QFileDialog.getOpenFileName(caption='CSV file to load')
        if fname:
            dlg = SelectCSVData(fname, parent=self)
            if dlg.exec():
                dset = dlg.dataset()
                data = dset.model.data
                self.qty.data = []
                for i, column in enumerate(data):
                    try:
                        name = self.table.horizontalHeaderItem(i).text()
                    except AttributeError:
                        name = 'x'
                    self.qty.data.append((name, column))
                self.fill_table()
                self.varchanged.emit()

    def edit_table(self):
        ''' Table was edited '''
        if not self.loading:
            self.qty.data = []
            for col in range(self.table.columnCount()):
                data = self.table.get_column(col, remove_nan=True)
                try:
                    name = self.table.horizontalHeaderItem(col).text()
                except AttributeError:
                    name = ''
                if '=' in name:
                    name, data = name.split('=')
                    name = name.strip()
                    data = data.strip()
                    with BlockedSignals(self.table):
                        for row in range(self.table.rowCount()):
                            self.table.setItem(row, col, widgets.ReadOnlyTableItem('<calculated>'))
                self.qty.data.append((name, data))
                self.varchanged.emit()

    def headerchanged(self):
        ''' The table header was edited '''
        varnames = [name for name, _ in self.qty.data if not name.startswith('u(')]
        self.loading = True
        self.predictor.clear()
        self.predictor.addItems(varnames)
        self.response.clear()
        self.response.addItems(varnames)
        self.predictor.setCurrentIndex(self.predictor.findText(self.qty.predictor_var))
        self.response.setCurrentIndex(self.response.findText(self.qty.response_var))
        self.varchanged.emit()
        self.loading = False

    def add_column(self):
        ''' Add a blank column to the table '''
        self.table.setColumnCount(self.table.columnCount() + 1)
        self.table.setHorizontalHeaderItem(self.table.columnCount()-1, QtWidgets.QTableWidgetItem('x'))
        self.varchanged.emit()

    def rem_column(self):
        ''' Remove a column from the table '''
        idx = self.table.selectedIndexes()
        if idx:
            col = idx[0].column()
            self.table.removeColumn(col)
        else:
            self.table.removeColumn(self.table.columnCount() - 1)


class MeasSysCurveWidget(QtWidgets.QDialog):
    ''' Popup widget for editing curve fit data '''
    def __init__(self, qty: SystemCurve, parent=None):
        super().__init__(parent=parent)
        font = self.font()
        font.setPointSize(10)
        self.setFont(font)
        gui_common.centerWindow(self, 800, 600)

        self.tab_data = CurveDataWidget(qty)
        self.tab_units = TabUnits(qty)
        self.tab = QtWidgets.QTabWidget()
        self.tab.addTab(self.tab_data, 'Measured Data')
        self.tab.addTab(self.tab_units, 'Units')
        self.btnok = QtWidgets.QPushButton('Done')
        self.btnok.clicked.connect(self.accept)

        self.tab_data.varchanged.connect(self.tab_units.fill_table)

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.tab)
        layout.addWidget(self.btnok)
        self.setLayout(layout)
