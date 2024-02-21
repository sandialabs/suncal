''' Widget for assigning a column to a name '''
from PyQt6 import QtWidgets
import numpy as np

from .. import gui_styles
from .. import gui_common


class AssignColumnWidget(QtWidgets.QDialog):
    ''' Dialog for assigning columns from a CSV file to a variable '''
    def __init__(self, data, datahdr, variables, exclusive=True, parent=None):
        super().__init__(parent=parent)
        self.data = data
        self.datahdr = datahdr
        self.variables = variables
        self.exclusive = exclusive

        font = self.font()
        font.setPointSize(10)
        self.setFont(font)
        self.setWindowTitle('Assign CSV columns to variables')
        gui_common.centerWindow(self, 800, 600)

        self.cmbVariable = QtWidgets.QComboBox()
        self.cmbVariable.addItems(['Not Used'] + variables)
        self.btnclear = QtWidgets.QPushButton('Clear Assignments')
        self.btnok = QtWidgets.QPushButton('OK')
        self.btncancel = QtWidgets.QPushButton('Cancel')
        self.btnok.setDefault(True)
        self.table = QtWidgets.QTableWidget()
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectColumns)
        self.table.currentCellChanged.connect(self.columnselected)
        self.fill_table()

        hlayout = QtWidgets.QHBoxLayout()
        hlayout.addWidget(QtWidgets.QLabel('Variable for Selected Column:'))
        hlayout.addWidget(self.cmbVariable)
        hlayout.addStretch()
        hlayout.addWidget(self.btnclear)
        blayout = QtWidgets.QHBoxLayout()
        blayout.addStretch()
        blayout.addWidget(self.btnok)
        blayout.addWidget(self.btncancel)
        layout = QtWidgets.QVBoxLayout()
        layout.addLayout(hlayout)
        layout.addWidget(self.table)
        layout.addLayout(blayout)
        self.setLayout(layout)

        self.btnok.clicked.connect(self.accept)
        self.btncancel.clicked.connect(self.reject)
        self.cmbVariable.currentIndexChanged.connect(self.assignmentchanged)
        self.btnclear.clicked.connect(self.clear_assignments)

    def fill_table(self):
        ''' Fill the table with data already in the model '''
        ncols = len(self.data)
        nrows = max(len(c) for c in self.data)
        self.table.setRowCount(nrows)
        self.table.setColumnCount(ncols)
        self.clear_assignments()
        labels = []
        for col in range(ncols):
            for row in range(len(self.data[col])):
                self.table.setItem(
                    row, col, QtWidgets.QTableWidgetItem(str(self.data[col][row])))

            # Try to match CSV columns with variables of same name
            if self.datahdr[col] in self.variables:
                labels.append(self.datahdr[col])
            else:
                labels.append('')
        self.table.setHorizontalHeaderLabels(labels)
        self.sethighlights()

    def columnselected(self, row, col, prow, pcol):
        ''' Column was selected, change combobox to match assignment '''
        with gui_common.BlockedSignals(self):
            try:
                varname = self.table.horizontalHeaderItem(col).text()
            except AttributeError:
                varname = None

            self.cmbVariable.setCurrentIndex(self.cmbVariable.findText(varname))
            if self.cmbVariable.currentIndex() < 0:
                self.cmbVariable.setCurrentIndex(0)  # Not used
            self.sethighlights()

    def assignmentchanged(self):
        ''' Column assignment combobox changed '''
        varname = self.cmbVariable.currentText()
        currentcol = self.table.currentColumn()
        if varname in ['Not Used', '']:
            self.table.setHorizontalHeaderItem(currentcol, QtWidgets.QTableWidgetItem(''))
        else:
            self.table.setHorizontalHeaderItem(currentcol, QtWidgets.QTableWidgetItem(varname))

        if self.exclusive:
            for colidx in range(self.table.columnCount()):
                if currentcol != colidx and self.table.horizontalHeaderItem(colidx).text() == varname:
                    self.table.setHorizontalHeaderItem(colidx, QtWidgets.QTableWidgetItem(''))
        self.sethighlights()

    def sethighlights(self):
        ''' Highlight columns that are assigned '''
        for colidx in range(self.table.columnCount()):
            if self.table.horizontalHeaderItem(colidx).text() not in ('', 'Not Used'):
                colcolor = gui_styles.color.column_highlight
            else:
                colcolor = gui_styles.color.transparent
            for row in range(self.table.rowCount()):
                self.table.item(row, colidx).setBackground(colcolor)

    def clear_assignments(self):
        ''' Remove all column assignments '''
        self.cmbVariable.setCurrentIndex(0)
        self.table.setHorizontalHeaderLabels(['' for _ in range(self.table.columnCount())])

    def get_column(self, column):
        ''' Get array of values for one column '''
        return self.data[column]

    def get_assignments(self):
        ''' Get dictionary of variable: data (2D) '''
        values = {}
        for varname in self.variables:
            data = []
            for col in range(self.table.columnCount()):
                if self.table.horizontalHeaderItem(col).text() == varname:
                    data.append(self.get_column(col))
            if len(data) > 0:
                values[varname] = np.atleast_2d(data)
        return values
