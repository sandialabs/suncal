''' Page for loading raw data from a CSV file '''

from io import StringIO
import csv
import numpy as np
from dateutil.parser import parse

from PyQt5 import QtWidgets, QtCore

from .. import dataset
from . import gui_widgets



def _gettype(val):
    ''' Determine type of string, whether it can be converted to float, date, or str. '''
    try:
        float(val)
        return 'float'
    except ValueError:
        try:
            parse(val)
            return 'date'
        except (ValueError, OverflowError):
            return 'str'


class SelectCSVData(QtWidgets.QDialog):
    ''' Widget for displaying raw CSV and letting user select which data ranges
        to import
    '''
    def __init__(self, fname, parent=None):
        super().__init__(parent=parent)
        font = self.font()
        font.setPointSize(10)
        self.setFont(font)
        gui_widgets.centerWindow(self, 900, 600)
        self.table = QtWidgets.QTableWidget()
        self.transpose = QtWidgets.QCheckBox('Transpose')
        self.dlgbutton = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        self.dlgbutton.rejected.connect(self.reject)
        self.dlgbutton.accepted.connect(self.checkdatarange)
        self.dlgbutton.button(QtWidgets.QDialogButtonBox.Ok).setEnabled(False)
        layout = QtWidgets.QVBoxLayout()
        hlayout = QtWidgets.QHBoxLayout()
        hlayout.addWidget(QtWidgets.QLabel('Select data range(s) to import'))
        hlayout.addStretch()
        hlayout.addWidget(self.transpose)
        layout.addLayout(hlayout)
        layout.addWidget(self.table)
        layout.addWidget(self.dlgbutton)
        self.setLayout(layout)

        self.loadfile(fname)
        self.table.itemSelectionChanged.connect(self.selection_change)

    def loadfile(self, fname):
        ''' Populate the table with CSV values '''
        if fname == '_clipboard_':
            rawcsv = QtWidgets.QApplication.instance().clipboard().text()
            csvfile = StringIO(rawcsv)
        else:
            csvfile = open(fname, 'r')
        try:
            dialect = csv.Sniffer().sniff(csvfile.read(1024), [',', ';', ' '])
        except (csv.Error, UnicodeDecodeError):
            dialect = None
        csvfile.seek(0)
        try:
            reader = csv.reader(csvfile, dialect)
        except (csv.Error, UnicodeDecodeError):
            QtWidgets.QMessageBox.warning(None, 'CSV Error', 'Could not determine CSV format.')
            csvfile.close()
            return
        lines = list(reader)
        csvfile.close()

        self.table.setRowCount(len(lines))
        self.table.setColumnCount(len(lines[0]))
        for row, columns in enumerate(lines):
            for col, val in enumerate(columns):
                self.table.setItem(row, col, QtWidgets.QTableWidgetItem(val.strip()))
                if self.table.item(row, col) is not None:
                    self.table.item(row, col).setFlags(QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEnabled)  # No editable
        self.table.resizeColumnsToContents()
        self.table.setRangeSelected(QtWidgets.QTableWidgetSelectionRange(0, 0, self.table.rowCount()-1, self.table.columnCount()-1), True)
        self.selection_change()

    def selection_change(self):
        rng = self.table.selectedRanges()
        self.dlgbutton.button(QtWidgets.QDialogButtonBox.Ok).setEnabled(len(rng) > 0)

    def checkdatarange(self):
        ''' Convert selected data into an array, with header if a header is detected '''
        columns = []
        # Get selected cell text
        if self.transpose.isChecked():
            for rng in self.table.selectedRanges():
                for row in range(rng.rowCount()):
                    rstart = rng.leftColumn()
                    rcount = rng.columnCount()
                    cidx = rng.topRow() + row
                    c = []
                    for i in range(rcount):
                        item = self.table.item(cidx, rstart+i)
                        c.append(item.text() if item else '')
                    columns.append(c)
        else:
            for rng in self.table.selectedRanges():
                for col in range(rng.columnCount()):
                    rstart = rng.topRow()
                    rcount = rng.rowCount()
                    cidx = rng.leftColumn() + col
                    c = []
                    for i in range(rcount):
                        item = self.table.item(rstart+i, cidx)
                        c.append(item.text() if item else '')
                    columns.append(c)

        # See if first row has same parsed datatype as second row
        # If it doesn't, use first row as header
        hdr = []
        for col in columns:
            if len(col) > 1 and _gettype(col[0]) != _gettype(col[1]):
                hdr = [c[0] for c in columns]
                columns = [c[1:] for c in columns]
                break

        # Convert each column to a float or a date if possible
        datcolumns = []
        for col in columns:
            try:
                datcol = [float(v) for v in col]
            except ValueError:
                try:
                    datcol = [parse(v) for v in col]
                except (ValueError, OverflowError):
                    datcol = col
            datcolumns.append(np.array(datcol))

        # self.columns will be array of floats or of datetimes
        self.columns = datcolumns
        self.header = hdr
        self.accept()

    def dataset(self):
        ''' Get DataSet object (assumes checkdatarange already called to validate) '''

        def padnans(v, fillval=np.nan):
            # https://stackoverflow.com/questions/40569220/efficiently-convert-uneven-list-of-lists-to-minimal-containing-array-padded-with
            lens = np.array([len(item) for item in v])
            mask = lens[:, None] > np.arange(lens.max())
            out = np.full(mask.shape, fillval)
            out[mask] = np.concatenate(v)
            return out

        data = self.columns
        hdr = self.header if len(self.header) > 0 else [f'Column {i}' for i in range(len(self.columns))]
        return dataset.DataSet(data, hdr)
