''' Tables and TableItems '''
from PyQt6 import QtWidgets, QtCore, QtGui
import numpy as np
from dateutil.parser import parse
import matplotlib.dates as mdates

from .. import gui_styles
from .. import gui_math
from ..gui_common import BlockedSignals


class ReadOnlyTableItem(QtWidgets.QTableWidgetItem):
    ''' Table Widget Item with read-only properties '''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setFlags(QtCore.Qt.ItemFlag.ItemIsEnabled | QtCore.Qt.ItemFlag.ItemIsSelectable)


class EditableTableItem(QtWidgets.QTableWidgetItem):
    ''' Table Widget Item with editable flags '''
    # Editable is already the default


class FloatTableItem(QtWidgets.QTableWidgetItem):
    ''' Table Widget Item that only accepts floats '''
    def setData(self, role, value):
        try:
            float(value)
        except (ValueError, TypeError):
            value = '0.0'
        super().setData(role, value)


class TableItemTex(QtWidgets.QTableWidgetItem):
    ''' TableWidgetItem formatted to display Math expression '''
    ROLE_ENTERED = QtCore.Qt.ItemDataRole.UserRole + 1    # Original, user-entered data

    def __init__(self, expr='', editable=True):
        super().__init__()
        if not editable:
            self.setFlags(QtCore.Qt.ItemFlag.ItemIsEnabled | QtCore.Qt.ItemFlag.ItemIsSelectable)
        self.setExpr(expr)
        gui_styles.darkmode_signal().connect(self.changecolormode)

    def changecolormode(self):
        ''' Update light/dark mode '''
        self.setExpr(self.data(self.ROLE_ENTERED))

    def setExpr(self, expr):
        ''' Remove display text, replace with rendered math '''
        px = gui_math.pixmap_from_latex(expr)
        self.setData(QtCore.Qt.ItemDataRole.DecorationRole, px)
        self.setData(self.ROLE_ENTERED, expr)


class EditableHeaderView(QtWidgets.QHeaderView):
    ''' Table Header that is user-editable by double-clicking.

        Credit: http://www.qtcentre.org/threads/12835-How-to-edit-Horizontal-Header-Item-in-QTableWidget
        Adapted for QT5.
    '''
    headeredited = QtCore.pyqtSignal()

    def __init__(self, orientation, floatonly=False, parent=None):
        super().__init__(orientation, parent)
        self.floatonly = floatonly
        self.line = QtWidgets.QLineEdit(parent=self.viewport())
        self.line.setAlignment(QtCore.Qt.AlignmentFlag.AlignTop)
        self.line.setHidden(True)
        self.line.blockSignals(True)
        self.sectionedit = 0
        self.sectionDoubleClicked.connect(self.editHeader)
        self.line.editingFinished.connect(self.doneEditing)

    def doneEditing(self):
        self.line.blockSignals(True)
        self.line.setHidden(True)
        if self.floatonly:
            try:
                value = float(self.line.text())
            except ValueError:
                value = '---'
        else:
            value = self.line.text()
        self.model().setHeaderData(self.sectionedit, QtCore.Qt.Orientation.Horizontal,
                                   str(value), QtCore.Qt.ItemDataRole.EditRole)
        self.line.setText('')
        self.setCurrentIndex(QtCore.QModelIndex())
        self.headeredited.emit()

    def editHeader(self, section):
        edit_geometry = self.line.geometry()
        edit_geometry.setWidth(self.sectionSize(section))
        edit_geometry.moveLeft(self.sectionViewportPosition(section))
        self.line.setGeometry(edit_geometry)
        self.line.setText(str(self.model().headerData(section, QtCore.Qt.Orientation.Horizontal)))
        self.line.setHidden(False)
        self.line.blockSignals(False)
        self.line.setFocus()
        self.line.selectAll()
        self.sectionedit = section


class FloatTableWidget(QtWidgets.QTableWidget):
    ''' Widget for entering a table of floats

        Args:
            movebyrows (bool): When done editing, move the selected cell to the next row (True)
                or the next column (False).
            headeredit (string): Editable header. If None, no editing. string options are 'str' or 'float'
                to restrict header values to strings or floats.
            paste_multicol (bool): Allow pasting multiple columns (and inserting columns as necessary)
    '''
    valueChanged = QtCore.pyqtSignal()
    headerChanged = QtCore.pyqtSignal()

    def __init__(self, movebyrows=False, headeredit=None, paste_multicol=True, parent=None):
        super().__init__(parent=parent)
        font = self.font()
        font.setPointSize(12)
        self.setFont(font)
        self.movebyrows = movebyrows
        self.paste_multicol = paste_multicol
        self.maxrows = None
        self.maxcols = None
        self.setRowCount(1)
        self.setColumnCount(0)
        if headeredit is not None:
            assert headeredit in ['str', 'float']
            self.setHorizontalHeader(EditableHeaderView(
                orientation=QtCore.Qt.Orientation.Horizontal, floatonly=(headeredit == 'float')))
            self.horizontalHeader().headeredited.connect(self.valueChanged)
            self.horizontalHeader().headeredited.connect(self.headerChanged)
        QtGui.QShortcut(QtGui.QKeySequence('Ctrl+v'), self).activated.connect(self._paste)
        QtGui.QShortcut(QtGui.QKeySequence('Ctrl+c'), self).activated.connect(self._copy)
        self.cellChanged.connect(self._itemchanged)

    def clear(self):
        ''' Clear table, but not header '''
        self.setRowCount(0)
        self.setRowCount(1)

    def _paste(self):
        ''' Handle pasting data into table '''
        with BlockedSignals(self):
            startrow = self.currentRow()
            startcol = self.currentColumn()
            clipboard = QtWidgets.QApplication.instance().clipboard().text()
            rowlist = clipboard.split('\n')
            if self.maxrows is not None:
                rowlist = rowlist[:self.maxrows]
            if self.maxcols is not None:
                rowlist = ['\t'.join(r.split()[:self.maxcols-startcol]) for r in rowlist]

            j = i = 0
            for i, row in enumerate(rowlist):
                collist = row.split()
                if self.paste_multicol:
                    for j, st in enumerate(collist):
                        try:
                            val = float(st)
                        except ValueError:
                            if st.lower() in ['pass', 'fail', 'true', 'false', 'yes', 'no', 'n/a', 'none', 'null']:
                                val = st
                            else:
                                try:
                                    parse(st)
                                    val = st
                                except (ValueError, OverflowError):
                                    val = '-'

                        if self.rowCount() <= startrow+i:
                            self.setRowCount(startrow+i+1)
                        if self.columnCount() <= startcol+j:
                            self.setColumnCount(startcol+j+1)
                        self.setItem(startrow+i, startcol+j, QtWidgets.QTableWidgetItem(str(val)))
                else:
                    if self.rowCount() <= startrow + i:
                        self.setRowCount(startrow+i+1)
                    st = row
                    try:
                        val = float(st)
                    except ValueError:
                        if st.lower() in ['pass', 'fail', 'true', 'false', 'yes', 'no', 'n/a', 'none', 'null']:
                            val = st
                        else:
                            try:
                                parse(st)
                                val = st
                            except (ValueError, OverflowError):
                                val = '-'
                    j = 0
                    self.setItem(startrow+i, startcol, QtWidgets.QTableWidgetItem(str(val)))
            self.clearSelection()
            self.setCurrentCell(startrow+i, startcol+j)
            if self.maxrows is None or startrow+i+1 < self.maxrows:
                self.insertRow(startrow+i+1)  # Blank row at end
        self.valueChanged.emit()

    def _copy(self):
        ''' Copy selected cells to clipboard '''
        clipboard = QtWidgets.QApplication.clipboard()
        clipboard.clear(mode=clipboard.Mode.Clipboard)
        ranges = self.selectedRanges()
        if len(ranges) < 1:
            return
        text = ''
        for rng in ranges:
            top = rng.topRow()
            bot = rng.bottomRow()
            lft = rng.leftColumn()
            rgt = rng.rightColumn()
            rows = []
            for row in range(top, bot+1):
                cols = []
                for col in range(lft, rgt+1):
                    item = self.item(row, col)
                    cols.append(item.text() if item else '')
                rows.append('\t'.join(cols))
        text = '\n'.join(rows)
        clipboard.setText(text, mode=clipboard.Mode.Clipboard)

    def _insertrow(self):
        ''' Insert a blank row in the table '''
        if self.maxrows is None or self.rowCount() < self.maxrows:
            self.insertRow(max(0, self.currentRow()))
            self.valueChanged.emit()

    def _removerow(self):
        ''' Remove row from table '''
        self.removeRow(self.currentRow())
        self.valueChanged.emit()

    def keyPressEvent(self, event):
        ''' Key was pressed. Capture delete key to clear selected items '''
        items = self.selectedItems()
        if event.key() == QtCore.Qt.Key.Key_Delete and len(items) > 0:
            with BlockedSignals(self):
                for item in items:
                    item.setText('')
            self.valueChanged.emit()
        else:
            super().keyPressEvent(event)

    def contextMenuEvent(self, event):
        menu = QtWidgets.QMenu(self)
        actCopy = QtGui.QAction('Copy', self)
        actPaste = QtGui.QAction('Paste', self)
        actPaste.setEnabled(QtWidgets.QApplication.instance().clipboard().text() != '')
        actInsert = QtGui.QAction('Insert Row', self)
        actRemove = QtGui.QAction('Remove Row', self)
        menu.addAction(actCopy)
        menu.addAction(actPaste)
        menu.addSeparator()
        menu.addAction(actInsert)
        menu.addAction(actRemove)
        actPaste.triggered.connect(self._paste)
        actCopy.triggered.connect(self._copy)
        actInsert.triggered.connect(self._insertrow)
        actRemove.triggered.connect(self._removerow)
        menu.popup(event.globalPos())

    def _itemchanged(self, row, col):
        ''' Item was changed. Add new row and move selected cell as appropriate. '''
        item = self.item(row, col)
        if item and item.text() != '':
            try:
                float(item.text())
            except ValueError:
                if item.text().lower() not in ['pass', 'true', 'fail', 'false',  'yes', 'no', 'none', 'n/a', 'null']:
                    try:
                        parse(item.text()).toordinal()
                    except (ValueError, OverflowError):
                        item.setText('-')
                    try:
                        parse(item.text()).toordinal()
                    except (ValueError, OverflowError):
                        item.setText('-')

        if row == self.rowCount() - 1 and item is not None and item.text() != '':
            # Edited last row. Add a blank one
            if self.maxrows is None or row+1 < self.maxrows:
                self.insertRow(row+1)
                self.setRowHeight(row+1, self.rowHeight(row))

        # Move cursor to next row or column
        # TODO: setCurrentCell triggers a Qt warning
        if self.movebyrows:
            self.setCurrentCell(row+1, col)
        elif col == self.columnCount() - 1:
            self.setCurrentCell(row+1, 0)
        else:
            self.setCurrentCell(row, col+1)
        self.valueChanged.emit()

    def has_dates(self, column=0):
        ''' Determine if the data has datetime in column '''
        hasdates = False
        for i in range(self.rowCount()):
            text = self.item(i, column).text() if self.item(i, column) else ''
            try:
                float(text)
            except ValueError:
                try:
                    mdates.date2num(parse(text))
                except (ValueError, OverflowError):
                    pass
                else:
                    hasdates = True
                    break
        return hasdates

    def get_column(self, column, remove_nan=False):
        ''' Get array of values for one column '''
        vals = []
        for i in range(self.rowCount()):
            text = self.item(i, column).text() if self.item(i, column) else ''
            try:
                vals.append(float(text))
            except ValueError:
                if text.lower() in ['p', 'pass', 't', 'true', 'y', 'yes']:
                    vals.append(1)
                elif text.lower() in ['f', 'fail', 'false', 'n', 'no']:
                    vals.append(0)
                else:
                    try:
                        vals.append(mdates.date2num(parse(text)))
                    except (ValueError, OverflowError):
                        if not remove_nan:
                            vals.append(np.nan)
        return np.asarray(vals)

    def get_columntext(self, column):
        ''' Get array of string values in column '''
        vals = []
        for i in range(self.rowCount()):
            try:
                vals.append(self.item(i, column).text())
            except AttributeError:
                vals.append('')
        return vals

    def get_table(self):
        ''' Get 2D array of values for entire table '''
        vals = []
        for col in range(self.columnCount()):
            vals.append(self.get_column(col))
        try:
            tbl = np.vstack(vals)
            while tbl.shape[1] > 0 and all(~np.isfinite(tbl[:, -1])):
                # Strip blank rows
                tbl = tbl[:, :-1]
        except ValueError:
            # No rows
            tbl = np.array([[]])

        return tbl

    def insert_data(self, data, col: int = 0, row: int = 0):
        data = np.atleast_2d(data)
        ncols, nrows = data.shape
        if self.rowCount() < row + nrows:
            self.setRowCount(row+nrows)
        if self.columnCount() < col + ncols:
            self.setColumnCount(col+ncols)

        for irow in range(nrows):
            for icol in range(ncols):
                self.setItem(irow+row, icol+col, QtWidgets.QTableWidgetItem(str(data[icol, irow])))
