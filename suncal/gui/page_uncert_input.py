from contextlib import suppress
from copy import deepcopy

import sympy
import numpy as np
from PyQt5 import QtWidgets, QtGui, QtCore
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from pint import PintError

from ..common import report, uparser, ttable, unitmgr
from ..uncertainty.variables import Typeb, RandomVariable
from ..uncertainty import Model
from . import gui_common
from . import gui_widgets
from . import page_dataimport


TREESTYLE = """
QTreeView {
    show-decoration-selected: 1;
}
"""


TABLESTYLE = """
QTableView {
    show-decoration-selected: 1;
}
QTableView::item:selected {
    border: 1px solid #567dbc;
}
QTableView::item:selected:active{
    background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1, stop: 0 #6ea1f1, stop: 1 #567dbc);
}
QTableView::item:selected:!active {
    background: palette(base);
    border: none;
}"""


class TableItemTex(QtWidgets.QTableWidgetItem):
    ROLE_ENTERED = QtCore.Qt.UserRole + 1    # Original, user-entered data

    ''' TableWidgetItem formatted to display Math expression '''
    def __init__(self, expr='', editable=True):
        super().__init__()
        if not editable:
            self.setFlags(QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsSelectable)
        self.setExpr(expr)

    def setExpr(self, expr):
        # Remove display text, replace with rendered math
        px = QtGui.QPixmap()
        ratio = QtWidgets.QApplication.instance().devicePixelRatio()
        tex = uparser.parse_math_with_quantities_to_tex(expr)
        px.loadFromData(report.Math.from_latex(tex).svg_buf(fontsize=16*ratio).read())
        px.setDevicePixelRatio(ratio)
        self.setData(QtCore.Qt.DecorationRole, px)
        self.setData(self.ROLE_ENTERED, expr)


class FunctionTableWidget(QtWidgets.QTableWidget):
    ''' Table for defining measurement model functions

        Signals
        -------
        funcchanged: One of the functions was modified
        funcremoved: A function was removed
        orderchange: Functions were reordered via drag/drop
        resizerows: Emitted when the number of rows in the table changes
    '''
    funcchanged = QtCore.pyqtSignal(list)
    resizerows = QtCore.pyqtSignal()
    COL_NAME = 0
    COL_EXPR = 1
    COL_UNIT = 2
    COL_DESC = 3
    COL_CNT = 4

    ROLE_ENTERED = QtCore.Qt.UserRole + 1    # Original, user-entered data

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setColumnCount(self.COL_CNT)
        self.setDragEnabled(True)
        self.setAcceptDrops(True)
        self.viewport().setAcceptDrops(True)
        self.setDragDropOverwriteMode(False)
        self.setDropIndicatorShown(True)
        self.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.setDragDropMode(QtWidgets.QAbstractItemView.InternalMove)
        self._delegate = gui_widgets.LatexDelegate()  # Assign to self - don't let the garbage collector eat it
        self.setItemDelegateForColumn(self.COL_NAME, self._delegate)
        self.setItemDelegateForColumn(self.COL_EXPR, self._delegate)
        self.setSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum)
        self.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.setStyleSheet(TABLESTYLE)
        self.verticalHeader().hide()
        self.verticalHeader().setDefaultSectionSize(48)
        self.setSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum)
        self.clear()
        self.addRow()
        self.cellChanged.connect(self.itemEdit)
        self.fixSize()

    def fixSize(self):
        ''' Adjust the size of the table to fit the number of rows '''
        height = max(self.horizontalHeader().height()+20,
                     self.verticalHeader().length() + self.horizontalHeader().height())
        self.setFixedHeight(height)
        self.resizerows.emit()

    def clear(self):
        ''' Clear everything in the table. Overrides TableWidget clear. '''
        self.setRowCount(0)
        self.setHorizontalHeaderLabels(['Name', 'Expression',  'Units', 'Description'])
        self.resizeColumnsToContents()
        # NOTE: with window width 1200, sum of columnwidths 1110 fills the tree
        self.setColumnWidth(self.COL_NAME, 100)
        self.setColumnWidth(self.COL_EXPR, 440)
        self.setColumnWidth(self.COL_UNIT, 120)
        self.setColumnWidth(self.COL_DESC, 460)
        self.fixSize()

    def get_config(self):
        ''' Get config dictionary of functions '''
        functions = []
        if self.is_valid():
            for row in range(self.rowCount()):
                units = self.item(row, self.COL_UNIT).text()
                functions.append({
                    'name': self.item(row, self.COL_NAME).data(self.ROLE_ENTERED),
                    'expr': self.item(row, self.COL_EXPR).data(self.ROLE_ENTERED),
                    'units': units if units else None,
                    'desc': self.item(row, self.COL_DESC).text()})
            functions = [f for f in functions if f['expr'] and f['name']]
        return functions

    def load_config(self, config):
        ''' Load the uncert project/model '''
        self.blockSignals(True)
        self.clear()
        for row, function in enumerate(config.get('functions', [])):
            self.addRow()
            self.setItem(row, self.COL_NAME, TableItemTex(function['name']))
            self.setItem(row, self.COL_EXPR, TableItemTex(function['expr']))
            self.setItem(row, self.COL_UNIT, gui_widgets.EditableTableItem(function.get('units', '')))
            self.setItem(row, self.COL_DESC, gui_widgets.EditableTableItem(function.get('desc', '')))
        self.blockSignals(False)
        self.funcchanged.emit(self.get_config())
        self.fixSize()

    def addRow(self):
        ''' Add an empty row to function table '''
        self.blockSignals(True)
        rows = self.rowCount()
        self.setRowCount(rows + 1)
        self.setItem(rows, self.COL_NAME, TableItemTex())
        self.setItem(rows, self.COL_EXPR, TableItemTex())
        self.setItem(rows, self.COL_UNIT, gui_widgets.EditableTableItem())
        self.setItem(rows, self.COL_DESC, gui_widgets.EditableTableItem())
        self.blockSignals(False)
        self.fixSize()

    def remRow(self):
        ''' Remove row at index from function table '''
        idx = self.selectedItems()
        if len(idx) > 0:
            idx = idx[0].row()
            self.removeRow(idx)
            self.fixSize()
            self.funcchanged.emit(self.get_config())

    def itemText(self, row, col):
        ''' Get text of item, if item exists. '''
        if self.item(row, col) is not None:
            if col in [self.COL_NAME, self.COL_EXPR]:
                return self.item.data(self.ROLE_ENTERED)
            return self.item(row, col).text()
        return ''

    def itemEdit(self, row, col):
        ''' A cell was changed. '''
        self.blockSignals(True)
        ok = False
        name = self.item(row, self.COL_NAME).data(self.ROLE_ENTERED)
        expr = self.item(row, self.COL_EXPR).data(self.ROLE_ENTERED)
        unit = self.item(row, self.COL_UNIT).text()
        if expr is None:
            expr = self.item(row, self.COL_EXPR).text()

        # Check Expression
        try:
            fn = uparser.parse_math_with_quantities(expr, name=name)
        except (ValueError, PintError):
            fn = None
        else:
            ok = True

        try:
            sname = uparser.parse_math(name)
        except ValueError:
            sname = None
            ok = False

        if self.item(row, self.COL_EXPR) is not None:
            if fn is not None:
                self.item(row, self.COL_EXPR).setBackground(QtGui.QBrush(gui_common.COLOR_OK))
            else:
                self.item(row, self.COL_EXPR).setBackground(QtGui.QBrush(gui_common.COLOR_INVALID))
                self.clearSelection()

        if col == self.COL_NAME and self.item(row, self.COL_NAME) is not None:
            if isinstance(sname, sympy.Basic):
                self.item(row, self.COL_NAME).setBackground(QtGui.QBrush(gui_common.COLOR_OK))
            else:
                self.item(row, self.COL_NAME).setBackground(QtGui.QBrush(gui_common.COLOR_INVALID))
                self.clearSelection()

        # Check units
        if unit != '':
            try:
                unit = uparser.parse_unit(unit)
            except ValueError:
                self.item(row, self.COL_UNIT).setBackground(QtGui.QBrush(gui_common.COLOR_INVALID))
                self.item(row, self.COL_UNIT).setText(unit)
                self.clearSelection()
            else:
                self.item(row, self.COL_UNIT).setBackground(QtGui.QBrush(gui_common.COLOR_OK))
                self.item(row, self.COL_UNIT).setText(f'{unit:~P}')
                unit = str(unit)
        else:
            unit = None

        self.blockSignals(False)
        if ok:
            self.funcchanged.emit(self.get_config())

    def drop_on(self, event):
        ''' Get row to drop on (see https://stackoverflow.com/a/43789304) '''
        index = self.indexAt(event.pos())
        if not index.isValid():
            return self.rowCount()
        return index.row() + 1 if self.is_below(event.pos(), index) else index.row()

    def dropEvent(self, event):
        ''' Process drop event '''
        self.blockSignals(True)
        if not event.isAccepted() and event.source() == self:
            drop_row = self.drop_on(event)
            rows = sorted(set(item.row() for item in self.selectedItems()))
            rows_to_move = [[QtWidgets.QTableWidgetItem(self.item(row_index, column_index))
                            for column_index in range(self.columnCount())]
                            for row_index in rows]
            for row_index in reversed(rows):
                self.removeRow(row_index)
                if row_index < drop_row:
                    drop_row -= 1

            for row_index, data in enumerate(rows_to_move):
                row_index += drop_row
                self.insertRow(row_index)
                for column_index, column_data in enumerate(data):
                    self.setItem(row_index, column_index, column_data)

            event.accept()
            for row_index in range(len(rows_to_move)):
                self.item(drop_row + row_index, 0).setSelected(True)
                self.item(drop_row + row_index, 1).setSelected(True)

            # Remove blank rows to maintain consistency with functions list
            for row_idx in reversed(range(self.rowCount())):
                if ((self.item(row_idx, self.COL_EXPR) is None or
                   self.item(row_idx, self.COL_EXPR).data(self.ROLE_ENTERED) == '')):
                    self.removeRow(row_idx)

        super().dropEvent(event)
        self.blockSignals(False)

    def is_below(self, pos, index):
        ''' Check if index is below position. Used for drag/drop '''
        rect = self.visualRect(index)
        margin = 2
        if pos.y() - rect.top() < margin:
            return False
        elif rect.bottom() - pos.y() < margin:
            return True
        return (rect.contains(pos, True) and
                not (int(self.model().flags(index)) & QtCore.Qt.ItemIsDropEnabled) and
                pos.y() >= rect.center().y())

    def is_valid(self):
        ''' Return True if all entries are valid. '''
        for row in range(self.rowCount()):
            if (self.item(row, self.COL_EXPR).background().color() == gui_common.COLOR_INVALID or
               self.item(row, self.COL_NAME).background().color() == gui_common.COLOR_INVALID or
               self.item(row, self.COL_UNIT).background().color() == gui_common.COLOR_INVALID):
                return False
        return True

    def contextMenuEvent(self, event):
        ''' Show right-click context menu '''
        item = self.itemAt(event.pos())
        if item:
            row = item.row()
            name = self.item(row, self.COL_NAME).data(self.ROLE_ENTERED)
            expr = self.item(row, self.COL_EXPR).data(self.ROLE_ENTERED)
            if expr is None:
                expr = self.item(row, self.COL_EXPR).text()
            try:
                fn = uparser.parse_math(expr, name=name)
            except ValueError:
                return

            menu = QtWidgets.QMenu()
            add = menu.addAction('Add measurement function')
            rem = menu.addAction('Remove measurement function')
            add.triggered.connect(self.addRow)
            rem.triggered.connect(self.remRow)

            solve = menu.addMenu('Solve for')
            for symbol in fn.free_symbols:
                act = solve.addAction(str(symbol))
                act.triggered.connect(
                    lambda x, fn=fn, name=name, var=str(symbol), row=row: self.solvefor(fn, name, var, row))
            menu.exec(event.globalPos())

        else:
            menu = QtWidgets.QMenu()
            add = menu.addAction('Add measurement function')
            add.triggered.connect(self.addRow)
            menu.exec(event.globalPos())

    def moveCursor(self, cursorAction, modifiers):
        ''' Override cursor so tab works as expected '''
        assert self.COL_CNT == 4   # If column defs change, need to update this tab-key behavior
        assert self.COL_DESC == 3
        if cursorAction == QtWidgets.QAbstractItemView.MoveNext:
            index = self.currentIndex()
            if index.isValid():
                if (index.column() in [self.COL_DESC] and
                   index.row()+1 < self.model().rowCount(index.parent())):
                    return index.sibling(index.row()+1, 1)
                elif index.column() in [self.COL_NAME, self.COL_EXPR, self.COL_UNIT]:
                    return index.sibling(index.row(), index.column()+1)
                else:
                    return QtCore.QModelIndex()
        elif cursorAction == QtWidgets.QAbstractItemView.MovePrevious:
            index = self.currentIndex()
            if index.isValid():
                if index.column() == self.COL_NAME:
                    if index.row() > 0:
                        return index.sibling(index.row()-1, self.COL_DESC)
                    else:
                        return QtCore.QModelIndex()
                else:
                    return index.sibling(index.row(), index.column()-1)
        return super().moveCursor(cursorAction, modifiers)

    def solvefor(self, fn, name, var, row):
        ''' Algebraically rearrange function on row, solving for var and replacing
            entry in Measurement Model table (after prompting user).

            name = fn(var, ...) --> solve for var

            Args:
                fn (sympy) Expression to solve
                name (string): Name of expression (name = fn)
                var (string): Name of variable to solve for
                row (int): Table row number for this function

            Notes:
                Uses sympy.solve(sympy.Eq(name, fn), var).
        '''
        sname = sympy.Symbol(name)
        svar = sympy.Symbol(var)

        try:
            # NOTE: can take too long for some functions. Could use a timeout.
            solutions = sympy.solve(sympy.Eq(sname, fn), svar)
        except NotImplementedError:  # Some solves crash with this if sympy doesn't know what to do
            solutions = []

        if len(solutions) == 0:
            QtWidgets.QMessageBox.warning(self, 'Suncal', 'No Solution Found!')
            return

        if len(solutions) == 1:
            solution = solutions[0]
        else:
            # Try to pick solution that's reversible
            revsolutions = []
            for s in solutions:
                stest = sympy.solve(sympy.Eq(svar, s), sname)
                if len(stest) > 0:
                    revsolutions.append(s)
            solution = revsolutions[0]

        mbox = QtWidgets.QMessageBox()
        mbox.setWindowTitle('Suncal')
        math = report.Math.from_sympy(sympy.Eq(svar, solution)).svg_b64()
        mbox.setText(f'Set model equation?<br><br><img src="{math}"/>')
        mbox.setInformativeText('Measured Quantity entries may be removed.<br><br>Note: Do not use for reversing '
                                'the calculation to determine required uncertainty of a measured quantity.')
        mbox.setStandardButtons(QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)
        ok = mbox.exec_()
        if ok == QtWidgets.QMessageBox.Yes:
            expr = str(solution)
            if fn is not None:
                # Must use delegate to set so equation renders properly
                self._delegate.setModelData(QtWidgets.QLineEdit(expr),
                                            self.model(),
                                            self.indexFromItem(self.item(row, self.COL_EXPR)))
                self._delegate.setModelData(QtWidgets.QLineEdit(var),
                                            self.model(),
                                            self.indexFromItem(self.item(row, self.COL_NAME)))


class MeasTableWidget(QtWidgets.QTableWidget):
    ''' Table Widget ... '''
    COL_VARNAME = 0
    COL_VALNAME = 1
    COL_VALUE = 2
    COL_UNITS = 3
    COL_DEGF = 4
    COL_DESC = 5
    COL_STDUNC = 6
    COL_PREVIEW = 7
    COL_BTN = 8
    COL_CNT = 9

    COL_DATA = COL_VALUE                   # setData lives in this column
    ROLE_ENTERED = QtCore.Qt.UserRole + 1  # Value as entered by user
    ROLE_CONFIGIDX = ROLE_ENTERED + 1      # Tuple. Index of (variableidx, uncertidx)

    COLOR2 = QtGui.QBrush(QtGui.QColor(246, 246, 246, 255))
    COLOR = {True: gui_common.COLOR_OK,
             False: gui_common.COLOR_INVALID}
    COLORINPT = {True: QtGui.QBrush(QtGui.QColor(246, 246, 246, 255)),
                 False: gui_common.COLOR_INVALID}

    resizerows = QtCore.pyqtSignal()
    changed = QtCore.pyqtSignal(list)

    def __init__(self, projitem):
        super().__init__()
        self.projitem = projitem
        self.clear()
        self.setStyleSheet(TABLESTYLE)
        self._delegate = gui_widgets.LatexDelegate()
        self.setItemDelegateForColumn(self.COL_VARNAME, self._delegate)
        self.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.verticalHeader().hide()
        self.itemActivated.connect(self.editItem)
        self.itemChanged.connect(self.itemchange)
        self.horizontalHeader().sectionResized.connect(self.col_resize)
        self.fixSize()
        self.config = []  # list of inputs, each one is a dict
        self.storedconfig = {}   # Stored configuration, for when variables are removed then put back

    def load_config(self, config):
        ''' Load values from configuration '''
        self.blockSignals(True)
        self.clear()
        self.config = config.get('inputs', [])
        for varidx, variable in enumerate(self.config):
            units = uparser.parse_unit(variable.get('units', ''))
            degf = variable.get('degf', np.inf)

            self.setRowCount(self.rowCount() + 1)
            row = self.rowCount() - 1
            varnameitem = TableItemTex(variable.get('name'), editable=False)
            varnameitem.setBackground(QtGui.QBrush(gui_common.COLOR_UNUSED))
            varvalueitem = gui_widgets.EditableTableItem(f'{variable.get("mean"):.5g}')
            self.setItem(row, self.COL_VARNAME, varnameitem)
            self.setItem(row, self.COL_VALNAME, gui_widgets.ReadOnlyTableItem('Measured'))
            self.setItem(row, self.COL_VALUE, varvalueitem)
            self.item(row, self.COL_DATA).setData(self.ROLE_CONFIGIDX, (varidx, None))

            self.setItem(row, self.COL_UNITS, gui_widgets.EditableTableItem(f'{units:~P}' if units else ''))
            self.setItem(row, self.COL_DESC, gui_widgets.EditableTableItem(variable.get('desc')))
            self.setItem(row, self.COL_PREVIEW, gui_widgets.ReadOnlyTableItem())
            self.setItem(row, self.COL_DEGF,
                         gui_widgets.ReadOnlyTableItem('inf' if not np.isfinite(degf) else f'{degf:.1f}'))
            self.setItem(row, self.COL_STDUNC, gui_widgets.ReadOnlyTableItem('± 1'))

            btn = gui_widgets.TreeButton('+')
            btn.setToolTip('Add Uncertainty Component')
            btn.clicked.connect(lambda x, varidx=varidx: self.add_uncertainty(varidx))
            self.setCellWidget(row, self.COL_BTN, btn)

            # Set colors for the variable row (grayish)
            for col in range(self.COL_CNT):
                with suppress(AttributeError):
                    self.item(row, col).setBackground(QtGui.QBrush(gui_common.COLOR_UNUSED))
            for col in [self.COL_VALUE, self.COL_UNITS, self.COL_DESC]:
                with suppress(AttributeError):
                    self.item(row, col).setBackground(QtGui.QBrush(self.COLOR2))

            # Fill in uncertainties under the variable
            for uncidx, uncert in enumerate(variable.get('uncerts')):
                self.setRowCount(self.rowCount() + 1)
                row = self.rowCount() - 1
                item = QtWidgets.QTableWidgetItem()
                item.setData(self.ROLE_CONFIGIDX, (varidx, uncidx))
                self.setItem(row, self.COL_VALUE, item)
                self.fill_uncertparams(row, varidx, uncidx)

        self.fixSize()
        self.trickle_units()
        self.blockSignals(False)
        self.changed.emit(self.get_config())

    def fill_uncertparams(self, row, varidx, uncidx):
        ''' Fill uncertainty parameters for the given row '''
        # Signals should already be blocked
        config = self.get_uncert_config(varidx, uncidx)
        typeb = Typeb(nominal=self.get_nominal(varidx), **config)

        # Distribution select widget
        cmbdist = gui_widgets.ComboNoWheel()
        cmbdist.addItems(gui_common.settings.getDistributions())
        cmbdist.setCurrentIndex(cmbdist.findText(config.get('dist')))
        if cmbdist.currentText() == '':
            cmbdist.addItem(config.get('dist'))
            cmbdist.setCurrentIndex(cmbdist.count()-1)
        self.setItem(row, self.COL_VALNAME, gui_widgets.ReadOnlyTableItem('Distribution'))
        self.setCellWidget(row, self.COL_VALUE, cmbdist)

        # Other top-row items
        self.setItem(row, self.COL_UNITS, gui_widgets.EditableTableItem(config.get('units')))
        self.setItem(row, self.COL_VARNAME, TableItemTex(config.get('name', '')))
        self.setItem(row, self.COL_DEGF, gui_widgets.EditableTableItem(str(config.get('degf', 'inf'))))
        self.setItem(row, self.COL_DESC, gui_widgets.EditableTableItem(config.get('desc', '')))
        self.setItem(row, self.COL_STDUNC, gui_widgets.ReadOnlyTableItem(
            f'± {typeb.uncertainty:.2g~P}' if typeb.units else f'± {typeb.uncertainty:.2g}'))
        btn = gui_widgets.TreeButton('–')  # endash
        btn.setToolTip('Remove Uncertainty Component')
        btn.clicked.connect(lambda x, varidx=varidx, uncidx=uncidx: self.remove_uncertainty(varidx, uncidx))
        self.setCellWidget(row, self.COL_BTN, btn)

        # Other uncertainty parameters (more rows)
        if config.get('dist') in ['norm', 'normal', 't']:
            newrows = 4
            if 'conf' in config:
                conf = float(config['conf'])
                k = ttable.k_factor(conf, config.get('degf', np.inf))
            else:
                k = float(config.get('k', 1))
                conf = ttable.confidence(k, config.get('degf', np.inf))
            uncstr = config.get('unc', k*float(config.get('std', 1)))  # Could be float or string
            with suppress(ValueError):
                uncstr = f'{uncstr:.5g}'

            self.insertRow(row+1)
            self.setItem(row+1, self.COL_VALNAME, gui_widgets.ReadOnlyTableItem('Confidence'))
            self.setItem(row+1, self.COL_VALUE, gui_widgets.EditableTableItem(f'{float(conf)*100:.2f}%'))
            self.item(row+1, self.COL_DATA).setData(self.ROLE_CONFIGIDX, (varidx, uncidx))
            self.insertRow(row+1)
            self.setItem(row+1, self.COL_VALNAME, gui_widgets.ReadOnlyTableItem('k'))
            self.setItem(row+1, self.COL_VALUE, gui_widgets.EditableTableItem(f'{k:.2f}'))
            self.item(row+1, self.COL_DATA).setData(self.ROLE_CONFIGIDX, (varidx, uncidx))
            self.insertRow(row+1)
            self.setItem(row+1, self.COL_VALNAME, gui_widgets.ReadOnlyTableItem('Uncertainty'))
            self.setItem(row+1, self.COL_VALUE, gui_widgets.EditableTableItem(uncstr))
            self.item(row+1, self.COL_DATA).setData(self.ROLE_CONFIGIDX, (varidx, uncidx))

        elif config.get('dist') == 'histogram':
            newrows = 1  # No more rows to add, just distribution box

        else:
            newrows = len(typeb.distribution.argnames) + 1
            for arg in reversed(sorted(typeb.distribution.argnames)):
                self.insertRow(row+1)
                self.setItem(row+1, self.COL_VALNAME, gui_widgets.ReadOnlyTableItem(arg))
                self.setItem(row+1, self.COL_VALUE, gui_widgets.EditableTableItem(str(typeb.kwargs.get(arg, 1))))
                self.item(row+1, self.COL_DATA).setData(self.ROLE_CONFIGIDX, (varidx, uncidx))

        # Preview Plot
        prev = UncertPreview()
        prev.setFixedSize(self.columnWidth(self.COL_PREVIEW)-1, self.rowHeight(0)*newrows)
        prev.replot(typeb)
        self.setSpan(row, self.COL_PREVIEW, newrows, 1)
        self.setCellWidget(row, self.COL_PREVIEW, prev)
        self.setSpan(row, self.COL_VARNAME, newrows, 1)
        self.setSpan(row, self.COL_UNITS, newrows, 1)
        self.setSpan(row, self.COL_DEGF, newrows, 1)
        self.setSpan(row, self.COL_DESC, newrows, 1)
        self.setSpan(row, self.COL_STDUNC, newrows, 1)
        self.setSpan(row, self.COL_BTN, newrows, 1)
        item = self.item(row, self.COL_VALUE)
        cmbdist.currentIndexChanged.connect(lambda y, item=item: self.change_dist(item))
        self.fixSize()

    def get_config(self):
        ''' Get config dictionary of all inputs '''
        return deepcopy(self.config)

    def get_uncert_config(self, varidx, uncidx):
        return self.config[varidx].get('uncerts')[uncidx]

    def get_var_config(self, varidx):
        return self.config[varidx]

    def get_nominal(self, varidx):
        return unitmgr.make_quantity(self.config[varidx]['mean'], self.config[varidx].get('units', 'dimensionless'))

    def is_valid(self):
        ''' Return True if all entries are valid (not red) '''
        for row in range(self.rowCount()):
            for col in range(self.columnCount()):
                with suppress(AttributeError):
                    if self.item(row, col).background().color() == gui_common.COLOR_INVALID:
                        return False
        return True

    def change_dist(self, item):
        ''' Distribution type in combobox was changed '''
        self.blockSignals(True)
        row = item.row()
        cfgidx = item.data(self.ROLE_CONFIGIDX)
        config = self.get_uncert_config(*cfgidx)
        typeb_old = Typeb(**config)
        distname = self.cellWidget(row, self.COL_VALUE).currentText()   # Dist combo is always first child
        oldrows = 3 if config.get('dist') in ['norm', 'normal', 't'] else len(typeb_old.distribution.argnames)

        # Remove old rows except the first one with dist combobox
        for r in range(oldrows):
            self.removeRow(row+1)

        # Old distribtuion parameters will remain in the dict, but are ignored by Distribution instance
        config['dist'] = distname
        self.fill_uncertparams(row, *cfgidx)
        self.blockSignals(False)
        self.fixSize()
        self.changed.emit(self.get_config())

    def varrow_change(self, item, varidx):
        ''' Variable row was edited. '''
        column = self.column(item)
        config = self.get_var_config(varidx)

        value = item.text()
        valid = True
        if column == self.COL_VALUE:
            try:
                value = np.float64(uparser.callf(value))
            except (ValueError, TypeError):
                valid = False
            else:
                config['mean'] = value
                item.setText(str(value))

        elif column == self.COL_DESC:
            config['desc'] = value

        elif column == self.COL_UNITS:
            try:
                units = uparser.parse_unit(value)
            except ValueError:
                valid = False
            else:
                value = f'{units:~P}'
                config['units'] = str(units)   # The ~P formatted one won't always parse back
                item.setText(value)

                # Ensure all component units are compatible
                for uncert in config.get('uncerts', []):
                    uncunit = unitmgr.parse_units(uncert.get('units', 'dimensionless'))
                    if uncunit.dimensionality != units.dimensionality:
                        uncert['units'] = str(units)
                    if uncert['units'] in ['degree_Celsius', 'degC', 'celsius']:
                        uncert['units'] = 'delta_degC'
                    elif uncert['units'] in ['degree_Fahrenheit', 'degF', 'fahrenheit']:
                        uncert['units'] = 'delta_degF'
        if valid:
            self.changed.emit(self.get_config())
        return valid

    def uncrow_change(self, item, varidx, uncidx):
        ''' Uncertainty row was edited '''
        row = self.row(item)
        column = self.column(item)
        config = self.get_uncert_config(varidx, uncidx)
        valid = True
        value = item.text()

        try:
            floatval = float(value.strip('%'))
        except ValueError:
            floatval = None

        if column == self.COL_DESC:
            config['desc'] = value

        elif column == self.COL_VARNAME:
            # Uses LatexDelegate, so need to get user-entered value from data
            enteredvalue = item.data(self.ROLE_ENTERED)
            config['name'] = enteredvalue

        elif column == self.COL_DEGF:
            valid = floatval is not None
            if valid:
                config['degf'] = floatval

                if config.get('dist') in ['normal', 'norm', 't']:
                    if 'k' in config:
                        k = float(config.get('k'))
                        self.item(row+3, self.COL_VALUE).setText(
                            f'{ttable.confidence(k, floatval)*100:.2f}%')
                    else:
                        conf = float(config.get('conf', .95))
                        self.item(row+2, self.COL_VALUE).setText(
                            f'{ttable.k_factor(conf, floatval):.2f}')

        elif column == self.COL_UNITS:
            valid, unitstr = self.validate_unit(value, self.get_nominal(varidx))
            if valid:
                item.setText(unitstr)
                config['units'] = unitstr

        elif column == self.COL_VALUE:
            paramname = self.item(row, self.COL_VALNAME).text()
            if paramname == 'Uncertainty':
                config['unc'] = value
                config.pop('std', None)

            elif paramname == 'k':
                valid = floatval is not None and floatval > 0
                if valid:
                    config['k'] = floatval
                    config.pop('conf', None)
                    degf = config.get('degf', np.inf)
                    self.item(row+1, self.COL_VALUE).setText(f'{ttable.confidence(floatval, degf)*100:.2f}%')

            elif paramname == 'Confidence':
                valid = floatval is not None and 0 < floatval < 100
                if valid:
                    config['conf'] = floatval/100  # Assume confidence in percent
                    config.pop('k', None)
                    degf = config.get('degf', np.inf)
                    self.item(row-1, self.COL_VALUE).setText(f'{ttable.k_factor(floatval/100, degf):.2f}')

            else:
                config[paramname] = value

        if valid:
            try:
                typeb = Typeb(nominal=self.get_nominal(varidx), **config)
                typeb.distribution.rvs()
            except (AttributeError, ValueError, PintError):
                valid = False

        if valid:
            self.changed.emit(self.get_config())
        return valid

    def itemchange(self, item):
        ''' An item in the table was edited. Validate the input. '''
        row = self.row(item)
        varidx, uncidx = self.item(row, self.COL_DATA).data(self.ROLE_CONFIGIDX)
        self.blockSignals(True)
        if uncidx is None:
            valid = self.varrow_change(item, varidx)
            item.setBackground(self.COLORINPT[valid])
        else:
            valid = self.uncrow_change(item, varidx, uncidx)
            item.setBackground(self.COLOR[valid])

        if valid:
            self.replotall()
            self.trickle_units()
        else:
            self.clearSelection()  # Remove highlight to see the red color

        self.blockSignals(False)
        self.changed.emit(self.get_config())

    def validate_unit(self, unitstr, nominal):
        # 1) convert degC of celsius into delta_degC; degF or fah. into delta_degF
        # 2) uparser.parse_unit
        # 3) check dimensionality with nominal
        if unitstr in ['degC', 'celsius']:
            unitstr = 'delta_degC'
        elif unitstr in ['degF', 'fahrenheit']:
            unitstr = 'delta_degF'
        try:
            units = uparser.parse_unit(unitstr)
        except ValueError:
            valid = False
        else:
            if unitmgr.has_units(nominal):
                valid = (nominal.units.dimensionality == units.dimensionality)
            unitstr = str(units)
        return valid, unitstr

    def trickle_units(self):
        ''' Trickle up/down changes in units and standard uncertainty '''
        self.blockSignals(True)
        for row in range(self.rowCount()):
            varidx, uncidx = self.item(row, self.COL_DATA).data(self.ROLE_CONFIGIDX)
            nominal = self.get_nominal(varidx)
            if uncidx is None:  # This row is a Variable
                config = self.get_var_config(varidx)
                randvar = RandomVariable(nominal)
                for uncert in config.get('uncerts', []):
                    randvar.typeb(**uncert)

                degf = randvar.degrees_freedom
                if unitmgr.has_units(randvar.uncertainty):
                    uncstr = f'± {randvar.uncertainty:.2g~P}'
                else:
                    uncstr = f'± {randvar.uncertainty:.2g}'
                self.item(row, self.COL_STDUNC).setText(uncstr)
                self.item(row, self.COL_DEGF).setText('inf' if not np.isfinite(degf) else f'{degf:.1f}')

            else:  # Uncertainty row
                config = self.get_uncert_config(varidx, uncidx)
                if unitmgr.has_units(nominal) and 'units' not in config:
                    config['units'] = str(nominal.units)
                typeb = Typeb(nominal=nominal, **config)
                degf = typeb.degf
                self.setItem(
                    row, self.COL_DEGF, gui_widgets.EditableTableItem(f'{degf:.1f}'))
                self.setItem(
                    row, self.COL_UNITS, gui_widgets.EditableTableItem(f'{typeb.units:~P}' if typeb.units else ''))
                if self.item(row, self.COL_STDUNC):  # Only applies to first row unc uncertainty component
                    self.item(row, self.COL_STDUNC).setText(
                        f'± {typeb.uncertainty:.2g~P}' if typeb.units else f'± {typeb.uncertainty:.2g}')
        self.blockSignals(False)

    def replotall(self):
        ''' Replot all preview plots '''
        done = []
        for row in range(self.rowCount()):
            prev = self.cellWidget(row, self.COL_PREVIEW)
            if prev is not None and prev not in done:
                varidx, uncidx = self.item(row, self.COL_DATA).data(self.ROLE_CONFIGIDX)
                config = self.get_uncert_config(varidx, uncidx)
                nominal = self.get_nominal(varidx)
                prev.replot(Typeb(nominal=nominal, **config))
                done.append(prev)

    def add_uncertainty(self, varidx, uncidx=0):
        ''' Add a new uncertianty component to the table '''
        varcfg = self.get_var_config(varidx)
        varname = varcfg.get('name')
        i = 1
        name = f'u{i}({varname})'
        while name in [u['name'] for u in varcfg.get('uncerts', [])]:
            i += 1
            name = f'u{i}({varname})'
        unccfg = {'name': name, 'dist': 'normal', 'unc': 1, 'k': 2}

        varcfg['uncerts'].insert(uncidx, unccfg)
        self.load_config({'inputs': self.config})
        self.fixSize()
        self.changed.emit(self.get_config())

    def remove_uncertainty(self, varidx, uncidx):
        ''' Remove selected uncertainty component
        '''
        varcfg = self.get_var_config(varidx)
        varcfg['uncerts'].pop(uncidx)
        self.load_config({'inputs': self.config})
        self.fixSize()
        self.changed.emit(self.get_config())

    def add_variables(self, varnames):
        ''' Add or remove variables from the table when the functions change '''
        # Add new ones
        for varname in varnames:
            existingvars = [v['name'] for v in self.config]
            if varname not in existingvars:
                if varname not in self.storedconfig:
                    self.config.append({'name': varname,
                                        'mean': 1,
                                        'uncerts': [{
                                            'name': f'u({varname})',
                                            'dist': 'normal',
                                            'unc': 1,
                                            'k': 2}]})
                else:
                    self.config.append(self.storedconfig[varname])

        # Remove (but cache) variables that are no longer used
        existingvars = [v['name'] for v in self.config]
        for oldvar in existingvars:
            if oldvar not in varnames:
                idx = [v['name'] for v in self.config].index(oldvar)
                self.storedconfig[oldvar] = self.config.pop(idx)
        self.load_config({'inputs': self.config})
        self.fixSize()

    def clear(self):
        ''' Clear the table '''
        super().clear()
        self.setRowCount(0)
        self.setColumnCount(self.COL_CNT)
        self.setColumnWidth(self.COL_VARNAME, 70)
        self.setColumnWidth(self.COL_VALNAME, 100)
        self.setColumnWidth(self.COL_VALUE, 100)
        self.setColumnWidth(self.COL_UNITS, 80)
        self.setColumnWidth(self.COL_DEGF, 80)
        self.setColumnWidth(self.COL_DESC, 290)
        self.setColumnWidth(self.COL_STDUNC, 90)
        self.setColumnWidth(self.COL_PREVIEW, 282)
        self.setColumnWidth(self.COL_BTN, 18)
        self.setHorizontalHeaderLabels(['Variable', 'Parameter', 'Value', 'Units', 'Degrees\nFreedom',
                                        'Description', 'Standard\nUncertainty', 'Preview', ''])
        self.fixSize()

    def fixSize(self):
        ''' Adjust the size of the table to fit the number of rows '''
        height = max(self.horizontalHeader().height()+20,
                     self.verticalHeader().length() + self.horizontalHeader().height())
        self.setFixedHeight(height)
        self.resizerows.emit()

    def col_resize(self, index, oldsize, newsize):
        ''' Column was resized. Resize the preview plot to fit '''
        if index == self.COL_PREVIEW:
            previews = [self.cellWidget(row, self.COL_PREVIEW) for row in range(self.rowCount())]
            previews = set(filter(None, previews))
            for prev in previews:
                prev.setFixedSize(newsize-1, prev.size().height())

    def contextMenuEvent(self, event):
        ''' Show right-click context menu '''
        item = self.itemAt(event.pos())
        menu = QtWidgets.QMenu()

        if item:
            row = self.row(item)
            varidx = uncidx = None
            with suppress(AttributeError):
                varidx, uncidx = self.item(row, self.COL_DATA).data(self.ROLE_CONFIGIDX)

            if uncidx is None:  # Variable row
                add = menu.addAction('Add uncertainty component')
                add.triggered.connect(lambda x, varidx=varidx: self.add_uncertainty(varidx))

            else:  # Uncert Row
                add = menu.addAction('Add uncertainty component')
                add.triggered.connect(lambda x, varidx=varidx, uncidx=uncidx: self.add_uncertainty(varidx, uncidx))
                rem = menu.addAction('Remove uncertainty component')
                rem.triggered.connect(lambda x, varidx=varidx, uncidx=uncidx: self.remove_uncertainty(varidx, uncidx))
                imp = menu.addAction('Import distribution from...')
                imp.triggered.connect(self.import_dist)
                hlp = menu.addAction('Distribution help...')

                def helppopup():
                    ''' Show distribution description '''
                    config = self.get_uncert_config(varidx, uncidx)
                    typeb = Typeb(**config)
                    dlg = gui_widgets.PopupHelp(typeb.distribution.helpstr())
                    dlg.exec_()

                hlp.triggered.connect(helppopup)
            menu.exec(event.globalPos())

    def import_dist(self):
        ''' Import a distribution from somewhere else in the project '''
        dlg = page_dataimport.DistributionSelectWidget(singlecol=True, project=self.projitem.project)
        ok = dlg.exec_()
        if ok:
            distargs = dlg.get_dist()
            distname = distargs.get('dist', 'normal')
            nominal = distargs.pop('expected', distargs.pop('median', distargs.pop('mean', 0)))
            if distname == 'histogram':
                distargs['histogram'] = (distargs['histogram'][0], np.array(distargs['histogram'][1])-nominal)

            newdist = Typeb(nominal=nominal, **distargs)  # Change to RandomVariable.value eventually

            selitem = self.currentItem()
            row = selitem.row()
            varidx, uncidx = self.item(row, self.COL_DATA).data(self.ROLE_CONFIGIDX)
            unccfg = self.get_uncert_config(varidx, uncidx)
            distname = unccfg.get('dist', 'normal')

            # Remove old rows if different distribution
            self.blockSignals(True)
            oldrows = 3 if distname in ['normal', 't'] else len(newdist.distribution.argnames)
            for r in range(oldrows):
                self.removeRow(row+1)

            self.config[varidx]['mean'] = nominal
            self.config[varidx]['uncerts'][uncidx] = distargs
            self.load_config({'inputs': self.config})
            self.trickle_units()


class UncertPreview(QtWidgets.QWidget):
    ''' Plot of uncertainty component PDF '''
    def __init__(self):
        super().__init__()
        self.setFixedSize(282, 120)
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.component = None

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.canvas)
        layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)

    def replot(self, typeb=None):  # comp is Typeb class
        ''' Update the plot with PDF of InptUncert comp '''
        if typeb is not None:
            self.component = typeb

        if self.component is not None:
            with plt.style.context({'font.size': 8}):
                self.figure.clf()
                ax = self.figure.add_subplot(1, 1, 1)
                x, y = self.component.pdf()
                ax.plot(x, y)
                ax.yaxis.set_ticks([])
                height = self.size().height()
                self.figure.subplots_adjust(left=0, right=1, bottom=20/height, top=.99)
            self.canvas.draw_idle()


class CorrelationTableWidget(QtWidgets.QTableWidget):
    ''' Widget for setting correlations between inputs '''
    resizerows = QtCore.pyqtSignal()

    def __init__(self):
        super().__init__()
        self.varnames = []
        self.clear()
        self.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.verticalHeader().hide()
        self.setSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum)
        self.setStyleSheet(TABLESTYLE)
        self.fixSize()

    def load_config(self, config):
        ''' Load configuration '''
        self.config = config
        self.setVarNames([v['name'] for v in self.config.get('inputs', [])])
        for corcfg in config.get('correlations', []):
            self.addRow(corcfg.get('var1'), corcfg.get('var2'), float(corcfg.get('cor')))

    def get_config(self):
        ''' Get correlation config list '''
        config = []
        for row in range(self.rowCount()):
            var1, var2, cor = self.getRow(row)
            if cor is not None:
                config.append({'var1': var1, 'var2': var2, 'cor': cor})
        return config

    def clear(self):
        ''' Clear and reset the table '''
        super().clear()
        self.setColumnCount(3)
        self.setRowCount(0)
        self.setHorizontalHeaderLabels(['Variable 1', 'Variable 2', 'Correlation Coefficient'])
        self.setColumnWidth(0, 100)
        self.setColumnWidth(1, 100)
        self.setColumnWidth(2, 200)
        self.varnames = []
        self.fixSize()

    def fixSize(self):
        ''' Adjust the size of the table to fit the number of rows '''
        height = max(self.horizontalHeader().height()+20,
                     self.verticalHeader().length() + self.horizontalHeader().height())
        self.setFixedHeight(height)
        self.resizerows.emit()

    def setVarNames(self, names):
        ''' Set names of available variables '''
        self.blockSignals(True)
        self.varnames = names
        for row in range(self.rowCount()):
            prev1 = self.cellWidget(row, 0).currentText()
            prev2 = self.cellWidget(row, 1).currentText()
            self.cellWidget(row, 0).clear()
            self.cellWidget(row, 1).clear()
            self.cellWidget(row, 0).addItems(names)
            self.cellWidget(row, 1).addItems(names)
            self.cellWidget(row, 0).setCurrentIndex(self.cellWidget(row, 0).findText(prev1))
            self.cellWidget(row, 1).setCurrentIndex(self.cellWidget(row, 1).findText(prev2))
        self.blockSignals(False)

    def remRow(self):
        ''' Remove selected row from correlation table '''
        self.removeRow(self.currentRow())
        self.cellChanged.emit(0, 0)
        self.fixSize()

    def addRow(self, name1=None, name2=None, corr=None):
        ''' Add a row to the table. '''
        row = self.rowCount()
        self.blockSignals(True)
        self.insertRow(row)
        v1 = QtWidgets.QComboBox()
        v1.addItems(self.varnames)
        v2 = QtWidgets.QComboBox()
        v2.addItems(self.varnames)
        self.setCellWidget(row, 0, v1)
        self.setCellWidget(row, 1, v2)
        val = gui_widgets.EditableTableItem('0')
        self.setItem(row, 2, val)
        if name1 is not None:
            self.setRow(row, name1, name2, corr)
        self.blockSignals(False)
        self.fixSize()

    def setRow(self, row, v1, v2, cor):
        ''' Set values to an existing row '''
        self.cellWidget(row, 0).setCurrentIndex(self.cellWidget(row, 0).findText(v1))
        self.cellWidget(row, 1).setCurrentIndex(self.cellWidget(row, 1).findText(v2))
        self.item(row, 2).setText(f'{cor:.4f}')

    def getRow(self, row):
        ''' Get values for one row in the table. Returns (input1, input2, corrleation) '''
        v1 = self.cellWidget(row, 0).currentText()
        v2 = self.cellWidget(row, 1).currentText()
        try:
            val = float(self.item(row, 2).text())
        except ValueError:
            return v1, v2, None
        val = max(val, -1)
        val = min(val, 1)
        return v1, v2, val

    def contextMenuEvent(self, event):
        ''' Right-click menu '''
        row = self.currentRow()
        menu = QtWidgets.QMenu()
        add = menu.addAction('Add correlation coefficient')
        add.triggered.connect(self.addRow)
        if row > -1:
            rem = menu.addAction('Remove correlation coefficient')
            rem.triggered.connect(self.remRow)
        menu.exec(event.globalPos())


class SettingsWidget(QtWidgets.QTableWidget):
    ''' Widget for Monte-Carlo settings '''
    COL_NAME = 0
    COL_VALUE = 1
    ROW_SAMPLES = 0
    ROW_SEED = 1
    ROW_SYMBOLIC = 2
    ROW_CNT = 3

    changed = QtCore.pyqtSignal(str, object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setColumnCount(2)
        self.setRowCount(self.ROW_CNT)
        self.setHorizontalHeaderLabels(['Setting', 'Value'])
        self.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.setStyleSheet(TABLESTYLE)
        self.verticalHeader().hide()
        self.setSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum)

        self.setItem(self.ROW_SAMPLES, self.COL_NAME, gui_widgets.ReadOnlyTableItem('Monte Carlo Samples'))
        self.setItem(self.ROW_SAMPLES, self.COL_VALUE,
                     gui_widgets.EditableTableItem(str(gui_common.settings.getSamples())))
        self.setItem(self.ROW_SEED, self.COL_NAME, gui_widgets.ReadOnlyTableItem('Random Seed'))
        self.setItem(self.ROW_SEED, self.COL_VALUE,
                     gui_widgets.EditableTableItem(str(gui_common.settings.getRandomSeed())))
        self.setItem(self.ROW_SYMBOLIC, self.COL_NAME, gui_widgets.ReadOnlyTableItem('Symbolic GUM Solution Only'))
        chkbox = QtWidgets.QCheckBox()
        chkbox.stateChanged.connect(lambda x: self.valuechange(self.item(self.ROW_SYMBOLIC, self.COL_NAME)))
        self.setCellWidget(self.ROW_SYMBOLIC, self.COL_VALUE, chkbox)

        self.resizeColumnsToContents()
        self.itemChanged.connect(self.valuechange)

        height = self.verticalHeader().length() + self.horizontalHeader().height()
        self.setFixedHeight(height)

    def get_config(self):
        ''' Get configuration from settings '''
        config = {}
        try:
            samples = int(float(self.item(self.ROW_SAMPLES, self.COL_VALUE).text()))
        except (TypeError, ValueError, OverflowError):
            samples = gui_common.settings.getSamples()
        config['samples'] = samples

        try:
            seed = int(float(self.item(self.ROW_SEED, self.COL_VALUE).text()))
        except (TypeError, ValueError, OverflowError):
            seed = gui_common.settings.getRandomSeed()
        config['seed'] = seed
        return config

    def valuechange(self, item):
        ''' A setting value was changed. '''
        row = self.row(item)
        name = self.item(row, self.COL_NAME).text()
        if row == self.ROW_SYMBOLIC:
            value = self.cellWidget(self.ROW_SYMBOLIC, self.COL_VALUE).isChecked()
        else:
            value = item.text()
        self.changed.emit(name, value)

    def setvalue(self, name, value):
        ''' Change the value of a setting '''
        item = self.findItems(name, QtCore.Qt.MatchExactly)
        assert len(item) == 1
        row = item[0].row()
        if row == self.ROW_SYMBOLIC:
            self.cellWidget(self.ROW_SYMBOLIC, self.COL_VALUE).setChecked(value)
        else:
            self.item(row, self.COL_VALUE).setText(str(value))


class PageInput(QtWidgets.QWidget):
    ''' Page for setting up input parameters '''
    calculate = QtCore.pyqtSignal()

    def __init__(self, projitem=None, parent=None):
        super().__init__(parent)
        self.projitem = projitem
        self.symbolicmode = False
        self.funclist = FunctionTableWidget()
        self.meastable = MeasTableWidget(self.projitem)
        self.corrtable = CorrelationTableWidget()
        self.btnCalc = QtWidgets.QPushButton('Calculate')
        self.description = QtWidgets.QPlainTextEdit()
        font = self.description.font()
        font.setPointSize(10)
        self.description.setFont(font)
        self.settings = SettingsWidget()

        self.panel = gui_widgets.WidgetPanel()
        self.panel.setStyleSheet(TREESTYLE)
        font = QtGui.QFont('Arial', 14)
        self.panel.setFont(font)
        self.panel.setAnimated(True)
        self.btnCalc.clicked.connect(self.calculate)

        _, funcbuttons = self.panel.add_widget('Measurement Model', self.funclist, buttons=True)
        funcbuttons.btnplus.setToolTip('Add measurement function')
        funcbuttons.btnminus.setToolTip('Remove measurement function')
        self.panel.add_widget('Measured Values and Uncertainties', self.meastable)
        _, corrbuttons = self.panel.add_widget('Correlations', self.corrtable, buttons=True)
        corrbuttons.btnplus.setToolTip('Add correlation coefficient')
        corrbuttons.btnminus.setToolTip('Remove correlation coefficient')
        self.panel.add_widget('Notes', self.description)
        self.panel.add_widget('Settings', self.settings)
        self.panel.expand('Measurement Model')
        self.panel.expand('Measured Values and Uncertainties')

        calclayout = QtWidgets.QHBoxLayout()
        calclayout.addStretch()
        calclayout.addWidget(self.btnCalc)
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.panel)
        layout.addLayout(calclayout)
        self.setLayout(layout)
        self.load_config(self.projitem.get_config())

        funcbuttons.plusclicked.connect(self.funclist.addRow)
        funcbuttons.minusclicked.connect(self.funclist.remRow)
        corrbuttons.plusclicked.connect(self.corrtable.addRow)
        corrbuttons.minusclicked.connect(self.corrtable.remRow)
        self.funclist.resizerows.connect(self.panel.fixSize)
        self.funclist.funcchanged.connect(self.functionchanged)
        self.meastable.resizerows.connect(self.panel.fixSize)
        self.corrtable.resizerows.connect(self.panel.fixSize)
        self.settings.changed.connect(self.settingchanged)
        self.panel.fixSize()

    def load_config(self, config):
        ''' Set the widgets to match the config dictionary '''
        self.blockSignals(True)
        self.funclist.load_config(config)
        self.meastable.load_config(config)
        self.corrtable.load_config(config)
        if 'correlations' in config:
            self.panel.expand('Correlations')
        self.settings.setvalue('Monte Carlo Samples', config.get('samples', gui_common.settings.getSamples()))
        self.settings.setvalue('Random Seed', config.get('seed', gui_common.settings.getRandomSeed()))
        self.description.setPlainText(config.get('description', ''))
        self.blockSignals(False)

        if 'description' in config:
            self.panel.expand('Notes')

    def functionchanged(self, config):
        ''' Model functions were changed. Extract variables and update variable table '''
        # config is just function list
        exprs = []
        for f in config:
            if f['expr'] and f['name']:
                exprs.append(f"{f['name']}={f['expr']}")
        model = Model(*exprs)
        varnames = model.varnames
        self.meastable.add_variables(varnames)
        self.corrtable.setVarNames(varnames)

    def is_valid(self):
        ''' Check if all inputs are valid '''
        return self.funclist.is_valid() and self.meastable.is_valid()

    def settingchanged(self, name, value):
        ''' A setting was changed in Settings table '''
        if name == 'Symbolic GUM Solution Only':
            self.symbolicmode = value
            self.meastable.setVisible(not value)
            self.panel.hide('Measured Values and Uncertainties', value)

    def get_config(self):
        ''' Convert input parameters into a Model config dictionary '''
        config = {}
        config['name'] = 'uncertainty'
        config['mode'] = 'uncertainty'
        config['functions'] = self.funclist.get_config()
        config['inputs'] = self.meastable.get_config()
        config['correlations'] = self.corrtable.get_config()
        config['description'] = self.description.toPlainText()
        config.update(self.settings.get_config())  # samples and seed
        return config
