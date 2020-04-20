''' Page for propagating uncertainty calculations '''

import re
from contextlib import suppress
import numpy as np
import sympy
from pint import DimensionalityError, UndefinedUnitError, OffsetUnitCalculusError
from PyQt5 import QtWidgets, QtGui, QtCore
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

from .. import uncertainty
from .. import uparser
from .. import report
from .. import plotting
from .. import distributions
from .. import ttable
from . import gui_common
from . import gui_widgets
from . import page_dataimport


TREESTYLE = """
QTreeView {
    show-decoration-selected: 1;
    font-size: 16px;
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
    ''' TableWidgetItem formatted to display Math expression '''
    def __init__(self, expr=''):
        super().__init__()
        self.setFlags(self.flags() | QtCore.Qt.ItemIsEditable)
        self.setExpr(expr)

    def setExpr(self, expr):
        # Remove display text, replace with rendered math
        px = QtGui.QPixmap()
        px.loadFromData(report.Math(expr).svg_buf().read())
        self.setData(QtCore.Qt.DecorationRole, px)
        self.setData(gui_widgets.ROLE_ORIGDATA, expr)


class FunctionTableWidget(QtWidgets.QTableWidget):
    ''' Table for defining measurement model functions

        Signals
        -------
        funcchanged: One of the functions was modified
        funcremoved: A function was removed
        orderchange: Functions were reordered via drag/drop
        resizerows: Emitted when the number of rows in the table changes
    '''
    funcchanged = QtCore.pyqtSignal(int, dict)
    funcremoved = QtCore.pyqtSignal(int)
    orderchange = QtCore.pyqtSignal()
    resizerows = QtCore.pyqtSignal()
    COL_NAME = 0
    COL_EXPR = 1
    COL_UNIT = 2
    COL_DESC = 3
    COL_REPT = 4
    COL_CNT = 5

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setColumnCount(5)
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
        height = max(self.horizontalHeader().height()+20, self.verticalHeader().length() + self.horizontalHeader().height())
        self.setFixedHeight(height)
        self.resizerows.emit()

    def clear(self):
        ''' Clear everything in the table. Overrides TableWidget clear. '''
        self.setRowCount(0)
        self.setHorizontalHeaderLabels(['Name', 'Expression',  'Units', 'Description', 'Report?'])
        self.resizeColumnsToContents()
        # NOTE: with window width 1200, sum of columnwidths 1110 fills the tree
        self.setColumnWidth(self.COL_NAME, 100)
        self.setColumnWidth(self.COL_EXPR, 400)
        self.setColumnWidth(self.COL_UNIT, 100)
        self.setColumnWidth(self.COL_DESC, 460)
        self.setColumnWidth(self.COL_REPT, 50)
        self.fixSize()

    def setFunclist(self, funclist):
        ''' Set a list of functions '''
        self.blockSignals(True)
        self.clear()
        self.blockSignals(False)
        for row, f in enumerate(funclist):
            self.addRow()
            # Let signals populate the other tables
            self.setItem(row, self.COL_NAME, TableItemTex(f.name))
            self.setItem(row, self.COL_EXPR, TableItemTex(str(f.function)))
            self.setItem(row, self.COL_UNIT, gui_widgets.EditableTableItem(f.outunits))
            self.setItem(row, self.COL_DESC, gui_widgets.EditableTableItem(f.desc))
            chk = QtWidgets.QCheckBox()
            chk.setCheckState(QtCore.Qt.Checked)
            chk.stateChanged.connect(lambda x, row=row, col=self.COL_REPT: self.itemEdit(row, col))
            self.setCellWidget(row, self.COL_REPT, chk)
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
        chk = QtWidgets.QCheckBox()
        chk.setCheckState(QtCore.Qt.Checked)
        chk.stateChanged.connect(lambda x, row=rows, col=self.COL_REPT: self.itemEdit(row, col))
        self.setCellWidget(rows, self.COL_REPT, chk)
        self.blockSignals(False)
        self.fixSize()

    def remRow(self):
        ''' Remove row at index from function table '''
        idx = self.selectedItems()
        if len(idx) > 0:
            idx = idx[0].row()
            self.removeRow(idx)
            self.funcremoved.emit(idx)
        self.fixSize()

    def itemText(self, row, col):
        ''' Get text of item, if item exists. '''
        if self.item(row, col) is not None:
            if col in [self.COL_NAME, self.COL_EXPR]:
                return self.item.data(gui_widgets.ROLE_ORIGDATA)
            else:
                return self.item(row, col).text()
        else:
            return ''

    def itemEdit(self, row, col):
        ''' A cell was changed. '''
        self.blockSignals(True)
        ok = False
        name = self.item(row, self.COL_NAME).data(gui_widgets.ROLE_ORIGDATA)
        expr = self.item(row, self.COL_EXPR).data(gui_widgets.ROLE_ORIGDATA)
        unit = self.item(row, self.COL_UNIT).text()
        if expr is None:
            expr = self.item(row, self.COL_EXPR).text()

        # Check Expression
        try:
            fn = uparser.parse_math(expr, name=name)
        except ValueError:
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
                self.item(row, self.COL_UNIT).setText(format(unit, '~P'))
                unit = str(unit)
        else:
            unit = None

        self.blockSignals(False)
        if ok:
            func = {'name': name,
                    'desc': self.itemText(row, self.COL_DESC),
                    'unit': unit,
                    'expr': expr}
            if self.cellWidget(row, self.COL_REPT):
                func['report'] = self.cellWidget(row, self.COL_REPT).checkState() == QtCore.Qt.Checked
            self.funcchanged.emit(row, func)

    # Drag Drop based on: https://stackoverflow.com/a/43789304
    def drop_on(self, event):
        ''' Get row to drop on '''
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
            rows_to_move = [[QtWidgets.QTableWidgetItem(self.item(row_index, column_index)) for column_index in range(self.columnCount())]
                            for row_index in rows]
            chkstate = [self.cellWidget(row_index, self.COL_REPT).checkState() for row_index in rows]
            for row_index in reversed(rows):
                self.removeRow(row_index)
                if row_index < drop_row:
                    drop_row -= 1

            for row_index, data in enumerate(rows_to_move):
                c = chkstate[row_index]
                row_index += drop_row
                self.insertRow(row_index)
                for column_index, column_data in enumerate(data):
                    self.setItem(row_index, column_index, column_data)
                chk = QtWidgets.QCheckBox()
                chk.setCheckState(c)
                chk.stateChanged.connect(lambda x, row=row_index, col=self.COL_REPT: self.itemEdit(row, col))
                self.setCellWidget(row_index, self.COL_REPT, chk)

            event.accept()
            for row_index in range(len(rows_to_move)):
                self.item(drop_row + row_index, 0).setSelected(True)
                self.item(drop_row + row_index, 1).setSelected(True)

            # Remove blank rows to maintain consistency with uncCalc.functions list
            for row_idx in reversed(range(self.rowCount())):
                if (self.item(row_idx, self.COL_EXPR) is None or self.item(row_idx, self.COL_EXPR).data(gui_widgets.ROLE_ORIGDATA) == ''):
                    self.removeRow(row_idx)

        super().dropEvent(event)
        self.blockSignals(False)
        self.orderchange.emit()

    def is_below(self, pos, index):
        ''' Check if index is below position. Used for drag/drop '''
        rect = self.visualRect(index)
        margin = 2
        if pos.y() - rect.top() < margin:
            return False
        elif rect.bottom() - pos.y() < margin:
            return True
        return rect.contains(pos, True) and not (int(self.model().flags(index)) & QtCore.Qt.ItemIsDropEnabled) and pos.y() >= rect.center().y()

    def isValid(self):
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
            name = self.item(row, self.COL_NAME).data(gui_widgets.ROLE_ORIGDATA)
            expr = self.item(row, self.COL_EXPR).data(gui_widgets.ROLE_ORIGDATA)
            if expr is None:
                expr = self.item(row, self.COL_EXPR).text()
            try:
                fn = uparser.parse_math(expr, name=name)
            except ValueError:
                return

            menu = QtWidgets.QMenu()
            add = menu.addAction('Add model equation')
            rem = menu.addAction('Remove model equation')
            add.triggered.connect(self.addRow)
            rem.triggered.connect(self.remRow)

            solve = menu.addMenu('Solve for')
            for symbol in fn.free_symbols:
                act = solve.addAction(str(symbol))
                act.triggered.connect(lambda x, fn=fn, name=name, var=str(symbol), row=row: self.solvefor(fn, name, var, row))
            menu.exec(event.globalPos())

        else:
            menu = QtWidgets.QMenu()
            add = menu.addAction('Add model equation')
            add.triggered.connect(self.addRow)
            menu.exec(event.globalPos())

    def moveCursor(self, cursorAction, modifiers):
        ''' Override cursor so tab works as expected '''
        assert self.COL_CNT == 5   # If column defs change, need to update this tab-key behavior
        assert self.COL_DESC == 3
        if cursorAction == QtWidgets.QAbstractItemView.MoveNext:
            index = self.currentIndex()
            if index.isValid():
                if index.column() in [self.COL_DESC, self.COL_REPT] and index.row()+1 < self.model().rowCount(index.parent()):
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

            Parameters
            ----------
            fn: sympy expression
                Expression to solve
            name: string
                Name of expression (name = fn)
            var: string
                Name of variable to solve for
            row: int
                Table row number for this function

            Notes
            -----
            Uses sympy.solve(sympy.Eq(name, fn), var).
        '''
        sname = sympy.Symbol(name)
        svar = sympy.Symbol(var)

        try:
            solutions = sympy.solve(sympy.Eq(sname, fn), svar)   # NOTE: can take too long for some functions. Need timeout.
        except NotImplementedError:  # Some solves crash with this if sympy doesn't know what to do
            solutions = []

        if len(solutions) == 0:
            QtWidgets.QMessageBox.warning(self, 'Uncertainty Calculator', 'No Solution Found!')
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
        mbox.setWindowTitle('Uncertainty Calculator')
        math = report.Math.from_sympy(sympy.Eq(svar, solution)).svg_b64()
        mbox.setText('Set model equation?' + '<br><br>' + '<img src="{}"/>'.format(math))
        mbox.setInformativeText('Measured Quantity entries may be removed.<br><br>Note: Do not use for reversing the calculation to determine required uncertainty of a measured quantity.')
        mbox.setStandardButtons(QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)
        ok = mbox.exec_()
        if ok == QtWidgets.QMessageBox.Yes:
            expr = str(solution)
            if fn is not None:
                # Must use delegate to set so equation renders properly
                self._delegate.setModelData(QtWidgets.QLineEdit(expr), self.model(), self.indexFromItem(self.item(row, self.COL_EXPR)))
                self._delegate.setModelData(QtWidgets.QLineEdit(var), self.model(), self.indexFromItem(self.item(row, self.COL_NAME)))


class MeasTableWidget(QtWidgets.QTableWidget):
    ''' Tree Widget for editing input nominal values '''
    COL_VARNAME = 0
    COL_VALUE = 1
    COL_UNITS = 2
    COL_DIST = 3
    COL_PARAM1 = 4
    COL_PARAM2 = 5
    COL_PARAM3 = 6
    COL_PARAM4 = 7
    COL_DEGF = 8
    COL_DESC = 9
    COL_STDUNC = 10
    COL_CNT = 11

    updated = QtCore.pyqtSignal(object)  # Emit when uncertainty changes, to signal replot
    resizerows = QtCore.pyqtSignal()

    def __init__(self, unccalc):
        super().__init__()
        self.unccalc = unccalc
        self.clear()
        self.setEditTriggers(QtWidgets.QTableWidget.NoEditTriggers)  # Only start editing manually

        self.inputlist = []
        self.setStyleSheet(TABLESTYLE)
        self._delegate = gui_widgets.LatexDelegate()
        self.setItemDelegateForColumn(self.COL_VARNAME, self._delegate)
        self.setItemDelegateForColumn(self.COL_VALUE, self._delegate)
        self.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.verticalHeader().hide()
        self.setSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum)

        self.itemActivated.connect(self.checkEdit)
        self.itemChanged.connect(self.itemchange)
        self.currentCellChanged.connect(self.rowchange)
        self.fixSize()

    def clear(self):
        ''' Clear the table '''
        super().clear()
        self.setRowCount(0)
        self.setColumnCount(self.COL_CNT)
        self.setColumnWidth(self.COL_VARNAME, 70)
        self.setColumnWidth(self.COL_VALUE, 80)
        self.setColumnWidth(self.COL_UNITS, 80)
        self.setColumnWidth(self.COL_DIST, 100)
        self.setColumnWidth(self.COL_PARAM1, 80)
        self.setColumnWidth(self.COL_PARAM2, 80)
        self.setColumnWidth(self.COL_PARAM3, 80)
        self.setColumnWidth(self.COL_PARAM4, 80)
        self.setColumnWidth(self.COL_DEGF, 80)
        self.setColumnWidth(self.COL_DESC, 290)
        self.setColumnWidth(self.COL_STDUNC, 90)
        self.setHorizontalHeaderLabels(['Variable', 'Value', 'Units', 'Distribution', 'Param 1', 'Param 2', 'Param 3', 'Param 4', 'Degrees\nFreedom', 'Description', 'Standard\nUncertainty'])
        self.fixSize()

    def fixSize(self):
        ''' Adjust the size of the table to fit the number of rows '''
        height = max(self.horizontalHeader().height()+20, self.verticalHeader().length() + self.horizontalHeader().height())
        self.setFixedHeight(height)
        self.resizerows.emit()

    def checkEdit(self, item):
        ''' Start editor if this item/column is editable '''
        column = item.column()
        row = item.row()
        uncert = None
        with suppress(AttributeError):
            uncert = self.item(row, self.COL_VALUE).data(gui_widgets.ROLE_UNCERT)

        if ((uncert is None and column in [self.COL_VALUE, self.COL_UNITS, self.COL_DESC]) or
            (uncert is not None and column not in [self.COL_VARNAME, self.COL_STDUNC])):
            self.editItem(item)  # Variable row

    def filltable(self, inputlist=None):
        ''' Fill the table with inputs defined in inputlist, or refresh using existing inputlist '''
        self.blockSignals(True)
        if inputlist is not None:
            self.inputlist = inputlist
        self.clear()
        COLOR2 = QtGui.QBrush(QtGui.QColor(246, 246, 246, 255))
        for inpt in self.inputlist:
            self.setRowCount(self.rowCount() + 1)
            row = self.rowCount() - 1
            inptitem = TableItemTex(inpt.name)
            inptitem.setBackground(QtGui.QBrush(gui_common.COLOR_UNUSED))
            valueitem = TableItemTex(format(inpt.nom, '.5g'))
            valueitem.setData(gui_widgets.ROLE_VARIABLE, inpt)  # Data is always in COL_VALUE
            self.setItem(row, self.COL_VARNAME, inptitem)
            self.setItem(row, self.COL_VALUE, valueitem)
            self.setItem(row, self.COL_UNITS, gui_widgets.EditableTableItem(format(inpt.units, '~P')))
            self.setItem(row, self.COL_DESC, gui_widgets.EditableTableItem(inpt.desc))
            self.setItem(row, self.COL_STDUNC, gui_widgets.ReadOnlyTableItem('± {:.2g~P}'.format(inpt.stdunc())))
            degf = inpt.degf()
            self.setItem(row, self.COL_DEGF, gui_widgets.ReadOnlyTableItem('inf' if not np.isfinite(degf) else format(degf, '.1f')))
            self.setItem(row, self.COL_DIST, gui_widgets.ReadOnlyTableItem())
            self.setSpan(row, self.COL_DIST, 1, 5)
            for col in [self.COL_DIST, self.COL_DEGF, self.COL_PARAM1, self.COL_PARAM2, self.COL_PARAM3, self.COL_PARAM4, self.COL_STDUNC]:
                with suppress(AttributeError):
                    self.item(row, col).setBackground(QtGui.QBrush(gui_common.COLOR_UNUSED))
            for col in [self.COL_VALUE, self.COL_UNITS, self.COL_DESC]:
                with suppress(AttributeError):
                    self.item(row, col).setBackground(QtGui.QBrush(COLOR2))

            for uncert in inpt.uncerts:
                self.setRowCount(self.rowCount() + 1)
                row = self.rowCount() - 1
                item = TableItemTex(uncert.name)
                item.setData(gui_widgets.ROLE_UNCERT, uncert)
                item.setData(gui_widgets.ROLE_VARIABLE, inpt)
                self.setItem(row, self.COL_VALUE, item)
                self.fill_uncertparams(row)

        self.blockSignals(False)
        self.fixSize()

    def fill_uncertparams(self, row):
        ''' Fill uncertainty parameters for the given row '''
        # Signals should already be blocked
        uncert = self.item(row, self.COL_VALUE).data(gui_widgets.ROLE_UNCERT)
        cmbdist = gui_widgets.ComboNoWheel()
        cmbdist.addItems(gui_common.settings.getDistributions())
        cmbdist.setCurrentIndex(cmbdist.findText(uncert.distname))
        if cmbdist.currentText() == '':
            cmbdist.addItem(uncert.distname)
            cmbdist.setCurrentIndex(cmbdist.count()-1)
        self.setCellWidget(row, self.COL_DIST, cmbdist)
        self.setItem(row, self.COL_UNITS, gui_widgets.EditableTableItem(format(uncert.units, '~P')))
        degf = uncert.degf
        self.setItem(row, self.COL_DEGF, gui_widgets.EditableTableItem('inf' if not np.isfinite(degf) else format(degf, '.1f')))
        self.setItem(row, self.COL_DESC, gui_widgets.EditableTableItem(uncert.desc))
        self.setItem(row, self.COL_STDUNC, gui_widgets.ReadOnlyTableItem('± {:.2g~P}'.format(uncert.std())))
        if uncert.distname in ['normal', 't']:
            if 'conf' in uncert.args:
                conf = float(uncert.args['conf'])
                k = ttable.t_factor(conf, degf)
            else:
                k = float(uncert.args.get('k', 1))
                conf = ttable.confidence(k, degf)
            uncstr = uncert.args.get('unc', k*float(uncert.args.get('std', 1)))  # Could be float or string
            with suppress(ValueError):
                uncstr = '{:.5g}'.format(uncstr)

            self.setItem(row, self.COL_PARAM1, gui_widgets.EditableTableItem(uncstr))
            self.setItem(row, self.COL_PARAM2, gui_widgets.EditableTableItem('{:.2f}'.format(k)))
            self.setItem(row, self.COL_PARAM3, gui_widgets.EditableTableItem('{:.2f}%'.format(float(conf)*100)))
            self.setItem(row, self.COL_PARAM4, gui_widgets.EditableTableItem(''))
        elif uncert.distname == 'histogram':
            for col in range(4):
                self.setItem(row, self.COL_PARAM1+col, gui_widgets.EditableTableItem(''))
        else:
            for col, arg in enumerate(sorted(uncert.distribution.argnames)):
                self.setItem(row, self.COL_PARAM1+col, gui_widgets.EditableTableItem(format(uncert.args.get(arg, 1))))
            for col in range(len(uncert.distribution.argnames), 4):
                self.setItem(row, self.COL_PARAM1+col, gui_widgets.EditableTableItem(''))

        item = self.item(row, self.COL_VALUE)
        cmbdist.currentIndexChanged.connect(lambda y, item=item: self.change_dist(item))

    def change_dist(self, item):
        ''' Distribution type in combobox was changed. item is the COL_VALUE item with
            data() of uncertainty component.
        '''
        self.blockSignals(True)
        row = item.row()
        uncert = item.data(gui_widgets.ROLE_UNCERT)
        distname = self.cellWidget(row, self.COL_DIST).currentText()   # Dist combo is always first child
        uncert.set_dist(distname)
        self.fill_uncertparams(row)
        self.rowchange(row, self.COL_DIST)  # To update header row
        self.blockSignals(False)
        self.updated.emit(uncert)

    def rowchange(self, row, column, oldrow=None, oldcolumn=None):
        ''' Selected row was changed - update header labels and refresh the preview plot '''
        if row is not None:
            uncert = None
            with suppress(AttributeError):
                uncert = self.item(row, self.COL_VALUE).data(gui_widgets.ROLE_UNCERT)
                if uncert.distname in ['normal', 't']:
                    arg = ['Uncertainty', 'k', 'Confidence']
                else:
                    arg = uncert.distribution.argnames.copy()
                    assert len(arg) <= 4
                for i in range(len(arg), 4):
                    arg.append('')

                self.setHorizontalHeaderLabels(['Variable', 'Value', 'Units', 'Distribution'] + arg + ['Degrees\nFreedom', 'Description', 'Standard\nUncertainty'])
        else:  # Unselected
            uncert = None
            self.setHorizontalHeaderLabels(['Variable', 'Value', 'Units', 'Distribution', 'Param 1', 'Param 2', 'Param 3', 'Param 4', 'Degrees\nFreedom', 'Description', 'Standard\nUncertainty'])
        self.updated.emit(uncert)

    def itemchange(self, item):
        ''' An item in the table was edited. Validate the input and update the model. '''
        COLOR = {True: gui_common.COLOR_OK, False: gui_common.COLOR_INVALID}
        self.blockSignals(True)
        column = self.column(item)
        row = self.row(item)
        inpt = self.item(row, self.COL_VALUE).data(gui_widgets.ROLE_VARIABLE)
        uncert = self.item(row, self.COL_VALUE).data(gui_widgets.ROLE_UNCERT)
        value = item.text().strip()
        obj = uncert if uncert is not None else inpt

        status = True
        if column == self.COL_VALUE:
            if item.data(gui_widgets.ROLE_ORIGDATA) == '':
                status = False
            elif uncert is not None:
                uncert.name = item.data(gui_widgets.ROLE_ORIGDATA)
            else:
                status = inpt.set_nom(item.data(gui_widgets.ROLE_ORIGDATA))

        elif column == self.COL_UNITS:
            status = obj.set_units(value)
            if status:
                item.setText(obj.get_unitstr())

        elif column == self.COL_DESC:
            obj.desc = value

        elif column == self.COL_PARAM1 and uncert.distname in ['normal', 't']:
            uncert.args['unc'] = value
            uncert.args.pop('std', None)
            if 'k' not in uncert.args:
                uncert.args['k'] = self.item(row, self.COL_PARAM2).text()

        elif column in [self.COL_DEGF, self.COL_PARAM1, self.COL_PARAM2, self.COL_PARAM3] and uncert.distname in ['normal', 't']:
            # For normal or t distributions where conf/k need extra processing
            value = value.strip('%')  # In case conf was entered with percent symbol
            try:
                value = float(value)
            except ValueError:
                status = False
            else:
                if column == self.COL_DEGF:
                    uncert.degf = value
                    status = uncert.updateparams()
                    if 'k' in uncert.args:
                        self.item(row, self.COL_PARAM3).setText('{:.2f}%'.format(ttable.confidence(float(uncert.args['k']), uncert.degf)*100))
                    elif 'conf' in uncert.args:
                        self.item(row, self.COL_PARAM2).setText('{:.2f}'.format(ttable.t_factor(float(uncert.args['conf']), uncert.degf)))

                elif column == self.COL_PARAM2:  # k
                    uncert.args['k'] = value
                    uncert.args.pop('conf', None)
                    self.item(row, self.COL_PARAM3).setText('{:.2f}%'.format(ttable.confidence(value, uncert.degf)*100))

                elif column == self.COL_PARAM3:  # confidence
                    uncert.args['conf'] = value/100   # Assume entry in percent
                    uncert.args.pop('k', None)
                    self.item(row, self.COL_PARAM2).setText('{:.2f}'.format(ttable.t_factor(value/100, uncert.degf)))
                    self.item(row, self.COL_PARAM3).setText('{:.2f}%'.format(value))

        elif column in [self.COL_DEGF]:
            try:
                value = float(value)
            except ValueError:
                status = False
            else:
                uncert.degf = value

        else:
            paramidx = column - self.COL_PARAM1
            if paramidx < len(uncert.distribution.argnames):  # When user fills in column that's not used
                paramname = uncert.distribution.argnames[paramidx]
                uncert.args[paramname] = value

        if uncert is not None:
            status = status and uncert.updateparams()

        item.setBackground(COLOR[status])
        if not status:
            self.clearSelection()

        item.valid = status
        self.blockSignals(False)
        if status:
            self.trickle_units()
            self.updated.emit(uncert)

    def trickle_units(self):
        ''' Trickle up/down changes in units and standard uncertainty '''
        self.blockSignals(True)
        for row in range(self.rowCount()):
            item = self.item(row, self.COL_VALUE)
            if item:
                uncert = item.data(gui_widgets.ROLE_UNCERT)
                inpt = item.data(gui_widgets.ROLE_VARIABLE)
                if uncert is None:  # Variable
                    degf = inpt.degf()
                    self.item(row, self.COL_VALUE).setExpr(str(inpt.nom))
                    self.item(row, self.COL_STDUNC).setText('± {:.2g~P}'.format(inpt.stdunc()))
                    self.item(row, self.COL_DEGF).setText('inf' if not np.isfinite(degf) else format(degf, '.1f'))
                else:  # Uncertainty
                    self.setItem(row, self.COL_DEGF, gui_widgets.EditableTableItem('{:.1f}'.format(uncert.degf)))
                    self.setItem(row, self.COL_UNITS, gui_widgets.EditableTableItem(format(uncert.units, '~P')))
                    self.item(row, self.COL_STDUNC).setText('± {:.2g~P}'.format(uncert.std()))
        self.blockSignals(False)

    def contextMenuEvent(self, event):
        ''' Show right-click context menu '''
        item = self.itemAt(event.pos())
        menu = QtWidgets.QMenu()

        if item:
            row = self.row(item)
            uncert = None
            with suppress(AttributeError):
                uncert = self.item(row, self.COL_VALUE).data(gui_widgets.ROLE_UNCERT)

            if uncert is None:  # Variable row
                inpt = self.item(row, self.COL_VALUE).data(gui_widgets.ROLE_VARIABLE)
                add = menu.addAction('Add uncertainty component')
                add.triggered.connect(lambda x, inpt=inpt, item=item: self.add_comp(inpt, row))

            else:  # Uncert Row
                inpt = self.item(row, self.COL_VALUE).data(gui_widgets.ROLE_VARIABLE)
                add = menu.addAction('Add uncertainty component')
                add.triggered.connect(lambda x, inpt=inpt, row=row: self.add_comp(inpt, row))
                rem = menu.addAction('Remove uncertainty component')
                rem.triggered.connect(lambda x, uncert=uncert, row=row: self.rem_comp(uncert, row))
                imp = menu.addAction('Import distribution from...')
                imp.triggered.connect(self.import_dist)
                hlp = menu.addAction('Distribution help...')
                def helppopup():
                    dlg = PopupHelp(uncert.distribution.helpstr())
                    dlg.exec_()
                hlp.triggered.connect(helppopup)
            menu.exec(event.globalPos())

    def import_dist(self):
        ''' Import a distribution from somewhere else in the project '''
        dlg = page_dataimport.DistributionSelectWidget(singlecol=True, project=self.unccalc.project)
        ok = dlg.exec_()
        if ok:
            distargs = dlg.get_dist()
            distname = distargs.pop('dist', 'normal')
            nominal = distargs.pop('expected', distargs.pop('median', distargs.pop('mean', 0)))
            if distname == 'histogram':
                distargs['histogram'] = (distargs['histogram'][0], np.array(distargs['histogram'][1])-nominal)

            selitem = self.currentItem()
            row = selitem.row()
            uncert = self.item(row, self.COL_VALUE).data(gui_widgets.ROLE_UNCERT)
            inpt = self.item(row, self.COL_VALUE).data(gui_widgets.ROLE_VARIABLE)
            inpt.set_nom(nominal)
            uncert.nom = nominal
            uncert.args = distargs
            uncert.set_dist(distname)
            uncert.updateparams()
            self.blockSignals(True)
            self.fill_uncertparams(row)
            self.trickle_units()
            self.blockSignals(False)
            self.updated.emit(uncert)

    def add_comp(self, inpt, row):
        ''' Add a blank uncertainty component '''
        self.blockSignals(True)
        i = 1
        name = 'u{}({})'.format(i, inpt.name)
        while name in [u.name for u in inpt.uncerts]:
            i += 1
            name = 'u{}({})'.format(i, inpt.name)
        unc = inpt.add_comp(name, unc=1.0, k=2)

        item = TableItemTex(unc.name)
        item.setData(gui_widgets.ROLE_UNCERT, unc)
        item.setData(gui_widgets.ROLE_VARIABLE, inpt)
        self.insertRow(row+1)
        self.setItem(row+1, self.COL_VALUE, item)
        self.fill_uncertparams(row+1)
        self.blockSignals(False)
        self.trickle_units()
        self.updated.emit(unc)
        self.fixSize()

    def rem_comp(self, unc, row):
        ''' Remove selected uncertainty component '''
        inpt = self.item(row, self.COL_VALUE).data(gui_widgets.ROLE_VARIABLE)
        idx = inpt.uncerts.index(unc)
        inpt.rem_comp(idx)
        self.removeRow(row)
        self.trickle_units()
        self.updated.emit(None)
        self.fixSize()

    def isValid(self):
        ''' Return True if all entries are valid (not red) '''
        for row in range(self.rowCount()):
            for col in range(self.columnCount()):
                with suppress(AttributeError):
                    if not self.item(row, col).valid:
                        return False
        return True

    def moveCursor(self, cursorAction, modifiers):
        ''' Override cursor so tab works as expected '''
        assert self.COL_CNT == 11   # If column defs change, need to update this tab-key behavior
        assert self.COL_STDUNC == 10
        assert self.COL_DIST == 3

        index = self.currentIndex()
        if index.isValid():
            uncert = index.sibling(index.row(), self.COL_VALUE).data(gui_widgets.ROLE_UNCERT)
            if cursorAction == QtWidgets.QAbstractItemView.MoveNext:
                if not uncert:  # Variable row
                    if index.column() == self.COL_UNITS:
                        return index.sibling(index.row(), self.COL_DESC)
                    elif index.column() == self.COL_DESC:
                        return index.sibling(index.row()+1, self.COL_VALUE)  # Jump to value on either variable or uncert row
                    else:
                        return index.sibling(index.row(), index.column()+1)
                else:  # Uncertainty row
                    if index.column() == self.COL_DESC:
                        return index.sibling(index.row()+1, self.COL_VALUE)
                    elif index.column() == self.COL_UNITS:
                        return index.sibling(index.row(), self.COL_PARAM1)  # Skip Distribution combobox
                    else:
                        return index.sibling(index.row(), index.column()+1)

            elif cursorAction == QtWidgets.QAbstractItemView.MovePrevious:
                if index.column() == self.COL_VALUE:
                    prevuncert = index.sibling(index.row()-1, self.COL_VALUE).data(gui_widgets.ROLE_UNCERT)
                    if not prevuncert:  # Going back to variable row
                        return index.sibling(index.row()-1, self.COL_DESC)
                    else:  # Going back to another uncertainty row
                        return index.sibling(index.row()-1, self.COL_DESC)
                elif not uncert and index.column() == self.COL_DESC:  # Variable row jump desc-> units
                    return index.sibling(index.row(), self.COL_UNITS)
                elif index.column() == self.COL_PARAM1:
                    return index.sibling(index.row(), self.COL_UNITS)  # Skip distribution combo
                else:
                    return index.sibling(index.row(), index.column()-1)

        return super().moveCursor(cursorAction, modifiers)


class UncertPreview(QtWidgets.QWidget):
    ''' Plot of uncertainty component PDF '''
    def __init__(self):
        super().__init__()
        self.setFixedSize(300, 250)
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.component = None

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.canvas)
        layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)

    def replot(self, comp=None):
        ''' Update the plot with PDF of InptUncert comp '''
        if comp is not None:
            self.component = comp

        if self.component is not None:
            self.figure.clf()
            ax = self.figure.add_subplot(1, 1, 1)
            x, y = self.component.pdf()
            ax.plot(x.magnitude, y)
            ax.text(.02, .9, '${}${}'.format(self.component.get_latex(), report.Unit(self.component.parentunits, bracket=True).latex()),
                    transform=ax.transAxes)
            ax.yaxis.set_ticks([])
            self.figure.tight_layout()
            self.canvas.draw_idle()


class PopupHelp(QtWidgets.QDialog):
    ''' Show a floating dialog window with a text message '''
    def __init__(self, text):
        super().__init__()
        gui_widgets.centerWindow(self, 600, 400)
        self.setModal(False)
        self.text = QtWidgets.QTextEdit()
        self.text.setReadOnly(True)
        font = QtGui.QFont('Courier')
        font.setStyleHint(QtGui.QFont.TypeWriter)
        self.text.setCurrentFont(font)
        self.text.setText(text)
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.text)
        self.setLayout(layout)


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

    def clear(self):
        ''' Clear and reset the table '''
        super().clear()
        self.setColumnCount(3)
        self.setRowCount(0)
        self.setHorizontalHeaderLabels(['Input 1', 'Input 2', 'Correlation Coefficient'])
        self.setColumnWidth(0, 100)
        self.setColumnWidth(1, 100)
        self.setColumnWidth(2, 200)
        self.varnames = []
        self.fixSize()

    def fixSize(self):
        ''' Adjust the size of the table to fit the number of rows '''
        height = max(self.horizontalHeader().height()+20, self.verticalHeader().length() + self.horizontalHeader().height())
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

    def addRow(self):
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
        v1.currentIndexChanged.connect(lambda idx, r=row, c=0: self.cmbchange(r, c))
        v2.currentIndexChanged.connect(lambda idx, r=row, c=1: self.cmbchange(r, c))
        self.blockSignals(False)
        self.fixSize()

    def setRow(self, row, v1, v2, cor):
        ''' Set values to an existing row '''
        self.cellWidget(row, 0).setCurrentIndex(self.cellWidget(row, 0).findText(v1))
        self.cellWidget(row, 1).setCurrentIndex(self.cellWidget(row, 1).findText(v2))
        self.item(row, 2).setText(format(cor, '.4f'))

    def getRow(self, row):
        ''' Get values for one row in the table. Returns (input1, input2, corrleation) '''
        v1 = self.cellWidget(row, 0).currentText()
        v2 = self.cellWidget(row, 1).currentText()
        val = self.item(row, 2).text()
        return v1, v2, val

    def cmbchange(self, row, col):
        ''' Combobox (input name) changed. '''
        self.cellChanged.emit(row, col)

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
    ROW_CNT = 2

    changed = QtCore.pyqtSignal(str, str)

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
        self.setItem(self.ROW_SAMPLES, self.COL_VALUE, gui_widgets.EditableTableItem(str(gui_common.settings.getSamples())))
        self.setItem(self.ROW_SEED, self.COL_NAME, gui_widgets.ReadOnlyTableItem('Random Seed'))
        self.setItem(self.ROW_SEED, self.COL_VALUE, gui_widgets.EditableTableItem(str(gui_common.settings.getRandomSeed())))
        self.resizeColumnsToContents()
        self.itemChanged.connect(self.valuechange)

        height = self.verticalHeader().length() + self.horizontalHeader().height()
        self.setFixedHeight(height)

    def valuechange(self, item):
        ''' A setting value was changed. '''
        row = self.row(item)
        name = self.item(row, self.COL_NAME).text()
        value = item.text()
        self.changed.emit(name, value)

    def setvalue(self, name, value):
        ''' Change the value of a setting '''
        item = self.findItems(name, QtCore.Qt.MatchExactly)
        assert len(item) == 1
        row = item[0].row()
        self.item(row, self.COL_VALUE).setText(str(value))


class PageInput(QtWidgets.QWidget):
    calculate = QtCore.pyqtSignal()

    ''' Page for setting up input parameters '''
    def __init__(self, unccalc, parent=None):
        super().__init__(parent)
        self.unccalc = unccalc
        # Set up widgets
        self.funclist = FunctionTableWidget()
        self.meastable = MeasTableWidget(unccalc)
        self.uncpreview = UncertPreview()
        self.corrtable = CorrelationTableWidget()
        self.btnCalc = QtWidgets.QPushButton('Calculate')
        self.description = QtWidgets.QPlainTextEdit()
        self.settings = SettingsWidget()

        self.panel = gui_widgets.WidgetPanel()
        self.panel.setStyleSheet(TREESTYLE)
        self.panel.setAnimated(True)
        self.btnCalc.clicked.connect(self.calculate)

        self.panel.add_widget('Measurement Model', self.funclist)
        self.panel.add_widget('Uncertainties', self.meastable)
        self.panel.add_widget('Uncertainty Preview', self.uncpreview)
        self.panel.add_widget('Correlations', self.corrtable)
        self.panel.add_widget('Notes', self.description)
        self.panel.add_widget('Settings', self.settings)
        self.panel.expand('Measurement Model')
        self.panel.expand('Values')
        self.panel.expand('Uncertainties')

        self.funclist.resizerows.connect(self.panel.fixSize)
        self.meastable.resizerows.connect(self.panel.fixSize)
        self.corrtable.resizerows.connect(self.panel.fixSize)
        self.meastable.updated.connect(self.uncpreview.replot)

        calclayout = QtWidgets.QHBoxLayout()
        calclayout.addStretch()
        calclayout.addWidget(self.btnCalc)
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.panel)
        layout.addLayout(calclayout)
        self.setLayout(layout)

    def set_unccalc(self, unccalc):
        ''' Set the uncertainty calc object '''
        self.unccalc = unccalc

    def isValid(self):
        ''' Check if all inputs are valid '''
        return self.funclist.isValid() and self.meastable.isValid()


#------------------------------------------------------------
# Output view control widgets
#------------------------------------------------------------
class OutputExpandedWidget(QtWidgets.QWidget):
    ''' Widget for controlling expanded uncertainties page '''
    changed = QtCore.pyqtSignal()

    def __init__(self):
        super().__init__()
        self.GUMexpanded = gui_widgets.GUMExpandedWidget(multiselect=True, dflt=[0, 1, 2, 3, 4])
        self.MCexpanded = gui_widgets.MCExpandedWidget(multiselect=True, dflt=[0, 1, 2, 3, 4])
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.GUMexpanded)
        layout.addWidget(self.MCexpanded)
        layout.addStretch()
        self.setLayout(layout)

        self.GUMexpanded.GUMtype.setCurrentIndex(0 if gui_common.settings.getCoverageTypeGUM() == 't' else 1)
        self.MCexpanded.MCtype.setCurrentIndex(0 if gui_common.settings.getCoverageTypeMC() == 'symmetric' else 1)
        self.GUMexpanded.set_buttons(gui_common.settings.getCoverageGUMt() + gui_common.settings.getCoverageGUMk())
        self.MCexpanded.set_buttons(gui_common.settings.getCoverageMC())
        self.GUMexpanded.changed.connect(self.changed)
        self.MCexpanded.changed.connect(self.changed)


class OutputPlotWidget(QtWidgets.QWidget):
    ''' Widget for controlling output plot view '''
    changed = QtCore.pyqtSignal()

    def __init__(self):
        super().__init__()
        self.cmbjoint = QtWidgets.QComboBox()
        self.cmbjoint.addItems(['Individual PDF', 'Joint PDF'])
        self.cmbjoint.setVisible(False)
        self.cmbhist = QtWidgets.QComboBox()
        self.cmbhist.addItems(['Histograms', 'PDFs'])
        self.cmbscat = QtWidgets.QComboBox()
        self.cmbscat.addItems(['Scatter', 'Contours'])
        self.cmbscat.setVisible(False)
        self.bins = gui_widgets.SpinWidget('Bins:')
        self.bins.spin.setRange(3, 1E8)
        self.bins.setValue(100)
        self.points = gui_widgets.SpinWidget('Points:')
        self.points.spin.setRange(1, 1E8)
        self.points.setValue(10000)
        self.points.setVisible(False)
        self.flist = gui_widgets.ListSelectWidget()

        self.showmc = QtWidgets.QCheckBox('Show Monte-Carlo Result')
        self.showgum = QtWidgets.QCheckBox('Show GUM Result')
        self.overlay = QtWidgets.QCheckBox('Overlay GUM and MC Plots')
        self.overlay.setVisible(False)
        self.eqscale = QtWidgets.QCheckBox('Equalize Scales')
        self.eqscale.setVisible(False)
        self.showmc.setChecked(True)
        self.showgum.setChecked(True)
        self.namedesc = QtWidgets.QCheckBox('Label by description')
        self.showleg = QtWidgets.QCheckBox('Show Legend')
        self.showleg.setChecked(True)
        self.intervals = QtWidgets.QCheckBox('Show Coverage Invervals')
        self.GUMexpanded = gui_widgets.GUMExpandedWidget(multiselect=True, dflt=[])
        self.MCexpanded = gui_widgets.MCExpandedWidget(multiselect=True, dflt=[1])
        self.GUMexpanded.setVisible(False)
        self.MCexpanded.setVisible(False)

        self.bins.valueChanged.connect(self.changed)
        self.points.valueChanged.connect(self.changed)
        self.showmc.stateChanged.connect(self.changed)
        self.showgum.stateChanged.connect(self.changed)
        self.overlay.stateChanged.connect(self.changed)
        self.eqscale.stateChanged.connect(self.changed)
        self.namedesc.stateChanged.connect(self.changed)
        self.showleg.stateChanged.connect(self.changed)
        self.GUMexpanded.changed.connect(self.changed)
        self.MCexpanded.changed.connect(self.changed)
        self.flist.checkChange.connect(self.changed)
        self.cmbhist.currentIndexChanged.connect(self.typechanged)
        self.cmbjoint.currentIndexChanged.connect(self.typechanged)
        self.cmbscat.currentIndexChanged.connect(self.typechanged)
        self.intervals.stateChanged.connect(self.toggleintervals)

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.cmbjoint)
        layout.addWidget(self.cmbhist)
        layout.addWidget(self.cmbscat)
        layout.addWidget(self.bins)
        layout.addWidget(self.points)
        layout.addWidget(self.flist)
        layout.addWidget(self.showmc)
        layout.addWidget(self.showgum)
        layout.addWidget(self.overlay)
        layout.addWidget(self.eqscale)
        layout.addWidget(self.namedesc)
        layout.addWidget(self.showleg)
        layout.addWidget(self.intervals)
        layout.addWidget(self.GUMexpanded)
        layout.addWidget(self.MCexpanded)
        layout.addStretch()
        self.setLayout(layout)

    def typechanged(self):
        ''' Plot type has changed '''
        assert self.cmbjoint.findText('Joint PDF') != -1  # Update below if names change
        assert self.cmbscat.findText('Scatter') != -1  # Update below if names change
        joint = self.cmbjoint.currentText() == 'Joint PDF' and len(self.flist) > 1
        self.cmbhist.setVisible(not joint)
        self.points.setVisible(joint and self.cmbscat.currentText() == 'Scatter')
        self.bins.setVisible(not self.points.isVisible())
        self.overlay.setVisible(joint)
        self.eqscale.setVisible(joint)
        self.cmbscat.setVisible(joint)
        if joint and self.intervals.isChecked():
            self.intervals.setChecked(False)
        self.intervals.setVisible(not joint)
        self.flist.setVisible((not joint and len(self.flist) > 1) or (joint and len(self.flist) > 2))
        self.changed.emit()

    def toggleintervals(self):
        self.GUMexpanded.setVisible(self.intervals.isChecked())
        self.MCexpanded.setVisible(self.intervals.isChecked())
        self.changed.emit()

    def contour(self):
        ''' Get contour on/off state '''
        assert self.cmbjoint.findText('Joint PDF') != -1  # Update below if names change
        assert self.cmbscat.findText('Contours') != -1  # Update below if names change
        assert self.cmbhist.findText('PDFs') != -1  # Update below if names change
        joint = self.cmbjoint.currentText() == 'Joint PDF' and len(self.flist) > 1
        return ((joint and self.cmbscat.currentText() == 'Contours') or (not joint and self.cmbhist.currentText() == 'PDFs'))

    def joint(self):
        ''' Is this a joint probability plot? '''
        return self.cmbjoint.currentText() == 'Joint PDF' and len(self.flist) > 1

    def set_funclist(self, flist):
        ''' Set list of function names '''
        self.flist.clear()
        self.flist.addItems(flist)
        self.flist.selectAll()
        joint = self.cmbjoint.currentText() == 'Joint PDF'
        self.flist.setVisible((not joint and len(self.flist) > 1) or (joint and len(self.flist) > 2))
        self.cmbjoint.setVisible(len(self.flist) > 1)


class OutputMCSampleWidget(QtWidgets.QWidget):
    ''' Widget for controlling MC input plot view '''
    changed = QtCore.pyqtSignal()

    def __init__(self):
        super().__init__()
        self.cmbjoint = QtWidgets.QComboBox()
        self.cmbjoint.addItems(['Individual PDF', 'Joint PDF'])
        self.cmbscat = QtWidgets.QComboBox()
        self.cmbscat.addItems(['Scatter', 'Contours'])
        self.cmbscat.setVisible(False)
        self.bins = gui_widgets.SpinWidget('Bins:')
        self.bins.spin.setRange(3, 1E8)
        self.bins.setValue(100)
        self.points = gui_widgets.SpinWidget('Points:')
        self.points.setValue(10000)
        self.points.spin.setRange(3, 1E8)
        self.points.setVisible(False)
        self.ilist = gui_widgets.ListSelectWidget()
        self.namedesc = QtWidgets.QCheckBox('Label by description')
        self.bins.valueChanged.connect(self.changed)
        self.points.valueChanged.connect(self.changed)
        self.cmbjoint.currentIndexChanged.connect(self.typechanged)
        self.cmbscat.currentIndexChanged.connect(self.typechanged)
        self.namedesc.stateChanged.connect(self.changed)
        self.ilist.checkChange.connect(self.changed)
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.cmbjoint)
        layout.addWidget(self.cmbscat)
        layout.addWidget(self.bins)
        layout.addWidget(self.points)
        layout.addWidget(self.ilist)
        layout.addWidget(self.namedesc)
        layout.addStretch()
        self.setLayout(layout)

    def typechanged(self):
        ''' Plot type has changed '''
        assert self.cmbjoint.findText('Joint PDF') != -1   # Update here if names change
        assert self.cmbscat.findText('Scatter') != -1
        joint = self.cmbjoint.currentText() == 'Joint PDF'
        self.cmbscat.setVisible(joint and len(self.ilist) > 1)
        self.points.setVisible(self.cmbjoint.currentText() == 'Joint PDF' and self.cmbscat.currentText() == 'Scatter')
        self.bins.setVisible(not self.points.isVisible())
        self.ilist.setVisible((joint and len(self.ilist) > 2) or (not joint and len(self.ilist) > 1))
        self.changed.emit()

    def set_inptlist(self, ilist):
        ''' Set list of input names '''
        assert self.cmbjoint.findText('Joint PDF') != -1   # Update here if names change
        joint = self.cmbjoint.currentText() == 'Joint PDF'
        self.ilist.clear()
        self.ilist.addItems(ilist)
        self.ilist.selectAll()
        self.cmbjoint.setVisible(len(ilist) > 1)
        self.cmbscat.setVisible(len(self.ilist) > 1 and joint)
        self.ilist.setVisible((joint and len(self.ilist) > 2) or (not joint and len(self.ilist) > 1))


class OutputMCDistributionWidget(QtWidgets.QWidget):
    ''' Widget for controlling MC Distribution Analysis View '''
    changed = QtCore.pyqtSignal()

    def __init__(self):
        super().__init__()
        self.cmbdist = QtWidgets.QComboBox()
        dists = gui_common.settings.getDistributions()
        dists = [d for d in dists if distributions.fittable(d)]
        self.cmbdist.addItems(dists)
        self.cmbdist.currentIndexChanged.connect(self.changed)
        self.cmbfunc = QtWidgets.QComboBox()
        self.cmbfunc.currentIndexChanged.connect(self.changed)
        self.flabel = QtWidgets.QLabel('Function')
        self.label = QtWidgets.QLabel()
        layout = QtWidgets.QVBoxLayout()
        dlayout = QtWidgets.QHBoxLayout()
        dlayout.addWidget(QtWidgets.QLabel('Distribution'))
        dlayout.addWidget(self.cmbdist)
        dlayout.addStretch()
        layout.addLayout(dlayout)
        self.flayout = QtWidgets.QHBoxLayout()
        self.flayout.addWidget(self.flabel)
        self.flayout.addWidget(self.cmbfunc)
        self.flayout.addStretch()
        layout.addLayout(self.flayout)
        layout.addWidget(QtWidgets.QLabel('Fit Parameters:'))
        layout.addWidget(self.label)
        layout.addStretch()
        self.setLayout(layout)

    def update_label(self, fitparams):
        ''' Update the label with parameter names. fitparams is dictionary of name:value '''
        if fitparams is not None:
            s = ''
            for name, val in fitparams.items():
                s += '{} = {:.4g}\n'.format(name, val)
            self.label.setText(s)

    def set_funclist(self, flist):
        self.blockSignals(True)
        self.cmbfunc.clear()
        self.cmbfunc.addItems(flist)
        self.flabel.setVisible(len(flist) > 1)
        self.cmbfunc.setVisible(len(flist) > 1)
        self.blockSignals(False)


class OutputMCConvergeWidget(QtWidgets.QWidget):
    ''' Widget for controlling MC Convergence Analysis View '''
    changed = QtCore.pyqtSignal()

    def __init__(self):
        super().__init__()
        self.relative = QtWidgets.QCheckBox('Relative to final value')
        self.relative.stateChanged.connect(self.changed)
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.relative)
        layout.addStretch()
        self.setLayout(layout)


class OutputGUMValidityWidget(QtWidgets.QWidget):
    ''' Widget for controlling display of GUM validity page '''
    changed = QtCore.pyqtSignal()

    def __init__(self):
        super().__init__()
        self.ndig = gui_widgets.SpinWidget('Significant Digits')
        self.ndig.spin.setRange(1, 5)
        self.ndig.setValue(1)
        self.ndig.valueChanged.connect(self.changed)
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.ndig)
        layout.addStretch()
        self.setLayout(layout)


class OutputGUMDerivationWidget(QtWidgets.QWidget):
    ''' Widget for controlling display of GUM derivation page '''
    changed = QtCore.pyqtSignal()

    def __init__(self):
        super().__init__()
        self.showvalues = QtWidgets.QCheckBox('Show derivation with values')
        self.showvalues.stateChanged.connect(self.changed)
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.showvalues)
        layout.addStretch()
        self.setLayout(layout)


class OutputReportGen(QtWidgets.QWidget):
    ''' Class for controlling full output report '''
    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.report = None  # Cache the report object, only refresh on button press
        self.btnRefresh = QtWidgets.QPushButton('Refresh')
        self.chkSummary = QtWidgets.QCheckBox('Summary')
        self.chkOutputPlot = QtWidgets.QCheckBox('Output Plots')
        self.chkInputs = QtWidgets.QCheckBox('Input Values')
        self.chkComponents = QtWidgets.QCheckBox('Uncertainty Budget')
        self.chkSensitivity = QtWidgets.QCheckBox('Sensitivity Coefficients')
        self.chkExpanded = QtWidgets.QCheckBox('Expanded Uncertainties')
        self.chkGUMderiv = QtWidgets.QCheckBox('GUM Derivation')
        self.chkGUMvalid = QtWidgets.QCheckBox('GUM Validity')
        self.chkMChist = QtWidgets.QCheckBox('MC Input Histograms')
        self.chkMCconverge = QtWidgets.QCheckBox('MC Convergence')
        self.chkSummary.setChecked(True)
        self.chkOutputPlot.setChecked(True)
        self.chkInputs.setChecked(True)
        self.chkComponents.setChecked(True)
        self.chkSensitivity.setChecked(True)
        self.chkExpanded.setChecked(True)
        self.chkGUMderiv.setChecked(True)
        self.chkGUMvalid.setChecked(True)
        self.chkMChist.setChecked(True)
        self.chkMCconverge.setChecked(True)
        # NOTE: pull other settings (expanded levels, etc.) from other pages
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.chkSummary)
        layout.addWidget(self.chkOutputPlot)
        layout.addWidget(self.chkInputs)
        layout.addWidget(self.chkExpanded)
        layout.addWidget(self.chkComponents)
        layout.addWidget(self.chkSensitivity)
        layout.addWidget(self.chkGUMderiv)
        layout.addWidget(self.chkGUMvalid)
        layout.addWidget(self.chkMChist)
        layout.addWidget(self.chkMCconverge)
        layout.addWidget(self.btnRefresh)
        self.setLayout(layout)


class PageOutput(QtWidgets.QWidget):
    ''' Page for viewing output values '''
    back = QtCore.pyqtSignal()

    def __init__(self, uncCalc, parent=None):
        super().__init__(parent)
        self.outputSelect = QtWidgets.QComboBox()
        self.outputSelect.addItems(['Summary', 'Comparison Plots', 'Expanded Uncertainties', 'Uncertainty Budget',
                                    'GUM Derivation', 'GUM Validity', 'Monte Carlo Distribution',
                                    'Monte Carlo Input Plots', 'Monte Carlo Convergence', 'Full Report'])
        self.outputPlot = OutputPlotWidget()
        self.outputExpanded = OutputExpandedWidget()
        self.outputMCsample = OutputMCSampleWidget()
        self.outputMCdist = OutputMCDistributionWidget()
        self.outputMCconv = OutputMCConvergeWidget()
        self.outputGUMvalid = OutputGUMValidityWidget()
        self.outputGUMderiv = OutputGUMDerivationWidget()
        self.outputReportSetup = OutputReportGen()
        self.txtOutput = gui_widgets.MarkdownTextEdit()
        self.fig = Figure()
        self.canvas = FigureCanvas(self.fig)
        self.toolbar = NavigationToolbar(self.canvas, self, coordinates=True)
        self.plotWidget = QtWidgets.QWidget()
        playout = QtWidgets.QVBoxLayout()   # Plot layout
        playout.addWidget(self.canvas)
        playout.addWidget(self.toolbar)
        self.plotWidget.setLayout(playout)

        self.ctrlStack = QtWidgets.QStackedWidget()
        self.ctrlStack.addWidget(QtWidgets.QWidget())  # 1 - Summary (blank - no controls)
        self.ctrlStack.addWidget(self.outputPlot)      # 2
        self.ctrlStack.addWidget(self.outputExpanded)  # 3
        self.ctrlStack.addWidget(QtWidgets.QWidget())  # 4 - Uncertainty budget (blank)
        self.ctrlStack.addWidget(self.outputGUMderiv)  # 5
        self.ctrlStack.addWidget(self.outputGUMvalid)  # 6
        self.ctrlStack.addWidget(self.outputMCdist)    # 7
        self.ctrlStack.addWidget(self.outputMCsample)  # 8
        self.ctrlStack.addWidget(self.outputMCconv)    # 9
        self.ctrlStack.addWidget(self.outputReportSetup)  # 10 - Full report (no controls)
        self.ctrlStack.addWidget(QtWidgets.QWidget())  # 11 - Warnings (blank)
        self.outputStack = QtWidgets.QStackedWidget()
        self.outputStack.addWidget(self.plotWidget)
        self.outputStack.addWidget(self.txtOutput)
        self.btnBack = QtWidgets.QPushButton('Back')
        self.btnBack.clicked.connect(self.goback)

        wlayout = QtWidgets.QHBoxLayout()   # Main Layout
        self.oplayout = QtWidgets.QVBoxLayout()  # Options along left side
        self.oplayout.addWidget(self.outputSelect)
        self.oplayout.addWidget(self.ctrlStack)
        self.oplayout.addStretch()
        self.oplayout.addStretch()
        self.oplayout.addWidget(self.btnBack)
        wlayout.addLayout(self.oplayout, stretch=0)
        wlayout.addWidget(self.outputStack, stretch=10)
        self.setLayout(wlayout)

        self.outputSelect.currentIndexChanged.connect(self.set_outputpage)
        self.outputExpanded.changed.connect(self.outputupdate)
        self.outputPlot.changed.connect(self.outputupdate)
        self.outputMCsample.changed.connect(self.outputupdate)
        self.outputMCconv.changed.connect(self.outputupdate)
        self.outputGUMderiv.changed.connect(self.outputupdate)
        self.outputGUMvalid.changed.connect(self.outputupdate)
        self.outputMCdist.changed.connect(self.outputupdate)
        self.outputReportSetup.btnRefresh.clicked.connect(self.refresh_fullreport)
        self.set_unccalc(uncCalc)

    def set_outputpage(self):
        ''' Change the output page '''
        self.ctrlStack.setCurrentIndex(self.outputSelect.currentIndex())
        self.outputupdate()

    def set_unccalc(self, unccalc):
        ''' Save the uncertainty calculator object used to show the output pages '''
        self.uncCalc = unccalc

    def refresh_fullreport(self):
        rptsetup = {'summary': self.outputReportSetup.chkSummary.isChecked(),
                    'outputs': self.outputReportSetup.chkOutputPlot.isChecked(),
                    'inputs': self.outputReportSetup.chkInputs.isChecked(),
                    'components': self.outputReportSetup.chkComponents.isChecked(),
                    'sens': self.outputReportSetup.chkSensitivity.isChecked(),
                    'expanded': self.outputReportSetup.chkExpanded.isChecked(),
                    'gumderv': self.outputReportSetup.chkGUMderiv.isChecked(),
                    'gumvalid': self.outputReportSetup.chkGUMvalid.isChecked(),
                    'gumvaliddig': self.outputGUMvalid.ndig.value(),
                    'mchist': self.outputReportSetup.chkMChist.isChecked(),
                    'mcconv': self.outputReportSetup.chkMCconverge.isChecked(),
                    'mcconvnorm': self.outputMCconv.relative.isChecked(),
                    'expandedparams': {'intervalsgum': self.outputExpanded.GUMexpanded.get_covlist(),
                                       'intervalsmc': self.outputExpanded.MCexpanded.get_covlist(),
                                       'norm': self.outputExpanded.GUMexpanded.GUMtype.currentText() == 'Normal/k',
                                       'shortest': self.outputExpanded.MCexpanded.MCtype.currentText() == 'Shortest'},
                    'gumvalues': self.outputGUMderiv.showvalues.isChecked(),
                    'mchistparams': {'joint': self.outputMCsample.cmbjoint.currentText() == 'Joint PDF' and len(self.outdata.ucalc.get_baseinputs()) > 1,
                                     'plotargs': {'bins': self.outputMCsample.bins.value(),
                                     'points': self.outputMCsample.points.value(),
                                     'contour': self.outputMCsample.cmbscat.currentText() == 'Contours',
                                     'labelmode': 'desc' if self.outputMCsample.namedesc.isChecked() else 'name',
                                     'inpts': self.outputMCsample.ilist.getSelectedIndexes(),
                                     'cmap': gui_common.settings.getColormap('cmapcontour')}},
                     'outplotparams': {'joint': self.outputPlot.joint(),
                                      'plotargs': {'showgum': self.outputPlot.showgum.isChecked(),
                                      'showmc': self.outputPlot.showmc.isChecked(),
                                      'equal_scale': self.outputPlot.eqscale.isChecked(),
                                      'overlay': self.outputPlot.overlay.isChecked(),
                                      'bins': self.outputPlot.bins.value(),
                                      'points': self.outputPlot.points.value(),
                                      'contour': self.outputPlot.contour(),
                                      'labelmode': 'desc' if self.outputPlot.namedesc.isChecked() else 'name',
                                      'funcs': self.outputPlot.flist.getSelectedIndexes(),
                                      'legend': self.outputPlot.showleg.isChecked(),
                                      'cmap': gui_common.settings.getColormap('cmapcontour'),
                                      'cmapmc': gui_common.settings.getColormap('cmapscatter')}}
                    }
        if self.outputPlot.intervals.isChecked():
            rptsetup['outplotparams']['plotargs']['intervals'] = self.outputPlot.MCexpanded.get_covlist()
            rptsetup['outplotparams']['plotargs']['intTypeMC'] = self.outputPlot.MCexpanded.MCtype.currentText()
            rptsetup['outplotparams']['plotargs']['intervalsGUM'] = self.outputPlot.GUMexpanded.get_covlist()
            rptsetup['outplotparams']['plotargs']['intTypeGUMt'] = self.outputPlot.GUMexpanded.GUMtype.currentText() == 'Student-t'

        self.outputReportSetup.report = self.outdata.report_all(**rptsetup)  # Cache report for displaying/saving
        self.outputupdate()
        return self.outputReportSetup.report

    def outputupdate(self):
        ''' Update the output page based on widget settings. '''
        PLOT = 0
        TEXT = 1
        option = self.outputSelect.currentText()

        # If these names change, make sure corresponding option in "if" below is updated.
        assert self.outputSelect.findText('Summary') != -1
        assert self.outputSelect.findText('Expanded Uncertainties') != -1
        assert self.outputSelect.findText('Uncertainty Budget') != -1
        assert self.outputSelect.findText('GUM Validity') != -1
        assert self.outputSelect.findText('GUM Derivation') != -1
        assert self.outputSelect.findText('Comparison Plots') != -1
        assert self.outputSelect.findText('Monte Carlo Input Plots') != -1
        assert self.outputSelect.findText('Monte Carlo Distribution') != -1
        assert self.outputSelect.findText('Full Report') != -1
        if (option in ['Summary', 'Expanded Uncertainties', 'Uncertainty Budget', 'GUM Validity', 'GUM Derivation',
                       'Monte Carlo Components', 'Full Report', 'Warnings']):
            self.outputStack.setCurrentIndex(TEXT)
        else:
            self.outputStack.setCurrentIndex(PLOT)

        if option == 'Summary':
            r = report.Report()
            r.hdr('Results', level=2)
            r.append(self.outdata.report_summary())
            self.txtOutput.setReport(r)

        elif option == 'Full Report':
            if self.outputReportSetup.report is None:
                self.refresh_fullreport()
            self.txtOutput.setReport(self.outputReportSetup.report)

        elif option == 'Expanded Uncertainties':
            assert self.outputExpanded.GUMexpanded.GUMtype.findText('Normal/k') != -1
            assert self.outputExpanded.MCexpanded.MCtype.findText('Shortest') != -1
            intervalsgum = self.outputExpanded.GUMexpanded.get_covlist()
            intervals = self.outputExpanded.MCexpanded.get_covlist()
            norm = self.outputExpanded.GUMexpanded.GUMtype.currentText() == 'Normal/k'
            shortest = self.outputExpanded.MCexpanded.MCtype.currentText() == 'Shortest'
            r = report.Report()
            r.hdr('Expanded Uncertainty', level=2)
            r.append(self.outdata.report_expanded(covlist=intervals, normal=norm, shortest=shortest, covlistgum=intervalsgum))
            self.txtOutput.setReport(r)

        elif option == 'Uncertainty Budget':
            self.txtOutput.setReport(self.outdata.report_allinputs())

        elif option == 'Warnings':
            self.txtOutput.setReport(self.outdata.report_warns())

        elif option == 'GUM Derivation':
            solve = self.outputGUMderiv.showvalues.isChecked()
            self.txtOutput.setReport(self.outdata.report_derivation(solve=solve))

        elif option == 'Monte Carlo Convergence':
            self.outdata.plot_converge(plot=self.fig, relative=self.outputMCconv.relative.isChecked())
            self.canvas.draw_idle()
            self.fig.tight_layout()

        elif option == 'Comparison Plots':
            assert self.outputPlot.GUMexpanded.GUMtype.findText('Student-t') != -1  # Update below if names change
            plotargs = {'showgum': self.outputPlot.showgum.isChecked(),
                        'showmc': self.outputPlot.showmc.isChecked(),
                        'equal_scale': self.outputPlot.eqscale.isChecked(),
                        'overlay': self.outputPlot.overlay.isChecked(),
                        'bins': self.outputPlot.bins.value(),
                        'points': self.outputPlot.points.value(),
                        'contour': self.outputPlot.contour(),
                        'labelmode': 'desc' if self.outputPlot.namedesc.isChecked() else 'name',
                        'funcs': self.outputPlot.flist.getSelectedIndexes(),
                        'legend': self.outputPlot.showleg.isChecked(),
                        'cmap': gui_common.settings.getColormap('cmapcontour'),
                        'cmapmc': gui_common.settings.getColormap('cmapscatter')}
            if self.outputPlot.intervals.isChecked():
                plotargs['intervals'] = self.outputPlot.MCexpanded.get_covlist()
                plotargs['intTypeMC'] = self.outputPlot.MCexpanded.MCtype.currentText()
                plotargs['intervalsGUM'] = self.outputPlot.GUMexpanded.get_covlist()
                plotargs['intTypeGUMt'] = self.outputPlot.GUMexpanded.GUMtype.currentText() == 'Student-t'

            assert self.outputPlot.GUMexpanded.GUMtype.findText('Student-t') != -1   # In case name changes from 'Student-t' it needs to be updated here

            if self.outputPlot.joint():
                self.outdata.plot_outputscatter(plot=self.fig, **plotargs)
            else:
                self.outdata.plot_pdf(plot=self.fig, **plotargs)
            self.canvas.draw_idle()

        elif option == 'Monte Carlo Input Plots':
            assert self.outputMCsample.cmbjoint.findText('Joint PDF') != -1  # Update below if names change
            assert self.outputMCsample.cmbscat.findText('Contours') != -1  # Update below if names change
            numinpts = len(self.outdata.ucalc.get_baseinputs())
            joint = self.outputMCsample.cmbjoint.currentText() == 'Joint PDF' and numinpts > 1
            plotargs = {'bins': self.outputMCsample.bins.value(),
                        'points': self.outputMCsample.points.value(),
                        'contour': self.outputMCsample.cmbscat.currentText() == 'Contours',
                        'labelmode': 'desc' if self.outputMCsample.namedesc.isChecked() else 'name',
                        'inpts': self.outputMCsample.ilist.getSelectedIndexes(),
                        'cmap': gui_common.settings.getColormap('cmapcontour')}
            if joint:
                self.outdata.plot_xscatter(plot=self.fig, **plotargs)
            else:
                self.outdata.plot_xhists(plot=self.fig, **plotargs)
            self.canvas.draw_idle()

        elif option == 'GUM Validity':
            ndig = self.outputGUMvalid.ndig.value()
            r = self.outdata.report_validity(ndig=ndig)
            self.txtOutput.setReport(r)

        elif option == 'Monte Carlo Distribution':
            fidx = self.outputMCdist.cmbfunc.currentIndex()
            dist = self.outputMCdist.cmbdist.currentText()
            y = self.outdata.foutputs[fidx].mc.samples.magnitude
            fitparams = plotting.fitdist(y, distname=dist, plot=self.fig, qqplot=True, bins=100, points=200)
            with suppress(IndexError):  # Raises if axes weren't added (maybe invalid samples)
                self.fig.axes[0].set_title('Distribution Fit')
                self.fig.axes[1].set_title('Probability Plot')
            self.fig.tight_layout()
            self.outputMCdist.update_label(fitparams)
            self.canvas.draw_idle()

        else:
            raise NotImplementedError

    def update(self, outdata):
        ''' Calculation run, update the page '''
        self.outdata = outdata
        funclist = [f.name for f in self.outdata.ucalc.functions if f.show]
        self.outputPlot.set_funclist(funclist)
        self.outputMCdist.set_funclist(funclist)
        self.outputMCsample.set_inptlist(self.outdata.ucalc.get_baseinputnames())
        self.outputReportSetup.report = None
        idx = self.outputSelect.findText('Warnings')
        if len(self.outdata.report_warns().get_md().strip()) > 0:
            if idx < 0 or idx is None:
                self.outputSelect.addItem('Warnings')
            self.outputSelect.setCurrentIndex(self.outputSelect.count()-1)
        elif idx >= 0:
            self.outputSelect.removeItem(idx)
        if self.outputSelect.currentText() == 'Full Report':
            self.refresh_fullreport()
            self.outputupdate()

    def goback(self):
        self.back.emit()


class UncertPropWidget(QtWidgets.QWidget):
    ''' Uncertainty propagation widget '''
    openconfigfolder = QtCore.QStandardPaths.standardLocations(QtCore.QStandardPaths.HomeLocation)[0]
    newtype = QtCore.pyqtSignal(object, str)

    PG_INPUT = 0
    PG_OUTPUT = 1

    def __init__(self, item, parent=None):
        super().__init__(parent)
        assert isinstance(item, uncertainty.UncertCalc)
        self.pginput = PageInput(item)
        self.pgoutput = PageOutput(item)
        self.stack = QtWidgets.QStackedWidget()
        self.stack.addWidget(self.pginput)
        self.stack.addWidget(self.pgoutput)

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.stack)
        self.setLayout(layout)

        # Menu
        self.menu = QtWidgets.QMenu('&Uncertainty')
        self.actChkUnits = QtWidgets.QAction('Check Units...', self)
        self.actSweep = QtWidgets.QAction('New uncertainty sweep from model', self)
        self.actReverse = QtWidgets.QAction('New reverse calculation from model', self)
        self.actImportDists = QtWidgets.QAction('Import uncertainty distributions...', self)
        self.actClear = QtWidgets.QAction('Clear inputs', self)
        self.actSaveReport = QtWidgets.QAction('Save Report...', self)
        self.actSaveSamplesCSV = QtWidgets.QAction('Text (CSV)...', self)
        self.actSaveSamplesNPZ = QtWidgets.QAction('Binary (NPZ)...', self)

        self.menu.addAction(self.actChkUnits)
        self.menu.addSeparator()
        self.menu.addAction(self.actImportDists)
        self.menu.addAction(self.actClear)
        self.menu.addSeparator()
        self.menu.addAction(self.actSweep)
        self.menu.addAction(self.actReverse)
        self.menu.addSeparator()
        self.menu.addAction(self.actSaveReport)
        self.mnuSaveSamples = QtWidgets.QMenu('Save Monte Carlo Samples')
        self.mnuSaveSamples.addAction(self.actSaveSamplesCSV)
        self.mnuSaveSamples.addAction(self.actSaveSamplesNPZ)
        self.menu.addMenu(self.mnuSaveSamples)
        self.actSaveReport.setEnabled(False)
        self.mnuSaveSamples.setEnabled(False)

        self.actImportDists.triggered.connect(self.importdistributions)
        self.actClear.triggered.connect(self.clearinput)
        self.actSweep.triggered.connect(lambda event, x=item: self.newtype.emit(x.get_config(), 'sweep'))
        self.actReverse.triggered.connect(lambda event, x=item: self.newtype.emit(x.get_config(), 'reverse'))
        self.actSaveReport.triggered.connect(self.save_report)
        self.actSaveSamplesCSV.triggered.connect(self.save_samples_csv)
        self.actSaveSamplesNPZ.triggered.connect(self.save_samples_npz)
        self.actChkUnits.triggered.connect(self.checkunits)

        self.pginput.calculate.connect(self.calculate)
        self.pginput.funclist.orderchange.connect(self.funcorderchanged)
        self.pginput.funclist.funcchanged.connect(self.funcchanged)
        self.pginput.funclist.funcremoved.connect(self.funcremoved)
        self.pginput.corrtable.cellChanged.connect(self.set_correlations)
        self.pginput.settings.changed.connect(self.setSetting)
        self.pginput.description.textChanged.connect(self.setDesc)
        self.pgoutput.back.connect(self.backbutton)
        self.set_calc(item)
        gui_common.set_plot_style()

    def get_menu(self):
        ''' Get menu for this widget '''
        return self.menu

    def backbutton(self):
        ''' Back button pressed. '''
        self.stack.setCurrentIndex(0)
        self.actSaveReport.setEnabled(False)
        self.mnuSaveSamples.setEnabled(False)

    def clearinput(self):
        ''' Clear all the input/output values '''
        self.pginput.funclist.clear()
        self.pginput.meastable.clear()
        self.pginput.corrtable.clear()
        self.pginput.description.setPlainText('')
        self.setSetting('Random Seed', gui_common.settings.getRandomSeed())
        self.setSetting('Monte Carlo Samples', gui_common.settings.getSamples())
        self.uncCalc.clearall()
        self.actSweep.setEnabled(False)
        self.actReverse.setEnabled(False)
        self.stack.setCurrentIndex(0)

    def checkunits(self):
        ''' Show units/dimensionality report '''
        dlg = gui_widgets.MarkdownTextEdit()
        dlg.setMinimumSize(800, 600)
        dlg.setReport(self.uncCalc.units_report())
        dlg.show()

    def funcchanged(self, row, fdict):
        ''' Function was changed. '''
        self.uncCalc.set_function(fdict['expr'], idx=row, name=fdict['name'], desc=fdict['desc'],
                                  show=fdict.get('report', True), outunits=fdict['unit'])
        self.uncCalc.add_required_inputs()
        baseinputs = self.uncCalc.get_baseinputs()
        self.pginput.meastable.filltable(baseinputs)
        self.pginput.corrtable.setVarNames(self.uncCalc.get_reqd_inputs())
        self.actSweep.setEnabled(True)
        self.actReverse.setEnabled(True)

    def funcremoved(self, row):
        ''' A function was removed from the list '''
        self.uncCalc.remove_function(row)

    def funcorderchanged(self):
        ''' Functions were reordered by drag/drop '''
        names = [self.pginput.funclist.item(r, self.pginput.funclist.COL_NAME).data(gui_widgets.ROLE_ORIGDATA) for r in range(self.pginput.funclist.rowCount())]
        self.uncCalc.reorder(names)

    def setDesc(self):
        ''' Description was edited. Save to uncCalc. '''
        self.uncCalc.longdescription = self.pginput.description.toPlainText()

    def setSetting(self, name, value):
        if name == 'Random Seed':
            try:
                seed = abs(int(float(value)))
            except (TypeError, ValueError, OverflowError):
                seed = None
                self.pginput.settings.setvalue(name, None)
            else:
                self.uncCalc.seed = seed

        elif name == 'Monte Carlo Samples':
            try:
                samples = int(float(value))  # cast to float first so exp-notation will work
            except (TypeError, ValueError, OverflowError):
                samples = 1000000
                self.pginput.settings.setvalue(name, samples)
            else:
                self.uncCalc.samples = samples

    def set_correlations(self):
        ''' Set correlation table in unc calc. '''
        self.uncCalc.clear_corr()
        for row in range(self.pginput.corrtable.rowCount()):
            v1, v2, val = self.pginput.corrtable.getRow(row)
            try:
                f = float(val)
                self.uncCalc.correlate_vars(v1, v2, f)
            except ValueError:
                if self.pginput.corrtable.item(row, 2) is not None:
                    self.pginput.corrtable.item(row, 2).setBackground(QtGui.QBrush(gui_common.COLOR_INVALID))
                self.pginput.corrtable.clearSelection()
            else:
                self.pginput.corrtable.item(row, 2).setBackground(QtGui.QBrush(gui_common.COLOR_OK))

    def importdistributions(self):
        ''' Load uncertainty components from data for multiple input variables '''
        varnames = self.uncCalc.get_baseinputnames()
        dlg = page_dataimport.DistributionSelectWidget(singlecol=False, project=self.uncCalc.project, coloptions=varnames)
        if dlg.exec_():
            dists = dlg.get_dist()

            for varname, params in dists.items():
                if varname == '_correlation_':
                    for (v1, v2), corr in params.items():
                        if corr != 0.:
                            self.pginput.corrtable.addRow()
                            self.pginput.corrtable.setRow(self.pginput.corrtable.rowCount()-1, v1, v2, corr)
                    self.pginput.panel.expand('Correlations')

                else:
                    nom = params.pop('expected', params.pop('median', params.pop('mean', None)))  # Don't pass along median, it's handled by Variable
                    if nom is not None:
                        self.uncCalc.set_input(varname, nom=nom)
                    self.uncCalc.set_uncert(varname, name=f'u({varname})', degf=params.get('df', np.inf),
                                            units=str(self.uncCalc.get_input(varname).units), **params)
                self.pginput.meastable.filltable()
                self.backbutton()

    def calculate(self):
        ''' Run the calculation '''
        msg = None
        if not (self.pginput.funclist.isValid() and self.pginput.meastable.isValid()):
            msg = 'Invalid input parameter!'

        elif len(self.uncCalc.functions) < 1:
            msg = 'No functions to compute!'

        elif self.uncCalc.check_circular():
            msg = 'Circular reference in function definitions'

        else:
            try:
                self.uncCalc.check_dimensionality()
            except (DimensionalityError, UndefinedUnitError) as e:
                msg = 'Units Error: {}'.format(e)
            except OffsetUnitCalculusError as e:
                badunit = re.findall(r'\((.+ )', str(e))[0].split()[0].strip(', ')
                msg = 'Ambiguous unit {}. Try "{}".'.format(badunit, 'delta_{}'.format(badunit))
            except RecursionError:
                msg = 'Error - possible circular reference in function definitions'

        if msg is None:
            try:
                self.uncCalc.calculate()
            except OffsetUnitCalculusError as e:
                badunit = re.findall(r'\((.+ )', str(e))[0].split()[0].strip(', ')
                msg = 'Ambiguous unit {}. Try "{}".'.format(badunit, 'delta_{}'.format(badunit))
            except RecursionError:
                msg = 'Error - possible circular reference in function definitions'
            except (TypeError, ValueError):
                msg = 'Error computing solution!'

        if msg is None:
            self.stack.setCurrentIndex(self.PG_OUTPUT)
            self.pgoutput.update(self.uncCalc.out)
            self.pgoutput.outputupdate()
            self.actSaveReport.setEnabled(True)
            self.mnuSaveSamples.setEnabled(True)
        else:
            QtWidgets.QMessageBox.warning(self, 'Uncertainty Calculator', msg)
            self.actSaveReport.setEnabled(False)
            self.mnuSaveSamples.setEnabled(False)

    def set_calc(self, calc):
        ''' Set calculator object to be displayed (e.g. loading from project file) '''
        self.uncCalc = calc
        self.pginput.corrtable.reset()
        self.pginput.funclist.blockSignals(True)
        self.pgoutput.set_unccalc(self.uncCalc)
        self.pginput.set_unccalc(self.uncCalc)
        self.pginput.funclist.setFunclist(calc.functions)
        self.pginput.corrtable.setVarNames([i.name for i in self.uncCalc.get_baseinputs()])
        self.pginput.settings.setvalue('Monte Carlo Samples', self.uncCalc.samples)
        self.pginput.settings.setvalue('Random Seed', self.uncCalc.seed)
        self.pginput.description.setPlainText(str(self.uncCalc.longdescription))

        if self.uncCalc._corr is not None:
            for v1, v2, c in self.uncCalc.get_corr_list():
                if c != 0.:
                    self.pginput.corrtable.addRow()
                    self.pginput.corrtable.setRow(self.pginput.corrtable.rowCount()-1, v1, v2, c)
        if self.uncCalc.longdescription:
            self.pginput.panel.expand('Notes')
        self.stack.setCurrentIndex(self.PG_INPUT)
        self.pginput.funclist.blockSignals(False)

    def get_report(self):
        ''' Get full report of curve fit, using page settings '''
        report = self.pgoutput.outputReportSetup.report
        if report is None and hasattr(self.pgoutput, 'outdata'):
            report = self.pgoutput.refresh_fullreport()
        return report

    def save_report(self):
        ''' Save full output report to file, using user's GUI settings '''
        gui_widgets.savereport(self.get_report())

    def save_samples_csv(self):
        ''' Save Monte-Carlo samples (inputs and outputs) to CSV file. This file can get big fast! '''
        fname, _ = QtWidgets.QFileDialog.getSaveFileName(caption='Select file to save', directory=self.openconfigfolder)
        if fname:
            self.uncCalc.save_samples(fname, 'csv')

    def save_samples_npz(self):
        ''' Save Monte-Carlo samples to NPZ (compressed numpy) file.

            Load into python using numpy.load() to return a dictionary with
            'samples' and 'hdr' keys.
        '''
        fname, _ = QtWidgets.QFileDialog.getSaveFileName(caption='Select file to save', directory=self.openconfigfolder)
        if fname:
            self.uncCalc.save_samples(fname, 'npz')
