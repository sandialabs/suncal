''' Page for propagating uncertainty calculations '''

import re
from contextlib import suppress
import numpy as np
import sympy
from pint import DimensionalityError, UndefinedUnitError, OffsetUnitCalculusError
from PyQt5 import QtWidgets, QtGui, QtCore
import matplotlib.pyplot as plt
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
        ratio = QtWidgets.QApplication.instance().devicePixelRatio()
        px.loadFromData(report.Math(expr).svg_buf(fontsize=16*ratio).read())
        px.setDevicePixelRatio(ratio)
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

    def setFunclist(self, model):
        ''' Set the measurement Model (list of functions) '''
        self.blockSignals(True)
        self.clear()
        self.blockSignals(False)
        for row, (name, expr, unit, desc) in enumerate(zip(model.outnames, model.exprs, model.outunits, model.descriptions)):
            self.addRow()
            # Let signals populate the other tables
            self.setItem(row, self.COL_NAME, TableItemTex(name))
            self.setItem(row, self.COL_EXPR, TableItemTex(expr))
            self.setItem(row, self.COL_UNIT, gui_widgets.EditableTableItem(unit))
            self.setItem(row, self.COL_DESC, gui_widgets.EditableTableItem(desc))
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
            add = menu.addAction('Add measurement function')
            rem = menu.addAction('Remove measurement function')
            add.triggered.connect(self.addRow)
            rem.triggered.connect(self.remRow)

            solve = menu.addMenu('Solve for')
            for symbol in fn.free_symbols:
                act = solve.addAction(str(symbol))
                act.triggered.connect(lambda x, fn=fn, name=name, var=str(symbol), row=row: self.solvefor(fn, name, var, row))
            menu.exec(event.globalPos())

        else:
            menu = QtWidgets.QMenu()
            add = menu.addAction('Add measurement function')
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
    COL_VALNAME = 1
    COL_VALUE = 2
    COL_UNITS = 3
    COL_DEGF = 4
    COL_DESC = 5
    COL_STDUNC = 6
    COL_PREVIEW = 7
    COL_BTN = 8
    COL_CNT = 9

    resizerows = QtCore.pyqtSignal()

    def __init__(self, unccalc):
        super().__init__()
        self.unccalc = unccalc
        self.clear()
        self.setEditTriggers(QtWidgets.QTableWidget.NoEditTriggers)  # Only start editing manually

        self.inputs = None  # uncertainty.Input object
        self.setStyleSheet(TABLESTYLE)
        self._delegate = gui_widgets.LatexDelegate()
        self.setItemDelegateForColumn(self.COL_VARNAME, self._delegate)
        self.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.verticalHeader().hide()

        self.itemActivated.connect(self.checkEdit)
        self.itemChanged.connect(self.itemchange)
        self.horizontalHeader().sectionResized.connect(self.col_resize)
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
        self.setHorizontalHeaderLabels(['Variable', 'Parameter', 'Value', 'Units', 'Degrees\nFreedom', 'Description', 'Standard\nUncertainty', 'Preview', ''])
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
            (uncert is not None and column not in [self.COL_VALNAME, self.COL_STDUNC])):
            self.editItem(item)

    def filltable(self, inputs=None):
        ''' Fill the table with inputs defined in Input object, or refresh using existing Input '''
        self.blockSignals(True)
        if inputs is not None:
            self.inputs = inputs
        self.clear()
        COLOR2 = QtGui.QBrush(QtGui.QColor(246, 246, 246, 255))
        for inpt in sorted(self.inputs, key=lambda x: x.name):
            self.setRowCount(self.rowCount() + 1)
            row = self.rowCount() - 1
            inptitem = TableItemTex(inpt.name)
            inptitem.setBackground(QtGui.QBrush(gui_common.COLOR_UNUSED))
            valueitem = gui_widgets.EditableTableItem(format(inpt.nom, '.5g'))
            valueitem.setData(gui_widgets.ROLE_VARIABLE, inpt)  # Data is always in COL_VALUE
            self.setItem(row, self.COL_VARNAME, inptitem)
            self.setItem(row, self.COL_VALNAME, gui_widgets.ReadOnlyTableItem('Measured'))
            self.setItem(row, self.COL_VALUE, valueitem)
            self.setItem(row, self.COL_UNITS, gui_widgets.EditableTableItem(format(inpt.units, '~P')))
            self.setItem(row, self.COL_DESC, gui_widgets.EditableTableItem(inpt.desc))
            self.setItem(row, self.COL_STDUNC, gui_widgets.ReadOnlyTableItem('± {:.2g~P}'.format(inpt.stdunc())))
            self.setItem(row, self.COL_PREVIEW, QtWidgets.QTableWidgetItem())
            btn = gui_widgets.TreeButton('+')
            btn.setToolTip('Add Uncertainty Component')
            self.setCellWidget(row, self.COL_BTN, btn)
            btn.clicked.connect(lambda x, inpt=inpt, row=row: self.add_comp(inpt))
            degf = inpt.degf()
            self.setItem(row, self.COL_DEGF, gui_widgets.ReadOnlyTableItem('inf' if not np.isfinite(degf) else format(degf, '.1f')))
            for col in range(self.COL_CNT):
                with suppress(AttributeError):
                    self.item(row, col).setBackground(QtGui.QBrush(gui_common.COLOR_UNUSED))
            for col in [self.COL_VALUE, self.COL_UNITS, self.COL_DESC]:
                with suppress(AttributeError):
                    self.item(row, col).setBackground(QtGui.QBrush(COLOR2))

            for uncert in inpt.uncerts:
                self.setRowCount(self.rowCount() + 1)
                row = self.rowCount() - 1
                item = QtWidgets.QTableWidgetItem()
                item.setData(gui_widgets.ROLE_UNCERT, uncert)
                item.setData(gui_widgets.ROLE_VARIABLE, inpt)
                item.setData(gui_widgets.ROLE_TOPITEM, item)
                self.setItem(row, self.COL_VALUE, item)
                self.fill_uncertparams(row)

        self.blockSignals(False)
        self.fixSize()

    def fill_uncertparams(self, row):
        ''' Fill uncertainty parameters for the given row '''
        # Signals should already be blocked
        inpt = self.item(row, self.COL_VALUE).data(gui_widgets.ROLE_VARIABLE)
        uncert = self.item(row, self.COL_VALUE).data(gui_widgets.ROLE_UNCERT)
        topitem = self.item(row, self.COL_VALUE).data(gui_widgets.ROLE_TOPITEM)
        cmbdist = gui_widgets.ComboNoWheel()
        cmbdist.addItems(gui_common.settings.getDistributions())

        cmbdist.setCurrentIndex(cmbdist.findText(uncert.distname))
        if cmbdist.currentText() == '':
            cmbdist.addItem(uncert.distname)
            cmbdist.setCurrentIndex(cmbdist.count()-1)

        self.setItem(row, self.COL_VALNAME, gui_widgets.ReadOnlyTableItem('Distribution'))
        self.setCellWidget(row, self.COL_VALUE, cmbdist)

        self.setItem(row, self.COL_UNITS, gui_widgets.EditableTableItem(format(uncert.units, '~P')))
        degf = uncert.degf
        self.setItem(row, self.COL_VARNAME, TableItemTex(uncert.name))
        self.setItem(row, self.COL_DEGF, gui_widgets.EditableTableItem('inf' if not np.isfinite(degf) else format(degf, '.1f')))
        self.setItem(row, self.COL_DESC, gui_widgets.EditableTableItem(uncert.desc))
        self.setItem(row, self.COL_STDUNC, gui_widgets.ReadOnlyTableItem('± {:.2g~P}'.format(uncert.std())))
        btn = gui_widgets.TreeButton(gui_common.CHR_ENDASH)
        btn.setToolTip('Remove Uncertainty Component')
        btn.clicked.connect(lambda x, uncert=uncert, row=row: self.rem_comp(uncert))
        self.setCellWidget(row, self.COL_BTN, btn)

        if uncert.distname in ['normal', 't']:
            newrows = 4
            if 'conf' in uncert.args:
                conf = float(uncert.args['conf'])
                k = ttable.t_factor(conf, degf)
            else:
                k = float(uncert.args.get('k', 1))
                conf = ttable.confidence(k, degf)
            uncstr = uncert.args.get('unc', k*float(uncert.args.get('std', 1)))  # Could be float or string
            with suppress(ValueError):
                uncstr = '{:.5g}'.format(uncstr)

            self.insertRow(row+1)
            self.setItem(row+1, self.COL_VALNAME, gui_widgets.ReadOnlyTableItem('Confidence'))
            self.setItem(row+1, self.COL_VALUE, gui_widgets.EditableTableItem('{:.2f}%'.format(float(conf)*100)))
            self.item(row+1, self.COL_VALUE).setData(gui_widgets.ROLE_UNCERT, uncert)
            self.item(row+1, self.COL_VALUE).setData(gui_widgets.ROLE_VARIABLE, inpt)
            self.item(row+1, self.COL_VALUE).setData(gui_widgets.ROLE_TOPITEM, topitem)
            self.insertRow(row+1)
            self.setItem(row+1, self.COL_VALNAME, gui_widgets.ReadOnlyTableItem('k'))
            self.setItem(row+1, self.COL_VALUE, gui_widgets.EditableTableItem('{:.2f}'.format(k)))
            self.item(row+1, self.COL_VALUE).setData(gui_widgets.ROLE_UNCERT, uncert)
            self.item(row+1, self.COL_VALUE).setData(gui_widgets.ROLE_VARIABLE, inpt)
            self.item(row+1, self.COL_VALUE).setData(gui_widgets.ROLE_TOPITEM, topitem)
            self.insertRow(row+1)
            self.setItem(row+1, self.COL_VALNAME, gui_widgets.ReadOnlyTableItem('Uncertainty'))
            self.setItem(row+1, self.COL_VALUE, gui_widgets.EditableTableItem(uncstr))
            self.item(row+1, self.COL_VALUE).setData(gui_widgets.ROLE_UNCERT, uncert)
            self.item(row+1, self.COL_VALUE).setData(gui_widgets.ROLE_VARIABLE, inpt)
            self.item(row+1, self.COL_VALUE).setData(gui_widgets.ROLE_TOPITEM, topitem)

        elif uncert.distname == 'histogram':
            newrows = 1  # No more rows to add, just distribution box

        else:
            newrows = len(uncert.distribution.argnames) + 1
            for r, arg in enumerate(reversed(sorted(uncert.distribution.argnames))):
                self.insertRow(row+1)
                self.setItem(row+1, self.COL_VALNAME, gui_widgets.ReadOnlyTableItem(arg))
                self.setItem(row+1, self.COL_VALUE, gui_widgets.EditableTableItem(format(uncert.args.get(arg, 1))))
                self.item(row+1, self.COL_VALUE).setData(gui_widgets.ROLE_UNCERT, uncert)
                self.item(row+1, self.COL_VALUE).setData(gui_widgets.ROLE_VARIABLE, inpt)
                self.item(row+1, self.COL_VALUE).setData(gui_widgets.ROLE_TOPITEM, topitem)

        prev = UncertPreview()
        prev.setFixedSize(self.columnWidth(self.COL_PREVIEW)-1, self.rowHeight(0)*newrows)
        prev.replot(uncert)
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

    def col_resize(self, index, oldsize, newsize):
        ''' Column was resized. Resize the preview plot to fit '''
        if index == self.COL_PREVIEW:
            previews = [self.cellWidget(row, self.COL_PREVIEW) for row in range(self.rowCount())]
            previews = set(filter(None, previews))
            for prev in previews:
                prev.setFixedSize(newsize-1, prev.size().height())

    def change_dist(self, item):
        ''' Distribution type in combobox was changed. item is the COL_VALUE item with
            data() of uncertainty component.
        '''
        self.blockSignals(True)
        row = item.row()
        uncert = item.data(gui_widgets.ROLE_UNCERT)
        distname = self.cellWidget(row, self.COL_VALUE).currentText()   # Dist combo is always first child
        oldrows = 3 if uncert.distname in ['normal', 't'] else len(uncert.distribution.argnames)

        # Remove old rows except the first one with dist combobox
        for r in range(oldrows):
            self.removeRow(row+1)

        uncert.set_dist(distname)
        self.fill_uncertparams(row)
        self.blockSignals(False)
        self.fixSize()

    def itemchange(self, item):
        ''' An item in the table was edited. Validate the input and update the model. '''
        COLOR = {True: gui_common.COLOR_OK,
                 False: gui_common.COLOR_INVALID}
        COLORINPT = {True: QtGui.QBrush(QtGui.QColor(246, 246, 246, 255)),
                     False: gui_common.COLOR_INVALID}
        self.blockSignals(True)
        column = self.column(item)
        row = self.row(item)
        inpt = self.item(row, self.COL_VALUE).data(gui_widgets.ROLE_VARIABLE)
        uncert = self.item(row, self.COL_VALUE).data(gui_widgets.ROLE_UNCERT)
        value = item.text().strip()
        obj = uncert if uncert is not None else inpt
        toprow = self.item(row, self.COL_VALUE).data(gui_widgets.ROLE_TOPITEM)
        toprow = row if toprow is None else toprow.row()

        status = True
        origdata = item.data(gui_widgets.ROLE_ORIGDATA)
        if column == self.COL_VARNAME:
            if origdata == '':
                status = False
            elif uncert is not None:
                uncert.name = item.data(gui_widgets.ROLE_ORIGDATA)

        elif column == self.COL_UNITS:
            status = obj.set_units(value)
            if status:
                item.setText(obj.get_unitstr())

        elif column == self.COL_DESC:
            obj.desc = value

        elif column == self.COL_VALUE:
            if origdata == '':
                status = False
            elif uncert is None:
                status = inpt.set_nom(item.text())
            else:
                try:
                    fvalue = float(value.strip('%'))
                except ValueError:
                    fvalue = None

                if self.item(row, self.COL_VALNAME).text() == 'Uncertainty':
                    uncert.args['unc'] = value
                    uncert.args.pop('std', None)
                    if 'k' not in uncert.args:
                        uncert.args['k'] = self.item(row+1, self.COL_VALUE).text()
                elif self.item(row, self.COL_VALNAME).text() == 'k':
                    if fvalue is None:
                        status = False
                    else:
                        uncert.args['k'] = value
                        uncert.args.pop('conf', None)
                        self.item(row+1, self.COL_VALUE).setText('{:.2f}%'.format(ttable.confidence(fvalue, uncert.degf)*100))
                elif self.item(row, self.COL_VALNAME).text() == 'Confidence':
                    if fvalue is None:
                        status = False
                    else:
                        uncert.args['conf'] = fvalue/100   # Assume entry in percent
                        uncert.args.pop('k', None)
                        self.item(row-1, self.COL_VALUE).setText('{:.2f}'.format(ttable.t_factor(fvalue/100, uncert.degf)))
                        self.item(row, self.COL_VALUE).setText('{:.2f}%'.format(fvalue))
                else:  # Other uncertainty parameters
                    paramidx = row - toprow - 1
                    paramname = uncert.distribution.argnames[paramidx]
                    uncert.args[paramname] = value

        elif column in [self.COL_DEGF] and uncert.distname in ['normal', 't']:
            # For normal or t distributions where conf/k need extra processing
            try:
                value = float(value)
            except ValueError:
                status = False
            else:
                uncert.degf = value
                status = uncert.updateparams()
                if 'k' in uncert.args:
                    self.item(toprow+3, self.COL_VALUE).setText('{:.2f}%'.format(ttable.confidence(float(uncert.args['k']), uncert.degf)*100))
                elif 'conf' in uncert.args:
                    self.item(toprow+2, self.COL_VALUE).setText('{:.2f}'.format(ttable.t_factor(float(uncert.args['conf']), uncert.degf)))

        elif column in [self.COL_DEGF]:
            # degf for other distributions
            try:
                value = float(value)
            except ValueError:
                status = False
            else:
                uncert.degf = value

        if uncert is not None:
            status = status and uncert.updateparams()
            item.setBackground(COLOR[status])
        else:
            item.setBackground(COLORINPT[status])

        if not status:
            self.clearSelection()

        item.valid = status
        self.blockSignals(False)
        if status:
            self.trickle_units()
            self.replotall()

    def trickle_units(self):
        ''' Trickle up/down changes in units and standard uncertainty '''
        self.blockSignals(True)
        for row in range(self.rowCount()):
            item = self.item(row, self.COL_VALUE)
            if item:
                uncert = item.data(gui_widgets.ROLE_UNCERT)
                inpt = item.data(gui_widgets.ROLE_VARIABLE)
                toprow = self.item(row, self.COL_VALUE).data(gui_widgets.ROLE_TOPITEM)
                toprow = row if toprow is None else toprow.row()
                if uncert is None:  # Variable
                    degf = inpt.degf()
                    self.item(toprow, self.COL_STDUNC).setText('± {:.2g~P}'.format(inpt.stdunc()))
                    self.item(toprow, self.COL_DEGF).setText('inf' if not np.isfinite(degf) else format(degf, '.1f'))
                else:  # Uncertainty
                    self.setItem(toprow, self.COL_DEGF, gui_widgets.EditableTableItem('{:.1f}'.format(uncert.degf)))
                    self.setItem(toprow, self.COL_UNITS, gui_widgets.EditableTableItem(format(uncert.units, '~P')))
                    self.item(toprow, self.COL_STDUNC).setText('± {:.2g~P}'.format(uncert.std()))
        self.blockSignals(False)

    def replotall(self):
        ''' Replot all preview plots '''
        done = []
        for row in range(self.rowCount()):
            prev = self.cellWidget(row, self.COL_PREVIEW)
            if prev is not None and prev not in done:
                prev.replot(self.item(row, self.COL_VALUE).data(gui_widgets.ROLE_UNCERT))
                done.append(prev)

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
                    dlg = gui_widgets.PopupHelp(uncert.distribution.helpstr())
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

            # Remove old rows if different distribution
            self.blockSignals(True)
            oldrows = 3 if uncert.distname in ['normal', 't'] else len(uncert.distribution.argnames)
            for r in range(oldrows):
                self.removeRow(row+1)

            inpt = self.item(row, self.COL_VALUE).data(gui_widgets.ROLE_VARIABLE)
            inpt.set_nom(nominal)
            # Find input line so it can be set to nominal
            for r in range(self.rowCount()):
                if self.item(r, self.COL_VALUE).data(gui_widgets.ROLE_VARIABLE) is inpt:
                    toprow = r
                    break
            self.item(toprow, self.COL_VALUE).setText(str(nominal))
            uncert.nom = nominal
            uncert.args = distargs
            uncert.set_dist(distname)
            uncert.updateparams()
            self.fill_uncertparams(row)
            self.trickle_units()
            self.fixSize()
            self.blockSignals(False)

    def add_comp(self, inpt, row=None):
        ''' Add a blank uncertainty component

            Parameters
            ----------
            inpt: InputVariable
                The input to add uncertainty to
            row: int
                Row in the table. If None, the new component will be added to
                the beginning of this input's list of uncertainties
        '''
        if row is None:  # Find first row of this input
            for r in range(self.rowCount()):
                if self.item(r, self.COL_VALUE).data(gui_widgets.ROLE_VARIABLE) is inpt:
                    row = r
                    break

        self.blockSignals(True)
        i = 1
        name = 'u{}({})'.format(i, inpt.name)
        while name in [u.name for u in inpt.uncerts]:
            i += 1
            name = 'u{}({})'.format(i, inpt.name)
        unc = inpt.add_comp(name, unc=1.0, k=2)

        item = self.item(row, self.COL_VALUE).data(gui_widgets.ROLE_TOPITEM)
        if item:
            if item.data(gui_widgets.ROLE_UNCERT).distname in ['normal', 't']:
                startrow = item.row() + 4
            else:
                startrow = item.row() + len(item.data(gui_widgets.ROLE_UNCERT).distribution.argnames) + 1
        else:
            startrow = row + 1

        item = QtWidgets.QTableWidgetItem()
        item.setData(gui_widgets.ROLE_UNCERT, unc)
        item.setData(gui_widgets.ROLE_VARIABLE, inpt)
        item.setData(gui_widgets.ROLE_TOPITEM, item)
        self.insertRow(startrow)
        self.setItem(startrow, self.COL_VALUE, item)
        self.fill_uncertparams(startrow)
        self.trickle_units()
        self.blockSignals(False)
        self.fixSize()

    def rem_comp(self, unc, row=None):
        ''' Remove selected uncertainty component

            Parameters
            ----------
            unc: InputUncert
                The uncertainty object to remove
            row: int
                Row in the table. If None, first row tied to this unc will
                be removed.
        '''
        if row is None:  # Find first row of this uncertainty
            for r in range(self.rowCount()):
                if self.item(r, self.COL_VALUE).data(gui_widgets.ROLE_UNCERT) is unc:
                    row = r
                    break

        inpt = self.item(row, self.COL_VALUE).data(gui_widgets.ROLE_VARIABLE)
        idx = inpt.uncerts.index(unc)
        inpt.rem_comp(idx)
        if unc.distname in ['normal', 't']:
            rcount = 4  # 3+1 for distribution
        else:
            rcount = len(unc.distribution.argnames) + 1  # +1 for distribution row

        item = self.item(row, self.COL_VALUE).data(gui_widgets.ROLE_TOPITEM)
        if item:
            startrow = item.row()
        else:
            startrow = row

        for i in range(rcount):
            self.removeRow(startrow)
        self.trickle_units()
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
        assert self.COL_CNT == 9
        assert self.COL_STDUNC == 6   # If column defs change, need to update this tab-key behavior

        # This thing is hideous. Keyboard navigation doesn't work well with table spans,
        # have to manually jump to top row of the spanned cells.

        def toprow(uncert, row=None):
            if row is None:
                row = index.row()
            prevrow = index.sibling(row, self.COL_VALUE)
            while prevrow.isValid() and prevrow.data(gui_widgets.ROLE_UNCERT) is uncert:
                row -= 1
                prevrow = index.sibling(row, self.COL_VALUE)
            return row + 1

        def botrow(uncert):
            row = index.row()
            nextrow = index.sibling(row, self.COL_VALUE)
            while nextrow.isValid() and nextrow.data(gui_widgets.ROLE_UNCERT) is uncert:
                row += 1
                nextrow = index.sibling(row, self.COL_VALUE)
            return row - 1

        index = self.currentIndex()
        if index.isValid():
            uncert = index.sibling(index.row(), self.COL_VALUE).data(gui_widgets.ROLE_UNCERT)
            if cursorAction == QtWidgets.QAbstractItemView.MoveNext:
                if not uncert:  # Variable row
                    if index.column() == self.COL_UNITS:
                        return index.sibling(index.row(), self.COL_DESC)
                    elif index.column() == self.COL_DESC:
                        nextuncert = index.sibling(index.row()+1, self.COL_VALUE).data(gui_widgets.ROLE_UNCERT)
                        if nextuncert is not None:
                            return index.sibling(index.row()+1, self.COL_VARNAME)  # Jump to uncertainty name
                        else:
                            return index.sibling(index.row()+1, self.COL_VALUE)  # No uncertainties for this variable
                    else:
                        return index.sibling(index.row(), index.column()+1)
                else:  # Uncertainty row
                    if index.column() == self.COL_DESC:
                        brow = botrow(uncert)
                        nextunc = index.sibling(brow+1, self.COL_VALUE).data(gui_widgets.ROLE_UNCERT)
                        if nextunc:
                            return index.sibling(brow+1, self.COL_VARNAME)  # Next row is Uncertainty
                        else:
                            return index.sibling(brow+1, self.COL_VALUE)  # Next row is Variable
                    elif index.column() == self.COL_VARNAME:
                        return index.sibling(toprow(uncert)+1, self.COL_VALUE)  # Skip distribution combobox
                    elif index.column() == self.COL_VALUE:  # Uncertainty parameter row, tab down before tabbing to Units
                        nextrow = index.sibling(index.row()+1, self.COL_VALUE)
                        if nextrow.isValid() and nextrow.data(gui_widgets.ROLE_UNCERT) is uncert:
                            return nextrow
                        else:
                            return index.sibling(toprow(uncert), self.COL_UNITS)
                    else:
                        return index.sibling(index.row(), index.column()+1)

            elif cursorAction == QtWidgets.QAbstractItemView.MovePrevious:
                if uncert is None:  # Variable Row
                    if index.column() == self.COL_VALUE:
                        prevuncert = index.sibling(index.row()-1, self.COL_VALUE).data(gui_widgets.ROLE_UNCERT)
                        if prevuncert:
                            return index.sibling(toprow(prevuncert, index.row()-1), self.COL_DESC)
                        else:
                            return index.sibling(index.row()-1, self.COL_DESC)
                    elif index.column() == self.COL_DESC:
                        return index.sibling(index.row(), self.COL_UNITS)
                    else:
                        return index.sibling(index.row(), index.column()-1)

                else:  # Uncertainty Row
                    if index.column() == self.COL_UNITS:
                        return index.sibling(botrow(uncert), self.COL_VALUE)
                    elif index.column() == self.COL_VALUE:
                        prevname = index.sibling(index.row()-1, self.COL_VALNAME)
                        if prevname.isValid() and prevname.data(QtCore.Qt.DisplayRole) == 'Distribution':
                            return index.sibling(index.row()-1, self.COL_VARNAME)
                        else:
                            return index.sibling(index.row()-1, self.COL_VALUE)
                    elif index.column() == self.COL_VARNAME:
                        return index.sibling(index.row()-1, self.COL_DESC)
                    else:
                        return index.sibling(index.row(), index.column()-1)
        return super().moveCursor(cursorAction, modifiers)


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

    def replot(self, comp=None):
        ''' Update the plot with PDF of InptUncert comp '''
        if comp is not None:
            self.component = comp

        if self.component is not None:
            with plt.style.context({'font.size': 8}):
                self.figure.clf()
                ax = self.figure.add_subplot(1, 1, 1)
                x, y = self.component.pdf()
                ax.plot(x.magnitude, y)
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
        self.setItem(self.ROW_SAMPLES, self.COL_VALUE, gui_widgets.EditableTableItem(str(gui_common.settings.getSamples())))
        self.setItem(self.ROW_SEED, self.COL_NAME, gui_widgets.ReadOnlyTableItem('Random Seed'))
        self.setItem(self.ROW_SEED, self.COL_VALUE, gui_widgets.EditableTableItem(str(gui_common.settings.getRandomSeed())))
        self.setItem(self.ROW_SYMBOLIC, self.COL_NAME, gui_widgets.ReadOnlyTableItem('Symbolic GUM Solution Only'))
        chkbox = QtWidgets.QCheckBox()
        chkbox.stateChanged.connect(lambda x: self.valuechange(self.item(self.ROW_SYMBOLIC, self.COL_NAME)))
        self.setCellWidget(self.ROW_SYMBOLIC, self.COL_VALUE, chkbox)

        self.resizeColumnsToContents()
        self.itemChanged.connect(self.valuechange)

        height = self.verticalHeader().length() + self.horizontalHeader().height()
        self.setFixedHeight(height)

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
    calculate = QtCore.pyqtSignal()

    ''' Page for setting up input parameters '''
    def __init__(self, unccalc, parent=None):
        super().__init__(parent)
        self.unccalc = unccalc
        self.symbolicmode = False
        # Set up widgets
        self.funclist = FunctionTableWidget()
        self.meastable = MeasTableWidget(unccalc)
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

        funcbuttons.plusclicked.connect(self.funclist.addRow)
        funcbuttons.minusclicked.connect(self.funclist.remRow)
        corrbuttons.plusclicked.connect(self.corrtable.addRow)
        corrbuttons.minusclicked.connect(self.corrtable.remRow)
        self.funclist.resizerows.connect(self.panel.fixSize)
        self.meastable.resizerows.connect(self.panel.fixSize)
        self.corrtable.resizerows.connect(self.panel.fixSize)
        self.settings.changed.connect(self.settingchanged)

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

    def settingchanged(self, name, value):
        ''' A setting was changed in Settings table '''
        if name == 'Symbolic GUM Solution Only':
            self.symbolicmode = value
            self.meastable.setVisible(not value)
            self.panel.hide('Measured Values and Uncertainties', value)


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

    allitems = ['Summary', 'Comparison Plots', 'Expanded Uncertainties', 'Uncertainty Budget',
                'GUM Derivation', 'GUM Validity', 'Monte Carlo Distribution',
                'Monte Carlo Input Plots', 'Monte Carlo Convergence', 'Full Report']

    def __init__(self, uncCalc, parent=None):
        super().__init__(parent)
        self.outputSelect = QtWidgets.QComboBox()
        self.outputSelect.addItems(self.allitems)
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
                    'mchistparams': {'joint': self.outputMCsample.cmbjoint.currentText() == 'Joint PDF' and len(self.outdata.inputs) > 1,
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
            rptsetup['outplotparams']['plotargs']['intervalsmc'] = self.outputPlot.MCexpanded.get_covlist()
            rptsetup['outplotparams']['plotargs']['intervaltypemc'] = self.outputPlot.MCexpanded.MCtype.currentText()
            rptsetup['outplotparams']['plotargs']['intervalsgum'] = self.outputPlot.GUMexpanded.get_covlist()
            rptsetup['outplotparams']['plotargs']['intervaltypegum'] = 't' if 'Student' in self.outputPlot.GUMexpanded.GUMtype.currentText() else 'k'

        self.outputReportSetup.report = self.outdata.report_all(**rptsetup)  # Cache report for displaying/saving
        self.outputupdate()
        return self.outputReportSetup.report

    def outputupdate(self):
        ''' Update the output page based on widget settings. '''
        PLOT = 0
        TEXT = 1
        option = self.outputSelect.currentText()
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
            self.txtOutput.setReport(self.outdata.gum.report_derivation(solve=solve))

        elif option == 'Monte Carlo Convergence':
            self.outdata.mc.plot_converge(plot=self.fig, relative=self.outputMCconv.relative.isChecked())
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
                plotargs['intervalsmc'] = self.outputPlot.MCexpanded.get_covlist()
                plotargs['intervaltypemc'] = self.outputPlot.MCexpanded.MCtype.currentText().lower()
                plotargs['intervalsgum'] = self.outputPlot.GUMexpanded.get_covlist()
                plotargs['intervaltypegum'] = 't' if 'Student' in self.outputPlot.GUMexpanded.GUMtype.currentText() else 'k'

            assert self.outputPlot.GUMexpanded.GUMtype.findText('Student-t') != -1   # In case name changes from 'Student-t' it needs to be updated here

            if self.outputPlot.joint():
                self.outdata.plot_correlation(plot=self.fig, **plotargs)
            else:
                self.outdata.plot_pdf(plot=self.fig, **plotargs)
            self.canvas.draw_idle()

        elif option == 'Monte Carlo Input Plots':
            assert self.outputMCsample.cmbjoint.findText('Joint PDF') != -1  # Update below if names change
            assert self.outputMCsample.cmbscat.findText('Contours') != -1  # Update below if names change
            numinpts = len(self.outdata.inputs)
            joint = self.outputMCsample.cmbjoint.currentText() == 'Joint PDF' and numinpts > 1
            plotargs = {'bins': self.outputMCsample.bins.value(),
                        'points': self.outputMCsample.points.value(),
                        'contour': self.outputMCsample.cmbscat.currentText() == 'Contours',
                        'labelmode': 'desc' if self.outputMCsample.namedesc.isChecked() else 'name',
                        'inpts': self.outputMCsample.ilist.getSelectedIndexes(),
                        'cmap': gui_common.settings.getColormap('cmapcontour')}
            if joint:
                self.outdata.mc.plot_xscatter(plot=self.fig, **plotargs)
            else:
                self.outdata.mc.plot_xhists(plot=self.fig, **plotargs)
            self.canvas.draw_idle()

        elif option == 'GUM Validity':
            ndig = self.outputGUMvalid.ndig.value()
            r = self.outdata.report_validity(ndig=ndig)
            self.txtOutput.setReport(r)

        elif option == 'Monte Carlo Distribution':
            fidx = self.outputMCdist.cmbfunc.currentIndex()
            dist = self.outputMCdist.cmbdist.currentText()
            y = self.outdata.mc.samples(fidx).magnitude
            fitparams = plotting.fitdist(y, distname=dist, plot=self.fig, qqplot=True, bins=100, points=200)
            with suppress(IndexError):  # Raises if axes weren't added (maybe invalid samples)
                self.fig.axes[0].set_title('Distribution Fit')
                self.fig.axes[1].set_title('Probability Plot')
            self.fig.tight_layout()
            self.outputMCdist.update_label(fitparams)
            self.canvas.draw_idle()

        else:
            raise NotImplementedError

    def update(self, outdata, symonly=False):
        ''' Calculation run, update the page '''
        self.outdata = outdata
        funclist = [self.outdata.names[i] for i in range(self.outdata.nouts) if self.outdata.show[i]]
        if not symonly:
            if self.outputSelect.count() < len(self.allitems):
                # Switched from symbolic back to full
                self.outputSelect.blockSignals(True)
                self.outputSelect.clear()
                self.outputSelect.addItems(self.allitems)
                self.outputSelect.blockSignals(False)
            self.outputPlot.set_funclist(funclist)
            self.outputMCdist.set_funclist(funclist)
            self.outputMCsample.set_inptlist(self.outdata.inputs.names)
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
        else:
            self.outputSelect.blockSignals(True)
            self.outputSelect.clear()
            self.outputSelect.addItems(['GUM Derivation'])
            self.outputSelect.blockSignals(False)
            self.ctrlStack.setCurrentIndex(0)
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
        self.uncCalc.clearout()
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
        try:
            self.uncCalc.set_function(fdict['expr'], idx=row, name=fdict['name'], desc=fdict['desc'],
                                      show=fdict.get('report', True), outunits=fdict['unit'])
        except RecursionError:
            QtWidgets.QMessageBox.warning(self, 'Uncertainty Calculator', 'Circular reference in function definitions')
        else:
            self.uncCalc.add_required_inputs()
            self.pginput.meastable.filltable(self.uncCalc.inputs)
            self.pginput.corrtable.setVarNames(self.uncCalc.required_inputs)
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
        varnames = self.uncCalc.inputs.names
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
                                            units=str(self.uncCalc.get_inputvar(varname).units), **params)
                self.pginput.meastable.filltable()
                self.backbutton()

    def calculate(self):
        ''' Run the calculation '''
        msg = None
        if not (self.pginput.funclist.isValid() and self.pginput.meastable.isValid()):
            msg = 'Invalid input parameter!'

        elif len(self.uncCalc.model.exprs) < 1:
            msg = 'No functions to compute!'

        elif self.uncCalc.model.check_circular():
            msg = 'Circular reference in function definitions'

        elif not self.pginput.symbolicmode:
            try:
                self.uncCalc.model.check_dimensionality()
            except (TypeError, DimensionalityError, UndefinedUnitError) as e:
                msg = 'Units Error: {}'.format(e)
            except OffsetUnitCalculusError as e:
                badunit = re.findall(r'\((.+ )', str(e))[0].split()[0].strip(', ')
                msg = 'Ambiguous unit {}. Try "{}".'.format(badunit, 'delta_{}'.format(badunit))
            except RecursionError:
                msg = 'Error - possible circular reference in function definitions'

        if msg is None:
            try:
                self.uncCalc.calculate(mc=not self.pginput.symbolicmode)
            except OffsetUnitCalculusError as e:
                badunit = re.findall(r'\((.+ )', str(e))[0].split()[0].strip(', ')
                msg = 'Ambiguous unit {}. Try "{}".'.format(badunit, 'delta_{}'.format(badunit))
            except RecursionError:
                msg = 'Error - possible circular reference in function definitions'
            except (TypeError, ValueError):
                msg = 'Error computing solution!'

        if msg is None:
            self.stack.setCurrentIndex(self.PG_OUTPUT)
            self.pgoutput.update(self.uncCalc.out, symonly=self.pginput.symbolicmode)
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
        self.pginput.funclist.setFunclist(self.uncCalc.model)
        self.pginput.corrtable.setVarNames(self.uncCalc.inputs.names)
        self.pginput.settings.setvalue('Monte Carlo Samples', self.uncCalc.inputs.nsamples)
        self.pginput.settings.setvalue('Random Seed', self.uncCalc.inputs.seed)
        self.pginput.description.setPlainText(str(self.uncCalc.longdescription))

        if len(self.uncCalc.inputs.corr_list) > 0:
            for v1, v2, c in self.uncCalc.inputs.corr_list:
                if c != 0.:
                    self.pginput.corrtable.addRow(v1, v2, c)
            self.pginput.corrtable.blockSignals(False)
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
        fname, _ = QtWidgets.QFileDialog.getSaveFileName(caption='Select file to save', directory=self.openconfigfolder, filter='CSV (*.csv)')
        if fname:
            self.uncCalc.save_samples(fname, 'csv')

    def save_samples_npz(self):
        ''' Save Monte-Carlo samples to NPZ (compressed numpy) file.

            Load into python using numpy.load() to return a dictionary with
            'samples' and 'hdr' keys.
        '''
        fname, _ = QtWidgets.QFileDialog.getSaveFileName(caption='Select file to save', directory=self.openconfigfolder, filter='Numpy NPZ (*.npz)')
        if fname:
            self.uncCalc.save_samples(fname, 'npz')
