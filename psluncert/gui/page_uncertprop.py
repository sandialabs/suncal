''' Page for propagating uncertainty calculations '''

import os
import sympy
import numpy as np

from PyQt5 import QtWidgets, QtGui, QtCore
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

from .. import uncertainty as uc
from .. import uparser
from .. import output
from .. import customdists
from . import gui_common
from . import gui_widgets
from . import page_data


class FunctionTableWidget(QtWidgets.QTableWidget):
    ''' Function definition widget '''
    funcchanged = QtCore.pyqtSignal(int, dict)
    funcremoved = QtCore.pyqtSignal(int)
    orderchange = QtCore.pyqtSignal()
    COL_NAME = 0
    COL_EXPR = 1
    COL_DESC = 2
    COL_REPT = 3

    def __init__(self, parent=None):
        super(FunctionTableWidget, self).__init__(parent)
        self.setColumnCount(4)
        self.setDragEnabled(True)
        self.setAcceptDrops(True)
        self.viewport().setAcceptDrops(True)
        self.setDragDropOverwriteMode(False)
        self.setDropIndicatorShown(True)
        self.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.setDragDropMode(QtWidgets.QAbstractItemView.InternalMove)
        self.setItemDelegate(gui_widgets.SympyDelegate())

        self.clear()
        self.addRow()
        self.cellChanged.connect(self.itemEdit)

    def clear(self):
        ''' Override clear to reset things correctly. '''
        self.setRowCount(0)
        self.setHorizontalHeaderLabels(['Name', 'Expression',  'Description', 'Report?'])
        self.resizeColumnsToContents()
        self.setColumnWidth(self.COL_EXPR, 300)
        self.setColumnWidth(self.COL_DESC, 400)

    def setFunclist(self, funclist):
        ''' Set a list of functions '''
        self.blockSignals(True)
        self.clear()
        self.blockSignals(False)
        for row, f in enumerate(funclist):
            self.addRow()
            # Let signals populate the other tables
            self.setItem(row, self.COL_NAME, gui_widgets.EditableTableItem(f.name))
            self.setItem(row, self.COL_EXPR, gui_widgets.EditableTableItem(str(f.function)))
            self.setItem(row, self.COL_DESC, gui_widgets.EditableTableItem(f.desc))
            chk = QtWidgets.QCheckBox()
            chk.setCheckState(QtCore.Qt.Checked)
            chk.stateChanged.connect(lambda x, row=row, col=self.COL_REPT: self.itemEdit(row, col))
            self.setCellWidget(row, self.COL_REPT, chk)

    def addRow(self):
        ''' Add an empty row to function table '''
        self.blockSignals(True)
        rows = self.rowCount()
        self.setRowCount(rows + 1)
        self.setItem(rows, self.COL_NAME, gui_widgets.EditableTableItem())
        self.setItem(rows, self.COL_EXPR, gui_widgets.EditableTableItem())
        self.setItem(rows, self.COL_DESC, gui_widgets.EditableTableItem())
        chk = QtWidgets.QCheckBox()
        chk.setCheckState(QtCore.Qt.Checked)
        chk.stateChanged.connect(lambda x, row=rows, col=self.COL_REPT: self.itemEdit(row, col))
        self.setCellWidget(rows, self.COL_REPT, chk)
        self.blockSignals(False)

    def remRow(self):
        ''' Remove row at index from function table '''
        idx = self.selectedItems()
        if len(idx) > 0:
            idx = idx[0].row()
            self.removeRow(idx)
            self.funcremoved.emit(idx)

    def itemText(self, row, col):
        ''' Get text of item, if item exists. '''
        if self.item(row, col) is not None:
            return self.item(row, col).text()
        else:
            return ''

    def itemEdit(self, row, col):
        ''' A cell was changed. '''
        self.blockSignals(True)
        ok = False
        name = self.item(row, self.COL_NAME).text()
        expr = self.item(row, self.COL_EXPR).data(gui_widgets.ROLE_ORIGDATA)
        if expr is None:
            expr = self.item(row, self.COL_EXPR).text()

        try:
            fn = uparser.check_expr(expr, name=name)
        except ValueError:
            fn = None
        else:
            ok = True

        try:
            sname = uparser.check_expr(name)
        except ValueError:
            sname = None
            ok = False

        if self.item(row, self.COL_EXPR) is not None:
            if fn is not None:
                self.item(row, self.COL_EXPR).setBackground(QtGui.QBrush(gui_common.COLOR_OK))
                self.item(row, self.COL_EXPR).setData(gui_widgets.ROLE_SYMPY, fn)
            else:
                self.item(row, self.COL_EXPR).setBackground(QtGui.QBrush(gui_common.COLOR_INVALID))
                self.item(row, self.COL_EXPR).setData(gui_widgets.ROLE_SYMPY, None)
                self.clearSelection()

        if col == self.COL_NAME and self.item(row, self.COL_NAME) is not None:
            if isinstance(sname, sympy.symbol.Symbol):
                self.item(row, self.COL_NAME).setBackground(QtGui.QBrush(gui_common.COLOR_OK))
                self.item(row, self.COL_NAME).setData(gui_widgets.ROLE_SYMPY, sname)
            else:
                self.item(row, self.COL_NAME).setBackground(QtGui.QBrush(gui_common.COLOR_INVALID))
                self.item(row, self.COL_NAME).setData(gui_widgets.ROLE_SYMPY, None)
                self.clearSelection()

        self.blockSignals(False)
        if ok:
            func = {'name': name,
                    'desc': self.itemText(row, self.COL_DESC),
                    'expr': expr,
                    }
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
            for row_idx in range(self.rowCount())[::-1]:
                if (self.item(row_idx, 1) is None or self.item(row_idx, 1).text() == ''):
                    self.removeRow(row_idx)

        super().dropEvent(event)
        self.blockSignals(False)
        self.orderchange.emit()

    def is_below(self, pos, index):
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
               self.item(row, self.COL_NAME).background().color() == gui_common.COLOR_INVALID):
                return False
        return True

    def contextMenuEvent(self, event):
        item = self.itemAt(event.pos())
        if item:
            row = item.row()
            name = self.item(row, self.COL_NAME).text()
            expr = self.item(row, self.COL_EXPR).data(gui_widgets.ROLE_ORIGDATA)
            if expr is None:
                expr = self.item(row, self.COL_EXPR).text()
            try:
                fn = uparser.check_expr(expr, name=name)
            except ValueError:
                return

            menu = QtWidgets.QMenu()
            solve = menu.addMenu('Solve for')
            for symbol in fn.free_symbols:
                act = solve.addAction(str(symbol))
                act.triggered.connect(lambda x, fn=fn, name=name, var=str(symbol), row=row: self.solvefor(fn, name, var, row))
            menu.exec(event.globalPos())

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
        except NotImplementedError:  # Some solves crash with this is sympy doesn't know what to do
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
        mbox.setText('Set model equation?' + '<br><br>' + output.tex_to_html(sympy.latex(sympy.Eq(svar, solution))))
        mbox.setInformativeText('Measured Quantity entries may be removed.<br><br>Note: Do not use for reversing the calculation to determine required uncertainty of a measured quantity.')
        mbox.setStandardButtons(QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)
        ok = mbox.exec_()
        if ok == QtWidgets.QMessageBox.Yes:
            expr = str(solution)
            if fn is not None:
                self.item(row, self.COL_NAME).setText(var)
                self.item(row, self.COL_EXPR).setText(expr)
                self.item(row, self.COL_EXPR).setData(gui_widgets.ROLE_ORIGDATA, expr)
                self.item(row, self.COL_NAME).setData(gui_widgets.ROLE_ORIGDATA, var)


class InputButtonSet(QtWidgets.QWidget):
    ''' Button Widget for each line in InputTreeWidget '''
    addrem = QtCore.pyqtSignal()
    customize = QtCore.pyqtSignal()

    # CSS stylesheet for nice round buttons
    buttonstyle = '''QToolButton {border: 1px solid #8f8f91; border-radius: 8px;
                     background-color: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1, stop: 0 #f6f7fa, stop: 1 #dadbde);}

                     QToolButton:pressed {border: 2px solid #8f8f91; border-radius: 8px; border-width: 2px;
                     background-color: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1, stop: 0 #dadbde, stop: 1 #f6f7fa);}

                     QToolButton:checked {border: 2px solid #8f8f91; border-radius: 8px; border-width: 1px;
                     background-color: qlineargradient(x1: 0, y1: 1, x2: 0, y2: 0, stop: 0 #dadbde, stop: 1 #7c7c7c);}
                     '''

    def __init__(self):
        super(InputButtonSet, self).__init__()
        self.btn_addrem = QtWidgets.QToolButton()
        self.btn_custom = QtWidgets.QToolButton()
        self.btn_addrem.setText('+')
        self.btn_custom.setText(gui_common.CHR_ELLIPSIS)
        self.btn_addrem.setToolTip('Add uncertainty component')
        self.btn_custom.setToolTip('Customize uncertainty distribution')
        self.btn_addrem.clicked.connect(self.addrem)
        self.btn_custom.clicked.connect(self.customize)
        self.btn_custom.setCheckable(True)
        self.setStyleSheet(self.buttonstyle)

        layout = QtWidgets.QHBoxLayout()
        layout.setSpacing(3)
        layout.addWidget(self.btn_addrem)
        layout.addWidget(self.btn_custom)
        layout.addStretch()
        self.setLayout(layout)

    def setCustomChecked(self, checked=True):
        ''' Set check state of customize button '''
        self.btn_custom.setChecked(checked)

    def set_mode(self, mode='input'):
        ''' Set buttonset mode.
            mode = 'input':  Show Add button only
            mode = 'uncert': Show Remove and Customize buttons
            mode = 'inptunc': Show Add and Customize buttons
        '''
        assert mode in ['input', 'uncert', 'inptunc']
        if mode == 'input':
            self.btn_addrem.setText('+')
            self.btn_addrem.setToolTip('Add uncertainty component')
            self.btn_custom.setVisible(False)
        elif mode == 'uncert':
            self.btn_addrem.setText(gui_common.CHR_ENDASH)
            self.btn_addrem.setToolTip('Remove uncertainty component')
            self.btn_custom.setVisible(True)
        else:
            self.btn_addrem.setText('+')
            self.btn_addrem.setToolTip('Add uncertainty component')
            self.btn_custom.setVisible(True)


class InputTreeWidget(QtWidgets.QTreeWidget):
    ''' Tree Widget for editing input values and their uncertainty components '''
    COL_NAME = 0
    COL_MEAN = 1
    COL_STDU = 2
    COL_DEGF = 3
    COL_DESC = 4
    COL_BTNS = 5
    COL_CNT = 6

    customedit = QtCore.pyqtSignal(object)  # Show Custom distribution widget for this input/uncertainty
    hidecustom = QtCore.pyqtSignal()   # Hide custom distribution widget
    edited = QtCore.pyqtSignal()  # A value has changed

    def __init__(self):
        super(InputTreeWidget, self).__init__()
        self.clear()
        self.setColumnCount(self.COL_CNT)
        self.setHeaderItem(QtWidgets.QTreeWidgetItem(['Name', 'Mean', 'Standard\nUncertainty', 'Degrees\nFreedom', 'Description', '']))
        self.inputlist = []
        self.setEditTriggers(QtWidgets.QTreeWidget.NoEditTriggers)  # Only start editing manually
        self.itemDoubleClicked.connect(self.checkEdit)
        self.itemExpanded.connect(self.update_fonts)
        self.itemCollapsed.connect(self.update_fonts)
        self.itemChanged.connect(self.itemchange)
        self.currentItemChanged.connect(self.rowchange)
        self.setItemDelegate(gui_widgets.SympyDelegate())
        self.setColumnWidth(self.COL_DESC, 250)

    def checkEdit(self, item, column):
        ''' Check if this item/column is editable, and start editor if it is. '''
        if self.editable(item, column):
            self.editItem(item, column)

    def editable(self, item, column):
        ''' Check if item/column is editable '''
        obj = item.data(0, gui_widgets.ROLE_OBJ)
        if column == self.COL_DESC:
            return True

        if column == self.COL_NAME and isinstance(obj, uc.InputUncert):
            return True

        if column == self.COL_MEAN and isinstance(obj, uc.InputVar):
            return True

        if column == self.COL_DEGF:
            if (isinstance(obj, uc.InputUncert) or
               (isinstance(obj, uc.InputVar) and not item.isExpanded() and (len(obj.uncerts) == 0 or (len(obj.uncerts) == 1 and obj.uncerts[0].name.startswith('u('))))):
                return True

        if column in [self.COL_STDU]:
            if isinstance(obj, uc.InputVar):
                if len(obj.uncerts) == 0:
                    return True
                if not item.isExpanded() and (len(obj.uncerts) == 1 and obj.uncerts[0].name.startswith('u(') and (not hasattr(obj.uncerts[0], 'customized') or obj.uncerts[0].customized is False)):
                    return True

            if isinstance(obj, uc.InputUncert):
                if not hasattr(obj, 'customized') or obj.customized is False:
                    return True
        return False

    def update_fonts(self):
        ''' Set font to bold if the item is editable. '''
        def set_bold(item, column, bold):
            font = item.font(column)
            font.setBold(bold)
            item.setFont(column, font)
        it = QtWidgets.QTreeWidgetItemIterator(self)
        self.blockSignals(True)
        while it.value():  # Of course they couldn't make it a real Python iterator...
            item = it.value()
            for col in range(self.COL_CNT):
                set_bold(item, col, self.editable(item, col))
            obj = item.data(0, gui_widgets.ROLE_OBJ)
            if item.isExpanded():
                self.itemWidget(item, self.COL_BTNS).set_mode('input')
            elif isinstance(obj, uc.InputVar):
                self.itemWidget(item, self.COL_BTNS).set_mode('inptunc')
                self.itemWidget(item, self.COL_BTNS).setCustomChecked(hasattr(obj.uncerts[0], 'customized'))
            else:
                self.itemWidget(item, self.COL_BTNS).setCustomChecked(hasattr(obj, 'customized'))
            it += 1
        self.blockSignals(False)

    def customchanged(self, params=None):
        ''' Custom uncertainty widget was changed. Update the std. uncertainty for the row.

            Parameters
            ----------
            params: dict
                Dictionary containing other parameters to update. Currently only 'mean' value
                if it should be set from the new distribution.
        '''
        self.blockSignals(True)
        it = QtWidgets.QTreeWidgetItemIterator(self)
        while it.value():  # Of course they couldn't make it a real Python iterator...
            item = it.value()
            obj = item.data(0, gui_widgets.ROLE_OBJ)
            item.setText(self.COL_STDU, str(obj.stdunc()) if isinstance(obj, uc.InputVar) else str(obj.std()))
            item.setText(self.COL_DEGF, str(obj.degf()) if isinstance(obj, uc.InputVar) else str(obj.degf))

            if params and isinstance(obj, uc.InputVar) and params.get('comp', None) in obj.uncerts:
                item.setText(self.COL_MEAN, str(params['mean']))
                obj.set_nom(params['mean'])

            it += 1
        self.blockSignals(False)

    def update(self, inputlist=None):
        ''' Set list of inputs to show. inputlist is list of InputVar objects. If omitted, tree will be
            refreshed with current inputlist.
        '''
        self.blockSignals(True)
        if inputlist is not None:
            self.inputlist = inputlist
        self.clear()

        top = self.invisibleRootItem()
        for row, inpt in enumerate(self.inputlist):
            collapse_unc = len(inpt.uncerts) == 0 or (len(inpt.uncerts) == 1 and inpt.uncerts[0].name.startswith('u('))

            if len(inpt.uncerts) == 0:
                inpt.add_comp(name='u({})'.format(inpt.name), std=1.0)

            newitem = QtWidgets.QTreeWidgetItem(top, [inpt.name, format(inpt.nom), format(inpt.stdunc()), format(inpt.degf()), inpt.desc, ''])
            newitem.setFlags(newitem.flags() | QtCore.Qt.ItemIsEditable)
            newitem.setData(0, gui_widgets.ROLE_OBJ, inpt)
            newitem.setData(0, gui_widgets.ROLE_VALID, True)
            newitem.setData(0, gui_widgets.ROLE_SYMPY, inpt.get_symbol())
            btn = InputButtonSet()
            btn.set_mode('input' if not collapse_unc else 'inptunc')
            if collapse_unc and len(inpt.uncerts) > 0 and hasattr(inpt.uncerts[0], 'customized') and inpt.uncerts[0].customized:
                btn.setCustomChecked(True)
            btn.addrem.connect(lambda x=inpt: self.add_comp(x))
            btn.customize.connect(lambda x=newitem: self.customizeuncert(x))

            self.setItemWidget(newitem, self.COL_BTNS, btn)
            for uidx, unc in enumerate(inpt.uncerts):
                newunc = QtWidgets.QTreeWidgetItem(newitem, [unc.name, '', format(unc.std()), format(unc.degf), unc.desc])
                newunc.setData(0, gui_widgets.ROLE_OBJ, unc)
                newunc.setData(0, gui_widgets.ROLE_VALID, True)
                newunc.setData(0, gui_widgets.ROLE_SYMPY, unc.get_symbol())
                newunc.setFlags(newunc.flags() | QtCore.Qt.ItemIsEditable)
                btn = InputButtonSet()
                btn.set_mode('uncert')
                btn.addrem.connect(lambda x=inpt, y=uidx: self.rem_comp(x, y))
                btn.customize.connect(lambda x=newunc: self.customizeuncert(x))
                if hasattr(unc, 'customized') and unc.customized:
                    btn.setCustomChecked(True)
                self.setItemWidget(newunc, self.COL_BTNS, btn)
            newitem.setExpanded(not collapse_unc)
        self.update_fonts()
        self.blockSignals(False)

    def customizeuncert(self, item):
        ''' Customize uncertainty button was clicked for this item '''
        self.setCurrentItem(item)
        obj = item.data(0, gui_widgets.ROLE_OBJ)
        if isinstance(obj, uc.InputVar):
            if len(obj.uncerts) == 0:
                obj.add_comp('u({})'.format(obj.name))  # Add the default uncert if it's not there
            obj = obj.uncerts[0]

        if hasattr(obj, 'customized') and obj.customized:
            delattr(obj, 'customized')
            obj.set_dist('normal')
            self.hidecustom.emit()
            self.update_fonts()
            return

        obj.customized = True
        self.customedit.emit(obj)
        self.update_fonts()

    def rowchange(self, item, olditem):
        ''' Selected row changed. Show custom dist if customized. '''
        obj = item.data(0, gui_widgets.ROLE_OBJ)
        if isinstance(obj, uc.InputVar):
            if len(obj.uncerts) == 1:
                obj = obj.uncerts[0]
            else:
                self.hidecustom.emit()
                return

        if hasattr(obj, 'customized') and obj.customized:
            self.customedit.emit(obj)
        else:
            self.hidecustom.emit()

    def add_comp(self, inpt):
        ''' Add new component to input '''
        inpt.add_comp('u{}({})'.format(len(inpt.uncerts)+1, inpt.name))
        self.update()

    def rem_comp(self, inpt, idx):
        ''' Remove component from input with index idx '''
        inpt.rem_comp(idx)
        if len(inpt.uncerts) == 0:
            inpt.add_comp('u({})'.format(inpt.name))  # Put back the hidden/default component
        self.update()

    def itemchange(self, item, column):
        ''' An item in the tree was changed by the user. '''
        COLOR = {True: gui_common.COLOR_OK, False: gui_common.COLOR_INVALID}
        self.blockSignals(True)
        obj = item.data(0, gui_widgets.ROLE_OBJ)
        text = item.text(column)
        status = True

        if column == self.COL_MEAN and isinstance(obj, uc.InputVar):
            status = obj.set_nom(text)
            if status:
                item.setText(column, format(obj.nom))
                for i in range(item.childCount()):
                    uitem = item.child(i)
                    uitem.data(0, gui_widgets.ROLE_OBJ).updateparams()
                    uitem.setData(self.COL_STDU, QtCore.Qt.DisplayRole, format(uitem.data(0, gui_widgets.ROLE_OBJ).std()))
                if item.childCount() > 0:
                    obj.uncerts[0].updateparams()
                item.setData(self.COL_STDU, QtCore.Qt.DisplayRole, format(obj.stdunc()))  # And combined stdunc
                self.blockSignals(False)
                self.edited.emit()
                self.blockSignals(True)

        elif column == self.COL_DESC:
            obj.desc = text

        elif column == self.COL_NAME and isinstance(obj, uc.InputUncert):
            obj.name = text
            try:
                sname = uparser.check_expr(text)
            except ValueError:
                sname = None
            item.setData(0, gui_widgets.ROLE_SYMPY, sname)

        elif column == self.COL_STDU:
            if isinstance(obj, uc.InputVar):
                inpt = obj
                if len(obj.uncerts) == 0:
                    obj.add_comp('u({})'.format(obj.name))  # Add default/hidden component
                else:
                    item.child(0).setText(column, text)
                obj = obj.uncerts[0]   # Point to default/collapsed uncertainty component
            else:
                inpt = item.parent().data(0, gui_widgets.ROLE_OBJ)

            obj.userargs.clear()
            obj.userargs['std'] = text
            status = obj.updateparams()
            if status:
                objstd = obj.std()
                item.setText(column, format(objstd) if np.isfinite(objstd) else '0')  # Show calculated value, not entered value
                if item.parent():     # Propagate change to up to InputVar
                    item.parent().setText(column, format(inpt.stdunc()))

        elif column == self.COL_DEGF:
            if isinstance(obj, uc.InputVar):
                if len(obj.uncerts) == 0:
                    obj.add_comp('u({})'.format(obj.name))  # Add default/hidden component
                else:
                    item.child(0).setText(column, text)
                obj = obj.uncerts[0]   # Point to default/hidden uncertainty component
            try:
                df = float(text)
            except ValueError:
                item.setText(column, format(obj.degf))
                status = False
            else:
                obj.degf = df
                status = obj.updateparams()
                objstd = obj.std()
                item.setText(self.COL_STDU, format(objstd) if np.isfinite(objstd) else '0')  # Show calculated value, not entered value
                if item.parent(): # Propagate change to up to InputVar or back to stdunc.
                    item.parent().setText(column, format(item.parent().data(0, gui_widgets.ROLE_OBJ).degf()))
                    item.parent().setText(self.COL_STDU, format(item.parent().data(0, gui_widgets.ROLE_OBJ).stdunc()))
                self.blockSignals(False)
                self.edited.emit()
                self.blockSignals(True)

        item.setBackground(column, COLOR[status])
        item.setData(0, gui_widgets.ROLE_VALID, status)
        if not status:
            self.clearSelection()
        self.blockSignals(False)

    def isValid(self):
        ''' Return True if all entries are valid (not red) '''
        it = QtWidgets.QTreeWidgetItemIterator(self)
        while it.value():  # Of course they couldn't make it a real Python iterator...
            item = it.value()
            if not item.data(0, gui_widgets.ROLE_VALID):
                return False
            it += 1
        return True


class DistributionWidget(QtWidgets.QWidget):
    ''' Widget for editing and plotting custom distributions '''
    changed = QtCore.pyqtSignal(object)

    def __init__(self, calc=None):
        super(DistributionWidget, self).__init__()
        self.unccalc = calc
        self.title = QtWidgets.QLabel('Customized Distribution for Monte-Carlo')
        self.param = QtWidgets.QLabel('Uncertainty:')
        self.table = DistributionEditTable()
        self.table.changed.connect(self.tablechange)
        self.fig = Figure(figsize=(2, 2))
        self.canvas = FigureCanvas(self.fig)
        self.btnImport = QtWidgets.QToolButton()
        self.btnImport.setToolTip('Import Distribution From...')
        self.btnImport.setIconSize(QtCore.QSize(28, 28))
        self.btnImport.setIcon(gui_common.load_icon('loaddist'))

        layout = QtWidgets.QVBoxLayout()
        hlayout = QtWidgets.QHBoxLayout()
        hlayout.addWidget(self.title)
        hlayout.addStretch()
        hlayout.addWidget(self.btnImport)
        layout.addLayout(hlayout)
        layout.addWidget(self.param)
        layout.addWidget(self.table, stretch=10)
        layout.addWidget(self.canvas, stretch=15)
        sp = self.sizePolicy()
        sp.setRetainSizeWhenHidden(True)
        self.setSizePolicy(sp)
        self.setLayout(layout)
        self.setMinimumWidth(400)
        self.btnImport.clicked.connect(self.importdist)

    def clear(self):
        ''' Clear the table '''
        self.table.clear()

    def set_component(self, comp):
        ''' Set the uncertainty component to display '''
        if comp is None and self.table.comp is None:
            return

        if comp is not None:
            self.param.setText('Uncertainty: {}'.format(comp.name))
        self.table.set_component(comp)
        self.plotdist()

    def tablechange(self):
        ''' Table was changed. Update the plot and emit changed signal. '''
        self.plotdist()
        self.changed.emit(None)

    def plotdist(self):
        ''' Update the plot with PDF of distribution '''
        comp = self.table.comp
        self.fig.clf()
        ax = self.fig.add_subplot(1, 1, 1)
        x, y = comp.pdf()
        ax.plot(x, y)
        ax.set_title('$'+comp.name+'$')
        self.fig.tight_layout()
        self.canvas.setVisible(True)
        self.canvas.draw_idle()

    def importdist(self):
        dlg = page_data.DistributionSelectWidget(project=self.unccalc.project)
        ok = dlg.exec_()
        if ok:
            distname, distargs, mean = dlg.get_dist()
            self.table.comp.userargs = distargs
            self.table.comp.set_dist(distname)
            self.table.comp.userargs = distargs
            self.table.set_component(None)
            params = {'comp': self.table.comp, 'mean': mean}
            if 'df' in distargs:
                params['df'] = distargs['df']
            self.changed.emit(params)
            self.plotdist()


class DistributionEditTable(QtWidgets.QTableWidget):
    ''' Table for editing parameters of an uncertainty distribution '''
    changed = QtCore.pyqtSignal()

    def __init__(self):
        super(DistributionEditTable, self).__init__()
        self.cellChanged.connect(self.valuechanged)
        self.comp = None

    def clear(self):
        ''' Clear and reset the table '''
        super(DistributionEditTable, self).clear()
        self.setColumnCount(3)
        self.setHorizontalHeaderLabels(['Parameter', 'Value', ''])
        self.resizeColumnsToContents()
        self.setColumnWidth(0, 120)
        self.setColumnWidth(1, 120)
        self.setColumnWidth(2, 30)

    def set_component(self, comp):
        ''' Set the table with rows for this distribution type '''
        self.blockSignals(True)
        self.clear()
        if comp is not None:
            self.comp = comp
        self.dist = QtWidgets.QComboBox()
        self.dist.addItems(gui_common.settings.getDistributions())
        self.dist.setCurrentIndex(self.dist.findText(self.comp.distname))
        if self.dist.currentText() == '':
            self.dist.addItem(self.comp.distname)
            self.dist.setCurrentIndex(self.dist.count()-1)

        self.kconf = QtWidgets.QComboBox()
        self.kconf.addItems(['k', 'confidence'])

        self.setRowCount(1)
        self.setItem(0, 0, QtWidgets.QTableWidgetItem('Distribution'))
        self.setCellWidget(0, 1, self.dist)
        self.btnhlp = QtWidgets.QToolButton()
        self.btnhlp.setText('?')
        self.btnhlp.clicked.connect(self.showhelp)
        self.setCellWidget(0, 2, self.btnhlp)

        if self.comp.distname in ['normal', 't']:
            self.setRowCount(3)
            self.setItem(1, 0, gui_widgets.ReadOnlyTableItem('Uncertainty'))
            self.setItem(1, 1, gui_widgets.EditableTableItem(format(self.comp.userargs.get('unc', self.comp.args.get('std', 1)))))
            self.setCellWidget(2, 0, self.kconf)
            if 'conf' in self.comp.userargs:
                self.kconf.setCurrentIndex(1)
                self.setItem(2, 1, gui_widgets.EditableTableItem(format(self.comp.userargs['conf'])))
            else:
                self.kconf.setCurrentIndex(0)
                self.setItem(2, 1, gui_widgets.EditableTableItem(format(self.comp.userargs.get('k', 1))))
        elif self.comp.distname == 'histogram':
            pass  # Don't add anything to table

        else:
            self.setRowCount(len(self.comp.args.keys())+1)
            for row, arg in enumerate(sorted(self.comp.userargs.keys())):
                self.setItem(row+1, 0, gui_widgets.ReadOnlyTableItem(arg))
                self.setItem(row+1, 1, gui_widgets.EditableTableItem(format(self.comp.userargs[arg])))
        self.dist.currentIndexChanged.connect(self.change_dist)
        self.kconf.currentIndexChanged.connect(self.valuechanged)
        self.blockSignals(False)

    def valuechanged(self):
        ''' A value was edited. '''
        args = {}
        assert self.dist.findText('normal') != -1  # Update below if names change
        assert self.dist.findText('t') != -1  # Update below if names change
        assert self.kconf.findText('k') != -1  # Update below if names change
        if self.dist.currentText() in ['normal', 't']:
            args['unc'] = self.item(1, 1).text()
            kval = self.item(2, 1).text()
            if self.kconf.currentText() == 'k':
                args['k'] = kval
            else:
                args['conf'] = kval
        else:
            for row in range(self.rowCount()-1):
                name = self.item(row+1, 0).text()
                val = self.item(row+1, 1).text()
                args[name] = val

        self.comp.userargs.clear()
        self.comp.userargs.update(args)
        status = self.comp.updateparams()
        if status is False and self.currentRow() > -1:
            self.item(self.currentRow(), 1).setBackground(gui_common.COLOR_INVALID)
            self.clearSelection()
        else:
            if self.currentRow() > -1 and self.item(self.currentRow(), 1):
                self.item(self.currentRow(), 1).setBackground(gui_common.COLOR_OK)
            self.changed.emit()

    def change_dist(self, index=None):
        ''' Distribution type has changed. Update fields. '''
        dist = self.dist.currentText()
        self.comp.set_dist(dist)
        self.set_component(None)
        if 'conf' in self.comp.userargs:
            self.kconf.setCurrentIndex(1)
        else:
            self.kconf.setCurrentIndex(0)
        self.changed.emit()

    def showhelp(self):
        ''' Show help string for this distribution '''
        p = PopupHelp(self.comp.helpstr)
        p.exec_()


class PopupHelp(QtWidgets.QDialog):
    ''' Show a floating dialog window with a text message '''
    def __init__(self, text):
        super(PopupHelp, self).__init__()
        self.setGeometry(600, 200, 600, 600)
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


class CorrelationTableWidget(QtWidgets.QWidget):
    ''' Widget for setting correlations between inputs '''
    cellChanged = QtCore.pyqtSignal(int, int)

    def __init__(self):
        super(CorrelationTableWidget, self).__init__()
        self.table = QtWidgets.QTableWidget()
        self.label = QtWidgets.QLabel('Correlations:')
        self.btnadd = QtWidgets.QToolButton()
        self.btnrem = QtWidgets.QToolButton()
        self.btnadd.setText('+')
        self.btnadd.setToolTip('Add correlation')
        self.btnrem.setText(gui_common.CHR_ENDASH)
        self.btnrem.setToolTip('Remove correlation')
        self.btnadd.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        self.btnrem.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        self.btnadd.clicked.connect(self.addRow)
        self.btnrem.clicked.connect(self.remRow)
        self.table.cellChanged.connect(self.cellChanged)
        self.varnames = []
        self.reset()

        layout = QtWidgets.QVBoxLayout()
        hlayout = QtWidgets.QHBoxLayout()
        hlayout.addWidget(self.label)
        hlayout.addWidget(self.btnadd)
        hlayout.addWidget(self.btnrem)
        layout.addLayout(hlayout)
        layout.addWidget(self.table)
        self.setLayout(layout)

    def reset(self):
        ''' Clear and reset the table '''
        self.table.setColumnCount(3)
        self.table.setRowCount(0)
        self.table.setHorizontalHeaderLabels(['Input 1', 'Input 2', 'Correlation\nCoefficient'])
        self.table.verticalHeader().hide()
        self.varnames = []
        self.table.resizeColumnsToContents()

    def setVarNames(self, names):
        ''' Set names of available variables '''
        self.table.blockSignals(True)
        self.varnames = names
        for row in range(self.table.rowCount()):
            prev1 = self.table.cellWidget(row, 0).currentText()
            prev2 = self.table.cellWidget(row, 1).currentText()
            self.table.cellWidget(row, 0).clear()
            self.table.cellWidget(row, 1).clear()
            self.table.cellWidget(row, 0).addItems(names)
            self.table.cellWidget(row, 1).addItems(names)
            self.table.cellWidget(row, 0).setCurrentIndex(self.table.cellWidget(row, 0).findText(prev1))
            self.table.cellWidget(row, 1).setCurrentIndex(self.table.cellWidget(row, 1).findText(prev2))
        self.table.blockSignals(False)

    def remRow(self):
        ''' Remove selected row from correlation table '''
        self.table.removeRow(self.table.currentRow())
        self.cellChanged.emit(0, 0)

    def addRow(self):
        ''' Add a row to the table. '''
        row = self.table.rowCount()
        self.table.blockSignals(True)
        self.table.insertRow(row)
        v1 = QtWidgets.QComboBox()
        v1.addItems(self.varnames)
        v1.setMaximumWidth(50)
        v2 = QtWidgets.QComboBox()
        v2.setMaximumWidth(50)
        v2.addItems(self.varnames)
        self.table.setCellWidget(row, 0, v1)
        self.table.setCellWidget(row, 1, v2)
        val = QtWidgets.QTableWidgetItem('0')
        val.setFlags(QtCore.Qt.ItemIsEditable | QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsSelectable)
        self.table.setItem(row, 2, val)
        v1.currentIndexChanged.connect(lambda idx, r=row, c=0: self.cmbchange(r, c))
        v2.currentIndexChanged.connect(lambda idx, r=row, c=1: self.cmbchange(r, c))
        self.table.blockSignals(False)

    def setRow(self, row, v1, v2, cor):
        ''' Set values to an existing row '''
        self.table.cellWidget(row, 0).setCurrentIndex(self.table.cellWidget(row, 0).findText(v1))
        self.table.cellWidget(row, 1).setCurrentIndex(self.table.cellWidget(row, 1).findText(v2))
        self.table.item(row, 2).setText(str(cor))

    def getRow(self, row):
        ''' Get values for one row in the table. Returns (input1, input2, corrleation) '''
        v1 = self.table.cellWidget(row, 0).currentText()
        v2 = self.table.cellWidget(row, 1).currentText()
        val = self.table.item(row, 2).text()
        return v1, v2, val

    def cmbchange(self, row, col):
        ''' Combobox (input name) changed. '''
        self.cellChanged.emit(row, col)


class SettingsWidget(QtWidgets.QWidget):
    ''' Widget for Monte-Carlo settings '''
    def __init__(self, parent=None):
        super(SettingsWidget, self).__init__(parent)
        self.txtSamples = QtWidgets.QLineEdit(str(gui_common.settings.getSamples()))
        self.txtSamples.setValidator(QtGui.QIntValidator(0, 1E8))
        self.txtSamples.validator().setBottom(0)
        self.txtSeed = QtWidgets.QLineEdit(str(gui_common.settings.getRandomSeed()))
        layout = QtWidgets.QFormLayout()
        layout.addRow('Monte Carlo Samples', self.txtSamples)
        layout.addRow('Random Seed', self.txtSeed)
        self.setLayout(layout)


class PageInput(QtWidgets.QWidget):
    calculate = QtCore.pyqtSignal()
    ''' Page for setting up input parameters '''
    def __init__(self, unccalc=None, parent=None):
        super(PageInput, self).__init__(parent)

        # Set up widgets
        self.btnAdd = QtWidgets.QToolButton()
        self.btnAdd.setText('+')
        self.btnAdd.setToolTip('Add model equation')
        self.btnRem = QtWidgets.QToolButton()
        self.btnRem.setText(gui_common.CHR_ENDASH)
        self.btnRem.setToolTip('Remove model equation')
        self.btnAdd.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        self.btnRem.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        self.funclist = FunctionTableWidget()
        self.inputtree = InputTreeWidget()
        self.corrtable = CorrelationTableWidget()
        self.corrtable.setVisible(False)
        self.distedit = DistributionWidget(unccalc)
        self.distedit.setVisible(False)
        self.btnCalc = QtWidgets.QPushButton('Calculate')
        self.description = QtWidgets.QPlainTextEdit()
        self.MCsettings = SettingsWidget()

        self.tab = QtWidgets.QTabWidget()
        self.tab.addTab(self.description, 'Notes')
        self.tab.addTab(self.corrtable, 'Correlations')
        self.tab.addTab(self.MCsettings, 'Settings')
        self.tab.setFixedWidth(400)

        self.btnAdd.clicked.connect(self.funclist.addRow)
        self.btnRem.clicked.connect(self.funclist.remRow)
        self.btnCalc.clicked.connect(self.calculate)
        self.inputtree.customedit.connect(self.customize)
        self.inputtree.hidecustom.connect(self.hidecustom)
        self.inputtree.edited.connect(self.updatecustom)
        self.distedit.changed.connect(self.inputtree.customchanged)

        layout = QtWidgets.QVBoxLayout()
        blayout = QtWidgets.QHBoxLayout()
        blayout.addWidget(QtWidgets.QLabel('Measurement Model:'))
        blayout.addWidget(self.btnAdd)
        blayout.addWidget(self.btnRem)
        blayout.addStretch()
        layout.addLayout(blayout)
        layout.addWidget(self.funclist)
        layout.addWidget(QtWidgets.QLabel('Measured Quantities:'))

        clayout = QtWidgets.QHBoxLayout()
        hlayout = QtWidgets.QVBoxLayout()
        hlayout.addWidget(self.inputtree, stretch=1)
        clayout.addLayout(hlayout, stretch=1)
        clayout.addWidget(self.tab)
        layout.addLayout(clayout, stretch=10)
        slayout = QtWidgets.QHBoxLayout()
        slayout.addStretch()
        slayout.addWidget(self.btnCalc)
        layout.addLayout(slayout)
        self.setLayout(layout)

    def customize(self, uncert):
        ''' Show Custom Distribution widget '''
        self.distedit.set_component(uncert)
        self.distedit.setVisible(True)
        self.tab.insertTab(0, self.distedit, 'Distribution')
        self.tab.setCurrentIndex(0)

    def updatecustom(self):
        ''' Update Custom Distribution widget '''
        self.distedit.set_component(None)

    def hidecustom(self):
        ''' Hide the custom distribution widget '''
        self.distedit.setVisible(False)
        tabtext = [self.tab.tabText(i) for i in range(self.tab.count())]
        if 'Distribution' in tabtext:
            self.tab.removeTab(tabtext.index('Distribution'))


#------------------------------------------------------------
# Output view control widgets
#------------------------------------------------------------
class OutputExpandedWidget(QtWidgets.QWidget):
    ''' Widget for controlling expanded uncertainties page '''
    changed = QtCore.pyqtSignal()

    def __init__(self):
        super(OutputExpandedWidget, self).__init__()
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
        super(OutputPlotWidget, self).__init__()
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
        super(OutputMCSampleWidget, self).__init__()
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
        super(OutputMCDistributionWidget, self).__init__()
        self.cmbdist = QtWidgets.QComboBox()
        dists = gui_common.settings.getDistributions()
        dists = [d for d in dists if hasattr(customdists.get_dist(d), 'fit')]
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
        super(OutputMCConvergeWidget, self).__init__()
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
        super(OutputGUMValidityWidget, self).__init__()
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
        super(OutputGUMDerivationWidget, self).__init__()
        self.showvalues = QtWidgets.QCheckBox('Show derivation with values')
        self.showvalues.stateChanged.connect(self.changed)
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.showvalues)
        layout.addStretch()
        self.setLayout(layout)


class OutputReportGen(QtWidgets.QWidget):
    ''' Class for controlling full output report '''
    def __init__(self, parent=None):
        super(OutputReportGen, self).__init__(parent=parent)
        self.report = None  # Cache the report object, only refresh on button press
        self.btnRefresh = QtWidgets.QPushButton('Refresh')
        self.chkSummary = QtWidgets.QCheckBox('Summary')
        self.chkOutputPlot = QtWidgets.QCheckBox('Output Plots')
        self.chkInputs = QtWidgets.QCheckBox('Input Values')
        self.chkComponents = QtWidgets.QCheckBox('Uncertainty Components')
        self.chkSensitivity = QtWidgets.QCheckBox('Sensitivity Coefficients')
        self.chkExpanded = QtWidgets.QCheckBox('Expanded Uncertainties')
        self.chkGUMderiv = QtWidgets.QCheckBox('GUM Derivation')
        self.chkGUMvalid = QtWidgets.QCheckBox('GUM Validity')
        self.chkMChist = QtWidgets.QCheckBox('MC Input Histograms')
        self.chkMCscat = QtWidgets.QCheckBox('MC Input Scatter Plots')
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
        super(PageOutput, self).__init__(parent)
        self.outputSelect = QtWidgets.QComboBox()
        self.outputSelect.addItems(['Summary', 'Comparison Plots', 'Expanded Uncertainties', 'Uncertainty Components',
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
        self.ctrlStack.addWidget(QtWidgets.QWidget())  # 4 - Uncertainty components (blank)
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

        rptsetup.update(gui_common.get_rptargs())  # Add in formatting
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
        assert self.outputSelect.findText('Uncertainty Components') != -1
        assert self.outputSelect.findText('GUM Validity') != -1
        assert self.outputSelect.findText('GUM Derivation') != -1
        assert self.outputSelect.findText('Comparison Plots') != -1
        assert self.outputSelect.findText('Monte Carlo Input Plots') != -1
        assert self.outputSelect.findText('Monte Carlo Distribution') != -1
        assert self.outputSelect.findText('Full Report') != -1
        if (option in ['Summary', 'Expanded Uncertainties', 'Uncertainty Components', 'GUM Validity', 'GUM Derivation',
                       'Monte Carlo Components', 'Full Report', 'Warnings']):
            self.outputStack.setCurrentIndex(TEXT)
        else:
            self.outputStack.setCurrentIndex(PLOT)

        if option == 'Summary':
            r = output.MDstring('## Results\n\n')
            r += self.outdata.report_summary(**gui_common.get_rptargs())
            self.txtOutput.setMarkdown(r)

        elif option == 'Full Report':
            if self.outputReportSetup.report is None:
                self.refresh_fullreport()
            self.txtOutput.setMarkdown(self.outputReportSetup.report)

        elif option == 'Expanded Uncertainties':
            assert self.outputExpanded.GUMexpanded.GUMtype.findText('Normal/k') != -1
            assert self.outputExpanded.MCexpanded.MCtype.findText('Shortest') != -1
            intervalsgum = self.outputExpanded.GUMexpanded.get_covlist()
            intervals = self.outputExpanded.MCexpanded.get_covlist()
            norm = self.outputExpanded.GUMexpanded.GUMtype.currentText() == 'Normal/k'
            shortest = self.outputExpanded.MCexpanded.MCtype.currentText() == 'Shortest'
            r = '## Expanded Uncertainty\n\n'
            r += self.outdata.report_expanded(covlist=intervals, normal=norm, shortest=shortest, covlistgum=intervalsgum, **gui_common.get_rptargs())
            self.txtOutput.setMarkdown(r)

        elif option == 'Uncertainty Components':
            self.txtOutput.setMarkdown(self.outdata.report_allinputs(**gui_common.get_rptargs()))

        elif option == 'Warnings':
            self.txtOutput.setMarkdown(self.outdata.report_warns(**gui_common.get_rptargs()))

        elif option == 'GUM Derivation':
            solve = self.outputGUMderiv.showvalues.isChecked()
            self.txtOutput.setMarkdown(self.outdata.report_derivation(solve=solve, **gui_common.get_rptargs()))

        elif option == 'Monte Carlo Convergence':
            self.outdata.plot_converge(fig=self.fig, relative=self.outputMCconv.relative.isChecked())
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
                self.outdata.plot_outputscatter(fig=self.fig, **plotargs)
            else:
                self.outdata.plot_pdf(fig=self.fig, **plotargs)
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
                self.outdata.plot_xscatter(fig=self.fig, **plotargs)
            else:
                self.outdata.plot_xhists(fig=self.fig, **plotargs)
            self.canvas.draw_idle()

        elif option == 'GUM Validity':
            ndig = self.outputGUMvalid.ndig.value()
            r = self.outdata.report_validity(ndig=ndig, **gui_common.get_rptargs())
            self.txtOutput.setMarkdown(r)

        elif option == 'Monte Carlo Distribution':
            fidx = self.outputMCdist.cmbfunc.currentIndex()
            dist = self.outputMCdist.cmbdist.currentText()
            y = self.outdata.foutputs[fidx].mc.samples[:, 0]
            fitparams = output.fitdist(y, dist=dist, fig=self.fig, qqplot=True, bins=100, points=200)
            try:
                self.fig.axes[0].set_title('Distribution Fit')
                self.fig.axes[1].set_title('Probability Plot')
            except IndexError:
                pass  # Axes weren't added (maybe invalid samples)
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
        if len(self.outdata.report_warns()) > 0:
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
    opencsvfolder = QtCore.QStandardPaths.standardLocations(QtCore.QStandardPaths.HomeLocation)[0]
    openconfigfolder = QtCore.QStandardPaths.standardLocations(QtCore.QStandardPaths.HomeLocation)[0]
    newtype = QtCore.pyqtSignal(object, str)

    PG_INPUT = 0
    PG_OUTPUT = 1

    def __init__(self, item, parent=None):
        super(UncertPropWidget, self).__init__(parent)
        assert isinstance(item, uc.UncertCalc)
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
        self.actSweep = QtWidgets.QAction('New uncertainty sweep from model', self)
        self.actReverse = QtWidgets.QAction('New reverse calculation from model', self)
        self.actLoadCSV = QtWidgets.QAction('Load uncertainties from file...', self)
        self.actLoadDist = QtWidgets.QAction('Use distribution from...', self)
        self.actClear = QtWidgets.QAction('Clear inputs', self)
        self.actSaveReport = QtWidgets.QAction('Save Report...', self)
        self.actSaveSamplesCSV = QtWidgets.QAction('Text (CSV)...', self)
        self.actSaveSamplesNPZ = QtWidgets.QAction('Binary (NPZ)...', self)

        self.menu.addAction(self.actSweep)
        self.menu.addAction(self.actReverse)
        self.menu.addSeparator()
        self.menu.addAction(self.actLoadCSV)
        self.menu.addAction(self.actClear)
        self.menu.addSeparator()
        self.menu.addAction(self.actSaveReport)
        self.mnuSaveSamples = QtWidgets.QMenu('Save Monte Carlo Samples')
        self.mnuSaveSamples.addAction(self.actSaveSamplesCSV)
        self.mnuSaveSamples.addAction(self.actSaveSamplesNPZ)
        self.menu.addMenu(self.mnuSaveSamples)
        self.actSaveReport.setEnabled(False)
        self.mnuSaveSamples.setEnabled(False)

        self.actLoadCSV.triggered.connect(self.loadfromcsv)
        self.actClear.triggered.connect(self.clearinput)
        self.actSweep.triggered.connect(lambda event, x=item: self.newtype.emit(x.get_config(), 'sweep'))
        self.actReverse.triggered.connect(lambda event, x=item: self.newtype.emit(x.get_config(), 'reverse'))
        self.actSaveReport.triggered.connect(self.save_report)
        self.actSaveSamplesCSV.triggered.connect(self.save_samples_csv)
        self.actSaveSamplesNPZ.triggered.connect(self.save_samples_npz)

        self.pginput.calculate.connect(self.calculate)
        self.pginput.funclist.orderchange.connect(self.funcorderchanged)
        self.pginput.funclist.funcchanged.connect(self.funcchanged)
        self.pginput.funclist.funcremoved.connect(self.funcremoved)
        self.pginput.corrtable.cellChanged.connect(self.set_correlations)
        self.pginput.MCsettings.txtSeed.editingFinished.connect(self.setSeed)
        self.pginput.MCsettings.txtSamples.editingFinished.connect(self.setSamples)
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

    def clearinput(self):
        ''' Clear all the input/output values '''
        self.pginput.funclist.clear()
        self.pginput.inputtree.update([])
        self.pginput.corrtable.reset()
        self.pginput.description.setPlainText('')
        self.pginput.MCsettings.txtSamples.setText(str(gui_common.settings.getSamples()))
        self.pginput.MCsettings.txtSeed.setText(str(gui_common.settings.getRandomSeed()))
        self.setSamples()
        self.setSeed()
        self.pginput.hidecustom()
        self.uncCalc = uc.UncertCalc(seed=gui_common.settings.getRandomSeed())
        self.pgoutput.set_unccalc(self.uncCalc)
        self.actSweep.setEnabled(False)
        self.stack.setCurrentIndex(0)

    def funcchanged(self, row, fdict):
        ''' Function was changed. '''
        self.uncCalc.set_function(fdict['expr'], idx=row, name=fdict['name'], desc=fdict['desc'], show=fdict.get('report', True))
        self.uncCalc.add_required_inputs()
        baseinputs = self.uncCalc.get_baseinputs()
        self.pginput.inputtree.update(baseinputs)
        self.pginput.corrtable.setVarNames(self.uncCalc.get_reqd_inputs())
        self.actSweep.setEnabled(True)

    def funcremoved(self, row):
        ''' A function was removed from the list '''
        self.uncCalc.remove_function(row)

    def funcorderchanged(self):
        ''' Functions were reordered by drag/drop '''
        names = [self.pginput.funclist.item(r, self.pginput.funclist.COL_NAME).text() for r in range(self.pginput.funclist.rowCount())]
        self.uncCalc.reorder(names)

    def setDesc(self):
        ''' Description was edited. Save to uncCalc. '''
        self.uncCalc.longdescription = self.pginput.description.toPlainText()

    def setSeed(self):
        ''' Seed textbox was edited. Save seed to uncCalc. '''
        try:
            seed = abs(int(float(self.pginput.MCsettings.txtSeed.text())))
        except (ValueError, OverflowError):
            seed = None
        self.uncCalc.seed = seed
        self.pginput.MCsettings.txtSeed.blockSignals(True)
        self.pginput.MCsettings.txtSeed.setText(str(seed))
        self.pginput.MCsettings.txtSeed.blockSignals(False)

    def setSamples(self):
        ''' Samples textbox was edited. Save to uncCalc. '''
        try:
            samples = int(float(self.pginput.MCsettings.txtSamples.text()))  # cast to float first so exp-notation will work
        except (ValueError, OverflowError):
            samples = 1000000
        self.uncCalc.samples = samples
        self.pginput.MCsettings.txtSamples.blockSignals(True)
        self.pginput.MCsettings.txtSamples.setText(str(samples))
        self.pginput.MCsettings.txtSamples.blockSignals(False)

    def set_correlations(self):
        ''' Set correlation table in unc calc. '''
        self.uncCalc.clear_corr()
        for row in range(self.pginput.corrtable.table.rowCount()):
            v1, v2, val = self.pginput.corrtable.getRow(row)
            try:
                f = float(val)
                self.uncCalc.correlate_vars(v1, v2, f)
            except ValueError:
                if self.pginput.corrtable.table.item(row, 2) is not None:
                    self.pginput.corrtable.table.item(row, 2).setBackground(QtGui.QBrush(gui_common.COLOR_INVALID))
                self.pginput.corrtable.table.clearSelection()
            else:
                self.pginput.corrtable.table.item(row, 2).setBackground(QtGui.QBrush(gui_common.COLOR_OK))

    def loadfromcsv(self):
        ''' Load type-A uncertainties from data in a CSV file. '''
        fname, self.opencsvfolder = QtWidgets.QFileDialog.getOpenFileName(caption='Select CSV file to load', directory=self.opencsvfolder)
        if fname:
            _, fnamebase = os.path.split(fname)
            dlg = page_data.UncertsFromCSV(fname, colnames=self.uncCalc.get_baseinputnames())
            ok = dlg.exec_()
            if ok:
                for item in dlg.get_statistics():
                    if 'corr' in item:
                        self.uncCalc.correlate_vars(item['corr'][0], item['corr'][1], item['coef'])
                        self.pginput.corrtable.addRow()
                        self.pginput.corrtable.setRow(self.pginput.corrtable.table.rowCount()-1, item['corr'][0], item['corr'][1], item['coef'])
                    else:
                        self.uncCalc.set_input(item['var'], nom=item['mean'])
                        self.uncCalc.set_uncert(item['var'], name='u({})'.format(item['var']), degf=item['degf'],
                                                unc=item['stdu'], desc='Type A uncertainty from {}'.format(fnamebase))
                self.pginput.inputtree.update()
                self.backbutton()

    def calculate(self):
        ''' Run the calculation '''
        valid = True
        if not (self.pginput.funclist.isValid() and self.pginput.inputtree.isValid()):
            valid = False

        if len(self.uncCalc.functions) < 1:
            valid = False

        if valid:
            try:
                self.uncCalc.calculate()
            except (ValueError, RecursionError):
                valid = False

        if valid:
            self.stack.setCurrentIndex(self.PG_OUTPUT)
            self.pgoutput.update(self.uncCalc.out)
            self.pgoutput.outputupdate()
            self.actSaveReport.setEnabled(True)
            self.mnuSaveSamples.setEnabled(True)
        else:
            QtWidgets.QMessageBox.warning(self, 'Uncertainty Calculator', 'Invalid Input Parameter!')
            self.actSaveReport.setEnabled(False)
            self.mnuSaveSamples.setEnabled(False)

    def set_calc(self, calc):
        ''' Set calculator object to be displayed (e.g. loading from project file) '''
        self.uncCalc = calc
        self.pginput.corrtable.reset()
        for inpt in self.uncCalc.get_baseinputs():
            for unc in inpt.uncerts:
                if unc.distname != 'normal' or 'conf' in unc.userargs or 'k' in unc.userargs:
                    unc.customized = True

        self.pginput.funclist.blockSignals(True)
        self.pgoutput.set_unccalc(self.uncCalc)
        self.pginput.funclist.setFunclist(calc.functions)
        self.pginput.MCsettings.txtSamples.setText(str(self.uncCalc.samples))
        self.pginput.corrtable.setVarNames([i.name for i in self.uncCalc.get_baseinputs()])
        self.pginput.MCsettings.txtSeed.setText(str(self.uncCalc.seed))
        self.pginput.description.setPlainText(str(self.uncCalc.longdescription))
        self.pginput.hidecustom()

        if self.uncCalc.longdescription != '':
            self.pginput.tab.setCurrentIndex([self.pginput.tab.tabText(i) for i in range(self.pginput.tab.count())].index('Notes'))

        if self.uncCalc._corr is not None:
            for v1, v2, c in self.uncCalc.get_corr_list():
                if c != 0.:
                    self.pginput.corrtable.addRow()
                    self.pginput.corrtable.setRow(self.pginput.corrtable.table.rowCount()-1, v1, v2, c)
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
        gui_widgets.savemarkdown(self.get_report())

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
