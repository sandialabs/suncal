''' Page for propagating uncertainty calculations '''

import os
import re
import sympy
from pint import DimensionalityError, UndefinedUnitError, OffsetUnitCalculusError

from PyQt5 import QtWidgets, QtGui, QtCore
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

from .. import uncertainty as uc
from .. import uparser
from .. import output
from .. import customdists
from .. import ttable
from . import gui_common
from . import gui_widgets
from . import page_data


TREESTYLE = """
QTreeView {
    show-decoration-selected: 1;
}
QTreeView::item {
    border: 1px solid #d9d9d9;
    border-top-color: transparent;
    border-bottom-color: transparent;
}
QTreeView::item:has-children {
    border-bottom-color: #d9d9d9;
}
QTreeView::item:selected {
    border: 1px solid #567dbc;
}
QTreeView::item:selected:active{
    background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1, stop: 0 #6ea1f1, stop: 1 #567dbc);
}
QTreeView::item:selected:!active {
    background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1, stop: 0 #6b9be8, stop: 1 #577fbf);
}"""

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
    background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1, stop: 0 #6b9be8, stop: 1 #577fbf);
}"""


def tree_bold(item, column):
    ''' Set the TreeWidgetItem font bold '''
    font = item.font(column)
    font.setBold(True)
    item.setFont(column, font)


class TreeRow(QtWidgets.QTreeWidgetItem):
    ''' TreeWidgetItem formatted for Input Tree '''
    def __init__(self, parent, obj, name, value='', text=''):
        super(TreeRow, self).__init__(parent, [name, value, text])
        self.setFlags(self.flags() | QtCore.Qt.ItemIsEditable)
        self.obj = obj
        self.valid = True
        tree_bold(self, 1)


class TreeRowTex(QtWidgets.QTreeWidgetItem):
    ''' TreeWidgetItem formatted for Input Tree, with first column displayed as latex '''
    def __init__(self, parent, obj, name, value='', text=''):
        super(TreeRowTex, self).__init__(parent, ['', value, text])
        self.setFlags(self.flags() | QtCore.Qt.ItemIsEditable)
        self.obj = obj
        self.valid = True
        tree_bold(self, 1)
        # Remove display text, replace with rendered math
        px = QtGui.QPixmap()
        px.loadFromData(output.tex_to_buf(obj.get_latex()).read())
        self.setData(0, QtCore.Qt.DecorationRole, px)
        self.setData(0, gui_widgets.ROLE_ORIGDATA, obj.name)


class TableItemTex(QtWidgets.QTableWidgetItem):
    ''' TableWidgetItem formatted for Functions Input Table '''
    def __init__(self, tex=''):
        super(TableItemTex, self).__init__()
        self.setFlags(self.flags() | QtCore.Qt.ItemIsEditable)

        # Remove display text, replace with rendered math
        px = QtGui.QPixmap()
        px.loadFromData(output.tex_to_buf(output.format_math(tex)).read())
        self.setData(QtCore.Qt.DecorationRole, px)
        self.setData(gui_widgets.ROLE_ORIGDATA, tex)


class FunctionTableWidget(QtWidgets.QTableWidget):
    ''' Function definition widget '''
    funcchanged = QtCore.pyqtSignal(int, dict)
    funcremoved = QtCore.pyqtSignal(int)
    orderchange = QtCore.pyqtSignal()
    COL_NAME = 0
    COL_EXPR = 1
    COL_UNIT = 2
    COL_DESC = 3
    COL_REPT = 4

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
        self._delegate = gui_widgets.LatexDelegate()  # Don't let the garbage collector eat it
        self.setItemDelegateForColumn(0, self._delegate)
        self.setItemDelegateForColumn(1, self._delegate)
        self.setStyleSheet(TABLESTYLE)
        self.clear()
        self.addRow()
        self.cellChanged.connect(self.itemEdit)

    def clear(self):
        ''' Override clear to reset things correctly. '''
        self.setRowCount(0)
        self.setHorizontalHeaderLabels(['Name', 'Expression',  'Units', 'Description', 'Report?'])
        self.resizeColumnsToContents()
        self.setColumnWidth(self.COL_EXPR, 400)
        self.setColumnWidth(self.COL_UNIT, 75)
        self.setColumnWidth(self.COL_DESC, 400)

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
            else:
                self.item(row, self.COL_EXPR).setBackground(QtGui.QBrush(gui_common.COLOR_INVALID))
                self.clearSelection()

        if col == self.COL_NAME and self.item(row, self.COL_NAME) is not None:
            if isinstance(sname, sympy.symbol.Symbol):
                self.item(row, self.COL_NAME).setBackground(QtGui.QBrush(gui_common.COLOR_OK))
            else:
                self.item(row, self.COL_NAME).setBackground(QtGui.QBrush(gui_common.COLOR_INVALID))
                self.clearSelection()

        # Check units
        if unit != '':
            try:
                unit = uc.get_units(unit)
            except ValueError:
                self.item(row, self.COL_UNIT).setBackground(QtGui.QBrush(gui_common.COLOR_INVALID))
                self.item(row, self.COL_UNIT).setText(unit)
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
               self.item(row, self.COL_NAME).background().color() == gui_common.COLOR_INVALID or
               self.item(row, self.COL_UNIT).background().color() == gui_common.COLOR_INVALID):
                return False
        return True

    def contextMenuEvent(self, event):
        item = self.itemAt(event.pos())
        if item:
            row = item.row()
            name = self.item(row, self.COL_NAME).data(gui_widgets.ROLE_ORIGDATA)
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
        mbox.setText('Set model equation?' + '<br><br>' + output.tex_to_html(sympy.latex(sympy.Eq(svar, solution))))
        mbox.setInformativeText('Measured Quantity entries may be removed.<br><br>Note: Do not use for reversing the calculation to determine required uncertainty of a measured quantity.')
        mbox.setStandardButtons(QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)
        ok = mbox.exec_()
        if ok == QtWidgets.QMessageBox.Yes:
            expr = str(solution)
            if fn is not None:
                # Must use delegate to set so equation renders properly
                self._delegate.setModelData(QtWidgets.QLineEdit(expr), self.model(), self.indexFromItem(self.item(row, self.COL_EXPR)))
                self._delegate.setModelData(QtWidgets.QLineEdit(var), self.model(), self.indexFromItem(self.item(row, self.COL_NAME)))


class TreeButton(QtWidgets.QToolButton):
    ''' Round button for use in a tree widget '''
    # CSS stylesheet for nice round buttons
    buttonstyle = '''QToolButton {border: 1px solid #8f8f91; border-radius: 8px;
                     background-color: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1, stop: 0 #f6f7fa, stop: 1 #dadbde);}

                     QToolButton:pressed {border: 2px solid #8f8f91; border-radius: 8px; border-width: 2px;
                     background-color: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1, stop: 0 #dadbde, stop: 1 #f6f7fa);}

                     QToolButton:checked {border: 2px solid #8f8f91; border-radius: 8px; border-width: 1px;
                     background-color: qlineargradient(x1: 0, y1: 1, x2: 0, y2: 0, stop: 0 #dadbde, stop: 1 #7c7c7c);}
                     '''

    def __init__(self, text):
        super(TreeButton, self).__init__(text=text)
        self.setStyleSheet(self.buttonstyle)
        self.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)


class InputTreeWidget(QtWidgets.QTreeWidget):
    ''' Tree Widget for editing input nominal values '''
    COL_NAME = 0
    COL_VALUE = 1
    COL_TEXT = 2
    COL_DEGF = 3
    COL_BTN = 4
    COL_CNT = 5

    replotdist = QtCore.pyqtSignal(object)  # Emit the uncert object to plot

    def __init__(self):
        super(InputTreeWidget, self).__init__()
        self.clear()
        self.setColumnCount(self.COL_CNT)
        self.setHeaderItem(QtWidgets.QTreeWidgetItem(['Parameter', 'Value', 'Standard Uncertainty', 'Deg. Freedom', '', '']))
        self.setEditTriggers(QtWidgets.QTableWidget.NoEditTriggers)  # Only start editing manually
        self.inputlist = []
        self.setColumnWidth(self.COL_NAME, 200)
        self.setColumnWidth(self.COL_VALUE, 100)
        self.setColumnWidth(self.COL_TEXT, 150)
        self.setColumnWidth(self.COL_DEGF, 100)
        self.setColumnWidth(self.COL_BTN, 20)
        self.setUniformRowHeights(True)
        self.setStyleSheet(TREESTYLE)
        self._delegate = gui_widgets.LatexDelegate()
        self.setItemDelegateForColumn(self.COL_NAME, self._delegate)

        self.itemChanged.connect(self.itemchange)
        self.itemDoubleClicked.connect(self.checkEdit)
        self.currentItemChanged.connect(self.rowchange)

    def checkEdit(self, item, column):
        ''' Check if this item/column is editable, and start editor if it is. '''
        if self.editable(item, column):
            self.editItem(item, column)

    def editable(self, item, column):
        ''' Check if this item/column is editable '''
        if column == self.COL_VALUE and item.childCount() == 0:
            return True
        elif column == self.COL_NAME and item.childCount() > 0 and isinstance(item.obj, uc.InputUncert):
            return True
        return False

    def rowchange(self, new, prev):
        ''' Selected row was changed '''
        if new is not None and isinstance(new.obj, uc.InputUncert):
            obj = new.obj
        else:
            obj = None
        self.replotdist.emit(obj)

    def filltable(self, inputlist=None):
        ''' Fill the entire table with inputs and uncertainties '''
        self.blockSignals(True)
        if inputlist is not None:
            self.inputlist = inputlist
        self.clear()
        topitem = self.invisibleRootItem()
        for row, inpt in enumerate(self.inputlist):
            newitem = TreeRowTex(topitem, inpt, '', '', '')
            TreeRow(newitem, inpt, 'Nominal', format(inpt.nom, '.5g'))
            TreeRow(newitem, inpt, 'Units', uc.printunit(inpt.units, fmt=''))
            TreeRow(newitem, inpt, 'Description', inpt.desc)
            uncitem = TreeRow(newitem, inpt, 'Uncertainties')
            addbutton = TreeButton('+')
            addbutton.clicked.connect(lambda x, inpt=inpt, uncitem=uncitem: self.add_comp(inpt, uncitem))
            uncitem.treeWidget().setItemWidget(uncitem, self.COL_BTN, addbutton)
            for unc in inpt.uncerts:
                compitem = TreeRowTex(uncitem, unc, unc.get_latex(), '', '')
                tree_bold(compitem, self.COL_NAME)
                rembutton = TreeButton(gui_common.CHR_ENDASH)
                rembutton.clicked.connect(lambda x, item=compitem, unc=unc: self.rem_comp(item, unc))
                compitem.treeWidget().setItemWidget(compitem, self.COL_BTN, rembutton)
                self.fill_uncertparams(compitem, unc)
        self.update_text()
        self.setColumnWidth(self.COL_NAME, 200)
        self.setColumnWidth(self.COL_VALUE, 200)
        self.setColumnWidth(self.COL_TEXT, 150)
        self.setColumnWidth(self.COL_DEGF, 100)
        self.setColumnWidth(self.COL_BTN, 20)
        self.blockSignals(False)

    def fill_uncertparams(self, compitem, uncobj):
        ''' Fill uncertainty parameters for uncertainty uncobj under treewidget compitem '''
        # Signals should already be blocked
        compitem.takeChildren()  # clear
        distitem = TreeRow(compitem, uncobj, 'Distribution')
        cmbdist = QtWidgets.QComboBox()
        cmbdist.addItems(gui_common.settings.getDistributions())
        cmbdist.setCurrentIndex(cmbdist.findText(uncobj.distname))
        if cmbdist.currentText() == '':
            cmbdist.addItem(uncobj.distname)
            cmbdist.setCurrentIndex(cmbdist.count()-1)
        compitem.treeWidget().setItemWidget(distitem, self.COL_VALUE, cmbdist)
        TreeRow(compitem, uncobj, 'units', format(uncobj.units, '~P'))
        degf = uncobj.degf
        TreeRow(compitem, uncobj, 'deg. freedom', 'inf')
        if uncobj.distname in ['normal', 't']:
            if 'conf' in uncobj.userargs:
                conf = float(uncobj.userargs['conf'])
                k = ttable.t_factor(conf, degf)
            else:
                k = float(uncobj.userargs.get('k', 1))
                conf = ttable.confidence(k, degf)
            TreeRow(compitem, uncobj, 'k', '{:.2f}'.format(k))
            TreeRow(compitem, uncobj, 'confidence', '{:.2f}%'.format(conf*100))
            TreeRow(compitem, uncobj, 'uncertainty', '{}'.format(uncobj.userargs.get('unc', uncobj.args.get('std', 1))))
        elif uncobj.distname == 'histogram':
            pass  # No extra widgets
        else:
            for row, arg in enumerate(sorted(uncobj.userargs.keys())):
                TreeRow(compitem, uncobj, arg, format(uncobj.userargs[arg]))
        cmbdist.currentIndexChanged.connect(lambda y, item=compitem, obj=uncobj: self.change_dist(item, obj))

    def import_dist(self):
        ''' Import a distribution from somewhere else in the project '''
        dlg = page_data.DistributionSelectWidget(project=self.parent().unccalc.project)
        ok = dlg.exec_()
        if ok:
            distname, distargs, mean = dlg.get_dist()

            selitem = self.currentItem()
            uncobj = selitem.obj

            # Get treeitem containing uncertainty name
            if selitem.childCount() > 0:
                topitem = selitem
            else:
                topitem = selitem.parent()

            inpt = topitem.parent().obj  # Up to "Uncertainties" treerow
            inpt.set_nom(mean)
            uncobj.nom = mean
            uncobj.userargs = distargs
            uncobj.set_dist(distname)
            uncobj.userargs = distargs
            uncobj.updateparams()

            self.blockSignals(True)
            self.fill_uncertparams(topitem, uncobj)
            nomitem = topitem.parent().parent().child(0)   # Get 'Nominal' tree item
            assert nomitem.text(self.COL_NAME) == 'Nominal'  # Make sure previous line works and things don't change
            nomitem.setText(self.COL_VALUE, format(inpt.nom, '.5g'))
            self.blockSignals(False)
            self.replotdist.emit(uncobj)

    def change_dist(self, compitem, uncobj):
        ''' Distribution type in combobox was changed '''
        self.blockSignals(True)
        distname = self.itemWidget(compitem.child(0), self.COL_VALUE).currentText()   # Dist combo is always first child
        uncobj.set_dist(distname)
        self.fill_uncertparams(compitem, uncobj)
        self.update_text()
        self.blockSignals(False)
        self.replotdist.emit(uncobj)

    def add_comp(self, inpt, uncitem):
        ''' Add a blank uncertainty component '''
        self.blockSignals(True)
        name = 'u{}({})'.format(len(inpt.uncerts)+1, inpt.name)
        unc = inpt.add_comp(name)

        compitem = TreeRowTex(uncitem, unc, unc.get_nameunicode())
        font = compitem.font(self.COL_NAME)
        font.setBold(True)
        compitem.setFont(self.COL_NAME, font)
        rembutton = TreeButton(gui_common.CHR_ENDASH)
        rembutton.clicked.connect(lambda x, item=compitem, unc=unc: self.rem_comp(item, unc))
        compitem.treeWidget().setItemWidget(compitem, self.COL_BTN, rembutton)
        self.fill_uncertparams(compitem, unc)
        self.blockSignals(False)
        self.update_text()

    def rem_comp(self, compitem, unc):
        ''' Remove selected uncertainty component '''
        inptitem = compitem.parent()
        inpt = inptitem.obj
        idx = inpt.uncerts.index(unc)
        inpt.rem_comp(idx)
        inptitem.removeChild(compitem)
        self.update_text()

    def itemchange(self, item, column):
        ''' An item in the table was edited. Validate the input and update the Uncert model. '''
        COLOR = {True: gui_common.COLOR_TEXT_OK, False: gui_common.COLOR_INVALID}
        self.blockSignals(True)

        obj = item.obj  # Either InputVar or InputUncert
        param = item.text(self.COL_NAME)
        value = item.text(self.COL_VALUE)
        status = True
        if param == 'Nominal' and isinstance(obj, uc.InputVar):
            status = obj.set_nom(value)
            if status:
                item.setText(self.COL_VALUE, format(obj.nom, '.5g'))

        elif param.lower() == 'units':
            status = obj.set_units(value)
            if status:
                item.setText(self.COL_VALUE, obj.get_unitstr())

        elif param == 'Description':
            obj.desc = value

        elif column == self.COL_NAME and isinstance(obj, uc.InputUncert):
            obj.name = item.data(self.COL_NAME, gui_widgets.ROLE_ORIGDATA)

        elif param == 'uncertainty':
            obj.userargs['unc'] = value
            obj.userargs.pop('std', None)

        elif param in ['k', 'confidence', 'deg. freedom']:
            # Must be floating points
            value =value.strip('%')  # In case conf was entered with percent symbol
            try:
                value = float(value)
            except ValueError:
                status = False
            else:
                if param == 'deg. freedom':
                    obj.degf = value
                    status = obj.updateparams()
                    kitems = [item.parent().child(i) for i in range(item.parent().childCount()) if item.parent().child(i).text(self.COL_NAME) == 'k']
                    if len(kitems) > 0:
                        if 'k' in obj.userargs:
                            confitem = [item.parent().child(i) for i in range(item.parent().childCount()) if item.parent().child(i).text(self.COL_NAME) == 'confidence'][0]
                            confitem.setText(self.COL_VALUE, '{:.2f}%'.format(ttable.confidence(obj.userargs['k'], obj.degf)*100))
                        elif 'conf' in obj.userargs:
                            kitem = [item.parent().child(i) for i in range(item.parent().childCount()) if item.parent().child(i).text(self.COL_NAME) == 'k'][0]
                            kitem.setText(self.COL_VALUE, '{:.2f}'.format(ttable.t_factor(obj.userargs['conf'], obj.degf)))

                elif param == 'k':
                    obj.userargs['k'] = value
                    obj.userargs.pop('conf', None)
                    confitem = [item.parent().child(i) for i in range(item.parent().childCount()) if item.parent().child(i).text(self.COL_NAME) == 'confidence'][0]
                    confitem.setText(self.COL_VALUE, '{:.2f}%'.format(ttable.confidence(value, obj.degf)*100))
                elif param == 'confidence':

                    obj.userargs['conf'] = value/100   # Assume entry in percent
                    obj.userargs.pop('k', None)
                    kitem = [item.parent().child(i) for i in range(item.parent().childCount()) if item.parent().child(i).text(self.COL_NAME) == 'k'][0]
                    kitem.setText(self.COL_VALUE, '{:.2f}'.format(ttable.t_factor(value/100, obj.degf)))
                    item.setText(self.COL_VALUE, '{:.2f}%'.format(value))
        else:
            obj.userargs[param] = value

        try:
            status = status and obj.updateparams()
        except AttributeError:
            pass

        item.setForeground(self.COL_VALUE, COLOR[status])
        self.blockSignals(False)
        if not status:
            self.clearSelection()
        else:
            self.update_text()
            if isinstance(obj, uc.InputUncert):
                self.replotdist.emit(obj)

    def update_text(self):
        ''' Uncertainty table changed downstream. Update the descriptive combined uncertaity column for each row.
            Also update units if they changed automatically.
        '''
        self.blockSignals(True)
        it = QtWidgets.QTreeWidgetItemIterator(self)
        while it.value():
            item = it.value()
            if isinstance(item.obj, uc.InputVar) and item.parent() is None:
                # Top level variable
                item.setText(self.COL_TEXT, '{:.5g~P} ± {:.2g~P}'.format(item.obj.mean(), item.obj.stdunc()))
                item.setText(self.COL_DEGF, '{:.1f}'.format(item.obj.degf()))
            elif isinstance(item.obj, uc.InputUncert) and item.childCount() > 0:
                # Uncert Component top
                item.setText(self.COL_TEXT, '± {:.2g~P}'.format(item.obj.std()))
                item.setText(self.COL_DEGF, '{:.1f}'.format(item.obj.degf))
            elif item.text(self.COL_NAME) == 'units':
                item.setText(self.COL_VALUE, format(item.obj.units, '~P'))
            it += 1

        self.blockSignals(False)

    def isValid(self):
        ''' Return True if all entries are valid (not red) '''
        it = QtWidgets.QTreeWidgetItemIterator(self)
        while it.value():
            item = it.value()
            if not item.valid:
                return False
            it += 1
        return True


class PopupHelp(QtWidgets.QDialog):
    ''' Show a floating dialog window with a text message '''
    def __init__(self, text):
        super(PopupHelp, self).__init__()
        self.setGeometry(600, 200, 600, 400)
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


class DistInfo(QtWidgets.QWidget):
    ''' Widget for displaying more info about distribution '''
    import_dist = QtCore.pyqtSignal()

    def __init__(self, parent=None):
        super(DistInfo, self).__init__(parent=parent)
        self.distname = QtWidgets.QLabel('Probability Distribution for ')
        self.distfig = Figure()
        self.distcanvas = FigureCanvas(self.distfig)
        self.btnImport = QtWidgets.QToolButton()
        self.btnImport.setIcon(gui_common.load_icon('loaddist'))
        self.btnImport.setToolTip('Import Distribution From...')
        self.btnHelp = QtWidgets.QToolButton()
        self.btnHelp.setIcon(gui_common.load_icon('help'))
        self.btnHelp.setToolTip('Distribution Parameter Help')
        blayout = QtWidgets.QHBoxLayout()
        blayout.addWidget(self.distname)
        blayout.addStretch()
        blayout.addWidget(self.btnImport)
        blayout.addWidget(self.btnHelp)
        layout = QtWidgets.QVBoxLayout()
        layout.addLayout(blayout)
        layout.addWidget(self.distcanvas)
        self.setLayout(layout)
        self.comp = None
        self.btnHelp.clicked.connect(self.showhelp)
        self.btnImport.clicked.connect(self.import_dist)

    def update_info(self, comp):
        self.comp = comp
        self.distfig.clf()
        ax = self.distfig.add_subplot(1, 1, 1)
        x, y = comp.pdf()
        ax.plot(x, y)
        self.distfig.tight_layout()
        self.distcanvas.draw_idle()
        self.distname.setText('Probability Distribution for {}'.format(comp.get_nameunicode()))
        self.helpstr = comp.helpstr

    def showhelp(self):
        ''' Show description of distribution parameters '''
        dlg = PopupHelp(self.comp.helpstr)
        dlg.exec_()


class PageInput(QtWidgets.QWidget):
    calculate = QtCore.pyqtSignal()

    ''' Page for setting up input parameters '''
    def __init__(self, unccalc, parent=None):
        super(PageInput, self).__init__(parent)
        self.unccalc = unccalc
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
        self.inputtable = InputTreeWidget()
        self.corrtable = CorrelationTableWidget()
        self.distinfo = DistInfo()
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

        blayout = QtWidgets.QHBoxLayout()
        blayout.addWidget(QtWidgets.QLabel('Measurement Model:'))
        blayout.addWidget(self.btnAdd)
        blayout.addWidget(self.btnRem)
        blayout.addStretch()
        hlayout = QtWidgets.QHBoxLayout()
        hlayout.addWidget(self.inputtable, stretch=10)
        hlayout.addWidget(self.tab)
        calclayout = QtWidgets.QHBoxLayout()
        calclayout.addStretch()
        calclayout.addWidget(self.btnCalc)
        layout = QtWidgets.QVBoxLayout()
        layout.addLayout(blayout)
        layout.addWidget(self.funclist, stretch=5)
        layout.addWidget(QtWidgets.QLabel('Measured Quantities:'))
        layout.addLayout(hlayout, stretch=8)
        layout.addLayout(calclayout)
        self.setLayout(layout)

        self.inputtable.replotdist.connect(self.update_disttab)
        self.distinfo.import_dist.connect(self.inputtable.import_dist)

    def set_unccalc(self, unccalc):
        ''' Set the uncertainty calc object '''
        self.unccalc = unccalc

    def update_disttab(self, comp=None):
        ''' Update the plot with PDF of distribution '''
        if comp is not None:
            self.tab.insertTab(0, self.distinfo, 'Distribution')
            self.distinfo.update_info(comp)
            self.tab.setCurrentIndex(0)
        elif self.tab.tabText(0) == 'Distribution':
            self.tab.removeTab(0)


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
            y = self.outdata.foutputs[fidx].mc.samples.magnitude
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
        self.actChkUnits = QtWidgets.QAction('Check Units...', self)
        self.actSweep = QtWidgets.QAction('New uncertainty sweep from model', self)
        self.actReverse = QtWidgets.QAction('New reverse calculation from model', self)
        self.actLoadCSV = QtWidgets.QAction('Load uncertainties from file...', self)
        self.actClear = QtWidgets.QAction('Clear inputs', self)
        self.actSaveReport = QtWidgets.QAction('Save Report...', self)
        self.actSaveSamplesCSV = QtWidgets.QAction('Text (CSV)...', self)
        self.actSaveSamplesNPZ = QtWidgets.QAction('Binary (NPZ)...', self)

        self.menu.addAction(self.actChkUnits)
        self.menu.addSeparator()
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
        self.actChkUnits.triggered.connect(self.checkunits)

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
        self.pginput.inputtable.clear()
        self.pginput.corrtable.reset()
        self.pginput.description.setPlainText('')
        self.pginput.MCsettings.txtSamples.setText(str(gui_common.settings.getSamples()))
        self.pginput.MCsettings.txtSeed.setText(str(gui_common.settings.getRandomSeed()))
        self.setSamples()
        self.setSeed()
        self.uncCalc = uc.UncertCalc(seed=gui_common.settings.getRandomSeed())
        self.pgoutput.set_unccalc(self.uncCalc)
        self.pginput.set_unccalc(self.uncCalc)
        self.actSweep.setEnabled(False)
        self.stack.setCurrentIndex(0)

    def checkunits(self):
        ''' Show units/dimensionality report '''
        dlg = gui_widgets.MarkdownTextEdit()
        dlg.setMinimumSize(800, 600)
        dlg.setMarkdown(self.uncCalc.units_report(**gui_common.get_rptargs()))
        dlg.show()

    def funcchanged(self, row, fdict):
        ''' Function was changed. '''
        self.uncCalc.set_function(fdict['expr'], idx=row, name=fdict['name'], desc=fdict['desc'],
                                  show=fdict.get('report', True), outunits=fdict['unit'])
        self.uncCalc.add_required_inputs()
        baseinputs = self.uncCalc.get_baseinputs()
        self.pginput.inputtable.filltable(baseinputs)
        self.pginput.update_disttab()
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
                self.pginput.inputtable.filltable()
                self.backbutton()

    def calculate(self):
        ''' Run the calculation '''
        msg = None
        if not (self.pginput.funclist.isValid() and self.pginput.inputtable.isValid()):
            msg = 'Invalid input parameter!'

        if len(self.uncCalc.functions) < 1:
            msg = 'No functions to compute!'

        try:
            self.uncCalc.check_dimensionality()
        except (DimensionalityError, UndefinedUnitError) as e:
            msg = 'Units Error: {}'.format(e)
        except OffsetUnitCalculusError as e:
            badunit = re.findall(r'\((.+ )', str(e))[0].split()[0].strip(', ')
            msg = 'Ambiguous unit {}. Try "{}".'.format(badunit, 'delta_{}'.format(badunit))

        if msg is None:
            try:
                self.uncCalc.calculate()
            except OffsetUnitCalculusError as e:
                badunit = re.findall(r'\((.+ )', str(e))[0].split()[0].strip(', ')
                msg = 'Ambiguous unit {}. Try "{}".'.format(badunit, 'delta_{}'.format(badunit))
            except (ValueError, RecursionError):
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
        self.pginput.MCsettings.txtSamples.setText(str(self.uncCalc.samples))
        self.pginput.corrtable.setVarNames([i.name for i in self.uncCalc.get_baseinputs()])
        self.pginput.MCsettings.txtSeed.setText(str(self.uncCalc.seed))
        self.pginput.description.setPlainText(str(self.uncCalc.longdescription))

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
