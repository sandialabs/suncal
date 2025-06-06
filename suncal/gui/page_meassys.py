''' Measurement System (Specific Measured Values) Page '''
import re
from PyQt6 import QtGui, QtWidgets, QtCore
from pint import OffsetUnitCalculusError, DimensionalityError, UndefinedUnitError
import sympy

from ..common import unitmgr, uparser
from ..common.limit import Limit
from ..project import ProjectMeasSys
from ..uncertainty.variables import Typeb
from ..meassys import SystemQuantity, SystemIndirectQuantity
from ..meassys.curve import SystemCurve
from .gui_settings import gui_settings
from . import gui_styles
from . import gui_common
from . import gui_math
from . import widgets
from .widgets.mqa import TypeADialog, ToleranceDelegate
from .delegates import SuncalDelegate, NoEditDelegate, EditDelegate, LatexDelegate, PopupDelegate
from .help_strings import SystemHelp, CurveHelp
from .page_uncert_output import PageOutput as PageGumOutput
from .page_uncert_input import CorrelationTableWidget
from .page_measys_curve import MeasSysCurveWidget
from .page_curvefit import PageOutputCurveFit as PageCurveOutput


class Settings(QtWidgets.QDialog):
    ''' Settings dialog '''
    def __init__(self, system: 'MeasureSystem', parent=None):
        super().__init__(parent=parent)
        self.samples = QtWidgets.QSpinBox()
        self.samples.setRange(10, 100000000)
        self.seed = QtWidgets.QLineEdit('None')
        self.chkCorrelate = QtWidgets.QCheckBox('')
        self.conf = widgets.PercentLineEdit('95')
        self.buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Ok |
                                                  QtWidgets.QDialogButtonBox.StandardButton.Cancel)
        self.conf.setValue(system.confidence)
        self.samples.setValue(system.samples)
        self.seed.setText(str(system.seed))
        self.chkCorrelate.setChecked(system.correlate_typeas)
        layout = QtWidgets.QFormLayout()
        layout.addRow('Monte Carlo Samples', self.samples)
        layout.addRow('Monte Carlo Random Seed', self.seed)
        layout.addRow('Level of Confidence for Expanded Uncertainties', self.conf)
        layout.addRow('Correlate quantities with Type A data of same length', self.chkCorrelate)
        tlayout = QtWidgets.QVBoxLayout()
        tlayout.addLayout(layout)
        tlayout.addWidget(self.buttons)
        self.setLayout(tlayout)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)


class TreeItem(QtWidgets.QTreeWidgetItem):
    ''' Tree item with optional Math and column enable/disable '''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setFlags(self.flags() | QtCore.Qt.ItemFlag.ItemIsEditable)
        gui_styles.darkmode_signal().connect(self.changecolormode)

    def changecolormode(self):
        ''' Update light/dark mode '''
        for i in range(self.columnCount()):
            if (data := self.data(i, LatexDelegate.ROLE_ENTERED)):
                self.setExpr(i, data)

    def enableEdit(self, *args):
        for i, enable in enumerate(args):
            self.setData(i, LatexDelegate.ROLE_DISABLE, not enable)
        return self

    def enableMath(self, *args):
        ''' Enable math mode on the columns '''
        for i, enable in enumerate(args):
            self.setData(i, LatexDelegate.ROLE_MATH_DISABLE, not enable)
            self.setExpr(i, self.text(i))
            self.setText(i, '')
        return self

    def setExpr(self, column, expr):
        ''' Remove display text, replace with rendered math '''
        px = gui_math.pixmap_from_latex(expr)
        self.setData(column, QtCore.Qt.ItemDataRole.DecorationRole, px)
        self.setData(column, LatexDelegate.ROLE_ENTERED, expr)

    def children(self):
        ''' Why is this not built in to QTreeWidgetItem?? '''
        return [self.child(i) for i in range(self.childCount())]


class FunctionTree(QtWidgets.QTreeWidget):
    ''' Tree widget for editing Functions '''
    COL_SYM = 0
    COL_EXPR = 1
    COL_UNITS = 2
    COL_TOLERANCE = 3
    COL_DESC = 4
    COL_CNT = 5

    equation_changed = QtCore.pyqtSignal()
    added = QtCore.pyqtSignal()

    def __init__(self, component: ProjectMeasSys):
        super().__init__()
        self.component = component
        self.loading: bool = False
        self._delegate = LatexDelegate()
        self._toldelegate = ToleranceDelegate(required=False)
        self.fill_tree()
        self.itemChanged.connect(self.edit_data)

    def set_background(self, item: QtWidgets.QTreeWidgetItem, column: int, valid: bool):
        ''' Set background color of a table item '''
        item.setBackground(column, gui_styles.color.transparent if valid else gui_styles.color.invalid)
        if not valid:
            self.clearSelection()

    def fill_tree(self):
        ''' Fill tree with component values '''
        self.loading = True
        self.clear()
        self.setColumnCount(self.COL_CNT)
        self.setHeaderLabels(['Symbol', 'Equation', 'Units', 'Tolerance', 'Description'])
        self.setItemDelegateForColumn(self.COL_SYM, self._delegate)
        self.setItemDelegateForColumn(self.COL_EXPR, self._delegate)
        self.setItemDelegateForColumn(self.COL_TOLERANCE, self._toldelegate)
        self.setColumnWidth(self.COL_SYM, 75)
        self.setColumnWidth(self.COL_EXPR, 255)
        self.setColumnWidth(self.COL_UNITS, 150)
        self.setColumnWidth(self.COL_TOLERANCE, 150)
        self.setColumnWidth(self.COL_DESC, 250)

        indirect_qty = [qty for qty in self.component.model.quantities if isinstance(qty, SystemIndirectQuantity)]
        for qty in indirect_qty:
            self.add_quantity(qty)
        self.loading = False

    def edit_data(self, item: TreeItem, column: int):
        ''' Item was changed, update the model '''
        if self.loading:
            return
        qty: SystemIndirectQuantity = item.data(0, LatexDelegate.ROLE_QUANTITY)
        if column == self.COL_SYM:
            try:
                value = item.data(self.COL_SYM, LatexDelegate.ROLE_ENTERED)
                mathvalue = uparser.parse_math(value)
            except ValueError:
                self.set_background(item, column, False)
            else:
                if isinstance(mathvalue, sympy.Symbol):  # name must be single symbol
                    qty.symbol = value
                    self.set_background(item, column, True)
                else:
                    self.set_background(item, column, False)

        elif column == self.COL_EXPR:  # May need additional logic if other rows are added
            try:
                value = item.data(self.COL_EXPR, LatexDelegate.ROLE_ENTERED)
                mathvalue = uparser.parse_math_with_quantities(value)
            except ValueError:
                self.set_background(item, column, False)
            else:
                if mathvalue is not None:
                    qty.equation = value   # Set to the string expression
                    self.equation_changed.emit()
                else:
                    self.set_background(item, column, False)

        elif column == self.COL_UNITS:
            try:
                unit = uparser.parse_unit(item.text(self.COL_UNITS))
            except ValueError:
                self.set_background(item, column, False)
            else:
                qty.outunits = str(unit)
                item.setText(self.COL_UNITS, f'{unit:~P}')
                self.set_background(item, column, True)

        elif column == self.COL_DESC:
            qty.description = item.text(self.COL_DESC)

        elif column == self.COL_TOLERANCE:
            tolerance = item.data(self.COL_TOLERANCE, SuncalDelegate.ROLE_TOLERANCE)
            qty.tolerance = tolerance

        else:
            raise NotImplementedError

    def add_quantity(self, qty: SystemIndirectQuantity):
        ''' Add a quantity to the tree '''
        item = TreeItem(
            [qty.symbol,
             qty.equation,
             str(qty.outunits) if qty.outunits else '',
             str(qty.tolerance) if qty.tolerance else '',  # Tolerance
             qty.description]
        ).enableMath(True, True)
        item.setData(0, SuncalDelegate.ROLE_QUANTITY, qty)
        item.setData(self.COL_TOLERANCE, SuncalDelegate.ROLE_TOLERANCE, qty.tolerance)
        self.addTopLevelItem(item)
        self.added.emit()

    def remove_quantity(self, qty: SystemIndirectQuantity, item: TreeItem):
        ''' Remove the quantity '''
        self.takeTopLevelItem(self.indexOfTopLevelItem(item))
        self.component.model.quantities.remove(qty)
        self.added.emit()

    def contextMenuEvent(self, event):
        ''' Show right-click context menu '''
        item = self.itemAt(event.pos())
        if item:
            menu = QtWidgets.QMenu()
            qty = item.data(0, SuncalDelegate.ROLE_QUANTITY)
            if qty:
                remove = menu.addAction('Remove Quantity')
                remove.triggered.connect(lambda x, qty=qty, item=item: self.remove_quantity(qty, item))
            menu.exec(event.globalPos())

    def moveCursor(self, cursorAction, modifiers):
        ''' Move by columns, not rows '''
        index = self.currentIndex()
        column = index.column()
        if column < self.COL_CNT:
            return index.sibling(index.row(), column+1)
        return super().moveCursor(cursorAction, modifiers)


class QtyValueDelegate(PopupDelegate):
    ''' Delegate for Quantity Value column '''
    def createEditor(self, parent, option, index):
        if index.model().data(index, PopupDelegate.ROLE_TYPEA):
            return TypeADialog(parent)
        elif index.model().data(index, PopupDelegate.ROLE_TYPEB):
            return widgets.PdfPopupDialog(parent)
        else:
            return super().createEditor(parent, option, index)

    def updateEditorGeometry(self, editor, option, index):
        if index.model().data(index, PopupDelegate.ROLE_TYPEA) or index.model().data(index, PopupDelegate.ROLE_TYPEB):
            return super().updateEditorGeometry(editor, option, index)
        editor.setGeometry(option.rect)  # no popup

    def setEditorData(self, editor, index):
        if index.model().data(index, PopupDelegate.ROLE_TYPEA):
            qty = index.model().data(index, PopupDelegate.ROLE_QUANTITY)
            editor.set_values(qty.typea)
            editor.autocorr.setChecked(qty.autocorrelation)
            if qty.num_newmeas is not None:
                editor.chkNewmeas.setChecked(True)
                editor.Nnewmeas.setValue(qty.num_newmeas)
        elif (typeb := index.model().data(index, PopupDelegate.ROLE_TYPEB)):
            qty = index.model().data(index, PopupDelegate.ROLE_QUANTITY)
            editor.dist = typeb.distribution
            editor.table.config = typeb.kwargs
            editor.table.fill_table()
        else:
            super().setEditorData(editor, index)

    def setModelData(self, editor, model, index):
        if index.model().data(index, PopupDelegate.ROLE_TYPEA):
            qty = index.model().data(index, PopupDelegate.ROLE_QUANTITY)
            qty.typea = editor.get_data()
            qty.autocorrelation = editor.autocorr.isChecked()
            qty.num_newmeas = editor.Nnewmeas.value() if editor.chkNewmeas.isChecked() else None
            size = editor.data_size()
            if size == 0:
                display = '   ± 0   ▾'
            else:
                try:
                    display = f'   ± {unitmgr.strip_units(qty.typea_uncertainty()):.4g} (k=1)   ▾'
                except (OffsetUnitCalculusError, DimensionalityError, UndefinedUnitError):
                    display = '   Units Error'
            index.model().setData(index, display, QtCore.Qt.ItemDataRole.DisplayRole)
        elif (typeb := index.model().data(index, PopupDelegate.ROLE_TYPEB)):
            qty = index.model().data(index, PopupDelegate.ROLE_QUANTITY)
            kwargs = editor.table.config
            kwargs['df'] = editor.table.degf
            if 'name' in kwargs:
                typeb.distname = kwargs.pop('name')
            typeb.set_kwargs(**kwargs)
            index.model().setData(index, f'   ±{typeb.uncertainty.magnitude:.4g} (k=1)   ▾', QtCore.Qt.ItemDataRole.DisplayRole)
        else:
            super().setModelData(editor, model, index)


class QuantityTree(QtWidgets.QTreeWidget):
    COL_SYM = 0
    COL_VALUE = 1
    COL_UNITS = 2
    COL_TOLERANCE = 3
    COL_DESC = 4
    COL_BTNA = 5
    COL_BTNB = 6
    COL_CNT = 7

    symbol_changed = QtCore.pyqtSignal()
    added = QtCore.pyqtSignal()

    def __init__(self, component: ProjectMeasSys):
        super().__init__()
        self.component = component
        self.loading: bool = False
        self._delegate = LatexDelegate()
        self._noeditdelegate = NoEditDelegate()
        self._toldelegate = ToleranceDelegate(required=False)
        self._valdelegate = QtyValueDelegate()
        self.fill_tree()
        self.itemChanged.connect(self.edit_data)

    def set_background(self, item: QtWidgets.QTreeWidgetItem, column: int, valid: bool):
        ''' Set background color of a table item '''
        item.setBackground(column, gui_styles.color.transparent if valid else gui_styles.color.invalid)
        if not valid:
            self.clearSelection()

    def fill_tree(self):
        ''' Fill tree with component values '''
        self.loading = True
        self.clear()
        self.setColumnCount(self.COL_CNT)
        self.setHeaderLabels(['Symbol', 'Value', 'Units', 'Tolerance', 'Description', '', ''])
        self.setItemDelegateForColumn(self.COL_SYM, self._delegate)
        self.setItemDelegateForColumn(self.COL_VALUE, self._valdelegate)
        self.setItemDelegateForColumn(self.COL_TOLERANCE, self._toldelegate)
        self.setColumnWidth(self.COL_SYM, 150)
        self.setColumnWidth(self.COL_VALUE, 175)
        self.setColumnWidth(self.COL_UNITS, 125)
        self.setColumnWidth(self.COL_TOLERANCE, 150)
        self.setColumnWidth(self.COL_DESC, 200)
        self.setColumnWidth(self.COL_BTNA, 20)
        self.setColumnWidth(self.COL_BTNB, 20)

        quantities = [qty for qty in self.component.model.quantities if isinstance(qty, SystemQuantity)]
        for qty in quantities:
            self.add_quantity(qty)
        self.loading = False

    def edit_data(self, item, column):
        ''' Item was changed, update the model '''
        if self.loading:
            return

        self.loading = True
        qty = item.data(0, SuncalDelegate.ROLE_QUANTITY)
        typeb = item.data(self.COL_VALUE, SuncalDelegate.ROLE_TYPEB)
        typea = item.data(self.COL_VALUE, SuncalDelegate.ROLE_TYPEA)
        if typeb is not None:
            self._edit_typeb_data(item, qty, typeb, column)
        elif typea is not None:
            self._edit_typea_data(item, qty, column)
        else:
            self._edit_quantity_data(item, qty, column)
        self.loading = False

    def _edit_quantity_data(self, item: TreeItem, qty: SystemQuantity, column: int):
        if column == self.COL_SYM:
            try:
                value = item.data(self.COL_SYM, LatexDelegate.ROLE_ENTERED)
                mathvalue = uparser.parse_math(value)
            except ValueError:
                self.set_background(item, column, False)
            else:
                if isinstance(mathvalue, sympy.Symbol):  # name must be single symbol
                    qty.symbol = value
                    self.set_background(item, column, True)
                    self.symbol_changed.emit()
                else:
                    self.set_background(item, column, False)

        elif column == self.COL_VALUE:
            try:
                value = float(item.text(self.COL_VALUE))
            except ValueError:
                self.set_background(item, column, False)
            else:
                qty.testpoint = value
                self.set_background(item, column, True)
                for typeb in qty.typebs:
                    typeb.nominal = value

        elif column == self.COL_UNITS:
            try:
                unit = uparser.parse_unit(item.text(self.COL_UNITS))
            except ValueError:
                self.set_background(item, column, False)
            else:
                qty.units = str(unit)
                item.setText(self.COL_UNITS, f'{unit:~P}')
                self.set_background(item, column, True)
                bunit = unitmgr.to_delta_units(qty.units)

                # Keep Type B units compatible
                for child in item.children():
                    typeb = child.data(self.COL_VALUE, SuncalDelegate.ROLE_TYPEB)
                    typea: bool = child.data(self.COL_VALUE, SuncalDelegate.ROLE_TYPEA)

                    if typea:
                        child.setText(self.COL_UNITS, f'{bunit:~P}')
                    elif typeb and typeb.units is None or unitmgr.is_dimensionless(typeb.units):
                        typeb.units = bunit
                        child.setText(self.COL_UNITS, f'{bunit:~P}')
                    elif typeb and typeb.units:
                        if typeb.units.dimensionality != unit.dimensionality:
                            typeb.units = bunit
                            child.setText(self.COL_UNITS, f'{bunit:~P}')

        elif column == self.COL_DESC:
            if item.data(self.COL_VALUE, SuncalDelegate.ROLE_TYPEA):
                qty.typea_description = item.text(self.COL_DESC)
            else:
                qty.description = item.text(self.COL_DESC)

        elif column == self.COL_TOLERANCE:
            qty.tolerance = item.data(self.COL_TOLERANCE, SuncalDelegate.ROLE_TOLERANCE)

        else:
            raise NotImplementedError

    def _edit_typea_data(self, item: TreeItem, qty: SystemQuantity, column: int):
        ''' Type A data was edited '''
        if qty.typea is not None:
            item.parent().setText(self.COL_VALUE, f'{unitmgr.strip_units(qty.expected()):.4g}')
            item.parent().enableEdit(True, False, True)
        else:
            item.parent().enableEdit(True, True, True)

    def _edit_typeb_data(self, item: TreeItem, qty: SystemQuantity, typeb: Typeb, column: int):
        if column == self.COL_SYM:
            typeb.name = item.data(self.COL_SYM, LatexDelegate.ROLE_ENTERED)

        elif column == self.COL_UNITS:
            unit = item.text(self.COL_UNITS)
            try:
                units = unitmgr.to_delta_units(unit)
            except ValueError:
                self.set_background(item, column, False)
            else:
                if qty.units and units.dimensionality == uparser.parse_unit(qty.units).dimensionality:
                    typeb.units = units
                    self.set_background(item, column, True)
                else:
                    self.set_background(item, column, False)

        elif column == self.COL_DESC:
            typeb.description = item.text(self.COL_DESC)

    def add_quantity(self, qty: SystemQuantity):
        ''' Add a quantity to the tree '''
        self.loading = True
        item = TreeItem(
            [qty.symbol,
             f'{qty.testpoint:.4g}',
             qty.units if qty.units else '',
             str(qty.tolerance) if qty.tolerance else '',
             qty.description]
        ).enableMath(True)
        self.addTopLevelItem(item)
        item.setData(0, SuncalDelegate.ROLE_QUANTITY, qty)
        item.setData(self.COL_TOLERANCE, SuncalDelegate.ROLE_TOLERANCE, qty.tolerance)
        self.setItemWidget(item, self.COL_BTNA, adda := widgets.RoundButton('A'))
        adda.setToolTip('Add Type A Uncertainty Data')
        adda.clicked.connect(lambda x, qty=qty, item=item: self.add_typea(qty, item))
        self.setItemWidget(item, self.COL_BTNB, addb := widgets.RoundButton('B'))
        addb.setToolTip('Add Type B Uncertainty Component')
        addb.clicked.connect(lambda x, qty=qty, item=item: self.add_uncertainty(qty, item))

        if qty.typea is not None:
            self.add_typea(qty, item)
        for typeb in qty.typebs:
            self.add_uncertainty_item(item, qty, typeb)

        self.loading = False
        self.added.emit()

    def add_uncertainty_item(self, item: TreeItem, qty: SystemQuantity, typeb: Typeb):
        ''' Add uncertainty row to the tree '''
        self.loading = True
        item.addChild(uitem := TreeItem([
            typeb.name,
            f'   ± {unitmgr.strip_units(typeb.uncertainty)} (k=1)    ▾',  # Type B
            f'{typeb.units:~P}' if typeb.units else '',
            '',  # Tolerance (NA)
            typeb.description
        ]).enableMath(True).enableEdit(True, False, True, False, True))
        uitem.setData(0, SuncalDelegate.ROLE_QUANTITY, qty)
        uitem.setData(self.COL_TOLERANCE, SuncalDelegate.ROLE_DISABLE, True)
        uitem.setData(self.COL_VALUE, SuncalDelegate.ROLE_TYPEB, typeb)
        uitem.setData(self.COL_VALUE, SuncalDelegate.ROLE_DISABLE, True)
        self.setItemWidget(uitem, self.COL_BTNB, remb := widgets.MinusButton())
        remb.setToolTip('Remove Type B Uncertainty Component')
        remb.clicked.connect(lambda x, qty=qty, typeb=typeb, item=uitem: self.remove_uncertainty(qty, item, typeb))

        item.setExpanded(True)
        self.loading = False
        self.added.emit()

    def remove_quantity(self, qty: SystemQuantity, item: TreeItem):
        ''' Remove the quantity '''
        self.takeTopLevelItem(self.indexOfTopLevelItem(item))
        self.component.model.quantities.remove(qty)
        self.symbol_changed.emit()
        self.added.emit()

    def add_uncertainty(self, qty: SystemQuantity, item: TreeItem):
        ''' Add a new uncertainty component '''
        typeb = Typeb(nominal=qty.testpoint)
        units = ''
        if qty.units is not None:
            if qty.units.lower() in ['celsius', 'degree_celsius', 'degc', '°c']:
                units = 'delta_degree_Celsius'
            elif qty.units.lower() in ['fahrenheit', 'degree_fahrenheit', 'degf', '°f']:
                units = 'delta_degree_Fahrenheit'
            else:
                units = str(qty.units)

        typeb.units = unitmgr.parse_units(units)
        qty.typebs.append(typeb)
        self.add_uncertainty_item(item, qty, typeb)
        self.added.emit()

    def remove_uncertainty(self, qty: SystemQuantity, item: TreeItem, typeb: Typeb):
        ''' Remove uncertainty component '''
        item.parent().removeChild(item)
        qty.typebs.remove(typeb)
        self.added.emit()

    def add_typea(self, qty: SystemQuantity, item: TreeItem):
        ''' Add Type A data '''
        for row in range(item.childCount()):
            if item.child(row).data(self.COL_VALUE, SuncalDelegate.ROLE_TYPEA):
                return  # Already present

        self.loading = True
        item.addChild(aitem := TreeItem([
            'Type A',
            '   ± 0   ▾',  # popup,
            '', '',
            qty.typea_description
        ]).enableEdit(False, False, False, False, False))
        aitem.setData(0, SuncalDelegate.ROLE_QUANTITY, qty)
        aitem.setData(self.COL_VALUE, SuncalDelegate.ROLE_TYPEA, True)
        aitem.setData(self.COL_VALUE, SuncalDelegate.ROLE_QUANTITY, qty)
        aitem.setData(self.COL_TOLERANCE, SuncalDelegate.ROLE_DISABLE, True)
        self.setItemWidget(aitem, self.COL_BTNB, rema := widgets.MinusButton())
        rema.setToolTip('Remove Type A Uncertainty Data')
        rema.clicked.connect(lambda x, qty=qty, item=aitem: self.remove_typea(qty, item))

        item.setExpanded(True)
        self.loading = False
        self.added.emit()

    def remove_typea(self, qty: SystemQuantity, item: TreeItem):
        ''' Remove the Type A data '''
        item.parent().removeChild(item)
        qty.typea = None
        self.added.emit()

    def contextMenuEvent(self, event):
        ''' Show right-click context menu '''
        item = self.itemAt(event.pos())
        if item:
            menu = QtWidgets.QMenu()
            qty = item.data(0, SuncalDelegate.ROLE_QUANTITY)
            typeb = item.data(self.COL_VALUE, SuncalDelegate.ROLE_TYPEB)
            typea = item.data(self.COL_VALUE, SuncalDelegate.ROLE_TYPEA)
            if typea:
                remtypea = menu.addAction('Remove Type A uncertainty data')
                remtypea.triggered.connect(lambda x, qty=qty, item=item: self.remove_typea(qty, item))
            elif typeb:
                remove = menu.addAction('Remove Uncertaity Component')
                remove.triggered.connect(lambda x, qty=qty, typeb=typeb, item=item: self.remove_uncertainty(qty, item, typeb))
            elif qty:
                addtypea = menu.addAction('Add Type &A Uncertainty Data')
                addtypea.triggered.connect(lambda x, qty=qty, item=item: self.add_typea(qty, item))
                addunc = menu.addAction('Add Type &B Uncertainty Component')
                addunc.triggered.connect(lambda x, qty=qty, item=item: self.add_uncertainty(qty, item))
                remove = menu.addAction('&Remove Quantity')
                remove.triggered.connect(lambda x, qty=qty, item=item: self.remove_quantity(qty, item))
            menu.exec(event.globalPos())

    def moveCursor(self, cursorAction, modifiers):
        ''' Move by columns, not rows '''
        index = self.currentIndex()
        column = index.column()
        if column < self.COL_BTNA:
            return index.sibling(index.row(), column+1)
        return super().moveCursor(cursorAction, modifiers)


class CurveTree(QtWidgets.QTreeWidget):
    COL_SYM = 0
    COL_MODEL = 1
    COL_DATA = 2
    COL_TOLERANCE = 3
    COL_DESC = 4
    COL_CNT = 5

    symbol_changed = QtCore.pyqtSignal()
    added = QtCore.pyqtSignal()

    def __init__(self, component: ProjectMeasSys):
        super().__init__()
        self.component = component
        self.loading: bool = False
        self._noeditdelegate = NoEditDelegate()
        self._editdelegate = EditDelegate()
        self._toldelegate = ToleranceDelegate(required=False)
        self.fill_tree()
        self.itemChanged.connect(self.edit_data)

    def set_background(self, item: QtWidgets.QTreeWidgetItem, column: int, valid: bool):
        ''' Set background color of a table item '''
        item.setBackground(column, gui_styles.color.transparent if valid else gui_styles.color.invalid)
        if not valid:
            self.clearSelection()

    def fill_tree(self):
        ''' Fill tree with component values '''
        self.loading = True
        self.clear()
        self.setColumnCount(self.COL_CNT)
        self.setHeaderLabels(['Symbol', 'Model', 'Data', 'Tolerance', 'Description'])
        self.setItemDelegateForColumn(self.COL_SYM, self._editdelegate)
        self.setItemDelegateForColumn(self.COL_MODEL, self._noeditdelegate)
        self.setItemDelegateForColumn(self.COL_TOLERANCE, self._toldelegate)
        self.setColumnWidth(self.COL_SYM, 125)
        self.setColumnWidth(self.COL_MODEL, 200)
        self.setColumnWidth(self.COL_DATA, 150)
        self.setColumnWidth(self.COL_TOLERANCE, 100)
        self.setColumnWidth(self.COL_DESC, 250)

        quantities = [qty for qty in self.component.model.quantities if isinstance(qty, SystemCurve)]
        for qty in quantities:
            self.add_quantity(qty)
        self.loading = False

    def edit_data(self, item, column):
        ''' Item was changed, update the model '''
        if self.loading:
            return
        qty: SystemCurve = item.data(0, SuncalDelegate.ROLE_QUANTITY)
        coeff: bool = item.data(0, SuncalDelegate.ROLE_COEFF)
        pred: str = item.data(0, SuncalDelegate.ROLE_PREDICT)

        if column == self.COL_DESC:
            qty.description = item.text(self.COL_DESC)

        elif coeff and column == self.COL_DATA:
            name = item.text(self.COL_SYM)
            guess = item.text(self.COL_DATA)
            try:
                guess = float(guess)
            except (TypeError, ValueError):
                self.set_background(item, self.COL_DATA, False)
            else:
                names = qty.coeff_names()
                idx = names.index(name)
                if qty.guess is None:
                    qty.guess = [1] * len(names)
                if len(qty.guess) < len(names):
                    qty.guess = qty.guess + [1] * (len(names) - len(qty.guess))
                qty.guess[idx] = guess
                self.set_background(item, self.COL_DATA, True)

        elif coeff and column == self.COL_TOLERANCE:
            tol = item.data(self.COL_TOLERANCE, SuncalDelegate.ROLE_TOLERANCE)
            qty.tolerances[coeff] = tol

        elif pred and column in [self.COL_SYM, self.COL_DATA, self.COL_TOLERANCE]:
            value = item.text(self.COL_DATA)
            tol = item.data(self.COL_TOLERANCE, SuncalDelegate.ROLE_TOLERANCE)
            name = item.text(self.COL_SYM)
            try:
                value = float(value)
            except (ValueError, TypeError):
                self.set_background(item, self.COL_DATA, False)
            else:
                qty.predictions.pop(pred, None)
                qty.predictions[name] = (value, tol)
                item.setData(0, SuncalDelegate.ROLE_PREDICT, name)
                self.set_background(item, self.COL_DATA, True)

    def add_quantity(self, qty: SystemCurve):
        ''' Add a quantity to the tree '''
        self.loading = True
        item = TreeItem([
            'Curve',
            '',  # Model combo
            '',  # Data popup
            '',  # Tolerance
            qty.description
        ]).enableEdit(False, False, False, False, True)
        self.addTopLevelItem(item)
        item.setData(0, SuncalDelegate.ROLE_QUANTITY, qty)
        item.setData(self.COL_TOLERANCE, SuncalDelegate.ROLE_DISABLE, True)
        model = QtWidgets.QComboBox()
        fitmodels = ['Line', 'Quadratic', 'Cubic', 'Quartic', 'Exponential', 'Exponential Decay', 'Log', 'Logistic Growth', 'Custom']
        model.addItems(fitmodels)
        cmodel = {
            'line': 'Line',
            'exp': 'Exponential',
            'decay': 'Exponential Decay',
            'log': 'Log',
            'logistic': 'Logistic Growth',
            'quad': 'Quadratic',
            'cubic': 'Cubic',
        }.get(qty.fitmodel, 'Custom')
        model.setCurrentText(cmodel)
        model.currentIndexChanged.connect(lambda x, qty=qty, widget=model, item=item: self.change_model(qty, model, item))
        data = QtWidgets.QToolButton()
        data.setText('Curve Data...')
        data.clicked.connect(lambda: MeasSysCurveWidget(qty, parent=self).exec())
        self.setItemWidget(item, self.COL_MODEL, model)
        self.setItemWidget(item, self.COL_DATA, data)
        self.add_child_items(qty, item)
        self.loading = False
        self.added.emit()

    def add_child_items(self, qty, item):
        ''' Add coefficients and predictions as tree children '''
        for name in qty.coeff_names():
            self.add_coefficient(name, qty, item)

        for name, (value, tolerance) in qty.predictions.values():
            self.add_prediction_item(name, value, tolerance, qty, item)

        item.setExpanded(True)

    def change_model(self, qty: SystemCurve, model: QtWidgets.QComboBox, item: TreeItem):
        ''' The fit model was changed '''
        fit = {
            'Line': 'line',
            'Exponential': 'exp',
            'Exponential Decay': 'decay',
            'Quadratic': 'poly',
            'Cubic': 'poly',
            'Quartic': 'poly',
            'Log': 'log',
            'Logistic Growth': 'logistic',
            'Custom': 'custom'
        }.get(model.currentText(), 'line')
        order = {
            'Quadratic': 2,
            'Cubic': 3,
            'Quartic': 4
        }.get(model.currentText(), 2)

        if fit == 'custom':
            fit, ok = QtWidgets.QInputDialog.getText(self, 'Custom Fit Model', 'Enter fit model as function of x')
            if not ok:
                return

        try:
            qty.polyorder = order
            qty.set_fitmodel(fit)
        except (TypeError, ValueError) as exc:
            QtWidgets.QMessageBox.warning(self, 'Suncal', exc)
        else:
            item.takeChildren()
            self.add_child_items(qty, item)

    def add_coefficient(self, name: str, qty: SystemCurve, item: TreeItem):
        ''' Add a coefficient row '''
        self.loading = True
        allnames = qty.coeff_names()
        idx = allnames.index(name)
        try:
            guess = qty.guess[idx]
        except (TypeError, AttributeError, IndexError):
            guess = 1.0

        tol = qty.tolerances.get(name)
        item.addChild(citem := TreeItem([
            name,
            'Fit Coefficient',  # no model
            str(guess),
            str(tol) if tol else '',
            qty.descriptions.get(name, '')  # Description
        ]).enableEdit(False, False, True, False, True))
        item.setData(0, SuncalDelegate.ROLE_QUANTITY, qty)
        citem.setData(0, SuncalDelegate.ROLE_COEFF, True)
        citem.setData(0, SuncalDelegate.ROLE_QUANTITY, qty)
        citem.setData(self.COL_TOLERANCE, SuncalDelegate.ROLE_TOLERANCE, tol)
        self.loading = False

    def add_prediction_item(self, name: str, value: float, tolerance: Limit, qty: SystemCurve, item: TreeItem):
        ''' Add a prediction value '''
        self.loading = True
        item.addChild(pitem := TreeItem([
            name,
            'Estimate at x =',
            str(value),
            str(tolerance) if tolerance else '',
            qty.descriptions.get(name, '')  # Description
        ]).enableEdit(True, False, True, False, True))
        pitem.setData(0, SuncalDelegate.ROLE_QUANTITY, qty)
        pitem.setData(0, SuncalDelegate.ROLE_PREDICT, name)
        pitem.setData(self.COL_TOLERANCE, SuncalDelegate.ROLE_TOLERANCE, tolerance)
        self.loading = False
        self.added.emit()

    def remove_quantity(self, qty: SystemCurve, item: TreeItem):
        ''' Remove the quantity '''
        self.takeTopLevelItem(self.indexOfTopLevelItem(item))
        self.component.model.quantities.remove(qty)
        self.added.emit()

    def add_prediction(self, qty: SystemCurve, item: TreeItem):
        ''' Add new precition value '''
        name = 0
        while f'x{name}' in qty.predictions:
            name += 1
        qty.predictions[f'x{name}'] = (0, None)
        self.add_prediction_item(f'x{name}', 0, None, qty, item)

    def remove_prediction(self, qty: SystemCurve, item: TreeItem, name: str):
        ''' Remove the prediction value '''
        qty.predictions.pop(name, None)
        item.parent().removeChild(item)
        self.added.emit()

    def contextMenuEvent(self, event):
        ''' Show right-click context menu '''
        item = self.itemAt(event.pos())
        if item:
            menu = QtWidgets.QMenu()
            qty = item.data(0, SuncalDelegate.ROLE_QUANTITY)
            coeff = item.data(0, SuncalDelegate.ROLE_COEFF)
            pred = item.data(0, SuncalDelegate.ROLE_PREDICT)
            if qty and not coeff and not pred:
                addpred = menu.addAction('Add Estimate Value')
                addpred.triggered.connect(lambda x, qty=qty, item=item: self.add_prediction(qty, item))
                remove = menu.addAction('Remove Quantity')
                remove.triggered.connect(lambda x, qty=qty, item=item: self.remove_quantity(qty, item))
            elif qty and pred:
                remove = menu.addAction('Remove Estimate Value')
                remove.triggered.connect(lambda x, qty=qty, item=item, pred=pred: self.remove_prediction(qty, item, pred))
            menu.exec(event.globalPos())

    def moveCursor(self, cursorAction, modifiers):
        ''' Move by columns, not rows '''
        index = self.currentIndex()
        column = index.column()
        if column < self.COL_CNT:
            return index.sibling(index.row(), column+1)
        return super().moveCursor(cursorAction, modifiers)


class MeasSysInput(QtWidgets.QWidget):
    ''' Widget for editing a full Measurement System '''
    change_help = QtCore.pyqtSignal()
    calculate = QtCore.pyqtSignal()

    def __init__(self, component: ProjectMeasSys, parent=None):
        super().__init__(parent)
        assert isinstance(component, ProjectMeasSys)
        self.loading: bool = False
        self.component = component
        font = self.font()
        font.setPointSize(12)
        self.setFont(font)
        self.btn_addqty = widgets.SmallToolButton('tolerance')
        self.btn_addeqn = widgets.SmallToolButton('equation')
        self.btn_addcurve = widgets.SmallToolButton('curve')
        self.btn_showrpt = widgets.SmallToolButton('report')
        self.btn_calc = QtWidgets.QPushButton('Calculate')
        self.btn_addqty.setToolTip('Add a direct-measured quantity.')
        self.btn_addeqn.setToolTip('Add an indirect measurement quantity, calculated via equation.')
        self.btn_addcurve.setToolTip('Add a curve fit calculation.')
        font = QtGui.QFont('Arial', 14)
        self.tree_func = FunctionTree(component)
        self.tree_qty = QuantityTree(component)
        self.tree_curve = CurveTree(component)
        self.label_func = QtWidgets.QLabel("<b>Measurement Equations</b>")
        self.label_qty = QtWidgets.QLabel("<b>Measured Quantities</b>")
        self.label_curve = QtWidgets.QLabel("<b>Curve Fit Quantities</b>")
        self.corrwidget = CorrelationWidget()
        self.corrwidget.setVisible(False)

        layout = QtWidgets.QVBoxLayout()
        blayout = QtWidgets.QHBoxLayout()
        blayout.addWidget(self.btn_addqty)
        blayout.addWidget(self.btn_addeqn)
        blayout.addWidget(self.btn_addcurve)
        blayout.addStretch()
        layout.addLayout(blayout)
        layout.addWidget(self.label_qty)
        layout.addWidget(self.tree_qty, stretch=2)
        layout.addWidget(self.corrwidget, stretch=1)
        layout.addWidget(self.label_curve)
        layout.addWidget(self.tree_curve, stretch=2)
        layout.addWidget(self.label_func)
        layout.addWidget(self.tree_func, stretch=2)
        layout.addStretch(stretch=0)
        b2layout = QtWidgets.QHBoxLayout()
        b2layout.addStretch()
        b2layout.addWidget(self.btn_calc)
        layout.addLayout(b2layout)
        self.setLayout(layout)

        self.tree_func.equation_changed.connect(self.add_missing_symbols)
        self.tree_qty.symbol_changed.connect(self.add_missing_symbols)
        self.tree_func.added.connect(self.set_visibility)
        self.tree_qty.added.connect(self.set_visibility)
        self.tree_curve.added.connect(self.set_visibility)
        self.corrwidget.changed.connect(self.set_correlation)
        self.btn_addeqn.clicked.connect(self.add_indirect)
        self.btn_addqty.clicked.connect(self.add_quantity)
        self.btn_addcurve.clicked.connect(self.add_curve)
        self.btn_calc.clicked.connect(self.calculate)
        self.set_visibility()

    def clear(self):
        ''' Clear everything '''
        self.tree_func.clear()
        self.tree_qty.clear()
        self.tree_curve.clear()
        self.set_visibility()

    def set_visibility(self):
        ''' Show/hide the trees '''
        qtys = [q for q in self.component.model.quantities if isinstance(q, SystemQuantity)]
        show_qty = len(qtys) > 0
        show_func = len([q for q in self.component.model.quantities if isinstance(q, SystemIndirectQuantity)]) > 0
        show_curve = len([q for q in self.component.model.quantities if isinstance(q, SystemCurve)]) > 0
        self.tree_func.setVisible(show_func)
        self.tree_qty.setVisible(show_qty)
        self.tree_curve.setVisible(show_curve)
        self.label_func.setVisible(show_func)
        self.label_qty.setVisible(show_qty)
        self.label_curve.setVisible(show_curve)
        names = [q.symbol for q in qtys]
        self.corrwidget.table.setVarNames(names)

    def add_missing_symbols(self):
        ''' Check measurement model and add any missing variables '''
        try:
            missing = self.component.model.missing_symbols()
        except ValueError as exc:
            # Possibly recursive function
            QtWidgets.QMessageBox.warning(self, 'Suncal', str(exc))
        else:
            if missing:
                for v in missing:
                    qty = SystemQuantity(v)
                    self.component.model.quantities.append(qty)
                self.tree_qty.fill_tree()

    def add_indirect(self):
        ''' Add an indirect quantity '''
        qty = SystemIndirectQuantity()
        qty.symbol = self.component.model.unused_symbol()
        self.component.model.quantities.append(qty)
        self.tree_func.add_quantity(qty)

    def add_quantity(self):
        ''' Add a Quantity to the system '''
        qty = SystemQuantity()
        qty.symbol = self.component.model.unused_symbol()
        self.component.model.quantities.append(qty)
        self.tree_qty.add_quantity(qty)

    def add_curve(self):
        ''' Add a CurveFit to the system '''
        qty = SystemCurve()
        self.component.model.quantities.append(qty)
        self.tree_curve.fill_tree()

    def set_correlation(self):
        ''' Set correlation coefficients '''
        corr = self.corrwidget.table.get_config()
        self.component.model.correlations = corr


class MeasSysOutput(QtWidgets.QWidget):
    ''' Output page for measurement system '''
    goback = QtCore.pyqtSignal()
    change_help = QtCore.pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.result = None
        self.loading: bool = True
        self.cmb_qty = QtWidgets.QComboBox()
        self.report = widgets.MarkdownTextEdit()
        self.gumout = PageGumOutput()
        self.curveout = PageCurveOutput()
        self.stack = QtWidgets.QStackedWidget()
        self.stack.addWidget(self.report)
        self.stack.addWidget(self.gumout)
        self.stack.addWidget(self.curveout)
        self.btn_back = QtWidgets.QPushButton('Back')
        self.btn_back.setSizePolicy(QtWidgets.QSizePolicy.Policy.Fixed, QtWidgets.QSizePolicy.Policy.Fixed)
        toplayout = QtWidgets.QHBoxLayout()
        toplayout.addWidget(QtWidgets.QLabel('Quantity:'))
        toplayout.addWidget(self.cmb_qty)
        toplayout.addStretch()
        layout = QtWidgets.QVBoxLayout()
        layout.addLayout(toplayout)
        layout.addWidget(self.stack)
        layout.addWidget(self.btn_back)
        self.setLayout(layout)

        self.btn_back.clicked.connect(self.goback)
        self.cmb_qty.currentIndexChanged.connect(self.select_quantity)
        self.gumout.back.connect(self.goback)
        self.curveout.btnBack.clicked.connect(self.goback)
        self.gumout.change_help.connect(self.change_help)
        self.curveout.change_help.connect(self.change_help)

    def set_result(self, result: 'SystemResult'):
        self.loading = True
        self.result = result
        symbols = [r.symbol for r in self.result.quantities]
        self.cmb_qty.clear()
        self.cmb_qty.addItems(['Summary'] + symbols)
        self.loading = False
        self.select_quantity()

    def select_quantity(self):
        ''' Quantity was selected from the dropdown '''
        if not self.loading:
            if self.cmb_qty.currentIndex() == 0:  # Summary
                self.btn_back.setVisible(True)
                self.stack.setCurrentIndex(0)
                self.report.setReport(self.result.report.summary())
            else:
                qtyname = self.cmb_qty.currentText()
                qty_result = self.result.get_result(qtyname)
                if isinstance(qty_result.qty, SystemQuantity):
                    self.stack.setCurrentIndex(0)
                    self.btn_back.setVisible(True)
                    self.report.setReport(qty_result.report.report_all())
                elif isinstance(qty_result.qty, SystemIndirectQuantity):
                    self.stack.setCurrentIndex(1)
                    self.btn_back.setVisible(False)
                    self.gumout.update(qty_result.meta.get('gumresult'))
                elif isinstance(qty_result.qty, SystemCurve):
                    self.stack.setCurrentIndex(2)
                    self.btn_back.setVisible(False)
                    self.curveout.set_output(qty_result.meta.get('fitresult'))
                else:
                    self.report.setText('No good')
                    self.btn_back.setVisible(True)
            self.change_help.emit()


class CorrelationWidget(QtWidgets.QWidget):
    changed = QtCore.pyqtSignal()

    def __init__(self):
        super().__init__()
        self.buttons = widgets.PlusMinusButton()
        self.table = CorrelationTableWidget()
        hlayout = QtWidgets.QHBoxLayout()
        hlayout.addWidget(QtWidgets.QLabel('<b>Correlation Coefficients</b>'))
        hlayout.addWidget(self.buttons)
        layout = QtWidgets.QVBoxLayout()
        layout.addLayout(hlayout)
        layout.addWidget(self.table)
        layout.addStretch()
        self.setLayout(layout)
        self.buttons.plusclicked.connect(self.table.addRow)
        self.buttons.minusclicked.connect(self.table.remRow)
        self.table.itemChanged.connect(self.changed)
        self.table.changed.connect(self.changed)
        self.buttons.minusclicked.connect(self.changed)
        self.buttons.plusclicked.connect(self.changed)


class MeasSysPage(QtWidgets.QWidget):
    ''' Main All-in-one Measurement System Page '''
    PG_INPUT = 0
    PG_OUTPUT = 1
    change_help = QtCore.pyqtSignal()

    def __init__(self, component: ProjectMeasSys, parent=None):
        super().__init__(parent=parent)
        self.component = component
        self.pg_input = MeasSysInput(component, parent=self)
        self.pg_output = MeasSysOutput(parent=self)
        self.stack = widgets.SlidingStackedWidget()
        self.stack.addWidget(self.pg_input)
        self.stack.addWidget(self.pg_output)
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.stack)
        self.setLayout(layout)
        self.pg_input.calculate.connect(self.calculate)
        self.pg_output.goback.connect(self.goback)

        self.menu = QtWidgets.QMenu('&Uncertainty')
        self.actChkUnits = QtGui.QAction('Check &Units...', self)
        self.actClear = QtGui.QAction('&Clear quantities', self)
        self.actCorr = QtGui.QAction('Show Co&rrelations', self)
        self.actCorr.setCheckable(True)
        self.actOpts = QtGui.QAction('&Options...', self)
        self.actSaveReport = QtGui.QAction('&Save Report...', self)
        self.menu.addAction(self.actChkUnits)
        self.menu.addAction(self.actClear)
        self.menu.addAction(self.actOpts)
        self.menu.addAction(self.actCorr)
        self.menu.addSeparator()
        self.menu.addAction(self.actSaveReport)
        self.actClear.triggered.connect(self.clearinput)
        self.actChkUnits.triggered.connect(self.checkunits)
        self.actSaveReport.triggered.connect(self.save_report)
        self.actOpts.triggered.connect(self.show_options)
        self.actCorr.triggered.connect(self.show_correlation)
        self.pg_output.change_help.connect(self.change_help)

    def get_menu(self):
        ''' Get the menu for this widget '''
        return self.menu

    def update_proj_config(self):
        ''' Save page setup back to project item configuration '''
        # GUI updates model in real time - nothing to do

    def checkunits(self):
        ''' Show units/dimensionality report '''
        self.update_proj_config()
        dlg = widgets.MarkdownTextEdit()
        dlg.setMinimumSize(800, 600)
        dlg.setReport(self.component.units_report())
        dlg.show()

    def save_report(self):
        ''' Save full report, asking user for settings/filename '''
        with gui_styles.LightPlotstyle():
            result = self.component.calculate()
            widgets.savereport(result.report.all_summaries())

    def clearinput(self):
        ''' Clear all the input/output values '''
        self.component.model.quantities = []
        self.pg_input.clear()
        self.stack.slideInRight(self.PG_INPUT)

    def show_correlation(self):
        ''' Show/hide correlation widget '''
        self.pg_input.corrwidget.setVisible(self.actCorr.isChecked())

    def show_options(self):
        ''' Show the options dialog '''
        dlg = Settings(self.component.model)
        if dlg.exec():
            try:
                seed = int(float(dlg.seed.text()))
            except (TypeError, ValueError, OverflowError):
                seed = gui_settings.randomseed
            try:
                samples = int(float(dlg.samples.value()))
            except (TypeError, ValueError, OverflowError):
                samples = gui_settings.samples
            self.component.model.seed = seed
            self.component.model.samples = samples
            self.component.model.correlate_typeas = dlg.chkCorrelate.isChecked()
            self.component.model.confidence = dlg.conf.value()

    def calculate(self):
        ''' Calculate the results '''
        try:
            result = self.component.calculate()
        except OffsetUnitCalculusError as exc:
            badunit = re.findall(r'\((.+ )', str(exc))[0].split()[0].strip(', ')
            QtWidgets.QMessageBox.warning(self, 'Suncal', f'Ambiguous unit {badunit}. Try "delta_{badunit}".')
        except (DimensionalityError, UndefinedUnitError) as exc:
            QtWidgets.QMessageBox.warning(self, 'Suncal', f'Units Error: {exc}')
        except TypeError as exc:
            if 'Improper input' in str(exc):
                QtWidgets.QMessageBox.warning(self, 'Suncal', 'Polynomial is overfit. Reduce order.')
            else:
                QtWidgets.QMessageBox.warning(self, 'Suncal', 'Error computing curve fit solution.')
        except RecursionError:
            QtWidgets.QMessageBox.warning(self, 'Suncal', 'Error - possible circular reference in function definitions')
        except RuntimeError as exc:
            if 'Optimal parameters' in str(exc):
                QtWidgets.QMessageBox.warning(self, 'Suncal', 'Curve fit failed to converge.')
            else:
                QtWidgets.QMessageBox.warning(self, 'Suncal', f'Error: {exc}')
        except ValueError as exc:
            if 'beta0' in str(exc):
                QtWidgets.QMessageBox.warning(self, 'Suncal', 'Please provide initial guess for curve fit.')
            elif 'operands' in str(exc):
                QtWidgets.QMessageBox.warning(self, 'Suncal', 'Curve columns must have equal lengths.')
            elif 'semidefinite' in str(exc):
                QtWidgets.QMessageBox.warning(self, 'Suncal', 'Invalid value in correlation - matrix not symmetric positive semidefinite.')
            else:
                QtWidgets.QMessageBox.warning(self, 'Suncal', f'Invalid expression')
                print(exc)
        else:
            self.pg_output.set_result(result)
            self.stack.slideInLeft(self.PG_OUTPUT)
            self.change_help.emit()

    def goback(self):
        ''' Return to input page '''
        self.stack.slideInRight(self.PG_INPUT)
        self.change_help.emit()

    def help_report(self):
        ''' Get the help report to display the current widget mode '''
        if self.stack.m_next == self.PG_INPUT:
            return SystemHelp.main()
        if self.pg_output.stack.currentIndex() == 1:
            return self.pg_output.gumout.help_report()
        elif self.pg_output.stack.currentIndex() == 2:
            if self.pg_output.curveout.outSelect.currentText() == 'Fit Plot':
                return CurveHelp.fit()
            elif self.pg_output.curveout.outSelect.currentText() == 'Estimates':
                return CurveHelp.prediction()
            elif self.pg_output.curveout.outSelect.currentText() == 'Waveform Features':
                return CurveHelp.waveform()
            elif self.pg_output.curveout.outSelect.currentText() == 'Interval':
                return CurveHelp.interval()
            elif self.pg_output.curveout.outSelect.currentText() == 'Residuals':
                return CurveHelp.residuals()
            elif self.pg_output.curveout.outSelect.currentText() == 'Correlations':
                return CurveHelp.correlations()
            elif self.pg_output.curveout.outSelect.currentText() == 'Monte Carlo':
                return CurveHelp.montecarlo()
        return SystemHelp.nohelp()


if __name__ == '__main__':
    import sys
    app = QtWidgets.QApplication(sys.argv)

    proj = ProjectMeasSys()
    main = MeasSysPage(proj)

    gui_common.centerWindow(main, 1200, 900)
    font = main.font()
    font.setPointSize(10)
    main.setFont(font)

    main.show()
    app.exec()
