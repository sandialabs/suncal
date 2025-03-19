''' Page for MQA '''
from PyQt6 import QtWidgets, QtCore, QtGui

from ..project import ProjectMqa
from ..uncertainty import Model
from ..mqa.mqa import MqaQuantity
from ..mqa.measure import MTE
from ..mqa.guardband import MqaGuardbandRule
from . import gui_common
from .gui_common import BlockedSignals
from . import gui_styles
from .gui_settings import gui_settings
from . import widgets
from .widgets.mqa import TypeBWidget, ToleranceDelegate, CheckPdf, ToleranceCheck
from .widgets.equipment import EquipmentSelect
from .help_strings import MqaHelp
from .delegates import SuncalDelegate, DropdownDelegate, PopupDelegate, NoEditDelegate


class EoprDelegate(DropdownDelegate):
    ''' Delegate for EOPR tree column '''
    def dropdown_menu(self, event, mode, option, index):
        ''' Show True/Observed popup menu '''
        menu = QtWidgets.QMenu()
        rtrue = index.model().data(index, SuncalDelegate.ROLE_TRUEEOPR)
        if rtrue is None:
            rtrue = False
        aobs = menu.addAction('Observed')
        atrue = menu.addAction('True')
        aobs.setCheckable(True)
        atrue.setCheckable(True)
        aobs.setChecked(not rtrue)
        atrue.setChecked(rtrue)
        atrue.triggered.connect(lambda x, index=index: index.model().setData(index, True, SuncalDelegate.ROLE_TRUEEOPR))
        aobs.triggered.connect(lambda x, index=index: index.model().setData(index, False, SuncalDelegate.ROLE_TRUEEOPR))
        menu.exec(event.globalPosition().toPoint())


class CostDelegate(PopupDelegate):
    ''' Delegate for Cost tree column '''
    def createEditor(self, parent, option, index):
        ''' Show cost editor '''
        editor = TabCosts(parent)
        editor.setAutoFillBackground(True)
        return editor

    def setEditorData(self, editor, index):
        ''' Fill cost editor data '''
        qty: MqaQuantity = index.model().data(index, SuncalDelegate.ROLE_QUANTITY)
        editor.set_quantity(qty)

    def setModelData(self, editor, model, index):
        ''' Save cost data to quantity '''
        qty: MqaQuantity = index.model().data(index, SuncalDelegate.ROLE_QUANTITY)
        qty.costs.annual.nuut = editor.nuut.value()
        qty.costs.annual.suut = editor.suut.value()
        qty.costs.annual.uut = editor.uut.value()
        qty.costs.annual.cal = editor.cal.value()
        qty.costs.annual.repair = editor.repair.value()
        qty.costs.annual.adjust = editor.adjust.value()
        qty.costs.annual.spare_startup = editor.spare_startup.value()
        qty.costs.annual.downtime.cal = editor.down_cal.value()
        qty.costs.annual.downtime.adj = editor.down_adj.value()
        qty.costs.annual.downtime.rep = editor.down_rep.value()
        qty.costs.item.cfa = editor.cf.value()
        qty.costs.annual.pe = editor.pe.value()
        self.emit_change(index)


class EquipSpec(QtWidgets.QWidget):
    ''' Widget for entering equipment spec as a ± value with reliability '''
    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.plusminus = widgets.FloatLineEdit('1', low=0)
        self.eopr = widgets.FloatLineEdit('91.67', low=0, high=100)
        layout = QtWidgets.QHBoxLayout()
        layout.setSpacing(0)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(QtWidgets.QLabel('±'))
        layout.addWidget(self.plusminus)
        layout.addWidget(QtWidgets.QLabel('@'))
        layout.addWidget(self.eopr)
        layout.addWidget(QtWidgets.QLabel('%'))
        layout.addStretch()
        self.setLayout(layout)


class EquipDelegate(DropdownDelegate, PopupDelegate):
    ''' Delegate for Equipment spec tree column '''
    uncert_source_changed = QtCore.pyqtSignal(object, object)  # qty, index

    def __init__(self, showindirect: bool = True):
        super().__init__()
        self.showindirect = showindirect

    def createEditor(self, parent, option, index):
        ''' Create the editor widget for the given equipment type '''
        qty: MqaQuantity = index.model().data(index, SuncalDelegate.ROLE_QUANTITY)
        mode = index.model().data(index, SuncalDelegate.ROLE_EQUIP_MODE)
        if mode in [None, 'tolerance']:
            editor = EquipSpec(parent)
            editor.setAutoFillBackground(True)
            return editor
        elif mode == 'equipment':
            editor = EquipmentSelect(qty.equipmentlist, parent=parent)
            editor.setAutoFillBackground(True)
            editor.selected.connect(lambda equip, index=index, qty=qty, editor=editor: self.equipment_selected(equip, qty, index, editor))
            return editor
        elif mode == 'indirect':
            editor = IndirectWidget(qty, parent=parent)
            editor.indirect_done.connect(lambda editor=editor: self.indirect_done(editor))
            editor.setAutoFillBackground(True)
            return editor
        return None

    def indirect_done(self, editor):
        ''' User finished editing an indirect equipment spec '''
        self.commitData.emit(editor)
        self.closeEditor.emit(editor, QtWidgets.QStyledItemDelegate.EndEditHint.NoHint)

    def equipment_selected(self, equiprng, qty, index, editor):
        ''' Equipment was selected from the popup '''
        qty.measurement.mte.equipment = equiprng
        self.set_displayrole(index, str(equiprng))
        self.closeEditor.emit(editor, QtWidgets.QStyledItemDelegate.EndEditHint.NoHint)

    def updateEditorGeometry(self, editor, option, index):
        ''' Set the editor geometry '''
        mode = index.model().data(index, SuncalDelegate.ROLE_EQUIP_MODE)
        if mode in [None, 'tolerance']:
            editor.setGeometry(option.rect)
        elif mode in ['equipment', 'indirect']:
            super().updateEditorGeometry(editor, option, index)

    def setEditorData(self, editor, index):
        ''' Fill editor widget with data from quantity '''
        qty: MqaQuantity = index.model().data(index, SuncalDelegate.ROLE_QUANTITY)
        mode = index.model().data(index, SuncalDelegate.ROLE_EQUIP_MODE)
        if mode in [None, 'tolerance']:
            editor.plusminus.setValue(qty.measurement.mte.accuracy_plusminus)
            editor.eopr.setValue(qty.measurement.mte.accuracy_eopr * 100)

    def dropdown_menu(self, event, mode, option, index):
        ''' Show the equipment type menu '''
        qty: MqaQuantity = index.model().data(index, SuncalDelegate.ROLE_QUANTITY)
        menu = QtWidgets.QMenu()
        mtol = menu.addAction('Tolerance')
        mequip = menu.addAction('Equipment List')
        mqty = menu.addAction('Another quantity')
        if self.showindirect:
            mind = menu.addAction('Indirect measurement')
            mind.setCheckable(True)
            mind.triggered.connect(lambda x, index=index, qty=qty: self.change_mte('indirect', qty, index))

        mtol.setCheckable(True)
        mequip.setCheckable(True)
        mqty.setCheckable(True)

        mode = index.model().data(index, SuncalDelegate.ROLE_EQUIP_MODE)
        if mode == 'quantity':
            mqty.setChecked(True)
        elif mode == 'indirect' and self.showindirect:
            mind.setChecked(True)
        elif mode == 'equipment':
            mequip.setChecked(True)
        else:
            mtol.setChecked(True)
        mqty.triggered.connect(lambda x, index=index, qty=qty: self.change_mte('quantity', qty, index))
        mtol.triggered.connect(lambda x, index=index, qty=qty: self.change_mte('tolerance', qty, index))
        mequip.triggered.connect(lambda x, index=index, qty=qty: self.change_mte('equipment', qty, index))
        menu.exec(event.globalPosition().toPoint())

    def change_mte(self, mode: str, qty: MqaQuantity, index):
        ''' The M&TE type was changed '''
        index.model().setData(index, mode, SuncalDelegate.ROLE_EQUIP_MODE)
        if mode == 'quantity':
            if qty.measurement.mte.quantity is None:
                calmte = MqaQuantity()
                calmte.child = qty
                calmte.equipmentlist = qty.equipmentlist
                calmte.enditem = False
                calmte.measurand.testpoint = qty.measurand.testpoint
                calmte.measurand.name = f'Calibration system for {qty.measurand.name}'
                calmte.measurand.units = qty.measurand.units
                qty.measurement.mte.quantity = calmte
            self.set_displayrole(index, 'Quantity ↓')
        elif mode == 'equipment':
            self.set_displayrole(index, str(qty.measurement.mte.equipment))
        elif mode == 'indirect':
            qty.measurement.mte.quantity = None
            self.set_displayrole(index, f'± {qty.measurement.mte_uncertainty(qty.measurand.testpoint)[1].expanded():.2g}')
        else:  # Tolerance
            qty.measurement.equation = None
            qty.measurement.mte.equipment = None
            qty.measurement.mte.quantity = None
            self.set_displayrole(index, f'± {qty.measurement.mte.accuracy_plusminus}')
        self.uncert_source_changed.emit(qty, index)

    def setModelData(self, editor, model, index):
        ''' Set the equipment spec in the quantity '''
        mode = index.model().data(index, SuncalDelegate.ROLE_EQUIP_MODE)
        qty: MqaQuantity = index.model().data(index, SuncalDelegate.ROLE_QUANTITY)
        if mode in [None, 'tolerance']:
            qty.measurement.mte.accuracy_eopr = editor.eopr.value() / 100
            qty.measurement.mte.accuracy_plusminus = editor.plusminus.value()
        self.emit_change(index)
        self.uncert_source_changed.emit(qty, index)


class GuardbandDelegate(DropdownDelegate):
    ''' Delegate for Guardbanding tree column '''
    def createEditor(self, parent, option, index):
        ''' Create the guardbanding tolerance editor '''
        editor = widgets.ToleranceWidget(required=False, parent=parent)
        editor.setAutoFillBackground(True)
        editor.cancel.connect(lambda editor=editor: self.cancel_edit(editor))
        return editor

    def dropdown_menu(self, event, mode, option, index):
        ''' Show guardbanding method/rule dropdown '''
        qty: MqaQuantity = index.model().data(index, SuncalDelegate.ROLE_QUANTITY)
        component: ProjectMqa = index.model().data(index, SuncalDelegate.ROLE_COMPONENT)

        menu = QtWidgets.QMenu()
        mmanual = menu.addAction('Manual')
        mnone = menu.addAction('None')
        mmanual.setCheckable(True)
        mnone.setCheckable(True)
        if qty.guardband.method == 'none':
            mnone.setChecked(True)
        elif qty.guardband.method == 'manual':
            mmanual.setChecked(True)
        mmanual.triggered.connect(lambda x, index=index, qty=qty: self.set_gbrule('manual', None, qty, index))
        mnone.triggered.connect(lambda x, index=index, qty=qty: self.set_gbrule('none', None, qty, index))
        for rule in component.model.gbrules.rule_names:
            mitem = menu.addAction(rule)
            mitem.setCheckable(True)
            mitem.triggered.connect(lambda x, index=index, rule=rule, qty=qty: self.set_gbrule('auto', rule, qty, index))
            if qty.guardband.method == 'auto' and qty.guardband.rule.name == rule:
                mitem.setChecked(True)
        menu.exec(event.globalPosition().toPoint())

    def setEditorData(self, editor, index):
        ''' Fill the tolerance widget with guardband from the quantity '''
        qty: MqaQuantity = index.model().data(index, SuncalDelegate.ROLE_QUANTITY)
        if qty.guardband.method == 'manual':
            editor.set_limit(qty.guardband.accept_limit)
        else:
            editor.set_limit(qty.result.guardband)

    def setModelData(self, editor, model, index):
        ''' Store the acceptance limit back in the quantity '''
        qty: MqaQuantity = index.model().data(index, SuncalDelegate.ROLE_QUANTITY)
        if editor.valid:
            qty.guardband.method = 'manual'
            qty.guardband.accept_limit = editor.limit()
        else:  # Clear button pressed
            qty.guardband.method = 'none'
        self.emit_change(index)

    def cancel_edit(self, editor):
        ''' Editing was canceled, remove the guardband. '''
        editor.valid = False
        self.commitData.emit(editor)
        self.closeEditor.emit(editor, QtWidgets.QStyledItemDelegate.EndEditHint.NoHint)

    def set_gbrule(self, method, rule, qty, index):
        ''' Guardbanding rule was changed '''
        qty.guardband.method = method
        component = index.model().data(index, SuncalDelegate.ROLE_COMPONENT)
        if method == 'auto':
            qty.guardband.rule = component.model.gbrules.by_name(rule)
        self.emit_change(index)


class CalibrationDelegate(DropdownDelegate, PopupDelegate):
    ''' Delegate for calibration tree column '''
    calc_interval = QtCore.pyqtSignal(object, object)  # qty, index
    calc_eopr = QtCore.pyqtSignal(object, object)   # qty, index

    def createEditor(self, parent, option, index):
        ''' Create the calibration editor widget '''
        editor = TabCalibration(parent)
        editor.setAutoFillBackground(True)
        return editor

    def setEditorData(self, editor, index):
        ''' Fill editor with data from quantity '''
        qty: MqaQuantity = index.model().data(index, SuncalDelegate.ROLE_QUANTITY)
        editor.set_quantity(qty)

    def dropdown_menu(self, event, mode, option, index):
        ''' Show interval commands in dropdown menu '''
        qty: MqaQuantity = index.model().data(index, SuncalDelegate.ROLE_QUANTITY)

        menu = QtWidgets.QMenu()
        newinterval = menu.addAction('Set new interval...')
        neweopr = menu.addAction('Set interval for new EOPR...')
        if qty.measurement.interval.reliability_model == 'none':
            newinterval.setEnabled(False)
            neweopr.setEnabled(False)

        neweopr.triggered.connect(lambda x, index=index, qty=qty: self.calc_interval.emit(qty, index))
        newinterval.triggered.connect(lambda x, index=index, qty=qty: self.calc_eopr.emit(qty, index))
        menu.exec(event.globalPosition().toPoint())

    def setModelData(self, editor, model, index):
        ''' Store data back to quantity '''
        qty: MqaQuantity = index.model().data(index, SuncalDelegate.ROLE_QUANTITY)
        qty.measurement.interval.reliability_model = editor.reliability.currentText().lower().replace(' ', '')
        qty.measurement.interval.years = editor.interval.value()
        qty.measurement.typebs = editor.typeb.typebs
        qty.measurement.calibration.stress_pre = editor.prestress.get_pdf()
        qty.measurement.calibration.stress_post = editor.poststress.get_pdf()
        qty.measurement.calibration.p_discard = editor.p_discard.value()
        qty.measurement.calibration.repair_limit = editor.repair.tolerance.limit() if editor.repair.chkbox.isChecked() else None
        qty.measurement.calibration.policy = editor.renewal.currentText().lower().replace('-', '')
        display = ''
        if qty.measurement.interval.reliability_model != 'none':
            display = f'{qty.measurement.interval.years:.1f} yr'
        elif qty.measurement.calibration.policy != 'never':
            display = qty.measurement.calibration.policy.title()
        self.set_displayrole(index, display)
        self.emit_change(index)


class IndirectWidget(QtWidgets.QDialog):
    ''' Widget for setting up indirect measurement as the M&TE '''
    indirect_done = QtCore.pyqtSignal()
    COL_SYMBOL = 0
    COL_VALUE = 1
    COL_EQUIP = 2

    def __init__(self, qty: MqaQuantity, parent=None):
        super().__init__(parent=parent, flags=QtCore.Qt.WindowType.Popup)
        self.qty = qty
        self.setMinimumSize(600, 500)
        self.equation = widgets.LineEditLabelWidget('Measurement Equation:', 'a + b')
        self.table = QtWidgets.QTableWidget()
        self.btnok = QtWidgets.QPushButton('Ok')
        self._equipdelegate = EquipDelegate(showindirect=False)
        self.table.setItemDelegateForColumn(self.COL_EQUIP, self._equipdelegate)
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.equation)
        layout.addWidget(self.table)
        layout.addWidget(self.btnok)
        self.setLayout(layout)
        self.fill_table()
        self.btnok.clicked.connect(self.update_qty)
        self.equation.editingFinished.connect(self.set_equation)

    def set_equation(self):
        ''' Set the measurement equation '''
        try:
            Model(self.equation.text())
        except ValueError as exc:
            QtWidgets.QMessageBox.warning(self, 'Suncal', exc)
        else:
            self.qty.measurement.equation = self.equation.text()
            self.fill_table()

    def fill_table(self):
        ''' Fill the table with the model variables '''
        if not self.qty.measurement.equation:
            self.qty.measurement.equation = 'a + b'

        self.table.clear()
        self.table.setColumnCount(3)
        self.table.setHorizontalHeaderLabels(['Variable', 'Testpoint', 'Equipment'])
        model = Model(self.qty.measurement.equation)
        self.table.setRowCount(len(model.varnames))
        for row, name in enumerate(model.varnames):
            mte = self.qty.measurement.mteindirect.get(name, MTE())
            value = self.qty.measurement.testpoints.get(name, 0)

            if mte.quantity is not None:
                equip = 'Quantity'
                mode = 'quantity'
            elif mte.equipment is not None:
                equip = str(mte.equipment)
                mode = 'equipment'
            else:
                equip = f'±{mte.accuracy_plusminus}'
                mode = 'tolerance'

            dummy_qty = MqaQuantity()
            dummy_qty.equipmentlist = self.qty.equipmentlist
            self.table.setItem(row, self.COL_SYMBOL, widgets.ReadOnlyTableItem(name))
            self.table.setItem(row, self.COL_VALUE, widgets.EditableTableItem(str(value)))
            self.table.setItem(row, self.COL_EQUIP, item := widgets.EditableTableItem(equip))
            item.setData(EquipDelegate.ROLE_EQUIP_MODE, mode)
            item.setData(EquipDelegate.ROLE_QUANTITY, dummy_qty)

    def update_qty(self):
        ''' Get dictionary of [str, MTE] and put in qty.measurement.mteindirect '''
        mteindirect = {}
        testpoints = {}
        for row in range(self.table.rowCount()):
            name = self.table.item(row, self.COL_SYMBOL).text()
            try:
                value = float(self.table.item(row, self.COL_VALUE).text())
            except (ValueError, TypeError):
                value = 0
            dummy_qty = self.table.item(row, self.COL_EQUIP).data(EquipDelegate.ROLE_QUANTITY)
            mte = dummy_qty.measurement.mte
            if mte.quantity is not None:
                mte.quantity.measurand.name = name
            mteindirect[name] = mte
            testpoints[name] = value
        self.qty.measurement.mteindirect.update(mteindirect)
        self.qty.measurement.testpoints.update(testpoints)
        self.indirect_done.emit()


class UtilityWidget(QtWidgets.QDialog):
    ''' Widget for editing end-item utility/performance '''
    def __init__(self, qty: MqaQuantity, parent=None):
        super().__init__(parent=parent, flags=QtCore.Qt.WindowType.Popup)
        self.qty = qty
        self.degrade = widgets.ToleranceCheck()
        self.failure = widgets.ToleranceCheck()
        self.psr = widgets.PercentLineEdit('100')
        self.degrade.setToolTip('Point at which the utility of the item begins to degrade. If omitted, item degrades at the tolerance.')
        self.failure.setToolTip('Point at which the utility of the item becomes zero. If omitted, item fails at the tolerance.')
        self.psr.setToolTip('Probability of a successful outcome, given the measurand is functional.')
        layout = QtWidgets.QFormLayout()
        layout.addRow('Degrade Limit', self.degrade)
        layout.addRow('Failure Limit', self.failure)
        layout.addRow('Successful Outcome Probability', self.psr)
        self.setLayout(layout)


class UtilityDelegate(PopupDelegate):
    ''' Delegate for utility tree column '''
    def createEditor(self, parent, option, index):
        ''' Create the utility editor '''
        qty = index.model().data(index, UtilityDelegate.ROLE_QUANTITY)
        editor = UtilityWidget(qty, parent=parent)
        editor.setAutoFillBackground(True)
        return editor

    def setEditorData(self, editor, index):
        ''' Fill editor with values from quantity '''
        qty = index.model().data(index, UtilityDelegate.ROLE_QUANTITY)
        editor.degrade.set_limit(qty.measurand.degrade_limit)
        editor.failure.set_limit(qty.measurand.fail_limit)
        editor.psr.setValue(qty.measurand.psr)

    def setModelData(self, editor, model, index):
        ''' Store editor values back to quantity '''
        qty = index.model().data(index, UtilityDelegate.ROLE_QUANTITY)
        qty.measurand.degrade_limit = editor.degrade.limit()
        qty.measurand.fail_limit = editor.failure.limit()
        qty.measurand.psr = editor.psr.value()
        if qty.measurand.fail_limit:
            display = str(qty.measurand.fail_limit)
        elif qty.measurand.degrade_limit:
            display = str(qty.measurand.degrade_limit)
        else:
            display = ''
        self.set_displayrole(index, display)
        self.emit_change(index)


class TabCalibration(QtWidgets.QDialog):
    ''' Calibration Process and Interval '''
    def __init__(self, parent=None):
        super().__init__(parent=parent, flags=QtCore.Qt.WindowType.Popup)
        self.qty: MqaQuantity = None
        self.loading: bool = False
        self.prestress = CheckPdf()
        self.poststress = CheckPdf()
        self.p_discard = widgets.FloatLineEdit('0', low=0, high=1)
        self.repair = ToleranceCheck()
        self.repair.setChecked(False)
        self.renewal = QtWidgets.QComboBox()
        self.renewal.addItems(['Never', 'Always', 'As-Needed'])
        self.repair.setEnabled(False)
        self.typeb = TypeBWidget()
        self.p_discard.setEnabled(False)
        self.reliability = QtWidgets.QComboBox()
        self.reliability.addItems(['None', 'Random Walk', 'Exponential'])
        self.interval = widgets.FloatLineEdit('1', low=0)
        self.interval.setEnabled(False)
        self.prestress.setToolTip('PDF of stresses occuring after item taken out of service but before calibration.')
        self.poststress.setToolTip('PDF of stresses occuring after calibration but before item is returned to service.')
        self.p_discard.setToolTip('Probability the item will be discarded if its adjustment or repair limit is exceeded.')
        self.repair.setToolTip('Limits at which a repair of the item is performed. Leave blank to repair when the item exceeds its tolerance.')
        self.renewal.setToolTip('Policy for performing adjustments during calibration. As-needed policy will adjust the item when it exceeds its tolerance.')
        self.reliability.setToolTip('Reliability model for evaluating uncertainty growth over the calibration interval.')
        self.interval.setToolTip('The calibration interval in years.')
        self.typeb.btn_add.setToolTip('Add a Type B uncertainty to the measurement')

        layout = QtWidgets.QFormLayout()
        layout.addRow('<br><font size=4><b>Calibration/Test Process</b></font>', QtWidgets.QWidget())
        layout.addRow('Renewal Policy', self.renewal)
        layout.addRow('Repair Limit', self.repair)
        layout.addRow('Probability of Discarding when OOT', self.p_discard)
        layout.addRow('Pre-measurement stress', self.prestress)
        layout.addRow('Post-measurement stress', self.poststress)
        layout.addRow('Additional process uncertainties', self.typeb)
        layout.addRow('<br><font size=4><b>Observed Interval</b></font>', QtWidgets.QWidget())
        layout.addRow('Reliability Model', self.reliability)
        layout.addRow('Observed Calibration Interval (years)', self.interval)
        self.setLayout(layout)
        self.reliability.currentIndexChanged.connect(self.change_model)
        self.renewal.currentIndexChanged.connect(self.change_renewal)

    def change_renewal(self):
        ''' Renewal policy changed '''
        if self.renewal.currentText() == 'Never':
            self.p_discard.setEnabled(False)
            self.repair.setEnabled(False)
        else:
            self.p_discard.setEnabled(True)
            self.repair.setEnabled(True)

    def change_model(self):
        ''' Reliability model changed '''
        if self.reliability.currentText() != 'None':
            self.interval.setEnabled(True)
        else:
            self.interval.setEnabled(False)

    def set_quantity(self, qty: MqaQuantity):
        ''' Set the Quantity to display '''
        self.qty = qty
        self.loading = True
        self.p_discard.setValue(self.qty.measurement.calibration.p_discard)
        self.interval.setValue(self.qty.measurement.interval.years)
        self.typeb.set_typebs(self.qty.measurement.typebs)
        if self.qty.measurement.calibration.stress_pre:
            self.prestress.chkbox.setChecked(True)
            self.prestress.pdf.set_distribution(self.qty.measurement.calibration.stress_pre.to_dist())
        else:
            self.prestress.chkbox.setChecked(False)
        if self.qty.measurement.calibration.stress_post:
            self.poststress.chkbox.setChecked(True)
            self.poststress.pdf.set_distribution(self.qty.measurement.calibration.stress_post.to_dist())
        else:
            self.poststress.chkbox.setChecked(False)

        self.reliability.setCurrentIndex(
            {'none': 0, 'randomwalk': 1, 'exponential': 2}.get(self.qty.measurement.interval.reliability_model, 0))

        if self.qty.measurement.calibration.repair_limit:
            self.repair.tolerance.set_limit(self.qty.measurement.calibration.repair_limit)
            self.repair.chkbox.setChecked(True)
        else:
            self.repair.chkbox.setChecked(False)
        self.renewal.setCurrentIndex(
            {'never': 0, 'always': 1, 'asneeded': 2}.get(self.qty.measurement.calibration.policy, 0))

        self.change_renewal()
        self.change_model()
        self.loading = False


class TabCosts(QtWidgets.QDialog):
    ''' Section for cost entry '''
    def __init__(self, parent=None):
        super().__init__(parent=parent, flags=QtCore.Qt.WindowType.Popup)
        self.qty: MqaQuantity = None
        self.loading: bool = False
        self.nuut = QtWidgets.QSpinBox()
        self.nuut.setMinimum(0)
        self.nuut.setMaximum(10000000)
        self.suut = widgets.PercentLineEdit('100')
        self.uut = widgets.FloatLineEdit('0', low=0)
        self.cal = widgets.FloatLineEdit('0', low=0)
        self.adjust = widgets.FloatLineEdit('0', low=0)
        self.repair = widgets.FloatLineEdit('0', low=0)
        self.spare_startup = widgets.FloatLineEdit('0', low=0)
        self.down_cal = widgets.FloatLineEdit('0', low=0)
        self.down_adj = widgets.FloatLineEdit('0', low=0)
        self.down_rep = widgets.FloatLineEdit('0', low=0)
        # End Item Parameters
        self.cf = widgets.FloatLineEdit('0', low=0)
        self.pe = widgets.PercentLineEdit('100')
        self.cal.setToolTip('The total cost of performing one calibration/measurement on this item.')
        self.adjust.setToolTip('The total cost of adjusting this item during calibration.')
        self.repair.setToolTip('The total cost to repair the item.')
        self.nuut.setToolTip('The number of identical items in the inventory.')
        self.spare_startup.setToolTip('Cost to startup a spare item')
        self.suut.setToolTip('Level of readiness of spares needed to cover downtime of the item')
        self.uut.setToolTip('Total cost of a new identical item')
        self.down_cal.setToolTip('Time in days required to perform a calibration. Include shipping or storage time.')
        self.down_adj.setToolTip('Time in days required to perform an adjustment of the item.')
        self.down_rep.setToolTip('Time in days required to perform a repair of the item. Include shipping or storage time.')
        self.cf.setToolTip('Cost of an unsuccessful outcome of the end item')
        self.pe.setToolTip('Probability of an unsuccessful outcome given the item fails')

        layout = QtWidgets.QFormLayout()
        layout.addRow('<br><font size=4><b>Calibration Costs</b></font>', QtWidgets.QWidget())
        layout.addRow('Calibration Cost', self.cal)
        layout.addRow('Adjustment Cost', self.adjust)
        layout.addRow('Repair Cost', self.repair)
        layout.addRow('Number of UUTs in Inventory', self.nuut)
        layout.addRow('<b>Spares</b>', QtWidgets.QWidget())
        layout.addRow('Spares Coverage', self.suut)
        layout.addRow('Cost of a UUT', self.uut)
        layout.addRow('Spare Startup Cost', self.spare_startup)
        layout.addRow('<b>Downtimes (days)</b>', QtWidgets.QWidget())
        layout.addRow('Calibration', self.down_cal)
        layout.addRow('Adjustment', self.down_adj)
        layout.addRow('Repair', self.down_rep)
        layout.addRow('<b>End Item Performance</b>', QtWidgets.QWidget())
        layout.addRow('Cost of unsuccessful outcome', self.cf)
        layout.addRow('Probability of unsuccessful outcome given failure', self.pe)
        self.setLayout(layout)

    def set_quantity(self, qty: MqaQuantity):
        ''' Set the Quantity to display '''
        self.qty = qty
        self.nuut.setValue(qty.costs.annual.nuut)
        self.suut.setValue(qty.costs.annual.suut)
        self.uut.setValue(qty.costs.annual.uut)
        self.cal.setValue(qty.costs.annual.cal)
        self.adjust.setValue(qty.costs.annual.adjust)
        self.repair.setValue(qty.costs.annual.repair)
        self.spare_startup.setValue(qty.costs.annual.spare_startup)
        self.down_cal.setValue(qty.costs.annual.downtime.cal)
        self.down_adj.setValue(qty.costs.annual.downtime.adj)
        self.down_rep.setValue(qty.costs.annual.downtime.rep)
        self.cf.setValue(qty.costs.item.cfa)
        self.pe.setValue(qty.costs.annual.pe)
        self.loading = False
        self.layout().setRowVisible(13, qty.enditem)
        self.layout().setRowVisible(14, qty.enditem)
        self.layout().setRowVisible(15, qty.enditem)


class TreeItem(QtWidgets.QTreeWidgetItem):
    ''' An editable tree item '''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setFlags(self.flags() | QtCore.Qt.ItemFlag.ItemIsEditable)


class MqaTree(QtWidgets.QTreeWidget):
    ''' Tree Widget with right-click to remove '''
    COL_NAME = 0
    COL_VALUE = 1
    COL_UNITS = 2
    COL_TOL = 3
    COL_UTIL = 4
    COL_EOPR = 5
    COL_EQUIP = 6
    COL_GB = 7
    COL_CAL = 8
    COL_COST = 9
    COL_TAR = 10
    COL_TUR = 11
    COL_PFA = 12
    COL_DESC = 13
    NCOLS = 14

    remove_qty = QtCore.pyqtSignal(object)
    update_report = QtCore.pyqtSignal(object)

    def __init__(self, component: ProjectMqa):
        super().__init__()
        self.component = component
        self.loading = False
        self._toldelegate = ToleranceDelegate()
        self._eoprdelegate = EoprDelegate()
        self._utildelegate = UtilityDelegate()
        self._equipdelegate = EquipDelegate()
        self._gbdelgate = GuardbandDelegate()
        self._costdelegate = CostDelegate()
        self._caldelegate = CalibrationDelegate()
        self._noeditdelegate = NoEditDelegate()
        self.setColumnCount(self.NCOLS)
        self.setItemDelegateForColumn(self.COL_TOL, self._toldelegate)
        self.setItemDelegateForColumn(self.COL_UTIL, self._utildelegate)
        self.setItemDelegateForColumn(self.COL_EOPR, self._eoprdelegate)
        self.setItemDelegateForColumn(self.COL_GB, self._gbdelgate)
        self.setItemDelegateForColumn(self.COL_EQUIP, self._equipdelegate)
        self.setItemDelegateForColumn(self.COL_COST, self._costdelegate)
        self.setItemDelegateForColumn(self.COL_CAL, self._caldelegate)
        self.setItemDelegateForColumn(self.COL_TAR, self._noeditdelegate)
        self.setItemDelegateForColumn(self.COL_TUR, self._noeditdelegate)
        self.setItemDelegateForColumn(self.COL_PFA, self._noeditdelegate)
        self._caldelegate.calc_eopr.connect(self.calc_eopr)
        self._caldelegate.calc_interval.connect(self.calc_interval)
        self._equipdelegate.uncert_source_changed.connect(self.parent_source_changed)

        self.fill_tree()
        self.itemChanged.connect(self.edit_data)
        self.currentItemChanged.connect(self.change_selected_qty)

    def change_selected_qty(self):
        ''' Selected row was changed '''
        qty = self.currentItem().data(self.COL_NAME, SuncalDelegate.ROLE_QUANTITY)
        self.update_report.emit(qty)

    def fill_tree(self):
        ''' Fill the tree with quantities '''
        self.loading = True
        self.clear()
        self.setColumnCount(self.NCOLS)
        self.setColumnWidth(self.COL_NAME, 125)
        self.setColumnWidth(self.COL_VALUE, 100)
        self.setColumnWidth(self.COL_UNITS, 80)
        self.setColumnWidth(self.COL_TOL, 130)
        self.setColumnWidth(self.COL_UTIL, 100)
        self.setColumnWidth(self.COL_EOPR, 100)
        self.setColumnWidth(self.COL_EQUIP, 130)
        self.setColumnWidth(self.COL_GB, 130)
        self.setColumnWidth(self.COL_TAR, 80)
        self.setColumnWidth(self.COL_TUR, 80)
        self.setColumnWidth(self.COL_PFA, 80)
        self.setColumnWidth(self.COL_DESC, 150)
        self.setHeaderLabels([
            'Quantity', 'Testpoint', 'Units', 'Tolerance', 'Utility', 'EOPR',
            'Equipment', 'Guardband', 'Measurement', 'Costs',
            'TAR', 'TUR', 'PFA %', 'Description'])
        self.headerItem().setToolTip(self.COL_NAME, 'Name of the item')
        self.headerItem().setToolTip(self.COL_VALUE, 'Expected or nominal value of this item')
        self.headerItem().setToolTip(self.COL_UNITS, 'Measurement units. Leave blank for dimensionless quantities.')
        self.headerItem().setToolTip(self.COL_TOL, 'Tolerance limit of the item')
        self.headerItem().setToolTip(self.COL_UTIL, 'End-item performance limits')
        self.headerItem().setToolTip(self.COL_EOPR, 'End-of-period reliability - percentage of similar devices that are found in tolerance at the time of calibration.')
        self.headerItem().setToolTip(self.COL_EQUIP, 'Equipment specification used to measure this item')
        self.headerItem().setToolTip(self.COL_GB, 'Enter acceptance limit or select guardbanding policy from dropdown.')
        self.headerItem().setToolTip(self.COL_CAL, 'Calibration/Measurement and Interval parameters')
        self.headerItem().setToolTip(self.COL_COST, 'Cost model')
        self.headerItem().setToolTip(self.COL_TAR, 'Test Accuracy Ratio (tolerance dividied by equipment accuracy)')
        self.headerItem().setToolTip(self.COL_TUR, 'Test Uncertainty Ratio (tolerance divided by total uncertainty)')
        self.headerItem().setToolTip(self.COL_PFA, 'Global (average) probability of false accept')
        self.headerItem().setToolTip(self.COL_DESC, 'Description of the item')

        for qty in self.component.model.quantities:
            self.add_row(qty)
        self.loading = False

    def add_row(self, qty: MqaQuantity, parent: TreeItem = None):
        ''' Add a row to the tree '''
        assert self.loading
        result = qty.calculate()
        tolerance = qty.measurand.tolerance
        uncert = qty.measurement.uncertainty(qty.measurand.testpoint).expanded()
        mode = qty.mqa_mode

        if qty.measurement.mte.quantity is not None:
            equipmode = 'quantity'
        elif qty.measurement.mte.equipment is not None:
            equipmode = 'equipment'
        else:
            equipmode = 'tolerance'

        item = TreeItem([
            qty.measurand.name,
            str(qty.measurand.testpoint),
            qty.measurand.units,
            str(tolerance),  # Tolerance
            str(qty.measurand.fail_limit) if qty.measurand.fail_limit else '',
            f'{qty.measurand.eopr_pct*100} %',
            f'± {uncert:.2g}',  # Equip
            str(result.guardband),  # GB
            f'{qty.measurement.interval.years:.1f} yr' if mode > MqaQuantity.Mode.BASIC else 'None',  # Cal
            f'{result.cost_annual.total:.0f}' if mode == MqaQuantity.Mode.COSTS else '',
            f'{result.capability.tar:.2f}' if tolerance else '',
            f'{result.capability.tur:.2f}' if tolerance else '',
            f'{result.risk.pfa_true*100:.2f}' if tolerance else '',
        ])
        item.setData(self.COL_NAME, SuncalDelegate.ROLE_QUANTITY, qty)
        item.setData(self.COL_COST, SuncalDelegate.ROLE_QUANTITY, qty)
        item.setData(self.COL_EQUIP, SuncalDelegate.ROLE_QUANTITY, qty)
        item.setData(self.COL_EQUIP, SuncalDelegate.ROLE_EQUIP_MODE, equipmode)
        item.setData(self.COL_GB, SuncalDelegate.ROLE_QUANTITY, qty)
        item.setData(self.COL_GB, SuncalDelegate.ROLE_COMPONENT, self.component)
        item.setData(self.COL_UTIL, SuncalDelegate.ROLE_QUANTITY, qty)
        item.setData(self.COL_CAL, SuncalDelegate.ROLE_QUANTITY, qty)
        item.setData(self.COL_TOL, ToleranceDelegate.ROLE_TOLERANCE, tolerance)
        item.setData(self.COL_EOPR, EoprDelegate.ROLE_TRUEEOPR, qty.measurand.eopr_true)
        if parent:
            parent.addChild(item)
        else:
            self.addTopLevelItem(item)

        if (parentqty := qty.measurement.mte.quantity):
            self.add_row(parentqty, item)
            item.setExpanded(True)

    def edit_data(self, item: TreeItem, column: int):
        ''' An item was edited '''
        if not self.loading:
            qty: MqaQuantity = item.data(0, SuncalDelegate.ROLE_QUANTITY)
            value = item.text(column)
            if column == self.COL_NAME:
                qty.measurand.name = value
            elif column == self.COL_UNITS:
                qty.measurand.units = value
            elif column == self.COL_VALUE:
                try:
                    value = float(value)
                except (TypeError, ValueError):
                    pass
                else:
                    qty.measurand.testpoint = value
                self.refresh_row(item)
            elif column == self.COL_EOPR:
                rtrue = item.data(self.COL_EOPR, EoprDelegate.ROLE_TRUEEOPR)
                qty.measurand.eopr_true = rtrue
                try:
                    value = float(value.strip(' %')) / 100
                except (TypeError, ValueError):
                    value = .9167
                else:
                    qty.measurand.eopr_pct = value
                self.refresh_row(item)

            elif column == self.COL_TOL:
                tol = item.data(self.COL_TOL, ToleranceDelegate.ROLE_TOLERANCE)
                qty.measurand.tolerance = tol
                self.refresh_row(item)

            elif column in [self.COL_GB, self.COL_COST, self.COL_UTIL, self.COL_CAL, self.COL_EQUIP]:
                self.refresh_row(item)

    def refresh_row(self, item: TreeItem):
        ''' Refresh the tree row with new calculated values '''
        qty: MqaQuantity = item.data(0, SuncalDelegate.ROLE_QUANTITY)
        qty.calculate(refresh=True)
        with BlockedSignals(self):
            if qty.guardband.method == 'none':
                item.setText(self.COL_GB, '')
            else:
                item.setText(self.COL_GB, str(qty.result.guardband))

            item.setText(self.COL_EQUIP, f'± {qty.measurement.mte_uncertainty(qty.measurand.testpoint)[1].expanded():.2g}')
            item.setText(self.COL_TAR, f'{qty.result.capability.tar:.2f}')
            item.setText(self.COL_TUR, f'{qty.result.capability.tur:.2f}')
            item.setText(self.COL_PFA, f'{qty.result.risk.pfa_true*100:.3f}')
            item.setText(self.COL_COST, f'{qty.result.cost_annual.total:.0f}' if qty.mqa_mode == MqaQuantity.Mode.COSTS else '')
        self.update_report.emit(qty)

    def contextMenuEvent(self, event):
        ''' Show context menu '''
        item = self.itemAt(event.pos())
        if item:
            menu = QtWidgets.QMenu(self)
            actRem = QtGui.QAction('Remove Quantity...', self)
            menu.addAction(actRem)
            actRem.triggered.connect(lambda x, item=item: self._removerow(item))
            menu.popup(event.globalPos())

    def _removerow(self, item):
        ''' Remove row from table '''
        qty = item.data(0, SuncalDelegate.ROLE_QUANTITY)
        self.remove_qty.emit(qty)

    def parent_source_changed(self, qty: MqaQuantity, index):
        ''' MTE parent quantity was changed - add or remove tree children '''
        self.loading = True
        item = self.itemFromIndex(index)
        item.takeChildren()
        parentqty = qty.measurement.mte.quantity
        indirect = qty.measurement.mteindirect

        if parentqty is not None:
            self.add_row(parentqty, item)

        for mte in indirect.values():
            if mte.quantity is not None:
                self.add_row(mte.quantity, item)

        item.setExpanded(True)
        self.loading = False

    def calc_interval(self, qty: MqaQuantity, index):
        ''' Calculate new interval for the input EOPR '''
        item = self.itemFromIndex(index)

        current_eopr = qty.measurand.eopr_pct * 100
        current_interval = qty.measurement.interval.years
        eopr, ok = QtWidgets.QInputDialog.getDouble(
            self, 'Desired End-of-period Reliability',
            f'Current: Interval = {current_interval:.2f} (years) at EOPR = {current_eopr:.4f}%.\n'
            'Enter desired EOPR% to calculate new interval.',
            current_eopr, .1, 99.999999, 4)

        if ok:
            qty.interval_for_eopr(eopr / 100)
            qty.calculate(refresh=True)
            item.setText(self.COL_EOPR, f'{eopr:.2f} %')

    def calc_eopr(self, qty: MqaQuantity, index):
        ''' Set the a new interval, recalculating EOPR '''
        item = self.itemFromIndex(index)
        assert qty.result is not None
        current_eopr = qty.measurand.eopr_pct * 100
        current_interval = qty.measurement.interval.years
        interval, ok = QtWidgets.QInputDialog.getDouble(
            self, 'New Interval',
            f'Current: Interval = {current_interval:.2f} (years) at EOPR = {current_eopr:.4f}%.\n'
            'Enter new interval in years.',
            current_interval, .1, 10000, 3)

        if ok:
            new_eopr = qty.eopr_for_interval(interval)
            qty.calculate(refresh=True)
            item.setText(self.COL_EOPR, f'{new_eopr*100:.2f} %')


class MqaInput(QtWidgets.QWidget):
    ''' MQA Input Page '''
    def __init__(self, component: ProjectMqa, parent=None):
        super().__init__(parent)
        self.component = component
        self.tree = MqaTree(component)
        self.tree.setStyleSheet("QTreeView::item { padding: 6px }")
        font = self.tree.font()
        font.setPointSize(12)
        self.tree.setFont(font)
        self.tree.header().setFont(font)
        self.qtyreport = QtyReport()
        self.btn_addqty = widgets.SmallToolButton('tolerance')
        self.btn_showrpt = widgets.SmallToolButton('report')
        self.btn_editequip = widgets.SmallToolButton('meter')
        self.btn_gbsettings = widgets.SmallToolButton('guardband')
        self.btn_editequip.setToolTip('Equipment List...')
        self.btn_showrpt.setToolTip('Generate Report...')
        self.btn_addqty.setToolTip('Add Measured Quantity...')
        self.btn_gbsettings.setToolTip('Guardband Policies...')
        self.splitter = QtWidgets.QSplitter()
        self.splitter.addWidget(self.tree)
        self.splitter.addWidget(self.qtyreport)
        self.splitter.setSizes([1, 0])

        layout = QtWidgets.QVBoxLayout()
        hlayout = QtWidgets.QHBoxLayout()
        hlayout.addWidget(self.btn_addqty)
        hlayout.addStretch()
        layout.addLayout(hlayout)
        layout.addWidget(self.splitter)

        blayout = QtWidgets.QHBoxLayout()
        blayout.addWidget(self.btn_editequip)
        blayout.addWidget(self.btn_gbsettings)
        blayout.addStretch()
        blayout.addWidget(self.btn_showrpt)
        layout.addLayout(blayout)
        self.setLayout(layout)

        self.tree.update_report.connect(self.qtyreport.refresh_report)


class QtyReport(QtWidgets.QWidget):
    ''' Report for one quantity '''
    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.qty: MqaQuantity = None
        self.report = widgets.MarkdownTextEdit()
        self.combo = QtWidgets.QComboBox()
        self.combo.addItems([
            'Reliability', 'Uncertainty', 'Reliability Plots',
            'Pre/Post Test PDFs', 'Reliability Decay', 'Utility Curve',
            'Costs'
        ])
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.combo)
        layout.addWidget(self.report)
        self.setLayout(layout)
        self.combo.currentIndexChanged.connect(lambda x: self.refresh_report())

    def refresh_report(self, qty=None):
        ''' Refresh the quantity report '''
        if qty is not None:
            self.qty = qty

        if self.qty is None:
            return

        result = self.qty.calculate()  # No refresh
        if self.combo.currentText() == 'Costs':
            rpt = result.report.report_cost(enditem=self.qty.enditem)
            rpt.hdr('Total', level=4)
            rpt.append(result.report.report_cost_total(enditem=self.qty.enditem))
        elif self.combo.currentText() == 'Uncertainty':
            rpt = result.uncertainty.report.report()
        elif self.combo.currentText() == 'Reliability Plots':
            rpt = result.report.report_reliability_plots()
        elif self.combo.currentText() == 'Pre/Post Test PDFs':
            rpt = result.report.report_prepost_pdf()
        elif self.combo.currentText() == 'Reliability Decay':
            rpt = result.report.report_decay()
        elif self.combo.currentText() == 'Utility Curve':
            rpt = result.report.report_utility()
        else:
            rpt = result.report.report()

        self.report.setReport(rpt)


class EquipmentEdit(QtWidgets.QWidget):
    ''' Widget for editing equipment list '''
    def __init__(self, component: ProjectMqa, parent=None):
        super().__init__(parent)
        self.component = component
        self.equip = widgets.EquipmentEdit(component.model.equipment)
        self.btn_back = widgets.LeftButton()
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.equip)
        layout.addStretch()
        layout.addWidget(self.btn_back)
        self.setLayout(layout)


class GuardbandRuleEditor(QtWidgets.QWidget):
    ''' Widget for editing guardband rules '''
    RULE_ROLE = QtCore.Qt.ItemDataRole.UserRole

    def __init__(self, component: ProjectMqa, parent=None):
        super().__init__(parent)
        self.component = component
        self.loading: bool = False
        self.table = QtWidgets.QTableWidget()
        font = self.table.font()
        font.setPointSize(12)
        self.table.setFont(font)
        self.buttons = widgets.PlusMinusButton()
        self.btn_back = widgets.LeftButton()
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.buttons)
        layout.addWidget(self.table)
        layout.addStretch()
        layout.addWidget(self.btn_back)
        self.setLayout(layout)
        self.fill_table()
        self.buttons.plusclicked.connect(self.add_rule)
        self.buttons.minusclicked.connect(self.rem_rule)
        self.table.itemChanged.connect(self.edited)

    def fill_table(self):
        ''' Fill the table with rules stored in the component '''
        self.loading = True
        self.table.clear()
        self.table.setColumnCount(3)
        self.table.setHorizontalHeaderLabels(['Rule Name', 'Method', 'Threshold'])
        self.table.setRowCount(len(self.component.model.gbrules.rules))
        methods = ['RDS', 'Dobbert', 'RP10', 'U95', 'PFA', 'CPFA']
        methods_lower = [m.lower() for m in methods]  # as stored in rule
        for row, rule in enumerate(self.component.model.gbrules.rules):
            item = QtWidgets.QTableWidgetItem(rule.name)
            item.setData(self.RULE_ROLE, rule)
            self.table.setItem(row, 0, item)
            self.table.setItem(row, 2, QtWidgets.QTableWidgetItem(str(rule.threshold)))
            widget = QtWidgets.QComboBox()
            widget.addItems(methods)
            widget.setCurrentIndex(methods_lower.index(rule.method))
            widget.currentIndexChanged.connect(lambda x, item=item: self.edited(item))
            self.table.setCellWidget(row, 1, widget)
        self.loading = False

    def add_rule(self):
        ''' Add a guardband rule '''
        self.table.setRowCount(self.table.rowCount() + 1)
        self.component.model.gbrules.rules.append(MqaGuardbandRule())
        self.fill_table()

    def rem_rule(self):
        ''' Remove the selected rule '''
        row = self.table.currentRow()
        if row >= 0:
            self.component.model.gbrules.rules.pop(row)
        self.fill_table()

    def edited(self, item: QtWidgets.QTableWidgetItem):
        ''' A rule was edited '''
        if item and not self.loading:
            row = item.row()
            rule = self.table.item(row, 0).data(self.RULE_ROLE)
            rule.name = self.table.item(row, 0).text()
            rule.method = self.table.cellWidget(row, 1).currentText().lower()
            try:
                rule.threshold = float(self.table.item(row, 2).text())
            except ValueError:
                self.table.item(row, 2).setText('4')


class MqaReport(QtWidgets.QWidget):
    ''' Widget for showing full report of all quantities '''
    def __init__(self, component, parent=None):
        super().__init__(parent)
        self.component = component
        self.report = widgets.MarkdownTextEdit()
        self.rpt_select = QtWidgets.QComboBox()
        self.rpt_select.addItems(['Table', 'Details'])
        self.btn_back = widgets.LeftButton()
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.rpt_select)
        layout.addWidget(self.report)
        layout.addWidget(self.btn_back)
        self.setLayout(layout)
        self.rpt_select.currentIndexChanged.connect(self.update)

    def update(self):
        ''' Update the report '''
        result = self.component.calculate()
        if self.rpt_select.currentText() == 'Table':
            self.report.setReport(result.report.summary())
        else:
            self.report.setReport(result.report.report_details())


class MqaWidget(QtWidgets.QWidget):
    ''' Widget for editing a full Measurement System '''
    STACK_TABLE = 0
    STACK_EQUIPMENT = 1
    STACK_GUARDBAND = 2
    STACK_REPORT = 3

    change_help = QtCore.pyqtSignal()

    def __init__(self, component, parent=None):
        super().__init__(parent)
        assert isinstance(component, ProjectMqa)
        self.component = component
        font = self.font()
        font.setPointSize(12)
        self.setFont(font)
        self.pg_input = MqaInput(component)
        self.equip = EquipmentEdit(component)
        self.gbedit = GuardbandRuleEditor(component)
        self.report = MqaReport(component)
        self.stack = widgets.SlidingStackedWidget()
        self.stack.addWidget(self.pg_input)
        self.stack.addWidget(self.equip)
        self.stack.addWidget(self.gbedit)
        self.stack.addWidget(self.report)

        self.actSaveReport = QtGui.QAction('&Save Report...', self)
        self.actUtility = QtGui.QAction('Show End-item Utility')
        self.actCal = QtGui.QAction('Show Measurement/Interval Options')
        self.actCost = QtGui.QAction('Show Cost Model')
        self.actTUR = QtGui.QAction('Show TUR')
        self.actTAR = QtGui.QAction('Show TAR')
        self.actPFA = QtGui.QAction('Show PFA')
        self.actDesc = QtGui.QAction('Show Description')
        self.actUtility.setCheckable(True)
        self.actCal.setCheckable(True)
        self.actCost.setCheckable(True)
        self.actTUR.setCheckable(True)
        self.actTAR.setCheckable(True)
        self.actPFA.setCheckable(True)
        self.actDesc.setCheckable(True)
        self.menu = QtWidgets.QMenu('M&QA')
        self.menu.addAction(self.actUtility)
        self.menu.addAction(self.actCal)
        self.menu.addAction(self.actCost)
        self.menu.addAction(self.actTUR)
        self.menu.addAction(self.actTAR)
        self.menu.addAction(self.actPFA)
        self.menu.addAction(self.actDesc)
        self.menu.addSeparator()
        self.menu.addAction(self.actSaveReport)
        self.actSaveReport.triggered.connect(self.save_report)
        self.actUtility.triggered.connect(self.enable_columns)
        self.actCal.triggered.connect(self.enable_columns)
        self.actCost.triggered.connect(self.enable_columns)
        self.actTUR.triggered.connect(self.enable_columns)
        self.actTAR.triggered.connect(self.enable_columns)
        self.actPFA.triggered.connect(self.enable_columns)
        self.actDesc.triggered.connect(self.enable_columns)

        self.report.btn_back.clicked.connect(self.goback)
        self.equip.btn_back.clicked.connect(self.goback)
        self.gbedit.btn_back.clicked.connect(self.goback)
        self.pg_input.btn_showrpt.clicked.connect(self.show_report)
        self.pg_input.btn_addqty.clicked.connect(self.add_quantity)
        self.pg_input.btn_editequip.clicked.connect(self.show_equip)
        self.pg_input.btn_gbsettings.clicked.connect(self.show_guardband)
        self.pg_input.tree.remove_qty.connect(self.rem_quantity)

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.stack)
        self.setLayout(layout)
        self.init_columns()

    def show_equip(self):
        ''' Show the Equipment List page. '''
        self.stack.slideInLeft(self.STACK_EQUIPMENT)
        self.change_help.emit()

    def show_guardband(self):
        ''' Show the Guardband Policy page. '''
        self.stack.slideInLeft(self.STACK_GUARDBAND)
        self.change_help.emit()

    def get_menu(self):
        ''' Get the menu for this widget '''
        return self.menu

    def update_proj_config(self):
        ''' Save page setup back to project item configuration '''
        # GUI updates model in real time - nothing to do

    def goback(self):
        ''' Slide back to table and refresh it '''
        self.stack.slideInRight(self.STACK_TABLE)
        self.change_help.emit()

    def add_quantity(self):
        ''' Add a Quantity to the system '''
        self.pg_input.tree.loading = True
        qty = self.component.model.add()
        self.pg_input.tree.add_row(qty)
        self.pg_input.tree.loading = False
        self.pg_input.tree.update_report.emit(qty)

    def rem_quantity(self, qty):
        ''' Remove quantity from the model '''
        self.component.model.quantities.remove(qty)
        self.pg_input.tree.fill_tree()

    def show_report(self):
        ''' Calculate and display the report '''
        result = self.component.calculate()
        self.report.report.setReport(result.report.summary())
        self.stack.slideInLeft(self.STACK_REPORT)
        self.change_help.emit()

    def save_report(self):
        ''' Save full report, asking user for settings/filename '''
        with gui_styles.LightPlotstyle():
            result = self.component.calculate()
            widgets.savereport(result.report.summary())

    def init_columns(self):
        ''' Initialize column visibility '''
        mode = self.component.model.mqa_mode()
        with BlockedSignals(self):
            if mode > MqaQuantity.Mode.BASIC or gui_settings.mqa_cal:
                self.actCal.setChecked(True)
            if mode == MqaQuantity.Mode.COSTS or gui_settings.mqa_cost:
                self.actCost.setChecked(True)

            self.actUtility.setChecked(gui_settings.mqa_performance)
            self.actTAR.setChecked(gui_settings.mqa_tar)
            self.actTUR.setChecked(gui_settings.mqa_tur)
            self.actPFA.setChecked(gui_settings.mqa_pfa)
            self.actDesc.setChecked(gui_settings.mqa_desc)
        self.enable_columns()

    def enable_columns(self):
        ''' Toggle column visibility '''
        gui_settings.mqa_performance = self.actUtility.isChecked()
        gui_settings.mqa_cal = self.actCal.isChecked()
        gui_settings.mqa_cost = self.actCost.isChecked()
        gui_settings.mqa_tur = self.actTUR.isChecked()
        gui_settings.mqa_tar = self.actTAR.isChecked()
        gui_settings.mqa_pfa = self.actPFA.isChecked()
        gui_settings.mqa_desc = self.actDesc.isChecked()
        self.pg_input.tree.setColumnHidden(MqaTree.COL_UTIL, not self.actUtility.isChecked())
        self.pg_input.tree.setColumnHidden(MqaTree.COL_CAL, not self.actCal.isChecked())
        self.pg_input.tree.setColumnHidden(MqaTree.COL_COST, not self.actCost.isChecked())
        self.pg_input.tree.setColumnHidden(MqaTree.COL_TUR, not self.actTUR.isChecked())
        self.pg_input.tree.setColumnHidden(MqaTree.COL_TAR, not self.actTAR.isChecked())
        self.pg_input.tree.setColumnHidden(MqaTree.COL_PFA, not self.actPFA.isChecked())
        self.pg_input.tree.setColumnHidden(MqaTree.COL_DESC, not self.actDesc.isChecked())

    def help_report(self):
        ''' Get the help report to display the current widget mode '''
        if self.stack.m_next == self.STACK_TABLE:
            return MqaHelp.main()
        if self.stack.m_next == self.STACK_GUARDBAND:
            return MqaHelp.guardband_rules()
        return MqaHelp.nohelp()


if __name__ == '__main__':
    import sys
    app = QtWidgets.QApplication(sys.argv)
    main = MqaWidget(ProjectMqa())

    gui_common.centerWindow(main, 1200, 900)
    font = main.font()
    font.setPointSize(10)
    main.setFont(font)

    main.show()
    app.exec()
