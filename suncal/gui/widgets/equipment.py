''' Widgets for editing and selecting measuring equipment '''
import yaml
from PyQt6 import QtWidgets, QtCore

from ...mqa.equipment import Equipment, Range, EquipmentList, MqaEquipmentRange
from .buttons import PlusMinusButton
from .combo import FloatLineEdit


class RangeEdit(QtWidgets.QWidget):
    ''' Widget with Range parameters '''
    changed = QtCore.pyqtSignal()

    def __init__(self, rng: Range, parent=None):
        super().__init__(parent=parent)
        self.rng = rng
        self.name = QtWidgets.QLineEdit()
        self.units = QtWidgets.QLineEdit()
        self.min = FloatLineEdit()
        self.max = FloatLineEdit()
        self.pct_reading = FloatLineEdit()
        self.pct_range = FloatLineEdit()
        self.const = FloatLineEdit()
        self.resolution = FloatLineEdit()
        self.formula = QtWidgets.QLineEdit()
        self.formula.setToolTip('Equation as function of measured value x')
        self.resolution.setToolTip('Smallest distinguishable change in indication')
        self.min.setToolTip('Smallest value that can be measured by this equipment range')
        self.max.setToolTip('Largest value that can be measured by this equipment range')
        self.pct_reading.setToolTip('Accuracy as a percent of measured value')
        self.pct_range.setToolTip('Accuracy as a percent of the range maximum')
        self.const.setToolTip('Constant component of accuracy for this equipment range')
        self.name.setToolTip('Descriptive name of the equipment range')
        self.units.setToolTip('Measurement units')

        layout = QtWidgets.QFormLayout()
        layout.addRow('Name', self.name)
        layout.addRow('Units', self.units)
        layout.addRow('Minimum Value', self.min)
        layout.addRow('Maximum Value', self.max)
        layout.addRow(r'% of Reading', self.pct_reading)
        layout.addRow(r'% of Range', self.pct_range)
        layout.addRow('Constant Uncertainty', self.const)
        layout.addRow('Resolution', self.resolution)
        layout.addRow('Uncertainty Formula', self.formula)
        self.setLayout(layout)

        self.name.editingFinished.connect(self.edit_rng)
        self.units.editingFinished.connect(self.edit_rng)
        self.min.editingFinished.connect(self.edit_rng)
        self.max.editingFinished.connect(self.edit_rng)
        self.pct_reading.editingFinished.connect(self.edit_rng)
        self.pct_range.editingFinished.connect(self.edit_rng)
        self.const.editingFinished.connect(self.edit_rng)
        self.resolution.editingFinished.connect(self.edit_rng)
        self.formula.editingFinished.connect(self.edit_rng)

    def set_rng(self, rng: Range):
        ''' Set the Range to display '''
        self.rng = rng
        self.name.setText(rng.name)
        self.units.setText(str(rng.units))
        self.min.setValue(rng.low)
        self.max.setValue(rng.high)
        self.pct_reading.setText(str(rng._accuracy.pct_reading))
        self.pct_range.setText(str(rng._accuracy.pct_range))
        self.const.setText(str(rng._accuracy.const))
        self.resolution.setText(str(rng._accuracy.resolution))
        self.formula.setText(str(rng._accuracy.formula))

    def clear(self):
        ''' Clear the values '''
        self.name.setText('')
        self.units.setText('')
        self.min.setText('')
        self.max.setText('')
        self.pct_reading.setText('')
        self.pct_range.setText('')
        self.const.setText('')
        self.resolution.setText('')
        self.formula.setText('')

    def edit_rng(self):
        ''' Values were edited, save back to model '''
        self.rng.name = self.name.text()
        self.rng.units = self.units.text()
        self.rng.low = self.min.value()
        self.rng.high = self.max.value()
        self.rng.set_accuracy(self.pct_reading.text(),
                              self.pct_range.text(),
                              self.const.text(),
                              self.resolution.text(),
                              self.formula.text(),
                              self.units.text())
        self.changed.emit()


class EquipEdit(QtWidgets.QWidget):
    ''' Widget for editing equipment details '''
    changed = QtCore.pyqtSignal()

    def __init__(self, equip: Equipment, parent=None):
        super().__init__(parent=parent)
        self.equip = equip
        self.make = QtWidgets.QLineEdit()
        self.model = QtWidgets.QLineEdit()
        self.serial = QtWidgets.QLineEdit()
        self.k = FloatLineEdit('')
        self.k.setPlaceholderText('Leave blank for Uniform')
        self.btn_addrng = QtWidgets.QPushButton('Add Range')

        flayout = QtWidgets.QFormLayout()
        flayout.addRow('Make', self.make)
        flayout.addRow('Model', self.model)
        flayout.addRow('Serial', self.serial)
        flayout.addRow('Coverage Factor', self.k)
        layout = QtWidgets.QVBoxLayout()
        layout.addLayout(flayout)
        layout.addWidget(self.btn_addrng)
        layout.addStretch()
        self.setLayout(layout)

        self.make.editingFinished.connect(self.edit_equip)
        self.model.editingFinished.connect(self.edit_equip)
        self.serial.editingFinished.connect(self.edit_equip)
        self.k.editingFinished.connect(self.edit_equip)

    def set_equipment(self, equip: Equipment):
        ''' Set the equipment to display '''
        self.equip = equip
        self.make.setText(equip.make)
        self.model.setText(equip.model)
        self.serial.setText(equip.serial)
        if equip.k is not None:
            self.k.setValue(equip.k)
        else:
            self.k.setText('')

    def edit_equip(self):
        ''' Equipment was changed, save back to model '''
        self.equip.make = self.make.text()
        self.equip.model = self.model.text()
        self.equip.serial = self.serial.text()
        if self.k.text():
            self.equip.k = self.k.value()
        else:
            self.equip.k = None
        self.changed.emit()


class EquipmentEdit(QtWidgets.QWidget):
    ''' Widget for editing the equipment list '''
    ROLE_EQUIP = QtCore.Qt.ItemDataRole.UserRole
    ROLE_RNG = ROLE_EQUIP + 1

    def __init__(self, equipment: EquipmentList = None, parent=None):
        super().__init__(parent=parent)
        if equipment is None:
            equipment = EquipmentList()
        self.equipment = equipment
        font = self.font()
        font.setPointSize(12)
        self.setFont(font)

        self.tree = QtWidgets.QTreeWidget()
        self.tree.setColumnCount(1)
        self.tree.setHeaderHidden(True)
        self.btn_addrem = PlusMinusButton()
        self.rngedit = RangeEdit(None)
        self.equipedit = EquipEdit(None)
        self.btn_import = QtWidgets.QPushButton('Import...')
        self.btn_export = QtWidgets.QPushButton('Export...')
        self.btn_import.setToolTip('Import equipment list from a previously saved file')
        self.btn_export.setToolTip('Export equipment list to a file')
        self.btn_addrem.btnplus.setToolTip('Add a new equipment model')
        self.btn_addrem.btnminus.setToolTip('Remove the selected equipment model')

        self.stack = QtWidgets.QStackedWidget()
        self.stack.addWidget(QtWidgets.QWidget())
        self.stack.addWidget(self.equipedit)
        self.stack.addWidget(self.rngedit)

        layout = QtWidgets.QHBoxLayout()
        llayout = QtWidgets.QVBoxLayout()

        blayout = QtWidgets.QHBoxLayout()
        blayout.addWidget(self.btn_addrem)
        blayout.addStretch()
        blayout.addWidget(self.btn_import)
        blayout.addWidget(self.btn_export)
        llayout.addLayout(blayout)
        llayout.addWidget(self.tree)
        layout.addLayout(llayout)
        layout.addWidget(self.stack)
        layout.addStretch()
        self.setLayout(layout)

        self.tree.itemClicked.connect(self.change_selected_item)
        self.btn_addrem.plusclicked.connect(self.add_equip)
        self.btn_addrem.minusclicked.connect(self.remove_item)
        self.equipedit.changed.connect(self.equip_edited)
        self.equipedit.btn_addrng.clicked.connect(self.add_rng)
        self.rngedit.changed.connect(self.rng_edited)
        self.btn_import.clicked.connect(self.import_list)
        self.btn_export.clicked.connect(self.export_list)
        self.fill_tree()

    def set_list(self, equipmentlist: EquipmentList):
        ''' Set the list of equipment to display '''
        if equipmentlist is None:
            equipmentlist = EquipmentList()
        self.equipment = equipmentlist
        self.fill_tree()

    def fill_tree(self):
        ''' Fill the tree '''
        self.tree.clear()
        for equip in self.equipment.equipment:
            item = QtWidgets.QTreeWidgetItem([str(equip)])
            item.setData(0, self.ROLE_EQUIP, equip)
            self.tree.addTopLevelItem(item)
            for rng in equip.ranges:
                rngitem = QtWidgets.QTreeWidgetItem([rng.name])
                rngitem.setData(0, self.ROLE_EQUIP, equip)
                rngitem.setData(0, self.ROLE_RNG, rng)
                item.addChild(rngitem)

    def change_selected_item(self, item: QtWidgets.QTreeWidgetItem):
        ''' The selected item was changed. Show appropriate widget '''
        rng = item.data(0, self.ROLE_RNG)
        equip = item.data(0, self.ROLE_EQUIP)

        if rng is not None:
            self.rngedit.set_rng(rng)
            self.stack.setCurrentWidget(self.rngedit)
        elif equip is not None:
            self.equipedit.set_equipment(equip)
            self.stack.setCurrentWidget(self.equipedit)

    def add_equip(self):
        ''' Add a new equipment '''
        equip = Equipment()
        self.equipment.equipment.append(equip)
        item = QtWidgets.QTreeWidgetItem(['New Equipment'])
        item.setData(0, self.ROLE_EQUIP, equip)
        self.tree.addTopLevelItem(item)
        self.change_selected_item(item)
        self.tree.setCurrentItem(item)

    def add_rng(self):
        ''' Add a range to the equipment'''
        equip = self.equipedit.equip
        rng = Range()
        rng.name = 'New Range'
        equip.add_range(rng)
        item = self.find_item(equip)
        rngitem = QtWidgets.QTreeWidgetItem([rng.name])
        rngitem.setData(0, self.ROLE_EQUIP, equip)
        rngitem.setData(0, self.ROLE_RNG, rng)
        item.addChild(rngitem)
        item.setExpanded(True)
        self.rngedit.set_rng(rng)
        self.stack.setCurrentWidget(self.rngedit)
        self.tree.setCurrentItem(item)

    def remove_item(self):
        ''' The selected item was changed. Show appropriate widget '''
        item = self.tree.currentItem()
        if item:
            rng: Range = item.data(0, self.ROLE_RNG)
            equip: Equipment = item.data(0, self.ROLE_EQUIP)
            if rng is not None:
                equip.ranges.remove(rng)
                item.parent().removeChild(item)
            elif equip is not None:
                self.equipment.equipment.remove(equip)
                self.tree.takeTopLevelItem(self.tree.indexOfTopLevelItem(item))

    def find_item(self, equip: Equipment) -> QtWidgets.QTreeWidgetItem:
        ''' Find tree item associated with the equipment '''
        items = [self.tree.topLevelItem(i) for i in range(self.tree.topLevelItemCount())]
        equips = [item.data(0, self.ROLE_EQUIP) for item in items]
        try:
            idx = equips.index(equip)
        except IndexError:
            return None
        return items[idx]

    def find_rng_item(self, rng: Range) -> QtWidgets.QTreeWidgetItem:
        ''' Find tree item associated with the range '''
        rngitems = []
        for i in range(self.tree.topLevelItemCount()):
            item = self.tree.topLevelItem(i)
            rngitems.extend([item.child(k) for k in range(item.childCount())])

        ranges = [item.data(0, self.ROLE_RNG) for item in rngitems]
        try:
            idx = ranges.index(rng)
        except IndexError:
            return None
        return rngitems[idx]

    def equip_edited(self):
        ''' The equipment was edited. Update tree label. '''
        equip = self.equipedit.equip
        item = self.find_item(equip)
        if item:
            item.setText(0, str(equip))

    def rng_edited(self):
        ''' The range was edited. Update tree label. '''
        rng = self.rngedit.rng
        item = self.find_rng_item(rng)
        if item:
            item.setText(0, str(rng.name))

    def export_list(self):
        ''' Export the equipment list '''
        fname, _ = QtWidgets.QFileDialog.getSaveFileName(caption='Select file to save')
        if fname:
            config = self.equipment.config()
            out = yaml.safe_dump(config, default_flow_style=False)

            with open(fname, 'w', encoding='utf-8') as f:
                f.write(out)

    def import_list(self):
        ''' Import the equipment list from file '''
        fname, _ = QtWidgets.QFileDialog.getOpenFileName(caption='Equipment file to load')
        if fname:
            with open(fname, 'r', encoding='utf-8') as f:
                dat = f.read()

            try:
                config = yaml.safe_load(dat)
            except yaml.YAMLError:
                QtWidgets.QMessageBox.warning(self, 'Suncal', 'Bad Equipment File Type')
                return

            try:
                equiplist = EquipmentList.from_config(config)
            except (ValueError, TypeError):
                QtWidgets.QMessageBox.warning(self, 'Suncal', 'Invalid Equipment List')
                return

            # Don't change the self.equipment object reference
            self.equipment.equipment[:] = equiplist.equipment
            self.fill_tree()


class EquipmentSelect(QtWidgets.QDialog):
    ''' Widget for selecting an equipment/range from the list '''
    ROLE_EQUIP = QtCore.Qt.ItemDataRole.UserRole
    ROLE_RNG = ROLE_EQUIP + 1

    selected = QtCore.pyqtSignal(object)

    def __init__(self, equipment: EquipmentList = None, parent=None):
        super().__init__(parent=parent, flags=QtCore.Qt.WindowType.Popup)
        self.equipment = equipment
        font = self.font()
        font.setPointSize(12)
        self.setFont(font)
        self.tree = QtWidgets.QTreeWidget()
        self.tree.setColumnCount(1)
        self.tree.setHeaderHidden(True)

        self.btn_select = QtWidgets.QPushButton('Select')
        self.btn_clear = QtWidgets.QPushButton('Unselect')
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.tree)
        hlayout = QtWidgets.QHBoxLayout()
        hlayout.addWidget(self.btn_select)
        hlayout.addWidget(self.btn_clear)
        hlayout.addStretch()
        layout.addLayout(hlayout)
        self.setLayout(layout)
        self.fill_tree()
        self.btn_select.clicked.connect(self.equip_selected)
        self.btn_clear.clicked.connect(self.equip_cleared)

    def fill_tree(self):
        ''' Fill the tree '''
        self.tree.clear()
        if not self.equipment.equipment:
            item = QtWidgets.QTreeWidgetItem(['No Equipment Defined.'])
            self.tree.addTopLevelItem(item)
        else:
            for equip in self.equipment.equipment:
                item = QtWidgets.QTreeWidgetItem([str(equip)])
                item.setData(0, self.ROLE_EQUIP, equip)
                self.tree.addTopLevelItem(item)
                for rng in equip.ranges:
                    rngitem = QtWidgets.QTreeWidgetItem([rng.name])
                    rngitem.setData(0, self.ROLE_EQUIP, equip)
                    rngitem.setData(0, self.ROLE_RNG, rng)
                    item.addChild(rngitem)

    def equip_selected(self):
        ''' Equipment was selected using "select" button '''
        item = self.tree.currentItem()
        if item is None:
            return
        equip = item.data(0, self.ROLE_EQUIP)
        rng = item.data(0, self.ROLE_RNG)
        self.selected.emit(MqaEquipmentRange(equip, rng))

    def equip_cleared(self):
        ''' Equipment was cleared using "Unselect" button '''
        self.selected.emit(None)
