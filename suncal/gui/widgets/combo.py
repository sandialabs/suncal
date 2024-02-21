''' Combination widgets, such as a LineEdit with Label, mostly
    for convenience.
'''
from PyQt6 import QtWidgets, QtCore, QtGui

from ..gui_common import InfValidator, BlockedSignals


class QHLine(QtWidgets.QFrame):
    ''' Horizontal divider line '''
    def __init__(self):
        super().__init__()
        self.setFrameShape(QtWidgets.QFrame.Shape.HLine)
        self.setFrameShadow(QtWidgets.QFrame.Shadow.Sunken)


class ComboNoWheel(QtWidgets.QComboBox):
    ''' ComboBox with scroll wheel disabled '''
    def __init__(self):
        super().__init__()
        self.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)

    def wheelEvent(self, event):
        ''' Only pass on the event if we have focus '''
        if self.hasFocus():
            super().wheelEvent(event)


class ComboLabel(QtWidgets.QWidget):
    ''' ComboBox with a label '''
    def __init__(self, label: str, items: list[str] = None):
        super().__init__()
        self._label = QtWidgets.QLabel(label)
        self._combo = QtWidgets.QComboBox()
        if items:
            self.combo.addItems(items)
        layout = QtWidgets.QHBoxLayout()
        layout.addWidget(self._label)
        layout.addWidget(self._combo)
        self.setLayout(layout)

    def __getattr__(self, name):
        ''' Get all other attributes from the combo widget '''
        return getattr(self._combo, name)


class SpinWidget(QtWidgets.QWidget):
    ''' Widget with label and spinbox '''
    valueChanged = QtCore.pyqtSignal()

    def __init__(self, label=''):
        super().__init__()
        self.label = QtWidgets.QLabel(label)
        self.spin = QtWidgets.QSpinBox()
        self.spin.setRange(int(0), int(1E8))
        self.spin.valueChanged.connect(self.valueChanged)
        layout = QtWidgets.QHBoxLayout()
        layout.addWidget(self.label)
        layout.addWidget(self.spin)
        layout.addStretch()
        self.setLayout(layout)

    def value(self):
        return int(self.spin.text())

    def setValue(self, value):
        self.spin.setValue(int(value))


class FloatLineEdit(QtWidgets.QLineEdit):
    ''' Line Edit with float validator '''
    def __init__(self, text='', low=None, high=None):
        super().__init__(text)
        self._validator = QtGui.QDoubleValidator()
        if low is not None:
            self._validator.setBottom(low)
        if high is not None:
            self._validator.setTop(high)
        self.setValidator(self._validator)

    def value(self):
        try:
            val = float(self.text())
        except ValueError:
            val = 0
        return val

    def setValue(self, value):
        self.setText(str(value))


class IntLineEdit(QtWidgets.QLineEdit):
    ''' Line Edit with integer validator '''
    def __init__(self, text='', low=None, high=None):
        super().__init__(text)
        self._validator = QtGui.QIntValidator()
        if low is not None:
            self._validator.setBottom(low)
        if high is not None:
            self._validator.setTop(high)
        self.setValidator(self._validator)

    def value(self):
        return int(self.text())

    def setValue(self, value):
        self.setText(str(int(value)))


class DoubleLineEdit(QtWidgets.QWidget):
    ''' Widget with two line edits for Doubles '''
    editingFinished = QtCore.pyqtSignal()

    def __init__(self, value1=0, value2=0, label1='', label2=''):
        super().__init__()
        self.line1 = QtWidgets.QLineEdit(str(value1))
        self.line2 = QtWidgets.QLineEdit(str(value2))
        self.line1.setValidator(InfValidator())
        self.line2.setValidator(InfValidator())
        layout = QtWidgets.QFormLayout()
        layout.addRow(label1, self.line1)
        layout.addRow(label2, self.line2)
        self.setLayout(layout)

        self.line1.editingFinished.connect(self.editingFinished)
        self.line2.editingFinished.connect(self.editingFinished)

    def getValue(self):
        ''' Return tuple value of two lines '''
        try:
            val1 = float(self.line1.text())
        except ValueError:
            val1 = 0
        try:
            val2 = float(self.line2.text())
        except ValueError:
            val2 = 0
        return val1, val2

    def setValue(self, value1, value2):
        ''' Set value of both lines '''
        self.line1.setText(f'{value1:.5g}')
        self.line2.setText(f'{value2:.5g}')


class LineEditLabelWidget(QtWidgets.QWidget):
    ''' Class for a line edit and label '''
    def __init__(self, label='', text=''):
        super().__init__()
        self._label = QtWidgets.QLabel(label)
        self._text = QtWidgets.QLineEdit(text)
        layout = QtWidgets.QHBoxLayout()
        layout.addWidget(self._label)
        layout.addWidget(self._text)
        layout.addStretch()
        layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)

    def __getattr__(self, name):
        ''' Get all other attributes from the lineedit widget '''
        return getattr(self._text, name)


class SpinBoxLabelWidget(QtWidgets.QWidget):
    ''' Class for a DoubleSpinBox and label '''
    def __init__(self, label='', value=0, rng=None):
        super().__init__()
        self._label = QtWidgets.QLabel(label)
        self._spinbox = QtWidgets.QDoubleSpinBox()
        self._spinbox.setValue(value)
        if rng is not None:
            self._spinbox.setRange(*rng)
        layout = QtWidgets.QHBoxLayout()
        layout.addWidget(self._label)
        layout.addWidget(self._spinbox)
        layout.addStretch()
        layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)

    def __getattr__(self, name):
        ''' Get all other attributes from the spinbox widget '''
        return getattr(self._spinbox, name)


class ListSelectWidget(QtWidgets.QListWidget):
    ''' List Widget with multi-selection on click '''
    checkChange = QtCore.pyqtSignal(int)

    def __init__(self):
        super().__init__()
        self.itemChanged.connect(self.itemcheck)

    def addItems(self, itemlist):
        self.clear()
        for i in itemlist:
            item = QtWidgets.QListWidgetItem(i)
            item.setFlags(item.flags() | QtCore.Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(QtCore.Qt.CheckState.Unchecked)
            self.addItem(item)

    def itemcheck(self, item):
        self.checkChange.emit(self.row(item))

    def getSelectedValues(self):
        sel = []
        for row in range(self.count()):
            if self.item(row).checkState() == QtCore.Qt.CheckState.Checked:
                sel.append(self.item(row).text())
        return sel

    def getSelectedIndexes(self):
        sel = []
        for row in range(self.count()):
            if self.item(row).checkState() == QtCore.Qt.CheckState.Checked:
                sel.append(row)
        return sel

    def selectAll(self):
        ''' Select all items '''
        with BlockedSignals(self):
            for i in range(self.count()):
                self.item(i).setCheckState(QtCore.Qt.CheckState.Checked)

    def selectIndex(self, idxs):
        ''' Select items with index in idxs '''
        with BlockedSignals(self):
            for i in range(self.count()):
                self.item(i).setCheckState(QtCore.Qt.CheckState.Checked if i in idxs else QtCore.Qt.CheckState.Unchecked)

    def selectNone(self):
        with BlockedSignals(self):
            for i in range(self.count()):
                self.item(i).setCheckState(QtCore.Qt.CheckState.Unchecked)
