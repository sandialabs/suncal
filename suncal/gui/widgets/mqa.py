''' Widgets common to MQA and Measurement System pages '''
from decimal import Decimal
import numpy as np
from PyQt6 import QtWidgets, QtCore

from ...uncertainty.variables import Typeb
from ...common.limit import Limit
from ..gui_common import BlockedSignals
from ..delegates import SuncalDelegate
from ..page_csvload import SelectCSVData
from .combo import FloatLineEdit
from .buttons import PlusMinusButton, PlusButton, MinusButton
from .table import FloatTableWidget
from .pdf import PdfPopupButton


class ToleranceDelegate(SuncalDelegate):
    def __init__(self, required: bool = True):
        super().__init__()
        self.required = required  # Hide the Cancel button

    def cancel_edit(self, editor):
        editor.valid = False
        self.commitData.emit(editor)
        self.closeEditor.emit(editor, QtWidgets.QStyledItemDelegate.EndEditHint.NoHint)

    def createEditor(self, parent, option, index):
        if index.model().data(index, self.ROLE_DISABLE):
            return None

        editor = ToleranceWidget(required=self.required, parent=parent)
        editor.setAutoFillBackground(True)
        editor.cancel.connect(lambda editor=editor: self.cancel_edit(editor))
        return editor

    def updateEditorGeometry(self, editor, option, index):
        editor.setGeometry(option.rect)

    def setEditorData(self, editor, index):
        tolerance = index.model().data(index, self.ROLE_TOLERANCE)
        if tolerance:
            editor.set_limit(tolerance)

    def setModelData(self, editor, model, index):
        if editor.valid:
            tolerance = editor.limit()
            index.model().setData(index, tolerance, self.ROLE_TOLERANCE)
            self.set_displayrole(index, str(tolerance))
        else:
            index.model().setData(index, None, self.ROLE_TOLERANCE)
            self.set_displayrole(index, '')


class ToleranceWidget(QtWidgets.QWidget):
    ''' Widget for entering an absolute tolerance, which may be:
        plus/minus value, max value, min value, or asymmetric + and -
        values, in a Table/Tree Item
    '''
    MODES = ['±', '>', '<', '↔']
    changed = QtCore.pyqtSignal(Limit)
    cancel = QtCore.pyqtSignal()

    def __init__(self, required: bool = True, parent=None):
        super().__init__(parent=parent)
        self.btnMode = QtWidgets.QToolButton()
        self.valid = True
        self.btnMode.setText('±')
        self.value = FloatLineEdit('0')
        self.value2 = FloatLineEdit('1')
        self.btnMode.clicked.connect(self.changemode)
        self.btnCancel = QtWidgets.QToolButton()
        self.btnCancel.setText('✗')
        self.value.editingFinished.connect(lambda: self.changed.emit(self.limit()))
        self.value2.editingFinished.connect(lambda: self.changed.emit(self.limit()))
        self.btnCancel.clicked.connect(self.cancel)
        layout = QtWidgets.QHBoxLayout()
        layout.setSpacing(0)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.value)
        layout.addWidget(self.btnMode)
        layout.addWidget(self.value2)
        if not required:
            layout.addWidget(self.btnCancel)
        layout.addStretch()
        self.setLayout(layout)

    def changemode(self):
        ''' Mode button pressed '''
        idx = self.MODES.index(self.btnMode.text()) + 1
        idx %= len(self.MODES)
        mode = self.MODES[idx]
        self.btnMode.setText(mode)
        self.set_enable()
        self.changed.emit(self.limit())

    def set_enable(self):
        ''' Enable/disable the first value entry '''
        if self.btnMode.text() not in ['↔', '±']:
            self.value.setText('')
            self.value.setEnabled(False)
        else:
            self.value.setText('0')
            self.value.setEnabled(True)

    def limit(self, nominal: str = None) -> Limit:
        ''' Get Limit '''
        mode = self.btnMode.text()

        if mode == '±':
            if nominal is None:
                nominal = Decimal(self.value.text())
            else:
                nominal = Decimal(nominal)
            lim = Limit.from_plusminus(nominal, Decimal(self.value2.text()))
        elif mode == '>':
            lim = Limit(Decimal(self.value2.text()), Decimal('inf'))
        elif mode == '<':
            lim = Limit(Decimal('-inf'), Decimal(self.value2.text()))
        else:
            # Ensure low <= high
            lims = sorted((Decimal(self.value.text()), Decimal(self.value2.text())))
            lim = Limit(*lims)
        return lim

    def set_limit(self, limit: Limit):
        ''' Set the limit to display '''
        if limit is None:
            return
        if limit.symmetric:
            self.btnMode.setText('±')
            self.set_enable()
            self.value.setText(str(limit.nominal))  # These are Decimal type
            self.value2.setText(str(limit.plusminus))
        elif not np.isfinite(limit.flow):
            self.btnMode.setText('<')
            self.set_enable()
            self.value2.setText(str(limit.high))
        elif not np.isfinite(limit.fhigh):
            self.btnMode.setText('>')
            self.set_enable()
            self.value2.setText(str(limit.low))
        else:
            self.btnMode.setText('↔')
            self.set_enable()
            self.value.setText(str(limit.low))
            self.value2.setText(str(limit.high))


class ToleranceCheck(QtWidgets.QWidget):
    ''' Tolerance edit and checkbox '''
    changed = QtCore.pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.tolerance = ToleranceWidget()
        self.chkbox = QtWidgets.QCheckBox()
        layout = QtWidgets.QHBoxLayout()
        layout.addWidget(self.chkbox)
        layout.addWidget(self.tolerance)
        layout.addStretch(3)
        self.setLayout(layout)
        layout.setContentsMargins(0, 0, 0, 0)
        self.tolerance.setEnabled(False)
        self.chkbox.stateChanged.connect(self.set_enable)
        self.tolerance.changed.connect(self.changed)

    def limit(self) -> Limit:
        if self.chkbox.isChecked():
            return self.tolerance.limit()
        return None

    def set_limit(self, limit: Limit):
        if limit:
            self.tolerance.set_limit(limit)
            self.chkbox.setChecked(True)
        else:
            self.chkbox.setChecked(False)

    def isChecked(self):
        return self.chkbox.isChecked()

    def setChecked(self, state: bool):
        self.chkbox.setChecked(state)

    def set_enable(self):
        ''' Change enable state based on checkbox '''
        if self.chkbox.isChecked():
            self.tolerance.setEnabled(True)
        else:
            self.tolerance.setEnabled(False)
        self.changed.emit()


class TypeADialog(QtWidgets.QDialog):
    ''' Entry of Type A measurement data for a quantity '''
    changed = QtCore.pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent=parent, flags=QtCore.Qt.WindowType.Popup)
        self.values = None
        font = self.font()
        font.setPointSize(10)
        self.setFont(font)
        self.setFixedSize(600, 800)
        self.plusminus = PlusMinusButton()
        self.plusminus.btnplus.setToolTip('Add Column')
        self.plusminus.btnminus.setToolTip('Remove Column')
        self.autocorr = QtWidgets.QCheckBox('Adjust for Autocorrelation')
        self.autocorr.setChecked(True)
        self.chkNewmeas = QtWidgets.QCheckBox('Use this data to estimate uncertainty in')
        self.Nnewmeas = QtWidgets.QSpinBox()
        self.Nnewmeas.setRange(1, 999)
        self.Nlabel = QtWidgets.QLabel('new measurements')
        self.table = FloatTableWidget()
        font = self.font()
        font.setPointSize(10)
        self.table.setFont(font)
        self.table.verticalHeader().setDefaultSectionSize(10)
        #self.btnload = QtWidgets.QPushButton('Load CSV...')

        blayout = QtWidgets.QHBoxLayout()
        blayout.addWidget(self.plusminus)
        blayout.addStretch()
        #blayout.addWidget(self.btnload)
        layout = QtWidgets.QVBoxLayout()
        layout.addLayout(blayout)
        layout.addWidget(self.table)
        layout.addWidget(self.autocorr)
        hlayout = QtWidgets.QHBoxLayout()
        hlayout.addWidget(self.chkNewmeas)
        hlayout.addWidget(self.Nnewmeas)
        hlayout.addWidget(self.Nlabel)
        hlayout.addStretch()
        layout.addLayout(hlayout)
        self.setLayout(layout)
        self.fill_table()

        self.plusminus.btnminus.clicked.connect(self.rem_column)
        self.plusminus.btnplus.clicked.connect(self.add_column)
        self.table.valueChanged.connect(self.changed)
        self.autocorr.stateChanged.connect(self.changed)
        #self.btnload.clicked.connect(self.loadcsv)

    def add_column(self):
        ''' Add column to the table '''
        self.table.setColumnCount(self.table.columnCount() + 1)
        if self.table.columnCount() > 1:
            self.autocorr.setEnabled(False)
            self.autocorr.setChecked(False)
        self.changed.emit()

    def rem_column(self):
        ''' Remove column from the table '''
        self.table.setColumnCount(self.table.columnCount() - 1)
        if self.table.columnCount() > 1:
            self.autocorr.setEnabled(False)
            self.autocorr.setChecked(False)
        else:
            self.autocorr.setEnabled(True)
            self.autocorr.setChecked(True)
        self.changed.emit()

    def loadcsv(self):
        ''' Load data from a CSV file '''
        fname, _ = QtWidgets.QFileDialog.getOpenFileName(caption='CSV file to load')
        if fname:
            dlg = SelectCSVData(fname, parent=self)
            if dlg.exec():
                dset = dlg.dataset()
                data = dset.model.data
                self.values = data
                self.fill_table()
            self.changed.emit()

    def set_values(self, values: np.ndarray):
        ''' Set the Quantity to display '''
        self.values = values
        self.fill_table()

    def fill_table(self):
        ''' Fill table with values stored in Quantity '''
        with BlockedSignals(self.table):
            self.table.clear()
            self.table.setRowCount(1)
            self.table.setColumnCount(1)
            if self.values is None:
                return

            data = np.squeeze(self.values)
            if data.size > 2 and data.ndim == 1:
                self.table.setRowCount(len(data))
                self.table.setColumnCount(1)
                for row, value in enumerate(data):
                    self.table.setItem(row, 0, QtWidgets.QTableWidgetItem(str(value)))

            elif data.size > 2 and data.ndim == 2:
                self.table.setRowCount(data.shape[0])
                self.table.setColumnCount(data.shape[1])
                for row, rowvals in enumerate(data):
                    for col, value in enumerate(rowvals):
                        self.table.setItem(row, col, QtWidgets.QTableWidgetItem(str(value)))

    def get_data(self):
        ''' Get measured data as 1D or 2D numpy array '''
        return np.atleast_1d(np.squeeze(self.table.get_table()))

    def data_size(self):
        if self.table.columnCount() == 0 or self.table.rowCount() == 0:
            return 0
        return np.count_nonzero(np.isfinite(self.table.get_table()))


class TypeBWidget(QtWidgets.QWidget):
    ''' Widget for entering a list of Type B PDFs '''
    changed = QtCore.pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.typebs: list[Typeb] = []
        self.widgets: list[TypeBRow] = []
        self.btn_add = PlusButton()
        self.btn_add.clicked.connect(self.add_row)
        self.flayout = QtWidgets.QVBoxLayout()
        self.flayout.addWidget(self.btn_add)
        self.setLayout(self.flayout)
        self.setContentsMargins(0, 0, 0, 0)

    def isvalid(self) -> bool:
        ''' Is distribution valid? '''
        return all(widget.pdf.isvalid() for widget in self.widgets)

    def add_row(self):
        ''' Add a new Type B uncertainty row '''
        typeb = Typeb()
        self.typebs.append(typeb)
        self._add_typeb(typeb)
        self.changed.emit()

    def rem_row(self, widget):
        ''' Remove the selected row '''
        idx = self.widgets.index(widget)
        self.flayout.removeWidget(widget)
        self.typebs.pop(idx)
        self.widgets.pop(idx)
        widget.deleteLater()
        self.changed.emit()

    def _add_typeb(self, typeb):
        ''' Add the Type B uncertainty to the widget '''
        widget = TypeBRow()
        widget.set_distribution(typeb.distribution)
        widget.name.setText(typeb.name)
        widget.changed.connect(lambda widget=widget: self.pdf_changed(widget))
        widget.removed.connect(self.rem_row)
        self.widgets.append(widget)
        # Insert before the "+"
        self.flayout.insertWidget(self.flayout.count()-1, widget)

    def set_typebs(self, typebs: list[Typeb]):
        ''' Set the Quantity to display '''
        self.typebs = typebs
        for w in self.widgets:
            self.flayout.removeWidget(w)
            w.deleteLater()
        self.widgets = []

        for typeb in self.typebs:
            self._add_typeb(typeb)

    def pdf_changed(self, widget):
        ''' A PDF was edited '''
        idx = self.widgets.index(widget)
        typeb = self.typebs[idx]
        dist = widget.get_distribution()
        if dist:
            typeb.name = widget.name.text()
            typeb.distribution = dist
            typeb.distname = dist.name
            typeb.kwargs = dist.kwds
            typeb.degrees_freedom = widget.pdf.degrees_freedom
        self.changed.emit()


class TypeBRow(QtWidgets.QWidget):
    ''' One row of Type B uncertainties '''
    changed = QtCore.pyqtSignal()
    removed = QtCore.pyqtSignal(object)

    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.name = QtWidgets.QLineEdit()
        self.name.setPlaceholderText('Uncertainty Name')
        self.pdf = PdfPopupButton()
        self.remove = MinusButton()
        self.name.setToolTip('Name of the uncertainty')
        self.pdf.setToolTip('Probability Distribution of the uncertainty component')
        self.remove.setToolTip('Remove this uncertainty component')
        layout = QtWidgets.QHBoxLayout()
        layout.addWidget(self.name)
        layout.addWidget(self.pdf)
        layout.addWidget(self.remove)
        layout.addStretch()
        layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)
        self.pdf.changed.connect(self.changed)
        self.name.editingFinished.connect(self.changed)
        self.remove.clicked.connect(lambda: self.removed.emit(self))

    def set_distribution(self, dist):
        ''' Set the distribution to display '''
        self.pdf.set_distribution(dist)

    def get_distribution(self):
        ''' Get the distribution '''
        return self.pdf.get_distribution()


class CheckPdf(QtWidgets.QWidget):
    ''' Checkbox and Pdf Dropdown '''
    def __init__(self, parent=None):
        super().__init__()
        self.chkbox = QtWidgets.QCheckBox(parent=parent)
        self.pdf = PdfPopupButton(parent=parent)
        self.pdf.setEnabled(False)
        layout = QtWidgets.QHBoxLayout()
        layout.addWidget(self.chkbox)
        layout.addWidget(self.pdf)
        layout.addStretch()
        self.setLayout(layout)
        layout.setContentsMargins(0, 0, 0, 0)
        self.chkbox.stateChanged.connect(self.chk_change)

    def chk_change(self):
        ''' Checkbox was clicked '''
        self.pdf.setEnabled(self.chkbox.isChecked())

    def get_pdf(self) -> 'Pdf':
        ''' Get the Pdf or None if not checked '''
        if self.chkbox.isChecked():
            return self.pdf.get_pdf()
        return None
