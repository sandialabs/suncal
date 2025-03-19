''' Widgets for PDF Popup '''
from contextlib import suppress
from PyQt6 import QtWidgets, QtCore
import numpy as np

from ...common import distributions, ttable
from .. import gui_common
from ..gui_settings import gui_settings
from .table import ReadOnlyTableItem, EditableTableItem
from .combo import ComboNoWheel
from ...mqa.pdf import Pdf


class PdfPopupTable(QtWidgets.QTableWidget):
    ''' A Table for editing a PDF '''
    COL_NAME = 0
    COL_VALUE = 1

    ROW_DIST = 0
    ROW_UNCERT = 1
    ROW_K = 2
    ROW_CONF = 3
    ROW_DEGF = 4

    changed = QtCore.pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.setColumnCount(2)
        self.setRowCount(1)
        self.setHorizontalHeaderLabels(['Parameter', 'Value'])
        self.dist = None
        self.degf = np.inf
        self.itemChanged.connect(self.edit_item)

    def fill_table(self):
        ''' Fill the table with the distribution parameters '''
        with gui_common.BlockedSignals(self):
            self.clear()
            self.setHorizontalHeaderLabels(['Parameter', 'Value'])

            config = self.dist.get_config()
            distname = config.get('dist')
            cmbdist = ComboNoWheel()
            cmbdist.addItems(gui_settings.distributions)
            cmbdist.setCurrentIndex(cmbdist.findText(distname))
            if cmbdist.currentText() == '':
                cmbdist.addItem(distname)
                cmbdist.setCurrentIndex(cmbdist.count()-1)
            self.setItem(self.ROW_DIST, self.COL_NAME, ReadOnlyTableItem('Distribution'))
            self.setCellWidget(self.ROW_DIST, self.COL_VALUE, cmbdist)

            if distname in ['norm', 'normal', 't']:
                self.setRowCount(5)
                if 'conf' in config:
                    conf = float(config['conf'])
                    k = ttable.k_factor(conf, config.get('degf', np.inf))
                else:
                    k = float(config.get('k', 1))
                    conf = ttable.confidence(k, config.get('degf', np.inf))
                uncstr = config.get('unc', k*float(config.get('std', 1)))  # Could be float or string
                with suppress(ValueError):
                    uncstr = f'{uncstr:.5g}'

                self.setItem(self.ROW_DEGF, self.COL_NAME, ReadOnlyTableItem('Degrees Freedom'))
                self.setItem(self.ROW_DEGF, self.COL_VALUE, EditableTableItem(f'{self.degf:.2f}'))
                self.setItem(self.ROW_CONF, self.COL_NAME, ReadOnlyTableItem('Confidence'))
                self.setItem(self.ROW_CONF, self.COL_VALUE, EditableTableItem(f'{float(conf)*100:.2f}%'))
                self.setItem(self.ROW_K, self.COL_NAME, ReadOnlyTableItem('k'))
                self.setItem(self.ROW_K, self.COL_VALUE, EditableTableItem(f'{k:.2f}'))
                self.setItem(self.ROW_UNCERT, self.COL_NAME, ReadOnlyTableItem('Uncertainty'))
                self.setItem(self.ROW_UNCERT, self.COL_VALUE, EditableTableItem(uncstr))
            else:
                self.setRowCount(len(self.dist.argnames)+1)
                for row, arg in enumerate(reversed(sorted(self.dist.argnames))):
                    self.setItem(row+1, self.COL_NAME, ReadOnlyTableItem(arg))
                    self.setItem(row+1, self.COL_VALUE, EditableTableItem(str(self.dist.kwds.get(arg, 1))))

            cmbdist.currentIndexChanged.connect(self.change_dist)

    def change_dist(self):
        ''' Distribution type in combobox was changed '''
        with gui_common.BlockedSignals(self):
            distname = self.cellWidget(self.ROW_DIST, self.COL_VALUE).currentText()
            if self.dist:
                cfg = self.dist.get_config()
            else:
                cfg = {}
            cfg['dist'] = distname
            self.dist = distributions.get_distribution(distname, **cfg)
            self.fill_table()
        self.changed.emit()

    def edit_item(self, item: QtWidgets.QTableWidgetItem):
        ''' An item was edited '''
        with gui_common.BlockedSignals(self):
            valid = True
            value = item.text()
            row = item.row()
            param = self.item(row, self.COL_NAME).text()
            if self.dist:
                config = self.dist.get_config()
            else:
                config = {}

            try:
                floatval = float(value.strip('%'))
            except ValueError:
                floatval = None

            if param == 'Uncertainty':
                config['unc'] = floatval
                config.pop('std', None)
            elif param == 'k':
                valid = floatval is not None and floatval > 0
                if valid:
                    config['k'] = floatval
                    self.item(self.ROW_CONF, self.COL_VALUE).setText(
                        f'{ttable.confidence(floatval, self.degf)*100:.2f}%')
            elif param == 'Confidence':
                valid = floatval is not None and 0 < floatval < 100
                if valid:
                    config['conf'] = floatval/100  # Assume confidence in percent
                    config.pop('k', None)
                    self.item(self.ROW_K, self.COL_VALUE).setText(
                        f'{ttable.k_factor(floatval/100, self.degf):.2f}')
            elif param == 'Degrees Freedom':
                valid = floatval is not None
                if valid:
                    self.degf = floatval
                if config.get('dist') in ['normal', 'norm', 't']:
                    config['df'] = floatval
                    if 'k' in config:
                        k = float(config.get('k'))
                        self.item(self.ROW_CONF, self.COL_VALUE).setText(
                            f'{ttable.confidence(k, floatval)*100:.2f}%')
                    else:
                        conf = float(config.get('conf', .95))
                        self.item(self.ROW_K, self.COL_VALUE).setText(
                            f'{ttable.k_factor(conf, floatval):.2f}')
            else:
                config[param] = floatval if floatval is not None else value

        if valid:
            config['name'] = config.get('dist', 'norm')
            try:
                self.dist = distributions.get_distribution(**config)
            except (ValueError, TypeError):
                self.dist = None
            self.changed.emit()
        else:
            self.dist = None


class PdfPopupButton(QtWidgets.QToolButton):
    ''' Button for showing a Distribution widget in a popup '''
    changed = QtCore.pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.setText('PDF')
        self.table = PdfPopupTable()
        self.popupAction = QtWidgets.QWidgetAction(self)
        self.popupAction.setDefaultWidget(self.table)
        self.setPopupMode(QtWidgets.QToolButton.ToolButtonPopupMode.InstantPopup)
        self.addAction(self.popupAction)
        self.table.changed.connect(self.changed)
        self.set_distribution(distributions.Distribution('norm', std=1))

    def isvalid(self) -> bool:
        ''' Pdf is valid '''
        return self.table.dist is not None

    def set_distribution(self, dist: distributions.Distribution):
        ''' Set the distribution '''
        self.table.dist = dist
        self.table.fill_table()

    def get_distribution(self) -> distributions.Distribution:
        ''' Get the Distribution'''
        return self.table.dist

    def get_pdf(self) -> Pdf:
        ''' Get the Pdf '''
        dist = self.get_distribution()
        return Pdf.from_dist(dist)

    @property
    def degrees_freedom(self) -> float:
        ''' Degrees of freedom '''
        return self.table.degf

    def get_config(self):
        ''' Get configuration dictionary '''
        dist = self.get_distribution()
        return {
            'dist': dist.get_config(),
            'degrees_freedom': self.table.degf
        }
