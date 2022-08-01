''' Widget for converting units and showing dimensionality '''

from PyQt5 import QtWidgets
from pint import DimensionalityError, UndefinedUnitError

from . import gui_common
from .. import unitmgr
from .. import report


class UnitsConverter(QtWidgets.QDialog):
    ''' Dialog for parsing and converting units. '''
    def __init__(self, parent=None):
        super().__init__(parent=parent)
        font = self.font()
        font.setPointSize(10)
        self.setFont(font)
        self.setWindowTitle('Units Converter')
        self.valuein = QtWidgets.QLineEdit('1.0')
        self.valueout = QtWidgets.QLabel('1.0')
        self.unitin = QtWidgets.QLineEdit('meter')
        self.unitout = QtWidgets.QLineEdit('meter')

        self.namein = QtWidgets.QLabel()
        self.nameout = QtWidgets.QLabel()
        self.abbrin = QtWidgets.QLabel()
        self.abbrout = QtWidgets.QLabel()
        self.dimin = QtWidgets.QLabel()
        self.dimout = QtWidgets.QLabel()
        self.dimok = QtWidgets.QLabel(gui_common.CHR_RARROW)
        self.msg = QtWidgets.QLabel()

        layout = QtWidgets.QVBoxLayout()
        hlayout = QtWidgets.QGridLayout()
        hlayout.addWidget(QtWidgets.QLabel('Value'), 0, 0)
        hlayout.addWidget(QtWidgets.QLabel('Units'), 0, 1)
        hlayout.addWidget(QtWidgets.QLabel('Value'), 0, 3)
        hlayout.addWidget(QtWidgets.QLabel('Units'), 0, 4)
        hlayout.addWidget(self.valuein, 1, 0)
        hlayout.addWidget(self.unitin, 1, 1)
        hlayout.addWidget(QtWidgets.QLabel(gui_common.CHR_RARROW), 1, 2)
        hlayout.addWidget(self.valueout, 1, 3)
        hlayout.addWidget(self.unitout, 1, 4)
        hlayout.addWidget(QtWidgets.QLabel('Unit Name:'), 2, 0)
        hlayout.addWidget(self.namein, 2, 1)
        hlayout.addWidget(self.nameout, 2, 4)
        hlayout.addWidget(QtWidgets.QLabel('Abbreviation:'), 3, 0)
        hlayout.addWidget(self.abbrin, 3, 1)
        hlayout.addWidget(self.abbrout, 3, 4)
        hlayout.addWidget(QtWidgets.QLabel('Dimensionality:'), 4, 0)
        hlayout.addWidget(self.dimin, 4, 1)
        hlayout.addWidget(self.dimok, 4, 2)
        hlayout.addWidget(self.dimout, 4, 4)
        layout.addLayout(hlayout)
        layout.addWidget(self.msg)
        self.setLayout(layout)
        self.update()

        self.valuein.editingFinished.connect(self.update)
        self.unitin.editingFinished.connect(self.update)
        self.unitout.editingFinished.connect(self.update)

    def update(self):
        ''' Recalculate and update the fields '''
        msg = []
        try:
            inpt = float(self.valuein.text())
        except ValueError:
            msg.append('Invalid number {}'.format(self.valuein.text()))
            inpt = 0

        try:
            uin = unitmgr.parse_units(self.unitin.text())
        except (AttributeError, UndefinedUnitError, ValueError, TypeError):
            msg.append('Undefined unit {}'.format(self.unitin.text()))
            self.namein.setText('---')
            self.abbrin.setText('---')
            self.dimin.setText('---')
            uin = None
        else:
            gui_common.setLabelTex(self.namein, report.Unit(uin, abbr=False).latex(escape=False))
            gui_common.setLabelTex(self.abbrin, report.Unit(uin, abbr=True).latex(escape=False))
            gui_common.setLabelTex(self.dimin, report.Unit(uin.dimensionality).latex(escape=False))

        try:
            uout = unitmgr.parse_units(self.unitout.text())
        except UndefinedUnitError:
            msg.append('Undefined unit {}'.format(self.unitout.text()))
            uout = None
            self.nameout.setText('---')
            self.abbrout.setText('---')
            self.dimout.setText('---')
        else:
            gui_common.setLabelTex(self.nameout, report.Unit(uout, abbr=False).latex(escape=False))
            gui_common.setLabelTex(self.abbrout, report.Unit(uout, abbr=True).latex(escape=False))
            gui_common.setLabelTex(self.dimout, report.Unit(uout.dimensionality).latex(escape=False))

        if uin is not None and uout is not None:
            try:
                valout = (inpt * uin).to(uout).magnitude
            except DimensionalityError:
                msg.append('Dimensionality Mismatch')
                self.dimok.setText(gui_common.CHR_X_RED)
            else:
                self.valueout.setText(format(valout, '.4g'))
                self.dimok.setText(gui_common.CHR_RARROW)

        if len(msg) > 0:
            self.valueout.setText('---')
            self.msg.setText('<font color="Red">' + '<br>'.join(msg) + '</font>')
        else:
            self.msg.setText('')
