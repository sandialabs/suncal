''' About box for GUI '''

import os
import sys
from PyQt5 import QtWidgets

from .. import version
from . import gui_common
from .licenses import licenses


def resource_path(relative):
    ''' Get absolute file path for resource. Will switch between pyInstaller tmp dir and gui folder '''
    try:
        base = sys._MEIPASS  # MEIPASS is added by pyinstaller
    except AttributeError:
        base = os.path.dirname(__file__)
    return os.path.join(base, relative)


class AboutUC(QtWidgets.QWidget):
    ''' Widget with the normal "About" information '''

    ABOUT = '''<font size=6>Uncertainty Calculator</font><br>
Version: {} - {}<br><br>
<font size=5>Primary Standards Lab<br>Sandia National Laboratories<br></font>
<font size=4>uncertainty@sandia.gov<br><br></font>
<font size=3>
Copyright 2019 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
<br>Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government
<br>retains certain rights in this software.
</font>'''.format(version.__version__, version.__date__)

    def __init__(self, parent=None):
        super(AboutUC, self).__init__(parent=parent)
        ico = gui_common.get_logo(pixmap=True)
        icolbl = QtWidgets.QLabel()
        icolbl.setPixmap(ico)
        llayout = QtWidgets.QVBoxLayout()
        llayout.addWidget(icolbl)
        llayout.addStretch()
        rlayout = QtWidgets.QVBoxLayout()
        rlayout.addWidget(QtWidgets.QLabel(self.ABOUT))
        rlayout.addStretch()
        layout = QtWidgets.QHBoxLayout()
        layout.addLayout(llayout)
        layout.addLayout(rlayout)
        self.setLayout(layout)


class AboutBox(QtWidgets.QDialog):
    ''' About dialog with copyright, credits, and license information '''
    def __init__(self, parent=None):
        super(AboutBox, self).__init__(parent=parent)
        self.setWindowTitle('PSL Uncertainty Calculator')
        self.setMinimumHeight(500)
        self.ok = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok)
        self.ok.accepted.connect(self.accept)
        self.txtLicense = QtWidgets.QTextEdit()
        self.txtLicense.setReadOnly(True)
        self.txtLicense.setHtml(licenses)
        self.tab = QtWidgets.QTabWidget()
        self.tab.addTab(AboutUC(), 'About')
        self.tab.addTab(self.txtLicense, 'Acknowledgements')
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.tab)
        layout.addWidget(self.ok)
        self.setLayout(layout)


def show():
    ''' Show the about dialog '''
    dlg = AboutBox()
    dlg.exec_()
