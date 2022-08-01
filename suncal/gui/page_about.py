''' About box for GUI '''

from PyQt5 import QtWidgets

from .. import version
from . import gui_common
from .licenses import licenses


class AboutUC(QtWidgets.QWidget):
    ''' Widget with the normal "About" information '''

    ABOUT = '''<font size=6>Uncertainty Calculator</font><br>
Version: {} - {}<br><br>
<font size=5>Primary Standards Lab<br>Sandia National Laboratories<br></font>
<font size=4>uncertainty@sandia.gov<br><br></font>
<font size=3>
Copyright 2019-2022 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
<br>Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government
<br>retains certain rights in this software.
</font>'''.format(version.__version__, version.__date__)

    def __init__(self, parent=None):
        super().__init__(parent=parent)
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(QtWidgets.QLabel(self.ABOUT))
        snlico = gui_common.get_snllogo(pixmap=True)
        ratio = QtWidgets.QApplication.instance().devicePixelRatio()
        snlico.setDevicePixelRatio(ratio)
        snllbl = QtWidgets.QLabel()
        snllbl.setPixmap(snlico)
        layout.addStretch()
        layout.addWidget(snllbl)
        self.setLayout(layout)


class AboutBox(QtWidgets.QDialog):
    ''' About dialog with copyright, credits, and license information '''
    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.setWindowTitle('Sandia PSL Uncertainty Calculator')
        self.setMinimumHeight(450)
        font = self.font()
        font.setPointSize(10)
        self.setFont(font)

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
