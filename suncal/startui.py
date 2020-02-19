#!/usr/bin/env python
'''
PSL Uncertainty Calculator

Launch the user interface.

PyInstaller croaks if it calls into gui/gui_main.py directly, so this file was added as a workaround.
'''

from PyQt5 import QtWidgets, QtCore, QtGui
from suncal import gui
from suncal.gui import gui_common
from suncal import version

import sys


message = '''<font size=6>Uncertainty Calculator</font><br>
Version: {} - {}<br><br>
<font size=4>Primary Standards Lab<br>Sandia National Laboratories<br></font>
<font size=4>uncertainty@sandia.gov<br><br></font>
<font size=3>
<br>
Copyright 2019-2020 National Technology & Engineering Solutions<br>of Sandia, LLC (NTESS).
Under the terms of Contract<br>DE-NA0003525 with NTESS, the U.S. Government<br>retains certain rights in this software.'''.format(version.__version__, version.__date__)


app = QtWidgets.QApplication(sys.argv)
app.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling)
app.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps)

pixmap = QtGui.QPixmap(480, 320)
pixmap.fill(app.palette().color(QtGui.QPalette.Window))
painter = QtGui.QPainter(pixmap)
painter.drawPixmap(10, 250, gui_common.get_snllogo(pixmap=True))
splash = QtWidgets.QSplashScreen(pixmap)
splash.showMessage(message)
splash.show()
QtCore.QTimer.singleShot(2000, splash.close)

app.processEvents()
app.setWindowIcon(gui.gui_common.get_logo())
main = gui.gui_main.MainGUI()
main.show()
app.exec_()
