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


QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling)
QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps)
app = QtWidgets.QApplication(sys.argv)
pxratio = app.devicePixelRatio()

message = f'''Suncal - Sandia Uncertainty Calculator

Version: {version.__version__} - {version.__date__}
Primary Standards Lab
Sandia National Laboratories
uncertainty@sandia.gov

Copyright 2019-2022 National Technology & Engineering
Solutions of Sandia, LLC (NTESS). Under the terms
of Contract DE-NA0003525 with NTESS, the U.S.
Government retains certain rights in this software.
'''

pixmap = QtGui.QPixmap(480*pxratio, 320*pxratio)
pixmap.fill(app.palette().color(QtGui.QPalette.Window))
painter = QtGui.QPainter(pixmap)
painter.drawPixmap(10*pxratio, 250*pxratio, gui_common.get_snllogo(pixmap=True))
pixmap.setDevicePixelRatio(pxratio)
splash = QtWidgets.QSplashScreen(pixmap, QtCore.Qt.SplashScreen)
font = splash.font()
font.setPointSize(12)
splash.setFont(font)

splash.showMessage(message)
splash.show()
splash.repaint()
QtCore.QTimer.singleShot(2000, splash.close)

app.processEvents()
app.setWindowIcon(gui.gui_common.get_logo())
main = gui.gui_main.MainGUI()
main.show()
app.exec_()
