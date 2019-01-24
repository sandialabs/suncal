''' PSL Uncertainty Calculator - Sandia National Labs

User Interface for Uncertainty Calculator

Usage:
>>> from PyQt4 import QtGui
>>> from UncertCalc import gui
>>> import sys
>>> app = QtGui.QApplication(sys.argv)
>>> main = gui.UncCalcGUI()
>>> main.show()
>>> app.exec_()
'''

from .gui_main import MainGUI
from ..version import __version__, __date__

