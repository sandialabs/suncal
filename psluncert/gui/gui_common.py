''' Functionality shared by gui pages. ALL pages should import this file to get exception handler
    if run independently of gui_main.
'''

import os
import sys
import traceback
from PyQt5 import QtWidgets, QtCore, QtGui
import matplotlib as mpl

from . import configmgr
from . import icons
from . import logo
from .. import output


# Characters and colors
CHR_ENDASH = u'\u2013' # Better minus sign for button
CHR_ELLIPSIS = u'\u2026'

COLOR_INVALID = QtGui.QBrush(QtCore.Qt.red)
COLOR_OK = QtGui.QBrush(QtCore.Qt.white)
COLOR_UNUSED = QtGui.QBrush(QtCore.Qt.gray)
COLOR_HIGHLIGHT = QtGui.QBrush(QtGui.QColor(255, 0, 0, 127))

settings = configmgr.Settings()

# Icon names for each calculation method
iconname = {'uncertainty': 'target',
            'curvefit': 'curvefit',
            'risk': 'risk',
            'sweep': 'targetlist',
            'reverse': 'calipers',
            'reversesweep': 'rulersweep',
            'anova': 'boxplot'}


# System exception handler. Not pretty, but better than just shutting down.
def handle_exception(exc_type, exc_value, exc_traceback):
    ''' Show exceptions in message box and print to console. '''
    if isinstance(exc_value, KeyboardInterrupt):
        return
    msg = ''.join(traceback.format_exception(exc_type, exc_value, exc_traceback))
    print(msg)
    msgbox = QtWidgets.QMessageBox()
    msgbox.setWindowTitle('Uncertainty Calculator')
    msgbox.setText('The following exception occurred.')
    msgbox.setInformativeText(msg)
    msgbox.exec_()
sys.excepthook = handle_exception


# Load Pandoc/Latex paths from settings into output module
pandoc = settings.getPandocPath()
latex = settings.getLatexPath()
if pandoc and os.path.exists(pandoc):
    output.pandoc_path = pandoc
if latex and os.path.exists(latex):
    output.latex_path = latex


def set_plot_style():
    ''' Configure matplotlib with plot style from saved settings '''
    mpl.rcParams.update(mpl.rcParamsDefault)
    mpl.style.use(settings.getStyle())
    for k, v in settings.getCustomStyle().items():
        try:
            mpl.style.use({k: v})  # Override anything in base style
        except ValueError:
            print('Bad parameter {} for key {}'.format(v, k))
    output.setup_mplparams()  # Override some font things always


# This function allows switching file path when run from pyInstaller.
def resource_path(relative):
    ''' Get absolute file path for resource. Will switch between pyInstaller tmp dir and gui folder '''
    try:
        base = sys._MEIPASS  # MEIPASS is added by pyinstaller
    except AttributeError:
        base = os.path.dirname(__file__)
    return os.path.join(base, relative)


def load_icon(name):
    ''' Load an icon from the icons file by name '''
    img = QtGui.QPixmap()
    img.loadFromData(QtCore.QByteArray.fromBase64(getattr(icons, name)), format='SVG')
    return QtGui.QIcon(img)


def get_logo(pixmap=False):
    ''' Load the app icon/logo '''
    img = QtGui.QPixmap()
    img.loadFromData(QtCore.QByteArray.fromBase64(logo.logo), format='PNG')
    if pixmap:
        return img
    else:
        return QtGui.QIcon(img)


def get_rptargs():
    ''' Get arguments to pass to reports from settings dialog '''
    return {'n': settings.getSigfigs(),
            'fmt': settings.getNumformat()}
