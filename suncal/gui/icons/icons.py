''' GUI Icon Lookup '''
from PyQt6 import QtCore, QtGui

from . import pngs
from . import snllogo
from .. import gui_styles


def icon(name):
    ''' Load an icon from the icons file by name '''
    if gui_styles.isdark():
        name += '_dark'
    img = QtGui.QPixmap()
    img.loadFromData(QtCore.QByteArray.fromBase64(getattr(pngs, name)), format='SVG')
    return QtGui.QIcon(img)


def appicon(pixmap=False):
    ''' Load the app icon/logo

        Args:
            pixmap: Return a QPixmap if True, a QIcon if False.
    '''
    img = QtGui.QPixmap()
    img.loadFromData(QtCore.QByteArray.fromBase64(snllogo.logo), format='PNG')
    if pixmap:
        return img
    return QtGui.QIcon(img)


def logo_snl(pixmap=False):
    ''' Load the SNL thunderbird icon/logo

        Args:
            pixmap: Return a QPixmap if True, a QIcon if False.
'''
    img = QtGui.QPixmap()
    img.loadFromData(QtCore.QByteArray.fromBase64(snllogo.logosnl), format='PNG')
    if pixmap:
        return img
    return QtGui.QIcon(img)
