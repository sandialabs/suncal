''' System-level GUI functions '''

import os
import sys
import traceback
from PyQt6 import QtWidgets, QtCore, QtGui


# Breakpoint handler for breakpoint() function to disable QT problems when breaking
def _qtbreakpoint(*args, **kwargs):
    from pdb import set_trace
    QtCore.pyqtRemoveInputHook()
    set_trace()


sys.breakpointhook = _qtbreakpoint


# System exception handler. Not pretty, but better than just shutting down.
def handle_exception(exc_type, exc_value, exc_traceback):
    ''' Show exceptions in message box and print to console. '''
    if isinstance(exc_value, KeyboardInterrupt):
        return
    msg = ''.join(traceback.format_exception(exc_type, exc_value, exc_traceback))
    print(msg)
    msgbox = QtWidgets.QMessageBox()
    msgbox.setWindowTitle('Suncal')
    msgbox.setText('The following exception occurred.')
    msgbox.setInformativeText(msg)
    msgbox.exec()


sys.excepthook = handle_exception


def centerWindow(window, w, h):
    ''' Set window geometry so it appears centered in the window
        If window size is too big, maximize it
    '''
    desktopsize = QtGui.QGuiApplication.primaryScreen().availableGeometry()
    window.setGeometry(desktopsize.width()//2 - w//2, desktopsize.height()//2 - h//2, w, h)  # Center window on screen
    if h >= desktopsize.height() or w >= desktopsize.width():
        window.showMaximized()


class InfValidator(QtGui.QDoubleValidator):
    ''' Double Validator that allows "inf" and "-inf" entry '''
    def validate(self, s, pos):
        ''' Validate the string '''
        if s.lower() in ['inf', '-inf']:
            return QtGui.QValidator.State.Acceptable, s, pos
        elif s.lower() in '-inf':
            return QtGui.QValidator.State.Intermediate, s, pos
        return super().validate(s, pos)


# This function allows switching file path when run from pyInstaller.
def resource_path(relative):
    ''' Get absolute file path for resource. Will switch between pyInstaller tmp dir and gui folder '''
    try:
        base = sys._MEIPASS  # MEIPASS is added by pyinstaller
    except AttributeError:
        base = os.path.dirname(__file__)
    return os.path.join(base, relative)


class BlockedSignals:
    ''' Context manager for blocking pyqt signals that
        restores the signal block to its previous state when done.

        with BlockedSignal(widget):
            ...
    '''
    def __init__(self, widget):
        self.widget = widget
        self._saved_state = self.widget.signalsBlocked()

    def __enter__(self):
        self._saved_state = self.widget.signalsBlocked()
        self.widget.blockSignals(True)

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.widget.blockSignals(self._saved_state)