''' Style and themes including dark and light mode themes '''
from contextlib import contextmanager
import logging
import matplotlib as mpl
from PyQt6 import QtWidgets, QtCore, QtGui

from ..common import plotting
from .gui_settings import gui_settings


def isdark() -> bool:
    ''' Is dark mode enabled? '''
    return QtWidgets.QApplication.instance().styleHints().colorScheme() == QtCore.Qt.ColorScheme.Dark


def darkmode_signal():
    ''' Signal emitted when dark mode changes '''
    return QtWidgets.QApplication.instance().styleHints().colorSchemeChanged


class _Color:
    ''' Color options for Light and Dark modes '''
    _DARK = {
        'transparent': QtGui.QBrush(QtCore.Qt.GlobalColor.transparent),
        'invalid': QtGui.QBrush(QtCore.Qt.GlobalColor.darkRed),
        'column_highlight': QtGui.QBrush(QtGui.QColor(68, 85, 68, 255)),
        'mathtext': '#cccccc'}
    _LIGHT = {
        'transparent': QtGui.QBrush(QtCore.Qt.GlobalColor.transparent),
        'invalid': QtGui.QBrush(QtCore.Qt.GlobalColor.darkRed),
        'column_highlight': QtGui.QBrush(QtGui.QColor(204, 255, 204, 255)),
        'mathtext': 'black'}

    @property
    def _themedict(self):
        ''' Get color dictionary for the current dark/light theme '''
        return self._DARK if isdark() else self._LIGHT

    @property
    def math(self) -> str:
        ''' Color of math text, as string for matplotlib '''
        return self._themedict.get('mathtext')

    @property
    def invalid(self) -> QtGui.QBrush:
        ''' Color for table cells with invalid input '''
        return self._themedict.get('invalid')

    @property
    def column_highlight(self) -> QtGui.QBrush:
        ''' Highlighted/selected column in data importer '''
        return self._themedict.get('column_highlight')

    @property
    def transparent(self) -> QtGui.QBrush:
        ''' Transparent color '''
        return self._themedict.get('transparent')

    def set_plot_style(self):
        ''' Configure matplotlib with plot style from saved settings '''
        plotting.activate_plotstyle(gui_settings.plot_style, dark=isdark())
        if (cycle := gui_settings.color_cycle):
            mpl.style.use({'axes.prop_cycle': cycle})
        for k, v in gui_settings.plotparams.items():
            try:
                mpl.style.use({k: v})  # Override anything in base style
            except ValueError:
                logging.warn(f'Bad parameter {v} for key {k}')


color = _Color()


@contextmanager
def LightPlotstyle():
    ''' Activate a light-mode plot style using context manager '''
    plotting.activate_plotstyle(gui_settings.plot_style, dark=False)
    yield
    plotting.activate_plotstyle(gui_settings.plot_style, dark=isdark())
