''' Render Latex Math '''
from PyQt6 import QtWidgets, QtGui

from ..common import uparser, report
from . import gui_styles


def pixmap_from_latex(expr: str, fontsize: float = 16) -> QtGui.QPixmap:
    ''' Render the LaTeX math expression to a QPixmap '''
    tex = uparser.parse_math_with_quantities_to_tex(expr)
    math = report.Math.from_latex(tex)
    return pixmap_from_reportmath(math, fontsize=fontsize)


def pixmap_from_sympy(expr, fontsize: float = 16) -> QtGui.QPixmap:
    ''' Render the Sympy expression to a QPixmap '''
    math = report.Math.from_sympy(expr)
    return pixmap_from_reportmath(math, fontsize=fontsize)


def pixmap_from_reportmath(math: report.Math, fontsize: float = 16) -> QtGui.QPixmap:
    ''' Render the report.Math object to a QPixmap '''
    color = gui_styles.color.math
    ratio = QtWidgets.QApplication.instance().primaryScreen().devicePixelRatio()
    svgbuf = math.svg_buf(fontsize=fontsize*ratio, color=color)
    px = QtGui.QPixmap()
    px.loadFromData(svgbuf.read())
    px.setDevicePixelRatio(ratio)
    return px
