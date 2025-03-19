#!/usr/bin/env python
''' PSL Uncertainty Calculator - User Interface Main '''
import sys
from PyQt6 import QtWidgets, QtCore, QtGui
import markdown

from suncal import gui
from suncal.gui import gui_common  # Install QT breakpoint hook
from suncal.gui import gui_math
from suncal.gui.icons import logo_snl, appicon
from suncal import version


def main():
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle('Fusion')  # Switches light/dark modes

    message = f'''Suncal - Sandia Uncertainty Calculator

    Version: {version.__version__} - {version.__date__}
    Primary Standards Lab
    Sandia National Laboratories
    uncertainty@sandia.gov

    Copyright 2019-2025 National Technology & Engineering
    Solutions of Sandia, LLC (NTESS). Under the terms
    of Contract DE-NA0003525 with NTESS, the U.S.
    Government retains certain rights in this software.
    '''
    pixmap = QtGui.QPixmap(int(480), int(320))
    pixmap.fill(app.palette().color(QtGui.QPalette.ColorRole.Window))
    painter = QtGui.QPainter(pixmap)
    painter.drawPixmap(int(10), int(250), logo_snl(pixmap=True))
    painter.end()
    splash = QtWidgets.QSplashScreen(pixmap)
    font = splash.font()
    font.setPointSize(12)
    splash.setFont(font)

    color = app.palette().color(QtGui.QPalette.ColorRole.WindowText)
    splash.showMessage(message, color=color)
    splash.show()
    splash.repaint()
    QtCore.QTimer.singleShot(3000, splash.close)

    app.processEvents()
    app.setWindowIcon(appicon())

    # This line forces Matplotlib to load in its fonts (taking ~1 sec),
    # and Markdown to load/cache its extension (~.5 sec) now
    # rather than when the user opens the first project component.
    gui_math.pixmap_from_latex('x')
    markdown.markdown('x', extensions=['markdown.extensions.tables'])

    main = gui.gui_main.MainGUI()
    main.show()
    app.exec()


if __name__ == '__main__':
    main()
