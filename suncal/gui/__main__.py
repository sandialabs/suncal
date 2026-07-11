#!/usr/bin/env python
''' PSL Uncertainty Calculator - User Interface Main '''

# From suncal root, run:
#     python -m nuitka suncal/gui
#
# nuitka-project: --enable-plugin=pyqt6
# nuitka-project: --include-package-data=suncal:*.mplstyle
# nuitka-project: --include-package-data=suncal.gui:SUNCALmanual.pdf
# nuitka-project: --product-name="Suncal - Sandia Uncertainty Calculator"
# nuitka-project-set: SUNVERSION = __import__("suncal").__version__
# nuitka-project: --product-version={SUNVERSION}
# nuitka-project: --file-version={SUNVERSION}
# nuitka-project: --file-description=Suncal
# nuitka-project: --copyright="Sandia National Laboratories"

# nuitka-project-if: {OS} == "Windows":
#    nuitka-project: --output-filename=Suncal.exe
#    nuitka-project: --windows-console-mode=attach
#    nuitka-project: --mode=onefile
#    nuitka-project: --windows-icon-from-ico=suncal/gui/icons/PSLcal_logo.ico
# nuitka-project-if: {OS} == "Darwin":
#   nuitka-project: --macos-create-app-bundle
#   nuitka-project: --macos-app-icon=suncal/gui/icons/PSLcal_logo.ico


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

    Copyright 2019-2026 National Technology & Engineering
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

    mainwindow = gui.gui_main.MainGUI()
    if len(sys.argv) > 1:
        mainwindow._load_project(sys.argv[1])

    mainwindow.show()
    app.exec()


if __name__ == '__main__':
    main()
