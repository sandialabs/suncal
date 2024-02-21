''' Tool/Dialog for sweeping risk curves '''
import sys
from collections import namedtuple
import numpy as np
from PyQt6 import QtWidgets, QtCore
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

from . import widgets
from . import gui_common
from ..risk.report.risk import risk_sweeper


class RiskSweepSetup(QtWidgets.QGroupBox):
    ''' Sweep controls '''
    def __init__(self, parent=None):
        super().__init__('Sweep Setup', parent=parent)
        items = ['Sweep (x) variable', 'Step (z) variable', 'Constant']
        self.itpvssigma = QtWidgets.QComboBox()
        self.itpvssigma.addItems(['In-tol probability %', 'SL/Process Std. Dev.'])
        self.itp = QtWidgets.QComboBox()
        self.tur = QtWidgets.QComboBox()
        self.gbf = QtWidgets.QComboBox()
        self.procbias = QtWidgets.QComboBox()
        self.testbias = QtWidgets.QComboBox()
        self.itp.addItems(items)
        self.tur.addItems(items)
        self.procbias.addItems(items)
        self.testbias.addItems(items)
        self.gbf.addItems(items + ['RDS', 'Dobbert', 'RP10', '95% Test'])
        self.tur.setCurrentIndex(1)  # Z
        self.gbf.setCurrentIndex(2)  # Fixed
        self.procbias.setCurrentIndex(2)  # Fixed
        self.testbias.setCurrentIndex(2)  # Fixed
        self.itpval = widgets.FloatLineEdit('90')
        self.turval = widgets.FloatLineEdit('4')
        self.gbfval = widgets.FloatLineEdit('1')
        self.procbiasval = widgets.FloatLineEdit('0')
        self.testbiasval = widgets.FloatLineEdit('0')

        self.xstart = widgets.FloatLineEdit('50')
        self.xstop = widgets.FloatLineEdit('90')
        self.xpts = widgets.FloatLineEdit('20', low=2)
        self.zvals = QtWidgets.QLineEdit()
        self.zvals.setText('1.5, 2, 3, 4')
        self.itpval.setVisible(False)
        self.turval.setVisible(False)
        self.plot3d = QtWidgets.QCheckBox('3D Plot')
        self.logy = QtWidgets.QCheckBox('Log Scale')
        self.plottype = QtWidgets.QComboBox()
        self.plottype.addItems(['PFA', 'PFR', 'Both'])
        self.btnrefresh = QtWidgets.QPushButton('Replot')
        layout = QtWidgets.QGridLayout()
        layout.addWidget(self.itpvssigma, 0, 0)
        layout.addWidget(self.itp, 0, 1)
        layout.addWidget(self.itpval, 0, 2)
        layout.addWidget(QtWidgets.QLabel('Test uncertainty ratio'), 1, 0)
        layout.addWidget(self.tur, 1, 1)
        layout.addWidget(self.turval, 1, 2)
        layout.addWidget(QtWidgets.QLabel('Guardband factor'), 2, 0)
        layout.addWidget(self.gbf, 2, 1)
        layout.addWidget(self.gbfval, 2, 2)
        layout.addWidget(QtWidgets.QLabel('Process Bias %'), 3, 0)
        layout.addWidget(self.procbias, 3, 1)
        layout.addWidget(self.procbiasval, 3, 2)
        layout.addWidget(QtWidgets.QLabel('Measurement Bias %'), 4, 0)
        layout.addWidget(self.testbias, 4, 1)
        layout.addWidget(self.testbiasval, 4, 2)
        layout.addWidget(widgets.QHLine(), 5, 0, 1, 3)
        layout.addWidget(QtWidgets.QLabel('Sweep (x) Start:'), 6, 0)
        layout.addWidget(self.xstart, 6, 1, 1, 2)
        layout.addWidget(QtWidgets.QLabel('Sweep (x) Stop:'), 7, 0)
        layout.addWidget(self.xstop, 7, 1, 1, 2)
        layout.addWidget(QtWidgets.QLabel('# Points (x):'), 8, 0)
        layout.addWidget(self.xpts, 8, 1, 1, 2)
        layout.addWidget(QtWidgets.QLabel('Step Values (z):'), 9, 0)
        layout.addWidget(self.zvals, 9, 1, 1, 2)
        layout.addWidget(widgets.QHLine(), 10, 0, 1, 3)
        layout.addWidget(QtWidgets.QLabel('Plot'), 11, 0)
        layout.addWidget(self.plottype, 11, 1)
        layout.addWidget(self.plot3d, 12, 1, 1, 2)
        layout.addWidget(self.logy, 13, 1, 1, 2)
        layout.addWidget(self.btnrefresh, 14, 1)
        toplayout = QtWidgets.QVBoxLayout()
        toplayout.addLayout(layout)
        toplayout.addStretch()
        self.setLayout(toplayout)

        self.itp.currentIndexChanged.connect(lambda i, x=self.itp: self.cmbchange(x))
        self.tur.currentIndexChanged.connect(lambda i, x=self.tur: self.cmbchange(x))
        self.gbf.currentIndexChanged.connect(lambda i, x=self.gbf: self.cmbchange(x))
        self.procbias.currentIndexChanged.connect(lambda i, x=self.procbias: self.cmbchange(x))
        self.testbias.currentIndexChanged.connect(lambda i, x=self.testbias: self.cmbchange(x))

    def cmbchange(self, boxchanged):
        ''' Combobox changed. Ensure only one sweep/step is selected,
            and enable constant fields
        '''
        # Check the box that was just changed first!
        boxes = [self.itp, self.tur, self.gbf, self.procbias, self.testbias]
        consts = [self.itpval, self.turval, self.gbfval, self.procbiasval, self.testbiasval]
        boxidx = boxes.index(boxchanged)
        boxes.pop(boxidx)
        c = consts.pop(boxidx)
        boxes.insert(0, boxchanged)
        consts.insert(0, c)

        havex = False
        havez = False
        for box, const in zip(boxes, consts):
            if 'Sweep' in box.currentText():
                if havex:
                    box.setCurrentIndex(2)  # Already have a sweep, set this to const
                const.setVisible(havex)
                havex = True
            elif 'Step' in box.currentText():
                if havez:
                    box.setCurrentIndex(2)
                const.setVisible(havez)
                havez = True
            elif 'Constant' not in box.currentText():
                const.setVisible(False)
            else:
                const.setVisible(True)

    def get_sweepvals(self):
        # Convert ALL variables into lists the same length.
        # Some will have all identical values
        xvals = np.linspace(float(self.xstart.text()), float(self.xstop.text()), num=int(self.xpts.text()))
        try:
            zvals = np.array([float(z) for z in self.zvals.text().split(',')])
        except ValueError:
            QtWidgets.QMessageBox.warning(self, 'Risk Sweep', 'Step values must be entered as comma-separated list.')
            return None

        # Defaults if not being swept
        sig0 = None
        itpval = float(self.itpval.text()) / 100  # Percent
        if 'SL' in self.itpvssigma.currentText():
            sig0 = float(self.itpval.text())
        turval = float(self.turval.text())
        pbias = float(self.procbiasval.text()) / 100
        tbias = float(self.testbiasval.text()) / 100
        if self.gbf.currentText() == 'Constant':
            gbfval = float(self.gbfval.text())
        else:
            gbfval = self.gbf.currentText().lower()
            gbfval = 'test' if 'test' in gbfval else gbfval

        if 'Step' in self.itp.currentText() and 'In-' in self.itpvssigma.currentText():
            zvar = 'itp'
        elif 'Step' in self.itp.currentText():
            zvar = 'sig0'
        elif 'Step' in self.tur.currentText():
            zvar = 'tur'
        elif 'Step' in self.gbf.currentText():
            zvar = 'gbf'
        elif 'Step' in self.procbias.currentText():
            zvar = 'pbias'
        elif 'Step' in self.testbias.currentText():
            zvar = 'tbias'
        else:
            zvar = 'none'
            zvals = [None]  # Need one item to loop

        if 'Sweep' in self.itp.currentText() and 'In-' in self.itpvssigma.currentText():
            xvar = 'itp'
        elif 'Sweep' in self.itp.currentText():
            xvar = 'sig0'
        elif 'Sweep' in self.tur.currentText():
            xvar = 'tur'
        elif 'Sweep' in self.gbf.currentText():
            xvar = 'gbf'
        elif 'Sweep' in self.procbias.currentText():
            xvar = 'pbias'
        elif 'Sweep' in self.testbias.currentText():
            xvar = 'tbias'
        else:
            QtWidgets.QMessageBox.warning(self, 'Risk Sweep', 'Please select a variable to sweep.')
            return None

        # Convert percent to decimal 0-1
        if xvar in ['itp', 'tbias', 'pbias']:
            xvals = xvals / 100
        if zvar in ['itp', 'tbias', 'pbias']:
            zvals = zvals / 100

        threed = self.plot3d.isChecked()
        y = self.plottype.currentText()
        logy = self.logy.isChecked()
        SweepSetup = namedtuple('SweepSetup', ['x', 'z', 'xvals', 'zvals', 'itp', 'tur', 'gbf', 'sig0',
                                               'pbias', 'tbias', 'threed', 'y', 'logy'])
        return SweepSetup(xvar, zvar, xvals, zvals, itpval, turval, gbfval, sig0, pbias, tbias, threed, y, logy)


class RiskSweeper(QtWidgets.QDialog):
    ''' Main dialog for risk sweep calculations '''
    def __init__(self, parent=None):
        super().__init__(parent=parent)
        gui_common.centerWindow(self, 800, 600)
        self.setWindowTitle('Suncal - Risk Curves Calculator')
        self.setup = RiskSweepSetup()
        self.fig = Figure()
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setStyleSheet("background-color:transparent;")
        self.toolbar = NavigationToolbar(self.canvas, self, coordinates=True)
        self.report = widgets.MarkdownTextEdit()

        outlayout = QtWidgets.QVBoxLayout()
        outlayout.addWidget(self.canvas)
        outlayout.addWidget(self.toolbar)
        outlayout.addWidget(self.report)
        self.outputwidget = QtWidgets.QWidget()
        self.outputwidget.setLayout(outlayout)
        self.splitter = QtWidgets.QSplitter()
        self.splitter.addWidget(self.setup)
        self.splitter.addWidget(self.outputwidget)
        layout = QtWidgets.QHBoxLayout()
        layout.addWidget(self.splitter)
        self.setLayout(layout)

        self.setup.btnrefresh.clicked.connect(self.replot_sweep)
        self.replot_sweep()
        self.canvas.draw_idle()

    def replot_sweep(self):
        ''' Plot generic PFA(R) sweep '''
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.CursorShape.WaitCursor)
        setup = self.setup.get_sweepvals()
        if setup is None:
            return  # No sweep variable
        rpt = risk_sweeper(self.fig,
                           xvar=setup.x,
                           zvar=setup.z,
                           xvals=setup.xvals,
                           zvals=setup.zvals,
                           yvar=setup.y,
                           threed=setup.threed,
                           logy=setup.logy,
                           gbmode=setup.gbf,
                           sig0=setup.sig0,
                           pbias=setup.pbias,
                           tbias=setup.tbias)
        self.fig.tight_layout()
        self.canvas.draw_idle()
        self.report.setReport(rpt)
        QtWidgets.QApplication.restoreOverrideCursor()


if __name__ == '__main__':
    # run using: "python -m suncal.gui.tool_riskcurves"
    app = QtWidgets.QApplication(sys.argv)
    main = RiskSweeper()
    main.show()
    app.exec()
