''' Widget for calculating and comparing t distributions '''

import numpy as np
from scipy import stats

from PyQt5 import QtWidgets, QtGui
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from . import gui_common
from . import gui_widgets
from .. import ttable


class TTableDialog(QtWidgets.QDialog):
    ''' Dialog for calculating t-distribution. Input any two of k, confidence, and degf,
        and solve for the third.
    '''
    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.setWindowTitle('T-table')
        font = self.font()
        font.setPointSize(10)
        self.setFont(font)
        self.cmbSolveFor = QtWidgets.QComboBox()
        self.cmbSolveFor.addItems(['Coverage Factor', 'Confidence', 'Degrees of Freedom'])
        self.degf = gui_widgets.LineEditLabelWidget('Degrees of Freedom', '30')
        self.degf.setValidator(QtGui.QIntValidator(1, 10000000))
        self.conf = gui_widgets.LineEditLabelWidget('Confidence Percent', '95.45')
        self.conf.setValidator(QtGui.QDoubleValidator(0, 100, 6))
        self.k = gui_widgets.LineEditLabelWidget('Coverage Factor', '2.00')
        self.k.setValidator(QtGui.QDoubleValidator(0, 1000, 6))
        self.k.setVisible(False)
        self.output = QtWidgets.QLabel()
        self.fig = Figure()
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setMaximumWidth(500)
        self.canvas.setMaximumHeight(500)

        hlayout = QtWidgets.QHBoxLayout()
        hlayout.addWidget(QtWidgets.QLabel('Solve for:'))
        hlayout.addWidget(self.cmbSolveFor)
        hlayout.addStretch()
        layout = QtWidgets.QVBoxLayout()
        layout.addLayout(hlayout)
        layout.addWidget(self.degf)
        layout.addWidget(self.conf)
        layout.addWidget(self.k)
        layout.addSpacing(5)
        layout.addStretch()
        layout.addWidget(self.output)
        layout.addWidget(self.canvas)
        self.setLayout(layout)

        self.calculate()
        self.degf.editingFinished.connect(self.calculate)
        self.conf.editingFinished.connect(self.calculate)
        self.k.editingFinished.connect(self.calculate)
        self.cmbSolveFor.currentIndexChanged.connect(self.change_solvefor)

    def calculate(self):
        ''' Run the calculation and plot '''
        if self.cmbSolveFor.currentText() == 'Coverage Factor':
            degf = float(self.degf.text())
            k = ttable.t_factor(float(self.conf.text())/100, float(self.degf.text()))
            self.output.setText('<font size=4>k: {:.4f}</font>'.format(k))
        elif self.cmbSolveFor.currentText() == 'Confidence':
            degf = float(self.degf.text())
            k = float(self.k.text())
            conf = ttable.confidence(k, degf) * 100
            self.output.setText('<font size=4>Confidence: {:.3f}%</font>'.format(conf))
        else:
            k = float(self.k.text())
            degf = ttable.degf(k, float(self.conf.text())/100)
            if degf > 1E6:
                self.output.setText('<font size=4>Degrees of Freedom: &infin;</font>')
            else:
                self.output.setText('<font size=4>Degrees of Freedom: {:.2f}</font>'.format(degf))

        self.fig.clf()
        ax = self.fig.add_subplot(1, 1, 1)
        xx = np.linspace(-4, 4, num=200)
        ax.plot(xx, stats.norm.pdf(xx), label='Normal')
        ax.plot(xx, stats.t.pdf(xx, df=degf), label='t (df={})'.format('{:.2f}'.format(degf) if degf < 1E6 else r'$\infty$'))
        ax.axvline(k, ls=':', color='black')
        ax.axvline(-k, ls=':', color='black')
        ax.set_ylabel('Probability Density Function')
        ax.legend(loc='upper right')
        self.fig.tight_layout()
        self.canvas.draw_idle()

    def change_solvefor(self):
        ''' Solve-for option changed. Show/hide inputs '''
        solvefor = self.cmbSolveFor.currentText()
        self.k.setVisible(solvefor != 'Coverage Factor')
        self.conf.setVisible(solvefor != 'Confidence')
        self.degf.setVisible(solvefor != 'Degrees of Freedom')
        self.calculate()
