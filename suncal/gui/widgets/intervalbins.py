''' Dialog for binning S2 Interval Data '''
from PyQt6 import QtWidgets, QtCore
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from .mkdown import MarkdownTextEdit
from ...intervals import datearray
from ...common import report


class BinData(QtWidgets.QDialog):
    ''' Dialog for binning historical pass/fail data for method S2 '''
    def __init__(self, component, parent=None):
        super().__init__(parent)
        font = self.font()
        font.setPointSize(10)
        self.setFont(font)
        self.component = component
        self.selectedbinidx = None
        self.press = None
        self.lastclickedbin = None
        self.binlines = []
        self.binlefts = []

        self.setWindowTitle('Select data bins')
        self.fig = Figure()
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setStyleSheet("background-color:transparent;")
        self.txtOutput = MarkdownTextEdit()
        self.nbins = QtWidgets.QSpinBox()
        self.nbins.setRange(2, 100)
        self.binwidth = QtWidgets.QSpinBox()
        self.binwidth.setRange(1, 9999)
        self.dlgbutton = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Ok |
                                                    QtWidgets.QDialogButtonBox.StandardButton.Cancel)

        layout = QtWidgets.QVBoxLayout()
        tlayout = QtWidgets.QHBoxLayout()
        tlayout.addWidget(self.canvas, stretch=2)
        tlayout.addWidget(self.txtOutput, stretch=1)
        layout.addLayout(tlayout)
        blayout = QtWidgets.QHBoxLayout()
        blayout.addWidget(QtWidgets.QLabel('Click and drag rectangles to adjust bin positions'))
        blayout.addSpacing(25)
        blayout.addWidget(QtWidgets.QLabel('Number of Bins'))
        blayout.addWidget(self.nbins)
        blayout.addSpacing(50)
        blayout.addWidget(QtWidgets.QLabel('Bin Width'))
        blayout.addWidget(self.binwidth)
        blayout.addStretch()
        layout.addLayout(blayout)
        layout.addWidget(self.dlgbutton)
        self.setLayout(layout)

        self.init_bins()
        self.init_plot()
        self.draw_bins()
        self.report_reliability()

        self.dlgbutton.rejected.connect(self.reject)
        self.dlgbutton.accepted.connect(self.accept)
        self.binwidth.valueChanged.connect(self.draw_bins)
        self.binwidth.valueChanged.connect(self.report_reliability)
        self.nbins.valueChanged.connect(self.set_nbins)

        self.canvas.mpl_connect('button_press_event', self.on_press)
        self.canvas.mpl_connect('button_release_event', self.on_release)
        self.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.canvas.mpl_connect('key_press_event', self.on_keypress)
        self.canvas.setFocusPolicy(QtCore.Qt.FocusPolicy.ClickFocus)

    def init_bins(self):
        ''' Initialize bins based on Interval Calc class '''
        if self.component.binlefts is None:
            self.nbins.setValue(9)
            self.reset_bins()
        else:
            self.binlefts = self.component.binlefts
            self.nbins.setValue(len(self.binlefts))
            self.binwidth.setValue(self.component.binwidth)

    def reset_bins(self, nbins=9):
        ''' Reset bins to default '''
        intmax = 0
        for val in self.component.assets.values():
            if 'startdates' not in val or val['startdates'] is None or len(val['startdates']) == 0:
                ti = np.diff(datearray(val['enddates']))
            else:
                ti = datearray(val['enddates']) - datearray(val['startdates'])
            if len(ti) > 0:
                intmax = max(intmax, ti.max())
        if intmax == 0:
            # No data yet
            self.binlefts = np.array([0, 180])
            self.binwidth.setValue(180)
        else:
            self.binlefts = np.linspace(0, intmax+1, num=nbins+1)[:-1]
            self.binwidth.setValue(int(self.binlefts[1] - self.binlefts[0]))
        self.report_reliability()

    def init_plot(self):
        self.fig.clf()
        self.ax = self.fig.add_subplot(1, 1, 1)

        i = 0
        for asset in self.component.assets.keys():
            pf, ti = self.component.passfails(asset)
            pf = np.asarray(pf)
            ti = np.asarray(ti)
            if len(ti) > 0:
                passes = np.array(ti, dtype=float)
                passes[pf == 0.] = np.nan
                fails = np.array(ti, dtype=float)
                fails[pf == 1.] = np.nan
                self.ax.plot(passes, np.ones_like(pf)*(i+1), marker='o', color='C0', ls='')
                self.ax.plot(fails, np.ones_like(pf)*(i+1), marker='x', color='C1', ls='')
                i += 1
        self.ax.set_ylabel('Asset')
        self.ax.set_xlabel('Time Since Calibration')
        self.canvas.draw_idle()

    def set_nbins(self):
        ''' Change the number of bins '''
        self.reset_bins(self.nbins.value())
        self.draw_bins()
        self.report_reliability()

    def draw_bins(self):
        ''' Add bin regions to plot '''
        for b in self.binlines:
            b.remove()
            del b
        self.binlines = []
        for i, left in enumerate(self.binlefts):
            color = 'C3' if i % 2 else 'C4'
            self.binlines.append(self.ax.axvspan(
                left, left+self.binwidth.value(), alpha=.3, ls='-', facecolor=color, edgecolor='black'))
        self.canvas.draw_idle()

    def on_press(self, event):
        ''' Mouse-click. See if click was in a bin rectangle '''
        if event.inaxes:
            for idx, b in enumerate(self.binlines):
                if b.contains(event)[0]:
                    self.selectedbinidx = idx
                    self.press = b.xy, (event.xdata, event.ydata)
                    self.lastclickedbin = self.selectedbinidx
                    self.canvas.setFocus()
                    break
            else:
                self.selectedbinidx = None

    def on_release(self, event):
        ''' Mouse was released. Update rectangle and reliability table '''
        self.selectedbinidx = None

        # Left edge of each rectangle
        self.binlefts = [b.xy[:, 0].min() for b in self.binlines]
        self.report_reliability()

    def on_keypress(self, event):
        ''' Handle key press event '''
        if self.lastclickedbin is not None and event.key in ['left', 'right']:
            increment = 1 if event.key == 'right' else -1

            # Don't hit another bin
            newx = self.binlefts[self.lastclickedbin] + increment
            if newx < 0:
                return
            if self.lastclickedbin < len(self.binlefts)-2:
                if newx + self.binwidth.value() > self.binlefts[self.lastclickedbin+1]:
                    return
            if self.lastclickedbin > 0:
                if newx < self.binlefts[self.lastclickedbin-1] + self.binwidth.value():
                    return

            self.binlefts[self.lastclickedbin] = newx
            rect = self.binlines[self.lastclickedbin]
            poly = rect.xy
            poly[:, 0] += increment
            rect.set_xy(poly)
            self.canvas.draw()
            self.report_reliability()

    def report_reliability(self):
        ''' Update text area with reliability table '''
        self.component.binlefts = self.binlefts
        self.component.binwidth = self.binwidth.value()
        params = self.component.params
        ti, ti0, ri, ni = params.ti, params.ti0, params.ri, params.ni
        hdr = ['Bin Range', 'Reliability', 'Measurements']
        rows = []
        for t0, t, r, n in zip(ti0, ti, ri, ni):
            rows.append([f'{t0:.0f} - {t:.0f}', f'{r:.3f}', f'{n:.0f}'])
        rpt = report.Report()
        rpt.table(rows, hdr)
        self.txtOutput.setReport(rpt)

    def on_motion(self, event):
        ''' Mouse drag of bin. Limit so bins can't overlap '''
        if self.selectedbinidx is not None and event.xdata is not None:
            rect = self.binlines[self.selectedbinidx]
            poly, (xpress, _) = self.press
            dx = event.xdata - xpress
            polytest = poly.copy()
            polytest[:, 0] += dx
            left = polytest[:, 0].min()
            right = polytest[:, 0].max()
            leftlim = (-np.inf if self.selectedbinidx == 0 else
                       self.binlines[self.selectedbinidx-1].xy[:, 0].max())
            rightlim = (np.inf if self.selectedbinidx == len(self.binlines)-1 else
                        self.binlines[self.selectedbinidx+1].xy[:, 0].min())
            if left < leftlim or right > rightlim:
                if left < leftlim:
                    dx += (leftlim-left)
                elif right > rightlim:
                    dx -= (right-rightlim)
                polytest = poly.copy()
                polytest[:, 0] += dx
            rect.set_xy(polytest)
            self.canvas.draw()

    def getbins(self):
        ''' Return list of left-edges, binwidth '''
        return self.binlefts, self.binwidth.value()
