''' Widgets for statistics settings '''
from PyQt6 import QtWidgets, QtCore, QtGui
import numpy as np

from ...common import distributions, ttable
from .. import gui_styles
from .. import gui_common
from ..gui_settings import gui_settings
from .table import ReadOnlyTableItem


class ExpandedConfidenceWidget(QtWidgets.QWidget):
    ''' Widget for setting expanded uncertainty confidence and type '''
    changed = QtCore.pyqtSignal()

    def __init__(self, showshortest=True):
        super().__init__()
        self.showshortest = showshortest
        self.confvalue = QtWidgets.QDoubleSpinBox()
        self.confvalue.setDecimals(3)
        self.confvalue.setSingleStep(1)
        self.confvalue.setRange(1, 99.999)
        self.confvalue.setValue(95)
        self.kvalue = QtWidgets.QDoubleSpinBox()
        self.kvalue.setDecimals(2)
        self.kvalue.setSingleStep(.1)
        self.kvalue.setRange(.1, 20)
        self.kvalue.setValue(2)
        self.cmbMode = QtWidgets.QComboBox()
        self.cmbMode.addItems(['Level of Confidence %:', 'Coverage Factor (k):'])
        self.cmbShortest = QtWidgets.QComboBox()
        self.cmbShortest.addItems(['Symmetric', 'Shortest'])
        self.lblshortest = QtWidgets.QLabel('Monte-Carlo Interval:')

        valuelayout = QtWidgets.QHBoxLayout()
        valuelayout.addWidget(self.confvalue)
        valuelayout.addWidget(self.kvalue)
        layout = QtWidgets.QGridLayout()
        layout.addWidget(self.cmbMode, 0, 0)
        layout.addLayout(valuelayout, 0, 1)
        if self.showshortest:
            layout.addWidget(self.lblshortest, 1, 0)
            layout.addWidget(self.cmbShortest, 1, 1)
        toplayout = QtWidgets.QVBoxLayout()
        toplayout.addLayout(layout)
        toplayout.addStretch()
        self.setLayout(toplayout)

        self.confvalue.valueChanged.connect(self.changed)
        self.kvalue.valueChanged.connect(self.changed)
        self.cmbShortest.currentIndexChanged.connect(self.changed)
        self.cmbMode.currentIndexChanged.connect(self.change_mode)
        self.cmbShortest.setVisible(showshortest)
        self.kvalue.setVisible(False)

    def change_mode(self):
        ''' Change from confidence to k-factor '''
        if self.cmbMode.currentIndex() == 0:  # Conf
            conf = ttable.confidence(self.kvalue.value(), 1E9) * 100
            self.confvalue.setValue(conf)
            self.confvalue.setVisible(True)
            self.kvalue.setVisible(False)
            self.cmbShortest.setVisible(self.showshortest)
            self.lblshortest.setVisible(self.showshortest)
        else:
            k = ttable.k_factor(self.confvalue.value()/100, 1E9)
            self.kvalue.setValue(k)
            self.confvalue.setVisible(False)
            self.kvalue.setVisible(True)
            self.cmbShortest.setVisible(False)
            self.lblshortest.setVisible(False)
        self.changed.emit()

    def get_params(self):
        ''' Get dictionary of {k: } or {conf: } for passing into
            other functions
        '''
        if self.cmbMode.currentIndex() == 0:  # Confidence
            params = {'conf': self.confvalue.value()/100}
        else:
            params = {'k': self.kvalue.value()}
        return params

    def get_shortest(self):
        return self.cmbShortest.currentText() == 'Shortest'


class DistributionEditTable(QtWidgets.QTableWidget):
    ''' Table for editing parameters of an uncertainty distribution

        Args:
            initargs (dict): Initial arguments for distribution
            locslider (bool): Show slider for median location of distribution
    '''
    changed = QtCore.pyqtSignal()

    ROLE_HISTDATA = QtCore.Qt.ItemDataRole.UserRole + 2    # Original, user-entered data

    def __init__(self, initargs=None, locslider=False):
        super().__init__()
        self.showlocslider = locslider
        self.range = (-2.0, 2.0)
        self.setMinimumWidth(200)
        self.setMaximumHeight(200)
        self.verticalHeader().setVisible(False)
        self.set_disttype(initargs)
        self.set_locrange(-4, 4)
        self.valuechanged()
        self.cellChanged.connect(self.valuechanged)

    def clear(self):
        ''' Clear and reset the table '''
        super().clear()
        self.setColumnCount(2)
        self.setHorizontalHeaderLabels(['Parameter', 'Value', ''])
        self.resizeColumnsToContents()
        self.setColumnWidth(0, 150)
        self.setColumnWidth(1, 100)

    def sliderchange(self):
        ''' Slider has changed, update loc in table '''
        rng = self.range[1] - self.range[0]
        val = self.locslide.value() / self.locslide.maximum() * rng + self.range[0]
        self.item(1, 1).setText(str(val))

    def set_locrange(self, low, high):
        ''' Set range for loc slider '''
        self.range = (low, high)

    def set_disttype(self, initargs=None):
        ''' Change distribution type, fill in required params '''
        if initargs is None:
            initargs = {}
        distname = initargs.pop('name', initargs.pop('dist', 'normal'))
        initargs.setdefault('median', initargs.get('mean', 0))
        bias = initargs['median'] - initargs.get('expected', initargs['median'])
        initargs.setdefault('bias', bias)

        # Want control to enter median value, not loc. Shift appropriately.
        stats_dist = distributions.get_distribution(distname, **initargs)
        argnames = stats_dist.argnames

        if 'loc' in argnames:
            argnames.remove('loc')
        if stats_dist.showshift:
            argnames = ['shift'] + argnames
            initargs.setdefault('shift', stats_dist.distargs.get('loc', 0))
        else:
            argnames = ['median'] + argnames

            try:
                median = stats_dist.median()
            except TypeError:
                median = 0
            else:
                median = median if np.isfinite(median) else 0
            initargs['median'] = median

        # Strip any units that may trickle in
        for k, v in initargs.items():
            if hasattr(v, 'magnitude'):
                initargs[k] = v.magnitude

        with gui_common.BlockedSignals(self):
            self.clear()
            dists = gui_settings.distributions
            self.cmbdist = QtWidgets.QComboBox()
            self.cmbdist.addItems(dists)
            if distname not in dists:
                self.cmbdist.addItem(distname)
            self.cmbdist.setCurrentIndex(self.cmbdist.findText(distname))
            self.setRowCount(len(argnames) + 1 + self.showlocslider)  # header + argnames
            self.setItem(0, 0, ReadOnlyTableItem('Distribution'))
            self.setCellWidget(0, 1, self.cmbdist)

            if distname == 'histogram':
                self.setRowCount(1)
                if self.showlocslider:
                    self.setRowCount(3)
                    self.setItem(1, 0, ReadOnlyTableItem('measurement'))
                    self.setItem(1, 1, QtWidgets.QTableWidgetItem(str(initargs.get('median', 0))))
                    self.setItem(2, 0, ReadOnlyTableItem('bias'))
                    self.setItem(2, 1, QtWidgets.QTableWidgetItem(str(initargs.get('bias', 0))))
                self.item(0, 0).setData(self.ROLE_HISTDATA, stats_dist.distargs)

            else:
                for row, arg in enumerate(argnames):
                    self.setItem(row+1, 0, ReadOnlyTableItem(arg))
                    self.setItem(row+1, 1, QtWidgets.QTableWidgetItem(str(initargs.get(arg, '1' if row > 0 else '0'))))
                if self.showlocslider:
                    self.setItem(row+2, 0, ReadOnlyTableItem('bias'))
                    self.setItem(row+2, 1, QtWidgets.QTableWidgetItem(str(initargs.get('bias', 0))))

            if self.showlocslider:
                self.locslidewidget = QtWidgets.QWidget()
                self.locslide = QtWidgets.QSlider(orientation=QtCore.Qt.Orientation.Horizontal)
                self.locslide.setRange(0, 200)  # Sliders always use ints.
                self.locslide.setValue(100)
                layout = QtWidgets.QHBoxLayout()
                layout.addWidget(QtWidgets.QLabel('measurement'))
                layout.addWidget(self.locslide)
                self.locslidewidget.setLayout(layout)
                self.setCellWidget(1, 0, self.locslidewidget)
                self.locslide.valueChanged.connect(self.sliderchange)
                self.setItem(1, 0, ReadOnlyTableItem(''))

            for row in range(self.rowCount()):
                self.setRowHeight(row, 40)

            self.cmbdist.currentIndexChanged.connect(self.distchanged)

    def distchanged(self):
        ''' Distribution combobox change '''
        self.set_disttype({'dist': self.cmbdist.currentText()})
        self.valuechanged()

    def valuechanged(self):
        ''' Table value has changed, update stats distribution. '''
        with gui_common.BlockedSignals(self):
            argvals = []
            distname = self.cmbdist.currentText()

            for r in range(self.rowCount()-1):
                try:
                    argvals.append(float(self.item(r+1, 1).text()))
                except ValueError:
                    argvals.append(None)
                    self.item(r+1, 1).setBackground(gui_styles.color.invalid)
                    self.clearSelection()
                else:
                    self.item(r+1, 1).setBackground(gui_styles.color.transparent)

            argnames = [self.item(r+1, 0).text() for r in range(self.rowCount()-1)]
            args = dict(zip(argnames, argvals))
            if '' in args:
                args['median'] = args.pop('')  # 'median' label is hidden when slider is used

            changed = False
            if distname == 'histogram':
                distargs = self.item(0, 0).data(self.ROLE_HISTDATA)
                self.statsdist = distributions.get_distribution(distname, **distargs)
                self.distbias = args.pop('bias', 0)
                if 'median' in args or '' in args:
                    median = args.pop('median', args.pop('', 0))
                    self.statsdist.set_median(median - self.distbias)
                    changed = True

            elif None not in argvals:
                self.distbias = args.pop('bias', 0)
                try:
                    self.statsdist = distributions.get_distribution(distname, **args)
                    if self.statsdist.showshift:
                        self.statsdist.set_shift(args.pop('shift', args.pop('median', 0)))
                    else:
                        self.statsdist.set_median(args.pop('median', 0) - self.distbias)
                except ZeroDivisionError:
                    self.statsdist = None
                else:
                    changed = True

        if changed:
            self.changed.emit()

    def contextMenuEvent(self, event):
        menu = QtWidgets.QMenu(self)
        actHelp = QtGui.QAction('Distribution Help...', self)
        menu.addAction(actHelp)
        actHelp.triggered.connect(self.helppopup)
        menu.popup(event.globalPos())

    def helppopup(self):
        dlg = PopupHelp(self.statsdist.helpstr())
        dlg.exec()


class PopupHelp(QtWidgets.QDialog):
    ''' Show a floating dialog window with a text message '''
    def __init__(self, text):
        super().__init__()
        gui_common.centerWindow(self, 600, 400)
        self.setWindowTitle('Distribution Info')
        self.setModal(False)
        self.text = QtWidgets.QTextEdit()
        self.text.setReadOnly(True)
        font = QtGui.QFont('Courier')
        font.setStyleHint(QtGui.QFont.StyleHint.TypeWriter)
        self.text.setCurrentFont(font)
        self.text.setText(text)
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.text)
        self.setLayout(layout)
