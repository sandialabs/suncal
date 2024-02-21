''' Page for changing the settings. '''

import os
import logging
from scipy import stats
from PyQt6 import QtCore, QtWidgets, QtGui
import matplotlib as mpl

from ..common import report
from . import gui_styles
from .gui_settings import gui_settings
from .gui_common import BlockedSignals
from .widgets import ColorMapDialog, QHLine


# List all distributions in scipy.stats
DISTS = [d for d in dir(stats) if isinstance(getattr(stats, d), (stats.rv_continuous, stats.rv_discrete))]
DISTS = ['normal', 'triangular', 'curvtrap', 'resolution'] + DISTS


class ColorButton(QtWidgets.QPushButton):
    ''' Widget to show a button and choose color '''
    colorChanged = QtCore.pyqtSignal(object)

    def __init__(self, color=None, parent=None):
        super().__init__(parent)
        self.color = color
        self.setMaximumWidth(32)
        self.setColor(color)
        self.pressed.connect(self.pickColor)

    def pickColor(self):
        ''' Button pressed '''
        dlg = QtWidgets.QColorDialog()
        if self.color:
            dlg.setCurrentColor(QtGui.QColor(self.color))
        if dlg.exec():
            self.setColor(dlg.currentColor().name())

    def setColor(self, color):
        ''' Set the color of the button '''
        if color != self.color:
            self.color = color
            self.colorChanged.emit(self.color)
        if self.color:
            self.setStyleSheet(f'background-color: {self.color}')
        else:
            self.setStyleSheet('')


class ColorMapButton(QtWidgets.QPushButton):
    ''' Widget to select a colormap '''
    cmapChanged = QtCore.pyqtSignal(str)

    def __init__(self, cmap=None, parent=None):
        super().__init__(parent)
        self.cmap = cmap
        self.pressed.connect(self.pickCmap)

    def pickCmap(self):
        ''' Button pressed '''
        dlg = ColorMapDialog(self, self.cmap)
        if dlg.exec() and dlg.selectedcmap:
            self.setCmap(dlg.selectedcmap)

    def setCmap(self, cmap):
        ''' Set colormap '''
        self.cmap = cmap
        self.setText(cmap)
        self.cmapChanged.emit(self.cmap)


class PgGeneral(QtWidgets.QWidget):
    ''' Page for General Settings '''
    def __init__(self, parent=None):
        super().__init__(parent)
        self.sigfigs = QtWidgets.QSpinBox()
        self.sigfigs.setMinimum(1)
        self.sigfigs.setMaximum(20)
        self.nformat = QtWidgets.QComboBox()
        self.nformat.addItems(['Auto', 'Decimal', 'Scientific', 'Engineering', 'SI'])

        flayout = QtWidgets.QFormLayout()
        flayout.addRow('Significant Figures', self.sigfigs)
        flayout.addRow('Number format', self.nformat)
        flayout.setHorizontalSpacing(20)
        self.setLayout(flayout)


class PgDistribution(QtWidgets.QWidget):
    ''' Page for distribution list '''
    def __init__(self, parent=None):
        super().__init__(parent)
        label = QtWidgets.QLabel('Enable these distributions:')
        self.dlist = QtWidgets.QListWidget()
        for name in DISTS:
            item = QtWidgets.QListWidgetItem(name)
            item.setFlags(item.flags() | QtCore.Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(QtCore.Qt.CheckState.Unchecked)
            self.dlist.addItem(item)

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(label)
        layout.addWidget(self.dlist)
        self.setLayout(layout)


class PgUnits(QtWidgets.QWidget):
    ''' Page for defining custom measurement units '''
    def __init__(self, parent=None):
        super().__init__(parent)
        label = QtWidgets.QLabel('Enter custom unit names and abbreviations\nas equations relating to other units. '
                                 'Example:\n\n    banana_dose = 78*nanosieverts = bn\n\n'
                                 'See Pint documentation for details.')
        self.unitdefs = QtWidgets.QTextEdit()
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(label)
        layout.addWidget(self.unitdefs)
        self.setLayout(layout)


class PgMonteCarlo(QtWidgets.QWidget):
    ''' Page for Monte-Carlo settings '''
    def __init__(self, parent=None):
        super().__init__(parent)
        self.txtSamples = QtWidgets.QLineEdit('1000000')
        self.txtSeed = QtWidgets.QLineEdit('None')

        flayout = QtWidgets.QFormLayout()
        flayout.addRow('Default Samples', self.txtSamples)
        flayout.addRow('Random Seed', self.txtSeed)
        layout = QtWidgets.QVBoxLayout()
        layout.addLayout(flayout)
        layout.addStretch()
        self.setLayout(layout)


class PgStyle(QtWidgets.QWidget):
    ''' Page for report settings '''
    def __init__(self, parent=None):
        super().__init__(parent)
        self.colorscustomized = False
        self.customrc = {}
        self.cmbStyle = QtWidgets.QComboBox()
        self.cmbStyle.addItems(['Suncal'] + mpl.style.available)
        self.colors = [ColorButton('black'),      # C0
                       ColorButton('black'),      # C1
                       ColorButton('black'),      # C2
                       ColorButton('black'),      # C3
                       ColorButton('black')]      # C4

        self.cmpGum = ColorMapButton('viridis')
        self.cmpScat = ColorMapButton('Purples')
        self.cmbStyle.currentIndexChanged.connect(self.stylechange)
        for clr in self.colors:
            clr.pressed.connect(self.colorchange)
        self.btnCustom = QtWidgets.QPushButton('Customize Style...')
        self.btnCustom.pressed.connect(self.customize)

        blayout = QtWidgets.QHBoxLayout()
        blayout.addWidget(QtWidgets.QLabel('Base Style:'))
        blayout.addWidget(self.cmbStyle)
        flayout = QtWidgets.QFormLayout()
        for i, colorwidget in enumerate(self.colors):
            flayout.addRow(f'Color {i+1}', colorwidget)
        f2layout = QtWidgets.QFormLayout()
        f2layout.addRow('Contour Colormap', self.cmpGum)
        f2layout.addRow('Discrete Colormap', self.cmpScat)
        flayout.setHorizontalSpacing(20)
        f2layout.setHorizontalSpacing(20)
        mainlayout = QtWidgets.QVBoxLayout()
        mainlayout.addLayout(blayout)
        mainlayout.addLayout(flayout)
        mainlayout.addLayout(f2layout)
        mainlayout.addWidget(self.btnCustom)
        self.setLayout(mainlayout)

    def stylechange(self, index):
        with BlockedSignals(self):
            stylename = self.cmbStyle.currentText()
            if stylename == 'Suncal' and gui_styles.isdark():
                stylename = 'suncal.common.style.suncal_dark'
            elif stylename == 'Suncal':
                stylename = 'suncal.common.style.suncal_light'

            with mpl.style.context(stylename):
                colorcycle = mpl.rcParams['axes.prop_cycle'].by_key()['color']

            for colorwidget, color in zip(self.colors, colorcycle):
                try:
                    colorwidget.setColor(color)
                except IndexError:
                    # Color not defined in this style
                    colorwidget.setColor('black')
        self.colorscustomized = False

    def colorchange(self):
        self.colorscustomized = True

    def get_colorcycle(self):
        return [c.color for c in self.colors]

    def customize(self):
        lines = []
        for k, v in self.customrc.items():
            lines.append(k + ': ' + v)
        text = '\n'.join(lines)

        label = '''Enter parameters compatible with Matplotlib rcParams file.\nFor example, "lines.linewidth: 2.0"'''
        dlg = DlgMultiLineEdit('Enter Parameters', label=label, text=text)
        ok = dlg.exec()
        if ok:
            lines = dlg.getText().splitlines()
            self.customrc = {}
            for line in lines:
                key, val = line.split(':', maxsplit=1)
                key = key.strip()
                val = val.strip()
                if key in list(mpl.rcParams.keys()):
                    self.customrc[key] = val
                else:
                    logging.warning('Unknown rcParam %s', key)


class PgReportOpts(QtWidgets.QWidget):
    ''' Page for default report format options '''
    def __init__(self, parent=None):
        super().__init__(parent)

        self.cmbFormat = QtWidgets.QComboBox()
        self.cmbFormat.addItems(['HTML', 'Markdown', 'PDF', 'Open Office ODT', 'Word DOCX'])
        self.cmbMath = QtWidgets.QComboBox()
        self.cmbMath.addItems(['Mathjax', 'Matplotlib'])
        self.cmbImage = QtWidgets.QComboBox()
        self.cmbImage.addItems(['SVG', 'PNG'])
        self.chkUnicode = QtWidgets.QCheckBox('Allow Unicode in Markdown')
        self.pandoc = QtWidgets.QLineEdit()
        self.latex = QtWidgets.QLineEdit()

        glayout = QtWidgets.QFormLayout()
        glayout.addRow('File Format', self.cmbFormat)
        glayout.addRow('Image Format', self.cmbImage)
        glayout.addRow('Math Renderer', self.cmbMath)
        glayout.addRow('', self.chkUnicode)
        playout = QtWidgets.QFormLayout()
        playout.addRow('Pandoc', self.pandoc)
        playout.addRow('PDFLatex', self.latex)

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(QtWidgets.QLabel('Default report format options:'))
        layout.addLayout(glayout)
        layout.addWidget(QHLine())
        layout.addWidget(QtWidgets.QLabel('Paths:'))
        layout.addLayout(playout)
        layout.addStretch()
        self.setLayout(layout)


class DlgMultiLineEdit(QtWidgets.QDialog):
    ''' Dialog for entering a multiline string '''
    def __init__(self, title, label='', text=''):
        super().__init__()
        font = self.font()
        font.setPointSize(10)
        self.setFont(font)
        self.setWindowTitle(title)
        label = QtWidgets.QLabel(label)
        self.text = QtWidgets.QPlainTextEdit()
        self.text.insertPlainText(text)
        btn = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Ok |
                                         QtWidgets.QDialogButtonBox.StandardButton.Cancel)
        btn.accepted.connect(self.accept)
        btn.rejected.connect(self.reject)
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(label)
        layout.addWidget(self.text)
        layout.addWidget(btn)
        self.setLayout(layout)

    def getText(self):
        return self.text.toPlainText()


class PgSettingsDlg(QtWidgets.QDialog):
    ''' Dialog for editing of settings '''
    def __init__(self, parent=None):
        super().__init__(parent)
        font = self.font()
        font.setPointSize(10)
        self.setFont(font)

        self.setWindowTitle('Suncal Settings')

        self.pgGeneral = PgGeneral(self)
        self.pgStyle = PgStyle(self)
        self.pgRptOpts = PgReportOpts(self)
        self.pgDist = PgDistribution(self)
        self.pgUnits = PgUnits(self)
        self.pgMC = PgMonteCarlo(self)
        self.list = QtWidgets.QListWidget()
        self.list.setMaximumWidth(150)
        self.list.addItems(['General', 'Plot Style', 'Report Format', 'Units',
                            'Distributions', 'Monte-Carlo'])
        self.list.currentRowChanged.connect(self.pagechange)
        self.stack = QtWidgets.QStackedWidget()
        self.stack.addWidget(self.pgGeneral)
        self.stack.addWidget(self.pgStyle)
        self.stack.addWidget(self.pgRptOpts)
        self.stack.addWidget(self.pgUnits)
        self.stack.addWidget(self.pgDist)
        self.stack.addWidget(self.pgMC)
        self.button = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Ok |
                                                 QtWidgets.QDialogButtonBox.StandardButton.Cancel |
                                                 QtWidgets.QDialogButtonBox.StandardButton.RestoreDefaults)
        self.button.accepted.connect(self.accept)
        self.button.rejected.connect(self.reject)
        self.button.button(QtWidgets.QDialogButtonBox.StandardButton.RestoreDefaults).clicked.connect(self.defaults)
        layout = QtWidgets.QHBoxLayout()
        layout.addWidget(self.list)
        layout.addWidget(self.stack, stretch=5)
        mlayout = QtWidgets.QVBoxLayout()
        mlayout.addLayout(layout)
        mlayout.addWidget(self.button)
        self.setLayout(mlayout)
        self.loadsettings()

    def loadsettings(self):
        ''' Load settings '''
        with BlockedSignals(self):
            self.pgGeneral.sigfigs.setValue(gui_settings.sigfigs)
            self.pgGeneral.nformat.setCurrentIndex(
                self.pgGeneral.nformat.findText(gui_settings.numformat,
                                                flags=QtCore.Qt.MatchFlag.MatchExactly |
                                                QtCore.Qt.MatchFlag.MatchFixedString))
            self.pgStyle.cmpGum.setCmap(gui_settings.colormap_contour)
            self.pgStyle.cmpScat.setCmap(gui_settings.colormap_scatter)
            self.pgStyle.cmbStyle.setCurrentIndex(self.pgStyle.cmbStyle.findText(gui_settings.plot_style))

            self.pgStyle.customrc = gui_settings.plotparams
            for i, clr in enumerate(self.pgStyle.colors):
                clr.setColor(gui_settings.get_color_from_cycle(i))

            self.pgMC.txtSamples.setText(str(gui_settings.samples))
            self.pgMC.txtSeed.setText(str(gui_settings.randomseed))
            self.pgUnits.unitdefs.setText(gui_settings.unit_defs)

            self.pgRptOpts.cmbFormat.setCurrentIndex(
                ['html', 'md', 'pdf', 'odt', 'docx'].index(gui_settings.rptformat))
            self.pgRptOpts.cmbMath.setCurrentIndex(
                ['mathjax', 'mpl'].index(gui_settings.rpt_mathformat))
            self.pgRptOpts.cmbImage.setCurrentIndex(
                ['svg', 'png'].index(gui_settings.rpt_imageformat))
            self.pgRptOpts.chkUnicode.setChecked(gui_settings.rpt_unicode)

            pandoc = gui_settings.pandoc_path
            if pandoc is None or pandoc == '':
                pandoc = report.pandoc_path  # Get from report module default if not defined in settings
            self.pgRptOpts.pandoc.setText(pandoc if pandoc else '')
            latex = gui_settings.latex_path
            if latex is None or latex == '':
                latex = report.latex_path  # Get from report module default if not defined in settings
            self.pgRptOpts.latex.setText(latex if latex else '')

            dists = gui_settings.distributions
            for i in range(self.pgDist.dlist.count()):
                item = self.pgDist.dlist.item(i)
                if item.text() in dists:
                    item.setCheckState(QtCore.Qt.CheckState.Checked)
                else:
                    item.setCheckState(QtCore.Qt.CheckState.Unchecked)

    def defaults(self):
        ''' Restore all default values '''
        gui_settings.set_defaults()
        self.loadsettings()

    def pagechange(self, index):
        ''' Change the page '''
        self.stack.setCurrentIndex(index)

    def accept(self):
        ''' Save settings and close the dialog. NOTE: NO VALIDATION CHECKING DONE! '''
        # Just write everything every time
        gui_settings.sigfigs = self.pgGeneral.sigfigs.value()
        gui_settings.numformat = self.pgGeneral.nformat.currentText().lower()

        gui_settings.plot_style = self.pgStyle.cmbStyle.currentText()
        if len(self.pgStyle.customrc):
            gui_settings.plotparams = self.pgStyle.customrc

        if self.pgStyle.colorscustomized:
            gui_settings.color_cycle = self.pgStyle.get_colorcycle()
        else:
            gui_settings.clear_color_cycle()

        gui_settings.colormap_contour = self.pgStyle.cmpGum.cmap
        gui_settings.colormap_scatter = self.pgStyle.cmpScat.cmap
        gui_settings.samples = self.pgMC.txtSamples.text()
        gui_settings.randomseed = self.pgMC.txtSeed.text()

        dlist = []
        for i in range(self.pgDist.dlist.count()):
            item = self.pgDist.dlist.item(i)
            if item.checkState() == QtCore.Qt.CheckState.Checked:
                dlist.append(item.text())
        gui_settings.distributions = dlist

        gui_settings.rptformat = ['html', 'md', 'pdf', 'odt', 'docx'][self.pgRptOpts.cmbFormat.currentIndex()]
        gui_settings.rpt_imageformat = ['svg', 'png'][self.pgRptOpts.cmbImage.currentIndex()]
        gui_settings.rpt_mathformat = ['mathjax', 'mpl'][self.pgRptOpts.cmbMath.currentIndex()]
        gui_settings.rpt_unicode = self.pgRptOpts.chkUnicode.isChecked()
        gui_settings.unit_defs = self.pgUnits.unitdefs.toPlainText()

        if self.pgRptOpts.pandoc.text() != report.pandoc_path and os.path.exists(self.pgRptOpts.pandoc.text()):
            # Only save if customized and pandoc is there
            gui_settings.pandoc_path = self.pgRptOpts.pandoc.text()
            report.pandoc_path = self.pgRptOpts.pandoc.text()
        if self.pgRptOpts.latex.text() != report.latex_path and os.path.exists(self.pgRptOpts.latex.text()):
            gui_settings.latex_path = self.pgRptOpts.latex.text()
            report.latex_path = self.pgRptOpts.latex.text()

        gui_styles.color.set_plot_style()
        super().accept()  # Let the dialog finish closing properly


if __name__ == '__main__':
    # Run just the settings dialog
    # run using: "python -m suncal.gui.page_settings"
    import sys
    app = QtWidgets.QApplication(sys.argv)
    main = PgSettingsDlg()
    main.show()
    app.exec()
