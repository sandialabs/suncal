''' Configuration Manager for Uncertainty Calculator User Interface.

The settings are saved using the Qt QSettings interface (and only apply to the GUI).
Settings are saved in an INI file to the location:
    (Windows) -- $APPDATA$/SandiaPSL/SandiaUncertCalc.ini
    (Mac/Unix) -- ~/.config/SandiaPSL/SandiaUncertCalc.ini
'''

import os
import scipy.stats as stats
from PyQt5 import QtCore, QtWidgets, QtGui
import matplotlib as mpl

from . import gui_common
from . import gui_widgets
from . import colormap
from .. import report
from .. import unitmgr

# List all distributions in scipy.stats!
DISTS = [d for d in dir(stats) if isinstance(getattr(stats, d), (stats.rv_continuous, stats.rv_discrete))]
DISTS = ['normal', 'triangular', 'curvtrap', 'resolution'] + DISTS

dfltcmap = 'viridis'
dfltstyle = 'bmh'
assert dfltstyle in mpl.style.available   # In case MPL changes styles or colormaps
assert hasattr(mpl.cm, dfltcmap)


class Settings(object):
    ''' Object for reading and writing configuration settings.

        In most cases, if an invalid value is provided, a default will be written to the settings file.
    '''
    dfltdists = ['normal', 'triangular',  'uniform', 't', 'gamma', 'lognorm', 'expon', 'curvtrap', 'beta',
                 'poisson', 'trapz', 'arcsine', 'resolution']

    def __init__(self):
        self.settings = QtCore.QSettings(QtCore.QSettings.IniFormat, QtCore.QSettings.UserScope,
                                         'SandiaPSL', 'SandiaUncertCalc')

    def sync(self):
        ''' Sync the settings to disk. QSettings already does this periodically. '''
        self.settings.sync()

    def setStyle(self, stylename):
        self.settings.setValue('style/name', stylename)

    def getStyle(self):
        return self.settings.value('style/name', dfltstyle, type=str)

    def setCustomStyle(self, styledict):
        self.settings.remove('rcparam')
        for k, v in styledict.items():
            self.settings.setValue('rcparam/'+k, v)

    def getCustomStyle(self):
        if 'rcparam' in self.settings.childGroups():
            self.settings.beginGroup('rcparam')
            params = {}
            for key in self.settings.childKeys():
                params[key] = self.settings.value(key)
            self.settings.endGroup()
            return params
        else:
            return {}

    def setColorCycle(self, colorlist):
        cyclestr = "cycler('color', ["
        cyclestr += ','.join(["'{}'".format(c) for c in colorlist])
        cyclestr += "])"
        self.settings.setValue('rcparam/axes.prop_cycle', cyclestr)

    def clearColorCycle(self):
        self.settings.remove('rcparam/axes.prop_cycle')

    def getColorFromCycle(self, index):
        cycle = self.settings.value('rcparam/axes.prop_cycle', None, type=str)
        if cycle:
            # Customized cycle
            colors = cycle[cycle.find('[')+1:cycle.find(']')].split(',')
            return colors[index]
        else:
            # Get from color C0, C1, etc.
            try:
                return mpl.rcParams['axes.prop_cycle'].by_key()['color'][index]
            except IndexError:
                return 'black'

    def getColormap(self, key):
        c = self.settings.value('style/'+key, dfltcmap, type=str)
        return c

    def setColormap(self, key, value):
        self.settings.setValue('style/'+key, value)

    def getDistributions(self):
        val = self.settings.value('montecarlo/distributions', self.dfltdists)
        return val

    def setDistributions(self, values):
        if len(values) == 0:
            values = ['normal']
        self.settings.setValue('montecarlo/distributions', values)

    def setSamples(self, value):
        try:
            value = int(value)
        except ValueError:
            value = 1000000
        value = max(value, 1)
        self.settings.setValue('montecarlo/samples', value)

    def getRandomSeed(self):
        val = self.settings.value('calculator/seed', 'None', type=str)
        try:
            val = int(val)
        except ValueError:
            val = None
        return val

    def setRandomSeed(self, value):
        # Store seed as string so None can be saved as well as int
        value = str(value)
        self.settings.setValue('calculator/seed', value)

    def getSamples(self):
        return self.settings.value('montecarlo/samples', 1000000, type=int)

    def setCoverageMC(self, values):
        self.settings.setValue('coverage/mc/levels', values)

    def setCoverageGUMt(self, values):
        self.settings.setValue('coverage/gum/levels/t', values)

    def setCoverageGUMk(self, values):
        values = [(v[-1]) for v in values]  # Strip 'k=' part
        self.settings.setValue('coverage/gum/levels/k', values)

    def setCoverageTypeGUM(self, value):
        assert value.lower() in ['t', 'k']
        self.settings.setValue('coverage/gum/type', value)

    def setCoverageTypeMC(self, value):
        assert value.lower() in ['symmetric', 'shortest']
        self.settings.setValue('coverage/mc/type', value)

    def getCoverageMC(self):
        # Store as strings so percent OR k=2 values can be saved
        vals = self.settings.value('coverage/mc/levels', ['99%', '95%', '90%', '85%', '68%'])
        return vals

    def getCoverageGUMt(self):
        return self.settings.value('coverage/gum/levels/t', ['99%', '95%', '90%', '85%', '68%'])

    def getCoverageGUMk(self):
        vals = self.settings.value('coverage/gum/levels/k', [2])
        try:
            vals = ['k = {}'.format(v) for v in vals]
        except TypeError:
            vals = ['k = 2']
        return vals

    def getCoverageTypeGUM(self):
        return self.settings.value('coverage/gum/type', 't')  # t or k

    def getCoverageTypeMC(self):
        return self.settings.value('coverage/mc/type', 'symmetric')  # symmetric or shortest

    def getFunc(self):
        return self.settings.value('calculator/function', 'f = x', type=str)

    def setFunc(self, value):
        self.settings.setValue('calculator/function', value)

    def getSigfigs(self):
        return self.settings.value('report/sigfigs', 2, type=int)

    def setSigfigs(self, value):
        self.settings.setValue('report/sigfigs', value)

    def getNumformat(self):
        return self.settings.value('report/numformat', 'auto', type=str)

    def setNumformat(self, value):
        assert value in ['auto', 'scientific', 'engineering', 'decimal', 'si']
        self.settings.setValue('report/numformat', value)

    def getRptFormat(self):
        return self.settings.value('report/format', 'html', type=str)

    def setRptFormat(self, value):
        assert value in ['html', 'pdf', 'md', 'docx', 'odt']
        self.settings.setValue('report/format', value)

    def getRptImgFormat(self):
        return self.settings.value('report/imgformat', 'svg', type=str)

    def setRptImgFormat(self, value):
        assert value in ['svg', 'png']
        self.settings.setValue('report/imgformat', value)

    def getRptMath(self):
        return self.settings.value('report/mathmode', 'mpl', type=str)

    def setRptMath(self, value):
        assert value in ['mathjax', 'mpl']
        self.settings.setValue('report/mathmode', value)

    def getRptMJURL(self):
        return self.settings.value('report/mathjaxurl', 'https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.4/MathJax.js', type=str)

    def setRptMJURL(self, value):
        self.settings.setValue('report/mathjaxurl', value)

    def getRptUnicode(self):
        return self.settings.value('report/unicode', True, type=bool)

    def setRptUnicode(self, value):
        assert value in [True, False]
        self.settings.setValue('report/unicode', value)

    def getPandocPath(self):
        return self.settings.value('report/pandoc', report.pandoc_path, type=str)

    def getLatexPath(self):
        return self.settings.value('report/latex', report.latex_path, type=str)

    def setPandocPath(self, value):  # Should have already checked that path is valid
        self.settings.setValue('report/pandoc', value)

    def setLatexPath(self, value):
        self.settings.setValue('report/latex', value)

    def getUnitDefs(self):
        custom = unitmgr.get_customunits()  # Get ones that were loaded from a calc config
        saved = self.settings.value('units/definitions', '', type=str)  # And ones saved in GUI config
        for item in custom.splitlines():
            if item not in saved:
                saved += ('\n' + item)
        return saved.strip()

    def setUnitDefs(self, value):
        self.settings.setValue('units/definitions', value)
        err = self.registerUnits()
        return err

    def registerUnits(self):
        ''' Register unit definitions with pint UnitRegistry '''
        unitdefs = self.getUnitDefs()
        err = unitmgr.register_units(unitdefs)
        return err

    def setDefaults(self):
        ''' Restore all values to default '''
        self.setStyle('bmh')
        self.settings.remove('rcparam')
        self.setColormap('cmapscatter', dfltcmap)
        self.setColormap('cmapcontour', dfltcmap)
        self.setDistributions(self.dfltdists)
        self.setSamples(1000000)
        self.setCoverageGUMt(['99%', '95%', '90%', '85%', '68%'])
        self.setCoverageGUMk(['k=2'])
        self.setCoverageMC(['99%', '95%', '90%', '85%', '68%'])
        self.setCoverageTypeGUM('t')
        self.setCoverageTypeMC('symmetric')
        self.setFunc('f = x')
        self.setSigfigs(2)
        self.setNumformat('auto')
        self.setRandomSeed(None)
        self.setRptFormat('html')
        self.setRptImgFormat('svg')
        self.setRptMath('mpl')
        self.setRptMJURL('https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.4/MathJax.js')
        self.setPandocPath(None)
        self.setLatexPath(None)
        self.sync()


#--------------------------------------------------------------------
# GUI Dialog for changing settings
#--------------------------------------------------------------------
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
        if dlg.exec_():
            self.setColor(dlg.currentColor().name())

    def setColor(self, color):
        ''' Set the color of the button '''
        if color != self.color:
            self.color = color
            self.colorChanged.emit(self.color)
        if self.color:
            self.setStyleSheet('background-color: {}'.format(self.color))
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
        dlg = colormap.ColorMapDialog(self, self.cmap)
        if dlg.exec_() and dlg.selectedcmap:
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
        self.txtFunc = QtWidgets.QLineEdit()
        self.sigfigs = QtWidgets.QSpinBox()
        self.sigfigs.setMinimum(1)
        self.sigfigs.setMaximum(20)
        self.nformat = QtWidgets.QComboBox()
        self.nformat.addItems(['Auto', 'Decimal', 'Scientific', 'Engineering', 'SI'])

        flayout = QtWidgets.QFormLayout()
        flayout.addRow('Default Function', self.txtFunc)
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
            item.setFlags(item.flags() | QtCore.Qt.ItemIsUserCheckable)
            item.setCheckState(QtCore.Qt.Unchecked)
            self.dlist.addItem(item)

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(label)
        layout.addWidget(self.dlist)
        self.setLayout(layout)


class PgUnits(QtWidgets.QWidget):
    ''' Page for defining custom measurement units '''
    def __init__(self, parent=None):
        super().__init__(parent)
        label = QtWidgets.QLabel('Enter custom unit names and abbreviations\nas equations relating to other units. Example:\n\n    banana_dose = 78*nanosieverts = bn\n\nSee Pint documentation for details.')
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


class PgExpanded(QtWidgets.QWidget):
    ''' Page for intervals and type of expanded uncertainty report '''
    def __init__(self, parent=None):
        super().__init__(parent)
        self.GUMcov = gui_widgets.GUMExpandedWidget()
        self.MCcov = gui_widgets.MCExpandedWidget()

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(QtWidgets.QLabel('Defaults for Expanded Report'))
        layout.addWidget(self.GUMcov)
        layout.addWidget(self.MCcov)
        self.setLayout(layout)

    def setCov(self, covgumt, covgumk, covmc, gumtype, mctype):
        ''' Set widget values '''
        self.GUMcov.GUMtype.setCurrentIndex(0 if gumtype == 't' else 1)
        self.MCcov.MCtype.setCurrentIndex(0 if mctype == 'symmetric' else 1)
        self.GUMcov.set_buttons(covgumt+covgumk)
        self.MCcov.set_buttons(covmc)

    def getCov(self):
        gumtype = 't' if self.GUMcov.GUMtype.currentIndex() == 0 else 'k'
        mctype = 'symmetric' if self.MCcov.MCtype.currentIndex() == 0 else 'shortest'
        gumcovt = self.GUMcov.covbuttons.get_covlist()
        gumcovk = self.GUMcov.kbuttons.get_covlist()
        return gumcovt, gumcovk, self.MCcov.get_covlist(), gumtype, mctype


class PgStyle(QtWidgets.QWidget):
    ''' Page for report settings '''
    def __init__(self, parent=None):
        super().__init__(parent)
        self.colorscustomized = False
        self.customrc = {}
        self.cmbStyle = QtWidgets.QComboBox()
        self.cmbStyle.addItems(mpl.style.available)
        self.clrHist = ColorButton('#992211')   # C0
        self.clrGum = ColorButton('#4455BB')    # C1
        self.clrScat = ColorButton('#4455BB')   # C3
        self.clrI = [ColorButton('black'),      # C4
                     ColorButton('black'),      # C5
                     ColorButton('black'),      # C6
                     ColorButton('black'),      # C7
                     ColorButton('black'),      # C8
                     ColorButton('black')]      # C9

        self.cmpGum = ColorMapButton('viridis')
        self.cmpScat = ColorMapButton('Purples')
        self.cmbStyle.currentIndexChanged.connect(self.stylechange)
        self.clrHist.pressed.connect(self.colorchange)
        self.clrGum.pressed.connect(self.colorchange)
        self.clrScat.pressed.connect(self.colorchange)
        [clr.pressed.connect(self.colorchange) for clr in self.clrI]
        self.btnCustom = QtWidgets.QPushButton('Customize Style...')
        self.btnCustom.pressed.connect(self.customize)

        blayout = QtWidgets.QHBoxLayout()
        blayout.addWidget(QtWidgets.QLabel('Base Style:'))
        blayout.addWidget(self.cmbStyle)
        flayout = QtWidgets.QHBoxLayout()
        flayout = QtWidgets.QFormLayout()
        flayout.addRow('Histograms', self.clrHist)
        flayout.addRow('PDFs', self.clrGum)
        flayout.addRow('Scatter Plots', self.clrScat)
        intlayout = QtWidgets.QHBoxLayout()
        intlayout.addWidget(QtWidgets.QLabel('Intervals:'))
        [intlayout.addWidget(i) for i in self.clrI]
        f2layout = QtWidgets.QFormLayout()
        f2layout.addRow('Contour Colormap', self.cmpGum)
        f2layout.addRow('Discrete Colormap', self.cmpScat)
        flayout.setHorizontalSpacing(20)
        f2layout.setHorizontalSpacing(20)
        mainlayout = QtWidgets.QVBoxLayout()
        mainlayout.addLayout(blayout)
        mainlayout.addLayout(flayout)
        mainlayout.addLayout(intlayout)
        mainlayout.addLayout(f2layout)
        mainlayout.addWidget(self.btnCustom)
        self.setLayout(mainlayout)

    def stylechange(self, index):
        self.blockSignals(True)
        mpl.rcParams.update(mpl.rcParamsDefault)
        mpl.style.use(self.cmbStyle.currentText())
        self.clrHist.setColor(mpl.rcParams['axes.prop_cycle'].by_key()['color'][0])
        self.clrGum.setColor(mpl.rcParams['axes.prop_cycle'].by_key()['color'][1])
        self.clrScat.setColor(mpl.rcParams['axes.prop_cycle'].by_key()['color'][3])
        for i, clr in enumerate(self.clrI):
            try:
                clr.setColor(mpl.rcParams['axes.prop_cycle'].by_key()['color'][4+i])
            except IndexError:
                # Color not defined in this style
                clr.setColor('black')
        self.blockSignals(False)
        self.colorscustomized = False

    def colorchange(self):
        self.colorscustomized = True

    def get_colorcycle(self):
        clist = [self.clrHist.color, self.clrGum.color, mpl.rcParams['axes.prop_cycle'].by_key()['color'][2], self.clrScat.color]
        clist.extend([c.color for c in self.clrI])
        return clist

    def customize(self):
        lines = []
        for k, v in self.customrc.items():
            lines.append(k + ': ' + v)
        text = '\n'.join(lines)

        label = '''Enter parameters compatible with Matplotlib rcParams file.\nFor example, "lines.linewidth: 2.0"'''
        dlg = DlgMultiLineEdit('Enter Parameters', label=label, text=text)
        ok = dlg.exec_()
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
                    print('Unknown rcParam {}'.format(key))


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
        self.mjurl = QtWidgets.QLineEdit()
        self.pandoc = QtWidgets.QLineEdit()
        self.latex = QtWidgets.QLineEdit()

        glayout = QtWidgets.QFormLayout()
        glayout.addRow('File Format', self.cmbFormat)
        glayout.addRow('Image Format', self.cmbImage)
        glayout.addRow('Math Renderer', self.cmbMath)
        glayout.addRow('Mathjax URL', self.mjurl)
        glayout.addRow('', self.chkUnicode)
        playout = QtWidgets.QFormLayout()
        playout.addRow('Pandoc', self.pandoc)
        playout.addRow('PDFLatex', self.latex)

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(QtWidgets.QLabel('Default report format options:'))
        layout.addLayout(glayout)
        layout.addWidget(gui_widgets.QHLine())
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
        btn = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
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

        self.settings = Settings()
        self.setWindowTitle('Uncertainty Calculator Settings')

        self.pgGeneral = PgGeneral(self)
        self.pgStyle = PgStyle(self)
        self.pgRptOpts = PgReportOpts(self)
        self.pgDist = PgDistribution(self)
        self.pgUnits = PgUnits(self)
        self.pgMC = PgMonteCarlo(self)
        self.pgExpand = PgExpanded(self)
        self.list = QtWidgets.QListWidget()
        self.list.setMaximumWidth(150)
        self.list.addItems(['General', 'Plot Style', 'Report Format', 'Units', 'Distributions', 'Monte-Carlo', 'Expanded Unc.'])
        self.list.currentRowChanged.connect(self.pagechange)
        self.stack = QtWidgets.QStackedWidget()
        self.stack.addWidget(self.pgGeneral)
        self.stack.addWidget(self.pgStyle)
        self.stack.addWidget(self.pgRptOpts)
        self.stack.addWidget(self.pgUnits)
        self.stack.addWidget(self.pgDist)
        self.stack.addWidget(self.pgMC)
        self.stack.addWidget(self.pgExpand)
        self.button = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel | QtWidgets.QDialogButtonBox.RestoreDefaults)
        self.button.accepted.connect(self.accept)
        self.button.rejected.connect(self.reject)
        self.button.button(QtWidgets.QDialogButtonBox.RestoreDefaults).clicked.connect(self.defaults)
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
        self.blockSignals(True)
        self.pgGeneral.txtFunc.setText(self.settings.getFunc())
        self.pgGeneral.sigfigs.setValue(self.settings.getSigfigs())
        self.pgGeneral.nformat.setCurrentIndex(self.pgGeneral.nformat.findText(self.settings.getNumformat(), flags=QtCore.Qt.MatchExactly | QtCore.Qt.MatchFixedString))  # Not MatchFixedString so not case sensitive
        self.pgStyle.cmpGum.setCmap(self.settings.getColormap('cmapcontour'))
        self.pgStyle.cmpScat.setCmap(self.settings.getColormap('cmapscatter'))
        self.pgStyle.cmbStyle.setCurrentIndex(self.pgStyle.cmbStyle.findText(self.settings.getStyle()))

        self.pgStyle.clrHist.setColor(self.settings.getColorFromCycle(0))
        self.pgStyle.clrGum.setColor(self.settings.getColorFromCycle(1))
        self.pgStyle.clrScat.setColor(self.settings.getColorFromCycle(3))
        self.pgStyle.customrc = self.settings.getCustomStyle()
        [clr.setColor(self.settings.getColorFromCycle(4+i)) for i, clr in enumerate(self.pgStyle.clrI)]

        self.pgMC.txtSamples.setText(str(self.settings.getSamples()))
        self.pgMC.txtSeed.setText(str(self.settings.getRandomSeed()))
        self.pgExpand.setCov(self.settings.getCoverageGUMt(), self.settings.getCoverageGUMk(),
                             self.settings.getCoverageMC(), self.settings.getCoverageTypeGUM(),
                             self.settings.getCoverageTypeMC())

        self.pgUnits.unitdefs.setText(self.settings.getUnitDefs())

        self.pgRptOpts.cmbFormat.setCurrentIndex(['html', 'md', 'pdf', 'odt', 'docx'].index(self.settings.getRptFormat()))
        self.pgRptOpts.cmbMath.setCurrentIndex(['mathjax', 'mpl'].index(self.settings.getRptMath()))
        self.pgRptOpts.cmbImage.setCurrentIndex(['svg', 'png'].index(self.settings.getRptImgFormat()))
        self.pgRptOpts.mjurl.setText(self.settings.getRptMJURL())
        self.pgRptOpts.chkUnicode.setChecked(self.settings.getRptUnicode())

        pandoc = self.settings.getPandocPath()
        if pandoc is None or pandoc == '':
            pandoc = report.pandoc_path  # Get from report module default if not defined in settings
        self.pgRptOpts.pandoc.setText(pandoc if pandoc else '')
        latex = self.settings.getLatexPath()
        if latex is None or latex == '':
            latex = report.latex_path  # Get from report module default if not defined in settings
        self.pgRptOpts.latex.setText(latex if latex else '')

        dists = self.settings.getDistributions()
        for i in range(self.pgDist.dlist.count()):
            item = self.pgDist.dlist.item(i)
            if item.text() in dists:
                item.setCheckState(QtCore.Qt.Checked)
            else:
                item.setCheckState(QtCore.Qt.Unchecked)
        self.blockSignals(False)

    def defaults(self):
        ''' Restore all default values '''
        self.settings.setDefaults()
        self.loadsettings()

    def pagechange(self, index):
        ''' Change the page '''
        self.stack.setCurrentIndex(index)

    def accept(self):
        ''' Save settings and close the dialog. NOTE: NO VALIDATION CHECKING DONE! '''
        # Just write everything every time
        self.settings.setFunc(self.pgGeneral.txtFunc.text())
        self.settings.setSigfigs(self.pgGeneral.sigfigs.value())
        self.settings.setNumformat(self.pgGeneral.nformat.currentText().lower())

        self.settings.setStyle(self.pgStyle.cmbStyle.currentText())
        if self.pgStyle.colorscustomized:
            self.settings.setColorCycle(self.pgStyle.get_colorcycle())
        else:
            self.settings.clearColorCycle()

        if len(self.pgStyle.customrc):
            self.settings.setCustomStyle(self.pgStyle.customrc)

        self.settings.setColormap('cmapcontour', self.pgStyle.cmpGum.cmap)
        self.settings.setColormap('cmapscatter', self.pgStyle.cmpScat.cmap)
        self.settings.setSamples(self.pgMC.txtSamples.text())
        self.settings.setRandomSeed(self.pgMC.txtSeed.text())

        dlist = []
        for i in range(self.pgDist.dlist.count()):
            item = self.pgDist.dlist.item(i)
            if item.checkState() == QtCore.Qt.Checked:
                dlist.append(item.text())
        self.settings.setDistributions(dlist)
        gumcovt, gumcovk, mccov, gumtype, mctype = self.pgExpand.getCov()
        self.settings.setCoverageGUMt(gumcovt)
        self.settings.setCoverageGUMk(gumcovk)
        self.settings.setCoverageMC(mccov)
        self.settings.setCoverageTypeGUM(gumtype)
        self.settings.setCoverageTypeMC(mctype)

        self.settings.setRptFormat(['html', 'md', 'pdf', 'odt', 'docx'][self.pgRptOpts.cmbFormat.currentIndex()])
        self.settings.setRptImgFormat(['svg', 'png'][self.pgRptOpts.cmbImage.currentIndex()])
        self.settings.setRptMath(['mathjax', 'mpl'][self.pgRptOpts.cmbMath.currentIndex()])
        self.settings.setRptMJURL(self.pgRptOpts.mjurl.text())
        self.settings.setRptUnicode(self.pgRptOpts.chkUnicode.isChecked())

        err = self.settings.setUnitDefs(self.pgUnits.unitdefs.toPlainText())
        if err:
            QtWidgets.QMessageBox.warning(self, 'Uncertainty Calculator', err)

        if self.pgRptOpts.pandoc.text() != report.pandoc_path and os.path.exists(self.pgRptOpts.pandoc.text()):  # Only save if customized and pandoc is there
            self.settings.setPandocPath(self.pgRptOpts.pandoc.text())
            report.pandoc_path = self.pgRptOpts.pandoc.text()
        if self.pgRptOpts.latex.text() != report.latex_path and os.path.exists(self.pgRptOpts.latex.text()):
            self.settings.setLatexPath(self.pgRptOpts.latex.text())
            report.latex_path = self.pgRptOpts.latex.text()

        super().accept()  # Let the dialog finish closing properly


if __name__ == '__main__':
    # Run just the settings dialog
    # run using: "python -m suncal.gui.configmgr"
    import sys
    app = QtWidgets.QApplication(sys.argv)
    main = PgSettingsDlg()
    main.show()
    app.exec_()
