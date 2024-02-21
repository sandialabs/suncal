''' Page for propagating uncertainty calculations '''

import re
from pint import DimensionalityError, UndefinedUnitError, OffsetUnitCalculusError
from PyQt6 import QtWidgets, QtCore, QtGui

from . import widgets
from . import gui_styles
from .gui_settings import gui_settings
from . import page_uncert_input, page_uncert_output
from .help_strings import UncertHelp


class UncertPropWidget(QtWidgets.QWidget):
    ''' Uncertainty propagation widget '''
    openconfigfolder = QtCore.QStandardPaths.standardLocations(
        QtCore.QStandardPaths.StandardLocation.HomeLocation)[0]
    newtype = QtCore.pyqtSignal(dict, str)
    change_help = QtCore.pyqtSignal()

    PG_INPUT = 0
    PG_OUTPUT = 1

    def __init__(self, component=None, parent=None):
        super().__init__(parent)
        self.component = component
        self.pginput = page_uncert_input.PageInput(component)
        self.pgoutput = page_uncert_output.PageOutput()
        self.stack = widgets.SlidingStackedWidget()
        self.stack.addWidget(self.pginput)
        self.stack.addWidget(self.pgoutput)

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.stack)
        self.setLayout(layout)

        # Menu
        self.menu = QtWidgets.QMenu('&Uncertainty')
        self.actChkUnits = QtGui.QAction('Check Units...', self)
        self.actSweep = QtGui.QAction('New uncertainty sweep from model', self)
        self.actReverse = QtGui.QAction('New reverse calculation from model', self)
        self.actMeasuredData = QtGui.QAction('Type A measurement data...', self)
        self.actClearComponents = QtGui.QAction('Clear uncertainty components', self)
        self.actClear = QtGui.QAction('Clear inputs', self)
        self.actSaveReport = QtGui.QAction('Save Report...', self)
        self.actSaveSamplesCSV = QtGui.QAction('Text (CSV)...', self)
        self.actSaveSamplesNPZ = QtGui.QAction('Binary (NPZ)...', self)

        self.menu.addAction(self.actChkUnits)
        self.menu.addSeparator()
        self.menu.addAction(self.actMeasuredData)
        self.menu.addAction(self.actClearComponents)
        self.menu.addAction(self.actClear)
        self.menu.addSeparator()
        self.menu.addAction(self.actSweep)
        self.menu.addAction(self.actReverse)
        self.menu.addSeparator()
        self.menu.addAction(self.actSaveReport)
        self.mnuSaveSamples = QtWidgets.QMenu('Save Monte Carlo Samples')
        self.mnuSaveSamples.addAction(self.actSaveSamplesCSV)
        self.mnuSaveSamples.addAction(self.actSaveSamplesNPZ)
        self.menu.addMenu(self.mnuSaveSamples)
        self.actSaveReport.setEnabled(False)
        self.mnuSaveSamples.setEnabled(False)

        self.actMeasuredData.triggered.connect(self.enter_typea_data)
        self.actClearComponents.triggered.connect(self.clearcomponents)
        self.actClear.triggered.connect(self.clearinput)
        self.actSweep.triggered.connect(lambda event: self.newtype.emit(self.pginput.get_config(), 'sweep'))
        self.actReverse.triggered.connect(lambda event: self.newtype.emit(self.pginput.get_config(), 'reverse'))
        self.actSaveReport.triggered.connect(self.save_report)
        self.actSaveSamplesCSV.triggered.connect(self.save_samples_csv)
        self.actSaveSamplesNPZ.triggered.connect(self.save_samples_npz)
        self.actChkUnits.triggered.connect(self.checkunits)

        self.pginput.calculate.connect(self.calculate)
        self.pgoutput.back.connect(self.backbutton)
        self.pgoutput.change_help.connect(self.change_help)

    def get_menu(self):
        ''' Get menu for this widget '''
        return self.menu

    def get_config(self):
        ''' Get component configuration from values entered in page '''
        return self.pginput.get_config()

    def update_proj_config(self):
        ''' Save page setup back to project item configuration '''
        self.component.load_config(self.get_config())

    def backbutton(self):
        ''' Back button pressed. '''
        self.stack.slideInRight(self.PG_INPUT)
        self.actSaveReport.setEnabled(False)
        self.mnuSaveSamples.setEnabled(False)
        self.change_help.emit()

    def clearcomponents(self):
        ''' Clear uncertianty components '''
        self.update_proj_config()
        self.component.clear_uncertainties()
        self.pginput.load_config(self.component.get_config())

    def clearinput(self):
        ''' Clear all the input/output values '''
        self.pginput.funclist.clear()
        self.pginput.meastable.clear()
        self.pginput.corrtable.clear()
        self.pginput.description.setPlainText('')
        self.pginput.settings.setvalue('Random Seed', gui_settings.randomseed)
        self.pginput.settings.setvalue('Monte Carlo Samples', gui_settings.samples)
        self.actSweep.setEnabled(False)
        self.actReverse.setEnabled(False)
        self.stack.slideInRight(self.PG_INPUT)

    def checkunits(self):
        ''' Show units/dimensionality report '''
        self.update_proj_config()
        dlg = widgets.MarkdownTextEdit()
        dlg.setMinimumSize(800, 600)
        dlg.setReport(self.component.units_report())
        dlg.show()

    def enter_typea_data(self):
        ''' Show dialog for entering Type A measurement data '''
        self.pginput.meastable.enter_typea_data()

    def calculate(self):
        ''' Run the calculation '''
        def err_msg(msg):
            QtWidgets.QMessageBox.warning(self, 'Suncal', msg)
            self.actSaveReport.setEnabled(False)
            self.mnuSaveSamples.setEnabled(False)

        if not (self.pginput.funclist.is_valid() and self.pginput.meastable.is_valid()):
            err_msg('Invalid input parameter!')
            return

        self.update_proj_config()
        config = self.get_config()

        if len(config.get('functions', [])) < 1:
            err_msg('No functions to compute!')
            return

        try:
            self.component.load_config(config)
        except RecursionError:
            err_msg('Circular reference in function definitions')
            return

        if not self.pginput.symbolicmode:
            # Check units/dimensionality
            try:
                self.component.model.eval()
            except OffsetUnitCalculusError as exc:
                badunit = re.findall(r'\((.+ )', str(exc))[0].split()[0].strip(', ')
                err_msg(f'Ambiguous unit {badunit}. Try "delta_{badunit}".')
                return
            except (TypeError, DimensionalityError, UndefinedUnitError) as exc:
                err_msg(f'Units Error: {exc}')
                return
            except RecursionError:
                err_msg('Error - possible circular reference in function definitions')
                return

        try:
            result = self.component.calculate(mc=not self.pginput.symbolicmode)
        except OffsetUnitCalculusError as exc:
            badunit = re.findall(r'\((.+ )', str(exc))[0].split()[0].strip(', ')
            err_msg(f'Ambiguous unit {badunit}. Try "delta_{badunit}".')
            return
        except (TypeError, DimensionalityError, UndefinedUnitError) as exc:
            err_msg(f'Units Error: {exc}')
            return
        except RecursionError:
            err_msg('Error - possible circular reference in function definitions')
            return
        except ValueError:
            err_msg('No Solution Found!\n\n')
            return

        self.stack.slideInLeft(self.PG_OUTPUT)
        self.pgoutput.update(result, symonly=self.pginput.symbolicmode)
        self.pgoutput.outputupdate()
        self.actSaveReport.setEnabled(True)
        self.mnuSaveSamples.setEnabled(True)
        self.change_help.emit()

    def get_report(self):
        ''' Get full report of curve fit, using page settings '''
        rpt = self.pgoutput.outputReportSetup.report
        if rpt is None and hasattr(self.pgoutput, 'outreport'):
            rpt = self.pgoutput.refresh_fullreport()
        return rpt

    def save_report(self):
        ''' Save full output report to file, using user's GUI settings '''
        with gui_styles.LightPlotstyle():
            widgets.savereport(self.pgoutput.generate_report())

    def save_samples_csv(self):
        ''' Save Monte-Carlo samples (inputs and outputs) to CSV file. This file can get big fast! '''
        fname, _ = QtWidgets.QFileDialog.getSaveFileName(
            caption='Select file to save', directory=self.openconfigfolder, filter='CSV (*.csv)')
        if fname:
            self.component.save_samples_csv(fname)

    def save_samples_npz(self):
        ''' Save Monte-Carlo samples to NPZ (compressed numpy) file.

            Load into python using numpy.load() to return a dictionary with
            'samples' and 'hdr' keys.
        '''
        fname, _ = QtWidgets.QFileDialog.getSaveFileName(
            caption='Select file to save', directory=self.openconfigfolder, filter='Numpy NPZ (*.npz)')
        if fname:
            self.component.save_samples_npz(fname)

    def help_report(self):
        ''' Get the help report to display the current widget mode '''
        if self.stack.m_next == self.PG_INPUT:
            return UncertHelp.inputs()
        else:
            return self.pgoutput.help_report()
