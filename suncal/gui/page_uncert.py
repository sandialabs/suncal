''' Page for propagating uncertainty calculations '''

import re
import numpy as np
from pint import DimensionalityError, UndefinedUnitError, OffsetUnitCalculusError
from PyQt5 import QtWidgets, QtCore

from . import gui_common
from . import gui_widgets
from . import page_uncert_input, page_uncert_output
from . import page_dataimport


class UncertPropWidget(QtWidgets.QWidget):
    ''' Uncertainty propagation widget '''
    openconfigfolder = QtCore.QStandardPaths.standardLocations(QtCore.QStandardPaths.HomeLocation)[0]
    newtype = QtCore.pyqtSignal(dict, str)

    PG_INPUT = 0
    PG_OUTPUT = 1

    def __init__(self, projitem=None, parent=None):
        super().__init__(parent)
        self.projitem = projitem
        self.pginput = page_uncert_input.PageInput(projitem)
        self.pgoutput = page_uncert_output.PageOutput()
        self.stack = gui_widgets.SlidingStackedWidget()
        self.stack.addWidget(self.pginput)
        self.stack.addWidget(self.pgoutput)

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.stack)
        self.setLayout(layout)

        # Menu
        self.menu = QtWidgets.QMenu('&Uncertainty')
        self.actChkUnits = QtWidgets.QAction('Check Units...', self)
        self.actSweep = QtWidgets.QAction('New uncertainty sweep from model', self)
        self.actReverse = QtWidgets.QAction('New reverse calculation from model', self)
        self.actImportDists = QtWidgets.QAction('Import uncertainty distributions...', self)
        self.actClear = QtWidgets.QAction('Clear inputs', self)
        self.actSaveReport = QtWidgets.QAction('Save Report...', self)
        self.actSaveSamplesCSV = QtWidgets.QAction('Text (CSV)...', self)
        self.actSaveSamplesNPZ = QtWidgets.QAction('Binary (NPZ)...', self)

        self.menu.addAction(self.actChkUnits)
        self.menu.addSeparator()
        self.menu.addAction(self.actImportDists)
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

        self.actImportDists.triggered.connect(self.importdistributions)
        self.actClear.triggered.connect(self.clearinput)
        self.actSweep.triggered.connect(lambda event: self.newtype.emit(self.pginput.get_config(), 'sweep'))
        self.actReverse.triggered.connect(lambda event: self.newtype.emit(self.pginput.get_config(), 'reverse'))
        self.actSaveReport.triggered.connect(self.save_report)
        self.actSaveSamplesCSV.triggered.connect(self.save_samples_csv)
        self.actSaveSamplesNPZ.triggered.connect(self.save_samples_npz)
        self.actChkUnits.triggered.connect(self.checkunits)

        self.pginput.calculate.connect(self.calculate)
        self.pgoutput.back.connect(self.backbutton)
        gui_common.set_plot_style()

    def get_menu(self):
        ''' Get menu for this widget '''
        return self.menu

    def get_config(self):
        return self.pginput.get_config()

    def update_proj_config(self):
        ''' Save page setup back to project item configuration '''
        self.projitem.load_config(self.get_config())

    def backbutton(self):
        ''' Back button pressed. '''
        self.stack.slideInRight(self.PG_INPUT)
        self.actSaveReport.setEnabled(False)
        self.mnuSaveSamples.setEnabled(False)

    def clearinput(self):
        ''' Clear all the input/output values '''
        self.pginput.funclist.clear()
        self.pginput.meastable.clear()
        self.pginput.corrtable.clear()
        self.pginput.description.setPlainText('')
        self.pginput.settings.setvalue('Random Seed', gui_common.settings.getRandomSeed())
        self.pginput.settings.setvalue('Monte Carlo Samples', gui_common.settings.getSamples())
        self.actSweep.setEnabled(False)
        self.actReverse.setEnabled(False)
        self.stack.slideInRight(self.PG_INPUT)

    def checkunits(self):
        ''' Show units/dimensionality report '''
        self.update_proj_config()
        dlg = gui_widgets.MarkdownTextEdit()
        dlg.setMinimumSize(800, 600)
        dlg.setReport(self.projitem.units_report())
        dlg.show()

    def importdistributions(self):
        ''' Load uncertainty components from data for multiple input variables '''
        self.update_proj_config()
        varnames = self.projitem.model.variables.names
        dlg = page_dataimport.DistributionSelectWidget(
            singlecol=False, project=self.projitem.project, coloptions=varnames)
        if dlg.exec_():
            dists = dlg.get_dist()

            config = self.projitem.get_config()

            for varname, params in dists.items():
                if varname == '_correlation_':
                    self.pginput.panel.expand('Correlations')
                    config.setdefault('correlations', [])
                    for (v1, v2), corr in params.items():
                        if corr != 0.:
                            self.pginput.corrtable.addRow()
                            self.pginput.corrtable.setRow(self.pginput.corrtable.rowCount()-1, v1, v2, corr)
                            config['correlations'].append({'var1': v1, 'var2': v2, 'cor': corr})

                else:
                    # Don't pass along median, it's handled by Variable
                    nom = params.pop('expected', params.pop('median', params.pop('mean', None)))
                    varconfigs = config.get('inputs', [])
                    cfgnames = [v.get('name') for v in varconfigs]
                    varconfig = varconfigs[cfgnames.index(varname)]
                    units = varconfig.get('units', None)
                    params.setdefault('name', f'u({varname})')
                    params.setdefault('degf', params.pop('df', np.inf))
                    params.setdefault('units', str(units) if units else None)
                    varconfig['mean'] = nom
                    uncnames = [u.get('name') for u in varconfig.get('uncerts')]
                    if params.get('name') in uncnames:
                        varconfig['uncerts'][uncnames.index(params.get('name'))] = params
                    else:
                        varconfig['uncerts'].append(params)

                self.pginput.meastable.load_config(config)
                self.backbutton()

    def calculate(self):
        ''' Run the calculation '''
        def err_msg(msg):
            QtWidgets.QMessageBox.warning(self, 'Suncal', msg)
            self.actSaveReport.setEnabled(False)
            self.mnuSaveSamples.setEnabled(False)

        if not (self.pginput.funclist.is_valid() and self.pginput.meastable.is_valid()):
            err_msg('Invalid input parameter!')
            return

        config = self.get_config()

        if len(config.get('functions', [])) < 1:
            err_msg('No functions to compute!')
            return

        try:
            self.projitem.load_config(config)
        except RecursionError:
            err_msg('Circular reference in function definitions')
            return

        if not self.pginput.symbolicmode:
            # Check units/dimensionality
            try:
                self.projitem.model.eval()
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
            result = self.projitem.calculate(mc=not self.pginput.symbolicmode)
        except (TypeError, DimensionalityError, UndefinedUnitError) as exc:
            err_msg(f'Units Error: {exc}')
            return
        except OffsetUnitCalculusError as exc:
            badunit = re.findall(r'\((.+ )', str(exc))[0].split()[0].strip(', ')
            err_msg(f'Ambiguous unit {badunit}. Try "delta_{badunit}".')
            return
        except RecursionError:
            err_msg('Error - possible circular reference in function definitions')
            return
        except (TypeError, ValueError):
            err_msg('No Solution Found!\n\n')
            return

        self.stack.slideInLeft(self.PG_OUTPUT)
        self.pgoutput.update(result, symonly=self.pginput.symbolicmode)
        self.pgoutput.outputupdate()
        self.actSaveReport.setEnabled(True)
        self.mnuSaveSamples.setEnabled(True)

    def get_report(self):
        ''' Get full report of curve fit, using page settings '''
        rpt = self.pgoutput.outputReportSetup.report
        if rpt is None and hasattr(self.pgoutput, 'outreport'):
            rpt = self.pgoutput.refresh_fullreport()
        return rpt

    def save_report(self):
        ''' Save full output report to file, using user's GUI settings '''
        gui_widgets.savereport(self.get_report())

    def save_samples_csv(self):
        ''' Save Monte-Carlo samples (inputs and outputs) to CSV file. This file can get big fast! '''
        fname, _ = QtWidgets.QFileDialog.getSaveFileName(
            caption='Select file to save', directory=self.openconfigfolder, filter='CSV (*.csv)')
        if fname:
            self.projitem.save_samples_csv(fname)

    def save_samples_npz(self):
        ''' Save Monte-Carlo samples to NPZ (compressed numpy) file.

            Load into python using numpy.load() to return a dictionary with
            'samples' and 'hdr' keys.
        '''
        fname, _ = QtWidgets.QFileDialog.getSaveFileName(
            caption='Select file to save', directory=self.openconfigfolder, filter='Numpy NPZ (*.npz)')
        if fname:
            self.projitem.save_samples_npz(fname)
