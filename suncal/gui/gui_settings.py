''' Configuration Preferences for Suncal GUI

The settings are saved using the Qt QSettings interface (and only apply to the GUI).
Settings are saved in an INI file to the location:
    (Windows) -- $APPDATA$/SandiaPSL/SandiaUncertCalc.ini
    (Mac/Unix) -- ~/.config/SandiaPSL/SandiaUncertCalc.ini
'''
from typing import Literal
import os
from PyQt6 import QtCore
import matplotlib as mpl

from ..common import report, unitmgr


DFLTCMAP = 'viridis'
DFLTSTYLE = 'Suncal'
assert hasattr(mpl.cm, DFLTCMAP)
DFLTDISTS = ['normal', 'triangular',  'uniform', 't', 'gamma', 'lognorm', 'expon',
             'curvtrap', 'beta', 'poisson', 'trapz', 'arcsine', 'resolution']


class Settings:
    _settings = QtCore.QSettings(QtCore.QSettings.Format.IniFormat,
                                 QtCore.QSettings.Scope.UserScope,
                                 'SandiaPSL', 'SandiaUncertCalc')

    def __init__(self):
        self.register_units()

        # Load Pandoc/Latex paths from settings into report module
        pandoc = self.pandoc_path
        latex = self.latex_path
        if pandoc and os.path.exists(pandoc):
            report.pandoc_path = pandoc
        if latex and os.path.exists(latex):
            report.latex_path = latex

    def sync(self):
        ''' Sync the settings to disk. QSettings already does this periodically. '''
        self._settings.sync()

    @property
    def plot_style(self) -> str:
        ''' Get plotting style '''
        return self._settings.value('style/theme', DFLTSTYLE, type=str)

    @plot_style.setter
    def plot_style(self, stylename: str) -> None:
        ''' Set plotting style '''
        # Note: setting key changed after 1.6.4 to revert everyone to new `Suncal` style
        # and avoid confusion with Light/Dark modes.
        self._settings.setValue('style/theme', stylename)

    @property
    def plotparams(self) -> dict[str, str]:
        ''' Get custom rcParams for Matplotlib '''
        if 'rcparam' in self._settings.childGroups():
            self._settings.beginGroup('rcparam')
            params = {}
            for key in self._settings.childKeys():
                params[key] = self._settings.value(key)
            self._settings.endGroup()
            return params
        else:
            return {}

    @plotparams.setter
    def plotparams(self, styledict: dict[str: str]) -> None:
        ''' Set custom rcParams for Matplotlib '''
        self._settings.remove('rcparam')
        for k, v in styledict.items():
            self._settings.setValue('rcparam/'+k, v)

    @property
    def color_cycle(self) -> str:
        ''' Get plot color cycle '''
        return self._settings.value('style/color_cycle', None, type=str)

    @color_cycle.setter
    def color_cycle(self, colorlist: list[str]) -> None:
        ''' Set plotting color cycle to override main plotting style '''
        cyclestr = "cycler('color', ["
        cyclestr += ','.join(f"'{c}'" for c in colorlist)
        cyclestr += "])"
        self._settings.setValue('style/color_cycle', cyclestr)

    def clear_color_cycle(self) -> None:
        ''' Clear custom plotting color cycle '''
        self._settings.remove('style/color_cycle')

    def get_color_from_cycle(self, index: int) -> str:
        ''' Get a color from the plotting color cycle '''
        cycle = self._settings.value('style/color_cycle', None, type=str)
        if cycle:
            # Customized cycle
            colors = cycle[cycle.find('[')+1:cycle.find(']')].split(',')
            return colors[index]

        # Get from color C0, C1, etc.
        try:
            return mpl.rcParams['axes.prop_cycle'].by_key()['color'][index]
        except IndexError:
            return 'black'

    @property
    def colormap_contour(self) -> str:
        ''' Get the colormap name for the plot type '''
        c = self._settings.value('style/cmapcontour', DFLTCMAP, type=str)
        return c

    @colormap_contour.setter
    def colormap_contour(self, value: str) -> None:
        ''' Set the colormap name '''
        self._settings.setValue('style/cmapcontour', value)

    @property
    def colormap_scatter(self) -> str:
        ''' Get the colormap name for the plot type '''
        c = self._settings.value('style/cmapscatter', DFLTCMAP, type=str)
        return c

    @colormap_scatter.setter
    def colormap_scatter(self, value: str) -> None:
        ''' Set the colormap name '''
        self._settings.setValue('style/cmapscatter', value)

    @property
    def distributions(self) -> list[str]:
        ''' Get enabled probability distributions '''
        val = self._settings.value('montecarlo/distributions', DFLTDISTS)
        return val

    @distributions.setter
    def distributions(self, values: list[str]) -> None:
        ''' Set enabled probability distributions '''
        if len(values) == 0:
            values = ['normal']
        self._settings.setValue('montecarlo/distributions', values)

    @property
    def samples(self) -> int:
        ''' Get the number of Monte Carlo samples '''
        return self._settings.value('montecarlo/samples', 1000000, type=int)

    @samples.setter
    def samples(self, value: int) -> None:
        ''' Set the number of Monte Carlo samples '''
        try:
            value = int(value)
        except ValueError:
            value = 1000000
        value = max(value, 1)
        self._settings.setValue('montecarlo/samples', value)

    @property
    def randomseed(self) -> str:
        ''' Get the random number seed '''
        val = self._settings.value('calculator/seed', 'None', type=str)
        try:
            val = int(val)
        except ValueError:
            val = None
        return val

    @randomseed.setter
    def randomseed(self, value: str) -> None:
        ''' Set the random number seed '''
        # Store seed as string so None can be saved as well as int
        value = str(value)
        self._settings.setValue('calculator/seed', value)

    @property
    def sigfigs(self) -> int:
        ''' Get default significant figures for reports '''
        return self._settings.value('report/sigfigs', 2, type=int)

    @sigfigs.setter
    def sigfigs(self, value: int) -> None:
        ''' Set default significant figures for reports '''
        self._settings.setValue('report/sigfigs', value)

    @property
    def numformat(self) -> Literal['auto', 'scientific', 'engineering', 'decimal', 'si']:
        ''' Get report numnber format '''
        return self._settings.value('report/numformat', 'auto', type=str)

    @numformat.setter
    def numformat(self, value: Literal['auto', 'scientific', 'engineering', 'decimal', 'si']):
        ''' Set report numnber format '''
        assert value in ['auto', 'scientific', 'engineering', 'decimal', 'si']
        self._settings.setValue('report/numformat', value)

    @property
    def rptformat(self) -> Literal['html', 'pdf', 'md', 'docx', 'odt']:
        ''' Get report document format '''
        return self._settings.value('report/format', 'html', type=str)

    @rptformat.setter
    def rptformat(self, value: Literal['html', 'pdf', 'md', 'docx', 'odt']) -> None:
        ''' Set report document format '''
        assert value in ['html', 'pdf', 'md', 'docx', 'odt']
        self._settings.setValue('report/format', value)

    @property
    def rpt_imageformat(self) -> Literal['svg', 'png']:
        ''' Get report image format '''
        return self._settings.value('report/imgformat', 'svg', type=str)

    @rpt_imageformat.setter
    def rpt_imageformat(self, value: Literal['svg', 'png']):
        ''' Set report image format '''
        assert value in ['svg', 'png']
        self._settings.setValue('report/imgformat', value)

    @property
    def rpt_mathformat(self) -> Literal['mathjax', 'mpl']:
        ''' Get report math format '''
        return self._settings.value('report/mathmode', 'mpl', type=str)

    @rpt_mathformat.setter
    def rpt_mathformat(self, value: Literal['mathjax', 'mpl']) -> None:
        ''' Set report math format '''
        assert value in ['mathjax', 'mpl']
        self._settings.setValue('report/mathmode', value)

    @property
    def rpt_unicode(self) -> bool:
        ''' Get report unicode format enabled '''
        return self._settings.value('report/unicode', True, type=bool)

    @rpt_unicode.setter
    def rpt_unicode(self, value: bool) -> None:
        ''' Set report unicode format enabled '''
        assert value in [True, False]
        self._settings.setValue('report/unicode', value)

    @property
    def pandoc_path(self) -> str:
        ''' Get path to Pandoc installation '''
        return self._settings.value('report/pandoc', report.pandoc_path, type=str)

    @pandoc_path.setter
    def pandoc_path(self, value: str) -> None:
        ''' Set path to Pandoc installation '''
        # Should have already checked that path is valid
        self._settings.setValue('report/pandoc', value)

    @property
    def latex_path(self) -> str:
        ''' Get path to Latex installation '''
        return self._settings.value('report/latex', report.latex_path, type=str)

    @latex_path.setter
    def latex_path(self, value: str) -> None:
        ''' Set path to Pandoc installation '''
        self._settings.setValue('report/latex', value)

    @property
    def unit_defs(self) -> str:
        ''' Get custom unit definitions '''
        custom = unitmgr.get_customunits()  # Get ones that were loaded from a calc config
        saved = self._settings.value('units/definitions', '', type=str)  # And ones saved in GUI config
        for item in custom.splitlines():
            if item not in saved:
                saved += ('\n' + item)
        return saved.strip()

    @unit_defs.setter
    def unit_defs(self, value: str) -> None:
        ''' Set custom unit definitions (and register the new units) '''
        self._settings.setValue('units/definitions', value)
        err = self.register_units()
        return err

    def register_units(self) -> str:
        ''' Register unit definitions with pint UnitRegistry '''
        err = unitmgr.register_units(self.unit_defs)
        return err

    def set_defaults(self) -> None:
        ''' Restore all values to default '''
        self.plot_style = 'Suncal'
        self._settings.remove('rcparam')
        self.clear_color_cycle()
        self.colormap_contour = DFLTCMAP
        self.colormap_scatter = DFLTCMAP
        self.distributions = DFLTDISTS
        self.samples = 1000000
        self.sigfigs = 2
        self.numformat = 'auto'
        self.randomseed = None
        self.rptformat = 'html'
        self.rpt_imageformat = 'svg'
        self.rpt_mathformat = 'mpl'
        self.pandoc_path = None
        self.latex_path = None
        self.sync()

    @property
    def report_args(self):
        ''' Get arguments to pass to reports from settings dialog '''
        return {'n': self.sigfigs,
                'fmt': self.numformat}

    @property
    def mqa_performance(self) -> bool:
        ''' MQA performance/utility enabled '''
        return self._settings.value('mqa/performance', False, type=bool)

    @mqa_performance.setter
    def mqa_performance(self, value: bool) -> None:
        ''' MQA features to enable '''
        self._settings.setValue('mqa/performance', value)

    @property
    def mqa_cal(self) -> bool:
        ''' MQA Calibration column enabled '''
        return self._settings.value('mqa/calibration', False, type=bool)

    @mqa_cal.setter
    def mqa_cal(self, value: bool) -> None:
        self._settings.setValue('mqa/calibration', value)

    @property
    def mqa_cost(self) -> bool:
        ''' MQA Cost column enabled '''
        return self._settings.value('mqa/cost', False, type=bool)

    @mqa_cost.setter
    def mqa_cost(self, value: bool) -> None:
        self._settings.setValue('mqa/cost', value)

    @property
    def mqa_tur(self) -> bool:
        ''' MQA TUR column enabled '''
        return self._settings.value('mqa/tur', False, type=bool)

    @mqa_tur.setter
    def mqa_tur(self, value: bool) -> None:
        self._settings.setValue('mqa/tur', value)

    @property
    def mqa_tar(self) -> bool:
        ''' MQA TAR column enabled '''
        return self._settings.value('mqa/tar', True, type=bool)

    @mqa_tar.setter
    def mqa_tar(self, value: bool) -> None:
        self._settings.setValue('mqa/tar', value)

    @property
    def mqa_pfa(self) -> bool:
        ''' MQA PFA column enabled '''
        return self._settings.value('mqa/pfa', True, type=bool)

    @mqa_pfa.setter
    def mqa_pfa(self, value: bool) -> None:
        self._settings.setValue('mqa/pfa', value)

    @property
    def mqa_desc(self) -> bool:
        return self._settings.value('mqa/desc', False, type=bool)

    @mqa_desc.setter
    def mqa_desc(self, value: bool) -> None:
        self._settings.setValue('mqa/desc', value)


gui_settings = Settings()
