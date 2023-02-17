''' Results of Monte Carlo calculation, with some additional analysis functions '''

from collections import namedtuple
import numpy as np

from ...common import unitmgr, reporter
from ..report.monte import ReportMonteCarlo
from ..report.cplx import ReportComplexMc

Expanded = namedtuple('Expanded', ['low', 'high', 'k', 'confidence'])


@reporter.reporter(ReportMonteCarlo)
class McResults:
    ''' Results of Monte Carlo uncertainty calculation

        Attributes:
            uncertainty (dict): Computed standard uncertainties for each model function
            expected (dict): Computed expected/mean values for each model function
            samples (dict): Random samples calculated for each model function
            varsamples (dict): Random samples generated for each input variable
            variables (tuple): Information about the input variables and uncertainties
            warns (list): Any warnings generated during the calculation
            descriptions (dict): Descriptions of model functions
            report (Report): Generate formatted reports of the results

        Methods:
            units: Convert the units of uncertainty and expected
            expand: Expand the uncertainty of a single function
            expanded: Get dictionary of expanded uncertainties for all model functions
            expect: Get a single expected value
            sensitivity: Calculate sensitivity coefficients and proportions
            correlation: Calculation correlation between model functions
    '''
    def __init__(self, functionsamples, variables, variablesamples, model, descriptions=None, warns=None):
        self.samples = functionsamples
        self.varsamples = variablesamples
        self.variables = variables
        self._model = model  # needed to run sensitivity
        self.warns = warns

        self.samples = {name: s[np.isfinite(s)] for name, s in self.samples.items()}  # Strip NANs
        self.expected = {name: np.nanmean(s) for name, s in self.samples.items()}
        self.uncertainty = {name: np.std(s, ddof=1) for name, s in self.samples.items()}
        self.descriptions = {} if descriptions is None else descriptions
        self._units = {}

    def units(self, **units):
        ''' Convert units of uncertainty results

            Args:
                **units: functionnames and unit string to convert each
                    model function result to
        '''
        self._units.update(units)
        self.expected = unitmgr.convert_dict(self.expected, self._units)
        self.uncertainty = unitmgr.convert_dict(self.uncertainty, self._units)
        self.samples = unitmgr.convert_dict(self.samples, self._units)
        return self

    def getunits(self):
        ''' Get the Pint units as currently configured '''
        return {fname: unitmgr.split_units(exp)[1] for fname, exp in self.expected.items()}

    @property
    def functionnames(self):
        ''' List of function names in model '''
        return list(self.expected.keys())

    @property
    def function_uncert_names(self):
        ''' List of uncertainty names (ie "u_f") in model '''
        return list(self.samples.keys())

    @property
    def variablenames(self):
        ''' List of variable names in the model '''
        return list(self.variables.expected.keys())

    @property
    def variable_uncert_names(self):
        ''' List of variable uncertainty names (ie "u_x") in model '''
        return list(self.variables.uncertainty.keys())

    def expect(self, name=None):
        ''' Get a single expected value

            Args:
                name: Name of the function
                k: Coverage factor (overrides confidence arg)
                conf: Level of confidence to expand to

            Returns:
                Expanded uncertainty
        '''
        if name is None:
            name = self.functionnames[0]
        return self.expected[name]

    def expand(self, name=None, shortest=False, conf=0.95):
        ''' Get a single expanded uncertainty

            Args:
                name: Name of the function
                shortest (bool): Use shortest interval instead of symmetric interval
                conf: Level of confidence to expand to

            Returns:
                Expanded uncertainty
        '''
        if name is None:
            name = self.functionnames[0]

        if shortest:
            # Find shortest interval by looping
            y = np.sort(self.samples[name])
            quant = int(conf*len(y))  # number of points in coverage range
            rmin = y[-1]-y[0]
            ridx = 0  # index of shortest found interval
            for rstart in range(len(y)-quant):
                if y[rstart+quant] - y[rstart] < rmin:  # This interval is shorter than previous ones
                    rmin = y[rstart+quant] - y[rstart]
                    ridx = rstart
            low = y[ridx]
            high = y[ridx+quant]
            k = unitmgr.strip_units((high-low) / (2*self.uncertainty[name]), reduce=True)
        else:
            quantiles = 100*(1-conf)/2, 100-100*(1-conf)/2
            low, high = np.nanpercentile(self.samples[name], quantiles)
            k = unitmgr.strip_units((high-low) / (2*self.uncertainty[name]), reduce=True)

        low = unitmgr.convert(low, self._units.get(name))
        high = unitmgr.convert(high, self._units.get(name))
        return Expanded(low, high, k, conf)

    def expanded(self, conf=0.95, shortest=False):
        ''' Expanded uncertainties

            Args:
                conf (float): Level of confidence in expanded interval
                shortest (bool): Use shortest interval instead of symmetric interval

            Returns:
                Expanded dictionaries
         '''
        assert 0 < conf < 1
        expanded = {}
        for fname in self.functionnames:
            expanded[fname] = self.expand(fname, conf=conf, shortest=shortest)
        return expanded

    def sensitivity(self):
        ''' Calculate sensitivity and proportions by sampling one variable at a time

            Returns:
                Tuple of (sensitivity, proportions) dictionaries
        '''
        # Set all sample arrays to the nominal
        # NOTE: don't use np.full here because it strips units
        ones = np.ones(len(self.samples[self.functionnames[0]]))
        variable_nom = {name: ones*x for name, x in self.variables.expected.items()}

        sensitivities = {name: {} for name in self.functionnames}
        proportions = {name: {} for name in self.functionnames}
        for varname in self.variablenames:
            samples = variable_nom.copy()              # Start with nominal values for all vars
            samples[varname] = self.varsamples[varname]  # And set this one to its real samples
            outsamples = self._model.eval(samples)    # Run it through the model
            stds = {name: x.std(ddof=1) for name, x in outsamples.items()}
            sens = {name: x/self.variables.uncertainty[varname] for name, x in stds.items()}
            prop = {name: (x/self.uncertainty[name])**2 for name, x in stds.items()}

            # {funcname: sens} -> for this varname  (Transposed)
            for funcname, value in sens.items():
                sensitivities[funcname][varname] = value

            for funcname, value in prop.items():
                proportions[funcname][varname] = unitmgr.strip_units(value, reduce=True)  # unitless

        McSensitivity = namedtuple('McSensitivity', ['sensitivity', 'proportions'])
        return McSensitivity(sensitivities, proportions)

    def correlation(self):
        ''' Calculate correlation coefficient between outputs

            Returns:
                Dictionary of {functionname: {functionname: correlation}}
        '''
        corrs = {}
        for i, func1 in enumerate(self.functionnames):
            cor1 = {}
            for j, func2 in enumerate(self.functionnames):
                if i == j:
                    cor1[func2] = 1.0
                elif i > j:
                    cor1[func2] = corrs[func2][func1]
                else:
                    cor1[func2] = np.corrcoef(unitmgr.strip_units(self.samples[func1]),
                                              unitmgr.strip_units(self.samples[func2]))[0, 1]
            corrs[func1] = cor1
        return corrs


@reporter.reporter(ReportComplexMc)
class McResultsCplx(McResults):
    ''' Results of Complex-valued GUM calculation '''
    def __init__(self, mcresults):
        super().__init__(mcresults.samples, mcresults.variables, mcresults.varsamples, mcresults._model, mcresults.warns)
        self._degrees = False
        self._mcresults = mcresults

    def degrees(self, degrees):
        self._degrees = degrees
        return self
