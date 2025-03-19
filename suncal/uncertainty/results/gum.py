''' Results of GUM calculation, with some additional analysis functions '''

import re
from collections import namedtuple
import sympy

from ...common import ttable, unitmgr, reporter
from ...common.style import latexchars
from ..report.gum import ReportGum
from ..report.cplx import ReportComplexGum


Expanded = namedtuple('Expanded', ['uncertainty', 'k', 'confidence'])
GumOutputData = namedtuple('GumOutput', ['uncertainty', 'Uy', 'Ux', 'Cx', 'degf', 'expected', 'functions'])


@reporter.reporter(ReportGum)
class GumResults:
    ''' Results of GUM uncertainty calculation

        Attributes:
            uncertainty (dict): Computed standard uncertainties for each model function
            expected (dict): Computed expected/mean values for each model function
            Uy (list): Covariance matrix of model functions
            Ux (list): Covariance matrix of model input variables
            Cx (list): Sensitivity matrix
            degf (dict): Effective degrees of freedom for each model function
            functions (list): The model functions as Sympy expressions
            symbolic (tuple): Symbolic expressions for uncertainty, Uy, Ux, Cx, and degf
            variables (tuple): Information about the input variables and uncertainties
            constants (dict): Constant quantity names and values, from brackets in model expression
            warns (list): Any warnings generated during the calculation
            descriptions (dict): Descriptions of model functions
            tolerances (dict): Tolerances for each function
            p_conform (dict): Probability of conformance with tolerance
            report (Report): Generate formatted reports of the results

        Methods:
            units: Convert the units of uncertainty and expected
            expand: Expand the uncertainty of a single function
            expanded: Get dictionary of expanded uncertainties for all model functions
            expect: Get a single expected value
            proportions: Calculate the proportion of uncertainty coming from each input
            sensitivity: Calculate sensitivity coefficients
            covariance: Calculate covariance between model functions
            correlation: Calculation correlation between model functions
    '''
    def __init__(self, numeric: GumOutputData, symbolic: GumOutputData, variables,
                 constants, descriptions=None, warns=None, tolerances=None, poc=None):
        self.uncertainty = numeric.uncertainty
        self.expected = numeric.expected
        self.Uy = numeric.Uy
        self.Ux = numeric.Ux
        self.Cx = numeric.Cx
        self.degf = numeric.degf
        self.functions = numeric.functions

        self._numeric = numeric
        self.symbolic = symbolic
        self.variables = variables
        self.constants = constants
        self.warns = warns
        self.descriptions = {} if descriptions is None else descriptions
        self.tolerances = {} if tolerances is None else tolerances
        self._units = {}  # user-defined units

    def units(self, **units):
        ''' Convert units of uncertainty results

            Args:
                **units: functionname and unit string to convert each
                    model function result to
        '''
        self._units.update(units)
        self.expected = unitmgr.convert_dict(self.expected, self._units)
        self.uncertainty = unitmgr.convert_dict(self.uncertainty, self._units)
        return self

    def getunits(self):
        ''' Get the Pint units as currently configured '''
        return {fname: unitmgr.get_units(exp) for fname, exp in self.expected.items()}

    def prob_conform(self):
        ''' Get probability of conformance '''
        poc = {}
        for fname, tol in self.tolerances.items():
            poc[fname] = tol.probability_conformance(
                self.expected.get(fname),
                self.uncertainty.get(fname),
                self.degf.get(fname))
        return poc

    @property
    def functionnames(self):
        ''' List of function names in model '''
        return list(self.expected.keys())

    @property
    def function_uncert_names(self):
        ''' List of uncertainty names (ie "u_f") in model '''
        return list(self.uncertainty.keys())

    @property
    def variablenames(self):
        ''' List of variable names in the model '''
        return list(self.variables.expected.keys())

    @property
    def variable_uncert_names(self):
        ''' List of variable uncertainty names (ie "u_x") in model '''
        return list(self.variables.uncertainty.keys())

    def expanded(self, conf=0.95, k=None):
        ''' Expanded uncertainties

            Args:
                conf (float): Level of confidence in interval
                k (float): Coverage factor for interval, overrides conf

            Returns:
                Dictionary of of Expanded(uncerts, kvalues, confidence)
                for each output function.
        '''
        assert 0 < conf < 1
        expanded = {}
        for name in self.functionnames:
            if k is None:
                kmult = ttable.k_factor(conf, self.degf[name])
            else:
                kmult = k
                conf = ttable.confidence(k, self.degf[name])
            expanded[name] = Expanded(self.uncertainty[name]*kmult, kmult, conf)
        return expanded

    def expand(self, name=None, k=None, conf=0.95):
        ''' Get a single expanded uncertainty

            Args:
                name: Name of the function
                k: Coverage factor (overrides confidence arg)
                conf: Level of confidence to expand to

            Returns:
                Expanded uncertainty
        '''
        if name is None:
            name = self.functionnames[0]
        if k is None:
            k = ttable.k_factor(conf, self.degf[name])
        return self.uncertainty[name]*k

    def expect(self, name=None):
        ''' Get a single expected value

            Args:
                name: Name of the function

            Returns:
                Expanded uncertainty
        '''
        if name is None:
            name = self.functionnames[0]
        return self.expected[name]

    def proportions(self):
        ''' Calculate proportion of uncertainty coming from each variable

            Returns:
                dictionary of {functionname: {variablename: proportion}}
        '''
        funcprops = {}
        for i, funcname in enumerate(self.functionnames):
            funcuncert2 = self.uncertainty[funcname]**2
            props = {}
            for j, varname in enumerate(self.variablenames):
                varuncert = self.variables.uncertainty[varname]
                props[varname] = ((self.Cx[i][j]*varuncert)**2 / funcuncert2)
                props[varname] = unitmgr.strip_units(props[varname], reduce=True)  # unitless

            resid = 1 - sum(props.values())
            if abs(resid) < 1E-7:
                resid = 0
            props['residual'] = resid
            funcprops[funcname] = props
        return funcprops

    def sensitivity(self, symbolic=False):
        ''' Get sensitivity. This is Cx, but in a dictionary.

            Args:
                symbolic (bool): Return symbolic/sympy expressions

            Returns:
                dictionary of {functionname: {variablename: sensitivity}}
        '''
        Cx = self.symbolic.Cx if symbolic else self.Cx
        funcs = {}
        for i, funcname in enumerate(self.functionnames):
            sens = dict(zip(self.variablenames, Cx[i]))
            funcs[funcname] = sens
        return funcs

    def covariance(self):
        ''' Get covariance [Uy] as dictionary.

            Returns:
                dictionary of {functionname: {functionname: covariance}}
        '''
        cov = {}
        for i, func1 in enumerate(self.functionnames):
            cov1 = {}
            for j, func2 in enumerate(self.functionnames):
                cov1[func2] = self.Uy[i][j]
            cov[func1] = cov1
        return cov

    def correlation(self):
        ''' Get correlation as dictionary. Uy / (u1*u2)

            Returns:
                dictionary of {functionname: {functionname: correlation}}
        '''
        corr = {}
        for i, func1 in enumerate(self.functionnames):
            cor1 = {}
            for j, func2 in enumerate(self.functionnames):
                try:
                    cor1[func2] = unitmgr.strip_units(
                        self.Uy[i][j] /
                        (self.uncertainty[func1] * self.uncertainty[func2]),
                        reduce=True)
                except ZeroDivisionError:
                    cor1[func2] = 0.0  # No uncertainty = no correlation
            corr[func1] = cor1
        return corr

    def covariance_inputs(self, symbolic=False):
        ''' Get covariance of inputs, Ux, as dictionary

            Args:
                symbolic (bool): Return symbolic/sympy expressions

            Returns:
                dictionary of {variablename: {variablename: covariance}}
        '''
        Ux = self.symbolic.Ux if symbolic else self.Ux
        funcs = {}
        for i, funcname in enumerate(self.variablenames):
            sens = dict(zip(self.variablenames, Ux[i]))
            funcs[funcname] = sens
        return funcs

    def correlation_inputs(self):
        ''' Get correlation of inputs as dictionary

            Returns:
                dictionary of {variablename: {variablename: correlation}}
        '''
        corr = {}
        for i, var1 in enumerate(self.variablenames):
            cor1 = {}
            for j, var2 in enumerate(self.variablenames):
                try:
                    cor1[var2] = unitmgr.strip_units(
                        self.Ux[i][j] /
                        (self.variables.uncertainty[var1]*self.variables.uncertainty[var2]),
                        reduce=True)
                except ZeroDivisionError:
                    cor1[var2] = 0.0
            corr[var1] = cor1
        return corr

    def has_correlated_inputs(self):
        ''' Determine whether any inputs are correlated '''
        for i in range(len(self.variablenames)):
            for j in range(len(self.variablenames)):
                if i < j and self.Ux[i][j] != 0:
                    return True
        return False

    def latexify(self, expr):
        ''' Convert sympy expression to Latex, substituting back any
            constant bracket quantities
        '''
        tex = sympy.latex(expr)
        for name, value in self.constants.items():
            base, num, _ = re.split('([0-9].*)', name, maxsplit=1)
            tex = tex.replace(f'{base}_{{{num}}}', f'[{value:~P}]')
        tex = f'${tex}$'  # encode/decode looks for $ or it will add its own
        return tex.encode('ascii', 'latex').decode().strip('$')


@reporter.reporter(ReportComplexGum)
class GumResultsCplx(GumResults):
    ''' Results of Complex-valued GUM calculation '''
    def __init__(self, gumresults):
        super().__init__(gumresults._numeric, gumresults.symbolic, gumresults.variables, gumresults.warns)
        self._degrees = False
        self._gumresults = gumresults

    def degrees(self, degrees):
        self._degrees = degrees
        return self
