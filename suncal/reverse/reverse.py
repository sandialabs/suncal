''' Reverse uncertainty propagation class '''

import logging
from dataclasses import dataclass
import sympy
import numpy as np

from ..uncertainty import Model
from ..common import unitmgr, reporter
from .report.reverse import ReportReverse, ReportReverseGum, ReportReverseMc


@reporter.reporter(ReportReverseGum)
@dataclass
class ResultsReverseGum:
    solvefor: str
    solvefor_value: float
    u_solvefor: str
    u_solvefor_value: float
    u_solvefor_expr: str
    u_forward_expr: str
    function: str
    funcname: str
    u_fname: str
    f_required: float
    uf_required: float


@reporter.reporter(ReportReverseMc)
@dataclass
class ResultsReverseMc:
    solvefor: str
    solvefor_value: float
    u_solvefor_value: float
    function: str
    funcname: str
    f_required: float
    uf_required: float
    mcresults: object
    reverse_model: object


@reporter.reporter(ReportReverse)
@dataclass
class ResultsReverse:
    gum: ResultsReverseGum
    montecarlo: ResultsReverseMc


class ModelReverse:
    ''' Reverse uncertainty calculation model

        Args:
            *exprs (str): Function expressions for the model
            solvefor (str): Variable to solve for
            targetnom (float): Target nominal value for function output
            targetunc (float): Target uncertainty for function output
            funcname (str): Name of function (for multi-function models)
            targetunits (str): Units for target nominal value
    '''
    def __init__(self, *exprs, solvefor, targetnom=None, targetunc=None, funcname=None, targetunits=None):
        self.model = Model(*exprs)
        if funcname is None:
            funcname = self.model.functionnames[-1]
        self.reverseparams = {'solvefor': solvefor,
                              'targetnom': targetnom if targetnom else 1,
                              'targetunc': targetunc if targetunc else 1,
                              'targetunits': targetunits,
                              'funcname': funcname}
        self.descriptions = {}

    @property
    def variables(self):
        ''' Get variables object '''
        return self.model.variables

    def var(self, name):
        ''' Get variable from model '''
        return self.model.var(name)

    def eval(self, values=None):
        ''' Evaluate the model at the values dictionary '''
        return self.model.eval(values)

    def calculate_gum(self):
        ''' Calculate reverse uncertainty propagation, GUM method '''
        funcname = self.reverseparams.get('funcname', self.model.functionnames[-1])
        solvefor = self.reverseparams['solvefor']
        targetnom = self.reverseparams.get('targetnom', 0)
        targetunc = self.reverseparams.get('targetunc', 0)
        targetunits = self.reverseparams.get('targetunits', None)
        targetnom = unitmgr.make_quantity(targetnom, targetunits)
        targetunc = unitmgr.make_quantity(targetunc, targetunits)

        # Calculate GUM symbolically then solve for uncertainty component
        symout = self.model.calculate_symbolic()  # uncerts, self.sympyexprs, degf, Uy, Ux, Cx

        corrvals = {k: v for k, v in self.model.variables.correlation_coefficients.items() if v == 0}
        u_forward_expr = symout.uncertainty['u_'+funcname]  # Symbolic expression for combined uncertainty
        u_forward_expr = u_forward_expr.subs(corrvals).simplify()  # Remove 0 correlation values from expression

        # Solve function for variable of interest
        func_reversed = sympy.solve(sympy.Eq(sympy.Symbol(funcname), self.model.basesympys[funcname]),
                                    sympy.Symbol(solvefor))[0]

        u_solvefor = sympy.Symbol('u_'+solvefor)  # Symbol for unknown uncertainty we're solving for
        u_forward = sympy.Symbol('u_'+funcname)

        try:
            # Solve for u_i, keep positive solution
            u_solvefor_expr = sympy.solve(sympy.Eq(u_forward, u_forward_expr), u_solvefor)[1]
        except IndexError:
            # Will fail with no solution for model f = x due to sqrt(x**2) not simplifying.
            u_solvefor_expr = u_forward
        else:
            u_solvefor_expr = u_solvefor_expr.subs({solvefor: func_reversed})  # Replace var with func_reversed
        u_solvefor_value = u_solvefor_expr.subs(self.model.variables.expected)
        inpts = self.model.variables.symbol_values()
        originput = inpts.pop(solvefor)
        inpts.update({funcname: targetnom})
        solvefor_value = sympy.lambdify(inpts.keys(), func_reversed, 'numpy')(**inpts)

        # Plug everything in
        inpts.update({str(u_forward): targetunc})
        u_solvefor_value = sympy.lambdify(inpts.keys(), u_solvefor_value, 'numpy')(**inpts)
        if not np.isreal(unitmgr.strip_units(u_solvefor_value)) or not np.isreal(unitmgr.strip_units(u_solvefor_value)):
            logging.warning('No real solution for reverse calculation.')
            u_solvefor_value = None

        if unitmgr.has_units(originput):
            solvefor_value.ito(originput.units)
            u_solvefor_value.ito(originput.units)

        return ResultsReverseGum(
            solvefor,
            solvefor_value,
            u_solvefor,
            u_solvefor_value,
            u_solvefor_expr,
            u_forward_expr,
            self.model.basesympys[funcname],
            sympy.Symbol(funcname),
            sympy.Symbol(f'u_{funcname}'),
            targetnom,
            targetunc)

    def monte_carlo(self, samples=1000000):
        ''' Calculate reverse uncertainty using Monte Carlo method. '''
        # Must account for correlation between f and input variables

        funcname = self.reverseparams.get('funcname', self.model.functionnames[-1])
        solvefor = self.reverseparams['solvefor']
        targetnom = self.reverseparams['targetnom']
        targetunc = self.reverseparams['targetunc']
        targetunits = self.reverseparams['targetunits']
        targetnom = unitmgr.make_quantity(targetnom, targetunits)
        targetunc = unitmgr.make_quantity(targetunc, targetunits)

        solvefor_expr = sympy.solve(sympy.Eq(sympy.Symbol(funcname), self.model.basesympys[funcname]),
                                    sympy.Symbol(solvefor))[0]
        revmodel = Model(f'{solvefor} = {solvefor_expr}')

        for origvarname in self.model.variables.names:
            if origvarname == solvefor:
                continue
            origunc = self.model.variables.uncertainties[origvarname]
            revmodel.var(origvarname).measure(self.model.variables.expected[origvarname])
            if np.isfinite(unitmgr.strip_units(origunc)) and origunc != 0:
                revmodel.var(origvarname).typeb(std=origunc)

        revmodel.var(funcname).measure(targetnom).typeb(std=targetunc)

        # Correlate variables: see GUM C.3.6 NOTE 3 - Estimate correlation from partials
        gumsensitivity = self.model.calculate_gum().sensitivity(symbolic=True)[funcname]

        inpts = []
        for vname in self.model.variables.names:
            if str(vname) == solvefor:
                continue
            part = gumsensitivity[vname]  # Cx from gum calculation
            inpts = self.model.variables.symbol_values()
            inpts.update({funcname: targetnom})
            ci = sympy.lambdify(inpts.keys(), part.subs({solvefor: solvefor_expr}), 'numpy')(**inpts)
            corr = unitmgr.strip_units(
                    (revmodel.var(str(vname)).uncertainty /
                     revmodel.var(funcname).uncertainty * ci))  # dimensionless
            if np.isfinite(corr):
                revmodel.variables.correlate(str(vname), funcname, corr)

        # Include existing correlations between inputs
        if self.model.variables.has_correlation():
            for v1 in self.model.variables.names:
                for v2 in self.model.variables.names:
                    if v1 == v2 or solvefor in (v1, v2):
                        continue
                    revmodel.variables.correlate(v1, v2, self.model.variables.get_correlation_coeff(v1, v2))

        mcresults = revmodel.monte_carlo(samples=samples)

        originput = inpts.pop(solvefor)
        solvefor_value = mcresults.expected[solvefor]
        u_solvefor_value = mcresults.uncertainty[solvefor]
        if unitmgr.has_units(originput):
            solvefor_value.ito(originput.units)
            u_solvefor_value.ito(originput.units)

        result = ResultsReverseMc(
            solvefor,
            solvefor_value,
            u_solvefor_value,
            self.model.basesympys[funcname],
            sympy.Symbol(funcname),
            targetnom,
            targetunc,
            mcresults,
            revmodel)

        mcsamples = unitmgr.strip_units(mcresults.samples[solvefor])
        if np.count_nonzero(np.isfinite(list(mcsamples))) < samples * .95:
            # less than 95% of trials resulted in real number, consider this a no-solution
            result.u_solvefor_value = None
            result.solvefor_value = None
        return result

    def calculate(self, mc=True, samples=1000000):
        ''' Calculate both GUM and Monte Carlo methods and return a ReportReverse '''
        gumresults = self.calculate_gum()
        mcresults = None
        if mc:
            mcresults = self.monte_carlo(samples=samples)
        return ResultsReverse(gumresults, mcresults)
