''' Measurement System Curve Fit Calculations '''
from typing import Sequence, Optional
from contextlib import suppress
import numpy as np
import sympy

from ..common import unitmgr, uparser
from ..common.limit import Limit
from ..sweep import UncertSweep
from ..curvefit import CurveFit, Array
from ..curvefit.fitparse import fit_callable
from ..uncertainty import Model
from .meas_result import SystemQuantityResult


class SystemCurve:
    ''' Fit curve in a Measurement System '''
    def __init__(self):
        self.data: list[tuple[str, Optional[Sequence[float] | str]]] = []   # (column-name, values|expr)
        self.fitmodel: str = 'line'
        self.fitexpr = None
        self.polyorder: int = 2
        self.predictor_var: str = 'x'
        self.response_var: str = 'y'
        self.odr: bool = False
        self.guess: list[float] = None
        self.tolerances: dict[str, Limit] = {}
        self.predictions: dict[str, float] = {}
        self.units: dict[str, str] = {}
        self.description = ''
        self.descriptions: dict[str, str] = {}  # for coefficients and predictions
        self.set_fitmodel('line')

    @property
    def symbol(self) -> str:
        ''' Get symbol for reports '''
        return self.fitmodel

    def set_fitmodel(self, model: str):
        ''' Set the curve fit function '''
        self.fitmodel = model  # "line", "polynomial", etc., or expression of x "a*x + b"
        _, self.fitexpr = fit_callable(self.fitmodel, self.polyorder)

    @property
    def rows(self) -> int:
        ''' Get minimum number of rows '''
        if len(self.data) == 0:
            return 0
        return min(len(data[1]) for data in self.data if not isinstance(data[1], str))

    def get_column(self, name):
        ''' Get the first column from data with this name '''
        for colname, dat in self.data:
            if name == colname:
                return dat
        return None

    def fill_column(self, name: str, value: float):
        ''' Fill a column with the value '''
        values = np.full(self.rows, value)
        self.data.append((name, values))

    def data_names(self) -> list[str]:
        ''' Get name of data input arrays '''
        names = []
        for name, _ in self.data:
            names.append(name)
        return list(set(names))

    def data_names_direct(self) -> list[str]:
        ''' Get data input arrays that are direct measurements '''
        names = self.data_names()
        exprs = [uparser.parse_math(n, raiseonerr=False) for n in names if not n.startswith('u(')]
        return [str(expr) for expr in exprs if expr and expr.is_symbol]

    def coeff_names(self) -> list[str]:
        ''' Get names of fit coefficients '''
        if self.fitexpr is None:
            return []
        argnames = sorted(str(s) for s in self.fitexpr.free_symbols)
        with suppress(ValueError):
            argnames.remove(self.predictor_var)
        return argnames

    def eval_data(self):
        ''' Calculate values and uncertainties of data columns '''
        columnnames = [d[0] for d in self.data]

        means = {}
        uncerts = {}

        # Start with Type A and Averages of multiple runs
        for name, value in self.data:
            if not name.startswith('u('):
                if columnnames.count(name) > 1 and not isinstance(value, str):
                    data = np.array([d[1] for d in self.data if d[0] == name])
                    means[name] = data.mean(axis=0)
                    uncerts[name] = data.std(ddof=1, axis=0)
                else:
                    means[name] = value

        # Now add Type B
        columnnames = means.keys()
        for name, value in self.data:
            if name.startswith('u('):
                name = name[2:-1]  # u(...)
                if name in uncerts:
                    # Have a Type A already
                    uncerts[name] = np.sqrt(uncerts[name]**2 + value**2)
                else:
                    uncerts[name] = value

        # Now calculate GUM expressions
        for name, expr in self.data:
            if isinstance(expr, str):
                # Expression, calculate it
                model = UncertSweep(Model(expr))
                for varname in model.varnames:
                    varuncert = uncerts.get(varname)
                    if varname in means:
                        model.var(varname).measure(means[varname])
                        model.add_sweep_nom(varname, means[varname])
                        if varuncert is not None and np.any(varuncert > 0):
                            model.var(varname).typeb(name=f'u({varname})', std=varuncert[0])
                            model.add_sweep_unc(varname, varuncert)
                gumresult = model.calculate_gum()
                means[name] = gumresult.expected()['f1']
                uncerts[name] = gumresult.uncertainties()['f1']
        return means, uncerts

    def fit(self) -> 'CurveFitResultsCombined':
        ''' Calculate curve fit coefficients '''
        means, uncerts = self.eval_data()
        x = means.get(self.predictor_var)
        y = means.get(self.response_var)
        if x is None:
            raise ValueError('Predictor (x) variable not defined')
        if y is None:
            raise ValueError('Response (y) variable not defined')
        if len(x) != len(y):
            raise ValueError('Curve Fit X and Y arrays must be same length')

        ux = uncerts.get(self.predictor_var)
        uy = uncerts.get(self.response_var)
        ux = np.full(len(x), 0) if ux is None else ux
        uy = np.full(len(y), 0) if uy is None else uy

        arr = Array(x, y, ux=ux, uy=uy)
        fit = CurveFit(
            arr,
            self.fitmodel,
            polyorder=self.polyorder,
            p0=self.guess,
            odr=self.odr,
            predictor_var=self.predictor_var)
        fit.predictions = self.predictions
        fit.tolerances = self.tolerances
        fit.xname = self.predictor_var
        fit.yname = self.response_var
        return fit.calculate_all(montecarlo=False, markov=False, gum=False)

    def calculate(self) -> list[SystemQuantityResult]:
        ''' Calculate curve results '''
        self.infer_units()
        fitresult = self.fit()
        fitlsq = fitresult.lsq
        quantities: list[SystemQuantityResult] = []
        for name, value, uncert in zip(
                fitlsq.setup.coeffnames,
                fitlsq.coeffs,
                fitlsq.uncerts):

            tol = self.tolerances.get(name)
            poc = tol.probability_conformance(value, uncert, fitlsq.degf) if tol else None
            units = self.units.get(name)
            quantities.append(SystemQuantityResult(
                symbol=name,
                value=unitmgr.make_quantity(value, units),
                uncertainty=unitmgr.make_quantity(uncert, units),
                units=units,
                degrees_freedom=fitlsq.degf,
                tolerance=tol,
                p_conformance=poc,
                qty=self,
                meta={'fitresult': fitresult}
            ))

        for name, (xvalue, tol) in self.predictions.items():
            value = fitlsq.y(xvalue)
            uncert = fitlsq.confidence_band(xvalue, k=1)  # Standard
            poc = tol.probability_conformance(value, uncert, fitlsq.degf) if tol else None
            units = self.units.get(self.response_var)
            quantities.append(SystemQuantityResult(
                symbol=name,
                value=unitmgr.make_quantity(value, units),
                uncertainty=unitmgr.make_quantity(uncert, units),
                units=units,
                degrees_freedom=fitlsq.degf,
                tolerance=tol,
                p_conformance=poc,
                qty=self,
                meta={'fitresult': fitresult,
                      'predictor': xvalue}
            ))
        return quantities

    def infer_units(self) -> dict[str, str]:
        ''' Infer units of fit coefficients from x and y units '''
        yunits = self.units.get(self.response_var)
        xunits = self.units.get(self.predictor_var)

        if not xunits or not yunits:
            # No units defined, can't predict anything
            return {}

        # Sympify the equation
        eq = self.fitexpr
        variables = [str(v) for v in list(eq.free_symbols)] + [self.response_var]

        # Subvals is what we currently know about units
        subvals = {v: 1 for v in variables}
        subvals[self.response_var] = unitmgr.make_quantity(1, yunits)
        subvals[self.predictor_var] = unitmgr.make_quantity(1, xunits)

        # Variables that are still unknown
        unknownvars = [k for k in subvals.keys() if k not in [self.predictor_var, self.response_var]]

        # Start with looking at functions in the expression. Whatever is
        # inside sin(...), exp(...), etc. must be unitless.
        funcs = list(eq.atoms(sympy.Function))
        funcs = sorted(funcs, key=lambda x: len(x.atoms(sympy.Function)))  # Go in order of complexity/nesting
        for func in funcs:
            for arg in func.args:
                for varname in unknownvars:
                    try:
                        v = sympy.solve(sympy.Eq(arg, 1), varname)
                    except NotImplementedError:  # sympy can't do it
                        continue
                    if v:
                        fn = sympy.lambdify(variables, v[0])
                        try:
                            subvals[varname] = fn(**subvals)  # Solve for the unit
                        except unitmgr.pint.PintError:
                            pass
                        else:
                            unknownvars.remove(varname)

        # Each term in a summation must have same dimension as response_var
        y = sympy.symbols(self.response_var)
        terms = [t[0] for t in eq.as_terms()[0]]
        terms = [sympy.Eq(t, y) for t in terms]
        step = 0
        while unknownvars:
            varname = unknownvars[0]
            for term in terms:
                if varname in [str(s) for s in term.free_symbols]:
                    v = sympy.solve(term, varname)
                    if v:
                        fn = sympy.lambdify(variables, v[0])
                        try:
                            subvals[varname] = fn(**subvals)
                        except unitmgr.pint.PintError:
                            pass
                        else:
                            unknownvars.remove(varname)
                            break
            step += 1
            if step > 4*len(subvals):  # Don't get stuck in endless loop
                print('Could not determine units of', unknownvars)
                break
            unknownvars = unknownvars[1:] + unknownvars[:1]  # Shift to next one

        self.units = {name: unitmgr.get_units(q) for name, q in subvals.items()}
        return self.units
