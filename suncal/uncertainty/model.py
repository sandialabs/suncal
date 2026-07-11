''' Measurement model for uncertainty propagation

    Either set of Sympy expressions, or list of callable Python functions
'''

import warnings
import logging
import inspect
import numpy as np
import sympy
from scipy.optimize import fsolve
from pint import DimensionalityError

from ..common import uparser, matrix, unitmgr
from ..common.limit import Limit
from .variables import Variables
from .results.gum import GumResults, GumOutputData
from .results.monte import McResults
from .results.uncertainty import UncertaintyResults


# Will get div/0 errors, for example, with degrees of freedom in W-S formula
# Can safely ignore them and let the result be inf.
np.seterr(divide='ignore', invalid='ignore', over='ignore')


class ModelBase:
    ''' Generic measurement model class. Not used directly. '''
    def __init__(self):
        self.variables = Variables()
        self.descriptions = {}
        self.tolerances: dict[str, Limit] = {}

    def var(self, name):
        ''' Get a variable from the model

            Args:
                name (str): Name of the variable to get

            Returns:
                RandomVariable
        '''
        return self.variables.get(name)

    @property
    def varnames(self):
        ''' Get list of variable names defined by the model '''
        return self.variables.names

    def measure_correlated(self, **kwargs):
        ''' Measure variables with correlation

            Args:
                **kwargs: variable name, value-array pairs
        '''
        for varname, value in kwargs.items():
            self.variables.get(varname).measure(value)

        for i, (var1, value1) in enumerate(kwargs.items()):
            for j, (var2, value2) in enumerate(kwargs.items()):
                if i < j:
                    value1 = np.atleast_1d(value1)
                    value2 = np.atleast_1d(value2)
                    if len(value1) == len(value2):
                        cor = np.corrcoef(value1, value2)
                        self.variables.correlate(var1, var2, cor)
                    else:
                        raise ValueError('Correlated samples must have same length')


class Model(ModelBase):
    ''' Measurement model of made from string expression parsed by Sympy

        Args:
            *exprs (str or sympy): Model functions as strings or Sympy expressions.
                Strings must by sympify-able.
    '''
    def __init__(self, *exprs):
        super().__init__()
        self.raw_exprs = exprs
        self.H = []  # H's - sympy
        self.Y = []  # Y's - sympy
        self.implicit = []  # bools, each model is implicit?
        self.constants = {}  # Bracketed quantities in expression
        self.functions = {}  # Processed and simplified as functions (only if no implicit)
        self.implicit_guesses = []

        for expr in exprs:
            if isinstance(expr, sympy.Basic):
                solvefor = f'f{len(self.H)+1}'
                self.Y.append(sympy.Symbol(solvefor))
                self.H.append(expr - self.Y[-1])
                self.implicit.append(False)
                self.implicit_guesses.append(1)

            elif ';' in expr:
                expr, solvefor = expr.split(';')
                guess = 1
                if '~' in solvefor:
                    solvefor, guess = solvefor.split('~')
                    guess = float(guess)
                self.Y.append(sympy.Symbol(solvefor.strip()))
                if '=' not in expr:
                    raise ValueError(f'Invalid expression {expr}')
                part1, part2 = expr.split('=')
                part1, part2 = part1.strip(), part2.strip()
                expr = part1 if part1 != '0' else part2
                expr, consts = uparser.parse_math_with_quantities(expr, nconsts=len(self.constants))
                self.constants.update(consts)
                self.H.append(expr)
                self.implicit.append(True)
                self.implicit_guesses.append(guess)

            else:
                if '=' not in expr:
                    solvefor = f'f{len(self.Y)+1}'
                else:
                    solvefor, expr = expr.split('=')
                solvefor_symbol = sympy.Symbol(solvefor.strip())
                self.Y.append(solvefor_symbol)
                expr, consts = uparser.parse_math_with_quantities(expr, nconsts=len(self.constants))
                self.constants.update(consts)
                self.implicit.append(solvefor_symbol in expr.free_symbols)
                self.H.append(expr - solvefor_symbol)
                self.implicit_guesses.append(1)

        variables = []
        for exp in self.H:
            variables.extend([str(s) for s in exp.free_symbols])
        variables = set(variables)
        [variables.discard(str(y)) for y in self.Y]
        [variables.discard(c) for c in self.constants.keys()]
        self.varsymbols = list(variables)
        self.variables = Variables(*[str(v) for v in self.varsymbols])
        self.functionnames = [str(s) for s in self.Y]

        if not any(self.implicit):
            self.H = self._build_baseexprs()
            self.functions = {str(y): h+y for y, h in zip(self.Y, self.H)}
            self.exprs = [str(e) for e in self.functions.values()]  # Expressions as input into the model

    def _build_baseexprs(self):
        ''' Parse expressions into base variables only (substitute any chained dependencies
            in fucntion list.)
        '''
        baseexprs = {}
        funcs = [h+y for h, y in zip(self.H, self.Y)]  # Revert to y = f(x) format
        for name, exp in zip(self.functionnames, funcs):
            oldfunc = None
            count = 0
            while oldfunc != exp and count < 100:
                oldfunc = exp
                for vname in exp.free_symbols:
                    if str(vname) in self.functionnames:
                        exp = exp.subs(vname, funcs[self.functionnames.index(str(vname))])
                count += 1
            if count >= 100:
                # This shouldn't happen since implicit models are handled separately now.
                raise RecursionError('Circular reference in function set')
            baseexprs[name] = exp

        return [exp - sympy.Symbol(name) for name, exp in baseexprs.items()]  # Back to H(y, x) = 0

    def _degrees_freedom(self, Cx):
        ''' Get expressions for degrees of freedom. Uses Cx sensitivity matrix,
            already computed
        '''
        degfsymbols = [sympy.Symbol(f'nu_{x}') for x in self.variables.names]
        uncertsymbols = [sympy.Symbol(f'u_{x}') for x in self.variables.names]
        N = len(uncertsymbols)

        degf = {}
        for i, funcname in enumerate(self.functionnames):
            denom = [(uncertsymbols[j]*Cx[i][j])**4/degfsymbols[j] for j in range(N)]
            denom = sympy.Add(*denom)
            if denom == 0:
                degf[f'nu_{funcname}'] = np.inf
            else:
                degf[f'nu_{funcname}'] = sympy.Symbol(f'u_{funcname}')**4 / denom
        return degf

    def eval(self, values=None):
        ''' Evaluate the expression at the provided values, or the expected values
            if not provided

            Args:
                values (dict): Dictionary of variablename : value
        '''
        if values is None:
            values = self.variables.expected
        values.update(self.constants)

        out = {}
        for solvefor, expr, implicit, guess in zip(self.Y, self.H, self.implicit, self.implicit_guesses):
            strname = str(solvefor)
            if not implicit:
                expr = expr + solvefor  # Convert back from f(x)-y = 0 to y = f(x)
                out.update(matrix.eval_dict({strname: expr}, values))  # Uses cached lambidfy
            else:
                out[strname] = solve_implicit(expr, solvefor, values, guess=guess)  # Use fsolve
        return out

    def expected(self):
        ''' Calculate expected value of all functions in model '''
        return self.eval()

    def calculate_symbolic(self):
        ''' Run the calculation, symbolic

            Returns:
                GumOutputData containing sympy expression for results
        '''
        cy = sympy.Matrix([[sympy.diff(exp, y) for y in self.Y] for exp in self.H])
        cx = sympy.Matrix([[sympy.diff(exp, sympy.Symbol(x)) for x in self.varsymbols] for exp in self.H])
        if cy.is_Identity or (-cy).is_Identity:
            c = cx
        else:
            c = sympy.MatMul(sympy.Inverse(cy), cx).doit()

        ux = sympy.Matrix(self.variables.covariance_symbolic())

        if len(cx) > 0:
            uy = sympy.MatMul(sympy.MatMul(c, ux), sympy.Transpose(c)).doit()
            uncerts = {f'u_{name}': sympy.sqrt(x) for name, x in zip(self.functionnames, uy.diagonal())}

        else:
            uy = ux
            uncerts = {f'u_{name}': 0 for name in self.functionnames}
        degf = self._degrees_freedom(c.tolist())
        return GumOutputData(uncerts, uy.tolist(), ux.tolist(), cy.tolist(), cx.tolist(), c.tolist(),
                             degf, self.Y, None, self.H, self.implicit)

    def calculate_gum(self):
        ''' Run the GUM calculation

            Returns:
                GumResults instance
        '''
        expected = self.expected()
        symbolic = self.calculate_symbolic()
        subvalues = self.variables.symbol_values()
        subvalues.update(self.constants)
        subvalues.update(expected)
        uncerts = matrix.eval_dict(symbolic.uncertainty, subvalues)
        subvalues.update(expected)
        subvalues.update(uncerts)  # degf needs to sub these too

        ux_correlation = self.variables.correlation_matrix() if self.variables.has_correlation() else None
        Cy = matrix.eval_matrix(symbolic.Cy, subvalues)
        Cx = matrix.eval_matrix(symbolic.Cx, subvalues)
        C = matrix.eval_matrix(symbolic.C, subvalues)
        Ux = matrix.eval_matrix(symbolic.Ux, subvalues)
        Uy = matrix.eval_matrix(symbolic.Uy, subvalues)
        degf = matrix.eval_dict(symbolic.degf, subvalues)
        degf = {name: unitmgr.strip_units(df, reduce=True) for name, df in degf.items()}
        degf = {name: np.inf if np.isnan(value) else value for name, value in degf.items()}
        degf = dict(zip(self.functionnames, degf.values()))        # Rename to use funciton name instead of nu_XXX
        uncerts = dict(zip(self.functionnames, uncerts.values()))  # Rename to use funciton name instead of u_XXX

        warns = []
        if not all(all(np.isfinite(u) for u in k) for k in Uy):
            warns.append('Overflow in GUM uncertainty calculation')

        outnumeric = GumOutputData(uncerts, Uy, Ux, Cy, Cx, C, degf, expected, ux_correlation, self.H, self.implicit)
        return GumResults(outnumeric, symbolic, self.variables.info, self.constants, self.descriptions, warns, self.tolerances)

    def monte_carlo(self, samples=1000000, copula='gaussian'):
        ''' Calculate Monte Carlo samples

            Args:
                samples (int): number of random samples
                copula (str): 'gaussian' or 't'

            Returns:
                McResults instance
        '''
        samplevalues = self.variables.sample(samples, copula=copula)
        samplevalues.update(self.constants)

        functions = {self.functionnames[i]: self.H[i]+self.Y[i] for i in range(len(self.functionnames)) if not self.implicit[i]}
        values = matrix.eval_dict(functions, samplevalues)

        implicits = {self.functionnames[i]: self.H[i] for i in range(len(self.functionnames)) if self.implicit[i]}
        for i, (name, H) in enumerate(implicits.items()):
            values[name] = solve_implicit(H, sympy.Symbol(name), samplevalues, guess=self.implicit_guesses[i])

        # Ensure all values are arrays (in case function itself is a constant)
        values = {name: np.full(samples, v) if np.isscalar(v) else v for name, v in values.items()}
        samplevalues = {name: np.full(samples, v) if np.isscalar(v) else v for name, v in samplevalues.items()}

        warns = []
        for fname, value in values.items():
            if not all(np.isfinite(np.atleast_1d(np.float64(unitmgr.strip_units(value))))):
                warns.append(f'Some Monte-Carlo samples in {fname} are NaN. Ignoring in statistics.')

        return McResults(values, self.variables.info, samplevalues, self, self.descriptions, warns, self.tolerances)

    def calculate(self, samples=1000000):
        ''' Run GUM and Monte Carlo calculation and generate a report '''
        gumresults = self.calculate_gum()
        mcresults = self.monte_carlo(samples=samples)
        return UncertaintyResults(gumresults, mcresults)


def infer_units(expr, solvefor, values):
    ''' Infer output units of an implicit model '''
    eqs = sympy.solve(sympy.Eq(expr, 0), solvefor)
    for eq in eqs:
        fn = sympy.lambdify(values.keys(), eq)
        result = fn(**values)
        if np.isfinite(result):
            return unitmgr.split_units(result)[1]
    return None


def solve_implicit(expr, solvefor, values, units=None, guess=1):
    ''' Solve the implicit measurement model for the given variable

        Args:
            expr: Sympy expression to solve for roots of
            solvefor: Sympy symbol to solve for
            values: Dictionary of other variables in the expression
            units: Units of the solvefor variable, if known. If not known,
                it tries to automatically figure them out
            guess: Initial guess for fsolve
    '''
    strname = str(solvefor)

    # If we have one solution, no fsolve needed
    eqs = sympy.solve(sympy.Eq(expr, 0), solvefor)
    if len(eqs) == 1:
        return matrix.eval_dict({str(solvefor): eqs[0]}, values)[strname]

    # Can't solve symbolically. Need iterative fsolve solution.
    varnames = [strname]
    varnames += [str(e) for e in expr.free_symbols if str(e) != strname]
    func = matrix._lambdify(tuple(varnames), expr)

    # Attempt to determine units of solvefor variable
    hasunits = any(unitmgr.has_units(v) for v in values.values())
    N = 0
    if hasunits and units is None:
        infervals = {}
        for name, val in values.items():
            try:
                infervals[name] = val[0]
                N = len(val)
            except IndexError:
                infervals[name] = val

        units = infer_units(expr, solvefor, infervals)

    # Solve for the solvefor variable, stripping units because they don't work with scipy.fsolve
    magvalues = {name: unitmgr.strip_units(values.get(name)) for name in varnames[1:]}
    if N > 0:
        out = np.zeros(N)
        for i in range(N):
            singlesample = []
            for name in varnames[1:]:
                val = magvalues.get(name)
                try:
                    singlesample.append(val[i])
                except IndexError:
                    singlesample.append(val)
            out[i] = fsolve(func, guess, args=tuple(singlesample))[0]
    else:
        out = fsolve(func, guess, args=tuple(magvalues.values()))[0]

    # Restore units
    if hasunits:
        out = unitmgr.make_quantity(out, units)
    return out


class ModelCallable(Model):
    ''' Measurement model made from Python-callable function (N-outputs).
        Cannot solve symbolically.

        To process units, provide unitsin and unitsout parameters. Any units will be
        striped from values before sending to function, then replaced with
        units defined by unitsout.

        Args:
            function (callable): Callable Python function. For multi-output
              measurement models, function should return a tuple or namedtuple
            names (str): Names of the parameters returned by function
            unitsin (list of str): Units associated with each argument to function
            unitsout (list of str): Units expected from each output of function
        '''
    def __init__(self, function, names=None, argnames=None, unitsin=None, unitsout=None):
        # unitsin, unitsout should be ureg units (not string)
        super().__init__()
        self.function = function  # N-output function
        self.functionnames = names
        self.unitsin = unitsin
        self.unitsout = unitsout
        self.variables = None
        self.argnames = argnames
        self._callable_name = None
        self._extract_args()

    def _extract_args(self):
        ''' Extract arguments to the function call, and wrap function to process units '''
        if self.argnames is None:
            params = inspect.signature(self.function).parameters
            if any(p.kind != inspect.Parameter.POSITIONAL_OR_KEYWORD for p in params.values()):
                raise ValueError('Callable function uses keyword arguments. Please specify argnames parameter.')
            self.argnames = sorted(list(params.keys()))

        if hasattr(self.function, '__name__'):
            self._callable_name = self.function.__name__
        elif isinstance(self.function, np.vectorize):
            self._callable_name = self.function.pyfunc.__name__
        else:
            self._callable_name = 'f'   # Shouldn't get here?

        # Wrap function with in/out units if specified
        if self.unitsin and self.unitsout:
            if not isinstance(self.unitsin, (list, tuple)):
                raise TypeError('unitsin parameter must be list')
            if not isinstance(self.unitsout, (list, tuple)):
                raise TypeError('unitsout parameter must be list')

            self.unitsin = [unitmgr.parse_units(u) if isinstance(u, str) else u for u in self.unitsin]
            self.unitsout = [unitmgr.parse_units(u) if isinstance(u, str) else u for u in self.unitsout]

            self.unitsout = self.unitsout[0] if len(self.unitsout) == 1 else self.unitsout
            self.function = unitmgr.ureg.wraps(self.unitsout, self.unitsin)(self.function)

        self.variables = Variables(*self.argnames)

    def _extract_output_names(self):
        ''' Attempt to determine function return value names '''
        # By delaying this until AFTER inputs are defined, units can properly propagate through
        # the function call when determining output structure
        if self.functionnames is None:
            out = uparser.callf(self.function, self.variables.expected)
            try:
                if hasattr(out, '_fields'):
                    # Namedtuple, use named fields
                    self.functionnames = out._fields
                else:
                    # Non-named tuple. Use Python function name with subscript
                    self.functionnames = [f'{self._callable_name}_{i+1}' for i in range(len(out))]
            except TypeError:
                # Fall back on the Python function name by itself
                self.functionnames = [self._callable_name]

        if self.unitsout is None:
            self.unitsout = [None]*len(self.functionnames)
        if not isinstance(self.unitsout, (tuple, list)):
            self.unitsout = [self.unitsout]

    def eval(self, values=None):
        ''' Evaluate the functions at the values

            Args:
                values (dict): Dictionary of variablename : value

            Returns:
                Dictionary of functioname : value
        '''
        self._extract_output_names()
        if values is None:
            values = self.variables.expected

        out = uparser.callf(self.function, values)
        if len(self.functionnames) > 1:
            return dict(zip(self.functionnames, out))
        return {self.functionnames[0]: out}

    def _eval_vectorized(self, values):
        ''' Evaluate MC samples by vectorizing the model '''
        try:
            samples = self.eval(values)
        except DimensionalityError:
            # Hack around Pint bug/inconsistency (see https://github.com/hgrecco/pint/issues/670, closed
            #   without solution)
            #   with x = np.arange(5) * units.dimensionless
            #   np.exp(x) --> returns dimensionless array
            #   2**x --> raises DimensionalityError
            # Since units/dimensionality has already been verified, this fix strips units and adds them back.
            values = {k: unitmgr.strip_units(v) for k, v in values.items()}
            samples = self.eval(values)
        except (TypeError, ValueError):
            # Call might have failed if function is not vectorizable. Use numpy vectorize
            # to broadcast over array and try again.
            logging.info('Vectorizing function {}...'.format(str(self.function)))

            # Vectorize will strip units - see https://github.com/hgrecco/pint/issues/828.
            # First, run a single sample through the function to determine what units come out
            outsingle = uparser.callf(self.function, {k: v[0] for k, v in values.items()})
            mcoutunits = str(outsingle.units) if unitmgr.has_units(outsingle) else None

            # Then apply those units to whole array of sampled values.
            # vectorize() will issue a UnitStripped warning, but we're handling it outside Pint, so ignore it.
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                out = unitmgr.make_quantity(uparser.callf(np.vectorize(self.function), values), mcoutunits)
                if len(self.functionnames) > 1:
                    samples = dict(zip(self.functionnames, out))
                else:
                    samples = {self.functionnames[0]: out}

        # Ensure float datatype (some functions may return dtype=object arrays)
        samples = {name: arr.astype(float, copy=False) for name, arr in samples.items()}
        return samples

    def _sensitivity(self):
        ''' Sensitivity matrix [Cx] '''
        means = self.variables.expected
        uncerts = self.variables.uncertainties
        delta = 1E-6  # delta parameter for numeric derivative

        CxT = []
        for name in self.variables.names:
            d1 = means.copy()
            dx = uncerts[name] * delta
            if dx == 0:
                dx = 1E-6
                if unitmgr.has_units(uncerts[name]):
                    dx *= unitmgr.get_units(uncerts[name])
            d1[name] = d1[name] + dx
            d2 = means.copy()
            d2[name] = d2[name] - dx
            result1 = self.eval(d1)
            result2 = self.eval(d2)
            CxT.append([((result1[fname]-result2[fname])/(2*dx)) for fname in self.functionnames])
        return matrix.transpose(CxT)

    def _degrees_freedom(self, Uy, Ux, Cx):
        ''' Get expressions for degrees of freedom. Uses Cx, Ux, Uy, already computed '''
        variabledegfs = self.variables.degrees_freedom
        uncerts = [np.sqrt(Ux[i][i]) for i in range(len(Ux))]
        degf = {}
        for i, funcname in enumerate(self.functionnames):
            denom = sum((u*c)**4/v for u, c, v in zip(uncerts, Cx[i], variabledegfs.values()))
            df = (Uy[i][i]**2 / denom)
            degf[funcname] = df
        return degf

    def expected(self):
        ''' Calculate expected value of all functions in model '''
        return self.eval(self.variables.expected)

    def calculate_gum(self):
        ''' Run the calculation

            Returns:
                GumResults instance
        '''
        expected = self.expected()
        Cx = self._sensitivity()
        CxT = matrix.transpose(Cx)
        Ux = self.variables.covariance()
        ux_correlation = self.variables.correlation_matrix() if self.variables.has_correlation() else None
        Uy = matrix.matmul(matrix.matmul(Cx, Ux), CxT)
        uncerts = {name: np.sqrt(x) for name, x in zip(self.functionnames, matrix.diagonal(Uy))}
        uncerts = dict(zip(self.functionnames, uncerts.values()))  # Rename to use funciton name instead of u_XXX
        degf = self._degrees_freedom(Uy, Ux, Cx)
        degf = {name: unitmgr.strip_units(df, reduce=False) for name, df in zip(self.functionnames, degf.values())}

        warns = []
        if not all(all(np.isfinite(u) for u in k) for k in Uy):
            warns.append('Overflow in GUM uncertainty calculation')

        outnumeric = GumOutputData(uncerts, Uy, Ux, None, Cx, Cx, degf, expected, ux_correlation, None, None)
        return GumResults(outnumeric, None, self.variables.info, None, None, self.tolerances)

    def monte_carlo(self, samples=1000000, copula='gaussian'):
        ''' Calculate Monte Carlo samples

            Args:
                samples (int): number of random samples
                copula (str): 'gaussian' or 't'

            Returns:
                McResults instance
        '''
        samples = self.variables.sample(samples, copula=copula)
        values = self._eval_vectorized(samples)

        warns = []
        for fname, value in values.items():
            if not all(np.isfinite(np.atleast_1d(np.float64(unitmgr.strip_units(value))))):
                warns.append(f'Some Monte-Carlo samples in {fname} are NaN. Ignoring in statistics.')

        return McResults(values, self.variables.info, samples, self, warns, self.tolerances)
