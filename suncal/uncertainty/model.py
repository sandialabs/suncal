''' Measurement model for uncertainty propagation

    Either set of Sympy expressions, or list of callable Python functions
'''

import warnings
import logging
import inspect
import numpy as np
import sympy
from pint import DimensionalityError

from ..common import uparser, matrix, unitmgr
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
        self.exprs = []
        self.sympys = []
        self.functionnames = []
        self.constants = {}  # Bracketed quantities in expression
        for expr in exprs:
            if isinstance(expr, sympy.Basic):
                name = f'f{len(self.exprs)+1}'
                self.sympys.append(expr)
                self.exprs.append(str(expr))
            else:
                if '=' in expr:
                    name, expr = expr.split('=')
                    name = name.strip()
                    expr = expr.strip()
                else:
                    name = f'f{len(self.exprs)+1}'

                symexprs, consts = uparser.parse_math_with_quantities(expr, name=name, nconsts=len(self.constants))
                self.constants.update(consts)
                self.sympys.append(symexprs)

                self.exprs.append(expr.strip())
            self.functionnames.append(name.strip())

        varnames, self.basesympys = self._build_baseexprs()

        for constname in self.constants.keys():
            varnames.remove(constname)

        self.variables = Variables(*varnames)

    def _build_baseexprs(self):
        ''' Parse expressions into base variables only (substitute any chained dependencies
            in fucntion list.)
        '''
        baseexprs = {}
        varnames = []
        for name, exp in zip(self.functionnames, self.sympys):
            oldfunc = None
            count = 0
            while oldfunc != exp and count < 100:
                oldfunc = exp
                for vname in exp.free_symbols:
                    if str(vname) in self.functionnames:
                        exp = exp.subs(vname, self.sympys[self.functionnames.index(str(vname))])
                count += 1
            if count >= 100:
                raise RecursionError('Circular reference in function set')
            baseexprs[name] = exp
            varnames.extend([str(s) for s in exp.free_symbols if str(s) not in self.functionnames])

        # varnames will be alpha sorted for sympy models, but not callables
        varnames = sorted(list(set(varnames)))
        return varnames, baseexprs

    def _sensitivity(self):
        ''' Sensitivity matrix (Cx), See GUM 6.2.1.3 '''
        Cx = []
        for exp in self.basesympys.values():
            Cx_row = []
            for var in self.varnames:
                Cx_row.append(sympy.Derivative(exp, sympy.Symbol(var), evaluate=True).simplify())
            Cx.append(Cx_row)
        return Cx

    def _degrees_freedom(self, Cx):
        ''' Get expressions for degrees of freedom. Uses Cx sensitivity matrix,
            already computed
        '''
        degfsymbols = [sympy.Symbol(f'nu_{x}') for x in self.variables.names]
        uncertsymbols = [sympy.Symbol(f'u_{x}') for x in self.variables.names]

        degf = {}
        for i, funcname in enumerate(self.functionnames):
            denom = [(u*c)**4/v for u, c, v in zip(uncertsymbols, Cx[i], degfsymbols)]
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
        return matrix.eval_dict(self.basesympys, values)

    def expected(self):
        ''' Calculate expected value of all functions in model '''
        return self.eval()

    def calculate_symbolic(self):
        ''' Run the calculation, symbolic

            Returns:
                GumOutputData containing sympy expression for results
        '''
        Cx = self._sensitivity()
        CxT = matrix.transpose(Cx)
        Ux = self.variables.covariance_symbolic()
        if len(Cx[0]) > 0:
            Uy = matrix.matmul(matrix.matmul(Cx, Ux), CxT)
        else:  # No variables in model
            Uy = Ux
        uncerts = {f'u_{name}': sympy.sqrt(x) for name, x in zip(self.functionnames, matrix.diagonal(Uy))}
        degf = self._degrees_freedom(Cx)
        return GumOutputData(uncerts, Uy, Ux, Cx, degf, self.basesympys, self.sympys)

    def calculate_gum(self):
        ''' Run the GUM calculation

            Returns:
                GumResults instance
        '''
        symbolic = self.calculate_symbolic()
        subvalues = self.variables.symbol_values()
        subvalues.update(self.constants)
        expected = matrix.eval_dict(symbolic.expected, subvalues)
        uncerts = matrix.eval_dict(symbolic.uncertainty, subvalues)
        subvalues.update(expected)
        subvalues.update(uncerts)  # degf needs to sub these too

        Cx = matrix.eval_matrix(symbolic.Cx, subvalues)
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

        outnumeric = GumOutputData(uncerts, Uy, Ux, Cx, degf, expected, self.sympys)
        return GumResults(outnumeric, symbolic, self.variables.info,  self.constants, self.descriptions, warns)

    def monte_carlo(self, samples=1000000, copula='gaussian'):
        ''' Calculate Monte Carlo samples

            Args:
                samples (int): number of random samples
                copula (str): 'gaussian' or 't'

            Returns:
                McResults instance
        '''
        samples = self.variables.sample(samples, copula=copula)
        samples.update(self.constants)
        values = matrix.eval_dict(self.basesympys, samples)

        warns = []
        for fname, value in values.items():
            if not all(np.isfinite(np.atleast_1d(np.float64(unitmgr.strip_units(value))))):
                warns.append(f'Some Monte-Carlo samples in {fname} are NaN. Ignoring in statistics.')

        return McResults(values, self.variables.info, samples, self, self.descriptions, warns)

    def calculate(self, samples=1000000):
        ''' Run GUM and Monte Carlo calculation and generate a report '''
        gumresults = self.calculate_gum()
        mcresults = self.monte_carlo(samples=samples)
        return UncertaintyResults(gumresults, mcresults)


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
                    dx *= unitmgr.split_units(uncerts[name])[1]
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
        Uy = matrix.matmul(matrix.matmul(Cx, Ux), CxT)
        uncerts = {name: np.sqrt(x) for name, x in zip(self.functionnames, matrix.diagonal(Uy))}
        uncerts = dict(zip(self.functionnames, uncerts.values()))  # Rename to use funciton name instead of u_XXX
        degf = self._degrees_freedom(Uy, Ux, Cx)
        degf = {name: unitmgr.strip_units(df, reduce=False) for name, df in zip(self.functionnames, degf.values())}

        warns = []
        if not all(all(np.isfinite(u) for u in k) for k in Uy):
            warns.append('Overflow in GUM uncertainty calculation')

        outnumeric = GumOutputData(uncerts, Uy, Ux, Cx, degf, expected, None)
        return GumResults(outnumeric, None, self.variables.info, None, None)

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

        return McResults(values, self.variables.info, samples, self, warns)
