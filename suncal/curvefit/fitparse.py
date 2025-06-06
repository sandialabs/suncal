''' Parse a curve fit expression to ensure it has an x variable and
    fit coefficients
'''

from collections import namedtuple
import numpy as np
import sympy

from ..common import uparser


def parse_fit_expr(expr, predictorvar: str = 'x'):
    ''' Check expr string for a valid curvefit function including an x variable
        and at least one fit parameter.

        Returns:
            func (callable): Lambdified function of expr
            symexpr (sympy): Sympy expression of function
            argnames (list of strings): Names of arguments (except x) to function
    '''
    uparser.parse_math(expr)  # Will raise if not valid expression
    symexpr = sympy.sympify(expr)
    argnames = sorted(str(s) for s in symexpr.free_symbols)
    if predictorvar not in argnames:
        raise ValueError(f'Expression must contain "{predictorvar}" variable.')
    argnames.remove(predictorvar)
    if len(argnames) == 0:
        raise ValueError('Expression must contain one or more parameters to fit.')
    # Make sure to specify 'numpy' so nans are returned instead of complex numbers
    func = sympy.lambdify([predictorvar] + argnames, symexpr, 'numpy')
    ParsedMath = namedtuple('ParsedMath', ['function', 'sympyexpr', 'argnames'])
    return ParsedMath(func, symexpr, argnames)


def fit_callable(model: str, polyorder: int = 2, predictor_var='x'):
    ''' Get fit callable and sympy expression for the function '''
    if model == 'line':
        expr = sympy.sympify('a + b*x')

        def func(x, b, a):
            return a + b*x

    elif model == 'exp':  # Full exponential
        expr = sympy.sympify('c + a * exp(x/b)')

        def func(x, a, b, c):
            return c + a * np.exp(x/b)

    elif model == 'decay':  # Exponential decay to zero (no c parameter)
        expr = sympy.sympify('a * exp(-x/b)')

        def func(x, a, b):
            return a * np.exp(-x/b)

    elif model == 'decay2':  # Exponential decay, using rate lambda rather than time constant tau
        expr = sympy.sympify('a * exp(-x*b)')

        def func(x, a, b):
            return a * np.exp(-x*b)

    elif model == 'log':
        expr = sympy.sympify('a + b * log(x-c)')

        def func(x, a, b, c):
            return a + b * np.log(x-c)

    elif model == 'logistic':
        expr = sympy.sympify('a / (1 + exp((x-c)/b)) + d')

        def func(x, a, b, c, d):
            return d + a / (1 + np.exp((x-c)/b))

    elif model == 'quad' or (model == 'poly' and polyorder == 2):
        expr = sympy.sympify('a + b*x + c*x**2')

        def func(x, a, b, c):
            return a + b*x + c*x*x

    elif model == 'cubic' or (model == 'poly' and polyorder == 3):
        expr = sympy.sympify('a + b*x + c*x**2 + d*x**3')

        def func(x, a, b, c, d):
            return a + b*x + c*x*x + d*x*x*x

    elif model == 'quartic' or (model == 'poly' and polyorder == 4):
        expr = sympy.sympify('a + b*x + c*x**2 + d*x**3 + f*x**4')

        def func(x, a, b, c, d, e):
            return a + b*x + c*x*x + d*x*x*x + e*x*x*x*x

    elif model == 'poly':
        def func(x, *p):
            return np.poly1d(p[::-1])(x)  # Coeffs go in reverse order (...e, d, c, b, a)

        polyorder = int(polyorder)
        if polyorder < 1 or polyorder > 12:
            raise ValueError('Polynomial order out of range')
        varnames = [chr(ord('a')+i) for i in range(polyorder+1)]
        expr = sympy.sympify('+'.join(f'{v}*x**{i}' for i, v in enumerate(varnames)))

    else:
        # actual expression as string
        func, expr, _ = parse_fit_expr(model, predictor_var)
    return func, expr
