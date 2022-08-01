'''
PSL Uncertainty Calculator - Sandia National Labs

Functions for evaluating a string expression into a value. Safe wrappers around sympy eval()
by analyzing the expression with the ast module.
'''

import ast
import numpy as np
import sympy

from . import unitmgr


# List of sympy functions allowed for converting strings to sympy
_functions = ['acos', 'asin', 'atan', 'atan2', 'cos', 'sin', 'tan',
              'log', 'cosh', 'sinh', 'tanh', 'atanh', 'acosh', 'asinh',
              'rad', 'deg', 'sqrt', 'exp', 'root',
              # Include functions we redefine
              'ln', 'log10', 'radian', 'degree', 'arccos', 'arcsin', 'arctan',
              ]

# Anything defined in sympy but NOT in _functions should be treated as a symbol, not a function
# for example, assume if the function string contains 'lex', we want a variable lex, not the LexOrder function.
_sympys = ['Symbol', 'Integer', 'Float', 'Rational', 'pi', 'E']  # But keep these as sympy objects, not symbols
_locals = dict((f, sympy.Symbol(f)) for f in dir(sympy) if '_' not in f and f not in _functions and f not in _sympys)

# Add some aliases for common stuff
_locals['inf'] = sympy.oo
_locals['e'] = sympy.E
_locals['ln'] = sympy.log
_locals['radian'] = sympy.rad
_locals['degree'] = sympy.deg
_locals['arccos'] = sympy.acos
_locals['arcsin'] = sympy.asin
_locals['arctan'] = sympy.atan

# And this one not defined as a single sympy func
_locals['log10'] = lambda x: sympy.log(x, 10)


def parse_unit(unitstr):
    ''' Parse the unit string and return a Pint unit instance '''
    if unitstr is None or unitstr.lower() == 'none':
        u = unitmgr.dimensionless
    else:
        try:
            u = unitmgr.parse_units(unitstr)
        except (ValueError, AttributeError, TypeError):
            raise ValueError('Cannot parse unit {}'.format(unitstr))
    return u


def parse_math(expr, name=None, allowcomplex=False, raiseonerr=True):
    ''' Parse the math expression string into a Sympy expression. Only basic
        math operations are allowed, such as 4-function math, exponents, and
        standard trigonometric and logarithm functions.

        Parameters
        ----------
        expr: string
            Expression to evaluate
        name: string (optional)
            Name of function to check that function is not self-recursive (e.g. f = 2*f)
        allowcomplex: bool
            Allow complex numbers (I) in expression
        raiseonerr: bool
            Raise exception on parse error. If False, None will be returned on error.

        Returns
        -------
        expr: Sympy expression
    '''
    if raiseonerr:
        return _parse_math(expr, name=name, allowcomplex=allowcomplex)

    else:
        try:
            expr = _parse_math(expr, name=name, allowcomplex=allowcomplex)
        except ValueError:
            expr = None
        return expr


def _parse_math(expr, fns=_functions, name=None, allowcomplex=False):
    ''' Parse the math expression and return Sympy expression if valid.
        Raise an error if not valid.

        Parameters
        ----------
        expr: string
            Expression to evaluate
        fns: string list, optional
            List of allowed functions in expression. Defaults to allowing basic trig
             and other functions listed in uparser._functions.
        name: string (optional)
            Name of function to check that function is not self-recursive (e.g. f = 2*f)
        allowcomplex: bool
            Allow complex numbers (I) in expression

        Notes
        -----
        Allows only 4-function math, exponents, and a few binary ops.
    '''
    if fns is None:
        fns = []
    allowed = [ast.Module, ast.Expr, ast.BinOp,
               ast.Name, ast.Num, ast.UnaryOp, ast.Load,
               ast.Add, ast.Mult, ast.Sub, ast.Div, ast.Pow,
               ast.USub, ast.UAdd, ast.Constant
               ]

    if not isinstance(expr, str):
        raise ValueError('Non string expression {}'.format(expr))

    expr = expr.replace('^', '**')  # Assume we want power, not bitwise XOR

    try:
        b = ast.parse(expr)
    except SyntaxError:
        raise ValueError('Invalid syntax in function: "{}"'.format(expr))

    for node in ast.walk(b):
        if type(node) == ast.Call:
            # Function call, must be in allowed list of functions
            if not hasattr(node.func, 'id') or node.func.id not in fns:
                raise ValueError('Invalid function call in "{}"'.format(expr))

        elif type(node) not in allowed:
            # Other operator. Must be in whitelist.
            raise ValueError('Invalid expression: "{}"'.format(expr))

    _locals['I'] = sympy.I if allowcomplex else sympy.Symbol('I')
    try:
        fn = sympy.sympify(expr, _locals)
    except (ValueError, TypeError, sympy.SympifyError):
        raise ValueError('Cannot sympify expression "{}"'.format(expr))

    if name and name in [str(i) for i in fn.free_symbols]:
        raise ValueError('Recursive function "{} = {}"'.format(name, expr))

    if not allowcomplex and sympy.I in fn.atoms(sympy.I):
        raise ValueError('Complex numbers not supported: "{} = {}"'.format(expr, str(fn)))

    return fn


def callf(func, vardict=None):
    ''' Call the function using variables defined in vardict dictionary. String will
        be validated before eval.

        Parameters
        ----------
        func:    string expression, python callable, or sympy expression
                 The function to evalueate.
        vardict: dictionary, optional
                 dictionary of arguments (name:value) to func

        Returns
        -------
        y:       output of function
    '''
    if vardict is None:
        vardict = {}

    if isinstance(func, str):
        # String expression. Convert to sympy.
        func = _parse_math(func)

    if isinstance(func, sympy.Basic):
        # Sympy expression. Try lambdify method with numpy so arrays can be computed element-wise.

        if func.has(sympy.zoo):
            # Stupid workaround if function contains a "complex infinity" (zoo) it won't lambidfy.
            # Shouldn't happen unless user enters something like "1/0" in input.
            # bug reported: https://github.com/sympy/sympy/issues/9439
            vardict['zoo'] = np.inf

        try:
            fn = sympy.lambdify(tuple(vardict.keys()), func, 'numpy')
            y = fn(**vardict)
        except (ZeroDivisionError, OverflowError):
            y = np.inf

    elif callable(func):
        # Python function. Just call it.  (NOTE: Put this after sympy. A sympy symbol is also callable!)
        y = func(**vardict)

    else:
        raise TypeError('Function {} is not callable'.format(func))
    return y
