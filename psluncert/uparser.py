'''
PSL Uncertainty Calculator - Sandia National Labs

Functions for evaluating a string expression into a value. Safe wrappers around sympy eval()
by analyzing the expression with the ast module.
'''

import ast
import numpy as np
import sympy


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
_locals = dict([(f, sympy.Symbol(f)) for f in dir(sympy) if '_' not in f and f not in _functions and f not in _sympys])

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


def check_expr(expr, fns=_functions, name=None):
    ''' Check the expression to ensure it is safe to eval/sympify. Only basic math operations are
        allowed. Raise exception if expr is invalid, otherwise returns a sympy expression.

        Parameters
        ----------
        expr: string
            Expression to evaluate
        fns: string list, optional
            List of allowed functions in expression. Defaults to allowing basic trig
             and other functions listed in uparser._functions.
        name: string (optional)
            Name of function to check that function is not self-recursive (e.g. f = 2*f)

        Notes
        -----
        Allows only 4-function math, exponents, and a few binary ops.
    '''
    if fns is None:
        fns = []
    allowed = [ast.Module, ast.Expr, ast.BinOp,
               ast.Name, ast.Num, ast.UnaryOp, ast.Load,
               ast.Add, ast.Mult, ast.Sub, ast.Div, ast.Pow,
               ast.USub, ast.UAdd
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

    try:
        fn = sympy.sympify(expr, _locals)
    except (ValueError, sympy.SympifyError):
        raise ValueError('Cannot sympify expression "{}"'.format(expr))

    if name and name in [str(i) for i in fn.free_symbols]:
        raise ValueError('Recursive function "{} = {}"'.format(name, expr))

    if sympy.I in fn.atoms(sympy.I):
        raise ValueError('Complex numbers not supported: "{} = {}"'.format(expr, str(fn)))

    return fn


def get_expr(expr):
    ''' Get expression, without raising. Return None if an error in parsing '''
    try:
        expr = check_expr(expr)
    except ValueError:
        expr = None
    return expr


def docalc(vardict, func):
    ''' Compute the output '''
    f = sympy.lambdify(tuple(vardict.keys()), func, 'numpy')
    return f(**vardict)


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
        func = check_expr(func)

    if hasattr(func, 'subs'):
        # Sympy expression. Try lambdify method with numpy so arrays can be computed element-wise.

        if func.has(sympy.zoo):
            # Stupid workaround if function contains a "complex infinity" (zoo) it won't lambidfy.
            # Shouldn't happen unless user enters something like "1/0" in input.
            # bug reported: https://github.com/sympy/sympy/issues/9439
            vardict['zoo'] = np.inf

        try:
            y = docalc(vardict, func)
        except (ZeroDivisionError, OverflowError):
            y = np.inf

    elif callable(func):
        # Python function. Just call it.  (NOTE: Put this after sympy. A sympy symbol is also callable!)
        y = func(**vardict)
        if isinstance(y, np.ndarray):
            # Weird case where func() returned an array with dtype=object. Make sure it's float64.
            y = y.astype(np.float64)

    else:
        raise TypeError('Function {} is not callable'.format(func))
    return y
