'''
Functions for evaluating a string expression into a value. Safe wrappers around sympy eval()
by analyzing the expression with the ast module.
'''

import ast
import re
from tokenize import TokenError
import numpy as np
import sympy
from pint import PintError

from . import unitmgr
from .style import latexchars


# List of sympy functions allowed for converting strings to sympy
_functions = ['acos', 'asin', 'atan', 'atan2', 'cos', 'sin', 'tan',
              'log', 'cosh', 'sinh', 'tanh', 'atanh', 'acosh', 'asinh',
              'rad', 'deg', 'sqrt', 'exp', 'root', 'coth', 'acoth',
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
        except (PintError, ValueError, AttributeError, TypeError, KeyError, TokenError) as exc:
            raise ValueError(f'Cannot parse unit {unitstr}') from exc
    return u


def parse_value(valuestr):
    ''' Parse the value + unit string and return a Pint Quantity '''
    try:
        q = unitmgr.parse_expression(valuestr)
    except (PintError, TokenError) as exc:
        raise ValueError(f'Cannot parse value {valuestr}') from exc
    return q


def parse_math(expr, name=None, allowcomplex=False, raiseonerr=True):
    ''' Parse the math expression string into a Sympy expression. Only basic
        math operations are allowed, such as 4-function math, exponents, and
        standard trigonometric and logarithm functions.

        Args:
            expr (string): Expression to evaluate
            name: string: Name of function to check that function is not
                self-recursive (e.g. f = 2*f)
            allowcomplex (bool): Allow complex numbers (I) in expression
            raiseonerr (bool): Raise exception on parse error. If False, None
                will be returned on error.

        Returns:
            expr: Sympy expression
    '''
    if raiseonerr:
        return _parse_math(expr, name=name, allowcomplex=allowcomplex)

    try:
        expr = _parse_math(expr, name=name, allowcomplex=allowcomplex)
    except (ValueError, AttributeError, TypeError, TokenError):
        expr = None
    return expr


def _replace_constants(expr, nconsts=0, raiseonerr=True):
    ''' Replace constant quantities in brackets, eg [360 deg],
        with a sympify-able variable.

        Args:
            expr (string): Math expression
            nconsts (int): Number of constants already replaced,
                eg in multi-output system
            raiseonerr (bool): Raise an exception if expr cannot be parsed

        Returns:
            expr (string): Sympify-able string expression
            consts (dict): Constant values name: Pint Quantities
    '''
    CONSTPREFIX = 'suncalconst'

    constants = {}  # name: Quantity value
    constexprs = re.findall(r'(\[.+?\])', expr)
    while True:
        cname = f'{CONSTPREFIX}{len(constants)+nconsts}'
        newexpr = re.sub(r'(\[.+?\])', cname, expr, count=1)
        if expr == newexpr:
            break

        if not raiseonerr:
            try:
                constants[cname] = unitmgr.ureg.parse_expression(constexprs[len(constants)])
            except PintError:
                return None, constants
        else:
            constants[cname] = unitmgr.ureg.parse_expression(constexprs[len(constants)])
        expr = newexpr
    return expr.strip(), constants


def parse_math_with_quantities(expr, name=None, nconsts=0, raiseonerr=True):
    ''' Parse math expression which may contain constant Quantites/units
        in brackets. Checks against allowed function list and for circular
        references (name as a variable in expr).

        Args:
            expr (string): Expression to evaluate
            name (string): Name of function to check that function is not
                self-recursive (e.g. f = 2*f)
            nconsts (int): Number of constants already defined in Model
                for multi-output measurements
            raiseonerr (bool): Raise exception on parse error. If False, None
                will be returned on error.

        Returns:
            expr: Sympy Expression
            consts (dict): Constant values name: Pint Quantities
    '''
    expr, constants = _replace_constants(expr, nconsts=nconsts)
    expr = parse_math(expr, name=name, raiseonerr=raiseonerr)
    return expr, constants


def parse_math_with_quantities_to_tex(expr):
    ''' Parse math expression containing constant Quantities, returning Latex.
        For display only - result is returned regardless of validity of expression.
     '''
    expr, constants = _replace_constants(expr)
    try:
        symexpr = sympy.sympify(expr, locals=_locals)
        tex = sympy.latex(symexpr)
    except (TypeError, sympy.SympifyError):
        tex = expr

    for name, value in constants.items():
        base, num, _ = re.split('([0-9].*)', name, maxsplit=1)
        tex = tex.replace(f'{base}_{{{num}}}', f'[{value:~P}]')
    tex = f'${tex}$'  # encode/decode looks for $ or it will add its own
    return tex.encode('ascii', 'latex').decode().strip('$')


def _parse_math(expr, fns=None, name=None, allowcomplex=False):
    ''' Parse the math expression and return Sympy expression if valid.
        Raise an error if not valid.

        Args:
            expr (string): Expression to evaluate
            fns (string list, optional): List of allowed functions in expression.
                Default allows basic trig and other functions listed in uparser._functions.
            name (string): Name of function to check that function is not
                self-recursive (e.g. f = 2*f)
            allowcomplex (bool): Allow complex numbers (I) in expression

        Notes:
            Default allows only 4-function math, exponents, and a few binary ops.
    '''
    if fns is None:
        fns = _functions
    allowed = (ast.Module, ast.Expr, ast.BinOp,
               ast.Name, ast.UnaryOp, ast.Load,
               ast.Add, ast.Mult, ast.Sub, ast.Div, ast.Pow,
               ast.USub, ast.UAdd, ast.Constant)

    if allowcomplex:
        fns = fns + ['re', 'im']

    if not isinstance(expr, str):
        raise ValueError(f'Non string expression {expr}')

    expr = expr.replace('^', '**')  # Assume we want power, not bitwise XOR

    try:
        b = ast.parse(expr)
    except SyntaxError as exc:
        raise ValueError(f'Invalid syntax in function: "{expr}"') from exc

    for node in ast.walk(b):
        if isinstance(node, ast.Call):
            # Function call, must be in allowed list of functions
            if not hasattr(node.func, 'id') or node.func.id not in fns:
                raise ValueError(f'Invalid function call in "{expr}"')

        elif not isinstance(node, allowed):
            # Other operator. Must be in whitelist.
            raise ValueError(f'Invalid expression: "{expr}"')

    if allowcomplex:
        _locals['I'] = sympy.I
        _locals['re'] = sympy.re
        _locals['im'] = sympy.im
    else:
        _locals['I'] = sympy.Symbol('I')
        _locals['re'] = sympy.Symbol('re')
        _locals['im'] = sympy.Symbol('im')

    try:
        fn = sympy.sympify(expr, _locals)
    except (ValueError, TypeError, sympy.SympifyError) as exc:
        raise ValueError(f'Cannot sympify expression "{expr}"') from exc

    if not hasattr(fn, 'free_symbols'):
        # Didn't sympify into an expression, possibly a function only (e.g. "sqrt")
        raise ValueError(f'Incomplete expression {expr}')

    if name and name in [str(i) for i in fn.free_symbols]:
        raise ValueError(f'Recursive function "{name} = {expr}"')

    if not allowcomplex and sympy.I in fn.atoms(sympy.I):
        raise ValueError(f'Complex numbers not supported: "{expr} = {fn}"')

    return fn


def callf(func, vardict=None):
    ''' Call the function using variables defined in vardict dictionary. String will
        be validated before eval.

        Args:
            func: string expression, python callable, or sympy expression to evaluate
            vardict: (dict): dictionary of arguments {name:value} to func

        Returns:
            y: output of function
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
        raise TypeError(f'Function {func} is not callable')
    return y
