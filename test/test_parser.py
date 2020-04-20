''' Test cases for uparser.py.
    Usage: run py.test from root folder.
'''
import pytest
import os
import numpy

import suncal.uparser as uparser

def test_parse_math_ok():
    ''' Test parse_math. These should evaluate ok, no exception raised. '''
    uparser.parse_math('1+1')
    uparser._parse_math('2*6+cos(30)', fns=['cos'])
    uparser.parse_math('2**2')
    uparser.parse_math('(10+5)/3')
    uparser.parse_math('a+b')
    uparser.parse_math('sqrt(-1)*sqrt(-1)')  # Complex not supported, but this evaluates to real


def test_parse_math_fail():
    ''' Test parse_math. These should raise ValueError. '''
    with pytest.raises(ValueError):
        uparser.parse_math('import os')   # imports disabled

    with pytest.raises(ValueError):
        uparser.parse_math('print("ABC")')  # builtin functions disabled

    with pytest.raises(ValueError):
        uparser.parse_math('import(os)')  # Syntax error

    with pytest.raises(ValueError):
        uparser.parse_math('os.system("ls")')  # non-allowed function

    with pytest.raises(ValueError):
        uparser.parse_math('().__class__')     # Hack to get at base classes

    with pytest.raises(ValueError):
        uparser._parse_math('sin(pi)', fns=None)  # No fn list given, sin not allowed

    with pytest.raises(ValueError):
        uparser.parse_math('lambda x: x+1')   # Lambdas disabled

    with pytest.raises(ValueError):
        uparser.parse_math('def x(): pass')   # Function def disabled

    with pytest.raises(ValueError):
        uparser.parse_math('numpy.pi')  # Attributes disabled

    with pytest.raises(ValueError):
        uparser.parse_math('2*f', name='f')  # Name parameter, can't be recursive

    with pytest.raises(ValueError):
        uparser.parse_math('#a+b')  # comments are ok, but here there's no expression before it

    with pytest.raises(ValueError):
        uparser.parse_math('sqrt(-1)')  # Imaginary numbers not supported


def test_call():
    ''' Test callf function, verify results are same as plain math. '''
    assert uparser.callf('2+2') == 2+2
    assert uparser.callf('4**2') == 4**2
    assert uparser.callf('4^2') == 4**2  # Replacing ^ with ** for user
    assert uparser.callf('cos(pi)') == numpy.cos(numpy.pi)
    assert uparser.callf('exp(-1)') == numpy.exp(-1)
    assert uparser.callf('ln(exp(1))') == numpy.log(numpy.exp(1))
    assert uparser.callf('x + y', {'x':3, 'y':4}) == 7

    #with pytest.raises(TimeoutError):
    #    uparser.callf('x**x**x**x**x', {'x':9})


def test_callf_sympy():
    ''' Test callf with a sympy expression '''
    import sympy
    a, b = sympy.symbols('a b')
    f = (a+b)/2
    assert uparser.callf(f, {'a':10, 'b':6}) == (10+6)/2


def test_callf_callable():
    ''' Test callf with python function '''

    def myfunc(a,b):
        return (a+b)/2
    assert uparser.callf(myfunc, {'a':10, 'b':6}) == (10+6)/2

    with pytest.raises(TypeError):
        uparser.callf(numpy)  # Object that won't translate into function

