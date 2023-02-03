''' Test complex number module '''
import numpy as np
import sympy

from suncal.uncertainty import ModelComplex
from suncal.common import uparser


def test_E15():
    ''' Problem E15 in NIST-1900, Voltage Reflection Coefficient '''
    np.random.seed(2038942)
    u = ModelComplex('Gamma = S22 - S12*S23/S13', magphase=False)
    u.var('S22').measure_magphase(.24776, 4.88683, .00337, .01392, k=1, degrees=False)
    u.var('S12').measure_magphase(.49935, 4.78595, .00340, .00835, k=1, degrees=False)
    u.var('S23').measure_magphase(.24971, 4.85989, .00170, .00842, k=1, degrees=False)
    u.var('S13').measure_magphase(.49952, 4.79054, .00340, .00835, k=1, degrees=False)
    gum = u.calculate_gum()
    mc = u.monte_carlo(samples=100000)

    # Compare with results in NIST1900
    assert np.isclose(gum.expected['Gamma_real'], .0074, atol=.00005)   # Real part
    assert np.isclose(gum.uncertainty['Gamma_real'], .005, atol=.0005)   # Real part
    assert np.isclose(gum.expected['Gamma_imag'], .0031, atol=.00005)   # Imag part
    assert np.isclose(gum.uncertainty['Gamma_imag'], .0045, atol=.0005)  # Imag part

    # Monte Carlo
    assert np.isclose(mc.expected['Gamma_real'], .0074, atol=.0005)   # Real part
    assert np.isclose(mc.uncertainty['Gamma_real'], .005, atol=.0005)   # Real part
    assert np.isclose(mc.expected['Gamma_imag'], .0031, atol=.0005)   # Imag part
    assert np.isclose(mc.uncertainty['Gamma_imag'], .0045, atol=.0005)  # Imag part

    # Check correlation coefficient
    assert np.isclose(gum.correlation()['Gamma_real']['Gamma_imag'], 0.0323, atol=.005)
    assert np.isclose(mc.correlation()['Gamma_real']['Gamma_imag'], 0.0323, atol=.005)


def test_parse():
    ''' Test parsing expression with cplx '''
    expr = 'I * omega * C'

    # Without allowing on complex numbers, I is a symbol
    assert sympy.Symbol('I') in uparser.parse_math(expr).free_symbols

    # I interpreted as sqrt(-1)
    assert sympy.Symbol('I') not in uparser.parse_math(expr, allowcomplex=True).free_symbols

    assert sympy.I in uparser.parse_math(expr, allowcomplex=True).atoms(sympy.I)

    u = ModelComplex('f=a+b')
    u.var('a').measure(1+1j, uncertainty=0)
    u.var('b').measure(1+1j, uncertainty=0)
    u._build_model_sympy()
    re, im = u.model.sympys

    ar, br = sympy.symbols('a_real b_real')
    assert (re - ar - br) == 0
    ai, bi = sympy.symbols('a_imag b_imag')
    assert (im - ai - bi) == 0
