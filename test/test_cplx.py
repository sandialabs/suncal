''' Test complex number module '''
import numpy as np
import sympy

import suncal
from suncal import uparser


def test_E15():
    ''' Problem E15 in NIST-1900, Voltage Reflection Coefficient '''
    np.random.seed(2038942)
    u = suncal.UncertComplex('Gamma = S22 - S12*S23/S13', magphase=False, samples=100000)
    u.set_input_magph('S22', .24776, 4.88683, .00337, .01392, k=1, degrees=False)
    u.set_input_magph('S12', .49935, 4.78595, .00340, .00835, k=1, degrees=False)
    u.set_input_magph('S23', .24971, 4.85989, .00170, .00842, k=1, degrees=False)
    u.set_input_magph('S13', .49952, 4.79054, .00340, .00835, k=1, degrees=False)
    out = u.calculate()

    # Compare with results in NIST1900
    assert np.isclose(out.gum.nom()[0].magnitude, .0074, atol=.00005)   # Real part
    assert np.isclose(out.gum.uncert()[0].magnitude, .005, atol=.0005)   # Real part
    assert np.isclose(out.gum.nom()[1].magnitude, .0031, atol=.00005)   # Imag part
    assert np.isclose(out.gum.uncert()[1].magnitude, .0045, atol=.0005)  # Imag part

    # Monte Carlo
    assert np.isclose(out.mc.nom()[0].magnitude, .0074, atol=.0005)   # Real part
    assert np.isclose(out.mc.uncert()[0].magnitude, .005, atol=.0005)   # Real part
    assert np.isclose(out.mc.nom()[1].magnitude, .0031, atol=.0005)   # Imag part
    assert np.isclose(out.mc.uncert()[1].magnitude, .0045, atol=.0005)  # Imag part

    # Check correlation coefficient
    assert np.isclose(u.out.gum.correlation(), 0.0323, atol=.005)
    assert np.isclose(u.out.mc.correlation(), 0.0323, atol=.005)


def test_parse():
    ''' Test parsing expression with cplx '''
    expr = 'I * omega * C'
    assert sympy.Symbol('I') in uparser.parse_math(expr).free_symbols  # Without allowing on complex numbers, I is a symbol
    assert sympy.Symbol('I') not in uparser.parse_math(expr, allowcomplex=True).free_symbols  # I interpreted as sqrt(-1)
    assert sympy.I in uparser.parse_math(expr, allowcomplex=True).atoms(sympy.I)

    u = suncal.UncertComplex('f=a+b')
    u.set_input('a', nom=1+1j)
    u.set_input('b', nom=1+1j)
    u._build_ucalc()
    re, im = u.ucalc.model.get_baseexprs()

    ar, br = sympy.symbols('a_real b_real')
    assert (re - ar - br) == 0
    ai, bi = sympy.symbols('a_imag b_imag')
    assert (im - ai - bi) == 0
