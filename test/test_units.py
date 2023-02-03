''' Test cases for unit conversions '''
import pytest

import numpy as np

from suncal import Model
from suncal.project import ProjectUncert
from suncal.common import report, uparser, unitmgr

ureg = unitmgr.ureg


def test_units():
    # Basic units propagation
    np.random.seed(12345)
    u = Model('P = J*V')
    u.var('J').measure(4, units='V').typeb(unc=.04, k=2, units='V')
    u.var('V').measure(20, units='mA').typeb(name='u(typeA)', std=.1, k=2).typeb(name='u(typeB)', std=.15, k=2)
    result = u.calculate().units(P='mW')
    assert str(result.gum.expected['P'].units) == 'milliwatt'
    assert np.isclose(result.gum.expected['P'].magnitude, 80)
    assert str(result.montecarlo.expected['P'].units) == 'milliwatt'
    assert np.isclose(result.montecarlo.expected['P'].magnitude, 80)
    assert 'mW' in str(result.report.gum.summary())
    assert 'mW' in str(result.report.gum.expanded())
    assert 'mW' in str(result.report.montecarlo.summary())
    assert 'mW' in str(result.report.montecarlo.expanded())

    # Change output to microwatts and recalculate
    result = u.calculate().units(P='uW')
    assert str(result.gum.expected['P'].units) == 'microwatt'
    assert np.isclose(result.gum.expected['P'].magnitude, 80000, rtol=.001)
    assert str(result.montecarlo.expected['P'].units) == 'microwatt'
    assert np.isclose(result.montecarlo.expected['P'].magnitude, 80000, rtol=.001)


def test_multifunc():
    ''' Test multiple functions in UncertCalc with different units '''
    # Start without units -- convert all inputs to base units and *1000 to get milliwatt
    np.random.seed(398232)
    u1 = Model('P = J*V*1000', 'R = V/J')
    u1.var('V').measure(10).typeb(std=.5)
    u1.var('J').measure(5).typeb(name='u_A', std=.05)  # 50 mA
    u1.var('J').typeb(name='u_B', std=.01)  # 10 mA = 10000 uA
    result1 = u1.calculate()

    meanP = result1.gum.expected['P']
    uncertP = result1.gum.uncertainty['P']
    meanR = result1.gum.expected['R']
    uncertR = result1.gum.uncertainty['R']

    # Now with units specified instead of converting first
    np.random.seed(398232)
    u = Model('P = J*V', 'R = V/J')
    u.var('V').measure(10, units='volts').typeb(std=.5, units='volts')
    u.var('J').measure(5, units='ampere')
    u.var('J').typeb(name='u_A', std=50, units='mA')    # Uncert not same units as variable
    u.var('J').typeb(name='u_B', std=10000, units='uA')
    result2 = u.calculate().units(P='mW', R='ohm')

    # And compare.
    assert np.isclose(result2.gum.expected['P'].magnitude, meanP)
    assert np.isclose(result2.gum.uncertainty['P'].magnitude, uncertP)
    assert str(result2.gum.expected['P'].units) == 'milliwatt'
    assert np.isclose(result2.montecarlo.expected['P'].magnitude, meanP, rtol=.0001)
    assert np.isclose(result2.montecarlo.uncertainty['P'].magnitude, uncertP, rtol=.001)
    assert str(result2.montecarlo.expected['P'].units) == 'milliwatt'

    assert np.isclose(result2.montecarlo.expected['R'].magnitude, meanR, rtol=.0001)
    assert np.isclose(result2.montecarlo.uncertainty['R'].magnitude, uncertR, rtol=.001)
    assert str(result2.montecarlo.expected['R'].units) == 'ohm'
    assert np.isclose(result2.montecarlo.expected['R'].magnitude, meanR, rtol=.0001)
    assert np.isclose(result2.montecarlo.uncertainty['R'].magnitude, uncertR, rtol=.001)
    assert str(result2.montecarlo.expected['R'].units) == 'ohm'


def test_load():
    ''' Load end-gauge problem WITH units '''
    proj = ProjectUncert.from_configfile('test/ex_endgauge_units.yaml')
    np.random.seed(345345)
    result = proj.calculate()
    assert np.isclose(result.gum.uncertainty['f_0'].magnitude, 32, atol=.2)
    assert str(result.gum.uncertainty['f_0'].units) == 'nanometer'
    assert np.isclose(result.montecarlo.uncertainty['f_0'].magnitude, 34, atol=.4)
    assert str(result.montecarlo.uncertainty['f_0'].units) == 'nanometer'


def test_parse():
    ''' Test parsing units, wrapper function '''
    assert uparser.parse_unit('meter') == ureg.meter
    assert uparser.parse_unit('m/s^2') == ureg.meter/ureg.second**2
    assert uparser.parse_unit(None) == ureg.dimensionless
    assert uparser.parse_unit('') == ureg.dimensionless


def test_print():
    ''' output module special handling for unit quantities '''
    assert report.Unit(ureg.dimensionless).plaintext() == ''
    assert report.Unit(ureg.meter).plaintext() == 'm'
    assert report.Unit(ureg.millivolt).plaintext() == 'mV'
    assert report.Unit(ureg.cm,).plaintext(abbr=False) == 'centimeter'
    assert report.Unit(ureg.cm).plaintext() == 'cm'
    assert report.Unit(ureg.cm).latex() == r'$\mathrm{cm}$'
    assert report.Unit(ureg.cm**2).latex() == r'$\mathrm{cm}^{2}$'
    assert report.Number(1*ureg.cm).string() == '1.0 cm'
    assert report.Number(1*ureg.cm).string(abbr=False) == '1.0 centimeter'


def test_power():
    ''' Test case for powers of dimensionless quantities '''
    # See https://github.com/hgrecco/pint/issues/670
    # Make sure we have a workaround since this inconsistency was closed without a fix
    #   with x = np.arange(5) * ureg.dimensionless
    #   np.exp(x) --> returns dimensionless array
    #   2**x --> raises DimensionalityError
    u = Model('f = 2**x')
    np.random.seed(8833293)
    u.var('x').measure(4).typeb(std=.1)  # No units / dimensionless
    result = u.calculate_gum()
    resultmc = u.monte_carlo()
    assert not unitmgr.has_units(result.uncertainty['f'])
    assert np.isclose(resultmc.expected['f'], 16.0, rtol=.01)
    assert np.isclose(resultmc.uncertainty['f'], result.uncertainty['f'], rtol=.02)


def test_welch():
    ''' Test welch-satterthwaite with units '''
    proj = ProjectUncert.from_configfile('test/ex_xrf.yaml')
    # XRF problem with Yu units in nm, others in um
    result = proj.calculate()
    assert np.isclose(result.gum.degf['Y_c'], 27.5, atol=.1)
    assert np.isclose(result.gum.expanded()['Y_c'].k, 2.05, atol=.01)
