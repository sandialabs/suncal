''' Test cases for unit conversions '''
import pytest

import numpy as np

import suncal as uc
from suncal import uncertainty
from suncal import curvefit
from suncal import report
from suncal import uparser

ureg = uncertainty.ureg


def test_units():
    # Basic units propagation
    u = uc.UncertCalc('J*V', units='mW', seed=12345)
    u.set_input('J', nom=4, unc=.04, k=2, units='V')
    u.set_input('V', nom=20, units='mA')
    u.set_uncert('V', name='u(typeA)', std=.1, k=2)
    u.set_uncert('V', name='u(typeB)', std=.15, k=2)
    u.calculate()
    outgum = u.out.get_output(method='gum')
    outmc = u.out.get_output(method='mc')
    assert str(outgum.units) == 'milliwatt'
    assert np.isclose(outgum.mean.magnitude, 80)
    assert str(outgum.mean.units) == 'milliwatt'
    assert str(outmc.units) == 'milliwatt'
    assert np.isclose(outmc.mean.magnitude, 80)
    assert str(outmc.mean.units) == 'milliwatt'
    assert 'mW' in str(outgum.report())
    assert 'mW' in str(outgum.report_expanded())
    assert 'mW' in str(outmc.report())
    assert 'mW' in str(outmc.report_expanded())

    # Change output to microwatts and recalculate
    u.functions[0].outunits = 'uW'
    u.calculate()
    outgum = u.out.get_output(method='gum')
    outmc = u.out.get_output(method='mc')
    assert str(outgum.units) == 'microwatt'
    assert np.isclose(outgum.mean.magnitude, 80000)
    assert str(outgum.mean.units) == 'microwatt'
    assert str(outmc.units) == 'microwatt'
    assert np.isclose(outmc.mean.magnitude, 80000)
    assert str(outmc.mean.units) == 'microwatt'


def test_multifunc():
    ''' Test multiple functions in UncertCalc with different units '''
    # Start without units -- convert all inputs to base units and *1000 to get milliwatt
    u1 = uc.UncertCalc(['P = J*V*1000', 'R = V/J'], seed=398232)
    u1.set_input('V', nom=10, std=.5)
    u1.set_input('J', nom=5)
    u1.set_uncert('J', name='u_A', std=.05)  # 50 mA
    u1.set_uncert('J', name='u_B', std=.01)  # 1 mA = 10000 uA
    u1.calculate()
    meanP = u1.out.get_output(fname='P', method='gum').mean.magnitude
    uncertP = u1.out.get_output(fname='P', method='gum').uncert.magnitude
    meanR = u1.out.get_output(fname='R', method='gum').mean.magnitude
    uncertR = u1.out.get_output(fname='R', method='gum').uncert.magnitude

    # Now with units specified instead of converting first
    u = uc.UncertCalc(['P = J*V', 'R = V/J'], units=['mW', 'ohm'], seed=398232)
    u.set_input('V', nom=10, std=.5, units='V')
    u.set_input('J', nom=5, units='ampere')
    u.set_uncert('J', name='u_A', std=50, units='mA')   # Uncert not same units as variable
    u.set_uncert('J', name='u_B', std=10000, units='uA')
    u.calculate()

    # And compare.
    outgumP = u.out.get_output(method='gum', fname='P')
    outmcP = u.out.get_output(method='mc', fname='P')
    assert np.isclose(outgumP.mean.magnitude, meanP)
    assert np.isclose(outgumP.uncert.magnitude, uncertP)
    assert str(outgumP.mean.units) == 'milliwatt'
    assert np.isclose(outmcP.mean.magnitude, meanP, rtol=.0001)
    assert np.isclose(outmcP.uncert.magnitude, uncertP, rtol=.001)
    assert str(outmcP.mean.units) == 'milliwatt'

    outgumR = u.out.get_output(method='gum', fname='R')
    outmcR = u.out.get_output(method='mc', fname='R')
    assert np.isclose(outgumR.mean.magnitude, meanR)
    assert np.isclose(outgumR.uncert.magnitude, uncertR)
    assert str(outgumR.mean.units) == 'ohm'
    assert np.isclose(outmcR.mean.magnitude, meanR, rtol=.0001)
    assert np.isclose(outmcR.uncert.magnitude, uncertR, rtol=.001)
    assert str(outmcR.mean.units) == 'ohm'


def test_load():
    ''' Load end-gauge problem WITH units '''
    u = uc.UncertCalc.from_configfile('test/ex_endgauge_units.yaml')
    u.seed = 8888
    u.calculate()
    GUM = u.out.get_output(method='gum')
    MC = u.out.get_output(method='mc')
    assert np.isclose(GUM.uncert.magnitude, 32, atol=.1)
    assert str(GUM.uncert.units) == 'nanometer'
    assert np.isclose(MC.uncert.magnitude, 34, atol=.2)
    assert str(MC.uncert.units) == 'nanometer'


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
    u = uc.UncertCalc('f = 2**x')
    u.seed = 8833293
    u.set_input('x', nom=4, std=.1)  # No units / dimensionless
    u.calculate()

    mc = u.out.get_output(method='mc')
    gum = u.out.get_output(method='gum')
    assert mc.uncert.units == ureg.dimensionless
    assert np.isclose(mc.mean.magnitude, 16.0, rtol=.01)
    assert np.isclose(mc.uncert.magnitude, gum.uncert.magnitude, rtol=.02)