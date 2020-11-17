''' Test cases for unit conversions '''
import pytest

import numpy as np

import suncal as uc
from suncal import uncertainty
from suncal import curvefit
from suncal import report
from suncal import uparser
from suncal import unitmgr

ureg = unitmgr.ureg


def test_units():
    # Basic units propagation
    u = uc.UncertCalc('J*V', units='mW', seed=12345)
    u.set_input('J', nom=4, unc=.04, k=2, units='V')
    u.set_input('V', nom=20, units='mA')
    u.set_uncert('V', name='u(typeA)', std=.1, k=2)
    u.set_uncert('V', name='u(typeB)', std=.15, k=2)
    u.calculate()
    assert str(u.out.gum._units[0]) == 'milliwatt'
    assert np.isclose(u.out.gum.nom().magnitude, 80)
    assert str(u.out.mc._units[0]) == 'milliwatt'
    assert np.isclose(u.out.mc.nom().magnitude, 80)
    assert str(u.out.mc.nom().units) == 'milliwatt'
    assert 'mW' in str(u.out.gum.report())
    assert 'mW' in str(u.out.gum.report_expanded())
    assert 'mW' in str(u.out.mc.report())
    assert 'mW' in str(u.out.mc.report_expanded())

    # Change output to microwatts and recalculate
    u.model.outunits = ['uW']
    u.calculate()
    assert str(u.out.gum._units[0]) == 'microwatt'
    assert np.isclose(u.out.gum.nom().magnitude, 80000)
    assert str(u.out.gum.nom().units) == 'microwatt'
    assert str(u.out.mc._units[0]) == 'microwatt'
    assert np.isclose(u.out.mc.nom().magnitude, 80000)
    assert str(u.out.mc.nom().units) == 'microwatt'


def test_multifunc():
    ''' Test multiple functions in UncertCalc with different units '''
    # Start without units -- convert all inputs to base units and *1000 to get milliwatt
    u1 = uc.UncertCalc(['P = J*V*1000', 'R = V/J'], seed=398232)
    u1.set_input('V', nom=10, std=.5)
    u1.set_input('J', nom=5)
    u1.set_uncert('J', name='u_A', std=.05)  # 50 mA
    u1.set_uncert('J', name='u_B', std=.01)  # 1 mA = 10000 uA
    u1.calculate()
    meanP = u1.out.gum.nom('P').magnitude
    uncertP = u1.out.gum.uncert('P').magnitude
    meanR = u1.out.gum.nom('R').magnitude
    uncertR = u1.out.gum.uncert('R').magnitude

    # Now with units specified instead of converting first
    u = uc.UncertCalc(['P = J*V', 'R = V/J'], units=['mW', 'ohm'], seed=398232)
    u.set_input('V', nom=10, std=.5, units='V')
    u.set_input('J', nom=5, units='ampere')
    u.set_uncert('J', name='u_A', std=50, units='mA')   # Uncert not same units as variable
    u.set_uncert('J', name='u_B', std=10000, units='uA')
    u.calculate()

    # And compare.
    assert np.isclose(u.out.gum.nom('P').magnitude, meanP)
    assert np.isclose(u.out.gum.uncert('P').magnitude, uncertP)
    assert str(u.out.gum.nom('P').units) == 'milliwatt'
    assert np.isclose(u.out.mc.nom('P').magnitude, meanP, rtol=.0001)
    assert np.isclose(u.out.mc.uncert('P').magnitude, uncertP, rtol=.001)
    assert str(u.out.mc.nom('P').units) == 'milliwatt'

    assert np.isclose(u.out.mc.nom('R').magnitude, meanR, rtol=.0001)
    assert np.isclose(u.out.mc.uncert('R').magnitude, uncertR, rtol=.001)
    assert str(u.out.mc.nom('R').units) == 'ohm'
    assert np.isclose(u.out.mc.nom('R').magnitude, meanR, rtol=.0001)
    assert np.isclose(u.out.mc.uncert('R').magnitude, uncertR, rtol=.001)
    assert str(u.out.mc.nom('R').units) == 'ohm'


def test_load():
    ''' Load end-gauge problem WITH units '''
    u = uc.UncertCalc.from_configfile('test/ex_endgauge_units.yaml')
    u.seed = 8888
    u.calculate()
    assert np.isclose(u.out.gum.uncert().magnitude, 32, atol=.1)
    assert str(u.out.gum.uncert().units) == 'nanometer'
    assert np.isclose(u.out.mc.uncert().magnitude, 34, atol=.2)
    assert str(u.out.mc.uncert().units) == 'nanometer'


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

    assert u.out.mc.uncert().units == ureg.dimensionless
    assert np.isclose(u.out.mc.nom().magnitude, 16.0, rtol=.01)
    assert np.isclose(u.out.mc.uncert().magnitude, u.out.gum.uncert().magnitude, rtol=.02)


def test_welch():
    ''' Test welch-satterthwaite with units '''
    u = uc.UncertCalc.from_configfile('test/ex_xrf.yaml')
    # XRF problem with Yu units in nm, others in um
    u.calculate()
    assert np.isclose(u.out.gum.degf(), 27.5, atol=.1)
    assert np.isclose(u.out.gum.expanded().k, 2.05, atol=.01)

