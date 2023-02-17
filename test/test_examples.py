''' Test calculator output using examples from GUM, NIST TN1900, etc. '''

import os
import pytest
import numpy as np

from suncal import Model, ModelCallable
from suncal.project import ProjectUncert


def test_GUMH2():
    ''' Test example H.2 from GUM (resistance, reactance, impedance with correlated inputs).
        Read raw measurement data from CSV file.
        See table H.4 on page 88 for expected results.

        Both GUM and Monte Carlo methods are tested.
    '''
    V, J, th = np.genfromtxt(os.path.join('test', 'test_GUMH2.csv'), delimiter=',', skip_header=1).T
    J = J / 1000
    R = 'R = V/J * cos(theta)'
    X = 'X = V/J * sin(theta)'
    Z = 'Z = V/J'
    u = Model(R, X, Z)
    np.random.seed(0)
    k = np.sqrt(len(V))
    u.var('V').measure(V.mean()).typeb(std=V.std(ddof=1)/k)
    u.var('J').measure(J.mean()).typeb(std=J.std(ddof=1)/k)
    u.var('theta').measure(th.mean()).typeb(std=th.std(ddof=1)/k)
    u.variables.set_correlation(np.corrcoef(np.vstack((J, V, th))), names=['J', 'V', 'theta'])
    res = u.calculate_gum()
    mc = u.monte_carlo()

    # Within 3-decimal places given in GUM
    assert np.isclose(res.expected['R'], 127.732, rtol=0, atol=0.001)
    assert np.isclose(res.expected['X'], 219.847, rtol=0, atol=0.001)
    assert np.isclose(res.expected['Z'], 254.260, rtol=0, atol=0.001)
    assert np.isclose(res.uncertainty['R'], 0.071, rtol=0, atol=0.001)
    assert np.isclose(res.uncertainty['X'], 0.295, rtol=0, atol=0.001)
    assert np.isclose(res.uncertainty['Z'], 0.236, rtol=0, atol=0.001)

    # Add some relative tolerance due to MC randomness
    assert np.isclose(mc.expected['R'], 127.732, rtol=1E-5, atol=0.001)
    assert np.isclose(mc.expected['X'], 219.847, rtol=1E-5, atol=0.001)
    assert np.isclose(mc.expected['Z'], 254.260, rtol=1E-5, atol=0.001)
    assert np.isclose(mc.uncertainty['R'], 0.071, rtol=1E-5, atol=0.001)
    assert np.isclose(mc.uncertainty['X'], 0.295, rtol=1E-5, atol=0.001)
    assert np.isclose(mc.uncertainty['Z'], 0.236, rtol=1E-5, atol=0.001)


def test_GUMH2_2():
    ''' Test GUM H2, but this time let suncal determine the Type A uncertainty
        automatically
    '''
    V, J, th = np.genfromtxt(os.path.join('test', 'test_GUMH2.csv'), delimiter=',', skip_header=1).T
    J = J / 1000
    R = 'R = V/J * cos(theta)'
    X = 'X = V/J * sin(theta)'
    Z = 'Z = V/J'
    u = Model(R, X, Z)
    np.random.seed(0)
    u.var('V').measure(V)  # Arrays go in to measure(), Type A is automatic
    u.var('J').measure(J)
    u.var('theta').measure(th)
    u.variables.set_correlation(np.corrcoef(np.vstack((J, V, th))), names=['J', 'V', 'theta'])
    res = u.calculate_gum()
    mc = u.monte_carlo()

    # Within 3-decimal places given in GUM
    assert np.isclose(res.expected['R'], 127.732, rtol=0, atol=0.001)
    assert np.isclose(res.expected['X'], 219.847, rtol=0, atol=0.001)
    assert np.isclose(res.expected['Z'], 254.260, rtol=0, atol=0.001)
    assert np.isclose(res.uncertainty['R'], 0.071, rtol=0, atol=0.001)
    assert np.isclose(res.uncertainty['X'], 0.295, rtol=0, atol=0.001)
    assert np.isclose(res.uncertainty['Z'], 0.236, rtol=0, atol=0.001)

    # Add some relative tolerance due to MC randomness
    assert np.isclose(mc.expected['R'], 127.732, rtol=1E-5, atol=0.001)
    assert np.isclose(mc.expected['X'], 219.847, rtol=1E-5, atol=0.001)
    assert np.isclose(mc.expected['Z'], 254.260, rtol=1E-5, atol=0.001)
    assert np.isclose(mc.uncertainty['R'], 0.071, rtol=1E-5, atol=0.001)
    assert np.isclose(mc.uncertainty['X'], 0.295, rtol=1E-5, atol=0.001)
    assert np.isclose(mc.uncertainty['Z'], 0.236, rtol=1E-5, atol=0.001)


def test_NIST6():
    ''' Example E13 - Thermal Expansion Coefficient from NIST.TN.1900. '''
    # NOTE: be careful using examples from NIST calculator, they like to do conversions on mean/std/args from user
    # NIST example gives values in mean/std rather than center/scale, their code scales it like this.
    u = Model('(L1-L0) / (L0 * (T1 - T0))')
    u.var('L0').measure(1.4999).typeb(name='uL0', dist='t', unc=.0001, df=3)
    u.var('T0').measure(288.15).typeb(name='uT0', dist='t', unc=.02, df=3)
    u.var('L1').measure(1.5021).typeb(name='uL1', dist='t', unc=.0002, df=3)
    u.var('T1').measure(373.10).typeb(name='uT1', dist='t', unc=.05, df=3)
    np.random.seed(0)
    gum = u.calculate_gum()
    mc = u.monte_carlo()

    # Same sigfigs as NIST NUM calculator.
    assert np.isclose(gum.expected['f1'], 1.7266E-5, rtol=0, atol=0.0001E-5)
    assert np.isclose(gum.uncertainty['f1'], 1.76E-6, rtol=0, atol=0.01E-6)
    assert np.isclose(mc.expected['f1'], 1.7268E-5, rtol=1E-3, atol=0.0001E-5)
    assert np.isclose(mc.uncertainty['f1'], 1.74E-6, rtol=1E-2, atol=.1E-6)


def test_NIST6_fn():
    ''' Same example using python callable instead of string. Will exercise numeric gradient. '''
    def therm(L1, L0, T1, T0):
        return (L1-L0)/(L0*(T1-T0))
    u = ModelCallable(therm)
    u.var('L0').measure(1.4999).typeb(name='uL0', dist='t', unc=.0001, df=3)
    u.var('T0').measure(288.15).typeb(name='uT0', dist='t', unc=.02, df=3)
    u.var('L1').measure(1.5021).typeb(name='uL1', dist='t', unc=.0002, df=3)
    u.var('T1').measure(373.10).typeb(name='uT1', dist='t', unc=.05, df=3)
    np.random.seed(0)
    gum = u.calculate_gum()
    mc = u.monte_carlo()

    # Same sigfigs as NIST NUM calculator.
    assert np.isclose(gum.expected['therm'], 1.7266E-5, rtol=0, atol=0.0001E-5)
    assert np.isclose(gum.uncertainty['therm'], 1.76E-6, rtol=0, atol=0.01E-6)
    assert np.isclose(mc.expected['therm'], 1.7268E-5, rtol=1E-3, atol=0.0001E-5)
    assert np.isclose(mc.uncertainty['therm'], 1.74E-6, rtol=1E-2, atol=.1E-6)


def test_NIST10():
    ''' Example 10 (Stefan-Boltzmann Const) from NIST calculator manual. Also checks coverage interval calc. '''
    def sigma(h, R, Rinf, e, alpha):
        N = 32 * np.pi**5 * h * R**4 * Rinf**4
        D = 15 * e**4 * .001**4 * (299792458**6) * alpha**8
        return N/D

    # Note: The (299792458**6) term makes numpy return a dtype=object array
    # for some reason, which can cause issues if model.eval does not cast
    # back to float.

    u = ModelCallable(sigma)
    u.var('h').measure(6.62606957E-34).typeb(std=.00000029E-34)
    u.var('R').measure(8.3144621).typeb(std=.0000075)
    u.var('Rinf').measure(10973731.568539).typeb(std=.000055)
    u.var('e').measure(5.4857990946E-4).typeb(std=.0000000022E-4)
    u.var('alpha').measure(7.2973525698e-3).typeb(std=.0000000024E-3)
    np.random.seed(0)
    gum = u.calculate_gum()
    mc = u.monte_carlo()

    assert np.isclose(gum.expected['sigma'], 5.67037E-8, rtol=0, atol=.00001E-8)
    assert np.isclose(gum.uncertainty['sigma'], 2.05E-13, rtol=0, atol=.01E-13)
    assert np.isclose(mc.expected['sigma'], 5.67037E-8, rtol=1E-3, atol=.00001E-8)
    assert np.isclose(mc.uncertainty['sigma'], 2.05E-13, rtol=1E-3, atol=.01E-13)
    low, hi, k, _ = mc.expand('sigma', conf=.99)
    assert np.isclose(low, 5.67032E-8, rtol=1E-6, atol=.00001E-8)  # 99% interval
    assert np.isclose(hi, 5.67043E-8, rtol=1E-6, atol=.00001E-8)
    low, hi, k, _ = mc.expand('sigma', conf=.68)
    assert np.isclose(low, 5.67035E-8, rtol=1E-6, atol=.00001E-8)  # 68% interval
    assert np.isclose(hi, 5.67039E-8, rtol=1E-6, atol=.00001E-8)


def test_NISTE3():
    ''' Example E3 from NIST.TN.1900 - Falling Ball Viscometer '''
    proj = ProjectUncert.from_configfile('test/ex_viscometer.yaml')
    proj.seed = 65465456
    result = proj.calculate()

    # MC results
    assert np.isclose(result.montecarlo.expected['mu_m'], 5.82, rtol=0, atol=.01)
    assert np.isclose(result.montecarlo.uncertainty['mu_m'], 1.11, rtol=1, atol=.01)
    low, hi, k, _ = result.montecarlo.expand('mu_m', conf=.95)
    assert np.isclose(low, 4.05, rtol=1E-3, atol=.01)
    assert np.isclose(hi, 8.39, rtol=1E-3, atol=.01)

    # GUM results
    assert np.isclose(result.gum.expected['mu_m'], 5.69, rtol=0, atol=.01)
    assert np.isclose(result.gum.uncertainty['mu_m'], 1.11, rtol=0, atol=.1)  # NIST Has some round-off error


def test_NISTE11():
    ''' Example E11 from NIST.TN.1900 - Step Attenuator '''
    proj = ProjectUncert.from_configfile('test/ex_stepatten.yaml')
    proj.seed = 65465456
    result = proj.calculate()
    low, hi, k, _ = result.montecarlo.expand('Lx', conf=.95)

    assert np.isclose(result.montecarlo.expected['Lx'], 30.043, rtol=0, atol=.001)
    assert np.isclose(result.montecarlo.uncertainty['Lx'], 0.0224, rtol=0, atol=.0005)
    assert np.isclose(low, 30.006, rtol=1E-3, atol=.001)
    assert np.isclose(hi, 30.081, rtol=1E-3, atol=.001)


def test_GUMSUP2():
    ''' Magnitude/Phase example from GUM supplement 2 '''
    proj = ProjectUncert.from_configfile('test/ex_magphase.yaml')
    proj.seed = 65465456
    result = proj.calculate()
    # Re = .001

    # Values from GUM Sup2, table 6
    assert np.isclose(result.gum.expected['mag'], .001, atol=.0005)  # Magnitude
    assert np.isclose(result.gum.expected['ph'], 0.000, atol=.001)  # Phase
    assert np.isclose(result.gum.uncertainty['mag'], .010, atol=.001)
    assert np.isclose(result.gum.uncertainty['ph'], 10.000, atol=.001)
    assert np.isclose(result.montecarlo.expected['mag'], .013, atol=.0005)
    assert np.isclose(result.montecarlo.expected['ph'],  0, atol=.01)
    assert np.isclose(result.montecarlo.uncertainty['mag'], .007, atol=.0005)
    assert np.isclose(result.montecarlo.uncertainty['ph'], 1.744, atol=.002)

    # Now with non-zero covariance, values from table 7 (row 1)
    proj.model.variables.correlate('re', 'im', 0.9)
    result = proj.calculate()
    assert np.isclose(result.gum.expected['mag'], .001, atol=.0005)  # Magnitude
    assert np.isclose(result.gum.expected['ph'], 0.000, atol=.001)  # Phase
    assert np.isclose(result.gum.uncertainty['mag'], .010, atol=.001)
    assert np.isclose(result.gum.uncertainty['ph'], 10.000, atol=.001)
    assert np.isclose(result.montecarlo.expected['mag'], .012, atol=.0005)
    assert np.isclose(result.montecarlo.expected['ph'],  -.556, atol=.005)
    assert np.isclose(result.montecarlo.uncertainty['mag'], .008, atol=.0005)
    assert np.isclose(result.montecarlo.uncertainty['ph'], 1.599, atol=.002)

    # And again with re = 0.01, table 7 row 2
    proj.model.var('re').measure(.01)
    result = proj.calculate()

    assert np.isclose(result.gum.expected['mag'], .010, atol=.0005)  # Magnitude
    assert np.isclose(result.gum.expected['ph'], 0.000, atol=.001)  # Phase
    assert np.isclose(result.gum.uncertainty['mag'], .010, atol=.001)
    assert np.isclose(result.gum.uncertainty['ph'], 1.000, atol=.001)
    assert np.isclose(result.montecarlo.expected['mag'], .015, atol=.0005)
    assert np.isclose(result.montecarlo.expected['ph'],  -.343, atol=.005)
    assert np.isclose(result.montecarlo.uncertainty['mag'], .008, atol=.0005)
    assert np.isclose(result.montecarlo.uncertainty['ph'], .903, atol=.002)


def test_DEGF():
    ''' Test degrees of freedom - Example from ENGR224 '''
    u = Model('a+b')  # Formula and means dont matter
    u.var('a').typeb(std=.57, degf=9)
    u.var('b').typeb(std=0.25)  # degf=inf
    gum = u.calculate_gum()
    assert np.isclose(gum.degf['f1'], 12.8, atol=.1)


def test_GUMH1():
    ''' Example from GUM H1. Good test of degrees of freedom, and reading degf from file. '''
    proj = ProjectUncert.from_configfile('test/ex_endgauge.yaml')
    proj.seed = 65465456
    gum = proj.model.calculate_gum()

    assert np.isclose(gum.uncertainty['l'], 32, atol=.4)
    assert np.isclose(gum.degf['l'], 16, atol=1)

    # Check combining multiple components into standard uncert and degf
    assert np.isclose(proj.model.var('d').uncertainty, 9.7, atol=.05)
    assert np.isclose(proj.model.var('d').degrees_freedom, 25.6, atol=.05)


def test_montecarlo():
    ''' Test Monte-Carlo using examples from 9.2 of GUM Supplement 1.
        Note: numpy uses same Mersenne Twister algorithm for pseudo-random number generation
        as recommended by the GUM.
    '''
    inpts = ['X{}'.format(i+1) for i in range(4)]
    u = Model('+'.join(inpts))
    [u.var(x).typeb(std=1) for x in inpts]
    np.random.seed(10)
    mc = u.monte_carlo()
    low, hi, k, _ = mc.expand('f1', conf=.95)

    # Values from Table 2 in GUM Supplement 1
    assert np.isclose(mc.expected['f1'], 0, atol=.005)
    assert np.isclose(mc.uncertainty['f1'], 2.0, atol=.005)
    assert np.isclose(hi, 3.92, atol=.01)
    assert np.isclose(low, -3.92, atol=.01)

    # And repeat using 9.2.3 - rectangular distributions
    u = Model('+'.join(inpts))
    [u.var(x).typeb(dist='uniform', a=np.sqrt(3)) for x in inpts]
    np.random.seed(10)

    mc = u.monte_carlo()
    low, hi, k, _ = mc.expand('f1', conf=.95)
    # Values from Table 3 in GUM Supplement 1
    assert np.isclose(mc.expected['f1'], 0, atol=.005)
    assert np.isclose(mc.uncertainty['f1'], 2.0, atol=.005)
    assert np.isclose(hi, 3.88, atol=.01)
    assert np.isclose(low, -3.88, atol=.01)


def test_NPLlog():
    ''' Test Monte-Carlo vs GUM for y=log(x), described by NPL DEM-ES-011 section 9.2. '''
    u = Model('y=log(x)')
    u.var('x').measure(.6).typeb(dist='uniform', a=.5)   # a=.1, b=1.1
    np.random.seed(12345)
    gum = u.calculate_gum()
    mc = u.monte_carlo()
    assert np.isclose(gum.expected['y'], -.511, atol=.001)  # Results from Table 9.2
    assert np.isclose(gum.uncertainty['y'], .481, atol=.001)
    assert np.isclose(mc.expected['y'], -.665, atol=.001)
    assert np.isclose(mc.uncertainty['y'], .606, atol=.001)

    # GUM expanded 95%
    p = gum.expand('y', conf=.95)
    assert np.isclose(gum.expected['y'] + p, .432, atol=.001)

    # MC expanded were calculated using shortest interval
    # From table 9.2: min=–1.895, max=0.095.
    mn, mx, k, _ = mc.expand('y', conf=.95, shortest=True)
    assert np.isclose(mn, -1.895, atol=.001)
    assert np.isclose(mx, 0.095, atol=.001)


def test_XRF():
    ''' X-Ray Fluorescence example from SNL ENGR224 (v2) course notes '''
    u = Model('yc = X1/X2*Yu')
    u.var('X1').measure(.1820).typeb(std=.00093, df=9)
    u.var('X2').measure(.1823).typeb(std=.00058, df=19)
    u.var('Yu').measure(.6978).typeb(std=.0026, df=19)
    gum = u.calculate_gum()
    assert np.isclose(gum.uncertainty['yc'], .00494, atol=.00001)   # Slide 113 in "v2" version
    assert np.isclose(gum.expected['yc'], .6967, atol=.0001)       # Slide 115
    assert np.isclose(gum.degf['yc'], 27.7, atol=.5)              # Slide 117 (slides have some roundoff error)
    assert np.isclose(gum.expand('yc', conf=.95), .0101, atol=.0001)  # Slide 118


def test_inductance():
    ''' Inductance calculation. Test wrapping callable with units '''
    from scipy.special import ellipk, ellipe

    def inductance_nagaoka(radius, length, N, mu0):
        ''' Calculate inductance using Nagaoka formula '''
        k = np.sqrt(4*radius**2 / (4*radius**2 + length**2))
        kprime = np.sqrt(1 - k**2)
        Kk = ellipk(k**2)
        Ek = ellipe(k**2)
        kL = 4/3/np.pi/kprime * ((kprime/k)**2 * (Kk - Ek) + Ek - k)
        return mu0 * np.pi * N**2 * radius**2 / length * kL

    u = ModelCallable(inductance_nagaoka,
                      unitsin=['meter', 'meter', 'dimensionless', 'henry/meter'],
                      unitsout=['henry'])
    u.var('radius').measure(2.7, units='mm').typeb(unc=.005, k=1, units='mm')
    u.var('length').measure(9, units='mm').typeb(unc=.01, k=1, units='mm')
    u.var('N').measure(100, units='dimensionless')  # No uncertainty
    u.var('mu0').measure(1.25663706212E-6, units='H/m').typeb(unc=0.00000000019E-6, k=2, units='H/m')
    gum = u.calculate_gum()
    mc = u.monte_carlo()

    assert np.isclose(gum.expected['inductance_nagaoka'].to('uH').magnitude, 25.21506)
    assert np.isclose(mc.expected['inductance_nagaoka'].to('uH').magnitude, 25.21506)
