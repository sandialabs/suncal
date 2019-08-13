''' Test calculator output using examples from GUM, NIST TN1900, etc. '''

import pytest
import os
import numpy
import sympy

import suncal as uc


def test_GUMH2():
    ''' Test example H.2 from GUM (resistance, reactance, impedance with correlated inputs).
        Read raw measurement data from CSV file.
        See table H.4 on page 88 for expected results.

        Both K-M and M-C methods are tested.
    '''
    V, J, th = numpy.genfromtxt(os.path.join('test', 'test_GUMH2.csv'), delimiter=',', skip_header=1).T
    J = J / 1000
    R = 'R = V/J * cos(theta)'
    X = 'X = V/J * sin(theta)'
    Z = 'Z = V/J'
    u = uc.UncertaintyCalc([R, X, Z], seed=0)  # Specify random seed so we don't get occasional monte-carlo failures
    k = numpy.sqrt(len(V))
    u.set_input('V', nom=V.mean())
    u.set_input('J', nom=J.mean())
    u.set_input('theta', nom=th.mean())
    u.set_uncert('V', 'u(V)', std=V.std(ddof=1)/k)
    u.set_uncert('J', 'u(J)', std=J.std(ddof=1)/k)
    u.set_uncert('theta', 'u(theta)', std=th.std(ddof=1)/k)
    u.set_correlation(numpy.corrcoef(numpy.vstack((V, J, th))))
    u.calculate()
    GUM0 = u.out.R.gum
    GUM1 = u.out.X.gum
    GUM2 = u.out.Z.gum

    assert numpy.isclose(GUM0.mean.magnitude, 127.732, rtol=0, atol=0.001) # Within 3-decimal places given in GUM
    assert numpy.isclose(GUM1.mean.magnitude, 219.847, rtol=0, atol=0.001)
    assert numpy.isclose(GUM2.mean.magnitude, 254.260, rtol=0, atol=0.001)
    assert numpy.isclose(GUM0.uncert.magnitude, 0.071, rtol=0, atol=0.001)
    assert numpy.isclose(GUM1.uncert.magnitude, 0.295, rtol=0, atol=0.001)
    assert numpy.isclose(GUM2.uncert.magnitude, 0.236, rtol=0, atol=0.001)

    MC0 = u.out.R.mc
    MC1 = u.out.X.mc
    MC2 = u.out.Z.mc
    assert numpy.isclose(MC0.mean.magnitude, 127.732, rtol=1E-5, atol=0.001) # Add some relative tolerance due to MC randomness
    assert numpy.isclose(MC1.mean.magnitude, 219.847, rtol=1E-5, atol=0.001)
    assert numpy.isclose(MC2.mean.magnitude, 254.260, rtol=1E-5, atol=0.001)
    assert numpy.isclose(MC0.uncert.magnitude, 0.071, rtol=1E-5, atol=0.001)
    assert numpy.isclose(MC1.uncert.magnitude, 0.295, rtol=1E-5, atol=0.001)
    assert numpy.isclose(MC2.uncert.magnitude, 0.236, rtol=1E-5, atol=0.001)


def test_NIST6():
    ''' Example E13 - Thermal Expansion Coefficient from NIST.TN.1900. '''
    # NOTE: be careful using examples from NIST calculator, they like to do conversions on mean/std/args from user
    # NIST example gives values in mean/std rather than center/scale, their code scales it like this.
    inputs = [{'name': 'L0', 'nom':1.4999, 'uncerts': [{'name': 'uL0', 'dist':'t', 'unc':.0001, 'df':3}]},
              {'name': 'T0',  'nom':288.15, 'uncerts': [{'name': 'uT0', 'dist':'t', 'unc':.02,   'df':3}]},
              {'name': 'L1', 'nom':1.5021, 'uncerts': [{'name': 'uL1', 'dist':'t', 'unc':.0002, 'df':3}]},
              {'name': 'T1', 'nom':373.10, 'uncerts': [{'name': 'uT1', 'dist':'t', 'unc':.05,   'df':3}]}]
    u = uc.UncertaintyCalc('(L1-L0) / (L0 * (T1 - T0))', inputs=inputs, seed=0)
    u.calculate()
    GUM = u.out.get_output(method='gum')
    MC = u.out.get_output(method='mc')

    assert numpy.isclose(GUM.mean.magnitude, 1.7266E-5, rtol=0, atol=0.0001E-5)  # Same sigfigs as NIST NUM calculator.
    assert numpy.isclose(GUM.uncert.magnitude, 1.76E-6, rtol=0, atol=0.01E-6)
    assert numpy.isclose(MC.mean.magnitude, 1.7268E-5, rtol=1E-3, atol=0.0001E-5)
    assert numpy.isclose(MC.uncert.magnitude, 1.74E-6, rtol=1E-2, atol=.1E-6)


def test_NIST6_fn():
    ''' Same example using python callable instead of string. Will exercise numeric gradient. '''
    def therm(L1, L0, T1, T0):
        return (L1-L0)/(L0*(T1-T0))
    inputs = [{'name': 'L0', 'nom':1.4999, 'uncerts': [{'name': 'uL0', 'dist':'t', 'unc':.0001, 'df':3}]},
              {'name': 'T0',  'nom':288.15, 'uncerts': [{'name': 'uT0', 'dist':'t', 'unc':.02,   'df':3}]},
              {'name': 'L1', 'nom':1.5021, 'uncerts': [{'name': 'uL1', 'dist':'t', 'unc':.0002, 'df':3}]},
              {'name': 'T1', 'nom':373.10, 'uncerts': [{'name': 'uT1', 'dist':'t', 'unc':.05,   'df':3}]}]
    u = uc.UncertaintyCalc(therm, inputs=inputs, seed=0)
    u.calculate()
    GUM = u.out.get_output(method='gum')
    MC = u.out.get_output(method='mc')

    assert numpy.isclose(GUM.mean.magnitude, 1.7266E-5, rtol=0, atol=0.0001E-5)  # Same sigfigs
    assert numpy.isclose(GUM.uncert.magnitude, 1.76E-6, rtol=0, atol=0.01E-6)
    assert numpy.isclose(MC.mean.magnitude, 1.7268E-5, rtol=1E-3, atol=0.0001E-5)
    assert numpy.isclose(MC.uncert.magnitude, 1.74E-6, rtol=1E-2, atol=.1E-6)


def test_NIST10():
    ''' Example 10 (Stefan-Boltzmann Const) from NIST calculator manual. Also checks coverage interval calc. '''
    def sigma(h, R, Rinf, e, alpha):
        N = 32 * numpy.pi**5 * h * R**4 * Rinf**4
        D = 15 * e**4 * .001**4 * (299792458**6) * alpha**8
        return N/D

    u = uc.UncertaintyCalc(sigma, seed=0)
    u.set_input('h', nom=6.62606957E-34)
    u.set_input('R', nom=8.3144621)
    u.set_input('Rinf', nom=10973731.568539)
    u.set_input('e', nom=5.4857990946E-4)
    u.set_input('alpha', nom=7.2973525698e-3)
    u.set_uncert('h', 'u(h)', std=.00000029E-34)
    u.set_uncert('R', 'u(R)', std=.0000075)
    u.set_uncert('Rinf', 'u(Rinf)', std=.000055)
    u.set_uncert('e', 'u(e)', std=.0000000022E-4)
    u.set_uncert('alpha', 'u(alpha)', std=.0000000024E-3)
    u.calculate()
    GUM = u.out.get_output(method='gum')
    MC = u.out.get_output(method='mc')

    assert numpy.isclose(GUM.mean.magnitude, 5.67037E-8, rtol=0, atol=.00001E-8)
    assert numpy.isclose(GUM.uncert.magnitude, 2.05E-13, rtol=0, atol=.01E-13)
    assert numpy.isclose(MC.mean.magnitude, 5.67037E-8, rtol=1E-3, atol=.00001E-8)
    assert numpy.isclose(MC.uncert.magnitude, 2.05E-13, rtol=1E-3, atol=.01E-13)
    low, hi, k = MC.expanded(.99)
    assert numpy.isclose(low.magnitude, 5.67032E-8, rtol=1E-6, atol=.00001E-8) # 99% interval
    assert numpy.isclose(hi.magnitude, 5.67043E-8, rtol=1E-6, atol=.00001E-8)
    low, hi, k = MC.expanded(.68)
    assert numpy.isclose(low.magnitude, 5.67035E-8, rtol=1E-6, atol=.00001E-8) # 68% interval
    assert numpy.isclose(hi.magnitude, 5.67039E-8, rtol=1E-6, atol=.00001E-8)


def test_NISTE3():
    ''' Example E3 from NIST.TN.1900 - Falling Ball Viscometer '''
    u = uc.UncertaintyCalc.from_configfile('test/ex_viscometer.yaml')
    u.seed = 0
    u.calculate()
    GUM = u.out.get_output(method='gum')
    MC = u.out.get_output(method='mc')

    # MC results
    assert numpy.isclose(MC.mean.magnitude, 5.82, rtol=0, atol=.01)
    assert numpy.isclose(MC.uncert.magnitude, 1.11, rtol=1, atol=.01)
    low, hi, k = MC.expanded(.95)
    assert numpy.isclose(low.magnitude, 4.05, rtol=1E-3, atol=.01)
    assert numpy.isclose(hi.magnitude, 8.39, rtol=1E-3, atol=.01)

    # GUM results
    assert numpy.isclose(GUM.mean.magnitude, 5.69, rtol=0, atol=.01)
    assert numpy.isclose(GUM.uncert.magnitude, 1.11, rtol=0, atol=.1)  # NIST Has some round-off error


def test_NISTE11():
    ''' Example E11 from NIST.TN.1900 - Step Attenuator '''
    u = uc.UncertaintyCalc.from_configfile('test/ex_stepatten.yaml')
    u.seed = 0
    u.calculate()
    GUM = u.out.get_output(method='gum')
    MC = u.out.get_output(method='mc')
    low, hi, k = MC.expanded(.95)

    assert numpy.isclose(MC.mean.magnitude, 30.043, rtol=0, atol=.001)
    assert numpy.isclose(MC.uncert.magnitude, 0.0224, rtol=0, atol=.0005)
    assert numpy.isclose(low.magnitude, 30.006, rtol=1E-3, atol=.001)
    assert numpy.isclose(hi.magnitude, 30.081, rtol=1E-3, atol=.001)


def test_GUMSUP2():
    ''' Magnitude/Phase example from GUM supplement 2 '''
    u = uc.UncertaintyCalc.from_configfile('test/ex_magphase.yaml')
    # Re = .001
    u.seed = 0
    u.calculate()
    GUM0 = u.out.get_output(fidx=0, method='gum')
    GUM1 = u.out.get_output(fidx=1, method='gum')
    MC0 = u.out.get_output(fidx=0, method='mc')
    MC1 = u.out.get_output(fidx=1, method='mc')

    # Values from GUM Sup2, table 6
    assert numpy.isclose(GUM0.mean.magnitude, .001, atol=.0005)  # Magnitude
    assert numpy.isclose(GUM1.mean.magnitude, 0.000, atol=.001)  # Phase
    assert numpy.isclose(GUM0.uncert.magnitude, .010, atol=.001)
    assert numpy.isclose(GUM1.uncert.magnitude, 10.000, atol=.001)
    assert numpy.isclose(MC0.mean.magnitude, .013, atol=.0005)
    assert numpy.isclose(MC1.mean.magnitude,  0, atol=.01)
    assert numpy.isclose(MC0.uncert.magnitude, .007, atol=.0005)
    assert numpy.isclose(MC1.uncert.magnitude, 1.744, atol=.002)

    # Now with non-zero covariance, values from table 7 (row 1)
    u.correlate_vars('re', 'im', 0.9)
    u.calculate()
    GUM0 = u.out.get_output(fidx=0, method='gum')
    GUM1 = u.out.get_output(fidx=1, method='gum')
    MC0 = u.out.get_output(fidx=0, method='mc')
    MC1 = u.out.get_output(fidx=1, method='mc')
    assert numpy.isclose(GUM0.mean.magnitude, .001, atol=.0005)  # Magnitude
    assert numpy.isclose(GUM1.mean.magnitude, 0.000, atol=.001)  # Phase
    assert numpy.isclose(GUM0.uncert.magnitude, .010, atol=.001)
    assert numpy.isclose(GUM1.uncert.magnitude, 10.000, atol=.001)
    assert numpy.isclose(MC0.mean.magnitude, .012, atol=.0005)
    assert numpy.isclose(MC1.mean.magnitude,  -.556, atol=.005)
    assert numpy.isclose(MC0.uncert.magnitude, .008, atol=.0005)
    assert numpy.isclose(MC1.uncert.magnitude, 1.599, atol=.002)

    # And again with re = 0.01, table 7 row 2
    u.set_input('re', nom=.01)
    u.calculate()
    GUM0 = u.out.get_output(fidx=0, method='gum')
    GUM1 = u.out.get_output(fidx=1, method='gum')
    MC0 = u.out.get_output(fidx=0, method='mc')
    MC1 = u.out.get_output(fidx=1, method='mc')
    assert numpy.isclose(GUM0.mean.magnitude, .010, atol=.0005)  # Magnitude
    assert numpy.isclose(GUM1.mean.magnitude, 0.000, atol=.001)  # Phase
    assert numpy.isclose(GUM0.uncert.magnitude, .010, atol=.001)
    assert numpy.isclose(GUM1.uncert.magnitude, 1.000, atol=.001)
    assert numpy.isclose(MC0.mean.magnitude, .015, atol=.0005)
    assert numpy.isclose(MC1.mean.magnitude,  -.343, atol=.005)
    assert numpy.isclose(MC0.uncert.magnitude, .008, atol=.0005)
    assert numpy.isclose(MC1.uncert.magnitude, .903, atol=.002)


# Test degrees of freedom calculation
def test_DEGF():
    ''' Example from ENGR224 '''
    u = uc.UncertaintyCalc('a+b', seed=0)  # Formula and means dont matter
    u.set_input('a', nom=1)
    u.set_input('b', nom=1)
    u.set_uncert('a', 'ua', std=0.57, degf=9)
    u.set_uncert('b', 'ub', std=0.25)  # Not provided, degf=inf
    u.calculate()
    GUM = u.out.get_output(method='gum')

    assert numpy.isclose(GUM.degf, 12.8, atol=.1)
    assert numpy.isclose(GUM.degf, 12.8, atol=.1)


def test_GUMH1():
    ''' Example from GUM H1. Good test of degrees of freedom, and reading degf from file. '''
    u = uc.UncertaintyCalc.from_configfile('test/ex_endgauge.yaml')
    u.seed = 0
    u.calculate(MC=False)
    GUM = u.out.get_output(method='gum')
    assert numpy.isclose(GUM.uncert.magnitude, 32, atol=.4)
    assert numpy.isclose(GUM.degf, 16, atol=1)

    # Check combining multiple components into standard uncert and degf
    assert numpy.isclose(u.get_input('d').stdunc().magnitude, 9.7, atol=.05)
    assert numpy.isclose(u.get_input('d').degf(), 25.6, atol=.05)


def test_montecarlo():
    ''' Test Monte-Carlo using examples from 9.2 of GUM Supplement 1.
        Note: numpy uses same Mersenne Twister algorithm for pseudo-random number generation
        as recommended by the GUM.
    '''
    inpts = ['X{}'.format(i+1) for i in range(4)]
    u = uc.UncertaintyCalc('+'.join(inpts), seed=0)
    [u.set_input(x, nom=0) for x in inpts]
    [u.set_uncert(x, std=1) for x in inpts]
    u.calculate(GUM=False)
    MC = u.out.get_output(method='mc')
    low, hi, k = MC.expanded(.95)
    # Values from Table 2 in GUM Supplement 1
    assert numpy.isclose(MC.mean.magnitude, 0, atol=.005)
    assert numpy.isclose(MC.uncert.magnitude, 2.0, atol=.005)
    assert numpy.isclose(hi.magnitude, 3.92, atol=.005)
    assert numpy.isclose(low.magnitude, -3.92, atol=.005)

    # And repeat using 9.2.3 - rectangular distributions
    u = uc.UncertaintyCalc('+'.join(inpts), seed=0)
    [u.set_input(x, nom=0) for x in inpts]
    [u.set_uncert(x, dist='uniform', a=numpy.sqrt(3)) for x in inpts]
    u.calculate(GUM=False)
    MC = u.out.get_output(method='mc')
    low, hi, k = MC.expanded(.95)
    # Values from Table 3 in GUM Supplement 1
    assert numpy.isclose(MC.mean.magnitude, 0, atol=.005)
    assert numpy.isclose(MC.uncert.magnitude, 2.0, atol=.005)
    assert numpy.isclose(hi.magnitude, 3.88, atol=.005)
    assert numpy.isclose(low.magnitude, -3.88, atol=.005)


def test_NPLlog():
    ''' Test Monte-Carlo vs GUM for y=log(x), described by NPL DEM-ES-011 section 9.2. '''
    u = uc.UncertaintyCalc('y=log(x)', seed=12345)
    u.set_input('x', nom=.6, dist='uniform', a=.5)   # a=.1, b=1.1
    u.calculate()
    assert numpy.isclose(u.out.y.gum.mean.magnitude, -.511, atol=.001)  # Results from Table 9.2
    assert numpy.isclose(u.out.y.gum.uncert.magnitude, .481, atol=.001)
    assert numpy.isclose(u.out.y.mc.mean.magnitude, -.665, atol=.001)
    assert numpy.isclose(u.out.y.mc.uncert.magnitude, .606, atol=.001)

    # GUM expanded 95%
    p, k = u.out.y.gum.expanded(.95)
    assert numpy.isclose(u.out.y.gum.mean.magnitude + p.magnitude, .432, atol=.001)

    # MC expanded were calculated using shortest interval
    # From table 9.2: min=–1.895, max=0.095.
    mn, mx, k = u.out.y.mc.expanded(.95, shortest=True)
    assert numpy.isclose(mn.magnitude, -1.895, atol=.001)
    assert numpy.isclose(mx.magnitude, 0.095, atol=.001)


def test_XRF():
    ''' X-Ray Fluorescence example from SNL ENGR224 (v2) course notes '''
    u = uc.UncertaintyCalc('yc = X1/X2*Yu')
    u.set_input('X1', nom=.1820, std=.00093, df=9)
    u.set_input('X2', nom=.1823, std=.00058, df=19)
    u.set_input('Yu', nom=.6978, std=.0026, df=19)
    u.calculate()
    assert numpy.isclose(u.out.yc.gum.uncert.magnitude, .00494, atol=.00001)   # Slide 113 in "v2" version
    assert numpy.isclose(u.out.yc.gum.mean.magnitude, .6967, atol=.0001)       # Slide 115
    assert numpy.isclose(u.out.yc.gum.degf, 27.7, atol=.5)              # Slide 117 (slides have some roundoff error)
    assert numpy.isclose(u.out.yc.gum.expanded(.95)[0].magnitude, .0101, atol=.0001) # Slide 118

