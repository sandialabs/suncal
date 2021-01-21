''' Test cases for uncertainty core. '''
import pytest

import os
import numpy as np
import sympy

import suncal as uc
from suncal import uncertainty
from suncal import curvefit


def test_chain():
    ''' Test chaining of functions '''
    u = uc.UncertaintyCalc(seed=0)
    u.set_input('x', nom=100, std=1)  # Test adding uncertainties via set_input()
    u.set_input('y', nom=50, std=1)
    u.set_function('x+y', name='f')
    u.set_function('2*f', name='g')
    u.set_function('2*g', name='h')
    u.calculate()
    assert np.isclose(u.out.gum.nom('f').magnitude, u.out.gum.nom('g').magnitude/2)
    assert np.isclose(u.out.gum.nom('g').magnitude, u.out.gum.nom('h').magnitude/2)
    assert np.isclose(u.out.gum.uncert('f').magnitude, u.out.gum.uncert('g').magnitude/2)
    assert np.isclose(u.out.gum.uncert('g').magnitude, u.out.gum.uncert('h').magnitude/2)

    assert np.isclose(u.out.mc.nom('f').magnitude, u.out.mc.nom('g').magnitude/2, rtol=.02)
    assert np.isclose(u.out.mc.nom('g').magnitude, u.out.mc.nom('h').magnitude/2, rtol=.02)
    assert np.isclose(u.out.mc.uncert('f').magnitude, u.out.mc.uncert('g').magnitude/2, rtol=.05)
    assert np.isclose(u.out.mc.uncert('g').magnitude, u.out.mc.uncert('h').magnitude/2, rtol=.05)

    # Now change the base equation and verify everything trickles down
    oldh_mean = u.out.gum.nom('h').magnitude
    oldh_unc = u.out.gum.uncert('h').magnitude
    u.set_function('(x+y)/2', name='f')
    u.calculate()
    assert oldh_mean/2 == u.out.gum.nom('h').magnitude
    assert oldh_unc/2 == u.out.gum.uncert('h').magnitude

    assert np.isclose(u.out.gum.nom('f').magnitude, u.out.gum.nom('g').magnitude/2)
    assert np.isclose(u.out.gum.nom('g').magnitude, u.out.gum.nom('h').magnitude/2)
    assert np.isclose(u.out.gum.uncert('f').magnitude, u.out.gum.uncert('g').magnitude/2)
    assert np.isclose(u.out.gum.uncert('g').magnitude, u.out.gum.uncert('h').magnitude/2)

    assert np.isclose(u.out.mc.nom('f').magnitude, u.out.mc.nom('g').magnitude/2, rtol=.02)
    assert np.isclose(u.out.mc.nom('g').magnitude, u.out.mc.nom('h').magnitude/2, rtol=.02)
    assert np.isclose(u.out.mc.uncert('f').magnitude, u.out.mc.uncert('g').magnitude/2, rtol=.05)
    assert np.isclose(u.out.mc.uncert('g').magnitude, u.out.mc.uncert('h').magnitude/2, rtol=.05)

    # Double-chaining, order shouldn't matter
    u2 = uc.UncertaintyCalc(['g = a*b*f', 'f = a+b*c+h', 'h=a+d'])
    u2.set_input('a', nom=10, std=1)
    u2.set_input('b', nom=10, std=1)
    u2.set_input('c', nom=1, std=.5)
    u2.set_input('d', nom=1, std=.5)
    u2.calculate()
    u2.out.report_sens()  # This was crashing before
    symbols = u2.model.inputnames
    assert 'a' in symbols
    assert 'b' in symbols
    assert 'c' in symbols
    assert 'd' in symbols
    assert 'f' not in symbols  # Functions were substituted out
    assert 'g' not in symbols
    assert 'h' not in symbols


def test_callable():
    ''' Test callable (named arguments) and vectorizable function as input '''
    def myfunc(a, b):
        return a + b**2

    u = uc.UncertaintyCalc(myfunc, seed=0)
    reqinpts = u.required_inputs
    assert 'a' in reqinpts
    assert 'b' in reqinpts
    assert len(reqinpts) == 2
    u.add_required_inputs()
    assert len(u.inputs) == 2  # a, b
    assert 'a' in u.model.inputnames
    assert 'b' in u.model.inputnames

    u.set_input('a', nom=5)
    u.set_input('b', nom=2)
    u.set_uncert('a', std=.05)
    u.set_uncert('b', std=.02)
    assert u.inputs.means()['a'] == 5

    u.calculate()
    assert np.isclose(u.out.gum.nom().magnitude, 9)


def test_callablekwargs():
    ''' Test callable with **kwargs '''
    def myfunc(**kwargs):
        x = kwargs.get('x')
        y = kwargs.get('y')
        return x * y

    with pytest.raises(ValueError):  # Error if function takes unnamed kwargs and kwnames parameter not specified
        u = uc.UncertaintyCalc(myfunc, seed=0)

    u = uc.UncertaintyCalc(myfunc, finnames=['x', 'y'], seed=0)
    reqinpts = u.required_inputs
    assert 'x' in reqinpts
    assert 'y' in reqinpts
    assert len(reqinpts) == 2

    u.set_input('x', nom=2)
    u.set_input('y', nom=4)
    u.set_uncert('x', std=.1)
    u.set_uncert('y', std=.2)
    u.calculate()
    assert np.isclose(u.out.gum.nom().magnitude, 8)


def test_chaincallable():
    ''' Test chaining callable functions '''
    def myfunc1(x, y):
        return x * y

    def myfunc2(myfunc1):
        return myfunc1 + 100

    def myfunc(x, y):
        m = myfunc1(x, y)
        return m, myfunc2(m)

    u = uc.UncertaintyCalc(myfunc, seed=0, samples=10)
    reqinpts = u.required_inputs
    assert 'x' in reqinpts
    assert 'y' in reqinpts
    u.add_required_inputs()
    assert 'x' in u.inputs.names
    assert 'y' in u.inputs.names
    assert len(u.model.outnames) == 2
    assert 'myfunc_1' in u.model.outnames  # Can't automagically get name when output is a tuple

    u.set_input('x', nom=2)
    u.set_input('y', nom=10)
    u.set_uncert('x', std=.1)
    u.set_uncert('y', std=.5)
    u.calculate()
    assert np.isclose(u.out.gum.nom(0).magnitude, 20)
    assert np.isclose(u.out.gum.nom(1).magnitude, 120)


@pytest.mark.filterwarnings('ignore')  # Will generate unitstripped warning due to use of np.vectorize with unit values
def test_vectorize():
    ''' Make sure non-vectorized functions can run. Also tests function with kwargs arguments '''
    # This function is not vectorizable as-is. Calculator will try it, fail, and then
    # try using np.vectorize() on it.
    def tcr(**kwargs):
        ''' Temperature Coefficient of Resistance from pairs of R, T measurements. '''
        R = np.array([kwargs.get('R{}'.format(i+1)) for i in range(len(kwargs)//2)])
        T = np.array([kwargs.get('T{}'.format(i+1)) for i in range(len(kwargs)//2)])
        p = np.polyfit(T-T[0], R/R[0]-1, deg=1)[0]
        return p

    varnames = ['T{}'.format(i+1) for i in range(4)] + ['R{}'.format(i+1) for i in range(4)]

    u = uc.UncertaintyCalc(tcr, finnames=varnames, samples=1000)
    for i, rval in enumerate([100, 100.1, 100.2, 100.3]):
        Rname = 'R{}'.format(i+1)
        u.set_input(Rname, nom=rval)
        u.set_uncert(Rname, std=.2)
    for i, tval in enumerate([20, 22, 24, 26]):
        Tname = 'T{}'.format(i+1)
        u.set_input(Tname, nom=tval)
        u.set_uncert(Tname, std=.05)
    u.calculate(GUM=False)
    assert np.isclose(u.out.mc.nom().magnitude, 0.0005, atol=.0001)


@pytest.mark.filterwarnings('ignore')  # Will generate unitstripped warning due to use of np.vectorize with unit values
def test_vectorize_units():
    def test(x):
        ''' A function that will fail if called with an array '''
        if x > 0:
            return x*2
        else:
            return x

    # First run has no units
    u = uc.UncertCalc(test, samples=1000)
    u.set_input('x', nom=100, std=1)
    u.calculate(gum=False)
    assert np.isclose(u.out.mc.nom().magnitude, 200, atol=1)
    assert str(u.out.mc._units[0]) == 'dimensionless'

    # Now input has units but output units are not specified
    u = uc.UncertCalc(test, samples=1000)
    u.set_input('x', nom=100, std=1, units='cm')
    u.calculate(gum=False)
    assert np.isclose(u.out.mc.nom().magnitude, 200, atol=1)
    assert str(u.out.mc._units[0]) == 'centimeter'

    # Finally, request a unit conversion on the output
    u = uc.UncertCalc(test, units='meter', samples=1000)
    u.set_input('x', nom=100, std=1, units='cm')
    u.calculate(gum=False)
    assert np.isclose(u.out.mc.nom().magnitude, 2, atol=.1)
    assert str(u.out.mc._units[0]) == 'meter'


@pytest.mark.filterwarnings('ignore')  # Will generate a np warning about degrees of freedom <= 0
def test_constant():
    ''' Functions can also be constant with no variables. '''
    u = uc.UncertaintyCalc()
    u.set_function('10', name='a')
    u.set_function('a+b')
    u.set_input('b', 5)
    u.set_uncert('b', std=0.1)
    u.calculate()
    assert np.isclose(u.out.gum.nom(0).magnitude, 10)
    assert np.isclose(u.out.gum.nom(1).magnitude, 15)


def test_readconfig():
    ''' Test read_configfile '''
    u = uc.UncertaintyCalc.from_configfile(os.path.join('test', 'test1.yaml'))
    assert u.model.outnames == ['f', 'g', 'h']
    assert u.model.exprs == ['(a + b) / c', 'a - b', 'b * c']
    assert u.inputs.nsamples == 1E6
    assert len(u.inputs) == 3
    a = u.get_inputvar('a')
    assert a.name == 'a'
    assert a.nom == 10.0
    assert float(a.uncerts[0].args['std']) == 0.2
    assert a.uncerts[0].distname == 'normal'
    assert a.degf() == 10
    b = u.get_inputvar('b')
    assert b.name == 'b'
    assert b.nom == 25.0
    assert float(b.uncerts[0].args['scale']) == 2.0
    assert float(b.uncerts[0].args['a']) == 5.0
    assert b.uncerts[0].distname == 'gamma'
    assert b.degf() == np.inf
    c = u.get_inputvar('c')
    assert c.name == 'c'
    assert c.nom == 2.0
    assert float(c.uncerts[0].args['std']) == 0.1
    assert c.uncerts[0].distname == 'normal'
    assert c.degf() == 88
    corlist = u.inputs.corr_list
    assert ('b', 'a', -0.36) in corlist or ('a', 'b', -0.36) in corlist
    assert ('a', 'c', -0.4) in corlist or ('c', 'a', -0.4) in corlist
    assert ('b', 'c', 0.86) in corlist or ('c', 'b', 0.86) in corlist

def test_saveconfig():
    ''' Test save_config and read_config, we can go in a circle. '''
    CHECK_FILE = os.path.join('test', 'TEST_SAVE.YAML')

    u = uc.UncertaintyCalc.from_configfile(os.path.join('test', 'test1.yaml'))
    u.save_config(CHECK_FILE)
    u2 = uc.UncertaintyCalc.from_configfile(CHECK_FILE)

    assert u.model.sympyexprs == u2.model.sympyexprs
    assert u.inputs[0].name == u2.inputs[0].name
    assert u.inputs[1].name == u2.inputs[1].name
    assert u.inputs[2].name == u2.inputs[2].name
    assert np.allclose(u.inputs.correlation(), u2.inputs.correlation(), rtol=1E-9, atol=0)
    assert u.inputs.nsamples == u2.inputs.nsamples
    os.remove(CHECK_FILE)

def test_addinputs():
    ''' Test add_required_inputs() function '''
    # For a string function...
    u = uc.UncertaintyCalc('a + b + c + d')
    u.set_input('a', nom=1)
    u.add_required_inputs()
    assert len(u.inputs) == 4
    assert 'b' in u.inputs.names
    assert 'c' in u.inputs.names
    assert 'd' in u.inputs.names

    # and a sympy function...
    x,y,z = sympy.symbols('x y z')
    f = x+y+z
    u = uc.UncertaintyCalc(f)
    u.add_required_inputs()
    assert len(u.inputs) == 3
    assert 'x' in u.inputs.names
    assert 'y' in u.inputs.names
    assert 'z' in u.inputs.names

    # and for callable...
    def myfunc(j,k,l):
        return j+k+l
    u = uc.UncertaintyCalc(myfunc)
    u.add_required_inputs()
    assert len(u.inputs) == 3
    assert 'j' in u.inputs.names
    assert 'k' in u.inputs.names
    assert 'l' in u.inputs.names

def test_checkinput():
    ''' Test InputVar.check_args '''
    i = uc.InputUncert('b', nom=10, dist='gamma', std=1)
    assert i.check_args() == True # 'alpha' parameter automatically set to 1

    i = uc.InputUncert('b', nom=10, dist='gamma', std=1, alpha=-1)
    assert i.check_args() == False # Invalid 'alpha' parameter value

    # ValueError if not all inputs are defined
    u = uc.UncertaintyCalc('a+b')
    with pytest.raises(ValueError):
        u.calculate(MC=False)

def test_reqargs():
    ''' Test InputVar.req_args() '''
    args = uc.InputUncert('b', dist='gamma').required_args
    assert 'alpha' in args

    args = uc.InputUncert('b', dist='t').required_args
    assert 'df' in args

    args = uc.InputUncert('b', dist='burr').required_args
    assert 'c' in args
    assert 'd' in args

@pytest.mark.filterwarnings('ignore')  # Will generate a np warning about degrees of freedom <= 0
def test_reserved():
    ''' Test that reserved sympy keywords/functions are properly handled. '''
    # "pi" is 3.14, not a symbol
    u = uc.UncertaintyCalc('pi', seed=0)
    u.calculate(MC=False)
    assert np.isclose(u.out.gum.nom().magnitude, np.pi)

    # "gamma" is a symbol, not gamma function
    u = uc.UncertaintyCalc('gamma/2')
    u.set_input('gamma', nom=10)
    u.calculate(MC=False)
    assert np.isclose(u.out.gum.nom().magnitude, 5)

    # But '"cos" is the sympy cosine, not a variable
    u = uc.UncertaintyCalc('cos(x)')
    u.set_input('x', nom=np.pi)
    u.calculate(MC=False)
    assert np.isclose(u.out.gum.nom().magnitude, np.cos(np.pi))

def test_reorder():
    ''' Test UncertCalc.reorder() '''
    u = uc.UncertaintyCalc(['f=a+b', 'g=a*b'])
    assert u.get_functionnames() == ['f', 'g']
    u.reorder(['g','f'])
    assert u.get_functionnames() == ['g', 'f']

def test_seed():
    ''' Test random seed argument '''
    # With seed, outputs will be the same
    u = uc.UncertaintyCalc('f=a+b', seed=10)
    u.set_input('a', nom=10)
    u.set_input('b', nom=5)
    u.set_uncert('a', std=.1)
    u.set_uncert('b', std=.05)
    u.calculate()
    vals = u.out.mc.samples('f')[:10]  # Check first 10 samples
    u.calculate()
    assert (u.out.mc.samples('f')[:10] == vals).all()

    # With seed=None, seed is randomized
    u = uc.UncertaintyCalc('f=a+b', seed=None)
    u.set_input('a', nom=10)
    u.set_input('b', nom=5)
    u.set_uncert('a', std=.1)
    u.set_uncert('b', std=.05)
    u.calculate()
    vals = u.out.mc.samples('f')[:10]  # Check first 10 samples
    u.calculate()
    assert not (u.out.mc.samples('f')[:10] == vals).all()

def test_change():
    ''' Test changing function name to an existing variable. Bug found in 0.09.

        setfunction -> R = V/I
        add_required_inputs
        setfunction -> V = IR
        add_required_inputs
        ==> R is now InputVar and V is InputFunc
    '''
    u = uc.UncertaintyCalc('R = V/I')
    u.add_required_inputs()
    assert 'V' in u.inputs.names
    assert 'I' in u.inputs.names
    assert isinstance(u.get_inputvar('V'), uncertainty.InputVar)
    assert isinstance(u.get_inputvar('I'), uncertainty.InputVar)

    u.set_function('I*R', idx=0, name='V')
    u.add_required_inputs()
    assert 'R' in u.inputs.names
    assert 'I' in u.inputs.names
    assert isinstance(u.get_inputvar('R'), uncertainty.InputVar)
    assert isinstance(u.get_inputvar('I'), uncertainty.InputVar)


def test_expanded():
    ''' Test expanded uncertainty of MC using shortest and symmetric intervals.
        Results should be similar for symmetric distributions
    '''
    np.random.seed(12345)
    x = np.linspace(0, 10, num=10)
    uy = .5
    y = 2*x + np.random.normal(loc=0, scale=uy, size=len(x))
    arr = curvefit.Array(x, y, uy=uy)
    fit = curvefit.CurveFit(arr)
    fit.calculate(mc=True)


def test_savesamples(tmpdir):
    ''' Test savesamples function, in txt and npz formats. '''
    np.random.seed(1111)
    u = uc.UncertCalc('f=a+b', units='meter', samples=20)
    u.set_input('a', nom=10, std=.1, units='cm')
    u.set_input('b', nom=20, std=.2, units='mm')
    u.calculate()
    sfile = os.path.join(tmpdir, 'samples.txt')
    nfile = os.path.join(tmpdir, 'samples.npz')
    u.save_samples(sfile, fmt='csv')
    u.save_samples(nfile, fmt='npz')

    # Load in and compare (only comparing output column here)
    loadedsamples = np.genfromtxt(sfile, skip_header=1)
    assert np.allclose(loadedsamples[:,2], u.out.mc.samples('f').magnitude)
    loadednpz = np.load(nfile)
    assert np.allclose(loadednpz['samples'][:,2], u.out.mc.samples('f').magnitude)
