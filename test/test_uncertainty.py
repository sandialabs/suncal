''' Test cases for uncertainty core. '''
import pytest

import os
import numpy as np
import sympy

from suncal import Model, ModelCallable
from suncal.project import ProjectUncert
from suncal.common import unitmgr


def test_chain():
    ''' Test chaining of functions '''
    u = Model('f=x+y', 'g=2*f', 'h=2*g')
    np.random.seed(0)
    u.var('x').measure(100).typeb(std=1)
    u.var('y').measure(50).typeb(std=1)
    gum = u.calculate_gum()
    mc = u.monte_carlo()

    assert np.isclose(gum.expected['f'], gum.expected['g']/2)
    assert np.isclose(gum.expected['g'], gum.expected['h']/2)
    assert np.isclose(gum.uncertainty['f'], gum.uncertainty['g']/2)
    assert np.isclose(gum.uncertainty['g'], gum.uncertainty['h']/2)

    assert np.isclose(mc.expected['f'], mc.expected['g']/2, rtol=.02)
    assert np.isclose(mc.expected['g'], mc.expected['h']/2, rtol=.02)
    assert np.isclose(mc.uncertainty['f'], mc.uncertainty['g']/2, rtol=.05)
    assert np.isclose(mc.uncertainty['g'], mc.uncertainty['h']/2, rtol=.05)

    # Double-chaining, order shouldn't matter
    u2 = Model('g = a*b*f', 'f = a+b*c+h', 'h=a+d')
    u2.var('a').measure(10).typeb(std=1)
    u2.var('b').measure(10).typeb(std=1)
    u2.var('c').measure(1).typeb(std=.5)
    u2.var('d').measure(1).typeb(std=.5)
    u2.calculate_gum()
    symbols = u2.variables.names
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

    u = ModelCallable(myfunc)
    np.random.seed(0)
    assert 'a' in u.variables.names
    assert 'b' in u.variables.names
    assert len(u.variables.names) == 2

    u.var('a').measure(5).typeb(std=0.05)
    u.var('b').measure(2).typeb(std=0.02)
    assert u.variables.expected['a'] == 5

    gum = u.calculate_gum()
    assert np.isclose(gum.expected['myfunc'], 9)


def test_callablekwargs():
    ''' Test callable with **kwargs '''
    def myfunc(**kwargs):
        x = kwargs.get('x')
        y = kwargs.get('y')
        return x * y

    with pytest.raises(ValueError):  # Error if function takes unnamed kwargs and kwnames parameter not specified
        u = ModelCallable(myfunc)

    u = ModelCallable(myfunc, argnames=['x', 'y'])
    np.random.seed(0)
    assert 'x' in u.variables.names
    assert 'y' in u.variables.names
    assert len(u.variables.names) == 2

    u.var('x').measure(2).typeb(std=0.1)
    u.var('y').measure(4).typeb(std=0.2)
    gum = u.calculate_gum()
    assert np.isclose(gum.expected['myfunc'], 8)


def test_chaincallable():
    ''' Test chaining callable functions '''
    def myfunc1(x, y):
        return x * y

    def myfunc2(myfunc1):
        return myfunc1 + 100

    def myfunc(x, y):
        m = myfunc1(x, y)
        return m, myfunc2(m)

    u = ModelCallable(myfunc)
    np.random.seed(0)
    assert 'x' in u.variables.names
    assert 'y' in u.variables.names
    assert len(u.variables.names) == 2

    u.var('x').measure(2).typeb(std=0.1)
    u.var('y').measure(10).typeb(std=0.5)
    gum = u.calculate_gum()
    u.monte_carlo(samples=10)
    assert np.isclose(gum.expected['myfunc_1'], 20)
    assert np.isclose(gum.expected['myfunc_2'], 120)


@pytest.mark.filterwarnings('ignore')  # Will generate unitstripped warning due to use of np.vectorize with unit values
def test_vectorize():
    ''' Make sure non-vectorized functions can run. Also tests function with kwargs arguments '''
    # This function is not vectorizable as-is. Calculator will try it, fail, and then
    # try using np.vectorize() on it.
    def tcr(**kwargs):
        ''' Temperature Coefficient of Resistance from pairs of R, T measurements. '''
        R = np.array([kwargs.get(f'R{i+1}') for i in range(len(kwargs)//2)])
        T = np.array([kwargs.get(f'T{i+1}') for i in range(len(kwargs)//2)])
        p = np.polyfit(T-T[0], R/R[0]-1, deg=1)[0]
        return p

    varnames = [f'T{i+1}' for i in range(4)] + [f'R{i+1}' for i in range(4)]

    np.random.seed(34234)
    u = ModelCallable(tcr, argnames=varnames)
    for i, rval in enumerate([100, 100.1, 100.2, 100.3]):
        Rname = f'R{i+1}'
        u.var(Rname).measure(rval).typeb(std=0.2)
    for i, tval in enumerate([20, 22, 24, 26]):
        Tname = f'T{i+1}'
        u.var(Tname).measure(tval).typeb(std=0.05)
    mc = u.monte_carlo(samples=1000)
    assert np.isclose(mc.expected['tcr'], 0.0005, atol=.0001)


@pytest.mark.filterwarnings('ignore')  # Will generate unitstripped warning due to use of np.vectorize with unit values
def test_vectorize_units():
    def test(x):
        ''' A function that will fail if called with an array '''
        if x > 0:
            return x*2
        else:
            return x

    # First run has no units
    u = ModelCallable(test)
    u.var('x').measure(100).typeb(std=1)
    mc = u.monte_carlo(samples=1000)
    assert np.isclose(mc.expected['test'], 200, atol=1)
    assert not unitmgr.has_units(mc.expected['test'])

    # Now input has units but output units are not specified
    u = ModelCallable(test)
    u.var('x').measure(100, units='cm').typeb(std=1, units='cm')
    mc = u.monte_carlo(samples=1000)
    assert np.isclose(mc.expected['test'], unitmgr.make_quantity(200, 'cm'), atol=1)
    assert str(mc.expected['test'].units) == 'centimeter'


@pytest.mark.filterwarnings('ignore')  # Will generate a np warning about degrees of freedom <= 0
def test_constant():
    ''' Functions can also be constant with no variables. '''
    u = Model('a=10', 'g=a+b')
    u.var('b').measure(5).typeb(std=0.1)
    gum = u.calculate_gum()
    assert np.isclose(gum.expected['a'], 10)
    assert np.isclose(gum.expected['g'], 15)


def test_readconfig():
    ''' Test read_configfile '''
    u = ProjectUncert.from_configfile(os.path.join('test', 'test1.yaml'))
    assert u.model.functionnames == ['f', 'g', 'h']
    assert u.model.exprs == ['(a + b) / c', 'a - b', 'b * c']
    assert u.nsamples == 1E6
    assert len(u.model.variables.names) == 3
    a = u.model.var('a')
    assert a.value == 10.0
    assert float(a._typeb[0].kwargs['std']) == 0.2
    assert a._typeb[0].distname == 'normal'
    assert a.degrees_freedom == 10
    b = u.model.var('b')
    assert b.value == 25.0
    assert float(b._typeb[0].kwargs['scale']) == 2.0
    assert float(b._typeb[0].kwargs['a']) == 5.0
    assert b._typeb[0].distname == 'gamma'
    assert b.degrees_freedom == np.inf
    c = u.model.var('c')
    assert c.value == 2.0
    assert float(c._typeb[0].kwargs['std']) == 0.1
    assert c._typeb[0].distname == 'normal'
    assert c.degrees_freedom == 88
    assert u.model.variables.get_correlation_coeff('b', 'a') == -0.36
    assert u.model.variables.get_correlation_coeff('a', 'c') == -0.4
    assert u.model.variables.get_correlation_coeff('b', 'c') == 0.86


def test_saveconfig():
    ''' Test save_config and read_config, we can go in a circle. '''
    CHECK_FILE = os.path.join('test', 'TEST_SAVE.YAML')

    u = ProjectUncert.from_configfile(os.path.join('test', 'test1.yaml'))
    u.save_config(CHECK_FILE)
    u2 = ProjectUncert.from_configfile(CHECK_FILE)

    assert u.model.sympys == u2.model.sympys
    assert u.model.variables.names[0] == u.model.variables.names[0]
    assert u.model.variables.names[1] == u.model.variables.names[1]
    assert u.model.variables.names[2] == u.model.variables.names[2]
    assert np.allclose(u.model.variables.correlation_matrix(),
                       u2.model.variables.correlation_matrix(),
                       rtol=1E-9, atol=0)
    assert u.nsamples == u2.nsamples
    os.remove(CHECK_FILE)


def test_addinputs():
    ''' Test add_required_inputs() function '''
    # For a string function...
    u = Model('a + b + c + d')
    assert len(u.variables.names) == 4
    assert 'b' in u.variables.names
    assert 'c' in u.variables.names
    assert 'd' in u.variables.names

    # and a sympy function...
    x, y, z = sympy.symbols('x y z')
    f = x+y+z
    u = Model(f)
    assert len(u.variables.names) == 3
    assert 'x' in u.variables.names
    assert 'y' in u.variables.names
    assert 'z' in u.variables.names

    # and for callable...
    def myfunc(j, k, l):
        return j+k+l
    u = ModelCallable(myfunc)
    assert len(u.variables.names) == 3
    assert 'j' in u.variables.names
    assert 'k' in u.variables.names
    assert 'l' in u.variables.names


@pytest.mark.filterwarnings('ignore')  # Will generate a np warning about degrees of freedom <= 0
def test_reserved():
    ''' Test that reserved sympy keywords/functions are properly handled. '''
    # "pi" is 3.14, not a symbol
    u = Model('pi')
    gum = u.calculate_gum()
    assert np.isclose(gum.expected['f1'], np.pi)

    # "gamma" is a symbol, not gamma function
    u = Model('gamma/2')
    u.var('gamma').measure(10)
    gum = u.calculate_gum()
    assert np.isclose(gum.expected['f1'], 5)

    # But '"cos" is the sympy cosine, not a variable
    u = Model('cos(x)')
    u.var('x').measure(np.pi)
    gum = u.calculate_gum()
    assert np.isclose(gum.expected['f1'], np.cos(np.pi))


def test_seed():
    ''' Test random seed argument '''
    # With seed, outputs will be the same
    u = Model('f=a+b')
    u.var('a').measure(10).typeb(std=.1)
    u.var('b').measure(5).typeb(std=0.05)
    proj = ProjectUncert(u)
    proj.seed = 10
    result = proj.calculate()
    vals = result.montecarlo.samples['f'][:10]  # Check first 10 samples
    assert (result.montecarlo.samples['f'][:10] == vals).all()

    # With seed=None, seed is randomized
    proj = ProjectUncert(u)
    proj.seed = None
    result = proj.calculate()
    assert not (result.montecarlo.samples['f'][:10] == vals).all()


def test_savesamples(tmpdir):
    ''' Test savesamples function, in txt and npz formats. '''
    np.random.seed(1111)
    u = Model('f=a+b')
    u.var('a').measure(10, units='cm').typeb(std=.1, units='cm')
    u.var('b').measure(20, units='mm').typeb(std=.2, units='mm')
    proj = ProjectUncert(u)
    proj.nsamples = 20
    proj.outunits = {'f': 'meter'}
    result = proj.calculate()

    sfile = os.path.join(tmpdir, 'samples.txt')
    nfile = os.path.join(tmpdir, 'samples.npz')
    proj.save_samples_csv(sfile)
    proj.save_samples_npz(nfile)

    # Load in and compare (only comparing output column here)
    loadedsamples = np.genfromtxt(sfile, skip_header=1)
    assert np.allclose(loadedsamples[:, 2], result.montecarlo.samples['f'].magnitude)
    loadednpz = np.load(nfile)
    assert np.allclose(loadednpz['samples'][:, 2], result.montecarlo.samples['f'].magnitude)
