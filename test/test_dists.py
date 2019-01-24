''' Test custom distributions '''
import numpy as np
from scipy import stats
import psluncert.customdists as customdists

# First cases are wrappers around scipy.stats distributions
def test_uniform():
    u = customdists.uniform(2)
    assert np.isclose(u.std(), 2/np.sqrt(3))   # Standard uncertainty of uniform distribution half-width 2
    assert np.isclose(u.mean(), 0)


def test_tri():
    u = customdists.triangular(3)
    assert np.isclose(u.std(), 3/np.sqrt(6))  # Std. Uncertainty of triangular distribution
    assert np.isclose(u.mean(), 0)


def test_arcsine():
    u = customdists.arcsine(5)
    assert np.isclose(u.mean(), 0)
    assert np.isclose(u.std(), 5/np.sqrt(2))  # Std. Uncertainty of arcsine distribution


# Curvilinear trapezoid is a custom subclass of stats.rv_continuous, so do a bit more testing on it.
def test_curvtrap():
    # Check variance/mean values
    a = 2; d = .5
    u = customdists.curvtrap(a=a, d=d)
    assert np.isclose(u.mean(), 0)
    assert np.isclose(u.var(), (2*a)**2/12 + d**2/9)  # Std Uncertainty of Ctrap (see GUM-S1, 6.4.3.3)

    a = 4; d = 1
    u = customdists.curvtrap(a=a, d=d)
    assert np.isclose(u.var(), (2*a)**2/12 + d**2/9)

    u = customdists.curvtrap(a=2, d=3)  # d can't be > a. Should return nan.
    assert not np.isfinite(u.mean())
    assert not np.isfinite(u.std())

    # PDF should integrate to ~1
    u = customdists.curvtrap(a=2, d=.5)
    x = np.linspace(-3,3,1000)
    integ = np.trapz(u.pdf(x), x)
    assert np.isclose(integ, 1)

    # Check PDF at a few points - see GUM-S1 6.4.3.2
    a = 1; d=.1
    # w [in gum] = a [in here] because it's symmetric here
    # x [in gum] = 0 (dist has mean=0)
    c = 1/(4*d)
    u = customdists.curvtrap(a=a, d=d)
    assert u.pdf(-1.5) == 0                                    # Out of range
    assert np.isclose(u.pdf(-0.95), (c * np.log((a+d)/.95)))   # Curving up
    assert np.isclose(u.pdf(0),     (c * np.log((a+d)/(a-d)))) # Flat area
    assert np.isclose(u.pdf(0.95),  (c * np.log((a+d)/.95)))   # Curving down
    assert u.pdf(1.5) == 0                                     # Out of range

    # Generate random samples, check range
    np.random.seed(1234)
    a = 3; d = 2
    u = customdists.curvtrap(a=a, d=d)
    samples = u.rvs(size=1000000)
    assert ((samples<a+d) & (samples>-a-d)).all()

    # Compare sample histogram with PDF (roughly)
    y, x = np.histogram(samples, bins=25, density=True)
    x = x[1:] - (x[1]-x[0])/2  # Shift bin edges back to match centers of bins
    assert np.allclose(y, u.pdf(x), rtol=.05)


def test_discrete():
    # Test discrete histogram and piecewise distributions
    np.random.seed(883355)
    x = np.random.normal(loc=5, scale=2, size=100000)
    d = customdists.hist(data=x)
    assert np.isclose(d.mean(), x.mean(), rtol=.01)
    assert np.isclose(d.std(), x.std(), rtol=.01)
    assert np.isclose(d.cdf(np.median(x)), 0.5, rtol=.01)  # Midpoint of pdf ~ 1/2

    # Pdf from x, y array
    xx = np.linspace(-10, 20, num=200)
    yy = stats.norm.pdf(xx, loc=5, scale=2)
    d2 = customdists.piecewise(xx, yy)
    assert np.isclose(d2.mean(), 5, rtol=.05)
    assert np.isclose(d2.std(), 2, rtol=.05)


def test_config():
    # Test customdists get_config and from_config

    # Check a custom distribution
    config = {'dist': 'curvtrap', 'median': 5, 'a': 1, 'd': .5}
    dist = customdists.from_config(config)
    assert np.isclose(dist.median(), 5)
    config2 = customdists.get_config(dist)
    assert config == config2

    # Check a scipy.stats distribution using loc/scale
    config = {'dist': 'expon', 'loc': 1, 'scale': 2}
    dist = customdists.from_config(config)
    config2 = customdists.get_config(dist)
    assert config == config2

    # Check a customized distribution with same name as scipy.stats distribution - use median/std params
    config = {'dist': 't', 'median': -1, 'std': 2, 'df': 10}
    dist = customdists.from_config(config)
    assert np.isclose(dist.median(), -1)
    config2 = customdists.get_config(dist)
    assert config == config2

    # Check a scipy.stats distribution with same name as customdist - use loc/scale params
    config = {'dist': 't', 'loc': -1, 'scale': 2, 'df': 10}
    dist = customdists.from_config(config)
    assert np.isclose(dist.median(), -1)
    config2 = customdists.get_config(dist)
    assert config == config2

    # Check a custom distribution that is named in both stats and customdists
    config = {'dist': 'arcsine', 'a': 2, 'median': 3}
    dist = customdists.from_config(config)
    assert np.isclose(dist.median(), 3)
    config2 = customdists.get_config(dist)
    assert config == config2


def test_customconfig():
    # Test round-trip get/from config on all other custom types
    config = {'dist': 'uniform', 'median': 5, 'a': 2}
    dist = customdists.from_config(config)
    assert np.isclose(dist.median(), 5)
    config2 = customdists.get_config(dist)
    assert config == config2

    config = {'dist': 'uniform', 'loc': 2, 'scale': 2}   # scipy.stats uniform
    dist = customdists.from_config(config)
    assert np.isclose(dist.median(), 3)
    config2 = customdists.get_config(dist)
    assert config == config2

    config = {'dist': 'normal', 'median': 5, 'std': 2}
    dist = customdists.from_config(config)
    assert np.isclose(dist.median(), 5)
    config2 = customdists.get_config(dist)
    assert config == config2

    config = {'dist': 'triangular', 'median': -3, 'a': 3}
    dist = customdists.from_config(config)
    assert np.isclose(dist.median(), -3)
    config2 = customdists.get_config(dist)
    assert config == config2

    config = {'dist': 'resolution', 'median': 10, 'a': 1}
    dist = customdists.from_config(config)
    assert np.isclose(dist.median(), 10)
    config2 = customdists.get_config(dist)
    assert config == config2
