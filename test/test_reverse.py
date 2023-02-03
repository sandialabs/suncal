''' Test reverse uncertainty and sweeps '''
import pytest

import numpy as np

from suncal import Model
from suncal.reverse import ModelReverse
from suncal.sweep import UncertSweep, UncertSweepReverse
from suncal.project import ProjectSweep, ProjectReverseSweep


def test_reverse():
    ''' Test reverse uncertainty calculation by running reverse calculation, plugging in and
        verifying the forward calculation matches.
    '''
    # Example of density measurement. What is required uncertainty of weight measurement to achieve
    # desired density range?
    expr = 'rho = w / (k*d**2*h)'
    k = 12.870369  # pi/4*2.54**3, no uncertainty
    h = .5  # inch
    d = .25  # inch
    ud = .001/2
    uh = .001/2

    # Required values for rho
    rho = 2.0  # g/cm3
    urho = .06/2

    np.random.seed(234283742)
    u = ModelReverse(expr, solvefor='w', targetnom=rho, targetunc=urho)
    u.var('h').measure(h).typeb(std=uh)
    u.var('d').measure(d).typeb(std=ud)
    u.var('k').measure(k)
    gum = u.calculate_gum()
    mc = u.monte_carlo()

    # MC and GUM should match for this problem
    assert np.isclose(gum.solvefor_value, mc.solvefor_value, atol=.0001)
    assert np.isclose(gum.u_solvefor_value, mc.u_solvefor_value, atol=.0001)

    # Plugging in w to forward calculation should result in the uncertainty value we want
    u2 = Model(expr)
    u2.var('h').measure(h).typeb(std=uh)
    u2.var('d').measure(d).typeb(std=ud)
    u2.var('k').measure(k)
    u2.var('w').measure(gum.solvefor_value).typeb(std=gum.u_solvefor_value)  # Using results from reverse calc
    gum = u2.calculate_gum()
    assert np.isclose(gum.expected['rho'], rho, atol=.01)     # rho uncertainty matches the requirement
    assert np.isclose(gum.uncertainty['rho'], urho, atol=.001)


def test_reversechain():
    ''' Reverse should work when chaining functions too. '''
    u = ModelReverse('f = x+y', 'g = f * 2', funcname='g', solvefor='y', targetnom=10, targetunc=.5)
    u.var('x').measure(10).typeb(std=2)
    u.var('y').measure(-10).typeb(std=.5)
    gum = u.calculate_gum()
    assert gum.uf_required == 0.5
    assert str(gum.function) == '2*x + 2*y'    # Should substitute base function for chain


def test_sweep():
    ''' Test sweeper. Sweep mean, uncertainty component, degf, and correlation '''
    u = Model('f = a+b')
    u.var('a').measure(10).typeb(name='u(a)', std=1)
    u.var('b').measure(5).typeb(std=.5)

    # Sweep nominal value
    s = UncertSweep(u)
    s.add_sweep_nom('a', values=np.array([9, 10, 11]))
    proj = ProjectSweep(s)
    result = proj.calculate()
    assert result.report.describe(0) == 'a = 9.0'   # Description of each sweep index
    assert result.report.describe(1) == 'a = 10'
    assert result.gum[0].expected['f'] == 14   # Verify mean values of GUM calculation
    assert result.gum[1].expected['f'] == 15
    assert result.gum[2].expected['f'] == 16
    assert 'f (GUM)' in proj.get_dataset()
    assert np.allclose(proj.get_dataset('f (GUM)').get_column('f'), np.array([14, 15, 16]))
    cfg = proj.get_config()
    s2 = ProjectSweep.from_config(cfg)
    assert cfg['sweeps'] == s2.get_config()['sweeps']

    # Sweep uncertainty value
    s = UncertSweep(u)
    s.add_sweep_unc('a', values=np.array([.5, 1.0, 1.5]))
    proj = ProjectSweep(s)
    result = proj.calculate()
    assert result.report.describe(0) == 'u_a = 0.50'   # Description of each sweep index
    assert result.report.describe(1) == 'u_a = 1.0'
    assert np.isclose(result.gum[0].uncertainty['f'], np.sqrt(0.5**2 + 0.5**2))  # Uncertainties should sweep
    assert np.isclose(result.gum[1].uncertainty['f'], np.sqrt(1**2 + .5**2))
    assert np.isclose(result.gum[2].uncertainty['f'], np.sqrt(1.5**2 + .5**2))
    assert 'f (GUM)' in proj.get_dataset()
    assert np.allclose(proj.get_dataset('f (GUM)').get_column('f'), np.array([15, 15, 15]))  # Mean value shouldnt change
    cfg = proj.get_config()
    s2 = ProjectSweep.from_config(cfg)
    assert cfg['sweeps'] == s2.get_config()['sweeps']

    # Sweep degrees of freedom
    s = UncertSweep(u)
    s.add_sweep_df('a', values=np.array([10, 20, 30]))
    proj = ProjectSweep(s)
    result = proj.calculate()
    assert result.report.describe(0) == 'a deg.f = 10'   # Description of each sweep index
    assert result.report.describe(1) == 'a deg.f = 20'
    assert np.isclose(result.gum[0].degf['f'], 15.625)  # Uncertainties should sweep
    assert np.isclose(result.gum[1].degf['f'], 31.25)
    assert np.isclose(result.gum[2].degf['f'], 46.875)
    cfg = proj.get_config()
    s2 = ProjectSweep.from_config(cfg)
    assert cfg['sweeps'] == s2.get_config()['sweeps']

    # Sweep correlation coefficient
    s = UncertSweep(u)
    s.add_sweep_corr('a', 'b', values=np.array([-1, 0, 1]))
    proj = ProjectSweep(s)
    result = proj.calculate()
    assert result.report.describe(0) == 'corr = -1.0'   # Description of each sweep index
    assert result.report.describe(1) == 'corr = 0.0'
    assert np.isclose(result.gum[0].uncertainty['f'], 0.5)  # Uncertainties should sweep
    assert np.isclose(result.gum[1].uncertainty['f'], 1.118, atol=.005)
    assert np.isclose(result.gum[2].uncertainty['f'], 1.5)
    cfg = proj.get_config()
    s2 = ProjectSweep.from_config(cfg)
    assert cfg['sweeps'] == s2.get_config()['sweeps']


def test_sweepreverse():
    ''' Run a reverse sweep '''
    s = UncertSweepReverse('f=a+b', solvefor='a', targetnom=15, targetunc=1.5)
    s.model.var('a').measure(10).typeb(std=1)
    s.model.var('b').measure(5).typeb(name='u(b)', std=1)
    s.add_sweep_unc('b', values=np.array([.5, 1.0, 1.5]))
    proj = ProjectReverseSweep(s)
    result = proj.calculate()
    assert 'f (GUM)' in proj.get_dataset()
    assert np.allclose(proj.get_dataset('f (GUM)').get_column('$u_{b}$'), np.array([.5, 1, 1.5]))
    assert '$u_{b}$' in str(result.report.summary())
    cfg = proj.get_config()
    s2 = ProjectReverseSweep.from_config(cfg)
    assert cfg['sweeps'] == s2.get_config()['sweeps']

    s = UncertSweepReverse('f = a+b', solvefor='a', targetnom=15, targetunc=1.5)
    s.model.var('a').measure(10).typeb(std=1)
    s.model.var('b').measure(5).typeb(name='u(b)', std=1)
    s.add_sweep_nom('b', values=np.array([4, 5, 6]))
    proj = ProjectReverseSweep(s)
    result = proj.calculate()
    assert 'f (GUM)' in proj.get_dataset()
    assert np.allclose(proj.get_dataset('f (GUM)').get_column('$b$'), np.array([4, 5, 6]))
    assert np.allclose(proj.get_dataset('f (GUM)').get_column('u(f)'), np.full(3, 1.12), atol=.005)
    cfg = proj.get_config()
    s2 = ProjectReverseSweep.from_config(cfg)
    assert cfg['sweeps'] == s2.get_config()['sweeps']

    s = UncertSweepReverse('f = a+b', solvefor='a', targetnom=15, targetunc=1.5)
    s.model.var('a').measure(10).typeb(std=1)
    s.model.var('b').measure(5).typeb(name='u(b)', std=1)
    s.add_sweep_df('b', values=np.array([5, 10, 15]))
    proj = ProjectReverseSweep(s)
    result = proj.calculate()
    assert 'f (GUM)' in proj.get_dataset()
    assert np.allclose(proj.get_dataset('f (GUM)').get_column('$b$ deg.f'), np.array([5, 10, 15]))
    cfg = proj.get_config()
    s2 = ProjectReverseSweep.from_config(cfg)
    assert cfg['sweeps'] == s2.get_config()['sweeps']

    s = UncertSweepReverse('f = a+b', solvefor='a', targetnom=15, targetunc=1.5)
    s.model.var('a').measure(10).typeb(std=1)
    s.model.var('b').measure(5).typeb(name='u(b)', std=1)
    s.add_sweep_corr('a', 'b', values=np.array([-.5, 0, 0.5]))
    proj = ProjectReverseSweep(s)
    result = proj.calculate()
    assert 'f (GUM)' in proj.get_dataset()
    assert np.allclose(proj.get_dataset('f (GUM)').get_column('$corr$'), np.array([-.5, 0, 0.5]))
    cfg = proj.get_config()
    s2 = ProjectReverseSweep.from_config(cfg)
    assert cfg['sweeps'] == s2.get_config()['sweeps']
