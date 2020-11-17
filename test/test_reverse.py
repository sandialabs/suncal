''' Test reverse uncertainty and sweeps '''
import pytest

import os
import numpy as np
import sympy

import suncal as uc
from suncal import reverse
from suncal import sweeper


def test_reverse():
    ''' Test reverse uncertainty calculation by running reverse calculation, plugging in and
        verifying the forward calculation matches.
    '''
    # Example of density measurement. What is required uncertainty of weight measurement to achieve
    # desired density range?
    expr = 'rho = w / (k*d**2*h)'
    k = 12.870369  # pi/4*2.54**3, no uncertainty
    h = .5  # inch
    d = .25 # inch
    ud = .001/2
    uh = .001/2

    # Required values for rho
    rho = 2.0  # g/cm3
    urho = .06/2

    np.random.seed(234283742)
    u = reverse.UncertReverse(expr, solvefor='w', targetnom=rho, targetunc=urho)
    u.set_input('h', nom=h, std=uh)
    u.set_input('d', nom=d, std=ud)
    u.set_input('k', nom=k)
    u.add_required_inputs()
    out = u.calculate()

    # MC and GUM should match for this problem
    assert np.isclose(out.gumdata['i'].magnitude, out.mcdata['i'].magnitude, atol=.0001)
    assert np.isclose(out.gumdata['u_i'].magnitude, out.mcdata['u_i'].magnitude, atol=.0001)

    # Plugging in w to forward calculation should result in the uncertainty value we want
    u2 = uc.UncertaintyCalc(expr)
    u2.set_input('h', nom=h, std=uh)
    u2.set_input('d', nom=d, std=ud)
    u2.set_input('k', nom=k)
    u2.set_input('w', nom=out.gumdata['i'].magnitude, std=out.gumdata['u_i'].magnitude)  # Using results from reverse calc
    u2.calculate()
    assert np.isclose(u2.out.gum.nom().magnitude, rho, atol=.01)     # rho uncertainty matches the requirement
    assert np.isclose(u2.out.gum.uncert().magnitude, urho, atol=.001)


def test_reversechain():
    ''' Reverse should work when chaining functions too. '''
    u = reverse.UncertReverse(['f = x+y', 'g = f * 2'], fidx=1, solvefor='y', targetnom=10, targetunc=.5)
    u.set_input('x', nom=10, std=2)
    u.set_input('y', nom=-10, std=.5)
    u.calculate()
    assert u.out.mcdata['uf_required'] == 0.5
    assert str(u.out.gumdata['f']) == '2*x + 2*y'    # Should substitute base function for chain


def test_sweep():
    ''' Test sweeper. Sweep mean, uncertainty component, degf, and correlation '''
    u = uc.UncertCalc('f = a+b')
    u.set_input('a', nom=10, std=1)
    u.set_input('b', nom=5, std=.5)
    u.calculate()

    # Sweep nominal value
    s = sweeper.UncertSweep(u)
    s.add_sweep_nom('a', values=np.array([9, 10, 11]))
    s.calculate()
    assert s.out.get_single_desc(0) == 'a = 9.0'   # Description of each sweep index
    assert s.out.get_single_desc(1) == 'a = 10'
    assert s.out.get_rptsingle(0).gum.nom().magnitude == 14   # Verify mean values of GUM calculation
    assert s.out.get_rptsingle(1).gum.nom().magnitude == 15
    assert s.out.get_rptsingle(2).gum.nom().magnitude == 16
    assert 'f (GUM)' in s.out.get_dataset()
    assert np.allclose(s.out.get_dataset('f (GUM)').get_column('f'), np.array([14, 15, 16]))
    cfg = s.get_config()
    s2 = sweeper.UncertSweep.from_config(cfg)
    assert cfg == s2.get_config()

    # Sweep uncertainty value
    s = sweeper.UncertSweep(u)
    s.add_sweep_unc('a', values=np.array([.5, 1.0, 1.5]))
    s.calculate()
    assert s.out.get_single_desc(0) == 'u_a = 0.50'   # Description of each sweep index
    assert s.out.get_single_desc(1) == 'u_a = 1.0'
    assert np.isclose(s.out.get_rptsingle(0).gum.uncert().magnitude, np.sqrt(0.5**2 + 0.5**2))  # Uncertainties should sweep
    assert np.isclose(s.out.get_rptsingle(1).gum.uncert().magnitude, np.sqrt(1**2 +.5**2))
    assert np.isclose(s.out.get_rptsingle(2).gum.uncert().magnitude, np.sqrt(1.5**2 + .5**2))
    assert 'f (GUM)' in s.out.get_dataset()
    assert np.allclose(s.out.get_dataset('f (GUM)').get_column('f'), np.array([15, 15, 15]))  # Mean value shouldnt change
    cfg = s.get_config()
    s2 = sweeper.UncertSweep.from_config(cfg)
    assert cfg == s2.get_config()

    # Sweep degrees of freedom
    s = sweeper.UncertSweep(u)
    s.add_sweep_df('a', values=np.array([10, 20, 30]))
    s.calculate()
    assert s.out.get_single_desc(0) == 'a deg.f = 10'   # Description of each sweep index
    assert s.out.get_single_desc(1) == 'a deg.f = 20'
    assert np.isclose(s.out.get_rptsingle(0).gum.degf(), 15.625)  # Uncertainties should sweep
    assert np.isclose(s.out.get_rptsingle(1).gum.degf(), 31.25)
    assert np.isclose(s.out.get_rptsingle(2).gum.degf(), 46.875)
    cfg = s.get_config()
    s2 = sweeper.UncertSweep.from_config(cfg)
    assert cfg == s2.get_config()

    # Sweep correlation coefficient
    s = sweeper.UncertSweep(u)
    s.add_sweep_corr('a', 'b', values=np.array([-1, 0, 1]))
    s.calculate()
    assert s.out.get_single_desc(0) == 'corr = -1.0'   # Description of each sweep index
    assert s.out.get_single_desc(1) == 'corr = 0.0'
    assert np.isclose(s.out.get_rptsingle(0).gum.uncert().magnitude, 0.5)  # Uncertainties should sweep
    assert np.isclose(s.out.get_rptsingle(1).gum.uncert().magnitude, 1.118, atol=.005)
    assert np.isclose(s.out.get_rptsingle(2).gum.uncert().magnitude, 1.5)
    cfg = s.get_config()
    s2 = sweeper.UncertSweep.from_config(cfg)
    assert cfg == s2.get_config()

def test_sweepreverse():
    ''' Run a reverse sweep '''
    u = reverse.UncertReverse('f = a+b', solvefor='a', targetnom=15, targetunc=1.5)
    u.set_input('a', nom=10, std=1)
    u.set_input('b', nom=5, std=1)
    s = sweeper.UncertSweepReverse(u)
    s.add_sweep_unc('b', values=np.array([.5, 1.0, 1.5]))
    s.calculate()
    assert 'f (GUM)' in s.out.get_dataset()
    assert np.allclose(s.out.get_dataset('f (GUM)').get_column('$u_{b}$'), np.array([.5, 1, 1.5]))
    assert '$u_{b}$' in str(s.out.report())
    cfg = s.get_config()
    s2 = sweeper.UncertSweepReverse.from_config(cfg)
    assert cfg == s2.get_config()

    u = reverse.UncertReverse('f = a+b', solvefor='a', targetnom=15, targetunc=1.5)
    u.set_input('a', nom=10, std=1)
    u.set_input('b', nom=5, std=1)
    s = sweeper.UncertSweepReverse(u)
    s.add_sweep_nom('b', values=np.array([4, 5, 6]))
    s.calculate()
    assert 'f (GUM)' in s.out.get_dataset()
    assert np.allclose(s.out.get_dataset('f (GUM)').get_column('$b$'), np.array([4, 5, 6]))
    assert np.allclose(s.out.get_dataset('f (GUM)').get_column('u(a)'), np.full(3, 1.12), atol=.005)
    cfg = s.get_config()
    s2 = sweeper.UncertSweepReverse.from_config(cfg)
    assert cfg == s2.get_config()

    u = reverse.UncertReverse('f = a+b', solvefor='a', targetnom=15, targetunc=1.5)
    u.set_input('a', nom=10, std=1)
    u.set_input('b', nom=5, std=1)
    s = sweeper.UncertSweepReverse(u)
    s.add_sweep_df('b', values=np.array([5, 10, 15]))
    s.calculate()
    assert 'f (GUM)' in s.out.get_dataset()
    assert np.allclose(s.out.get_dataset('f (GUM)').get_column('$b$ deg.f'), np.array([5, 10, 15]))
    cfg = s.get_config()
    s2 = sweeper.UncertSweepReverse.from_config(cfg)
    assert cfg == s2.get_config()

    u = reverse.UncertReverse('f = a+b', solvefor='a', targetnom=15, targetunc=1.5)
    u.set_input('a', nom=10, std=1)
    u.set_input('b', nom=5, std=1)
    s = sweeper.UncertSweepReverse(u)
    s.add_sweep_corr('a', 'b', values=np.array([-.5, 0, 0.5]))
    s.calculate()
    assert 'f (GUM)' in s.out.get_dataset()
    assert np.allclose(s.out.get_dataset('f (GUM)').get_column('$corr$'), np.array([-.5, 0, 0.5]))
    cfg = s.get_config()
    s2 = sweeper.UncertSweepReverse.from_config(cfg)
    assert cfg == s2.get_config()
