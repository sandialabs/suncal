''' Test implicit measurement models '''
import numpy as np

from suncal import Model



def test_implicit():
    ''' Test reverse uncertainty calculation as an implicit model '''
    np.random.seed(7777)
    u = Model(
        'mw*gl*(1-rho_a/rho_w)/(A0*(1+lambd*p)*(1+alpha*dT)) - p = 0; p~4.1E4',
    )
    u.var('mw').measure(1.4, units='kg').typeb(std=1.5E-6, units='kg')
    u.var('gl').measure(9.792, units='m/s^2').typeb(std=2.5E-9, units='m/s^2')
    u.var('A0').measure(3.35E-4, units='m^2').typeb(std=1E-5, units='m^2')
    u.var('lambd').measure(4E-6, units='Pa^-1')
    u.var('alpha').measure(2E-5, units='1/K')
    u.var('dT').measure(2.2, units='delta_degC').typeb(std=5E-2, units='delta_degC')
    u.var('rho_a').measure(7.844E-8, units='kg/m^3').typeb(std=6.42E-8, units='kg/m^3')
    u.var('rho_w').measure(7800, units='kg/m^3').typeb(std=45.033, units='kg/m^3')
    gum = u.calculate_gum()
    mc = u.monte_carlo(samples=5000)

    assert np.isclose(gum.expected['p'].magnitude, 35790, atol=300)
    assert(np.isclose(gum.expected['p'], mc.expected['p'], atol=100))
    assert(np.isclose(gum.uncertainty['p'], mc.uncertainty['p'], atol=20))