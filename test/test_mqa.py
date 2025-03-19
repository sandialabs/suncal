import numpy as np

from suncal import Limit
from suncal.mqa.mqa import MqaQuantity
from suncal.mqa.measure import Typeb
from suncal.project import ProjectMqa


def test_cannonball():
    ''' Test the Cannon Ball example from Draft RP-19 '''
    ball = MqaQuantity()

    # Define the measurand and its tolerances
    ball.measurand.name = 'Canonball'
    ball.measurand.testpoint = 19.7
    ball.measurand.tolerance = Limit(19.51, 19.81)
    ball.measurand.degrade_limit = Limit(19.575, 19.808)
    ball.measurand.fail_limit = Limit(19.48, 19.827)
    ball.measurand.eopr_pct = 0.95
    ball.measurand.eopr_true = True  # 95% is a True EOPR, not an Observed EOPR

    # Specify the M&TE used to measure the ball
    ball.measurement.mte.accuracy_plusminus = .004
    ball.measurement.mte.accuracy_eopr = .95

    # No guardbanding
    ball.guardband.method = 'none'

    # Add other uncertainties
    ball.measurement.typebs = [
        Typeb('normal', name='repeatability', nominal=19.7, std=.0648),
        Typeb('uniform', name='resolution', nominal=19.7, a=.0005),
    ]

    result = ball.calculate()
    assert np.isclose(result.risk.cpfa_true, .02, atol=.05)  # Draft RP19 says 1.990%
    assert np.isclose(result.reliability.success, .956, atol=.001)  # Draft RP19 does not complete the integral...
    assert np.isclose(result.uncertainty.stdev, .0648, atol=.0001)


def test_solar():
    ''' Test Solar Experiment example from Draft RP-19 '''
    proj = ProjectMqa.from_configfile(r'test\ex_solarexperiment.yaml')
    result = proj.calculate()

    solar = result.quantities[0]
    lamp = solar.uncertainty.parent
    comp = lamp.uncertainty.parent

    # RP19 Table 5-8
    assert np.isclose(solar.eopr.true.pct, .99994, atol=.00001)        # True EOP = 99.994%
    assert np.isclose(solar.reliability.aop.pct, .99996, atol=.00001)  # True AOP = 99.99(7)%
    assert np.isclose(solar.risk.pfa_true, 0.0, atol=.00005)           # PFA = 0.00%
    assert np.isclose(solar.risk.pfr_true, 0.0, atol=.00005)           # PFR = 0.00%

    # RP19 Table 5-9
    assert np.isclose(lamp.eopr.true.pct, .9986, atol=.0001)           # True EOP = 99.86%
    assert np.isclose(lamp.reliability.aop.pct, .9991, atol=.00005)    # True AOP = 99.91%
    assert np.isclose(lamp.risk.pfa_true, 0.0004, atol=.00005)         # PFA = 0.04%
    assert np.isclose(lamp.risk.pfr_true, 0.0017, atol=.00005)         # PFR = 0.17%

    # RP19 Table 5-10
    assert np.isclose(comp.eopr.true.pct, .9527, atol=.0001)           # True EOP = 95.27%
    assert np.isclose(comp.reliability.aop.pct, .9733, atol=.0004)     # True AOP = 97.33%
    assert np.isclose(comp.reliability.bop.pct, .9943, atol=.0004)     # True BOP = 99.43%
    assert np.isclose(comp.risk.pfa_true, 0.0057, atol=.00005)         # PFA = 0.57%
    assert np.isclose(comp.risk.pfr_true, 0.0084, atol=.00005)         # PFR = 0.84%

    total = solar.total_costs()

    # Table 5-12
    assert np.isclose(total.cal, 176637, atol=1)       # RP19 = 176637
    assert np.isclose(total.adj, 10, atol=5)           # RP19 = 10
    assert np.isclose(total.support, 178687, atol=1)   # RP19 = 178687
    assert np.isclose(total.total, 178690, atol=15)    # RP19 = 178690
    assert np.isclose(total.spare_cost, 29801, atol=5) # RP19 = 29801
