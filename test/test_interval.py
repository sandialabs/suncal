''' Test cases for itervals module '''

import pytest

import numpy as np
from suncal.intervals import BinomialInterval, VariablesInterval
from suncal.intervals import TestInterval as _TestInterval  # Use a different name or pytest thinks this class is a test case!
from suncal.intervals import TestIntervalAssets as _TestIntervalAssets


def testA3():
    ''' Test method A3 - TestInterval Method '''
    # Asset 1 from 2019 NCSL Symposium Tutorial on Intervals
    y = np.array([1,1,0,1,0,1,1,1,1,0,1,1,1,1,1,0,1,1,1,0,])
    intv = _TestIntervalAssets(I0=365, Rt=.95)  # Existing interval I0=365 days.
    intv.updateasset('A', startdates=np.arange(0, 365*len(y), 365),
                     enddates=np.arange(365, 365*(len(y)+1), 365),
                     passfail=y)
    intv.calculate()
    assert np.round(intv.out.interval) == 231
    assert np.isclose(intv.out.results['rejection'], .9949, atol=.00005)  # Rejection conf
    assert np.isclose(intv.out.results['RL'], .6516, atol=.00005)         # Lower reliability conf limit
    assert np.isclose(intv.out.results['RU'], .8286, atol=.00005)         # Upper reliability conf limit

    # Asset 2 from 2019 NCSL Symposium Tutorial on Intervals
    y = np.array([1,1,1,0,1,0,0,1,1,1,0,0,1,1,1,0,1,1,1,1])
    intv = _TestIntervalAssets(I0=365, Rt=.95)  # Existing interval I0=365 days.
    intv.updateasset('A', startdates=np.arange(0, 365*len(y), 365),
                     enddates=np.arange(365, 365*(len(y)+1), 365),
                     passfail=y)
    intv.calculate()
    assert np.round(intv.out.interval) == 205
    assert np.isclose(intv.out.results['rejection'], .9993, atol=.00005)  # Rejection conf
    assert np.isclose(intv.out.results['RL'], .6000, atol=.00005)         # Lower reliability conf limit
    assert np.isclose(intv.out.results['RU'], .7843, atol=.00005)         # Upper reliability conf limit

    # Asset 3 from 2019 NCSL Symposium Tutorial on Intervals
    y = np.array([1,1,1,1,0,1,0,1,1,1,0,1,1,1,1,1,1,0,1,1])
    intv = _TestIntervalAssets(I0=730, Rt=.95)
    intv.updateasset('A', startdates=np.arange(0, 730*len(y), 730),
                     enddates=np.arange(730, 730*(len(y)+1), 730),
                     passfail=y)
    intv.calculate()
    assert np.round(intv.out.interval) == 523
    assert np.isclose(intv.out.results['rejection'], .9682, atol=.00005)  # Rejection conf
    assert np.isclose(intv.out.results['RL'], .7041, atol=.00005)         # Lower reliability conf limit
    assert np.isclose(intv.out.results['RU'], .8716, atol=.00005)         # Upper reliability conf limit

    # Other made up numbers, input directly without from_data()
    intv = _TestInterval(intol=42, n=50, I0=365, Rt=.9, conf=.3)
    intv.calculate()
    assert np.round(intv.out.interval) == 329
    assert np.isclose(intv.out.results['rejection'], .7557, atol=.00005)  # Rejection conf
    assert np.isclose(intv.out.results['RL'], .8067, atol=.00005)         # Lower reliability conf limit
    assert np.isclose(intv.out.results['RU'], .8664, atol=.00005)         # Upper reliability conf limit


@pytest.mark.filterwarnings('ignore')  # Will generate runtime/optimize warning in minimization loop
def test_variables():
    ''' Test variables method using data from NASA-HDBK-8739.19-5 '''
    # Data in Table 7-2.
    dt = np.array([70., 86., 104., 135., 167., 173.])
    deltas = np.array([.1, .11, .251, .299, .403, .615])
    intv = VariablesInterval(dt, deltas, u0=.28, m=2, y0=10.03, utarget=.5, rlimits=(9, 11), rconf=.9)
    intv.calculate()

    # Compare curve-fit results with Table 7-3
    assert np.isclose(intv.out.fit.b[0], 0.00015741)
    assert np.isclose(intv.out.fit.b[1], 0.00001674)
    assert np.isclose(intv.out.fit.cov[0,0], 0.00000101)
    assert np.isclose(intv.out.fit.cov[1,1], 4.549E-11)
    assert np.isclose(intv.out.fit.cov[0,1], -6.607E-9)
    assert np.isclose(intv.out.fit.syx, 0.0708, atol=.00005)

    # Compare reliability target method with Table 7-4
    assert np.isclose(intv.out.reliability.interval, 140.1, atol=.05)
    assert np.isclose(intv.out.fit.predict_deviation(intv.out.reliability.interval)[0], 10.38, atol=.005)

    # Compare uncertainty target method with Table 7-7
    assert np.isclose(intv.out.uncertainty.interval, 327.15, atol=.005)
    assert np.isclose(intv.out.fit.predict_deviation(intv.out.uncertainty.interval)[0], 11.87, atol=.005)

    # Compute reliability model with one-sided limits, compare with Table 7-5 and 7-6
    intv.rlimits = (None, 11)
    intv.calc_reliability_target()
    assert np.isclose(intv.out.reliability.interval, 171.75, atol=.005)

    # Another problem in NASA-8739. They calculate the single-sided interval case
    # with results in Table 7-6, but the initial uncertainty is already less
    # than the limit at t=0! Assigned interval should be 0 or N/A. Table 7-6
    # lists 153.04, which is correct in that's when the 95% conf line crosses
    # the lower limit, but going the wrong direction.
    intv = VariablesInterval(dt, deltas, u0=.28, m=2, y0=9.03, utarget=.5, rconf=.9)
    intv.calc_reliability_target()
    assert np.isclose(intv.out.reliability.interval, 0, atol=.005)  # Table 7-6
