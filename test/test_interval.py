''' Test cases for itervals module '''

import pytest

import numpy as np
from suncal.intervals import S2Params, s2_binom_interval
from suncal.intervals import A3Params, a3_testinterval
from suncal.intervals import VariablesData, variables_reliability_target, variables_uncertainty_target


def testA3():
    ''' Test method A3 - TestInterval Method '''
    # Asset 1 from 2019 NCSL Symposium Tutorial on Intervals
    y = np.array([1,1,0,1,0,1,1,1,1,0,1,1,1,1,1,0,1,1,1,0,])
    params = A3Params.from_assets([{'startdates': np.arange(0, 365*len(y), 365),
                                    'enddates': np.arange(365, 365*(len(y)+1), 365),
                                    'passfail': y
                                    }],
                                  target=0.95,
                                  I0=365)
    result = a3_testinterval(params)
    assert np.round(result.interval) == 231
    assert np.isclose(result.rejection, .9949, atol=.00005)  # Rejection conf
    assert np.isclose(result.reliability_range[0], .6516, atol=.00005)         # Lower reliability conf limit
    assert np.isclose(result.reliability_range[1], .8286, atol=.00005)         # Upper reliability conf limit

    # Asset 2 from 2019 NCSL Symposium Tutorial on Intervals
    y = np.array([1,1,1,0,1,0,0,1,1,1,0,0,1,1,1,0,1,1,1,1])
    params = A3Params.from_assets([{'startdates': np.arange(0, 365*len(y), 365),
                                    'enddates': np.arange(365, 365*(len(y)+1), 365),
                                    'passfail': y
                                    }],
                                  target=0.95,
                                  I0=365)
    result = a3_testinterval(params)
    assert np.round(result.interval) == 205
    assert np.isclose(result.rejection, .9993, atol=.00005)  # Rejection conf
    assert np.isclose(result.reliability_range[0], .6000, atol=.00005)         # Lower reliability conf limit
    assert np.isclose(result.reliability_range[1], .7843, atol=.00005)         # Upper reliability conf limit

    # Asset 3 from 2019 NCSL Symposium Tutorial on Intervals
    y = np.array([1,1,1,1,0,1,0,1,1,1,0,1,1,1,1,1,1,0,1,1])
    params = A3Params.from_assets([{'startdates': np.arange(0, 730*len(y), 730),
                                    'enddates': np.arange(730, 730*(len(y)+1), 730),
                                    'passfail': y
                                    }],
                                  target=0.95,
                                  I0=730)
    result = a3_testinterval(params)
    assert np.round(result.interval) == 523
    assert np.isclose(result.rejection, .9682, atol=.00005)  # Rejection conf
    assert np.isclose(result.reliability_range[0], .7041, atol=.00005)         # Lower reliability conf limit
    assert np.isclose(result.reliability_range[1], .8716, atol=.00005)         # Upper reliability conf limit

    # Other made up numbers, input directly without from_data()
    params = A3Params(intol=42, n=50, I0=365, target=.9, conf=.3)
    result = a3_testinterval(params)
    assert np.round(result.interval) == 329
    assert np.isclose(result.rejection, .7557, atol=.00005)  # Rejection conf
    assert np.isclose(result.reliability_range[0], .8067, atol=.00005)         # Lower reliability conf limit
    assert np.isclose(result.reliability_range[1], .8664, atol=.00005)         # Upper reliability conf limit


def test_s2():
    ''' Test S2 Method. Can't find any full published data to compare
        against, so this just exercises the calculation to make sure
        it completes.
    '''
    # Reliability data from Table D-1 in RP1
    ti = [4, 7, 10, 13, 21, 28, 40, 48]           # Weeks between calibrations
    ni = np.array([4, 6, 14, 13, 22, 49, 18, 6])  # Number of calibrations in each interval of ti
    ri = [1.0, .83333, .6429, .6154, .5455, .4082, .5000, .3333]    # Observed measurement reliability
    params = S2Params(target=.75, ti=ti, ri=ri, ni=ni)
    result = s2_binom_interval(params)
    result.report.summary()


@pytest.mark.filterwarnings('ignore')  # Will generate runtime/optimize warning in minimization loop
def test_variables():
    ''' Test variables method using data from NASA-HDBK-8739.19-5 '''
    # Data in Table 7-2.
    dt = np.array([70., 86., 104., 135., 167., 173.])
    deltas = np.array([.1, .11, .251, .299, .403, .615])
    params = VariablesData(dt, deltas, u0=.28, y0=10.03)
    result_utarget = variables_uncertainty_target(params, utarget=0.5, order=2)
    result_rtarget = variables_reliability_target(params, rel_lo=9, rel_high=11, rel_conf=.9, order=2)

    # Compare curve-fit results with Table 7-3
    assert np.isclose(result_utarget.b[0], 0.00015741)
    assert np.isclose(result_utarget.b[1], 0.00001674)
    assert np.isclose(result_utarget.cov[0, 0], 0.00000101)
    assert np.isclose(result_utarget.cov[1, 1], 4.549E-11)
    assert np.isclose(result_utarget.cov[0, 1], -6.607E-9)
    assert np.isclose(result_utarget.syx, 0.0708, atol=.00005)

    # Compare uncertainty target method with Table 7-7
    assert np.isclose(result_utarget.interval, 327.15, atol=.005)

    # Compare reliability target method with Table 7-4
    assert np.isclose(result_rtarget.interval, 140.1, atol=.05)

    # Compute reliability model with one-sided limits, compare with Table 7-5 and 7-6
    result_rtarget = variables_reliability_target(params, rel_lo=None, rel_high=11, rel_conf=.9, order=2)
    assert np.isclose(result_rtarget.interval, 171.75, atol=.005)

    # Another problem in NASA-8739. They calculate the single-sided interval case
    # with results in Table 7-6, but the initial uncertainty is already less
    # than the limit at t=0! Assigned interval should be 0 or N/A. Table 7-6
    # lists 153.04, which is correct in that's when the 95% conf line crosses
    # the lower limit, but going the wrong direction.
    params = VariablesData(dt, deltas, u0=.28, y0=9.03)
    result_rtarget = variables_reliability_target(params, rel_lo=9, rel_high=None, rel_conf=.9, order=2)
    assert np.isclose(result_rtarget.interval, 0, atol=.005)  # Table 7-6
