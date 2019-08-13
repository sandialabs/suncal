''' Test t-table calculations. Test values based on values in GUM Table G.2. '''

import numpy as np
from suncal import ttable

def test_t():
    # Test calculation of k given conf and degf
    assert np.isclose(ttable.t_factor(conf=.6827, degf=1), 1.84, atol=.005)
    assert np.isclose(ttable.t_factor(conf=.95, degf=1), 12.71, atol=.005)
    assert np.isclose(ttable.t_factor(conf=.9973, degf=5), 5.51, atol=.005)
    assert np.isclose(ttable.t_factor(conf=.90, degf=20), 1.72, atol=.005)
    assert np.isclose(ttable.t_factor(conf=.9545, degf=1E9), 2.00, atol=.005)

def test_conf():
    # Test calculation of confidence given k and degf. Tolerance is higher since tp values in GUM
    # Table were rounded
    assert np.isclose(ttable.confidence(tp=1.84, degf=1), .6827, atol=.005)
    assert np.isclose(ttable.confidence(tp=235.8, degf=1), .9973, atol=.005)
    assert np.isclose(ttable.confidence(tp=1.81, degf=10), .90, atol=.005)
    assert np.isclose(ttable.confidence(tp=2.09, degf=20), .95, atol=.005)
    assert np.isclose(ttable.confidence(tp=3.00, degf=1E9), .9973, atol=.005)

def test_degf():
    # Test calculation of degf given k and confidence.
    assert np.isclose(ttable.degf(tp=1.84, conf=.6827), 1, atol=.1)
    assert np.isclose(ttable.degf(tp=19.21, conf=.9973), 2, atol=.1)
    assert np.isclose(ttable.degf(tp=2.09, conf=.95), 19.5, atol=.5)  # This 2.09 shows up in table on two rows! Fractional part is between 19 and 20
    assert np.isclose(ttable.degf(tp=3.11, conf=.99), 11, atol=.1)
    assert ttable.degf(tp=2.00, conf=.9545) > 1E12   # Infinity
