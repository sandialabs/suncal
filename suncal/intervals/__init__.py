''' Calculate optimal calibration intervals based on historical data

    Implements methods "A3" and "S2" defined in NCSLI Recommended Practice 1,
    and the "Variables" method.
'''

from .testa3 import A3Params, a3_testinterval, datearray
from .binoms2 import S2Params, s2_binom_interval
from .variables import (VariablesData,
                        variables_reliability_target,
                        variables_uncertainty_target,
                        ResultsVariablesInterval)
