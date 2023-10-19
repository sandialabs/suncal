''' Calculate false accept and reject risk using Monte Carlo method '''
from collections import namedtuple
import numpy as np

from ..common import distributions


def PFAR_MC(dist_proc, dist_test, LL, UL, GBL=0, GBU=0, N=100000, testbias=0):
    ''' Probability of False Accept/Reject using Monte Carlo Method

        Args:
            dist_proc (stats.rv_frozen or distributions.Distribution):
              Distribution of possible unit under test values from process
            dist_test (stats.rv_frozen or distributions.Distribution):
              Distribution of possible test measurement values
            LL (float):Lower specification limit (absolute)
            UL (float): Upper specification limit (absolute)
            GBL (float): Lower guardband, as offset. Test limit is LL + GBL.
            GBU (float): Upper guardband, as offset. Test limit is UL - GBU.
            testbias (float): Bias (difference between distribution median and expected value)
              in test distribution
            approx (bool): Approximate using discrete probability distribution. This
              uses trapz integration so it may be faster than letting scipy integrate
              the actual pdf function.
            N (int): Number of Monte Carlo samples

        Returns:
            pfa: False accept probability, P(OOT and Accepted)
            pfr: False reject probability, P(IT and Rejected)
            proc_samples: Monte Carlo samples for uut
            test_samples: Monte Carlo samples for test measurement
            cpfa: Conditional False Accept Probability, P(OOT | Accepted)
    '''
    Result = namedtuple('MCRisk', ['pfa', 'pfr', 'process_samples', 'test_samples', 'cpfa'])
    proc_samples = dist_proc.rvs(size=N)
    expected = dist_test.median() - testbias
    kwds = distributions.get_distargs(dist_test)
    locorig = kwds.pop('loc', 0)
    try:
        # Works for normal stats distributions, but not rv_histograms
        test_samples = dist_test.dist.rvs(loc=proc_samples-(expected-locorig), size=N, **kwds)
    except TypeError:
        # Works for histograms, but not regular distributions...
        test_samples = dist_test.dist(**kwds).rvs(loc=proc_samples-(expected-locorig), size=N)
    except ValueError:
        # Invalid parameter in kwds
        test_samples = np.array([])
        return Result(np.nan, np.nan, None, None)

    # Unconditional False Accept
    FA = np.count_nonzero(((test_samples < UL-GBU) & (test_samples > LL+GBL)) &
                          ((proc_samples > UL) | (proc_samples < LL))) / N
    FR = np.count_nonzero(((test_samples > UL-GBU) | (test_samples < LL+GBL)) &
                          ((proc_samples < UL) & (proc_samples > LL))) / N

    # Conditional False Accept
    p_intol_and_accepted = np.count_nonzero(((test_samples < UL-GBU) & (test_samples > LL+GBL)) &
                                            ((proc_samples <= UL) & (proc_samples >= LL))) / N
    p_accepted = np.count_nonzero(((test_samples < UL-GBU) & (test_samples > LL+GBL))) / N
    try:
        cpfa = 1 - p_intol_and_accepted / p_accepted
    except ZeroDivisionError:
        cpfa = 0.
    return Result(FA, FR, proc_samples, test_samples, cpfa)
