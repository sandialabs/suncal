''' Test Interval Method, A3 '''
from dataclasses import dataclass
import numpy as np
from scipy import stats
from dateutil.parser import parse

from .report.attributes import ReportIntervalA3
from ..common import reporter


def datearray(dates):
    ''' Convert array to ordinal date. Input can be datetime or string '''
    if len(dates) == 0:
        dates = np.array([])
    elif hasattr(dates[0], 'toordinal'):
        dates = [d.toordinal() for d in dates]
    elif isinstance(dates[0], str):
        dates = [parse(d).toordinal() for d in dates]
    return np.asarray(dates)


@reporter.reporter(ReportIntervalA3)
@dataclass
class ResultsTestInterval:
    ''' Results of a Test Interval (A3) Calculation.

        Attributes:
            interval: Recommended calibration interval
            calculated: Calculated interval
            rejection: Rejection confidence
            reliability_range: Lower limit of reliability range
            reliability_hi: Upper limit of reliability range
            reliability_observed: Observed reliability
            intol: Number of calibrations in tolerance
            n: Total number of calibrations used to determine interval
            unused: Number of unused calibrations
    '''
    interval: float
    calculated: float
    rejection: float
    reliability_range: tuple[float, float]
    reliability_observed: float
    intol: float
    n: int
    unused: int = None


@dataclass
class A3Params:
    ''' Parameters for Method A3

        intol (int): Number of calibrations in tolerance during the
            interval period I0
        n (int): Total number of calibrations performed during the
            interval period I0
        I0 (float): Current interval
        target (float): Reliability target
        maxchange (float): Maximum change factor. New interval will not
            be more than maxchange*I0.
        conf (float): Interval change confidence. Interval will be
            changed from I0 if at least this much confidence that the
            new interval is better.
        mindelta (float): Minimum change in days
        minint (float): Shortest recommended interval
        maxint (float): Longest recommended interval
    '''
    intol: int = 0
    n: int = 1
    I0: float = 365
    target: float = 0.9
    maxchange: float = 2
    conf: float = 0.5
    mindelta: float = 5
    minint: float = 14
    maxint: float = 1826
    unused: int = 0

    @classmethod
    def from_assets(cls,
                    assets,
                    tolerance: float = 56,
                    threshold: float = 0.5,
                    I0: float = 365,
                    target: float = 0.9,
                    maxchange: float = 2,
                    conf: float = 0.5,
                    mindelta: float = 5,
                    minint: float = 14,
                    maxint: float = 1826):
        ''' Get number in-tolerance calibrations

            Args:
                assets (list): List of asset dictionaries, each item having keys
                    'startdates', 'enddates', 'passfail' with array values.
                tolerance (float): Tolerance to apply to historical intervals
                    in days. Calibrations intervals outside of I0 +/- tolerance
                    will be discarded.
                thresh (float): Threshold to apply to historical intervals.
                    Calibration intervals outside thresh*I0 will be discarded
                I0 (float): Assigned interval being tested
                target (float): Reliability target
                maxchange (float): Maximum change factor. New interval will not
                    be more than maxchange*I0.
                conf (float): Interval change confidence. Interval will be
                    changed from I0 if at least this much confidence that the
                    new interval is better.
                mindelta (float): Minimum change in days
                minint (float): Shortest recommended interval
                maxint (float): Longest recommended interval
        '''
        passes = 0
        totalused = 0
        total = 0
        for val in assets:
            if 'enddates' not in val or 'passfail' not in val:
                continue
            ends = datearray(val['enddates'])
            sortidx = np.argsort(ends)
            y = np.asarray(val['passfail'])[sortidx]
            ends = ends[sortidx]

            if 'startdates' not in val or val['startdates'] is None or len(val['startdates']) == 0:
                ddate = np.diff(ends)
                y = y[1:]
                total += (len(val['passfail']) - 1)
            else:
                starts = datearray(val['startdates'])[sortidx]
                ddate = np.asarray(ends) - np.asarray(starts)
                total += len(ddate)

            tolabs = min(tolerance, I0*threshold)
            try:
                use = np.where((y >= 0) & (abs(ddate - I0) <= tolabs))
            except TypeError:  # pass/fail are still strings
                y = np.array([1. if v.lower() in ['p', 'pass', 'true', 'yes'] else 0 for v in y])
                use = np.where((y >= 0) & (abs(ddate - I0) <= tolabs))
            y = y[use]
            passes += np.count_nonzero(y)
            totalused += len(y)
        return cls(passes, totalused, I0, target, maxchange, conf, mindelta, minint, maxint, unused=total-totalused)


def a3_testinterval(params: A3Params) -> ResultsTestInterval:
    ''' Calculate optimal interval using A3 Test Interval Method
        as described in NCSLI RP-1.
    '''
    if params.n == 0:
        return ResultsTestInterval(interval=np.nan,
                                   calculated=np.nan,
                                   rejection=1,
                                   reliability_range=(0, 1),
                                   reliability_observed=np.nan,
                                   intol=np.nan,
                                   n=np.nan)

    a = 1/params.maxchange
    b = params.maxchange

    Rt = params.target
    R0 = params.intol/params.n

    if np.isclose(R0, 0) or np.isclose(R0, 1):
        Q = 1 - stats.binom.pmf(params.intol, n=params.n, p=Rt)
    elif R0 > Rt:
        Q = stats.binom.cdf(params.intol, n=params.n, p=Rt) - 1
    else:
        Q = 1 - 2*stats.binom.cdf(params.intol, n=params.n, p=Rt)

    w = 10**((R0-Rt)/(1-Q))
    v = 10**((R0-Rt)*Q)

    if R0 > Rt:
        if Q == 1 or w > b:
            Icalc = b
        else:
            Icalc = w
    else:
        if v < a:
            Icalc = a
        else:
            Icalc = v
    Icalc = Icalc * params.I0

    if params.intol == 0:
        RL = 0
        RU = 1-(1-params.conf)**(1/params.n)
    elif params.intol == 1:
        RL = (1-params.conf)**(1/params.n)
        RU = 1
    else:
        RL = 1 - stats.beta.ppf((1+params.conf)/2, a=params.n-params.intol+1, b=params.intol)
        RU = 1 - stats.beta.ppf((1-params.conf)/2, a=params.n-params.intol, b=params.intol+1)

    # Accept new interval if Q > conf and within absolute limits
    if Q > params.conf and abs(Icalc - params.I0) >= params.mindelta:
        newI = min(max(Icalc, params.minint), params.maxint)
    else:
        newI = params.I0

    return ResultsTestInterval(interval=newI,
                               calculated=Icalc,
                               rejection=max(Q, 0),
                               reliability_range=(RL, RU if np.isfinite(RU) else 1.0),
                               reliability_observed=R0,
                               intol=params.intol,
                               n=params.n,
                               unused=params.unused)
