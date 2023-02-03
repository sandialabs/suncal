''' Test Interval Method, A3 '''

from collections import namedtuple
import numpy as np
from scipy import stats
from dateutil.parser import parse

from .report.attributes import ReportIntervalA3

A3Result = namedtuple('A3Result', 'interval calculated rejection RL RU Robserved intol n unused')


def datearray(dates):
    ''' Convert array to ordinal date. Input can be datetime or string '''
    if len(dates) == 0:
        dates = np.array([])
    elif hasattr(dates[0], 'toordinal'):
        dates = [d.toordinal() for d in dates]
    elif isinstance(dates[0], str):
        dates = [parse(d).toordinal() for d in dates]
    return np.asarray(dates)


class ResultsTestInterval:
    ''' Results of a Test Interval (A3) Calculation.

        Attributes:
            interval: Recommended calibration interval
            calculated: Calculated interval
            rejection: Rejection confidence
            RL: Lower limit of reliability range
            RU: Upper limit of reliability range
            Robserved: Observed reliability
            intol: Number of calibrations in tolerance
            n: Total number of calibrations used to determine interval
            unused: Number of unused calibrations
    '''
    def __init__(self, interval, calculated, rejection, RL, RU, Robserved, intol, n, unused):
        self.interval = interval
        self.calculated = calculated
        self.rejection = rejection
        self.RL = RL
        self.RU = RU
        self.Robserved = Robserved
        self.intol = intol
        self.n = n
        self.unused = unused
        self.report = ReportIntervalA3(self)

    def _repr_markdown_(self):
        return self.report.summary().get_md()


class TestInterval:
    ''' Interval Test Method (A3) from RP1

        Args:
            intol (int): Number of calibrations in tolerance during the interval period I0
            n (int): Total number of calibrations performed during the interval period I0
            I0 (float): Current interval
            Rt (float): Reliability target
            maxchange (float): Maximum ratio allowable change in interval. Equal to the
              "b" parameter in RP1 Method A3.
            conf (float): Interval change confidence. Interval will be changed from I0 if at
              least this much confidence that the new interval is better.
    '''
    def __init__(self, intol=0, n=1, I0=365, Rt=.9, maxchange=2, conf=.5, mindelta=5, minint=14,
                 maxint=1826, unused=None):
        self.intol = intol
        self.n = n
        self.unused = unused
        self.I0 = I0
        self.Rtarget = Rt
        self.maxchange = maxchange
        self.conf = conf
        self.mindelta = mindelta
        self.minint = minint
        self.maxint = maxint

    def update(self, intol, n):
        ''' Update parameters. Don't change if None.

        Args:
            intol (int): Number of calibrations in tolerance during the interval period I0
            n (int): Total number of calibrations performed during the interval period I0
        '''
        self.intol = intol if intol is not None else self.intol
        self.n = n if n is not None else self.n

    def update_params(self, I0=365, Rt=.9, maxchange=2, conf=.5, mindelta=5, minint=14, maxint=1826):
        ''' Update calculation parameters

            Args:
                I0 (float): Current interval
                Rt (float): Reliability target
                maxchange (float): Maximum ratio allowable change in interval. Equal to the
                "b" parameter in RP1 Method A3.
                conf (float): Interval change confidence. Interval will be changed from I0 if at
                least this much confidence that the new interval is better.
                maxchange (float): Maximum change factor. New interval will not be more than
                maxchange*I0.
                mindelta (float): Minimum change in days
                minint (float): Shortest recommended interval
                maxint (float): Longest recommended interval
        '''
        self.I0 = I0
        self.Rtarget = Rt
        self.maxchange = maxchange
        self.conf = conf
        self.mindelta = mindelta
        self.minint = minint
        self.maxint = maxint

    def calculate(self):
        ''' Run the calculation '''
        if self.n == 0:
            return ResultsTestInterval(interval=np.nan,
                                       calculated=np.nan,
                                       rejection=1,
                                       RL=0,
                                       RU=1,
                                       Robserved=np.nan,
                                       intol=np.nan,
                                       n=np.nan,
                                       unused=None)

        a = 1/self.maxchange
        b = self.maxchange

        Rt = self.Rtarget
        R0 = self.intol/self.n

        if np.isclose(R0, 0) or np.isclose(R0, 1):
            Q = 1 - stats.binom.pmf(self.intol, n=self.n, p=Rt)
        elif R0 > Rt:
            Q = stats.binom.cdf(self.intol, n=self.n, p=Rt) - 1
        else:
            Q = 1 - 2*stats.binom.cdf(self.intol, n=self.n, p=Rt)

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
        Icalc = Icalc * self.I0

        if self.intol == 0:
            RL = 0
            RU = 1-(1-self.conf)**(1/self.n)
        elif self.intol == 1:
            RL = (1-self.conf)**(1/self.n)
            RU = 1
        else:
            RL = 1 - stats.beta.ppf((1+self.conf)/2, a=self.n-self.intol+1, b=self.intol)
            RU = 1 - stats.beta.ppf((1-self.conf)/2, a=self.n-self.intol, b=self.intol+1)

        # Accept new interval if Q > conf and within absolute limits
        if Q > self.conf and abs(Icalc - self.I0) >= self.mindelta:
            newI = min(max(Icalc, self.minint), self.maxint)
        else:
            newI = self.I0

        return ResultsTestInterval(interval=newI,
                                   calculated=Icalc,
                                   rejection=max(Q, 0),
                                   RL=RL,
                                   RU=RU if np.isfinite(RU) else 1.0,
                                   Robserved=R0,
                                   intol=self.intol,
                                   n=self.n,
                                   unused=self.unused)


class TestIntervalAssets:
    ''' Test Interval method using data from individual assets

        Args:
            I0 (float): Current interval
            Rt (float): Reliability target
    '''
    def __init__(self, I0=180, Rt=0.95):
        self.assets = {}
        self.I0 = I0
        self.Rtarget = Rt
        self.tol = 56
        self.thresh = .5
        self.maxchange = 2
        self.conf = 0.5
        self.mindelta = 5
        self.minint = 14
        self.maxint = 1826

    def updateasset(self, assetname, enddates, passfail, startdates=None, **kwargs):
        ''' Update the asset calibration data

            Args:
                assetname (string): Name of the asset (key into self.assets dict)
                enddates (array): List of ending dates for each cal cycle
                passfail (array):List of pass/fail (1/0) values for each cal
                startdates (array): List of starting dates for each cal cycle
                **kwargs: not used, kept for signature compatibility
        '''
        self.assets[assetname] = {'startdates': startdates,
                                  'enddates': enddates,
                                  'passfail': passfail}

    def update_params(self, I0=365, Rt=.9, maxchange=2, conf=.5,
                      mindelta=5, minint=14, maxint=1826, tol=56, thresh=999):
        ''' Update calculation parameters

            Args:
                I0 (float): Current interval
                Rt (float): Reliability target
                maxchange (float): Maximum ratio allowable change in interval. Equal to the
                  "b" parameter in RP1 Method A3.
                conf (float): Interval change confidence. Interval will be changed from I0 if at
                  least this much confidence that the new interval is better.
                maxchange (float): Maximum change factor. New interval will not be more than
                  maxchange*I0.
                mindelta (float): Minimum change in days
                minint (float): Shortest recommended interval
                maxint (float): Longest recommended interval
                tol (float): Tolerance to apply to historical intervals in days. Calibrations intervals
                  outside of this tolerance will be discarded
                thresh (float): Threshold to apply to historical intervals. Calibration intervals
                  outside thresh*I0 will be discarded
        '''
        self.I0 = I0
        self.Rtarget = Rt
        self.maxchange = maxchange
        self.conf = conf
        self.mindelta = mindelta
        self.minint = minint
        self.maxint = maxint
        self.tol = tol
        self.thresh = thresh

    def remasset(self, assetname):
        ''' Remove the asset from the set '''
        self.assets.pop(assetname, None)

    def to_testinterval(self):
        ''' Convert to summarized TestInterval class '''
        intol, n, total = self.get_intol()
        return TestInterval(intol, n, I0=self.I0, Rt=self.Rtarget, maxchange=self.maxchange,
                            mindelta=self.mindelta, minint=self.minint, maxint=self.maxint,
                            unused=total-n, conf=self.conf)

    def get_intol(self):
        ''' Get number in-tolerance calibrations

            Returns:
                passes (int): Number of passing calibrations with the assigned interval
                totalused (int): Total number of calibrations that meet the tolerance and
                  threshold criteria
                total (int): Total number of calibrations in the data set
         '''
        passes = 0
        totalused = 0
        total = 0
        for val in self.assets.values():
            ends = datearray(val['enddates'])
            sortidx = np.argsort(ends)
            y = np.asarray(val['passfail'])[sortidx]
            ends = ends[sortidx]

            if val['startdates'] is None:
                ddate = np.diff(ends)
                y = y[1:]
                total += (len(val['passfail']) - 1)
            else:
                starts = datearray(val['startdates'])[sortidx]
                ddate = np.asarray(ends) - np.asarray(starts)
                total += len(ddate)

            tolabs = min(self.tol, self.I0*self.thresh)
            try:
                use = np.where((y >= 0) & (abs(ddate - self.I0) <= tolabs))
            except TypeError:  # pass/fail are still strings
                y = np.array([1. if v.lower() in ['p', 'pass', 'true', 'yes'] else 0 for v in y])
                use = np.where((y >= 0) & (abs(ddate - self.I0) <= tolabs))
            y = y[use]
            passes += np.count_nonzero(y)
            totalused += len(y)
        return passes, totalused, total

    def calculate(self):
        ''' Calculate '''
        calc = self.to_testinterval()
        return calc.calculate()
