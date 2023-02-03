''' Interval analysis using variables data

    Based on Castrup "Calibration Intervals from Variables Data" which determines
    how much a device drifts over a certain amount of time. Two methods are calculated:
    1) Uncertainty Target Method: stop the interval when measurement uncertainty exceeds limit
    2) Reliability Target Method: stop the interval when predicted value+uncertainty exceeds fixed tolerance
'''

from itertools import combinations
from dataclasses import dataclass
from collections import namedtuple
import logging
import warnings
import numpy as np
from scipy.optimize import fsolve, OptimizeWarning

from ..common import ttable, reporter
from .report.variables import (ReportIntervalVariables,
                               ReportIntervalVariablesUncertainty,
                               ReportIntervalVariablesReliability)
from .binoms2 import datearray
from . import fit

warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=OptimizeWarning)


@reporter.reporter(ReportIntervalVariablesUncertainty)
@dataclass
class ResultsUncertaintyTargetInterval:
    ''' Results from Uncertainty Target interval calculation

        Attributes:
            interval: Calculated interval
            target: Target uncertainty
            u0: Uncertainty at start of interval
            k: Coverage factor
            dt: Array of delta-time values between calibrations
            deltas: Array of delta values between calibrations
            m: Polynomial order for fit curve
            y0: Initial value
            b: Polynomial fit parameters
            cov: Covariance between polynomial fit parameters
            Syx: Standard Error in Fit
    '''
    interval: float
    target: float
    u0: float
    k: float
    dt: np.ndarray
    deltas: np.ndarray
    m: int
    y0: float
    b: np.ndarray
    cov: np.ndarray
    syx: float


@reporter.reporter(ReportIntervalVariablesReliability)
@dataclass
class ResultsReliabilityTargetInterval:
    ''' Results from Reliability Target interval calculation

        Attributes:
            interval: Calculated optimal interval
            u0: initial uncertainty
            LL: Lower reliability limit
            UL: Upper reliability limit
            dt: array of delta-time values between calibrations
            deltas: array of deltas between calibrations
            y0: Initial value
            m: Polynomial order for fit curve
            conf: Confidence
            k: Coverage factor
            b: Polynomial fit parameters
            cov: Covariance between polynomial fit parameters
            Syx: Standard Error in Fit
    '''
    interval: float
    u0: float
    LL: float
    UL: float
    dt: np.ndarray
    deltas: np.ndarray
    y0: float
    m: int
    conf: float
    k: float
    b: np.ndarray
    cov: np.ndarray
    syx: float


@reporter.reporter(ReportIntervalVariables)
@dataclass
class ResultsVariablesInterval:
    ''' Results from both reliability and uncertainty target interval calculations

        Attributes:
            uncertaintytarget: Results of Uncertainty Target method
            reliabilitytarget: Results of Reliability Target method
    '''
    uncertaintytarget: ResultsUncertaintyTargetInterval
    reliabilitytarget: ResultsReliabilityTargetInterval


class VariablesInterval:
    ''' Calculate interval by variables data method as described in
        Castrup "Calibration Intervals from Variables Data"

        Args:
            dt (array):Array of times since last calibration
            deltas (array): Array of deviation from prior calibration for each x value
            u0 (float): Time-of-test uncertainty of measurement
            y0 (float): Initial value at 0 time since calibration
            m (int): Order of polynomial to fit. Automatically selected if None.
            maxm (int): Maximum order of polynomial to fit (if m is None)
    '''
    def __init__(self, dt=None, deltas=None, u0=0, y0=0, m=1, maxm=1,
                 utarget=0.5, rlimits=(-1, 1), rconf=.95, kvalue=1):
        self.u0 = u0
        self.y0 = y0
        self.kvalue = kvalue
        self.maxm = maxm
        self.m = m
        self.utarget = utarget
        self.rlimits = rlimits
        self.rconf = rconf
        self.t = np.array([])
        self.deltas = np.array([])
        self.fitparams = None
        if dt is not None and deltas is not None:
            self.update(dt, deltas)

    def update(self, t, deltas):
        ''' Set the drift data for this calculation

            Args:
                t (array): time since last calibration
                deltas (array): deviation from last calibration
        '''
        self.t = np.asarray(t).astype(float)
        self.deltas = np.asarray(deltas).astype(float)

    def update_params(self, u0=0, y0=0, m=1, utarget=.5, rlimitL=-1, rlimitU=1,
                      rconf=0.95, kvalue=1, calcrel=True, calcunc=True):
        ''' Update calculation parameters

            Args:
                u0: uncertainty at time 0
                y0: value at time 0
                m: polynomial order
                utarget: target uncertainty to stop the interval
                rlimitL, rlimitU: lower and upper reliability limits
                rconf: reliability target confidence
                kvalue: coverage factor for uncertainty target
        '''
        self.u0 = u0
        self.kvalue = kvalue
        self.y0 = y0
        self.m = m
        self.utarget = utarget
        self.rlimits = (rlimitL, rlimitU)
        self.rconf = rconf

    def fitcurve(self):
        ''' Fit curve to the t vs delta data '''
        self.t = np.asarray(self.t).astype(float)
        self.deltas = np.asarray(self.deltas).astype(float)
        if len(self.t) == 0 or len(self.deltas) == 0 or len(self.t) != len(self.deltas):
            return

        if self.m is None:
            self._select_order()

        b, cov, syx = fit.fitpoly(self.t, self.deltas, m=self.m)
        FitResults = namedtuple('FitResults', ['b', 'cov', 'syx'])
        return FitResults(b, cov, syx)

    def _select_order(self):
        ''' Select polynomial order m for best fit of x, y.
            Limit to maximum order of maxm.
        '''
        # (Castrup section 6)
        smin = np.inf
        m = 1
        for k in range(1, self.maxm+1):
            _, _, syx = fit.fitpoly(self.t, self.deltas, m=k)
            if syx < smin:
                smin = syx
                m = k
        self.m = m
        return self.m

    def calc_uncertainty_target(self):
        ''' Calculate uncertainty target method
        '''
        b, cov, syx = self.fitcurve()

        def target(t):
            uk1 = self.u0 / self.kvalue
            target = self.utarget / self.kvalue
            return self.kvalue * (uk1**2 + fit.u_pred(t, b, cov, syx)**2 - target**2)

        intv, _, ier, mesg = fsolve(target, x0=self.t.max(), full_output=True)
        if ier != 1:
            interval = 0
            logging.info('No solution found: %s', mesg)
        else:
            interval = intv[0]

        params = {'interval': interval,
                  'target': self.utarget,
                  'u0': self.u0,
                  'k': self.kvalue,
                  'dt': self.t,
                  'deltas': self.deltas,
                  'm': self.m,
                  'y0': self.y0,
                  'b': b,
                  'cov': cov,
                  'syx': syx}
        return ResultsUncertaintyTargetInterval(**params)

    def calc_reliability_target(self):
        ''' Calculate reliability target method
        '''
        LL, UL = self.rlimits
        if UL is None or LL is None:
            k = ttable.t_onetail(self.rconf, len(self.t)-self.m)
        else:
            k = ttable.k_factor(self.rconf, len(self.t)-self.m)

        b, cov, syx = self.fitcurve()

        if all(b == 0):
            # NO slope. Interval is infinite
            interval = np.inf
        else:
            def upper_lim(t):
                return (fit.y_pred(t, b, y0=self.y0) +
                        k * np.sqrt(self.u0**2 + fit.u_pred(t, b, cov, syx)**2))

            def lower_lim(t):
                return (fit.y_pred(t, b, y0=self.y0) -
                        k * np.sqrt(self.u0**2 + fit.u_pred(t, b, cov, syx)**2))

            t = []
            if (UL is not None and upper_lim(0) > UL) or (LL is not None and lower_lim(0) < LL):
                # Already outside the limits at t=0! Set interval to 0.
                t = [0]

            else:
                if UL is not None:
                    intv, _, ier, _ = fsolve(lambda x: upper_lim(x) - UL, x0=self.t.mean(), full_output=True)
                    if ier == 1:  # Solution found
                        t.append(intv)

                if LL is not None:
                    intv, _, ier, _ = fsolve(lambda x: lower_lim(x) - LL, x0=self.t.mean(), full_output=True)
                    if ier == 1:  # Solution found
                        t.append(intv)

            t = np.array(t)
            try:
                interval = t[t > 0].min()
            except ValueError:  # All intervals are negative
                interval = 0

        params = {'interval': interval,
                  'u0': self.u0,
                  'LL': LL,
                  'UL': UL,
                  'dt': self.t,
                  'deltas': self.deltas,
                  'y0': self.y0,
                  'm': self.m,
                  'conf': self.rconf,
                  'k': k,
                  'b': b,
                  'cov': cov,
                  'syx': syx}
        return ResultsReliabilityTargetInterval(**params)

    def calculate(self):
        ''' Calculate both reliability target and uncertainty target methods
        '''
        if len(self.t) == 0 or len(self.deltas) == 0 or len(self.t) != len(self.deltas):
            return None

        params_ut = self.calc_uncertainty_target()
        params_rt = self.calc_reliability_target()
        return ResultsVariablesInterval(params_ut, params_rt)


class VariablesIntervalAssets:
    ''' Variables interval analysis from data on individual assets '''
    def __init__(self, u0=0, y0=0, m=1, maxm=1,
                 utarget=0.5, rlimits=(-1, 1), rconf=.95,
                 use_alldeltas=False, kvalue=1, name='interval'):
        self.name = name
        self.description = ''
        self.u0 = u0
        self.kvalue = kvalue
        self.y0 = y0
        self.maxm = maxm
        self.m = m
        self.utarget = utarget
        self.rlimits = rlimits
        self.rconf = rconf
        self.use_alldeltas = use_alldeltas
        self.assets = {}

    def updateasset(self, assetname, enddates, asfound, startdates=None, asleft=None, **kwargs):
        ''' Update the asset calibration data

            Args:
                assetname (string): Name of the asset (key into self.assets dict)
                enddates (array): List of ending dates for each cal cycle
                asfound (array): List of as-found calibration values
                asleft (array): List of as-left calibration values, if different from as-found
                startdates (array): List of starting dates for each cal cycle

            Keyword arguments not used. For call signature compatibility
            with other class.
        '''
        self.assets[assetname] = {'startdates': startdates,
                                  'enddates': enddates,
                                  'asleft': asleft,
                                  'asfound': asfound}

    def update_params(self, u0=0, y0=0, m=1, utarget=.5, rlimitL=-1, rlimitU=1,
                      rconf=0.95, calcrel=True, calcunc=True, kvalue=1):
        ''' Update calculation parameters

            Args:
                u0: uncertainty at time 0
                y0: value at time 0
                m: polynomial order
                utarget: target uncertainty to stop the interval
                rlimitL, rlimitU: lower and upper reliability limits
                rconf: reliability target confidence
                kvalue: coverage factor for uncertainty target
         '''
        self.u0 = u0
        self.y0 = y0
        self.kvalue = kvalue
        self.m = m
        self.utarget = utarget
        self.rlimits = (rlimitL, rlimitU)
        self.rconf = rconf

    def remasset(self, assetname):
        ''' Remove asset '''
        self.assets.pop(assetname, None)

    def get_deltas(self):
        ''' Convert as-found an as-left values to deltaT, deltaV '''
        dt_all = np.array([])
        deltas_all = np.array([])

        for asset, val in self.assets.items():
            yfound = val['asfound']
            yleft = val['asleft']

            if self.use_alldeltas and yleft is not None:
                raise ValueError('Cannot use_alldeltas when yleft != yfound')
            if self.use_alldeltas and val['startdates'] is not None:
                raise ValueError('Cannot use_alldeltas when startdate != enddate')

            # Determine delta_t and delta_v depending on what information is
            # given.
            if val['startdates'] is None:
                # No start dates. Assume start = end of last interval
                # and drop the first one
                dt = np.diff(datearray(val['enddates']))
                if yleft is not None and yfound is not None:
                    deltas = np.asarray(yfound[1:]) - np.asarray(yleft[:-1])
                elif yfound is not None:
                    deltas = np.diff(np.asarray(yfound))
                else:
                    deltas = np.diff(np.asarray(yleft))

            else:
                dt = datearray(val['enddates']) - datearray(val['startdates'])
                if yleft is not None:
                    deltas = np.diff(np.asarray(yfound))
                else:
                    deltas = np.asarray(yfound[1:]) - np.asarray(yleft[:-1])
                    dt = dt[1:]

            if len(dt) == 0:
                continue

            assert len(dt) == len(deltas)

            if np.all(dt == dt[0]) and not self.use_alldeltas:
                raise ValueError('All intervals are the same. Try using use_alldeltas=True.')

            if self.use_alldeltas:
                dt = np.array([v[1]-v[0] for v in list(combinations(val['enddates'], 2))])
                deltas = np.array([v[1]-v[0] for v in list(combinations(yfound, 2))])
                assert len(dt) == len(deltas)

            dt_all = np.concatenate((dt_all, dt))
            deltas_all = np.concatenate((deltas_all, deltas))
        return dt_all, deltas_all

    def to_variablesinterval(self):
        ''' Convert assets into VariablesInterval '''
        dt, deltas = self.get_deltas()
        v = VariablesInterval(dt, deltas, u0=self.u0, y0=self.y0, m=self.m, maxm=self.maxm,
                              utarget=self.utarget, rlimits=self.rlimits, rconf=self.rconf, kvalue=self.kvalue)
        return v

    def calc_uncertainty_target(self):
        calc = self.to_variablesinterval()
        return calc.calc_uncertainty_target()

    def calc_reliability_target(self):
        calc = self.to_variablesinterval()
        return calc.calc_reliability_target()

    def calculate(self):
        ''' Run the calculation '''
        calc = self.to_variablesinterval()
        return calc.calculate()
