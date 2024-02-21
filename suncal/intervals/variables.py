''' Interval analysis using variables data

    Based on Castrup "Calibration Intervals from Variables Data" which determines
    how much a device drifts over a certain amount of time. Two methods are calculated:
    1) Uncertainty Target Method: stop the interval when measurement uncertainty exceeds limit
    2) Reliability Target Method: stop the interval when predicted value+uncertainty exceeds fixed tolerance
'''
from typing import Sequence
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
            k: Coverage factor of u0 and target uncertainty
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
            k_u0: Coverage factor associated with u0
            LL: Lower reliability limit
            UL: Upper reliability limit
            dt: array of delta-time values between calibrations
            deltas: array of deltas between calibrations
            y0: Initial value
            m: Polynomial order for fit curve
            conf: Confidence
            k: Coverage factor for projected reliability
            b: Polynomial fit parameters
            cov: Covariance between polynomial fit parameters
            Syx: Standard Error in Fit
    '''
    interval: float
    u0: float
    k_u0: float
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


@dataclass
class VariablesData:
    ''' Data for Variables interval method

        Args:
            dt: Array of times since last calibration
            deltas: Array of deviation from prior calibration for each x value
            u0: Time-of-test uncertainty of measurement
            y0: Initial value at 0 time since calibration
            kvalue: Coverage factor associated with u0 and uncertainty target value
    '''
    dt: Sequence[float]
    deltas: Sequence[float]
    u0: float = 0
    y0: float = 0
    kvalue: float = 1

    @classmethod
    def from_assets(cls,
                    assets: list[dict[str, float]],
                    u0: float = 0,
                    y0: float = 0,
                    kvalue: float = 1,
                    use_alldeltas: bool = False) -> 'VariablesData':
        ''' Generate the VariablesData from list of assets '''
        dt_all = np.array([])
        deltas_all = np.array([])

        for asset in assets:
            yfound = asset.get('asfound', [])
            yleft = asset.get('asleft', [])

            if len(yleft) == 0:
                yleft = None
            if len(yfound) == 0:
                yfound = None

            if use_alldeltas and yleft is not None:
                raise ValueError('Cannot use_alldeltas when yleft != yfound')
            if use_alldeltas and asset.get('startdates') is not None:
                raise ValueError('Cannot use_alldeltas when startdate != enddate')

            # Determine delta_t and delta_v depending on what information is
            # given.
            if asset.get('startdates') is None or len(asset['startdates']) == 0:
                # No start dates. Assume start = end of last interval
                # and drop the first one
                dt = np.diff(datearray(asset['enddates']))
                if yleft is not None and yfound is not None:
                    deltas = np.asarray(yfound[1:]) - np.asarray(yleft[:-1])
                elif yfound is not None:
                    deltas = np.diff(np.asarray(yfound))
                else:
                    deltas = np.diff(np.asarray(yleft))

            else:
                dt = datearray(asset['enddates']) - datearray(asset['startdates'])
                if yleft is None:
                    deltas = np.diff(np.asarray(yfound))
                    dt = dt[1:]
                else:
                    deltas = np.asarray(yfound) - np.asarray(yleft)

            if len(dt) == 0:
                continue

            assert len(dt) == len(deltas)

            if np.all(dt == dt[0]) and not use_alldeltas:
                raise ValueError('All intervals are the same. Try using use_alldeltas=True.')

            if use_alldeltas:
                dt = np.array([v[1]-v[0] for v in list(combinations(asset['enddates'], 2))])
                deltas = np.array([v[1]-v[0] for v in list(combinations(yfound, 2))])
                assert len(dt) == len(deltas)

            dt_all = np.concatenate((dt_all, dt))
            deltas_all = np.concatenate((deltas_all, deltas))

        return cls(dt=dt_all, deltas=deltas_all, u0=u0, y0=y0, kvalue=kvalue)


FitResults = namedtuple('FitResults', ['b', 'cov', 'syx'])


def _fitcurve(t: Sequence[float], delta: Sequence[float],
              order: int = 1, maxorder: int = 1) -> FitResults:
    ''' Fit curve to the t vs delta data

        Args:
            t: Time values
            delta: Deviation from prior value at each time value
            order: Order of polynomial fit, or None to auto-select
            maxorder: Maximum order of polynomial fit when auto-selecting
    '''
    t = np.asarray(t).astype(float)
    delta = np.asarray(delta).astype(float)
    if len(t) == 0 or len(delta) == 0 or len(t) != len(delta):
        raise ValueError

    if order is None:
        order = _select_order(t, delta, maxorder)

    b, cov, syx = fit.fitpoly(t, delta, m=order)
    return FitResults(b, cov, syx)


def _select_order(t: Sequence[float], delta: Sequence[float], maxorder: int = 1) -> int:
    ''' Select polynomial order m for best fit of x, y.
        Limit to maximum order of maxorder.
    '''
    # (Castrup section 6)
    smin = np.inf
    m = 1
    for k in range(1, maxorder+1):
        _, _, syx = fit.fitpoly(t, delta, m=k)
        if syx < smin:
            smin = syx
            m = k
    return m


def variables_uncertainty_target(data: VariablesData,
                                 utarget: float = 0.5,
                                 order: int = 1,
                                 maxorder: int = 1,
                                 ) -> ResultsUncertaintyTargetInterval:
    ''' Calculate interval using uncertainty target method

        Args:
            data: The historical calibration data
            utarget: Target uncertainty at which to end the interval
            order: Polynomial order for fit curve, or none to auto-choose
            maxorder: Maximum polynomial order for automatic order selection
    '''
    dt = np.asarray(data.dt)
    deltas = np.asarray(data.deltas)
    b, cov, syx = _fitcurve(dt, deltas, order, maxorder)

    def target(t):
        uk1 = data.u0 / data.kvalue
        target = utarget / data.kvalue
        return data.kvalue * (uk1**2 + fit.u_pred(t, b, cov, syx)**2 - target**2)

    intv, _, ier, mesg = fsolve(target, x0=dt.max(), full_output=True)
    if ier != 1:
        interval = 0
        logging.info('No solution found: %s', mesg)
    else:
        interval = intv[0]

    results = {'interval': interval,
               'target': utarget,
               'u0': data.u0,
               'k': data.kvalue,
               'dt': dt,
               'deltas': deltas,
               'm': order,
               'y0': data.y0,
               'b': b,
               'cov': cov,
               'syx': syx}
    return ResultsUncertaintyTargetInterval(**results)


def variables_reliability_target(data: VariablesData,
                                 rel_lo: float = -1,
                                 rel_high: float = 1,
                                 rel_conf: float = 0.95,
                                 order: int = 1,
                                 maxorder: int = 1
                                 ) -> ResultsUncertaintyTargetInterval:
    ''' Calculate interval using reliability target method

        Args:
            data: The historical calibration data
            rel_lo: Lower reliability limit
            rel_high: Upper reliability limit
            rel_conf: Level of confidence for reliability
            order: Polynomial order for fit curve, or none to auto-choose
            maxorder: Maximum polynomial order for automatic order selection
    '''
    dt = np.asarray(data.dt)
    deltas = np.asarray(data.deltas)

    LL, UL = rel_lo, rel_high
    if UL is None or LL is None:
        k = ttable.t_onetail(rel_conf, len(dt)-order)
    else:
        k = ttable.k_factor(rel_conf, len(dt)-order)

    b, cov, syx = _fitcurve(dt, deltas, order, maxorder=maxorder)

    if all(b == 0):
        # NO slope. Interval is infinite
        interval = np.inf
    else:
        def upper_lim(t):
            return (fit.y_pred(t, b, y0=data.y0) +
                    k * np.sqrt(data.u0**2 + fit.u_pred(t, b, cov, syx)**2))

        def lower_lim(t):
            return (fit.y_pred(t, b, y0=data.y0) -
                    k * np.sqrt(data.u0**2 + fit.u_pred(t, b, cov, syx)**2))

        t = []
        if (UL is not None and upper_lim(0) > UL) or (LL is not None and lower_lim(0) < LL):
            # Already outside the limits at t=0! Set interval to 0.
            t = [0]

        else:
            if UL is not None:
                intv, _, ier, _ = fsolve(lambda x: upper_lim(x) - UL, x0=dt.mean(), full_output=True)
                if ier == 1:  # Solution found
                    t.append(intv)

            if LL is not None:
                intv, _, ier, _ = fsolve(lambda x: lower_lim(x) - LL, x0=dt.mean(), full_output=True)
                if ier == 1:  # Solution found
                    t.append(intv)

        t = np.array(t)
        try:
            interval = t[t > 0].min()
        except ValueError:  # All intervals are negative
            interval = 0

    params = {'interval': interval,
              'u0': data.u0,
              'k_u0': data.kvalue,
              'LL': LL,
              'UL': UL,
              'dt': dt,
              'deltas': deltas,
              'y0': data.y0,
              'm': order,
              'conf': rel_conf,
              'k': k,
              'b': b,
              'cov': cov,
              'syx': syx}
    return ResultsReliabilityTargetInterval(**params)
