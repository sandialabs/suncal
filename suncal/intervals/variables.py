''' Interval analysis using variables data

    Based NCSLI RP-1 Variables Method V1. Two methods are available:
    1) Uncertainty Target Method: stop the interval when measurement uncertainty exceeds limit
    2) Reliability Target Method: stop the interval when reliability falls below target
'''
from typing import Sequence
from itertools import combinations
from dataclasses import dataclass
from collections import namedtuple
import logging
import warnings
import numpy as np
from scipy.optimize import fsolve, OptimizeWarning
from scipy import stats

from ..common import reporter
from .report.variables import ReportIntervalVariablesUncertainty, ReportIntervalVariablesReliability
from .binoms2 import datearray
from . import fit

warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=OptimizeWarning)

FitResults = namedtuple('FitResults', ['b', 'cov', 'syx', 'order', 'degf'])


@reporter.reporter(ReportIntervalVariablesUncertainty)
@dataclass
class ResultsUncertaintyTargetInterval:
    ''' Results from Uncertainty Target interval calculation

        Attributes:
           interval: Calculated interval
           target: Target uncertainty
           fit: Curve fit results
            data: Input data with attribute history
    '''
    interval: float
    target: float
    fit: FitResults
    data: 'VariablesData'


@reporter.reporter(ReportIntervalVariablesReliability)
@dataclass
class ResultsReliabilityTargetInterval:
    ''' Results from Reliability Target interval calculation

        Attributes:
            interval: Calculated interval
            target: Reliability target
            LL: Lower tolerance of attribute
            UL: Upper tolerance of attribute
            final_value: Predicted attribte value at end of interval
            projected_uncert: Predicted total uncertainty at end of interval
            projected_degf: Degrees of freedom of projected_uncert
            forecast_uncert: Predicted uncertainty due to curve fit
            fit: Curve fit results
            data: Input data with attribute history
    '''
    interval: float
    target: float
    LL: float
    UL: float
    final_value: float
    projected_uncert: float
    projected_degf: float
    forecast_uncert: float
    fit: FitResults
    data: 'VariablesData'


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


class VariablesData:
    ''' Data for Variables interval method

        Args:
            dt: Array of times since last calibration
            deltas: Array of deviation from prior calibration for each x value
            u0: Time-of-test uncertainty of measurement
            y0: Initial value at 0 time since calibration
            u0_degf: Degrees of freedom associated with u0
    '''
    def __init__(self, dt: Sequence[float], deltas: Sequence[float],
                 u0: float = 0, y0: float = 0, u0_degf: float = np.inf):
        self.dt = np.asarray(dt)
        self.deltas = np.asarray(deltas)
        self.u0 = u0
        self.y0 = y0
        self.u0_degf = u0_degf

    @classmethod
    def from_assets(cls,
                    assets: list[dict[str, float]],
                    u0: float = 0,
                    y0: float = 0,
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

        return cls(dt=dt_all, deltas=deltas_all, u0=u0, y0=y0)

    def uncertainty(self, t, fitresult: FitResults):
        ''' Predict total uncertainty at time t '''
        uconf = fit.u_conf(t, fitresult.b, fitresult.cov)
        sigb = np.sqrt(uconf**2 + self.u0**2)
        degf = sigb**4 / (uconf**4/fitresult.degf + self.u0**4/self.u0_degf)
        return sigb, degf

    def _fitcurve(self, order: int = None, maxorder: int = 1) -> FitResults:
        ''' Fit curve to the t vs delta data

            Args:
                order: Order of polynomial fit, or None to auto-select
                maxorder: Maximum order of polynomial fit when auto-selecting
        '''
        t = np.asarray(self.dt).astype(float)
        delta = np.asarray(self.deltas).astype(float)
        if len(t) == 0 or len(delta) == 0 or len(t) != len(delta):
            raise ValueError

        if order is None:
            order = _select_order(t, delta, maxorder)

        b, cov, syx = fit.fitpoly(t, delta, m=order)
        return FitResults(b, cov, syx, order, len(t)-order)


class VariablesReliabilityTarget(VariablesData):
    ''' Reliability Target '''

    def reliability(self, t, LL, UL, fitresult: FitResults):
        ''' Predict reliability at time t as the probabiliy the predicted
            uncertainty falls between the tolerance limits
        '''
        conf = fit.u_conf(t, fitresult.b, fitresult.cov)
        sigb = np.sqrt(conf**2 + self.u0**2)
        degf = sigb**4 / ((conf**4/fitresult.degf + self.u0**4/self.u0_degf))
        mu = fit.y_pred(t, fitresult.b, y0=self.y0)
        if UL is not None and LL is not None:
            return stats.t.cdf(x=(UL-mu)/sigb, df=degf) - stats.t.cdf(x=(LL-mu)/sigb, df=degf)
        if UL is not None:
            return stats.t.cdf(x=(UL-mu)/sigb, df=degf)
        return 1 - stats.t.cdf(x=(LL-mu)/sigb, df=degf)

    def calculate(
            self,
            LL: float = 0,
            UL: float = 1,
            target_reliability: float = 0.95,
            order: int = None,
            maxorder: int = 3
            ) -> ResultsUncertaintyTargetInterval:
        ''' Calculate interval using reliability target method

            Args:
                LL: Lower tolerance limit
                UL: Upper tolerance limit
                target_reliability: Reliability at which to end the interval
                order: Polynomial order for fit curve, or none to auto-choose
                maxorder: Maximum polynomial order for automatic order selection
        '''
        fitresult = self._fitcurve(order, maxorder=maxorder)

        if all(fitresult.b == 0):
            # NO slope. Interval is infinite
            interval = np.inf
            mu = upred = degf = uprojected = np.nan
        else:
            intv, _, ier, mesg = fsolve(lambda x: self.reliability(x, LL, UL, fitresult)-target_reliability,
                                        x0=self.dt.max(), full_output=True)
            if ier == 1:  # Solution found
                interval = intv[0]
                mu = fit.y_pred(interval, fitresult.b, y0=self.y0)
                upred = fit.u_pred(interval, fitresult.b, fitresult.cov, self.u0)
                uprojected, degf = self.uncertainty(interval, fitresult)
            else:
                logging.warning(mesg)
                interval = mu = uprojected = upred = degf = np.nan

        params = {
            'interval': interval,
            'target': target_reliability,
            'LL': LL,
            'UL': UL,
            'final_value': mu,
            'projected_uncert': uprojected,
            'projected_degf': degf,
            'forecast_uncert': upred,
            'fit': fitresult,
            'data': self
            }
        return ResultsReliabilityTargetInterval(**params)


class VariablesUncertaintyTarget(VariablesData):
    ''' Uncertainty Target '''
    def calculate(
            self,
            utarget: float = 0.5,
            order: int = 1,
            maxorder: int = 1,
            ) -> ResultsUncertaintyTargetInterval:
        ''' Calculate interval using uncertainty target method

            Args:
                utarget: Target uncertainty at which to end the interval
                order: Polynomial order for fit curve, or none to auto-choose
                maxorder: Maximum polynomial order for automatic order selection
        '''
        fitresult = self._fitcurve(order, maxorder=maxorder)

        def target(t):
            return fit.u_pred(t, fitresult.b, fitresult.cov, self.u0)**2 - utarget**2

        intv, _, ier, mesg = fsolve(target, x0=self.dt.max(), full_output=True)
        if ier != 1:
            interval = 0
            logging.info('No solution found: %s', mesg)
        else:
            interval = intv[0]

        results = {
            'interval': interval,
            'target': utarget,
            'fit': fitresult,
            'data': self
            }
        return ResultsUncertaintyTargetInterval(**results)
