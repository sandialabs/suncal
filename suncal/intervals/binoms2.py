''' Binomial Method S2 '''
from typing import Sequence
from dataclasses import dataclass, field
import warnings
import logging
from collections import namedtuple
import numpy as np
from scipy import stats
from scipy.optimize import OptimizeWarning, minimize, approx_fprime, brentq, curve_fit, fsolve
from scipy.special import binom
from dateutil.parser import parse

from .report.attributes import ReportIntervalS2
from . import s2models


Observed = namedtuple('Observed', 'ti ri ni binleft')
S2Result = namedtuple('S2Result', 'interval theta modelfunction F Fcrit C accept target tr guess observed Ng G')

# Curve fit solver will throw warnings due to poor fits. Filter them out.
warnings.filterwarnings('ignore', category=RuntimeWarning, module='scipy')
warnings.filterwarnings('ignore', category=OptimizeWarning, module='scipy')


def datearray(dates):
    ''' Convert array to ordinal date. Input can be datetime or string '''
    if len(dates) == 0:
        dates = np.array([])
    elif hasattr(dates[0], 'toordinal'):
        dates = [d.toordinal() for d in dates]
    elif isinstance(dates[0], str):
        dates = [parse(d).toordinal() for d in dates]
    return np.asarray(dates)


def _count_groups(x):
    ''' Return group count (N_G in RP1) with how many models result in a "similar" interval. '''
    idx = np.argsort(x)
    x = np.maximum(x[idx], 0)
    diff = np.diff(x)
    mean = diff.mean() * 1.5
    groups = np.argwhere(diff > mean)
    bins = [0]
    if len(groups) == 0:
        bins.append(x[-1])
    else:
        start = 0
        for group in groups:
            vals = x[start:int(group[0])+1]
            bins.append(vals[-1]+.1)
            start = int(group[0])+1

    binidx = np.digitize(x, bins=bins)-1
    cnt = []
    for b in binidx:
        cnt.append(len(binidx) - np.count_nonzero(binidx-b))
    cnt = np.array(cnt)[np.argsort(idx)]
    return cnt


def neglikelihood(theta, model, ti, ni, gi):
    ''' Negative of likelihood function (negative log D-5 in RP1) '''
    modelr = model(ti, *theta)
    prod = np.prod(binom(ni, gi) * modelr**gi * (1-modelr)**(ni-gi))
    return np.nan_to_num(-np.log(prod), posinf=1000, neginf=-1000)


def reliability_variance(tau, theta, model, ni, ti):
    ''' Compute variance in reliability model (RP-1 eq D-29)

        Args:
            tau (float): Value of t at which to evaluate variance
            theta (tuple): Fit parameters for reliability model
            model (callable): Reliability model
            ni (array): Number of measurements in the bin
            ti (array): Observed intervals (right edge of bin)

        Returns:
            variance (float): Variance in the reliability model
    '''
    W = []
    for n, t in zip(ni, ti):
        r = model(t, *theta)
        W.append(n / r / (1-r))
    W = np.diag(W)

    D = []
    for t in ti:
        D.append(approx_fprime(theta, lambda p: model(t, *p)))
    D = np.asarray(D)

    d = approx_fprime(theta, lambda p: model(tau, *p))
    var_r = d.T @ np.linalg.inv(D.T @ W @ D) @ d
    return var_r


class ResultsBinomIntervalS2:
    ''' Results from Binomial Interval (S2) Method

        Attributes:
            interval: Recommended interval using best fit reliability model
            best: Name of best fit reliability model

        Methods:
            model: Get results from one reliability model
    '''
    def __init__(self, methodresults, conf: float = 0.95):
        self.models = {k: methodresults[k] for k in sorted(
            methodresults, key=lambda x: (methodresults[x].G), reverse=True)}
        self.best = max(self.models, key=lambda x: self.models[x].G)
        self.interval = self.models[self.best].interval
        self.conf = conf
        self.report = ReportIntervalS2(self)

    def _repr_markdown_(self):
        return self.report.summary().get_md()

    def model(self, name=None):
        ''' Get results for one reliability model '''
        if name is None:
            name = self.best
        return self.models.get(name)

    def reliability(self, t, name=None):
        ''' Get Reliability Curve at t '''
        results = self.model(name)
        model = results.modelfunction
        theta = results.theta
        return model(t, *theta)

    def expanded(self, name=None, conf=.95):
        ''' Expanded uncertainty range for one reliability model

            Args:
                name (str): Name of the reliability model
                conf (float): Confidence level (0-1) for uncertainty
        '''
        results = self.model(name)
        model = results.modelfunction
        theta = results.theta
        target = results.target
        ti = results.observed.ti
        ni = results.observed.ni
        kfactor = stats.norm.interval(confidence=conf)[1]

        def tau_f1(tau, model, theta, ni, ti, target):
            return model(tau, *theta) - kfactor * np.sqrt(reliability_variance(tau, theta, model, ni, ti)) - target

        def tau_f2(tau, model, theta, ni, ti, target):
            return model(tau, *theta) + kfactor * np.sqrt(reliability_variance(tau, theta, model, ni, ti)) - target

        try:
            tau_l = brentq(tau_f1, a=0, b=ti.max(), args=(model, theta, ni, ti, target), xtol=.01)
        except (ValueError, np.linalg.LinAlgError):
            # Probably no zero crossings
            tau_l = 0

        try:
            tau_u = brentq(tau_f2, a=0, b=ti.max(), args=(model, theta, ni, ti, target), xtol=.01)
        except (ValueError, np.linalg.LinAlgError):
            # Probably no zero crossings
            tau_u = np.inf

        return tau_l, tau_u


def get_passfails(asset):
    ''' Get list of interval, passfail values

        Returns:
            passfail: list of pass/fail values
            ti: list of intervals
    '''
    pf = np.array(asset['passfail'])
    try:
        pf.mean()
    except TypeError:
        pf = np.array([1. if v.lower() in ['p', 'pass', 'true', 'yes'] else 0. for v in pf])
    ends = datearray(asset['enddates'])
    sortidx = np.argsort(ends)
    pf = pf[sortidx]
    ends = ends[sortidx]

    if asset.get('startdates') is None or len(asset['startdates']) == 0:
        ti = np.diff(ends)
        pf = pf[1:]
    else:
        starts = datearray(np.array(asset['startdates'])[sortidx])
        ti = ends - starts
    return list(pf), list(ti)


@dataclass
class S2Params:
    ''' Parameters for Binomial Interval Method (S2)

        Args:
            target: Target reliability (0-1) (Rt in RP1)
            ti: Observed intervals (right edge of bin)
            ri: Observed reliability for each interval (Ri in RP1)
            ni: Number of measurements in this interval
            ti0: Observed interval, left edge of bin (optional)
    '''
    target: float = 0.95
    ti: Sequence[float] = field(default_factory=list)
    ri: Sequence[float] = field(default_factory=list)
    ni: Sequence[float] = field(default_factory=list)
    ti0: Sequence[float] = field(default_factory=list)

    @classmethod
    def from_assets(cls,
                    assets: list[dict[str, float]],
                    target: float = 0.95,
                    bins: int = 8,
                    binlefts: Sequence[float] = None,
                    binwidth: float = None):
        ''' Generate S2Params from list of assets

            Args:
                assets: List of asset calibrations
                target: Reliability target to solve for
                bins (int): Number of bins if binlefts is None
                binlefts (list): List of left-edges of each bin
                binwidth (float): Width of all bins
        '''
        R = []
        t = []
        t0 = []
        ni = []
        passfails = []
        testintervals = []

        for asset in assets:
            pf, ti = get_passfails(asset)
            passfails.extend(pf)
            testintervals.extend(ti)

        testintervals = np.array(testintervals)
        passfails = np.array(passfails)

        # Includes left and right edges
        if binlefts is None:
            binedges = np.histogram_bin_edges(testintervals, bins=bins)
            binlefts = binedges[:-1]
            binwidth = binedges[1] - binedges[0]

        for left in binlefts:
            idx = (testintervals > np.floor(left)) & (testintervals <= np.ceil(left+binwidth))
            if len(testintervals[idx]) > 0:
                R.append(passfails[idx].mean())
                t.append(left+binwidth)
                t0.append(left)
                ni.append(len(testintervals[idx]))
        return cls(target=target, ti=t, ri=R, ni=ni, ti0=t0)


def s2_binom_interval(params: S2Params,
                      p0: dict[str, Sequence[float]] = None,
                      bounds: dict[str, Sequence[Sequence[float]]] = None,
                      conf: float = 0.95
                      ) -> ResultsBinomIntervalS2:
    ''' Optimize interval using Binomial Method S2

        Args:
            params: Parameters for method S2
            p0: Initial guess for theta of each model. Dictionary key
                is model name, value is array of initial theta values.
            bounds: Search bounds for each parameter. Dictionary key
                is model name, value is ((min, max), (min, max), ...)
                for each theta in the model.
            conf: Confidence level for reporting uncertainty on interval
    '''
    p0 = {} if p0 is None else p0
    bounds = {} if bounds is None else bounds
    ni = np.asarray(params.ni)
    ti = np.asarray(params.ti)
    ti0 = np.array([]) if params.ti0 is None else np.asarray(params.ti0)
    ri = np.asarray(params.ri)

    k = len(ni)  # Number of intervals/bins
    n = sum(ni)  # Total number of measurements made
    gi = (ni * ri).astype(int)

    if k < 2:
        raise ValueError('Not enough data to compute interval')

    results = {}
    for name, model in s2models.models.items():
        # Defaults if things fail
        interval = 0
        theta = None
        F = np.inf
        Fcrit = 0
        C = 100
        accept = False

        if name in p0:
            p0 = p0[name]
            bounds = bounds.get(name, None)
        elif name in s2models.guessers:
            p0, bounds = s2models.guessers.get(name)(ti, ri)
        else:
            raise ValueError(f'Need initial guess p0 for model {name}')

        try:
            result = minimize(neglikelihood, p0,
                              bounds=bounds,
                              args=(model, ti, ni, gi))
        except (RuntimeError, TypeError):
            logging.warning('%s failed to converge!', name)
            theta = curve_fit(model, ti, ri, p0=p0)[0]  # Fall back on fit to ti, Ri
        else:
            theta = result.x

        m = len(theta)  # Number of fit parameters

        # Find intersection of fit and Rtarget
        interval = fsolve(lambda t: model(t, *theta)-params.target, x0=ti[0])[0]
        tr = fsolve(lambda t: model(t, *theta)-(1-params.target), x0=ti[-1])[0]
        interval = max(interval, 0)
        tr = max(tr, 0)

        if np.isclose(model(interval, *theta), params.target, atol=.01):
            interval = np.round(interval)
            se2 = sum(ni * ri * (1 - ri)) / (n-k)  # RP1 eq D-19
            sl2 = sum(ni * (ri - model(ti, *theta))**2) / (k-m)  # RP1 eq D-23
            F = sl2/se2
            Fcrit = stats.f.ppf(0.95, dfn=k-m, dfd=n-k)  # Critical F parameter
            C = stats.f.cdf(F, dfn=k-m, dfd=n-k) * 100   # Rejection Confidence
            if not np.isfinite(C):
                C = 100.
            accept = F < Fcrit
        else:
            interval = 0.

        results[name] = {
            'interval': interval,
            'theta': theta,
            'modelfunction': model,
            'F': F,
            'Fcrit': Fcrit,
            'C': C,
            'accept': accept,
            'target': params.target,
            'tr': tr,  # Eq D-25 in RP1
            'guess': p0,
            'observed': Observed(
                ti=ti,
                ri=ri,
                ni=ni,
                binleft=ti0 if len(ti0) else [0] + list(ti)[:-1])
            }

    # Group them by interval similarity to compute figure of merit
    interval = np.array([r['interval'] for r in results.values()])
    Ng = _count_groups(interval)  # Number of models in a "group" with similar interval result

    for idx, name in enumerate(s2models.models.keys()):
        results[name]['Ng'] = Ng[idx]
        results[name]['G'] = Ng[idx] / (results[name]['C']/100) * results[name]['tr'] ** 0.25

    return ResultsBinomIntervalS2({name: S2Result(**r) for name, r in results.items()}, conf=conf)
