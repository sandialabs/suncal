''' Binomial Method S2 '''

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
            vals = x[start:int(group)+1]
            bins.append(vals[-1]+.1)
            start = int(group)+1

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
    def __init__(self, methodresults):
        self.models = {k: methodresults[k] for k in sorted(
            methodresults, key=lambda x: (methodresults[x].G), reverse=True)}
        self.best = max(self.models, key=lambda x: self.models[x].G)
        self.interval = self.models[self.best].interval
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


class BinomialInterval:
    ''' Class for calculating calibration interval by Binomial Method (S2 in RP1)

        Args:
            Rtarget (float): Target reliability (0-1)
            ti (array): Observed intervals (right edge of bin)
            ti0 (array): Observed interval, left edge of bin
            Ri (array):Observed reliability for each interval
            ni (array): Number of measurements in this interval
            conf (float): Level of confidence

        Notes:
            ti, Ri, and ni parameters are used to set up calculation if calibration
            data has already been binned into discrete intervals with reliability.
            Otherwise, use from_passfail() method to set up calculation based on
            individual measurement points. If ti0 is None, there will be no gaps
            between bins.
    '''
    def __init__(self, Rtarget=0.95, ti=None, ti0=None, Ri=None, ni=None):
        self.Rtarget = Rtarget
        self.ti0 = np.asarray(ti0).astype(float) if ti0 is not None else np.array([])
        self.ti = np.asarray(ti).astype(float) if ti is not None else np.array([])
        self.Ri = np.asarray(Ri).astype(float) if Ri is not None else np.array([])
        self.ni = np.asarray(ni).astype(float) if ni is not None else np.array([])
        self.models = s2models.models
        self.guessers = s2models.guessers

        # Initial guess override (commented values are example coefficients from RP1)
        self.p0 = {
                  # 'Weibull': [.02, 2.0],
                  # 'Random Walk': [.3, .05],
                  # 'Restricted Walk': [0, 2, 0.3],
                  # 'Modified Gamma': [0.2],
                  # 'Mortality Drift': [.005, .001],
                  # 'Warranty': [1.5, 10],
                  # 'Drift': [10, .5, .5],
                  # 'Log Normal': [0.25, 1],
                  # 'Exponential': [.1],
                  # 'Mixed Exponential': [5, 2],
                  }

        # Override bounds for theta minimization. ((min, max), (min, max)).. for each theta
        self.bounds = {
                      # 'Exponential': ((0, 0),),
                      # ...
                      }

    def update(self, ti, ri, ni, ti0=None):
        ''' Update calibration data. Don't change if None.

            Args:
                ti (array): Observed intervals (right edge of bin)
                ri (array):Observed reliability for each interval
                ni (array): Number of measurements in this interval
                ti0 (array): Observed interval, left edge of bin
        '''
        self.ti = ti if ti is not None else self.ti
        self.ti0 = ti0 if ti0 is not None else self.ti0
        self.Ri = ri if ri is not None else self.Ri
        self.ni = ni if ni is not None else self.ni

    def update_params(self, Rt):
        ''' Update parameters, reliability and confidence

            Args:
                Rtarget (float): Target reliability (0-1)
        '''
        self.Rtarget = Rt

    def set_p0(self, model, p0):
        ''' Set initial guess for fitting model

            Args:
                model (str): Name of reliability model
                p0 (array): Initial guess for fit parameters
        '''
        self.p0[model] = p0

    def add_model(self, name, func, p0=None):
        ''' Add a reliability model

            Args:
                name (string): Name of reliability model
                func (callable): Function taking time as first argument, and other
                  arguments as fit parameters define the model
        '''
        self.models[name] = func
        if callable(p0):
            self.guessers[name] = p0
        elif p0 is not None:
            self.p0[name] = p0

    def calculate(self):
        ''' Calculate intervals using each model '''
        k = len(self.ni)  # Number of intervals/bins
        n = sum(self.ni)  # Total number of measurements made
        gi = (self.ni * self.Ri).astype(int)

        if k < 2:
            warnings.warn('Not enough data to compute interval')
            self.results = None
            return self.results

        results = {}
        for name, model in self.models.items():
            # Defaults if things fail
            interval = 0
            theta = None
            F = np.inf
            Fcrit = 0
            C = 100
            accept = False

            if name in self.p0:
                p0 = self.p0[name]
                bounds = self.bounds.get(name, None)
            elif name in self.guessers:
                p0, bounds = self.guessers.get(name)(self.ti, self.Ri)
            else:
                raise ValueError(f'Need initial guess p0 for model {name}')

            try:
                result = minimize(neglikelihood, p0,
                                  bounds=bounds,
                                  args=(model, self.ti, self.ni, gi))
            except (RuntimeError, TypeError):
                logging.warning('%s failed to converge!', name)
                theta = curve_fit(model, self.ti, self.Ri, p0=p0)[0]  # Fall back on fit to ti, Ri
            else:
                theta = result.x

            m = len(theta)  # Number of fit parameters

            # Find intersection of fit and Rtarget
            interval = fsolve(lambda t: model(t, *theta)-self.Rtarget, x0=self.ti[0])[0]
            tr = fsolve(lambda t: model(t, *theta)-(1-self.Rtarget), x0=self.ti[-1])[0]
            interval = max(interval, 0)
            tr = max(tr, 0)

            if np.isclose(model(interval, *theta), self.Rtarget, atol=.01):
                interval = np.round(interval)
                se2 = sum(self.ni * self.Ri * (1 - self.Ri)) / (n-k)                # RP1 eq D-19
                sl2 = sum(self.ni * (self.Ri - model(self.ti, *theta))**2) / (k-m)  # RP1 eq D-23
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
                'target': self.Rtarget,
                'tr': tr,  # Eq D-25 in RP1
                'guess': p0,
                'observed': Observed(
                    ti=self.ti,
                    ri=self.Ri,
                    ni=self.ni,
                    binleft=self.ti0 if len(self.ti0) else [0] + list(self.ti)[:-1])
                }

        # Group them by interval similarity to compute figure of merit
        interval = np.array([r['interval'] for r in results.values()])
        Ng = _count_groups(interval)  # Number of models in a "group" with similar interval result

        for idx, name in enumerate(self.models.keys()):
            results[name]['Ng'] = Ng[idx]
            results[name]['G'] = Ng[idx] / (results[name]['C']/100) * results[name]['tr'] ** 0.25

        return ResultsBinomIntervalS2({name: S2Result(**r) for name, r in results.items()})


class BinomialIntervalAssets:
    ''' Binomial Interval from individual asset's data

        Args:
            Rt (float): Reliability target
            bins (int): Number of bins for converting calibrations into (time, reliability) values
            conf (float): Level of confidence
            binlefts (array): Left edge of interval bins
            binwidth (array): Width of interval bins
    '''
    def __init__(self, Rt=0.9, bins=10, binlefts=None, binwidth=None):
        self.Rtarget = Rt
        self.bins = bins
        self.binlefts = binlefts
        self.binwidth = binwidth
        self.assets = {}

    def updateasset(self, assetname, enddates, passfail, startdates=None, **kwargs):
        ''' Update the asset calibration data

            Args:
                assetname (string): Name of the asset (key into self.assets dict)
                enddates (array): List of ending dates for each cal cycle
                passfail (array): List of pass/fail (1/0) values for each cal
                startdates (array): List of starting dates for each cal cycle
                **kwargs: Not used, but kept for compatibility
        '''
        self.assets[assetname] = {
            'startdates': startdates,
            'enddates': enddates,
            'passfail': passfail}

    def update_params(self, Rt=.9, bins=10, binlefts=None, binwidth=None, **kwargs):
        ''' Update target and bins parameters

        Args:
            Rt (float): Reliability target
            bins (int): Number of bins for converting calibrations into (time, reliability) values
            binlefts (array): Left edge of interval bins
            binwidth (array): Width of interval bins
        '''
        self.Rtarget = Rt
        self.bins = bins
        self.binlefts = binlefts
        self.binwidth = binwidth

    def remasset(self, assetname):
        ''' Remove asset from set '''
        self.assets.pop(assetname, None)

    def get_passfails(self, asset):
        ''' Get list of interval, passfail values

            Returns:
                passfail: list of pass/fail values
                ti: list of intervals
        '''
        # Ensure sorted date order
        val = self.assets.get(asset)

        pf = np.array(val['passfail'])
        try:
            pf.mean()
        except TypeError:
            pf = np.array([1. if v.lower() in ['p', 'pass', 'true', 'yes'] else 0. for v in pf])
        ends = datearray(val['enddates'])
        sortidx = np.argsort(ends)
        pf = pf[sortidx]
        ends = ends[sortidx]

        if val['startdates'] is None:
            ti = np.diff(ends)
            pf = pf[1:]
        else:
            starts = datearray(np.array(val['startdates'])[sortidx])
            ti = ends - starts
        return list(pf), list(ti)

    def get_reliability(self, binlefts=None, binwidth=None):
        ''' Convert assets into arrays of dt, Ri, n

            If parameters are not provided, attributes defined in class
            will be used.

            Args:
                binlefts (list): List of left-edges of each bin
                binwidth (float): Width of all bins
        '''
        R = []
        t = []
        t0 = []
        ni = []
        passfails = []
        testintervals = []

        for asset in self.assets:
            pf, ti = self.get_passfails(asset)
            passfails.extend(pf)
            testintervals.extend(ti)

        testintervals = np.array(testintervals)
        passfails = np.array(passfails)

        binlefts = binlefts if binlefts is not None else self.binlefts
        binwidth = binwidth if binwidth is not None else self.binwidth

        # Includes left and right edges
        if binlefts is None:
            binedges = np.histogram_bin_edges(testintervals, bins=self.bins)
            binlefts = binedges[:-1]
            binwidth = binedges[1] - binedges[0]

        for left in binlefts:
            idx = (testintervals > np.floor(left)) & (testintervals <= np.ceil(left+binwidth))
            if len(testintervals[idx]) > 0:
                R.append(passfails[idx].mean())
                t.append(left+binwidth)
                t0.append(left)
                ni.append(len(testintervals[idx]))
        return t, t0, R, ni

    def to_binomialinterval(self):
        ''' Convert assets into BinomialInterval '''
        t, ti0, R, ni = self.get_reliability()
        return BinomialInterval(self.Rtarget, ti=t, Ri=R, ni=ni, ti0=ti0)

    def calculate(self):
        ''' Calculate both methods '''
        calc = self.to_binomialinterval()
        return calc.calculate()
