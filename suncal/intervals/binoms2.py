''' Binomial Method S2 '''

import warnings
import logging
from collections import namedtuple
import numpy as np
from scipy import stats
from scipy.optimize import OptimizeWarning
from dateutil.parser import parse

from ..curvefit import CurveFit, Array
from ..common import ttable
from .report.attributes import ReportIntervalS2
from . import s2models


Binned = namedtuple('Binned', 'interval binleft reliability number')
S2Result = namedtuple('S2Result', 'interval interval_range conf theta F Fcrit C accept arr target guess binned Ng G')

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


class ResultsBinomIntervalS2:
    ''' Results from Binomial Interval (S2) Method

        Attributes:
            interval: Recommended interval using best fit reliability model
            best: Name of best fit reliability model

        Methods:
            method: Get results from one reliability model
    '''
    def __init__(self, methodresults):
        self._methodresults = {k: methodresults[k] for k in sorted(
            methodresults, key=lambda x: (methodresults[x].Ng, methodresults[x].G), reverse=True)}
        self.best = max(self._methodresults, key=lambda x: self._methodresults[x].G)
        self.interval = self._methodresults[self.best].interval
        self.report = ReportIntervalS2(self)

    def _repr_markdown_(self):
        return self.report.summary().get_md()

    def method(self, name):
        return self._methodresults.get(name)


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
    def __init__(self, Rtarget=0.95, ti=None, ti0=None, Ri=None, ni=None, conf=0.95):
        self.Rtarget = Rtarget
        self.conf = conf
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

    def update_params(self, Rt, conf):
        ''' Update parameters, reliability and confidence

            Args:
                Rtarget (float): Target reliability (0-1)
                conf (float): Level of confidence
        '''
        self.conf = conf
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
        arr = Array(self.ti, self.Ri)   # Fitting to right edge of each bin
        k = len(arr)      # Number of intervals/bins
        n = sum(self.ni)  # Total number of measurements made

        if k < 2:
            warnings.warn('Not enough data to compute interval')
            self.results = None
            return self.results

        results = {}
        for name, model in self.models.items():
            # Defaults if things fail
            interval = 0
            tau_u = tau_l = 0
            theta = None
            F = np.inf
            Fcrit = 0
            C = 100
            accept = False

            if name in self.p0:
                p0 = self.p0[name]
            elif name in self.guessers:
                p0 = self.guessers.get(name)(arr.x, arr.y)
            else:
                p0 = None  # Use curve_fit default

            fit = CurveFit(arr, model, p0=p0)
            try:
                fitresult = fit.calculate_lsq()
            except (RuntimeError, TypeError):
                # Fit failed to converge, use defaults
                logging.warning('%s failed to converge!', name)
            else:
                theta = fitresult.coeffs
                m = len(theta)  # Number of fit parameters

                # Find intersection of fit and Rtarget numerically
                # fsolve has problems if y has nans, which may be the case
                # as t->0. Numerical result is fine as we round to
                # nearest integer anyway.
                xx = np.linspace(0, arr.x.max(), num=1000)
                yy = fitresult.y(xx)
                yy[~np.isfinite(yy)] = 1E99
                interval = xx[np.argmin(abs(yy - self.Rtarget))]

                kfactor = ttable.k_factor(self.conf, k-m)
                yunc = kfactor * fitresult.confidence_band(xx)
                yunc[~np.isfinite(yunc)] = 1E99
                tau_u = xx[np.argmin(abs(yy+yunc - self.Rtarget))]
                tau_l = xx[np.argmin(abs(yy-yunc - self.Rtarget))]

                if np.isclose(fitresult.y(interval), self.Rtarget, atol=.01):
                    interval = np.round(interval)
                    se2 = sum(self.ni * self.Ri * (1 - self.Ri)) / (n-k)                # RP1 eq D-19
                    sl2 = sum(self.ni * (self.Ri - model(self.ti, *theta))**2) / (k-m)  # RP1 eq D-23
                    F = sl2/se2
                    Fcrit = stats.f.ppf(self.conf, dfn=k-m, dfd=n-k)  # Critical F parameter
                    C = stats.f.cdf(F, dfn=k-m, dfd=n-k) * 100        # Rejection Confidence
                    accept = F < Fcrit
                else:
                    interval = tau_l = tau_u = 0

            results[name] = {
                'interval': interval,
                'interval_range': (tau_l, tau_u),
                'conf': self.conf,
                'theta': theta,
                'F': F,
                'Fcrit': Fcrit,
                'C': C,
                'accept': accept,
                'arr': arr,
                'target': self.Rtarget,
                'guess': p0,
                'binned': Binned(
                    interval=self.ti,
                    binleft=self.ti0 if len(self.ti0) else [0] + list(self.ti)[:-1],
                    reliability=self.Ri,
                    number=self.ni)
                }

        # Group them by interval similarity to compute figure of merit
        interval = np.array([r['interval'] for r in results.values()])
        Ng = _count_groups(interval)  # Number of models in a "group" with similar interval result

        for idx, name in enumerate(self.models.keys()):
            results[name]['Ng'] = Ng[idx]
            results[name]['G'] = Ng[idx] / (results[name]['C']/100) * results[name]['interval'] ** 0.25

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
    def __init__(self, Rt=0.9, bins=10, conf=0.95, binlefts=None, binwidth=None):
        self.Rtarget = Rt
        self.bins = bins
        self.binlefts = binlefts
        self.binwidth = binwidth
        self.conf = conf
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

    def update_params(self, Rt=.9, conf=.95, bins=10, binlefts=None, binwidth=None):
        ''' Update target, conf, and bins parameters

        Args:
            Rt (float): Reliability target
            conf (float): Level of confidence
            bins (int): Number of bins for converting calibrations into (time, reliability) values
            binlefts (array): Left edge of interval bins
            binwidth (array): Width of interval bins
        '''
        self.conf = conf
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
        return BinomialInterval(self.Rtarget, ti=t, Ri=R, ni=ni, ti0=ti0, conf=self.conf)

    def calculate(self):
        ''' Calculate both methods '''
        calc = self.to_binomialinterval()
        return calc.calculate()
