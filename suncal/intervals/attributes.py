''' Calculations for Interval Analysis using Attributes Data

- Binomial Method (S2)
- Test Interval Method (A3)

Methods A1, A2 are discouraged and thus not implemented. No implementation of Method S1 is given,
as Method S2 is favored over S1.

Use: Variables data method if variables data is available at various intervals.
     Method S2 if lots of data at various intervals is available. Use A3 if most calibrations
     are in the same interval range.
'''

import warnings
import yaml
import numpy as np
from scipy import stats
from scipy.optimize import curve_fit, OptimizeWarning
from dateutil.parser import parse
import matplotlib.pyplot as plt

from .. import output
from .. import report
from .. import plotting
from .. import curvefit
from .. import ttable

# Curve fit solver will throw warnings due to poor fits. Filter them out.
warnings.filterwarnings('ignore', category=RuntimeWarning, module='scipy')
warnings.filterwarnings('ignore', category=OptimizeWarning, module='scipy')


def datearray(dates):
    ''' Convert array to ordinal date. Input can be datetime or string '''
    if len(dates) == 0:
        return np.array([])
    elif hasattr(dates[0], 'toordinal'):
        dates =[d.toordinal() for d in dates]
    elif isinstance(dates[0], str):
        dates = [parse(d).toordinal() for d in dates]
    return np.asarray(dates)


# Reliability Models
def R_exp(t, theta):
    return np.exp(-theta * t)

def R_weibull(t, theta1, theta2):
    return np.exp(-(theta1*t)**theta2)

def R_expmixed(t, theta1, theta2):
    return (1+theta1*t)**(-theta2)

def R_walk(t, theta1, theta2):
    Q = 1/np.sqrt(theta1+theta2*t)
    return 2*(stats.norm.cdf(Q)) - 1

def R_restrictedwalk(t, theta1, theta2, theta3):
    Q = 1/np.sqrt(theta1+theta2*(1-np.exp(-theta3*t)))
    return 2*(stats.norm.cdf(Q)) - 1

def R_gamma(t, theta):
    return np.exp(-theta*t) * sum([1, theta*t, ((theta*t)**2)/2, ((theta*t)**3)/6])

def R_mortality(t, theta1, theta2):
    return np.exp(-(theta1*t + theta2*t**2))

def R_warranty(t, theta1, theta2):
    return 1/(1 + np.exp(theta1*(t-theta2)))

def R_drift(t, theta1, theta2, theta3):
    # -1 (-0.5 for each cdf) to shift to the right place (PHI(x) in RP1 = cdf(x)-0.5)
    # RP1 seems to have wrong theta values for the plot in D-11, but 2.5, 2.5, and 0.5 seem to match?
    return stats.norm.cdf(theta1 + theta3*t) + stats.norm.cdf(theta2 - theta3*t) - 1

def R_lognorm(t, theta1, theta2):
    return 1 - stats.norm.cdf(np.log(theta1*t)/theta2)


# Functions for determining a reasonable initial guess for curve fit
def guess_exp(t, y):
    y = np.nan_to_num(-np.log(y))  # Linearize and fit straight line
    theta1 = curve_fit(lambda x, a: x*a, t, y)[0][0]
    return theta1

def guess_weibull(t, y):
    y[y == 1] = .99999
    t = np.nan_to_num(np.log(t))
    y = np.nan_to_num(np.log(-np.log(y)))  # Linearize
    coef = np.polyfit(t, y, deg=1)
    theta2 = coef[0]
    theta1 = np.exp(coef[1]/theta2)
    return theta1, theta2

def guess_expmixed(t, y):
    # Use Approximation: (1+x)**r ~= (1+xr)
    y = 1-y
    theta1theta2 = curve_fit(lambda x, a: x*a, t, y)[0][0]
    theta1 = theta1theta2/2
    theta2 = theta1theta2/theta1
    return theta1, theta2

def guess_walk(t, y):
    yy = stats.norm.ppf((y+1)/2)**-2    # Invert the cdf
    theta1 = yy[yy > 0][0]**2
    theta2 = 1/t[np.argmin(abs(y-(y.max()+y.min())/2))]
    return theta1, theta2

def guess_rwalk(t, y):
    y = (stats.norm.ppf((y+1)/2))**-2
    theta1 = np.nanmin(y)   # t->0
    t1plust2 = y[-1] if np.isfinite(y[-1]) else np.nanmean(y)    # t->inf
    theta2 = t1plust2 - theta1
    theta3 = 1/t.mean()     # ~ decay rate
    return theta1, theta2, theta3

def guess_gamma(t, y):
    yy = np.nan_to_num(-np.log(y))   # Ignore the SUM terms...
    theta1 = curve_fit(lambda x, a: x*a, t, yy)[0][0]
    return theta1

def guess_mortality(t, y):
    yy = np.nan_to_num(np.log(y))  # Quadratic after linearizing
    theta1, theta2 = curve_fit(lambda x, a, b: b*x**2 -a*x, t, yy)[0]
    return theta1, theta2

def guess_warranty(t, y):
    yy = np.nan_to_num(np.log(1/y-1))  # Invert/linearize
    theta1, theta1theta2 = np.polyfit(t, yy, deg=1)
    theta2 = -theta1theta2/theta1
    return theta1, theta2

def guess_drift(t, y):
    t1overt3 = t.mean()  # Inflection point ~= theta1/theta3
    theta1 = theta2 = 2
    theta3 = theta1/t1overt3
    return theta1, theta2, theta3

def guess_lognorm(t, y):
    ythresh = (y.max()+y.min())/2
    tthresh = t[np.abs(y-ythresh).argmin()]
    theta1 = 1/tthresh
    theta2 = 1/(t.max()/2)
    return theta1, theta2


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
        for i in range(len(groups)):
            vals = x[start:int(groups[i])+1]
            bins.append(vals[-1]+.1)
            start = int(groups[i])+1

    binidx = np.digitize(x, bins=bins)-1
    cnt = []
    for b in binidx:
        cnt.append(len(binidx) - np.count_nonzero(binidx-b))
    cnt = np.array(cnt)[np.argsort(idx)]
    return cnt


class BinomialInterval(object):
    ''' Class for calculating calibration interval by Binomial Method (S2 in RP1)

        Parameters
        ----------
        Rtarget: float
            Target reliability (0-1)
        ti: array
            Observed intervals (right edge of bin)
        ti0: array, optional
            Observed interval, left edge of bin
        Ri: array
            Observed reliability for each interval
        ni: array
            Number of measurements in this interval

        Notes:
        ------
        ti, Ri, and ni parameters are used to set up calculation if calibration
        data has already been binned into discrete intervals with reliability.
        Otherwise, use from_passfail() method to set up calculation based on
        individual measurement points. If ti0 is None, there will be no gaps
        between bins.
    '''
    models = {'Exponential': R_exp,
              'Weibull': R_weibull,
              'Mixed Exponential': R_expmixed,
              'Random Walk': R_walk,
              'Restricted Walk': R_restrictedwalk,
              'Modified Gamma': R_gamma,
              'Mortality Drift': R_mortality,
              'Warranty': R_warranty,
              'Drift': R_drift,
              'Log Normal': R_lognorm}

    guessers = {'Exponential': guess_exp,
              'Weibull': guess_weibull,
              'Mixed Exponential': guess_expmixed,
              'Random Walk': guess_walk,
              'Restricted Walk': guess_rwalk,
              'Modified Gamma': guess_gamma,
              'Mortality Drift': guess_mortality,
              'Warranty': guess_warranty,
              'Drift': guess_drift,
              'Log Normal': guess_lognorm}

    def __init__(self, Rtarget=0.95, ti=None, ti0=None, Ri=None, ni=None, conf=0.95, name='interval'):
        self.name = name
        self.description = ''
        self.Rtarget = Rtarget
        self.conf = conf
        self.ti0 = np.asarray(ti0).astype(float) if ti0 is not None else np.array([])  # Calibration interval (right side of bin)
        self.ti = np.asarray(ti).astype(float) if ti is not None else np.array([])  # Calibration interval (right side of bin)
        self.Ri = np.asarray(Ri).astype(float) if Ri is not None else np.array([])  # Observed reliability for that interval
        self.ni = np.asarray(ni).astype(float) if ni is not None else np.array([])  # Number of measurements made at that interval

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
        ''' Update calibration data. Don't change if None. '''
        self.ti = ti if ti is not None else self.ti
        self.ti0 = ti0 if ti0 is not None else self.ti0
        self.Ri = ri if ri is not None else self.Ri
        self.ni = ni if ni is not None else self.ni

    def update_params(self, Rt, conf):
        ''' Update parameters, reliability and confidence '''
        self.conf = conf
        self.Rtarget = Rt

    def set_p0(self, model, p0):
        ''' Set initial guess for fitting model '''
        self.p0[model] = p0

    def add_model(self, name, func, p0=None):
        ''' Add a reliability model

            Parameters
            ----------
            name: string
                Name of reliability model
            func: callable
                Function taking time as first argument, and other arguments to define the model
        '''
        self.models[name] = func
        if callable(p0):
            self.guessers[name] = p0
        elif p0 is not None:
            self.p0[name] = p0

    def calculate(self):
        ''' Calculate intervals using each model '''
        arr = curvefit.Array(self.ti, self.Ri)   # Fitting to right edge of each bin
        k = len(arr)      # Number of intervals/bins
        n = sum(self.ni)  # Total number of measurements made

        if k < 2:
            warnings.warn('Not enough data to compute interval')
            self.results = None
            self.out = BinomialIntervalOutput(results=None)
            return self.out

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

            fit = curvefit.CurveFit(arr, model, p0=p0)
            try:
                fit.calculate()
            except (RuntimeError, TypeError):
                # Fit failed to converge, use defaults
                print(name, 'failed to converge!')
                pass
            else:
                theta = fit.out.lsq.coeffs
                m = len(theta)  # Number of fit parameters

                # Find intersection of fit and Rtarget numerically
                # fsolve has problems if y has nans, which may be the case
                # as t->0. Numerical result is fine as we round to
                # nearest integer anyway.
                xx = np.linspace(0, arr.x.max(), num=1000)
                yy = fit.out.lsq.y(xx)
                yy[~np.isfinite(yy)] = 1E99
                interval = xx[np.argmin(abs(yy - self.Rtarget))]

                kfactor = ttable.t_factor(self.conf, k-m)
                yunc = kfactor * fit.out.lsq.u_conf(xx)
                yunc[~np.isfinite(yunc)] = 1E99
                tau_u = xx[np.argmin(abs(yy+yunc - self.Rtarget))]
                tau_l = xx[np.argmin(abs(yy-yunc - self.Rtarget))]

                if np.isclose(fit.out.lsq.y(interval), self.Rtarget, atol=.01):
                    interval = np.round(interval)
                    se2 = sum(self.ni * self.Ri * (1 - self.Ri)) / (n-k)                # RP1 eq D-19
                    sl2 = sum(self.ni * (self.Ri - model(self.ti, *theta))**2) / (k-m)  # RP1 eq D-23
                    F = sl2/se2
                    Fcrit = stats.f.ppf(self.conf, dfn=k-m, dfd=n-k)    # Critical F parameter (dfn = numerator, dfd=denominator)
                    C = stats.f.cdf(F, dfn=k-m, dfd=n-k) * 100          # Rejection Confidence
                    accept = F < Fcrit
                else:
                    interval = tau_l = tau_u = 0

            results[name] = {'interval': interval,
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
                             'binned': {'interval': self.ti, 'binleft': self.ti0, 'reliability': self.Ri, 'number': self.ni}
                             }

        # Group them by interval similarity to compute figure of merit
        I = np.array([r['interval'] for r in results.values()])
        Ng = _count_groups(I)  # Number of models in a "group" with similar interval result

        for idx, name in enumerate(self.models.keys()):
            results[name]['Ng'] = Ng[idx]
            results[name]['G'] = Ng[idx] / (results[name]['C']/100) * results[name]['interval'] ** 0.25

        self.results = results
        self.out = BinomialIntervalOutput(self.results)
        return self.out

    def get_config(self):
        ''' Get configuration dictionary '''
        d = {}
        d['mode'] = 'intervalbinom'
        d['name'] = self.name
        d['desc'] = self.description
        d['Rt'] = self.Rtarget
        d['conf'] = self.conf
        d['ti'] = list(self.ti)
        d['ri'] = list(self.Ri)
        d['ni'] = list(self.ni)
        return d

    @classmethod
    def from_config(cls, config):
        ''' Load interval object from config dictionary '''
        new = cls(Rtarget=config.get('Rt', .95),
                  ti=config.get('ti', []),
                  Ri=config.get('ri', []),
                  ni=config.get('ni', []),
                  conf=config.get('conf', 0.95),
                  name=config.get('name', 'interval'))
        new.description = config.get('desc', '')
        return new

    def save_config(self, fname):
        ''' Save configuration to file.

            Parameters
            ----------
            fname: string or file
                File name or file object to save to
        '''
        d = self.get_config()
        out = yaml.dump([d], default_flow_style=False)
        try:
            fname.write(out)
        except AttributeError:
            with open(fname, 'w') as f:
                f.write(out)

    @classmethod
    def from_configfile(cls, fname):
        ''' Read and parse the configuration file. Returns a new Risk
            instance.

            Parameters
            ----------
            fname: string or file
                File name or open file object to read configuration from
        '''
        try:
            try:
                yml = fname.read()  # fname is file object
            except AttributeError:
                with open(fname, 'r') as fobj:  # fname is string
                    yml = fobj.read()
        except UnicodeDecodeError:
            # file is binary, can't be read as yaml
            return None

        try:
            config = yaml.safe_load(yml)
        except yaml.YAMLError:
            return None  # Can't read YAML

        u = cls.from_config(config[0])  # config yaml is always a list
        return u


class BinomialIntervalAssets(object):
    ''' Binomial Interval from individual asset's data '''
    def __init__(self, Rt=0.9, bins=10, conf=0.95, binlefts=None, binwidth=None, name='interval'):
        self.name = name
        self.description = ''
        self.Rtarget = Rt
        self.bins = bins
        self.binlefts = binlefts
        self.binwidth = binwidth
        self.conf = conf
        self.assets = {}

    def updateasset(self, assetname, enddates, passfail, startdates=None, **kwargs):
        ''' Update the asset calibration data

            Parameters
            ----------
            assetname: string
                Name of the asset (key into self.assets dict)
            enddates: array
                List of ending dates for each cal cycle
            passfail: array
                List of pass/fail (1/0) values for each cal
            startdates: array (optional)
                List of starting dates for each cal cycle

            Keyword arguments not used. For call signature compatibility
            with other class.
        '''
        self.assets[assetname] = {'startdates': startdates,
                                  'enddates': enddates,
                                  'passfail': passfail}

    def update_params(self, Rt=.9, conf=.95, bins=10, binlefts=None, binwidth=None):
        ''' Update target, conf, and bins parameters '''
        self.conf = conf
        self.Rtarget = Rt
        self.bins = bins
        self.binlefts = binlefts
        self.binwidth = binwidth

    def remasset(self, assetname):
        ''' Remove asset '''
        self.assets.pop(assetname, None)

    def get_passfails(self, asset):
        ''' Get list of interval, passfail values '''
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

            Parameters
            ----------
            binlefts: list
                List of left-edges of each bin
            binwidth: float
                Width of all bins
            
            If parameters are not provided, self attributes
            will be used.
        '''
        R = []
        t = []
        t0 = []
        ni = []
        passfails = []
        testintervals = []

        for asset in self.assets.keys():
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
        self.calc = self.to_binomialinterval()
        self.out = self.calc.calculate()
        return self.out

    def get_config(self):
        ''' Get configuration dictionary '''
        d = {}
        d['mode'] = 'intervalbinomasset'
        d['name'] = self.name
        d['desc'] = self.description
        d['target'] = self.Rtarget
        d['bins'] = self.bins
        d['binlefts'] = self.binlefts
        d['binwidth'] = self.binwidth
        d['assets'] = {}
        for a, vals in self.assets.items():
            d['assets'][a] = {'startdates': list(vals['startdates']) if vals['startdates'] is not None else None,
                              'enddates': list(vals['enddates']) if vals['enddates'] is not None else None,
                              'passfail': list(vals['passfail']) if vals['passfail'] is not None else None}
        return d

    @classmethod
    def from_config(cls, config):
        ''' Load interval object from config dictionary '''
        new = cls(Rt=config.get('target', .9),
                  bins=config.get('bins', 10),
                  binlefts=config.get('binlefts', None),
                  binwidth=config.get('binwidth', None),
                  conf=config.get('conf', 0.95),
                  name=config.get('name', 'interval'))
        new.description = config.get('desc', '')
        new.assets = config.get('assets', {})
        return new

    def save_config(self, fname):
        ''' Save configuration to file.

            Parameters
            ----------
            fname: string or file
                File name or file object to save to
        '''
        d = self.get_config()
        out = yaml.dump([d], default_flow_style=False)
        try:
            fname.write(out)
        except AttributeError:
            with open(fname, 'w') as f:
                f.write(out)

    @classmethod
    def from_configfile(cls, fname):
        ''' Read and parse the configuration file. Returns a new Risk
            instance.

            Parameters
            ----------
            fname: string or file
                File name or open file object to read configuration from
        '''
        try:
            try:
                yml = fname.read()  # fname is file object
            except AttributeError:
                with open(fname, 'r') as fobj:  # fname is string
                    yml = fobj.read()
        except UnicodeDecodeError:
            # file is binary, can't be read as yaml
            return None

        try:
            config = yaml.safe_load(yml)
        except yaml.YAMLError:
            return None  # Can't read YAML

        u = cls.from_config(config[0])  # config yaml is always a list
        return u


class BinomialIntervalOutput(output.Output):
    ''' Output report for Binomial interval calculation '''
    def __init__(self, results):
        if results is not None:
            self.results = {k: results[k] for k in sorted(results, key=lambda x: (results[x]['Ng'], results[x]['G']), reverse=True)}
            self.best = max(self.results, key=lambda x: self.results[x]['G'])
            self.interval = results[self.best]['interval']
        else:
            self.results = None
            self.best = None
            self.interval = None

    def report(self, **kwargs):
        ''' Report of best model '''
        return self.report_model(self.best)

    def report_all(self, **kwargs):
        ''' Report everything '''
        rpt = self.report_allmodels(**kwargs)
        with plt.style.context(plotting.plotstyle):
            fig = self.plot_allmodels(**kwargs)
        rpt.plot(fig)
        plt.close(fig)
        rpt.append(self.report_bins(**kwargs))
        return rpt

    def report_bins(self, **kwargs):
        ''' Report table of binned data '''
        hdr = ['Range', 'Reliability', 'Number of measurements']
        rows = []
        
        ti = self.results[self.best]['binned']['interval']
        ri = self.results[self.best]['binned']['reliability']
        ni = self.results[self.best]['binned']['number']
        binleft = self.results[self.best]['binned']['binleft']
        if binleft is None:
            binleft = ti - ti[0]

        for bleft, t, r, n in zip(binleft, ti, ri, ni):
            rows.append(['{:.0f} - {:.0f}'.format(bleft, t), format(r, '.3f'), format(n, '.0f')])
        rpt = report.Report(**kwargs)
        rpt.hdr('Binned reliability data', level=2)
        rpt.table(rows, hdr)
        return rpt

    def report_model(self, model, **kwargs):
        ''' Report of one model '''
        hdr = ['Interval', 'Model', 'Rejection Confidence', '{:.1f}% Confidence Interval Range'.format(self.results[model]['conf']*100)]
        rows = [[format(self.results[model]['interval'], '.1f'),
                 model,
                 '{:.1f}%'.format(self.results[model]['C']),
                 '{:.1f} - {:.1f}'.format(*self.results[model]['interval_range'])]]
        rpt = report.Report(**kwargs)
        rpt.hdr('Best Fit Model', level=2)
        rpt.table(rows, hdr)
        return rpt

    def report_allmodels(self, **kwargs):
        ''' Report a table of all reliability models for comparison '''
        hdr = ['Reliability Model', 'Interval', 'Rejection Confidence', 'F-Test', 'Figure of Merit']
        rows = []
        for name, r in self.results.items():
            rows.append([name, format(r['interval'], '.0f'), '{:.2f}%'.format(r['C']), str(r['accept']), format(r['G'], '.2f')])
        rpt = report.Report(**kwargs)
        rpt.table(rows, hdr)
        return rpt

    def report_summary(self, **kwargs):
        ''' Summary report of best model, including plot '''
        if self.results is None:
            rpt = report.Report(**kwargs)
            rpt.txt('Not enough data to compute interval')
        else:
            rpt = self.report(**kwargs)
            with plt.style.context(plotting.plotstyle):
                fig = self.plot(**kwargs)
                rpt.plot(fig)
                plt.close(fig)
                rpt.hdr('All Models', level=3)
                rpt.append(self.report_allmodels(**kwargs))
                fig = self.plot_allmodels(**kwargs)
                rpt.plot(fig)
                plt.close(fig)
            rpt.append(self.report_bins(**kwargs))
        return rpt

    def plot(self, **kwargs):
        ''' Plot of best model '''
        fig = self.plot_model(self.best, **kwargs)
        fig.suptitle(self.best)
        return fig

    def plot_model(self, model, ax=None, axlabels=True, **kwargs):
        ''' Plot individual model '''
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.figure

        ax.plot(self.results[model]['arr'].x, self.results[model]['arr'].y*100, ls='', marker='o')
        if self.results[model]['theta'] is not None:
            xx = np.linspace(0, self.results[model]['arr'].x.max(), num=100)
            yy = BinomialInterval.models[model](xx, *self.results[model]['theta'])
            ax.plot(xx, yy*100, ls='-')
        if self.results[model]['interval'] > 0:
            ax.axvline(self.results[model]['interval'], ls=':', color='black')
        ax.axhline(self.results[model]['target']*100, ls=':', color='black')
        if axlabels:
            ax.set_xlabel('Interval Days')
            ax.set_ylabel('Reliability %')
        return fig

    def plot_allmodels(self, fig=None, **kwargs):
        ''' Plot all the models '''
        if fig is None:
            fig = plt.figure(figsize=(10, 10))
        fig.clf()

        axs = plotting.axes_grid(len(BinomialInterval.models), fig=fig, maxcols=3)
        for ax, modelname in zip(axs, self.results.keys()):
            self.plot_model(modelname, ax=ax, axlabels=False, **kwargs)
            ax.set_title(modelname)
        fig.tight_layout()
        return fig


class TestInterval(object):
    ''' Interval Test Method (A3) from RP1

        Parameters
        ----------
        intol: int
            Number of calibrations in tolerance during the interval period I0
        n: int
            Total number of calibrations performed during the interval period I0
        I0: float
            Current interval
        Rt: float
            Reliability target
        maxchange: float
            Maximum ratio allowable change in interval. Equal to the "b" parameter in RP1 Method A3.
        conf: float
            Interval change confidence. Interval will be changed from I0 if at least this much
            confidence that the new interval is better.
    '''
    def __init__(self, intol=0, n=1, I0=365, Rt=.9, maxchange=2, conf=.5, mindelta=5, minint=14,
                 maxint=1826, unused=None, name='interval'):
        self.name = name
        self.description = ''
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
        ''' Update parameters. Don't change if None. '''
        self.intol = intol if intol is not None else self.intol
        self.n = n if n is not None else self.n

    def update_params(self, I0=365, Rt=.9, maxchange=2, conf=.5, mindelta=5, minint=14, maxint=1826):
        ''' Update calculation parameters '''
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
            result = {'interval': np.nan,
                      'calculated': np.nan,
                      'rejection': 1,
                      'RL': 0,
                      'RU': 1,
                      'Robserved': np.nan,
                      'intol': np.nan,
                      'n': np.nan,
                      'unused': None,
                      'conf': self.conf}
            self.out = TestIntervalOutput(result)
            return self.out

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
                I = b
            else:
                I = w
        else:
            if v < a:
                I = a
            else:
                I = v
        I = I * self.I0

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
        if Q > self.conf and abs(I - self.I0) >= self.mindelta:
            newI = min(max(I, self.minint), self.maxint)
        else:
            newI = self.I0

        result = {'interval': newI,
                  'calculated': I,
                  'rejection': max(Q, 0),
                  'RL': RL,
                  'RU': RU,
                  'Robserved': R0,
                  'intol': self.intol,
                  'n': self.n,
                  'unused': self.unused,
                  'conf': self.conf}
        self.out = TestIntervalOutput(result)
        return self.out

    def get_config(self):
        ''' Get configuration dictionary '''
        d = {}
        d['mode'] = 'intervaltest'
        d['name'] = self.name
        d['desc'] = self.description
        d['intol'] = self.intol
        d['total'] = self.n
        d['I0'] = self.I0
        d['target'] = self.Rtarget
        d['maxchange'] = self.maxchange
        d['conf'] = self.conf
        d['mindelta'] = self.mindelta
        d['minint'] = self.minint
        d['maxint'] = self.maxint
        return d

    @classmethod
    def from_config(cls, config):
        ''' Load interval object from config dictionary '''
        new = cls(intol=config.get('intol', 0),
                  n=config.get('total', 0),
                  I0=config.get('I0', 365),
                  Rt=config.get('Rtarget', .95),
                  maxchange=config.get('maxchange', 2),
                  conf=config.get('conf', .5),
                  mindelta=config.get('mindelta', 5),
                  minint=config.get('conf', 14),
                  maxint=config.get('conf', 1826),
                  name=config.get('name', 'interval')
                  )
        new.description = config.get('desc', '')
        return new

    def save_config(self, fname):
        ''' Save configuration to file.

            Parameters
            ----------
            fname: string or file
                File name or file object to save to
        '''
        d = self.get_config()
        out = yaml.dump([d], default_flow_style=False)
        try:
            fname.write(out)
        except AttributeError:
            with open(fname, 'w') as f:
                f.write(out)

    @classmethod
    def from_configfile(cls, fname):
        ''' Read and parse the configuration file. Returns a new Risk
            instance.

            Parameters
            ----------
            fname: string or file
                File name or open file object to read configuration from
        '''
        try:
            try:
                yml = fname.read()  # fname is file object
            except AttributeError:
                with open(fname, 'r') as fobj:  # fname is string
                    yml = fobj.read()
        except UnicodeDecodeError:
            # file is binary, can't be read as yaml
            return None

        try:
            config = yaml.safe_load(yml)
        except yaml.YAMLError:
            return None  # Can't read YAML

        u = cls.from_config(config[0])  # config yaml is always a list
        return u


class TestIntervalAssets(object):
    ''' Test Interval method using data from individual assets '''
    def __init__(self, I0=180, Rt=0.95, name='interval'):
        self.name = name
        self.description = ''
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

            Parameters
            ----------
            assetname: string
                Name of the asset (key into self.assets dict)
            enddates: array
                List of ending dates for each cal cycle
            passfail: array
                List of pass/fail (1/0) values for each cal
            startdates: array (optional)
                List of starting dates for each cal cycle
                
            Keyword arguments not used. For call signature compatibility
            with other class.
        '''
        self.assets[assetname] = {'startdates': startdates,
                                  'enddates': enddates,
                                  'passfail': passfail}

    def update_params(self, I0=365, Rt=.9, maxchange=2, conf=.5, mindelta=5, minint=14, maxint=1826, tol=56, thresh=999):
        ''' Update calculation parameters '''
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
        ''' Remove asset '''
        self.assets.pop(assetname, None)

    def to_testinterval(self):
        ''' Convert to summarized TestInterval class '''
        intol, n, total = self.get_intol()
        return TestInterval(intol, n, I0=self.I0, Rt=self.Rtarget, maxchange=self.maxchange,
                            mindelta=self.mindelta, minint=self.minint, maxint=self.maxint,
                            unused=total-n, conf=self.conf)

    def get_intol(self):
        ''' Get number in-tolerance and total of usable calibrations, and total number of all calibrations '''
        passes = 0
        totalused = 0
        total = 0
        for aname, val in self.assets.items():
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
        self.calc = self.to_testinterval()
        self.out = self.calc.calculate()
        return self.out

    def get_config(self):
        ''' Get configuration dictionary '''
        d = {}
        d['mode'] = 'intervaltestasset'
        d['name'] = self.name
        d['desc'] = self.description
        d['I0'] = self.I0
        d['target'] = self.Rtarget
        d['tol'] = self.tol
        d['thresh'] = self.thresh
        d['maxchange'] = self.maxchange
        d['conf'] = self.conf
        d['mindelta'] = self.mindelta
        d['minint'] = self.minint
        d['maxint'] = self.maxint
        d['assets'] = {}
        for a, vals in self.assets.items():
            d['assets'][a] = {'startdates': list(vals['startdates']) if vals['startdates'] is not None else None,
                              'enddates': list(vals['enddates']) if vals['enddates'] is not None else None,
                              'passfail': list(vals['passfail']) if vals['passfail'] is not None else None}
        return d

    @classmethod
    def from_config(cls, config):
        ''' Load interval object from config dictionary '''
        new = cls(name=config.get('name', 'interval'))
        new.description = config.get('desc', '')
        new.assets = config.get('assets', {})
        new.I0 = config.get('I0', 365)
        new.Rtarget = config.get('Rtarget', .95)
        new.tol = config.get('tol', 56)
        new.thresh = config.get('thresh', .5)
        new.maxchange = config.get('maxchange', 2)
        new.conf = config.get('conf', .5)
        new.mindelta = config.get('mindelta', 5)
        new.minint = config.get('minint', 14)
        new.maxint = config.get('maxint', 1826)
        return new

    def save_config(self, fname):
        ''' Save configuration to file.

            Parameters
            ----------
            fname: string or file
                File name or file object to save to
        '''
        d = self.get_config()
        out = yaml.dump([d], default_flow_style=False)
        try:
            fname.write(out)
        except AttributeError:
            with open(fname, 'w') as f:
                f.write(out)

    @classmethod
    def from_configfile(cls, fname):
        ''' Read and parse the configuration file. Returns a new Risk
            instance.

            Parameters
            ----------
            fname: string or file
                File name or open file object to read configuration from
        '''
        try:
            try:
                yml = fname.read()  # fname is file object
            except AttributeError:
                with open(fname, 'r') as fobj:  # fname is string
                    yml = fobj.read()
        except UnicodeDecodeError:
            # file is binary, can't be read as yaml
            return None

        try:
            config = yaml.safe_load(yml)
        except yaml.YAMLError:
            return None  # Can't read YAML

        u = cls.from_config(config[0])  # config yaml is always a list
        return u


class TestIntervalOutput(output.Output):
    ''' Output report for Test Interval (A3) calculation '''
    def __init__(self, results):
        self.results = results
        self.interval = results['interval']

    def report(self, **kwargs):
        rows = [['Suggested Interval', report.Number(self.results['interval'], fmin=0)],
                ['Calculated Interval', report.Number(self.results['calculated'], fmin=0)],
                ['Current Interval Rejection Confidence', '{:.2f}%'.format(self.results['rejection']*100)],   # Confidence with which the original I0 interval was rejected
                ['True reliability range', '{:.2f}% - {:.2f}%'.format(self.results['RL']*100, self.results['RU']*100)],
                ['Observed Reliability', '{:.2f}% ({} / {})'.format(self.results['Robserved']*100, self.results['intol'], self.results['n'])],
                ['Number of calibrations used', '{:.0f}'.format(self.results['n'])]
                ]
        if self.results['unused'] is not None:
            rows.append(['Rejected calibrations (wrong interval)', format(self.results['unused'])])
        rpt = report.Report(**kwargs)
        rpt.table(rows, hdr=['Parameter', 'Value'])
        return rpt
