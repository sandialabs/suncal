''' Interval analysis using variables data

    Based on Castrup "Calibration Intervals from Variables Data" which determines
    how much a device drifts over a certain amount of time. Two methods are calculated:
    1) Uncertainty Target Method: stop the interval when a specific uncertainty is reached
    2) Reliability Target Method: stop the interval when some predetermined reliability is reached
'''

import warnings
import yaml
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from itertools import combinations
from scipy.optimize import fsolve, OptimizeWarning

from .. import output
from .. import report
from .. import plotting
from ..ttable import t_factor, t_onetail
from .attributes import datearray


warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=OptimizeWarning)

# Redefine the curve fit here rather than using curvefit.py for speed/efficiency
def fitpoly(x, y, m=1):
    ''' Fit polynomial, order m, with zero intercept

        Returns
        -------
        b: array
            Polynomial coefficients where
            y = b[0] * x + b[1]*x**2 ... + b[i]*x**(i+1)
        cov: array
            Covariance matrix of b parameters
    '''
    x = np.atleast_1d(x)
    y = np.atleast_1d(y)

    T = np.vstack([x**n for n in range(1, m+1)]).T  # Castrup (7)
    TTinv = np.linalg.inv(T.T @ T)
    b = TTinv @ T.T @ y                             # Castrup (6)

    rss = sum((y - y_pred(x, b))**2)                # Castrup (5)
    s2 = rss / (len(x)-m)                           # Castrup (10)
    S = s2 * np.eye(m)                              # Castrup (9)
    cov = TTinv @ S                                 # Castrup (8)
    return b, cov, np.sqrt(s2)


def y_pred(x, b, y0=0):
    ''' Predict y at the x value given b polynomial coefficients from fitpoly() '''
    scalar = not np.asarray(x).shape
    x = np.atleast_1d(x)
    y = np.zeros(len(x))
    m = len(b)
    for i, xval in enumerate(x):
        tprime = np.array([xval**n for n in range(1, m+1)])
        y[i] = tprime @ b                           # Castrup (12)
    if scalar:
        return y0 + y[0]
    else:
        return y0 + y


def u_pred(x, b, cov, syx):
    ''' Prediction band at x (based on residual scatter) '''
    scalar = not np.asarray(x).shape
    x = np.atleast_1d(x)
    upred = np.zeros(len(x))
    m = len(b)
    for i, xval in enumerate(x):
        tprime = np.array([xval**n for n in range(1, m+1)])
        # upred[i] = syx * np.sqrt(1 + tprime.T @ cov @ tprime)  # Castrup (19) appears to be wrong???
        upred[i] = np.sqrt(syx**2 + (tprime.T @ cov @ tprime))
    if scalar:
        return upred[0]
    else:
        return upred


def u_conf(x, b, cov):
    ''' Confidence band at x (based on residual scatter) '''
    scalar = not np.asarray(x).shape
    x = np.atleast_1d(x)
    uconf = np.zeros(len(x))
    m = len(b)
    for i, xval in enumerate(x):
        tprime = np.array([xval**n for n in range(1, m+1)])
        uconf[i] = np.sqrt(tprime.T @ cov @ tprime)
    if scalar:
        return uconf[0]
    else:
        return uconf


class VariablesInterval(object):
    ''' Calculate interval by variables data method as described in
        Castrup "Calibration Intervals from Variables Data"

        Parameters
        ----------
        dt: array
            Array of times since last calibration
        deltas: array
            Array of deviation from prior calibration for each x value
        u0: float
            Time-of-test uncertainty of measurement
        y0: float
            Initial value at 0 time since calibration
        m: int
            Order of polynomial to fit. Automatically selected if None.
        maxm: int
            Maximum order of polynomial to fit (if m is None)
    '''
    def __init__(self, dt=None, deltas=None, u0=0, y0=0, m=1, maxm=1,
                 utarget=0.5, rlimits=(-1, 1), rconf=.95, name='interval'):
        self.name = name
        self.description = ''
        self.u0 = u0
        self.y0 = y0
        self.maxm = maxm
        self.m = m
        self.utarget = utarget
        self.rlimits = rlimits
        self.rconf = rconf
        self.out = IntervalOutput(None, None, None)
        self.t = np.array([])
        self.y = np.array([])
        self.calcrel = True
        self.calcunc = True
        if dt is not None and deltas is not None:
            self.update(dt, deltas)

    def update(self, t, y):
        ''' Set t, y data for calculation where x is time since last calibration and y
            is deviation from last calibration
        '''
        self.t = np.asarray(t).astype(float)
        self.y = np.asarray(y).astype(float)
        if len(self.t) == 0 or len(self.y) == 0 or len(self.t) != len(self.y):
            self.out = IntervalOutput(None, None, None)
            return self.out

        if self.m is None:
            self._select_order()

        b, cov, syx = fitpoly(self.t, self.y, m=self.m)
        self.out = IntervalOutput(self.out.uncertainty, self.out.reliability, FitOutput(self.t, self.y, b, cov, syx, self.y0, self.u0))
        return self.out.fit

    def update_params(self, u0=0, y0=0, m=1, utarget=.5, rlimitL=-1, rlimitU=1, rconf=0.95, calcrel=True, calcunc=True):
        ''' Update calculation parameters '''
        self.u0 = u0
        self.y0 = y0
        self.m = m
        self.utarget = utarget
        self.rlimits = (rlimitL, rlimitU)
        self.rconf = rconf
        self.calcrel = calcrel
        self.calcunc = calcunc

    def _select_order(self):
        ''' Select polynomial order m for best fit of x, y.
            Limit to maximum order of maxm.
        '''
        # (Castrup section 6)
        smin = np.inf
        m = 1
        for k in range(1, self.maxm+1):
            _, _, syx = fitpoly(self.t, self.y, m=k)
            if syx < smin:
                smin = syx
                m = k
        self.m = m
        return self.m

    def calc_uncertainty_target(self):
        ''' Calculate uncertainty target method
        '''
        def target(t):
            return self.u0**2 + u_pred(t, self.out.fit.b, self.out.fit.cov, self.out.fit.syx)**2 - self.utarget**2

        intv, info, ier, mesg = fsolve(lambda x: target(x), x0=self.t.max(), full_output=True)
        if ier != 1:
            interval = 0
            print('No solution found:', mesg)
        else:
            interval = intv[0]

        result = IntervalUncertOutput(interval=interval, b=self.out.fit.b, cov=self.out.fit.cov,
                                    syx=self.out.fit.syx, target=self.utarget, u0=self.u0,
                                    t=self.t, y=self.y, m=self.m)
        self.out = IntervalOutput(result, self.out.reliability, self.out.fit)
        return result

    def calc_reliability_target(self):
        ''' Calculate reliability target method
        '''
        LL, UL = self.rlimits
        if UL is None or LL is None:
            k = t_onetail(self.rconf, len(self.t)-self.m)
        else:
            k = t_factor(self.rconf, len(self.t)-self.m)

        if all(self.out.fit.b == 0):
            # NO slope. Interval is infinite
            interval = np.inf
        else:
            def upper_lim(t):
                return y_pred(t, self.out.fit.b, y0=self.y0) + k * np.sqrt(self.u0**2 + u_pred(t, self.out.fit.b, self.out.fit.cov, self.out.fit.syx)**2)

            def lower_lim(t):
                return y_pred(t, self.out.fit.b, y0=self.y0) - k * np.sqrt(self.u0**2 + u_pred(t, self.out.fit.b, self.out.fit.cov, self.out.fit.syx)**2)

            t = []
            if (UL is not None and upper_lim(0) > UL) or (LL is not None and lower_lim(0) < LL):
                # Already outside the limits at t=0! Set interval to 0.
                t = [0]

            else:
                if UL is not None:
                    intv, info, ier, mesg = fsolve(lambda x: upper_lim(x) - UL, x0=self.t.mean(), full_output=True)
                    if ier == 1:  # Solution found
                        t.append(intv)

                if LL is not None:
                    intv, info, ier, mesg = fsolve(lambda x: lower_lim(x) - LL, x0=self.t.mean(), full_output=True)
                    if ier == 1:  # Solution found
                        t.append(intv)

            t = np.array(t)
            try:
                interval = t[t > 0].min()
            except ValueError:  # All intervals are negative
                interval = 0

        result = IntervalReliabilityOutput(interval=interval, b=self.out.fit.b, cov=self.out.fit.cov,
                                           syx=self.out.fit.syx, u0=self.u0, LL=LL, UL=UL,
                                           x=self.t, y=self.y, y0=self.y0, m=self.m, k=k)
        self.out = IntervalOutput(self.out.uncertainty, result, self.out.fit)
        return result

    def calculate(self):
        ''' Calculate both reliability target and uncertainty target methods
        '''
        if len(self.t) == 0 or len(self.y) == 0 or len(self.t) != len(self.y):
            self.out = IntervalOutput(None, None, None)
            return self.out

        if self.calcunc:
            self.calc_uncertainty_target()
        if self.calcrel:
            self.calc_reliability_target()
        return self.out

    def get_config(self):
        ''' Get configuration dictionary '''
        d = {}
        d['mode'] = 'intervalvariables'
        d['name'] = self.name
        d['desc'] = self.description
        d['u0'] = self.u0
        d['y0'] = self.y0
        d['maxm'] = self.maxm
        d['m'] = self.m
        d['utarget'] = self.utarget
        d['rlimits'] = list(self.rlimits)
        d['rconf'] = self.rconf
        d['dt'] = list(self.t)
        d['deltas'] = list(self.y)
        return d

    @classmethod
    def from_config(cls, config):
        ''' Load interval object from config dictionary '''
        new = cls(dt=config.get('dt', []),
                  deltas=config.get('deltas', []),
                  u0=config.get('u0', 0),
                  y0=config.get('y0', 0),
                  m=config.get('m', None),
                  maxm=config.get('maxm', 1),
                  utarget=config.get('utarget', 0.5),
                  rlimits=config.get('rlimits', (-1, 1)),
                  rconf=config.get('rconf', 0.95),
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


class VariablesIntervalAssets(object):
    def __init__(self, u0=0, y0=0, m=1, maxm=1,
                 utarget=0.5, rlimits=(-1, 1), rconf=.95,
                 use_alldeltas=False, name='interval'):
        self.name = name
        self.description = ''
        self.u0 = u0
        self.y0 = y0
        self.maxm = maxm
        self.m = m
        self.utarget = utarget
        self.rlimits = rlimits
        self.rconf = rconf
        self.calcrel = True
        self.calcunc = True
        self.use_alldeltas = use_alldeltas
        self.assets = {}

    def updateasset(self, assetname, enddates, asfound, startdates=None, asleft=None):
        ''' Update the asset calibration data

            Parameters
            ----------
            assetname: string
                Name of the asset (key into self.assets dict)
            enddates: array
                List of ending dates for each cal cycle
            asfound: array
                List of as-found calibration values
            asleft: array (optional)
                List of as-left calibration values, if different from as-found
            startdates: array (optional)
                List of starting dates for each cal cycle
        '''
        self.assets[assetname] = {'startdates': startdates,
                                  'enddates': enddates,
                                  'asleft': asleft,
                                  'asfound': asfound}

    def update_params(self, u0=0, y0=0, m=1, utarget=.5, rlimitL=-1, rlimitU=1, rconf=0.95, calcrel=True, calcunc=True):
        ''' Update calculation parameters '''
        self.u0 = u0
        self.y0 = y0
        self.m = m
        self.utarget = utarget
        self.rlimits = (rlimitL, rlimitU)
        self.rconf = rconf
        self.calcrel = calcrel
        self.calcunc = calcunc

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
                              utarget=self.utarget, rlimits=self.rlimits, rconf=self.rconf)
        v.calcrel = self.calcrel
        v.calcunc = self.calcunc
        return v

    def calculate(self):
        ''' Run the calculation '''
        self.calc = self.to_variablesinterval()
        self.out = self.calc.calculate()
        return self.out

    def get_config(self):
        ''' Get configuration dictionary '''
        d = {}
        d['mode'] = 'intervalvariablesasset'
        d['name'] = self.name
        d['desc'] = self.description
        d['u0'] = self.u0
        d['y0'] = self.y0
        d['maxm'] = self.maxm
        d['m'] = self.m
        d['utarget'] = self.utarget
        d['rlimits'] = list(self.rlimits)
        d['rconf'] = self.rconf
        d['assets'] = {}
        for a, vals in self.assets.items():
            d['assets'][a] = {'startdates': list(vals['startdates']) if vals['startdates'] is not None else None,
                              'enddates': list(vals['enddates']) if vals['enddates'] is not None else None,
                              'asleft': list(vals['asleft']) if vals['asleft'] is not None else None,
                              'asfound': list(vals['asfound']) if vals['asfound'] is not None else None}
        return d

    @classmethod
    def from_config(cls, config):
        ''' Load interval object from config dictionary '''
        new = cls(u0=config.get('u0', 0),
                  y0=config.get('y0', 0),
                  m=config.get('m', None),
                  maxm=config.get('maxm', 1),
                  utarget=config.get('utarget', 0.5),
                  rlimits=config.get('rlimits', (-1, 1)),
                  rconf=config.get('rconf', 0.95),
                  name=config.get('name', 'interval'))
        new.assets = config.get('assets', {})
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


class IntervalOutput(output.Output):
    ''' Report for both methods of interval

        Parameters
        ----------
        uncertainty: IntervalUncertOutput
            Results of uncertainty target method
        reliability: IntervalReliabilityOutput
            Results of reliability target method
    '''
    def __init__(self, uncertainty, reliability, fit):
        self.uncertainty = uncertainty
        self.reliability = reliability
        self.fit = fit

    def report(self, **kwargs):
        ''' Generate formatted report '''
        hdr = ['Method', 'Interval']
        rows = []
        if self.uncertainty is not None:
            rows.append(['Uncertainty Target', format(self.uncertainty.interval, '.2f') if self.uncertainty.interval else 'N/A'])
        if self.reliability is not None:
            rows.append(['Reliability Target', format(self.reliability.interval, '.2f') if self.reliability.interval else 'N/A'])
        rpt = report.Report(**kwargs)
        rpt.table(rows, hdr)
        if self.uncertainty is not None and self.uncertainty.t is not None:
            fig = self.uncertainty.plot()
            rpt.plot(fig)
            plt.close(fig)  # Prevent showing duplicate plot in Jupyter
        if self.reliability is not None and self.reliability.x is not None:
            fig = self.reliability.plot()
            rpt.plot(fig)
            plt.close(fig)  # Prevent showing duplicate plot in Jupyter
        return rpt


class FitOutput(output.Output):
    ''' Report for curve fit to deviation vs time since last cal

        Parameters
        ----------
        t: array
        y: array
            The t and y (delta time and deviation from prior) data
        b: array
            Polynomial fit coefficients
        cov: array
            Covariance matrix from polyfit
        syx: float
            RSS sum-squared of residuals from curve fit
        y0: float
            Initial y value at 0 time since last calibration
        u0: float
            Time-of-test uncertainty
    '''
    def __init__(self, t, y, b, cov, syx, y0, u0):
        self.t = np.asarray(t)
        self.y = np.asarray(y)
        self.y0 = y0
        self.b = b
        self.cov = cov
        self.syx = syx
        self.u0 = u0

    def report(self, **kwargs):
        ''' Generate formatted report '''
        rpt = report.Report(**kwargs)
        rpt.hdr('Fit line', level=3)
        rows = [[chr(ord('a')+i), '{:.4e}'.format(v)] for i, v in enumerate(self.b)]
        hdr = ['Parameter', 'Value']
        rpt.table(rows, hdr=hdr)
        return rpt

    def predict_deviation(self, x, **kwargs):
        ''' Predict deviation and uncertainty in deviation at x time since calibration

            Parameters
            ----------
            x: float or array
                Time value at which to predict deviation

            Keyword Arguments
            -----------------
            k: float
                K-value for prediction. Cannot be used with conf argument.
            conf: float
                Confidence for prediction (0-1). Cannot be used with k argument.

            Returns
            -------
            ypred: float or array
                Predicted deviation from prior at time x
            u_ypred: float or array
                Uncertainty in ypred prediction
        '''
        if 'k' in kwargs and 'conf' in kwargs:
            raise ValueError('Specify only one of k or conf.')

        if 'k' in kwargs:
            k = kwargs.get('k')
        else:
            k = t_factor(kwargs.get('conf', 0.95), len(self.t)-len(self.b))
        return y_pred(x, self.b, self.y0), k * np.sqrt(u_pred(x, self.b, self.cov, self.syx)**2 + self.u0**2)

    def plot(self, conf=.95):
        ''' Plot fit line '''
        xx = np.linspace(0, self.t.max())
        fit = y_pred(xx, self.b)
        k = t_factor(conf, len(self.t)-len(self.b))
        upred = k*np.sqrt(u_pred(xx, self.b, self.cov, self.syx)**2 + self.u0**2)

        with mpl.style.context(plotting.plotstyle):
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
            ax.plot(xx, fit, color='C1', label='Fit')
            ax.plot(xx, fit+upred, color='C4', ls='--', label='k={:.2f}'.format(k))
            ax.plot(xx, fit-upred, color='C4', ls='--')
            ax.plot(self.t, self.y, marker='o', ls='', label='Measurements')
            ax.set_xlabel('Time Since Calibration')
            ax.set_ylabel('Predicted Value')
            ax.legend(bbox_to_anchor=(1, 1))
        return fig


class IntervalUncertOutput(output.Output):
    ''' Report for Uncertainty Target method '''
    def __init__(self, interval, b, cov, syx, target, u0, t, y, m):
        self.interval = interval
        self.b = b
        self.cov = cov
        self.syx = syx
        self.target = target
        self.u0 = u0
        self.t = t
        self.y = y
        self.m = m

    def report(self, **kwargs):
        ''' Print the interval and fit parameters '''
        rpt = report.Report(**kwargs)
        if self.interval:
            rpt.hdr('Interval: {:.2f}\n\n'.format(self.interval), level=3)
        else:
            rpt.hdr('Interval: N/A', level=3)
        if self.t is not None:
            fig = plt.figure()
            self.plot(fig)
            rpt.plot(fig)
            plt.close()  # Prevent showing duplicate plot in Jupyter
        return rpt

    def plot(self, fig=None, **kwargs):
        ''' Plot the interval, fit line, limits, etc. '''
        xx = np.linspace(0, max(self.interval, self.t.max()))
        fit = y_pred(xx, self.b)
        upred = np.sqrt(u_pred(xx, self.b, self.cov, self.syx)**2 + self.u0**2)

        with mpl.style.context(plotting.plotstyle):
            if fig is None:
                fig = plt.figure()
            fig.clf()
            ax = fig.add_subplot(1, 1, 1)
            ax.plot(xx, fit, color='C1', label='Fit')
            ax.plot(xx, fit+upred, color='C4', ls='--', label='k=1')
            ax.plot(xx, fit-upred, color='C4', ls='--')
            ax.plot(xx, fit+self.target, color='C2', label='Target')
            ax.plot(xx, fit-self.target, color='C2')
            ax.axvline(self.interval, color='C3', label='Interval')
            ax.set_xlabel('Time Since Calibration')
            ax.set_ylabel('Predicted Value')
            ax.set_title('Uncertainty Target')
            ax.legend(bbox_to_anchor=(1, 1))
        return fig


class IntervalReliabilityOutput(output.Output):
    ''' Report for Reliability Target method '''
    def __init__(self, interval, b, cov, syx, u0, LL, UL, x, y, y0, m, k):
        self.interval = interval
        self.b = b
        self.cov = cov
        self.syx = syx
        self.u0 = u0
        self.LL = LL
        self.UL = UL
        self.x = x
        self.y = y
        self.y0 = y0
        self.m = m
        self.k = k

    def report(self, **kwargs):
        ''' Print the interval and fit parameters '''
        rpt = report.Report(**kwargs)
        if self.interval is not None:
            rpt.hdr('Interval: {:.2f}\n\n'.format(self.interval), level=3)
        else:
            rpt.hdr('Interval: N/A', level=3)
        if self.x is not None:
            fig = plt.figure()
            self.plot(fig)
            rpt.plot(fig)
            plt.close()  # Prevent showing duplicate plot in Jupyter
        return rpt

    def plot(self, fig=None, **kwargs):
        t = self.interval
        x = self.x
        y = self.y
        y0 = self.y0
        k = self.k
        UL = self.UL
        LL = self.LL

        tmax = max(t, x.max())
        xx = np.linspace(0, tmax)
        fit = y_pred(xx, self.b, y0=y0)
        upred = np.sqrt(u_pred(xx, self.b, self.cov, self.syx)**2 + self.u0**2)

        with mpl.style.context(plotting.plotstyle):
            if fig is None:
                fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
            ax.plot(x, y+y0, marker='o', ls='')
            ax.plot(xx, fit, color='C1', ls='-', label='Fit')
            if not np.isclose(k, 1, rtol=.02):
                ax.plot(xx, fit+k*upred, color='C4', ls=':', label='k={:.2f}'.format(k))
                ax.plot(xx, fit-k*upred, color='C4', ls=':')
            ax.plot(xx, fit+upred, color='C4', ls='--', label='k=1')
            ax.plot(xx, fit-upred, color='C4', ls='--')
            ax.set_title('Reliability Target')

            if LL is not None:
                ax.axhline(LL, color='C0', label='Limit')
            if UL is not None:
                ax.axhline(UL, color='C0')
            ax.axvline(t, color='C3', label='Interval')
            ax.set_xlabel('Time Since Calibration')
            ax.set_ylabel('Predicted Value')
            ax.legend(bbox_to_anchor=(1, 1))
        return fig
