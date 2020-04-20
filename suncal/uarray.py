'''
Array Uncertainty Module

This module provides tools to assist with finding uncertainty
from arrays. For example, find the uncertainty of a pulse
width or uncertainty in the slope of a linear regression.

Start by defining an Array object, using x, y and ux, uy data.
Alternatively, define a DataSetSummary object using measured data,
where uncertainty values will be determined from the standard deviation
of each measurement group. The resulting Array object can be sampled,
generating random variates of the array.

To use the array in an Uncertainty Calculator instance, use an ArrayFunc
instance. The function argument to ArrayFunc must take x and y arrays
as arguments, and return a scalar value.


See examples folder example usage.
'''

import numpy as np
import scipy.stats as stat

from . import uncertainty
from . import out_uncert
from . import uparser


class Array(object):
    ''' Array with uncertainty in y and maybe x

        Parameters
        ----------
        x, y: float arrays
            X and Y data of the array

        ux, uy: float or float arrays
            uncertainty in X and Y values

        name: string
            Name for the array

        xdate: bool
            X-values can be interpreted as ordinal date value

        Attributes
        ----------
        xsamples, ysamples: float arrays
            Sampled x and y values
    '''
    def __init__(self, x, y, ux=0., uy=0., name='', xdate=False):
        self.name = name
        self.x = np.asarray(x, dtype=np.float64)
        self.y = np.asarray(y, dtype=np.float64)
        if isinstance(ux, float) or isinstance(ux, int) or len(ux) == 1:
            ux = np.ones_like(self.x) * ux
        if isinstance(uy, float) or isinstance(uy, int) or len(uy) == 1:
            uy = np.ones_like(self.y) * uy
        self.ux = np.asarray(ux, dtype=np.float64)
        self.uy = np.asarray(uy, dtype=np.float64)
        self.uy_estimate = None  # uy estimated from residuals in curve fit
        self.xsamples = None
        self.ysamples = None
        self.sampledvalues = None
        self.xdate = xdate

    def __len__(self):
        return len(self.x)

    def has_ux(self):
        ''' Does the array have x-uncertainties? '''
        return not all(self.ux == 0)

    def has_uy(self):
        ''' Does the array have y-uncertainties? '''
        return not all(self.uy == 0)

    def sample(self, samples=1000):
        ''' Generate random samples of the array '''
        if self.xsamples is None or self.ysamples is None:
            if self.uy_estimate is not None:
                uy = self.uy_estimate
            else:
                uy = self.uy

            # NOTE: Currently only normal distributions can be used here
            distfunc_y = stat.norm(loc=self.y, scale=uy)
            distfunc_x = stat.norm(loc=self.x, scale=self.ux)
            samples = int(samples)
            self.xsamples = np.empty((len(self.x), samples))
            self.ysamples = np.empty((len(self.x), samples))
            for i in range(samples):
                self.xsamples[:, i] = distfunc_x.rvs()
                self.ysamples[:, i] = distfunc_y.rvs()

    def clear(self):
        ''' Clear sampled data '''
        self.xsamples = None
        self.ysamples = None

    def clear_uyestimate(self):
        self.uy_estimate = None

    def save_file(self, fname):
        ''' Save array to file (x, y, uy, ux) '''
        arr = np.vstack((self.x, self.y))
        hdr = 'x y'
        if self.has_uy():
            arr = np.vstack((arr, self.uy))
            hdr = hdr + ' uy'
        if self.has_ux():
            arr = np.vstack((arr, self.ux))
            hdr = hdr + ' ux'
        np.savetxt(fname, arr.transpose(), header=hdr)

    def get_numpy(self):
        ''' Return numpy array of x, y, uy, ux columns '''
        arr = np.vstack((self.x, self.y, self.uy, self.ux)).transpose()
        return arr


class ArrayFunc(uncertainty.InputFunc):
    ''' Input function that operates on an array as its only variable.
        Essentially uses each element of the array as an input for GUM calculation.

        Parameters
        ----------
        function: callable
            Function that takes x and y arrays as the only arguments and
            returns a scalar
        arr: Array
            The array to operate on
        name: string
            Name for the function
        desc: string
            Description of the function
    '''
    def __init__(self, function=None, arr=None, name=None, units=None, desc='', seed=None):
        if isinstance(arr, Array):
            arr = [arr]
        super().__init__(function, variables=arr, name=name, desc=desc)
        self.ftype = 'array'
        self.seed = seed
        self._degf = np.inf
        self.units = uparser.parse_unit(units)
        self.outputs = {}

    @property
    def nom(self):
        ''' Get nominal value of array '''
        return self.mean() * self.units

    def clear(self):
        ''' Clear sampled values '''
        super().clear()
        self.outputs = {}

    def calculate(self, **kwargs):
        ''' Calculate GUM and Monte-Carlo methods. kwargs passed to calc_MC and calc_GUM '''
        samples = kwargs.get('samples', 5000)
        mc = self.calc_MC(samples=samples)
        gum = self.calc_GUM()
        return out_uncert.FuncOutput([gum, mc], self)

    def calc_MC(self, **kwargs):
        ''' Calculate Monte Carlo method.

            Keyword Arguments
            -----------------
            samples: int
                Number of Monte Carlo samples
            sensitivity: boolean
                Calculate sensitivity (requires additional Monte Carlo runs for each input)

            Returns
            -------
            BaseOutputMC object
        '''
        if self.seed is not None:
            np.random.seed(self.seed)
        samples = kwargs.get('samples', 5000)
        sens = kwargs.get('sensitivity', False)
        return super().calc_MC(samples=samples, sensitivity=sens)

    def calc_GUM(self, correlation=None):
        ''' Calculate uncertainty using GUM method.

            Parameters
            ----------
            correlation: array
                Correlation matrix to apply (CURRENTLY IGNORED)

            Returns
            -------
            BaseOutput object
        '''
        basevar = self.variables[0]  # The array
        xmeans = basevar.x.copy()
        ymeans = basevar.y.copy()
        mean, cov, grad = _GUM(self.function, xmeans, ymeans, basevar.ux, basevar.uy)
        params = {'mean': mean,
                  'uncert': np.sqrt(np.diag(cov))[0],
                  'inputs': [basevar],
                  'function': str(self),
                  'degf': self._degf}
        self.outputs['gum'] = out_uncert.create_output('gum', **params)
        return self.outputs['gum']

    def sample(self, samples=10000):
        ''' Generate and return Monte Carlo samples '''
        if self.sampledvalues is not None:
            return self.sampledvalues

        for v in self.variables:
            v.clear()
            v.sample(samples)   # Generate MC samples of array points

        self.sampledvalues = np.zeros(samples)
        for i in range(samples):
            self.sampledvalues[i] = self.function(self.variables[0].xsamples[:, i],
                                                  self.variables[0].ysamples[:, i])
        self.sampledvalues = self.sampledvalues * self.units
        return self.sampledvalues


class ArrayThresh(ArrayFunc):
    ''' Input function for finding threshold crossing

        Parameters
        ----------
        arr: Array object
            The array to operate on
        name: string
            Name of the function
        desc: string
            Description of the function
        thresh: float
            Y-value of threshold crossing
        edge: string
            Which edge to detect. 'first', 'last', 'median'
    '''
    def __init__(self, arr=None, name=None, desc='', thresh=0, edge='first', units=None, seed=None):
        self.thresh = thresh
        self.edge = edge
        super().__init__(function=self._function, arr=arr, name=name, desc=desc, units=units, seed=seed)

    def _function(self, x, y):
        ''' Wrap getedge function. '''
        return getedge(x, y, thresh=self.thresh, edge=self.edge)

    def calculate(self, **kwargs):
        ''' Calculate using different methods

            Keyword Arguments
            -----------------
            gum: bool
                Calculate GUM method
            mc: bool
                Calculate Monte Carlo method
            lsq: bool
                Calculate analytical Least Squares method
            samples: int
                Number of Monte Carlo samples

            Returns
            -------
            FuncOutput object
        '''
        samples = kwargs.get('samples', 5000)
        lsqout = self.calc_LSQ()
        mcout = self.calc_MC(samples=samples)
        gumout = self.calc_GUM()
        self.out = out_uncert.FuncOutput([lsqout, mcout, gumout], self)
        return self.out

    def mean(self):
        ''' Get mean value '''
        if 'lsq' not in self.outputs:
            self.calc_LSQ()
        return self.outputs['lsq'].mean * self.units

    def stdunc(self):
        ''' Get standard uncertainty value '''
        if 'lsq' not in self.outputs:
            self.calc_LSQ()
        return self.outputs['lsq'].uncert * self.units

    def calc_LSQ(self):
        ''' Calculate analytical solution using "rectangles of uncertainty" method.

            Returns
            -------
            BaseOutput object
        '''
        arr = self.variables[0]
        idx = _getedge_idx(arr.x, arr.y, thresh=self.thresh, edge=self.edge)
        xval = getedge(arr.x, arr.y, thresh=self.thresh, edge=self.edge)

        # Interpolate uncertainty between u.uy[idx] and u.uy[idx+1]
        if np.ndim(arr.uy) == 0:
            yunc = arr.uy
        elif arr.uy[idx] == arr.uy[idx+1]:
            yunc = arr.uy[idx]
        else:
            yunc = np.interp(xval, arr.x[idx:idx+2], arr.uy[idx:idx+2])

        xhi = getedge(arr.x, arr.y, thresh=self.thresh+yunc, edge=self.edge)
        xlo = getedge(arr.x, arr.y, thresh=self.thresh-yunc, edge=self.edge)
        if xhi and xlo:
            ux = abs(xhi-xlo)/2  # half the total spread
        elif xhi:
            ux = abs(xhi-xval)  # Only spread on side with intersection
        else:
            ux = abs(xlo-xval)

        params = {'mean': xval, 'uncert': ux}
        self.outputs['lsq'] = out_uncert.create_output('lsq', **params)
        return self.outputs['lsq']


def _GUM(func, xmeans, ymeans, ux, uy):
    ''' Calculate GUM uncertainty on the function. Function takes arrays
        of x, y values as arguments.

        Parameters
        ----------
        func: callable
            Function to operate on. Arguments must be x and y arrays
        xmeans, ymeans: arrays
            Mean values of array data
        ux, uy: arrays
            Uncertainties in x and y

        Returns
        -------
        mean: array
            Mean value(s) of function evaluated at xmeans, ymeans
        cov: array
            Covariance matrix of function parameters. Uncertainty in
            parameters is sqrt(diag(cov)).
        grad: array, 2D
            Gradient of each function return value along [xarray, yarray].
            Each point along each array is treated as an input variable.
            Ref: GUM-Supplement 2, section 6.2
    '''
    xmeans = xmeans.astype('float')  # int arrays don't work
    ymeans = ymeans.astype('float')
    xmeansorg = xmeans.copy()
    ymeansorg = ymeans.copy()
    mean = func(xmeans, ymeans)

    nvars = len(np.atleast_1d(mean))  # Number of output parameters from func
    lenx = len(xmeans)
    grad = np.zeros((nvars, lenx*2))
    ui = np.zeros(lenx*2)

    for i, x in enumerate(xmeans):
        if ux[i] != 0:
            dx = np.float64(ux[i]) / 1E6
            xmeans[i] += dx
            val1 = func(xmeans, ymeans)
            xmeans[i] -= 2*dx
            val2 = func(xmeans, ymeans)
            xmeans[i] = xmeansorg[i]  # Restore value
            grad[:, i] = ((val1-val2)/(2*dx))
            ui[i] = ux[i]
    for i, y in enumerate(ymeans):
        if uy[i] != 0:
            dy = np.float64(uy[i]) / 1E6
            ymeans[i] += dy
            val1 = func(xmeans, ymeans)
            ymeans[i] -= 2*dy
            val2 = func(xmeans, ymeans)
            ymeans[i] = ymeansorg[i]  # Restore value
            grad[:, i+lenx] = (val1-val2)/(2*dy)
            ui[i+lenx] = uy[i]
    cov = grad @ np.diag(ui*ui) @ grad.T
    return mean, cov, grad


def _getedge_idx(x, y, thresh, edge='first'):
    ''' Get index of threshold crossing.

        Parameters
        ----------
        x, y: float arrays
            X and Y data
        thresh: float
            Y-value threshold
        edge: string
            Which edge to return if multiple crossings. One of
            'first', 'last', 'median', or 'all'

        Returns
        -------
        idx: int
            Index of point just before threshold crossing
    '''
    x = np.asarray(x)
    y = np.asarray(y)
    idx  = ((y > thresh)[1:] & (y < thresh)[:-1]).nonzero()[0]
    idx2 = ((y < thresh)[1:] & (y > thresh)[:-1]).nonzero()[0]
    idx = np.concatenate((idx, idx2))
    idx.sort()

    if edge == 'first':
        return idx[0]
    elif edge == 'last':
        return idx[-1]
    elif edge == 'median':
        return int(np.median(idx))
    elif edge == 'all':
        return idx


def getedge(x, y, thresh, edge='first'):
    ''' Get value of threshold crossing, interpolated between points

        Parameters
        ----------
        x, y: float arrays
            X and Y data
        thresh: float
            Y-value threshold
        edge: string
            Which edge to return if multiple crossings. One of
            'first', 'last', 'median', or 'all'

        Returns
        -------
        x1: float
            Value of x where y==thresh.
    '''
    idx = _getedge_idx(x, y, thresh, edge)
    x1, x2 = x[idx], x[idx+1]
    y1, y2 = y[idx], y[idx+1]
    return x1 + (thresh - y1) / ((y2-y1) / (x2-x1))
