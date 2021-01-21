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
