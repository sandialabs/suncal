''' Functions for generating Risk Curves '''

import numpy as np

from .risk import PFA, PFR, PFR_norm, PFA_norm
from . import guardband_tur


def PFA_sweep_simple(xvar='itp', zvar='TUR', xvals=None, zvals=None,
                     GBFdflt=1, itpdflt=.95, TURdflt=4, sig0=None, tbias=0, pbias=0, risk='PFA'):
    ''' Sweep PFA vs. itp, tur, or gbf for producing risk curves in simple mode

        Args:
            xvar (string): Sweep variable for x axis - 'itp', 'tur', 'gbf', 'tbias', 'pbias'
            zvar (string): Step variable - 'itp', 'tur', 'gbf', 'tbias', 'pbias'
            xvals (array): List of sweep values for x axis
            zvals (array): List of step values for step/z axis
            GBFdlft (float): Default guardband value, if gbf is not being swept
            itpdflt (float): Default itp value, if itp is not being swept
            TURdflt (float): Default tur value, if tur is not being swept
            sig0 (float): Process standard deviation in terms of #SL, overrides itp
            tbias (float): Default test measurement bias
            pbias (float): Default process distribution bias
            risk (string): Calculate 'PFA' or 'PFR'

        Returns:
            risk (array): 2D array (shape len(xvals) x len(zvals)) of risk values
    '''
    assert xvar.lower() in ['itp', 'tur', 'gbf', 'tbias', 'pbias', 'sig0']
    assert zvar.lower() in ['itp', 'tur', 'gbf', 'tbias', 'pbias', 'sig0', 'none']
    assert risk.lower() in ['pfa', 'pfr']

    if zvar == 'none':
        zvals = [None]
        xx = np.array([xvals])
        zz = np.array([])
    else:
        xx, zz = np.meshgrid(xvals, zvals)
    riskfunc = PFR_norm if risk.lower() == 'pfr' else PFA_norm

    if xvar.lower() == 'itp':
        itp = xx
    elif zvar.lower() == 'itp':
        itp = zz
    else:
        itp = np.full(xx.shape, itpdflt)

    if xvar.lower() == 'tur':
        TUR = xx
    elif zvar.lower() == 'tur':
        TUR = zz
    else:
        TUR = np.full(xx.shape, TURdflt)

    if xvar.lower() == 'gbf':
        GBF = xx
    elif zvar.lower() == 'gbf':
        GBF = zz
    elif isinstance(GBFdflt, str):
        gbmethod = {'rds': guardband_tur.rss,
                    'rss': guardband_tur.rss,
                    'dobbert': guardband_tur.dobbert,
                    'rp10': guardband_tur.rp10,
                    'test': guardband_tur.test95}.get(GBFdflt)
        GBF = np.array([[gbmethod(t) for t in turrow] for turrow in TUR])
    else:
        GBF = np.full(xx.shape, GBFdflt)

    if xvar.lower() == 'tbias':
        tbias = xx
    elif zvar.lower() == 'tbias':
        tbias = zz
    else:
        tbias = np.full(xx.shape, tbias)

    if xvar.lower() == 'pbias':
        pbias = xx
    elif zvar.lower() == 'pbias':
        pbias = zz
    else:
        pbias = np.full(xx.shape, pbias)

    if xvar.lower() == 'sig0':
        sig0 = 1/xx
    elif zvar.lower() == 'sig0':
        sig0 = 1/zz
    else:
        sig0 = np.full(xx.shape, 1/sig0 if sig0 is not None else None)

    curves = np.empty_like(xx)
    for zidx in range(len(zvals)):
        for xidx in range(len(xvals)):
            curves[zidx, xidx] = riskfunc(itp[zidx, xidx], TUR[zidx, xidx], GBF[zidx, xidx],
                                          sig0=sig0[zidx, xidx],
                                          biastest=tbias[zidx, xidx], biasproc=pbias[zidx, xidx])
    return curves


def PFA_sweep(xvarparam, zvarparam, xvardist=None, zvardist=None, xvals=None, zvals=None,
              dist_proc=None, dist_test=None, LL=-1, UL=1, GBL=0, GBU=0, testbias=0,
              risk='PFA', approx=True):
    ''' Sweep PFA vs. any distribution parameter for producing risk curves

        Args:
            xvarparam (string): Name of distribution parameter for sweep variable, or 'gb', 'gbl', 'gbu'
            zvarparam (string):Name of distribution parameter for step variable, or 'gb', 'gbl', 'gbu'
            xvardist (Distribution): Distribution to change in x sweep
            zvardist (Distribution): Distribution to change in z step
            xvals (array): List of sweep values for x axis
            zvals (array): List of step values for step/z axis
            dist_proc (Distribution): Process distribution
            dist_test (Distribution): Test measurement distribution
            LL, UL (float): Lower and upper specification limits
            GBL, GBU (float): Lower and upper guardbands, absolute
            testbias (float): Bias in test measurement
            risk (string): Calculate 'PFA' or 'PFR'
            approx (bool): Use trapezoidal approximation for integral (faster but less accurate)

        Returns:
            risk (array): 2D array (shape len(xvals) x len(zvals)) of risk values
    '''
    riskfunc = PFR if risk == 'pfr' else PFA

    xx, zz = np.meshgrid(xvals, zvals)
    curves = np.empty_like(xx)
    for zidx, z in enumerate(zvals):
        if zvarparam.lower() == 'gb':
            GBL = z
            GBU = z
        elif zvarparam.lower() == 'gbl':
            GBL = z
        elif zvarparam.lower() == 'gbu':
            GBU = z
        elif zvarparam.lower() == 'bias':
            testbias = z
        else:
            zvardist.update_kwds(**{zvarparam: z})

        for xidx, x in enumerate(xvals):
            if xvarparam.lower() == 'gb':
                GBL = x
                GBU = x
            elif xvarparam.lower() == 'gbl':
                GBL = x
            elif xvarparam.lower() == 'gbu':
                GBU = x
            elif xvarparam.lower() == 'bias':
                testbias = x
            else:
                xvardist.update_kwds(**{xvarparam: x})

            curves[zidx, xidx] = riskfunc(dist_proc, dist_test, LL, UL, GBL, GBU, testbias)
    return curves
