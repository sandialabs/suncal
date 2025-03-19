''' Uncertainty functions for max, min, and peak-peak '''
import numpy as np
from scipy import stats

from ..uncertarray import Array, EventUncert


def maximum(wf: Array) -> EventUncert:
    ''' Uncertainty in maximum value of waveform.
        Also returns time of maximum value as a component
    '''
    ymax = wf.y.max()
    if all(wf.uy) == 0.:
        idx = wf.y.argmax()
        tunc = EventUncert(wf.x[idx], 0, wf.x[idx], wf.x[idx])
        yunc = EventUncert(ymax, 0, ymax, ymax, components={'time': tunc})
        return yunc

    mn = (wf.y - 6*wf.uy).min()
    mx = (wf.y + 6*wf.uy).max()
    xx = np.linspace(mn, mx, 1000)
    yy = np.ones_like(xx)

    for i in range(len(wf.x)):
        yy *= stats.norm.cdf(xx, loc=wf.y[i], scale=wf.uy[i])

    try:
        ylow = xx[np.where(yy > .025)[0][0]]
    except IndexError:
        ylow = xx[0]

    try:
        yhigh = xx[np.where(yy > .975)[0][0]]
    except IndexError:
        yhigh = xx[-1]

    pt = np.zeros_like(wf.x)
    for i in range(len(wf.x)):
        pt[i] = 1 - stats.norm.cdf(ymax, loc=wf.y[i], scale=wf.uy[i])
    pt = np.cumsum(pt / sum(pt))

    try:
        tlow = wf.x[np.where(pt > .025)[0][0]]
    except IndexError:
        tlow = wf.x[0]

    try:
        thigh = wf.x[np.where(pt > .975)[0][0]]
    except IndexError:
        thigh = wf.x[-1]

    yexpected = (ylow+yhigh)/2
    texpected = (tlow+thigh)/2

    tunc = EventUncert(texpected, (thigh-texpected)/2, tlow, thigh)
    yunc = EventUncert(yexpected, (yhigh-yexpected)/2, ylow, yhigh,
                       components={'time': tunc})
    return yunc


def minimum(wf: Array) -> EventUncert:
    ''' Uncertainty in minimum value of waveform
        Also returns time of minimum value as a component
    '''
    wf = Array(wf.x, -wf.y, wf.ux, wf.uy)
    out = maximum(wf)
    out.nominal = -out.nominal
    out.low = -out.low
    out.high = -out.high
    return out


def peak_peak(wf: Array) -> EventUncert:
    ''' Uncertainty in peak-to-peak '''
    mx = maximum(wf)
    mn = minimum(wf)
    pkpk = mx.nominal - mn.nominal
    u_pkpk = np.sqrt(mx.uncert**2 + mn.uncert**2)
    components = {
        'max': mx,
        'min': mn,
    }
    return EventUncert(pkpk, u_pkpk, pkpk-u_pkpk*2, pkpk+u_pkpk*2,
                       components=components)
