import numpy as np

from ..uncertarray import Array, EventUncert
from . import extrema
from .threshold import first_over_idx, first_under_idx, interp_crossing, threshold_crossing_uncertainty


def rise_time(wf: Array, y10: float = None, y90: float = None) -> float:
    ''' Rise time calculation '''
    if y10 is None and y90 is None:
        pkpk = wf.y.max() - wf.y.min()
        delta = pkpk / 10
        y10 = wf.y.min() + delta
        y90 = wf.y.max() - delta

    starti = first_over_idx(wf, y10)
    tstart = interp_crossing(wf, y10)
    tend = interp_crossing(wf, y90, starti=starti)
    return tend-tstart


def fall_time(wf: Array, y10: float = None, y90: float = None) -> float:
    ''' Fall time calculation '''
    if y10 is None and y90 is None:
        pkpk = wf.y.max() - wf.y.min()
        delta = pkpk / 10
        y10 = wf.y.min() + delta
        y90 = wf.y.max() - delta

    starti = first_under_idx(wf, y10)
    tstart = interp_crossing(wf, y10, direction='falling')
    tend = interp_crossing(wf, y90, starti=starti, direction='falling')
    return tend-tstart


def u_rise_time(wf: Array, y10: float = None, y90: float = None) -> EventUncert:
    ''' Uncertainty in rise time from v10 to v90. '''
    if y10 is None and y90 is None:
        wmin = extrema.minimum(wf)
        wmax = extrema.maximum(wf)
        pkpk = extrema.peak_peak(wf)
        y10 = wmin.nominal + pkpk.nominal / 10
        uy10 = np.sqrt(wmin.uncert**2 + (pkpk.uncert/10)**2)
        y90 = wmax.nominal - pkpk.nominal / 10
        uy90 = np.sqrt(wmax.uncert**2 + (pkpk.uncert/10)**2)
    else:
        wmax = wmin = None
        uy10 = 0
        uy90 = 0

    starti = first_over_idx(wf, y10)
    start = threshold_crossing_uncertainty(wf, y10, uy10, direction='rising')
    end = threshold_crossing_uncertainty(wf, y90, uy90, direction='rising', starti=starti)

    rise = end.nominal - start.nominal
    uncert = np.sqrt(start.uncert**2 + end.uncert**2)
    components = {
        'start': start,
        'end': end,
        'max': EventUncert(y90, uy90, y90-uy90*2, y90+uy90*2),
        'min': EventUncert(y10, uy10, y10-uy10*2, y10+uy10*2),
    }
    return EventUncert(
        rise,
        uncert,
        rise-uncert*2,
        rise+uncert*2,
        components=components)


def u_fall_time(wf: Array, y10: float = None, y90: float = None) -> EventUncert:
    ''' Uncertainty in fall time from v90 to v10. '''
    if y10 is None and y90 is None:
        wmin = extrema.minimum(wf)
        wmax = extrema.maximum(wf)
        pkpk = extrema.peak_peak(wf)
        y10 = wmin.nominal + pkpk.nominal / 10
        uy10 = np.sqrt(wmin.uncert**2 + (pkpk.uncert/10)**2)
        y90 = wmax.nominal - pkpk.nominal / 10
        uy90 = np.sqrt(wmax.uncert**2 + (pkpk.uncert/10)**2)
    else:
        wmax = wmin = None
        uy10 = 0
        uy90 = 0

    starti = first_under_idx(wf, y90)
    start = threshold_crossing_uncertainty(wf, y90, uy90, direction='falling')
    end = threshold_crossing_uncertainty(wf, y10, uy10, direction='falling', starti=starti)

    rise = end.nominal - start.nominal
    uncert = np.sqrt(start.uncert**2 + end.uncert**2)
    components = {
        'end': end,
        'start': start,
        'max': EventUncert(y90, uy90, y90-uy90*2, y90+uy90*2),
        'min': EventUncert(y10, uy10, y10-uy10*2, y10+uy10*2),
    }

    return EventUncert(
        rise,
        uncert,
        rise-uncert*2,
        rise+uncert*2,
        components=components)


def pulse_width(wf: Array) -> float:
    ''' Calculate pulse width as Full-Width-Half-Max '''
    h = (wf.y.min() + wf.y.max())/2
    starti = first_over_idx(wf, h)
    tstart = interp_crossing(wf, h)
    tend = interp_crossing(wf, h, 'falling', starti+1)
    return tend-tstart


def u_pulse_width(wf: Array) -> EventUncert:
    ''' Calculate pulse width and uncertainty '''
    mx = extrema.maximum(wf)
    mn = extrema.minimum(wf)
    h = (mx.nominal + mn.nominal) / 2

    time = mx.components.get('time').nominal
    try:
        timeidx = np.where(wf.x > time)[0][0]
    except IndexError:
        timeidx = first_over_idx(wf, h) + 1

    start = threshold_crossing_uncertainty(wf, h, direction='rising')
    end = threshold_crossing_uncertainty(wf, h, direction='falling', starti=timeidx)
    width = end.nominal - start.nominal
    uwidth = np.sqrt(start.uncert**2 + end.uncert**2)

    components = {
        'max': mx,
        'min': mn,
        'start': start,
        'end': end,
        'h': h,
    }
    return EventUncert(width, uwidth, width-uwidth*2, width+uwidth*2,
                       components=components)
