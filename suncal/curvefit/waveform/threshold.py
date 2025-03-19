''' Threshold uncertainty calculations '''
from typing import Literal
import numpy as np
from scipy import stats

from ..uncertarray import Array, EventUncert


CrossingDirection = Literal['rising', 'falling']


def _interp(ta: float, tb: float, ya: float, yb: float, threshold: float) -> float:
    ''' Interpolate between two points '''
    return ta + (threshold - ya) * (tb-ta)/(yb-ya)


def first_over_idx(wf: Array, threshold: float, starti: int = 0) -> int:
    ''' Get index of first point over the threshold '''
    try:
        return np.where(wf.y[starti:] > threshold)[0][0] + starti
    except IndexError:
        return -1


def first_under_idx(wf: Array, threshold: float, starti: int = 0) -> int:
    ''' Get index of first point under the threshold '''
    try:
        return np.where(wf.y[starti:] < threshold)[0][0] + starti
    except IndexError:
        return -1


def interp_crossing(
        wf: Array,
        threshold: float,
        direction: CrossingDirection = 'rising',
        starti: int = 0) -> float:
    ''' Interpolated crossing time '''
    if direction == 'rising':
        idx = first_over_idx(wf, threshold, starti)
    else:
        idx = first_under_idx(wf, threshold, starti)

    return _interp(wf.x[idx-1], wf.x[idx], wf.y[idx-1], wf.y[idx], threshold)


def threshold_crossing_uncertainty(
            wf: Array,
            threshold: float,
            uthreshold: float | None = None,
            direction: CrossingDirection = 'rising',
            starti: int = 0
            ) -> Array:
    ''' Uncertainty in interpolated point over threshold '''
    # Slice the Waveform
    wf = Array(
        wf.x[starti:],
        wf.y[starti:],
        wf.ux[starti:],
        wf.uy[starti:]
    )

    if all(wf.uy) == 0 and all(wf.ux) == 0:
        expect = interp_crossing(wf, threshold, direction)
        return EventUncert(expect, 0, expect, expect)

    # Probability that each data point is above/below the threshold
    prob = np.zeros_like(wf.y)
    if uthreshold is None:
        for i, (yval, yunc) in enumerate(zip(wf.y, wf.uy)):
            prob[i] = stats.norm.cdf(threshold, loc=yval, scale=yunc)

    else:
        # thresh has Uncertainty. Prob(Y - H > 0)
        for i, (yval, yunc) in enumerate(zip(wf.y, wf.uy)):
            mu = yval - threshold
            std = np.sqrt(yunc**2 + uthreshold**2)
            prob[i] = stats.norm.cdf(0, loc=mu, scale=std)

    if direction == 'rising':
        prob = 1-prob

    # Calculate probability that each point was FIRST over the threshold
    prob_first = np.zeros_like(wf.y)
    for i, yval in enumerate(wf.y):
        prob_first[i] = prob[i] * np.prod(1-prob[:i])

    # This fails if there are no crossings or crossing is too close to end of Waveform
    # assert np.isclose(sum(prob_first), 1)

    # Find indices where there is some probability there is a crossing.
    # Only calculate remaining steps on these points (CDF can return
    # NAN if too small)
    index = np.where(prob_first > 1E-6)[0]

    # For each pair of points, interpolate the threshold crossing time
    # See interp_wf for details
    means = np.zeros(len(wf.y))
    sigmas = np.zeros(len(wf.y))
    expect1 = np.zeros(len(wf.y))
    expect2 = np.zeros(len(wf.y))
    for i in index:
        means[i], sigmas[i], expect1[i], expect2[i] = _interp_wf(wf, i, threshold, uthreshold, direction=direction)

    # Form a mixture distribution using means/sigmas of the interpolated points,
    # weighted by prob_first
    expectation = sum(prob_first[index] * means[index])
    variance = sum(prob_first[index] * (sigmas[index]**2 + means[index]**2)) - expectation**2
    stdev = np.sqrt(variance)

    # Calculate the CDF of the mixture distribution over the range of interest
    cdf = 0
    xx = np.linspace(expectation - stdev * 6, expectation + stdev * 6, 500)
    for i in index:
        cdf += prob_first[i] * stats.norm.cdf(xx, means[i], sigmas[i])

    # Find 95% Uncertainty region from 2.5% and 97.5% CDF
    try:
        t1 = xx[np.where(cdf > 0.025)[0][0]]
    except IndexError:
        t1 = np.nan
    try:
        t2 = xx[np.where(cdf > .975)[0][0]]
    except IndexError:
        t2 = np.nan

    return EventUncert(expectation, stdev, t1, t2)


def _interp_wf(
        wf: Array,
        idx: int,
        threshold: float,
        uthreshold: float | None = None,
        direction: CrossingDirection = 'rising') -> tuple[float, float, float, float]:
    ''' Interpolate time point of threshold crossing,
        returning time value and uncertainty. Uses truncated normal
        distribution to generate expectation and variance
        of each point given the point before is below and given the
        point after is above the threshold.

        t = t1 + (y - y1) * (t2-t1)/(y2-y1)

        Returns:
            time: Interpolated time value
            utime: Uncertainty in interpolated value
            ya, yb: Expectation value of the two points
    '''
    if uthreshold is None:
        uthreshold = 0

    # Extract the pair of points
    tb = wf.x[idx]
    utb = wf.ux[idx]
    uyb = wf.uy[idx]
    ta = wf.x[idx-1]
    uta = wf.ux[idx-1]
    uya = wf.uy[idx-1]

    ya_1 = wf.y[idx-1]  # Uncondional expectation of y
    yb_1 = wf.y[idx]
    if direction == 'falling':
        ya_1, yb_1 = yb_1, ya_1
        uya, uyb = uyb, uya
        ta, tb, uta, utb = tb, ta, utb, uta

    # Find expectation and variance of ya given ya < threshold
    # See https://en.wikipedia.org/wiki/Mills_ratio#Inverse_Mills_ratio
    # and https://en.wikipedia.org/wiki/Truncated_normal_distribution
    alpha = (threshold - ya_1) / uya
    lambd = stats.norm.pdf(alpha) / stats.norm.cdf(alpha)
    if np.isfinite(lambd):
        ya = ya_1 - uya * lambd
        uya = uya * np.sqrt(1 - alpha*lambd - lambd**2)
    else:  # lambd -> 0
        ya = ya_1

    # Expectation of yb given yb > threshold
    alpha = (threshold - yb_1) / uyb
    lambd = stats.norm.pdf(alpha) / (1 - stats.norm.cdf(alpha))
    if np.isfinite(lambd):
        yb = yb_1 + uyb * lambd
        uyb = uyb * np.sqrt(1 + alpha*lambd - lambd**2)
    else:  # lambd -> 0
        yb = yb_1

    # Interpolate between expectations
    tinterp = ta + (threshold - ya) * (tb-ta)/(yb-ya)

    # Apply GUM fomrula on tinterp to get uncertainty
    utinterp = np.sqrt(
        uthreshold**2 * ((tb-ta)/(yb-ya))**2 +
        uta**2 * (1 - (threshold-ya)/(yb-ya))**2 +
        utb**2 * ((threshold-ya) / (tb-ya))**2 +
        uya**2 * (((threshold-ya)*(tb-ta)/(yb-ya)**2) - (tb-ta)/(yb-ya))**2 +
        uyb**2 * ((threshold-ya)*(tb-ta)/(yb-ya)**2)**2
    )
    return tinterp, utinterp, ya, yb
