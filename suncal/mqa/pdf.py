''' Probability density functions for MQA '''
from typing import Sequence, Optional
from dataclasses import dataclass
from scipy import stats
from scipy.integrate import simpson, cumulative_simpson
from scipy.optimize import root_scalar
from scipy.signal import convolve
import numpy as np

from ..common import distributions
from ..common.limit import Limit


@dataclass
class PdfStats:
    ''' Statistics on the distribution. These are calculated
        on-demand but cached in this dataclass.
    '''
    mean: Optional[float] = None
    median: Optional[float] = None
    variance: Optional[float] = None
    stdev: Optional[float] = None
    N: Optional[int] = None
    domain: Optional[tuple[float, float]] = None


class Pdf:
    ''' Discrete representation of a probability density (mass) function,
        centered about zero
    '''
    NUM = 1001

    def __init__(self, x: Sequence[float], y: Sequence[float]):
        self._x = x
        self._y = y
        self._stats = PdfStats()
        self.setup = {}  # Parameters to save regarding how Pdf was created

    def __add__(self, other):
        if hasattr(other, '_x'):  # isinstance(other, Pdf)
            x, y1, y2 = self._align(other)
            return Pdf(x, y1 + y2)
        else:  # float/int/etc.
            return Pdf(self._x, self._y + other)

    def __sub__(self, other):
        return Pdf(self._x, self._y - other)

    def __rsub__(self, other):
        return Pdf(self._x, other - self._y)

    def __mul__(self, other):
        if hasattr(other, '_x'):  # isinstance(other, Pdf)
            x, y1, y2 = self._align(other)
            return Pdf(x, y1*y2)
        else:  # float/int/etc.
            return Pdf(self._x, self._y * other)

    def __truediv__(self, other):
        if hasattr(other, '_x'):  # isinstance(other, Pdf)
            x, y1, y2 = self._align(other)
            return Pdf(x, y1 / y2)
        else:  # float/int/etc.
            return Pdf(self._x, self._y / other)

    @property
    def N(self):
        ''' Number of points in discrete representation of pdf '''
        if self._stats.N is None:
            self._stats.N = len(self._x)
        return self._stats.N

    @property
    def domain(self) -> tuple[float, float]:
        ''' Domain of the pdf '''
        if self._stats.domain is None:
            self._stats.domain = self._x[0], self._x[-1]
        return self._stats.domain

    @property
    def mean(self) -> float:
        ''' Mean of the pdf '''
        if self._stats.mean is None:
            self._stats.mean = simpson(self._x * self._y, x=self._x)
        return self._stats.mean

    @property
    def variance(self) -> float:
        ''' Calculate variance of the pdf '''
        if self._stats.variance is None:
            mu = self.mean
            self._stats.variance = simpson(self._y*(self._x-mu)**2, x=self._x)
        return self._stats.variance

    @property
    def std(self) -> float:
        ''' Standard deviation of the pdf '''
        if self._stats.stdev is None:
            self._stats.stdev = np.sqrt(self.variance)
        return self._stats.stdev

    @property
    def median(self) -> float:
        ''' Calculate median '''
        if self._stats.median is None:
            cs = cumulative_simpson(self._y, x=self._x, initial=0)
            mx = cs[-1] / 2  # should be 0.5 for true PDF
            self._stats.median = np.interp(mx, cs, self._x)
        return self._stats.median

    def _align(self, other: 'Pdf') -> tuple[float, 'Pdf', 'Pdf']:
        ''' Align the two Pdfs to use the same x values '''
        domain1 = self.domain
        domain2 = other.domain
        domain = (min(domain1[0], domain2[0]),
                  max(domain1[1], domain2[1]))
        N = max(self.N, other.N)
        x = np.linspace(*domain, N)
        return x, self.pdf(x), other.pdf(x)

    def pdf(self, x):
        ''' Get PDF evaluated at x '''
        return np.interp(x, self._x, self._y)

    def cdf(self, x):
        ''' Get CDF evaluated at x '''
        return self.integrate(-np.inf, x)

    def itp(self, limit: Limit) -> float:
        ''' Get in-tolerance probability  '''
        return self.integrate(limit.flow, limit.fhigh)

    def given_y(self, y: float) -> 'Pdf':
        ''' Shift the nominal to y and return a new Pdf instance '''
        newx = self._x - self.median + y
        return Pdf(newx, self._y)

    def integrate(self, a: float = -np.inf, b: float = np.inf) -> float:
        ''' Integrate (numerically) the Pdf from a to b '''
        if (np.isfinite(a) and np.isfinite(b)):
            xx = np.linspace(a, b, self.N)
        elif np.isfinite(a):
            xx = np.linspace(a, self.domain[1], self.N)
        elif np.isfinite(b):
            xx = np.linspace(self.domain[0], b)
        else:
            xx = np.linspace(*self.domain, self.N)
        return simpson(self.pdf(xx), x=xx)

    def integrate_fgiveny(self, a: float, b: float) -> 'Pdf':
        ''' Integrate (numerically) f(x|y) dy from a to b, returning a Pdf '''
        # Same as convolving a step function between a and b
        width = b - a
        domain = (min(self.domain[0], a-width/2) if np.isfinite(a) else self.domain[0],
                  max(self.domain[1], b+width/2) if np.isfinite(b) else self.domain[1])
        xx = np.linspace(*domain, self.N)
        window = np.zeros_like(xx)
        window[np.where((xx >= a) & (xx <= b))] = 1
        return self.convolve(Pdf(xx, window))

    def convolve(self, f2: 'Pdf') -> 'Pdf':
        ''' Convolve this Pdf with f2(x|y), integrating f1(x)*f2(x|y) dx
            over inf, returning f(y)
        '''
        domain1 = self.domain
        domain2 = f2.domain
        domain = (min(domain1[0], domain2[0]),
                  max(domain1[1], domain2[1]))

        yy = np.linspace(*domain, num=self.NUM)
        dy = yy[1] - yy[0]
        pdf1 = self.pdf(yy) * dy
        pdf2 = f2.pdf(yy) * dy
        out = convolve(pdf1, pdf2, 'same') / dy
        return Pdf(yy, out)

    # Could also build the dists from B.1 of RP19 directly using classmethods.
    @classmethod
    def from_stdev(cls, nominal: float = 0, stdev: float = 1):
        ''' Create Pdf from nominal and standard deviation '''
        dist = stats.norm(loc=nominal, scale=stdev)
        new = cls.from_dist(dist)
        new.setup = {
            'from': 'stdev',
            'nominal': nominal,
            'stdev': stdev
        }
        return new

    @classmethod
    def from_dist(cls, dist: distributions.Distribution):
        ''' Set up distribution from a scipy rv_continuous '''
        rng = dist.interval(1-1E-12)
        x = np.linspace(*rng, cls.NUM)
        y = dist.pdf(x)
        pdf = cls(x, y)
        try:
            shape = dist.name
        except AttributeError:  # dist is a scipy.rv_continuous
            shape = dist.dist.name
        pdf.setup = {'from': 'scipy',
                     'shape': shape,
                     'args': list(dist.args),
                     'kwds': dist.kwds}
        return pdf

    def to_dist(self) -> distributions.Distribution:
        ''' Create a scipy distribution from the Pdf. Only works if
            setup "from" is "scipy", "stdev", "limit", "itp".
        '''
        mode = self.setup.get('from')
        if not mode:
            raise ValueError('Cannot convert this Pdf to scipy distribution')

        if mode == 'scipy':
            kwds = self.setup.get('kwds', {})
            name = self.setup.get('shape', 'norm')
            args = self.setup.get('args', [])
            assert len(args) == 0  # Shouldn't happen from GUI
            return distributions.get_distribution(name, **kwds)

        elif mode == 'stdev':
            nominal = self.setup.get('nominal', 0)
            stdev = self.setup.get('stdev', 1)
        elif mode in ['limit', 'itp']:
            nominal = self.setup.get('nominal', 0)
            itp = self.setup.get('itp', .95)
            tol = self.setup.get('tol', 1)
            stdev = tol / stats.norm.ppf((1+itp)/2)
        else:
            raise ValueError(f'Unknown Pdf mode {mode}')

        return distributions.get_distribution('normal', loc=nominal, std=stdev)

    @classmethod
    def from_itp(cls, nom: float = 0, itp: float = 0.95, tolerance: Limit = Limit()):
        ''' Set up distribution from in-tolerance probability '''
        if not tolerance.onesided and np.isclose(float(tolerance.center), nom):
            # The nice two-sided symmetric case
            pm = float(tolerance.plusminus)
            pdf = cls.from_stdev(nom, pm/stats.norm.ppf((1+itp)/2))

        else:
            # Have to bracket
            def risk(stdev):
                ''' Specific risk minus ITP (find zero of this) '''
                rl = stats.norm(nom, stdev).cdf(tolerance.flow)
                ru = 1 - stats.norm(nom, stdev).cdf(tolerance.fhigh)
                return 1 - (rl + ru) - itp

            span = max([x for x in (1, abs(nom-tolerance.flow), abs(tolerance.fhigh-nom)) if np.isfinite(x)])
            try:
                stdev = root_scalar(risk, bracket=(span/100, span*3), x0=span/2)
            except ValueError:
                pdf = cls.from_stdev(nom, np.nan)
            else:
                pdf = cls.from_stdev(nom, stdev.root)
                if not stdev.converged:
                    pdf = cls.from_stdev(nom, np.nan)

        pdf.setup = {'from': 'itp',
                     'nominal': nom,
                     'itp': itp,
                     'tol': tolerance}
        return pdf

    @classmethod
    def from_fit(cls, data: Sequence[float], shape='norm'):
        ''' Set up distribution using normal fit to sampled data '''
        dtype = getattr(stats, shape)
        dist = dtype.fit(data)
        pdf = cls.from_dist(dist)
        pdf.setup = {'from': 'fit',
                     'fitdata': data}
        return pdf

    @classmethod
    def cosine_utility(cls, degrade: Limit, fail: Limit):
        ''' Cosine utility function where xd is degraded and xf is failure points '''
        # RP-19 Eq. 4-3, adapted for single-sided tolerances

        if not degrade.onesided:
            xd_above = degrade.fhigh
            xd_below = degrade.flow
        elif np.isfinite(degrade.flow):
            xd_above = degrade.flow
            xd_below = float('-inf')
        else:
            xd_above = float('inf')
            xd_below = degrade.fhigh

        if not fail.onesided:
            xf_above = fail.fhigh
            xf_below = fail.flow
        elif np.isfinite(fail.flow):
            xf_above = fail.flow
            xf_below = float('-inf')
        else:
            xf_above = float('inf')
            xf_below = fail.fhigh

        low = min(x for x in (xd_above, xd_below, xf_above, xf_below) if np.isfinite(x))
        high = max(x for x in (xd_above, xd_below, xf_above, xf_below) if np.isfinite(x))
        span = high-low
        low, high = low-2*span, high+2*span
        xx = np.linspace(low, high, Pdf.NUM)

        yy = np.zeros_like(xx)

        above = np.where((xx > xd_above) & (xx < xf_above))
        below = np.where((xx < xd_below) & (xx > xf_below))
        inside = np.where((xx >= xd_below) & (xx <= xd_above))

        if (np.isfinite(xd_below) and np.isfinite(xf_below)):
            yy[below] = np.cos((xd_below - xx[below])*np.pi / (2*(xd_below-xf_below)))**2

        if (np.isfinite(xd_above) and np.isfinite(xf_above)):
            yy[above] = np.cos((xx[above] - xd_above)*np.pi / (2*(xf_above-xd_above)))**2

        yy[inside] = 1
        pdf = cls(xx, yy)
        pdf.setup = {'from': 'cosine',
                     'low': low,
                     'high': high}
        return pdf

    @classmethod
    def step(cls, low: float, high: float):
        ''' Step function: 1 between low and high, 0 elsewhere '''
        if not np.isfinite(low):
            delta = min(.1, abs(high/10))
            x = np.linspace(high-delta, high+delta, cls.NUM)
            y = np.zeros_like(x)
            y[(x <= high)] = 1.0
        elif not np.isfinite(high):
            delta = min(.1, abs(low/10))
            x = np.linspace(low-delta, low+delta, cls.NUM)
            y = np.zeros_like(x)
            y[(x >= low)] = 1.0
        else:
            width = high-low
            x = np.linspace(low-width, high+width, cls.NUM)
            y = np.zeros_like(x)
            y[(x >= low) & (x <= high)] = 1.0
        pdf = cls(x, y)
        pdf.setup = {'from': 'step',
                     'low': low,
                     'high': high}
        return pdf
