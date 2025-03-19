''' Tolerance or Limit to be verified by an MQA quantity '''
from decimal import Decimal
import math
from scipy import stats

from ..risk import specific_risk


class Limit:
    ''' Test/Calibration Specification Limit or Tolerance

        Args:
            low: Lower limit
            high: Upper limit
            nominal: Nominal value
    '''
    def __init__(self,
                 low: Decimal = Decimal('-1.0'),
                 high: Decimal = Decimal('1.0'),
                 nominal: Decimal | None = None,
                 units: str = ''):
        self.low = Decimal(low)
        self.high = Decimal(high)
        self.nominal = Decimal(nominal) if nominal is not None else None
        self.units = units

        if self.symmetric and self.nominal is not None:
            self._str = f'{self.nominal} ± {self.plusminus}'
        elif self.symmetric:
            self._str = f'{self.center} ± {self.plusminus}'
        elif not math.isfinite(self.low):
            self._str = f'< {self.high}'
        elif not math.isfinite(self.high):
            self._str = f'> {self.low}'
        else:
            self._str = f'({low}, {high})'

    @classmethod
    def from_plusminus(cls, nominal: Decimal, pm: Decimal):
        ''' Create limit from nominal and +/- value '''
        nominal = Decimal(nominal)
        pm = Decimal(pm)
        return cls(nominal-pm, nominal+pm, nominal=nominal)

    def __repr__(self):
        return f'<Limit: {str(self)}>'

    def __str__(self):
        return self._str

    def __iter__(self):
        ''' Allow unpacking the limits '''
        return iter((float(self.low), float(self.high)))

    @property
    def flow(self) -> float:
        ''' Low limit as float '''
        return float(self.low)

    @property
    def fhigh(self) -> float:
        ''' High limit as float '''
        return float(self.high)

    @property
    def center(self) -> Decimal:
        ''' Midpoint between the two limits '''
        return (self.low + self.high) / 2

    @property
    def plusminus(self) -> Decimal:
        ''' Get plus/minus value of limit '''
        if self.nominal is not None:
            return self.high - self.nominal
        return self.high - self.center

    @property
    def onesided(self) -> bool:
        ''' The tolerance is one-sided? '''
        return not math.isfinite(self.center)

    @property
    def symmetric(self) -> bool:
        ''' Is the limit symmetric about nominal? '''
        if self.onesided:
            return False
        if self.nominal is not None:
            center = self.nominal
            return math.isclose(center - self.low,
                                self.high - center)
        return False

    def probability_conformance(self, nominal: float, uncert: float, degf: float) -> float:
        ''' Probability the measured results conform with limits '''
        return 1 - self.specific_risk(nominal, uncert, degf)

    def probability_conformance_95(self, low95: float, high95: float) -> float:
        ''' Probabiliy of conformance given endpoints of 95% coverage region '''
        # Have to assume symmetric/normal centered between limits
        nominal = (low95 + high95) / 2
        plusminus = high95 - nominal
        return self.probability_conformance(nominal, plusminus, math.inf)

    def specific_risk(self, nominal: float, uncert: float, degf: float) -> float:
        ''' Probability the measured results do not conform with limits '''
        if math.isfinite(degf) and degf > 2:
            scale = uncert / math.sqrt(degf/(degf-2))  # Convert standard dev. to scipy's "scale" parameter
        else:
            scale = uncert
        scale = max(scale, 1E-99)  # Don't allow 0
        dist = stats.t(loc=nominal, scale=scale, df=degf)
        return specific_risk(dist, float(self.low), float(self.high)).total

    def config(self) -> dict:
        ''' Get configuration for a Limit '''
        return {
            'low': str(self.low),
            'high': str(self.high),
            'nominal': str(self.nominal) if self.nominal is not None else None,
            'units': str(self.units)
        }

    @classmethod
    def from_config(cls, config: dict):
        ''' Create limits from configuration '''
        if config is None:
            return None

        nom = config.get('nominal')
        return cls(
            Decimal(config.get('low', '-inf')),
            Decimal(config.get('high', 'inf')),
            Decimal(nom) if nom is not None else None,
            config.get('units', '')
        )
