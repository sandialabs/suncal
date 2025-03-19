''' Equipment definition for MQA '''
from typing import Optional
from decimal import Decimal
from dataclasses import dataclass
from uuid import uuid4
import math

from ..common import unitmgr
from ..common.uparser import callf
from ..common.limit import Limit
from .pdf import Pdf


@dataclass
class Accuracy:
    ''' Accuracy statement for an M&TE Range

        Args:
            pct_reading: Percent of reading/nominal
            pct_range: Percent of range
            const: A constant uncertainty
            resolution: Resolution uncertainty (uniform distribution)
            formula: An evaluatable function of x to add to the
                uncertainty, such as (x**2/1000)

        All components are RSS'd to obtain total accuracy, at same coverage
        factor as entered spec.
    '''
    def __init__(self,
                 pct_reading: str = '0',
                 pct_range: str = '0',
                 const: str = '0',
                 resolution: str = '0',
                 formula: Optional[str] = None,
                 units: str = ''):
        self.pct_reading = Decimal(pct_reading)
        self.pct_range = Decimal(pct_range)
        self.const = Decimal(const)
        self.resolution = Decimal(resolution)
        self.formula = formula
        self.units = units

    def set_values(self,
                   pct_reading: str = '0',
                   pct_range: str = '0',
                   const: str = '0',
                   resolution: str = '0',
                   formula: Optional[str] = None,
                   units: str = ''):
        self.pct_reading = Decimal(pct_reading)
        self.pct_range = Decimal(pct_range)
        self.const = Decimal(const)
        self.resolution = Decimal(resolution)
        self.formula = formula
        self.units = units

    def evaluate(self, nominal: float, rng: float = 0) -> float:
        ''' Evaluate combined accuracy at the nominal value '''
        if self.units and not unitmgr.has_units(nominal):
            nominal = unitmgr.make_quantity(nominal, self.units)

        uncert = unitmgr.make_quantity(float(self.const), self.units)
        uncert += float(self.pct_reading/100) * nominal
        uncert += float(self.pct_range/100) * unitmgr.make_quantity(rng, self.units)
        uncert += 2 * unitmgr.make_quantity(float(self.resolution), self.units) / math.sqrt(12)
        if self.formula and self.formula.lower().strip() != 'none':
            uncert += callf(self.formula, {'x': nominal})
        return uncert

    def config(self) -> dict:
        ''' Get accuracy configuration dictionary '''
        return {
            'pct_reading': str(self.pct_reading),
            'pct_range': str(self.pct_range),
            'const': str(self.const),
            'resolution': str(self.resolution),
            'formula': self.formula,
            'units': self.units
        }

    @classmethod
    def from_config(cls, config) -> 'Accuracy':
        ''' Create Accuracy from configuration '''
        return cls(**config)


class Range:
    ''' M&TE Range Definition

        Args:
            idn: Unique range identifier (filled automatically)
            name: Name of the range
            units: Measurement units (only used to check compatibility)
            low: Lowest value measurable in the range
            high: Highest value measurable in the range
            accuracy: Accuracy statement for the range
    '''
    def __init__(self, idn: str | None = None, name: str = '', units: str = '',
                 low: float = -math.inf, high: float = 10.,
                 accuracy: Accuracy | None = None):
        self.idn = idn if idn else uuid4().hex
        self.name = name
        self.units = units
        self.low = low
        self.high = high
        self._accuracy = accuracy if accuracy else Accuracy()

    def check_range(self, nominal) -> bool:
        ''' Check whether measured value is within the Range '''
        if self.units:
            if not unitmgr.has_units(nominal):
                nominal = unitmgr.make_quantity(nominal, self.units)
            return unitmgr.make_quantity(self.low, self.units) <= nominal <= unitmgr.make_quantity(self.high, self.units)

        # Unitless compare
        return self.low <= nominal <= self.high

    def set_accuracy(self,
                     pct_reading: str = '0',
                     pct_range: str = '0',
                     const: str = '0',
                     resolution: str = '0',
                     formula: Optional[str] = None,
                     units: str = ''):
        ''' Set the accuracy specification '''
        self._accuracy.set_values(
            pct_reading, pct_range, const, resolution, formula, units=units)

    def accuracy(self, nominal: float) -> float:
        ''' Evaluate measurement uncertainty at the nominal value
            with same coverage factor as entered spec
        '''
        if self.check_range(nominal):
            return self._accuracy.evaluate(nominal, self.high)
        return float('nan')

    def config(self) -> dict:
        ''' Get configuration dictionary '''
        return {
            'idn': self.idn,
            'name': self.name,
            'units': self.units,
            'low': self.low,
            'high': self.high,
            'accuracy': self._accuracy.config() if self._accuracy else {}
        }

    @classmethod
    def from_config(cls, cfg: dict) -> 'Range':
        ''' Create MteRange from config dictionary '''
        cfg['accuracy'] = Accuracy.from_config(cfg.get('accuracy', {}))
        return cls(**cfg)


class Equipment:
    ''' Measuring and Test Equipment

        Args:
            idn: Unique equipment identifier (filled automatically)
            make: Equipment make/manufacturer
            model: Equipment model number
            serial: Equipment serial number
            ranges: List of equipment ranges
            k: Coverage factor. Leave None for sqrt(3) uniform distribution.
    '''
    def __init__(self, idn: str | None = None, make: str = '', model: str = '',
                 serial: str = '', ranges: list[Range] | None = None,
                 k: float = None,
                 ):
        self.idn = idn if idn else uuid4().hex
        self.make = make
        self.model = model
        self.serial = serial
        self.ranges = ranges if ranges else []
        self.k = k

    def __str__(self):
        if self.serial:
            return f'{self.make} {self.model} - {self.serial}'
        return f'{self.make} {self.model}'

    @property
    def coverage_factor(self):
        ''' Get coverage factor, using sqrt(3) if not defined '''
        if self.k is None:
            return math.sqrt(3)
        return self.k

    def add_range(self, rng: Range) -> None:
        ''' Add a Range to the M&TE '''
        self.ranges.append(rng)

    def auto_range(self, value: float) -> Range:
        ''' Return the smallest range that is greater than value '''
        for rng in self.ranges:
            if rng.check_range(value):
                return rng
        return None

    def accuracy(self, nominal: float) -> float:
        ''' Plus/minus value at k=1 '''
        rng = self.auto_range(nominal)
        if rng is None:
            return float('nan')
        return rng.accuracy(nominal) / self.coverage_factor

    def config(self) -> dict:
        ''' Get configuration dictionary '''
        return {
            'idn': self.idn,
            'make': self.make,
            'model': self.model,
            'serial': self.serial,
            'ranges': [r.config() for r in self.ranges],
            'k': self.k,
        }

    @classmethod
    def from_config(cls, cfg: dict):
        ''' Create M&TE from config dictionary '''
        cfg['ranges'] = [Range.from_config(c) for c in cfg.pop('ranges', [])]
        return cls(**cfg)


class EquipmentList:
    ''' List of all equipment '''
    def __init__(self, equipment: list[Equipment] = None):
        if equipment is None:
            equipment = []
        self.equipment = list(equipment)

    def locate(self, idn: str) -> Equipment:
        ''' Locate equipment by id '''
        for equip in self.equipment:
            if equip.idn == idn:
                return equip
        return None

    def locate_range(self, idn: str) -> tuple[Range, Equipment]:
        ''' Locate equipment and range by id '''
        for equip in self.equipment:
            for rng in equip.ranges:
                if rng.idn == idn:
                    return rng, equip
        return None, None

    def config(self):
        ''' Get configuration dictionary '''
        return [equip.config() for equip in self.equipment]

    @classmethod
    def from_config(cls, config: list[dict]):
        ''' Create equipment list from configuration '''
        new = cls()
        new.equipment = [Equipment.from_config(e) for e in config]
        return new


class MqaEquipmentRange:
    ''' Equipment used in a measurement '''
    def __init__(self, equipment: Equipment = None, rng: Range = None):
        self.equipment: Equipment = equipment
        self.range: Range = rng

    def __str__(self):
        try:
            name = f'{self.equipment.make} {self.equipment.model}'
        except AttributeError:
            name = 'Undefined Equipment'

        if self.range:
            name += f' [{self.range.name}]'
        return name

    def check_range(self, measured: float) -> bool:
        ''' Check whether measured value is within range '''
        if self.range:
            return self.range.check_range(measured)
        return self.equipment.auto_range(measured) is not None

    def tolerance(self, nominal: float) -> Limit:
        ''' Get the tolerance of the equipment at the nominal value '''
        if not self.range and not self.equipment:
            return None

        if self.range:
            accuracy = self.range.accuracy(nominal)
            accuracy = unitmgr.match_units(accuracy, nominal)
        elif self.equipment:
            accuracy = self.equipment.accuracy(nominal)
            accuracy = unitmgr.match_units(accuracy, nominal)

        return Limit.from_plusminus(
            unitmgr.strip_units(nominal),
            unitmgr.strip_units(accuracy))

    def pdf(self, nominal: float) -> Pdf:
        ''' Get Pdf of measurement uncertainty, centered at 0??'''
        if self.equipment is None:
            return None

        tol = self.tolerance(nominal)
        if tol is None:
            return None

        return Pdf.from_stdev(0, float(tol.plusminus) / self.equipment.coverage_factor)
