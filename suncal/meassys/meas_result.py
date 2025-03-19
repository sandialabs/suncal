''' Results of a Measurement System calculation '''
from dataclasses import dataclass, field

from ..common import reporter, ttable
from ..common.limit import Limit
from ..uncertainty.variables import RandomVariable
from .report.meassys import SystemQuantityReport, SystemReport


@reporter.reporter(SystemQuantityReport)
@dataclass
class SystemQuantityResult:
    ''' Result from one quantity in a measurement system
        May be from a SystemQuantity, SystemIndirect, or SystemCurve

        Args:
            symbol: Symbol/variable name for this quantity
            value: Average/expected value of the quantity
            uncertainty: Standard uncertainty
            units: Units of measure (string)
            degrees_freedom: Effective degrees of freedom
            tolerance: Optional tolerance for pr(conformance)
            p_conformance: Probability of conformance to the tolerance, if defined
            confidence: Level of confidence for calculating expanded uncertainty
            qty: The SystemQuantity defining the calculation
            meta: Dictionary of metadata based on the type of calculation used to determine this quantity
    '''
    symbol: str
    value: float
    uncertainty: float  # standard
    units: str
    degrees_freedom: float
    tolerance: Limit
    p_conformance: float
    qty: 'SystemQuantity|SystemIndirectQuantity|SystemCurve'
    meta: dict = field(default_factory=dict)

    def expanded(self, conf=None) -> float:
        ''' Expanded uncertainty '''
        if conf is None:
            conf = self.meta.get('confidence', .95)
        k = ttable.k_factor(conf, self.degrees_freedom)
        return self.uncertainty * k

    def randomvariable(self) -> RandomVariable:
        ''' Convert the coefficient into a RandomVariable '''
        if hasattr(self.qty, 'randomvariable'):
            rv = self.qty.randomvariable()
        else:
            rv = RandomVariable(self.value)
            rv.typeb(std=self.uncertainty, df=self.degrees_freedom)
        return rv


@reporter.reporter(SystemReport)
@dataclass
class SystemResult:
    ''' Result for a system of quantities '''
    quantities: list[SystemQuantityResult] = None
    confidence: float = .95

    def get_result(self, symbol):
        symbols = [q.symbol for q in self.quantities]
        try:
            idx = symbols.index(symbol)
        except (ValueError, IndexError):
            return None
        return self.quantities[idx]
