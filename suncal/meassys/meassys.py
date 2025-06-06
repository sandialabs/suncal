''' Specific Meausurement Result of a Quantity '''
import string
from contextlib import suppress
import numpy as np

from ..common import unitmgr
from ..common.limit import Limit
from ..uncertainty import Model
from ..uncertainty.variables import Typeb, RandomVariable

from .curve import SystemCurve
from .meas_result import SystemQuantityResult, SystemResult


class SystemQuantity:
    ''' One direct quantity in a measurement system '''
    def __init__(self, symbol='a'):
        super().__init__()
        self.name = ''
        self.symbol = symbol
        self.units = None
        self.description = ''
        self.testpoint: float = 0
        self.autocorrelation: bool = True
        self.num_newmeas: int = None

        self.typebs: list[Typeb] = []  # Other Type Bs
        self.typea: np.ndarray | None = None  # Type A R&R data
        self.typea_description: str = ''
        self.tolerance: Limit | None = None

        self.result: SystemQuantityResult | None = None

    def randomvariable(self) -> RandomVariable:
        ''' Get a RandomVariable instance for this measurement '''
        rv = RandomVariable()
        if self.typea is not None:
            rv.measure(self.typea, units=self.units, autocor=self.autocorrelation, num_new_meas=self.num_newmeas)
        else:
            rv.measure(self.testpoint, units=self.units, autocor=self.autocorrelation, num_new_meas=self.num_newmeas)
        rv._typeb[:] = self.typebs[:]
        return rv

    def expected(self):
        ''' Expected/average value of the quantity. Uses Average of the repeatability
            data if provided, otherwise returns the testpoint.
        '''
        val = np.nanmean(self.typea) if self.typea is not None else self.testpoint
        return unitmgr.make_quantity(val, self.units)

    def typea_uncertainty(self):
        ''' Standard uncertainty '''
        if self.typea is not None:
            rv = RandomVariable()
            rv.measure(self.typea, units=self.units, autocor=self.autocorrelation, num_new_meas=self.num_newmeas)
            return rv.uncertainty
        return 0

    def calculate(self):
        ''' Calculate uncertainty of the quantity '''
        rv = self.randomvariable()
        mean = self.expected()
        uncert = rv.uncertainty
        degf = rv.degrees_freedom
        if self.tolerance:
            poc = self.tolerance.probability_conformance(
                unitmgr.strip_units(mean),
                unitmgr.strip_units(uncert),
                degf)
        else:
            poc = 1

        self.result = SystemQuantityResult(
            symbol=self.symbol,
            value=mean,
            uncertainty=uncert,
            units=self.units,
            degrees_freedom=degf,
            tolerance=self.tolerance,
            p_conformance=poc,
            qty=self,
        )
        return self.result


class SystemIndirectQuantity:
    ''' An indirect measurement quantity, calculated using an equation '''
    def __init__(self):
        super().__init__()
        self.name = ''
        self.symbol = 'a'
        self.description = ''
        self.equation: str = ''
        self.tolerance: Limit | None = None
        self.outunits: str = None
        self.result: SystemQuantityResult | None = None
        # Calculate happens in the MeasureSystem, not here, because
        # all Indirect quantities need to be calculated jointly

    @property
    def variablenames(self) -> list[str]:
        ''' Get names of variables in the model '''
        if not self.symbol:
            self.symbol = 'a'
        if not self.equation:
            self.equation = '1'
        gummodel = Model(f'{self.symbol} = {self.equation}')
        return gummodel.varnames


class MeasureSystem:
    ''' Measurement system of specific measured results '''
    def __init__(self):
        self.quantities: list[SystemQuantity | SystemIndirectQuantity | SystemCurve] = []
        self.confidence: float = .95
        self.samples: int = 1000000
        self.seed: int = None
        self.correlate_typeas = True
        self.correlations: list[dict] = []   # {'var1': var1, 'var2': var2, 'cor': cor}

    def defined_symbols(self) -> list[str]:
        ''' Get a list of symbols defined in the measurement system '''
        symbols = []
        for qty in self.quantities:
            if isinstance(qty, SystemCurve):
                symbols.extend(qty.coeff_names())
                symbols.extend(list(qty.predictions.keys()))
            else:
                symbols.append(qty.symbol)
        return symbols

    def missing_symbols(self) -> list[str]:
        ''' Get any symbols from a SystemIndirectQuantity expression that are not defined '''
        defined = self.defined_symbols()
        needed: list[str] = []
        for qty in self.quantities:
            if isinstance(qty, SystemIndirectQuantity):
                needed.extend(qty.variablenames)
        missing = [v for v in needed if v not in defined]
        return missing

    def unused_symbol(self) -> str:
        ''' Get next unused symbol '''
        used = self.defined_symbols()
        letters = list(string.ascii_letters)
        for symbol in used:
            with suppress(ValueError):
                letters.remove(symbol)

        if len(letters) > 0:
            return letters[0]
        return used[0] + '0'

    def gummodel(self):
        results = []
        direct_qtys = [qty for qty in self.quantities if isinstance(qty, SystemQuantity)]
        indirect_qtys = [qty for qty in self.quantities if isinstance(qty, SystemIndirectQuantity)]
        curve_qtys = [qty for qty in self.quantities if isinstance(qty, SystemCurve)]

        # Do all direct quantities first
        for qty in direct_qtys:
            results.append(qty.calculate())

        # Then curve fit quantities
        for qty in curve_qtys:
            # Curves generate multiple QuantityResults.
            results.extend(qty.calculate())

        gummodel = None
        outunits = {}
        if indirect_qtys:
            equations = []

            # Build a single measurement model and calculate all at once
            # so that multiple function outputs may be correlated
            for qty in indirect_qtys:
                equations.append(f'{qty.symbol} = {qty.equation}')
                if qty.outunits:
                    outunits[qty.symbol] = qty.outunits

            gummodel = Model(*equations)
            for qty_result in results:
                symbol = qty_result.symbol
                if symbol in gummodel.varnames:
                    gummodel.variables.variables[symbol] = qty_result.randomvariable()
        return gummodel, outunits

    def calculate(self) -> SystemResult:
        ''' Calculate all quantities '''
        if self.seed is not None:
            np.random.seed(self.seed)

        results = []
        direct_qtys = [qty for qty in self.quantities if isinstance(qty, SystemQuantity)]
        indirect_qtys = [qty for qty in self.quantities if isinstance(qty, SystemIndirectQuantity)]
        curve_qtys = [qty for qty in self.quantities if isinstance(qty, SystemCurve)]

        # Do all direct quantities first
        for qty in direct_qtys:
            results.append(qty.calculate())

        # Then curve fit quantities
        for qty in curve_qtys:
            # Curves generate multiple QuantityResults.
            results.extend(qty.calculate())

        # Then indirect
        if indirect_qtys:
            equations = []
            outunits = {}

            # Build a single measurement model and calculate all at once
            # so that multiple function outputs may be correlated
            for qty in indirect_qtys:
                equations.append(f'{qty.symbol} = {qty.equation}')
                if qty.outunits:
                    outunits[qty.symbol] = qty.outunits

            gummodel = Model(*equations)
            for qty_result in results:
                symbol = qty_result.symbol
                if symbol in gummodel.varnames:
                    gummodel.variables.variables[symbol] = qty_result.randomvariable()

            # Calculate correlations
            if self.correlate_typeas:
                for i, qty in enumerate(direct_qtys):
                    if qty.symbol in gummodel.varnames and qty.typea is not None:
                        for qty2 in direct_qtys[i+1:]:
                            if qty2.symbol in gummodel.varnames and qty2.typea is not None:
                                if qty.typea.ndim == qty2.typea.ndim == 1 and len(qty.typea) == len(qty2.typea):
                                    corrcoef = np.corrcoef(qty.typea, qty2.typea)[0][1]
                                    gummodel.variables.correlate(qty.symbol, qty2.symbol, corrcoef)

            # User-entered correlations, override the Type A computed correlations
            for corr in self.correlations:
                gummodel.variables.correlate(corr.get('var1'), corr.get('var2'), corr.get('cor'))

            # Correlate curvefit variables
            fitresults = []
            for r in results:
                if f := r.meta.get('fitresult', None):
                    if f not in fitresults:
                        fitresults.append(f)

            for fitresult in fitresults:
                curveresult = fitresult.method('lsq')
                cov = curveresult.covariance
                names = curveresult.setup.coeffnames
                for i, name1 in enumerate(names):
                    for j, name2 in enumerate(names[i+1:]):
                        if name1 in gummodel.varnames and name2 in gummodel.varnames:
                            u1 = curveresult.uncerts[i]  # Convert covariance to correlation
                            u2 = curveresult.uncerts[j]
                            gummodel.variables.correlate(name1, name2, cov[i, j] / u1 / u2)

            # Calculate all indirect at once
            gumout = gummodel.calculate(samples=self.samples)
            gumout.units(**outunits)

            # and split into unique SystemQuantityResults
            for qty in indirect_qtys:
                uncert = gumout.gum.uncertainty[qty.symbol]
                mean = gumout.gum.expected[qty.symbol]
                degf = gumout.gum.degf[qty.symbol]
                if qty.tolerance:
                    poc = qty.tolerance.probability_conformance(
                        unitmgr.strip_units(mean),
                        unitmgr.strip_units(uncert),
                        degf)
                else:
                    poc = 1

                results.append(SystemQuantityResult(
                    symbol=qty.symbol,
                    value=mean,
                    uncertainty=uncert,
                    units=outunits.get(qty.symbol),
                    degrees_freedom=degf,
                    tolerance=qty.tolerance,
                    p_conformance=poc,
                    qty=qty,
                    meta={'gumresult': gumout}
                ))

        for result in results:
            result.meta['confidence'] = self.confidence

        return SystemResult(results, self.confidence)
