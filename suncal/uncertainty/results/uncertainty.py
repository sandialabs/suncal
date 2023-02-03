
from dataclasses import dataclass

from ...common import reporter
from ..report.uncertainty import ReportUncertainty
from ..report.cplx import ReportComplex
from .monte import McResults
from .gum import GumResults


@reporter.reporter(ReportUncertainty)
@dataclass
class UncertaintyResults:
    ''' Results of GUM and Monte Carlo uncertainty calculation

        Attributes:
            gum: Results of GUM calculation
            montecarlo: Results of Monte Carlo calculation
            report (Report): Generate formatted reports of the results

        Methods:
            units: Convert the units of uncertainty and expected
    '''
    gum: GumResults
    montecarlo: McResults

    @property
    def functionnames(self):
        ''' List of function names in model '''
        if self.gum is not None:
            return self.gum.functionnames
        if self.montecarlo is not None:
            return self.montecarlo.functionnames
        return None

    @property
    def variablenames(self):
        ''' List of input variable names in model '''
        if self.gum is not None:
            return self.gum.variablenames
        if self.montecarlo is not None:
            return self.montecarlo.variablenames
        return None

    @property
    def descriptions(self):
        ''' Dictionary of function descriptions '''
        if self.gum is not None and self.gum.descriptions is not None:
            return self.gum.descriptions
        if self.montecarlo is not None and self.montecarlo.descriptions is not None:
            return self.montecarlo.descriptions

    def units(self, **units):
        ''' Convert units of uncertainty results

            Args:
                **units: functionnames and unit string to convert each
                    model function result to
        '''
        if self.gum is not None:
            self.gum.units(**units)
        if self.montecarlo is not None:
            self.montecarlo.units(**units)
        return self

    def getunits(self):
        ''' Get Pint units currently configured in result '''
        if self.gum is not None:
            return self.gum.getunits()
        if self.montecarlo is not None:
            return self.montecarlo.getunits()
        return {}


@reporter.reporter(ReportComplex)
class UncertaintyCplxResults:
    ''' Results of GUM and Monte Carlo uncertainty calculation
        with Complex numbers

        Attributes:
            gum: Results of GUM calculation
            montecarlo: Results of Monte Carlo calculation
    '''
    def __init__(self, gumresults, mcresults):
        self.gum = gumresults
        self.montecarlo = mcresults
        self.componentresults = UncertaintyResults(self.gum._gumresults, self.montecarlo._mcresults)
        self._degrees = False

    def units(self, **units):
        ''' Convert units of uncertainty results

            Args:
                **units: functionnames and unit string to convert each
                    model function result to
        '''
        if self.gum is not None:
            self.gum.units(**units)
        if self.montecarlo is not None:
            self.montecarlo.units(**units)
        return self

    def degrees(self, degrees):
        self._degrees = degrees
        if self.gum is not None:
            self.gum.degrees(degrees)
        if self.montecarlo is not None:
            self.montecarlo.degrees(degrees)
        return self
