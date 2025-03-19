''' Results of an Uncertainty Sweep '''

from dataclasses import dataclass
import numpy as np

from ...common import reporter, unitmgr
from ...uncertainty.results.uncertainty import UncertaintyResults
from ..report.sweep import ReportSweepGum, ReportSweepMc, ReportSweep


@reporter.reporter(ReportSweepGum)
class GumSweepResults:
    ''' Class to hold results of a GUM uncertainty sweep

        Args:
            gumresults (list): List of GumResults objects
    '''
    def __init__(self, gumresults, sweeplist):
        self.gumresults = gumresults
        self.sweeplist = sweeplist
        self.functionnames = list(self.gumresults[0].functionnames)
        self.variablenames = list(self.gumresults[0].variablenames)

    def __len__(self):
        return len(self.sweeplist[0]['values'])

    def __getitem__(self, index):
        return self.gumresults[index]

    def expected(self):
        ''' Get expected values as dict of {funcname: [val1, val2,...] } '''
        values = [g.expected for g in self.gumresults]
        expected = {}
        for funcname in self.functionnames:
            units = unitmgr.get_units(values[0][funcname])
            units = str(units) if units else None
            expected[funcname] = unitmgr.make_quantity(
                np.asarray([unitmgr.strip_units(v[funcname]) for v in values]), units)
        return expected

    def uncertainties(self):
        ''' Get uncertainty values as dict of {funcname: [val1, val2,...] } '''
        values = [g.uncertainty for g in self.gumresults]
        uncertainty = {}
        for funcname in self.functionnames:
            units = unitmgr.get_units(values[0][funcname])
            units = str(units) if units else None
            uncertainty[funcname] = unitmgr.make_quantity(
                np.asarray([unitmgr.strip_units(v[funcname]) for v in values]), units)
        return uncertainty

    def expand(self, conf=0.95):
        ''' Extract expanded uncertainties at each sweep point

            Args:
                conf (float): Level of confidence in the interval
        '''
        expanded = []
        for gumresult in self.gumresults:
            expanded.append(gumresult.expand(conf=conf))
        return expanded


@reporter.reporter(ReportSweepMc)
class McSweepResults:
    ''' Class to hold results of a Monte Carlo uncertainty sweep '''
    def __init__(self, mcresults, sweeplist):
        self.mcresults = mcresults  # List of McResults objects
        self.sweeplist = sweeplist
        self.functionnames = list(self.mcresults[0].functionnames)
        self.variablenames = list(self.mcresults[0].variablenames)

    def __len__(self):
        return len(self.sweeplist[0]['values'])

    def __getitem__(self, index):
        return self.mcresults[index]

    def expected(self):
        ''' Get expected values as dict of {funcname: [val1, val2,...] } '''
        values = [g.expected for g in self.mcresults]
        expected = {}
        for funcname in self.functionnames:
            units = unitmgr.get_units(values[0][funcname])
            units = str(units) if units else None
            expected[funcname] = unitmgr.make_quantity(
                np.asarray([unitmgr.strip_units(v[funcname]) for v in values]), units)
        return expected

    def uncertainties(self):
        ''' Get uncertainties values as dict of {funcname: [val1, val2,...] } '''
        values = [mc.uncertainty for mc in self.mcresults]
        uncertainty = {}
        for funcname in self.functionnames:
            units = unitmgr.get_units(values[0][funcname])
            units = str(units) if units else None
            uncertainty[funcname] = unitmgr.make_quantity(
                np.asarray([unitmgr.strip_units(v[funcname]) for v in values]), units)
        return uncertainty

    def expand(self, conf=.95, shortest=False):
        ''' Extract expanded uncertainties at each sweep point

            Args:
                conf (float): Level of confidence in the interval
                shortest (bool): Use shortest instead of symmetric interval
        '''
        expanded = []
        for mcresult in self.mcresults:
            expanded.append(mcresult.expand(conf=conf, shortest=shortest))
        return expanded


@reporter.reporter(ReportSweep)
@dataclass
class SweepResults:
    gum: GumSweepResults
    montecarlo: McSweepResults
    sweeplist: list

    def __len__(self):
        return len(self.sweeplist[0]['values'])

    def __getitem__(self, index):
        return UncertaintyResults(self.gum[index], self.montecarlo[index])
