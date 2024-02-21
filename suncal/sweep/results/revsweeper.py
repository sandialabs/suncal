''' Sweep a Reverse Uncertainty calcualtion '''

from dataclasses import dataclass

from ...common import reporter
from ...reverse.reverse import ResultsReverse
from ..report.revsweep import ReportReverseSweep, ReportReverseSweepGum, ReportReverseSweepMc


@reporter.reporter(ReportReverseSweepGum)
@dataclass
class ResultReverseSweepGum:
    ''' Results of reverse sweep using GUM method '''
    resultlist: list
    sweeplist: list

    def __len__(self):
        return len(self.sweeplist)

    def __getitem__(self, index):
        ''' Get results of single point at index '''
        return self.resultlist[index]


@reporter.reporter(ReportReverseSweepMc)
@dataclass
class ResultReverseSweepMc:
    ''' Results of reverse sweep using Monte Carlo method '''
    resultlist: list
    sweeplist: list

    def __len__(self):
        return len(self.sweeplist)

    def __getitem__(self, index):
        ''' Get results of single point at index '''
        return self.resultlist[index]


@reporter.reporter(ReportReverseSweep)
@dataclass
class ResultReverseSweep:
    ''' Results of Reverse Sweep using both GUM and Monte Carlo methods '''
    gum: ResultReverseSweepGum
    montecarlo: ResultReverseSweepMc
    sweeplist: list

    def __len__(self):
        return len(self.sweeplist)

    def __getitem__(self, index):
        ''' Get results of single point at index '''
        return ResultsReverse(self.gum[index], self.montecarlo[index])
