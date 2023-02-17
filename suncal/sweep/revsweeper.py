''' Sweep a Reverse Uncertainty calcualtion '''

from dataclasses import dataclass

from ..common import unitmgr, reporter
from ..uncertainty.model import Model
from ..reverse import ModelReverse
from ..reverse.reverse import ResultsReverse
from .sweeper import UncertSweep
from .report.revsweep import ReportReverseSweep, ReportReverseSweepGum, ReportReverseSweepMc


def model_copy(model, **reverseparams):
    ''' Make a deep copy of the uncertasinty Model '''
    exprs = [f'{name}={expr}' for name, expr in zip(model.functionnames, model.exprs)]
    modelcopy = ModelReverse(*exprs, **reverseparams)
    for varname in modelcopy.model.variables.names:
        var = modelcopy.model.var(varname)
        var.measure(model.variables.expected[varname])
        for typeb in model.variables.variables[varname]._typeb:
            units = str(typeb.units) if typeb.units is not None else None
            var.typeb(dist=typeb.distname, description=typeb.description,
                      units=units, name=typeb.name, **typeb.kwargs)
    return modelcopy


@reporter.reporter(ReportReverseSweepGum)
@dataclass
class ResultReverseSweepGum:
    resultlist: object
    sweeplist: list

    def __len__(self):
        return len(self.sweeplist)

    def __getitem__(self, index):
        ''' Get results of single point at index '''
        return self.resultlist[index]


@reporter.reporter(ReportReverseSweepMc)
@dataclass
class ResultReverseSweepMc:
    resultlist: object
    sweeplist: list

    def __len__(self):
        return len(self.sweeplist)

    def __getitem__(self, index):
        ''' Get results of single point at index '''
        return self.resultlist[index]


@reporter.reporter(ReportReverseSweep)
@dataclass
class ResultReverseSweep:
    gum: ResultReverseSweepGum
    montecarlo: ResultReverseSweepMc
    sweeplist: list

    def __len__(self):
        return len(self.sweeplist)

    def __getitem__(self, index):
        ''' Get results of single point at index '''
        return ResultsReverse(self.gum[index], self.montecarlo[index])


class UncertSweepReverse(UncertSweep):
    ''' Sweep a reverse propagation calculator

        Args:
            *exprs (str): Function expressions for the model
            solvefor (str): Variable to solve for
            targetnom (float): Target nominal value for function output
            targetunc (float): Target uncertainty for function output
            funcname (str): Name of function (for multi-function models)
            targetunits (str): Units for target nominal value
    '''
    def __init__(self, *exprs, solvefor, targetnom=None, targetunc=None, funcname=None, targetunits=None):
        self.model = Model(*exprs)
        if funcname is None:
            funcname = self.model.functionnames[-1]
        self.reverseparams = {'solvefor': solvefor,
                              'targetnom': targetnom,
                              'targetunc': targetunc,
                              'targetunits': targetunits,
                              'funcname': funcname}
        self.sweeplist = []

    @property
    def variables(self):
        ''' Get variables object '''
        return self.model.variables

    @property
    def functionnames(self):
        return self.model.functionnames

    @property
    def varnames(self):
        return self.model.varnames

    @property
    def constants(self):
        return self.model.constants

    @property
    def basesympys(self):
        return self.model.basesympys

    @property
    def var(self):
        return self.model.var

    def _sweep_models(self):
        ''' Iterate one Model instance for each sweep point '''
        if len(self.sweeplist) == 0:
            raise ValueError('No sweeps defined.')

        N = 0
        for sweepparams in self.sweeplist:
            # Note: all N's should be the same...
            N = max(N, len(sweepparams.get('values', [])))

        for sweepidx in range(N):
            modelcopy = model_copy(self.model, **self.reverseparams)  # Reverse model

            for sweepparams in self.sweeplist:
                inptname = sweepparams.get('var', None)
                comp = sweepparams.get('comp', 'nom')
                param = sweepparams.get('param', None)
                values = sweepparams.get('values', [])

                if inptname == 'corr':
                    modelcopy.model.variables.correlate(sweepparams['var1'], sweepparams['var2'], values[sweepidx])
                elif comp == 'nom':
                    inptvar = modelcopy.var(inptname)
                    units = str(inptvar.units) if inptvar.units else None
                    inptvar.measure(unitmgr.make_quantity(values[sweepidx], units))
                elif param == 'df':
                    inptvar = modelcopy.var(inptname)
                    inptvar.get_typeb(comp).degf = values[sweepidx]
                else:
                    inptvar = modelcopy.var(inptname)
                    comp = inptvar.get_typeb(comp)
                    units = str(inptvar.units) if inptvar.units else None
                    comp.set_kwargs(**{param: unitmgr.make_quantity(values[sweepidx], units)})

            yield modelcopy

    def calculate_gum(self):
        ''' Calculate reverse propagation sweep. '''
        resultlist = []
        for model in self._sweep_models():
            resultlist.append(model.calculate_gum())
        return ResultReverseSweepGum(resultlist, self.sweeplist)

    def monte_carlo(self, samples=1000000):
        ''' Calculate Monte Carlo reverse propagation sweep '''
        resultlist = []
        for model in self._sweep_models():
            resultlist.append(model.monte_carlo(samples=samples))
        return ResultReverseSweepMc(resultlist, self.sweeplist)

    def calculate(self, mc=True, samples=1000000):
        ''' Run GUM and Monte Carlo calculations and return ReportSweep '''
        gumresults = self.calculate_gum()
        mcresults = None
        if mc:
            mcresults = self.monte_carlo(samples=samples)
        return ResultReverseSweep(gumresults, mcresults, self.sweeplist)
