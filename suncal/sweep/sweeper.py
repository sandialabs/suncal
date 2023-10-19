''' Methods for running multiple uncertainty calculations sweeping over input arrays. '''

from ..uncertainty.model import Model
from .results.sweep import GumSweepResults, McSweepResults, SweepResults


def model_copy(model):
    ''' Make a deep copy of the uncertainty Model '''
    exprs = [f'{name}={expr}' for name, expr in zip(model.functionnames, model.exprs)]
    modelcopy = Model(*exprs)
    modelcopy.descriptions = model.descriptions
    for varname in modelcopy.variables.names:
        var = modelcopy.var(varname)
        var.measure(model.variables.expected[varname], description=model.var(varname).description)
        for typeb in model.variables.variables[varname]._typeb:
            var.typeb(dist=typeb.distname, description=typeb.description,
                      units=typeb.units, name=typeb.name, **typeb.kwargs)
    return modelcopy


class UncertSweep:
    ''' Class for running multiple uncertainty calculations over a range of input values

        Args:
            model (Model): Measurement model to sweep
    '''
    def __init__(self, model):
        self.model = model
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

    def add_sweep_nom(self, varname, values):
        ''' Add sweep of nominal value.

            Args:
                varname (string): Name of variable to sweep nominal value
                values (array): Values for sweep
        '''
        d = {'var': varname, 'comp': 'nom', 'values': values}
        self.sweeplist.append(d)

    def add_sweep_df(self, varname, values, comp=None):
        ''' Add sweep of degrees of freedom.

            Args:
                varname (string): Name of variable to sweep deg.freedom value
                values (array): Values for sweep
        '''
        if comp is None:
            comp = f'u({varname})'
        d = {'var': varname, 'comp': comp, 'param': 'df', 'values': values}
        self.sweeplist.append(d)

    def add_sweep_corr(self, var1, var2, values):
        ''' Add sweep of correlation coefficient between var1 and var2

            Args:
                var1 (str): Name of variable 1
                var2 (str): Name of variable 2
                values (array): Correlation values to sweep
        '''
        d = {'var': 'corr', 'var1': var1, 'var2': var2, 'values': values}
        self.sweeplist.append(d)

    def add_sweep_unc(self, varname, values, comp=None, param='std'):
        ''' Add sweep of uncertainty component parameter.

            Args:
                varname (string): Name of variable to sweep uncertainty component
                values (array): Values for sweep
                comp (string): Name of uncertainty component. Defaults to u_{varname}.
                param (string): Name of uncertainty parameter in distribution.
                index (int): Index of sweep item in sweeplist. Use None to add a new item
                  to the list.
        '''
        if comp is None:
            comp = f'u({varname})'
        d = {'var': varname, 'comp': comp, 'param': param, 'values': values}
        self.sweeplist.append(d)

    def _sweep_models(self):
        ''' Iterate through a Model instance for each sweep point '''
        if len(self.sweeplist) == 0:
            raise ValueError('No sweeps defined.')

        N = 0
        for sweepparams in self.sweeplist:
            # Note: all N's should be the same...
            N = max(N, len(sweepparams.get('values', [])))

        for sweepidx in range(N):
            modelcopy = model_copy(self.model)

            for sweepparams in self.sweeplist:
                inptname = sweepparams.get('var', None)
                comp = sweepparams.get('comp', 'nom')
                param = sweepparams.get('param', None)
                values = sweepparams.get('values', [])

                if inptname == 'corr':
                    modelcopy.variables.correlate(sweepparams['var1'], sweepparams['var2'], values[sweepidx])
                elif comp == 'nom':
                    inptvar = modelcopy.var(inptname)
                    units = str(inptvar.units) if inptvar.units else None
                    inptvar.measure(values[sweepidx], units=units)
                elif param == 'df':
                    inptvar = modelcopy.var(inptname)
                    inptvar.get_typeb(comp).degf = values[sweepidx]
                else:
                    inptvar = modelcopy.var(inptname)
                    typeb = inptvar.get_typeb(comp)
                    typeb.set_kwargs(**{param: values[sweepidx]})
            yield modelcopy

    def calculate_gum(self):
        ''' Calculate using GUM method '''
        resultlist = []
        for model in self._sweep_models():
            resultlist.append(model.calculate_gum())
        return GumSweepResults(resultlist, self.sweeplist)

    def monte_carlo(self, samples=1000000):
        ''' Calculate using Monte Carlo method '''
        resultlist = []
        for model in self._sweep_models():
            resultlist.append(model.monte_carlo(samples=samples))
        return McSweepResults(resultlist, self.sweeplist)

    def calculate(self, samples=1000000):
        ''' Run GUM and Monte Carlo calculations and return ReportSweep '''
        gumresult = self.calculate_gum()
        mcresult = self.monte_carlo(samples=samples)
        return SweepResults(gumresult, mcresult, self.sweeplist)
