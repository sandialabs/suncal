''' Reverse uncertainty project component '''

import numpy as np

from .component import ProjectComponent
from ..common import unitmgr
from ..uncertainty import Model
from ..uncertainty.report.units import units_report
from ..reverse import ModelReverse
from .proj_uncert import ProjectUncert


class ProjectReverse(ProjectComponent):
    ''' Reverse Uncertainty project component '''
    def __init__(self, model=None, name='reverse'):
        super().__init__()
        self.name = name
        if model is not None:
            self.model = model
        else:
            self.model = ModelReverse('f=x', solvefor='x', targetnom=1)
            self.model.var('x').typeb(dist='normal', unc=1, k=2, name='u(x)')
        self.nsamples = 1000000
        self.seed = None
        self.outunits = {}
        self.result = None
        self.longdescription = None
        self.project = None  # Parent project

    def calculate(self, mc=True):
        ''' Run the calculation '''
        if self.seed:
            np.random.seed(self.seed)

        self.result = self.model.calculate(mc=mc, samples=self.nsamples)
        return self.result

    def units_report(self, **kwargs):
        ''' Create report showing units of all parameters '''
        return units_report(self.model.model, self.outunits)

    def get_dists(self):
        ''' Get distributions resulting from this calculation. If name is none,
            return a list of available distribution names.
        '''
        dists = {}
        if self.result is None:
            return dists

        for funcname in self.model.functionnames:
            if self.result.montecarlo:
                name = f'{funcname} (MC)'
                samples = self.result.montecarlo.samples[funcname]
                dists[name] = {
                    'samples': samples,
                    'median': np.median(samples),
                    'expected': self.result.montecarlo.expected[funcname]}

            if self.result.gum:
                name = f'{funcname} (GUM)'
                dists[name] = {
                    'mean': unitmgr.strip_units(self.result.gum.expected[funcname]),
                    'std': unitmgr.strip_units(self.result.gum.uncertainty[funcname]),
                    'df': self.result.gum.degf[funcname]}

        return dists

    def load_config(self, config):
        ''' Load configuration into the model '''
        if 'unitdefs' in config:
            unitmgr.register_units(config['unitdefs'])

        exprs = [f'{func["name"]}={func["expr"]}' for func in config.get('functions')]
        model = Model(*exprs)

        self.model.model = model
        self.name = config.get('name', 'reverse')
        self.outunits = {func['name']: func.get('units') for func in config.get('functions')}
        self.model.descriptions = {func['name']: func.get('desc') for func in config.get('functions')}
        self.nsamples = config.get('samples', 1000000)
        self.longdescription = config.get('longdescription')
        self.seed = config.get('seed')

        for variable in config.get('inputs', []):
            modelvar = model.var(variable['name'])
            modelvar.description = variable.get('desc', '')
            units = variable.get('units')
            value = unitmgr.make_quantity(variable.get('typea', variable.get('mean')), units)
            modelvar.measure(value, num_new_meas=variable.get('numnewmeas'))
            for uncert in variable.get('uncerts'):
                units = uncert.pop('units', None)
                desc = uncert.pop('desc', None)
                dist = uncert.pop('dist', 'normal')
                modelvar.typeb(dist, description=desc, units=units, **uncert)

        for cor in config.get('correlations', []):
            self.model.model.variables.correlate(cor['var1'], cor['var2'], cor['cor'])

        self.model.reverseparams = config.get('reverse', {})
        if 'targetunits' not in self.model.reverseparams:
            if 'func' in self.model.reverseparams:
                self.model.reverseparams['funcname'] = self.model.model.functionnames[self.model.reverseparams.get('func')]
                self.model.reverseparams.pop('func')
            funcname = self.model.reverseparams.get('funcname')
            self.model.reverseparams['targetunits'] = self.outunits.get(funcname)

    def get_config(self):
        ''' Get configuration dictionary for calculation '''
        proj = ProjectUncert(self.model.model)
        proj.nsamples = self.nsamples
        proj.seed = self.seed
        proj.outunits = self.outunits
        proj.longdescription = self.longdescription
        proj.model.descriptions = self.model.descriptions
        d = proj.get_config()
        d['name'] = self.name
        d['mode'] = 'reverse'
        d['reverse'] = self.model.reverseparams
        return d
