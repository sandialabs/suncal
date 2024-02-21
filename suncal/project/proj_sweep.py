''' Uncertainty sweep project component '''

import numpy as np

from .component import ProjectComponent
from .proj_uncert import ProjectUncert
from ..sweep import UncertSweep
from ..uncertainty import Model
from ..uncertainty.report.units import units_report
from ..common import unitmgr


class ProjectSweep(ProjectComponent):
    ''' Uncertainty project component '''
    def __init__(self, model=None, name='sweep'):
        super().__init__(name=name)
        if model is not None:
            self.model = model
        else:
            self.model = UncertSweep(Model('f=x'))
            self.model.model.var('x').typeb(dist='normal', unc=1, k=2, name='u(x)')
        self.nsamples = 1000000
        self.seed = None
        self.outunits = {}

    def calculate(self):
        ''' Run the calculation '''
        if self.seed:
            np.random.seed(self.seed)
        self._result = self.model.calculate(samples=self.nsamples)
        return self._result

    def units_report(self, **kwargs):
        ''' Create report showing units of all parameters '''
        return units_report(self.model, self.outunits)

    def get_arrays(self):
        d = {}
        for name in self.model.model.functionnames:
            if self._result is not None and self.result.gum is not None:
                d[f'{name} (GUM)'] = self.to_array(gum=True, funcname=name)
            if self._result is not None and self.result.montecarlo is not None:
                d[f'{name} (MC)'] = self.to_array(gum=False, funcname=name)
        return d

    def to_array(self, gum=True, funcname=None):
        ''' Return dictionary {x: y: uy:} of swept data and uncertainties

            Args:
                gum (bool): Use gum (True) or monte carlo (False) values
                funcidx (int): Index of function in calculator as y values
        '''
        xvals = unitmgr.strip_units(self.result.report._sweepvals[0])  # Only first x value
        if gum:
            yvals = unitmgr.strip_units(self.result.gum.expected()[funcname])
            uyvals = unitmgr.strip_units(self.result.gum.uncertainties()[funcname])
        else:
            yvals = unitmgr.strip_units(self.result.montecarlo.expected()[funcname])
            uyvals = unitmgr.strip_units(self.result.montecarlo.uncertainties()[funcname])
        return {'x': xvals, 'y': yvals, 'u(y)': uyvals}

    def load_config(self, config):
        ''' Load configuration into this project '''
        puncert = ProjectUncert()
        puncert.load_config(config)
        self.model.model = puncert.model
        self.name = puncert.name
        self.outunits = puncert.outunits
        self.nsamples = puncert.nsamples
        self.description = puncert.description
        self.seed = puncert.seed
        self.model.sweeplist = []

        sweeps = config.get('sweeps', [])
        for sweep in sweeps:
            var = sweep['var']
            comp = sweep.get('comp', None)
            param = sweep.get('param', None)
            values = sweep['values']
            if var == 'corr':
                self.model.add_sweep_corr(sweep.get('var1', None), sweep.get('var2', None), values)
            elif comp == 'nom':
                self.model.add_sweep_nom(var, values)
            elif param == 'df':
                self.model.add_sweep_df(var, values, comp)
            else:
                self.model.add_sweep_unc(var, values, comp, param)

    def get_config(self):
        ''' Get configuration dictionary '''
        puncert = ProjectUncert(self.model.model)
        puncert.outunits = self.outunits
        puncert.description = self.description
        puncert.seed = self.seed
        puncert.nsamples = self.nsamples
        d = puncert.get_config()
        d['name'] = self.name
        d['mode'] = 'sweep'
        sweeps = []
        for sweep in self.model.sweeplist:
            sweep['values'] = list(sweep.get('values', []))
            sweeps.append(sweep)
        d['sweeps'] = sweeps
        return d
