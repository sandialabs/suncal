''' Reverse sweep project component '''

import numpy as np

from .proj_sweep import ProjectSweep
from ..sweep import UncertSweepReverse
from ..common import unitmgr


class ProjectReverseSweep(ProjectSweep):
    ''' Reverse Sweep project component '''
    def __init__(self, model=None, name='revsweep'):
        super().__init__(model, name)
        if model is None:
            self.model = UncertSweepReverse('f=x', solvefor='x', targetnom=1)
            self.model.model.var('x').typeb(dist='normal', unc=1, k=2, name='u(x)')

    def calculate(self, mc=True):
        ''' Run the calculation '''
        if self.seed:
            np.random.seed(self.seed)
        self._result = self.model.calculate(mc=mc, samples=self.nsamples)
        return self._result

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
            yvals = [unitmgr.strip_units(g.solvefor_value) for g in self.result.gum.resultlist]
            uyvals = [unitmgr.strip_units(g.u_solvefor_value) for g in self.result.gum.resultlist]
        else:
            yvals = [unitmgr.strip_units(g.solvefor_value) for g in self.result.montecarlo.resultlist]
            uyvals = [unitmgr.strip_units(g.u_solvefor_value) for g in self.result.montecarlo.resultlist]
        return {'x': xvals, 'y': yvals, 'u(y)': uyvals}

    def get_config(self):
        ''' Get configuration dictionary '''
        d = super().get_config()
        d['mode'] = 'reversesweep'
        d['reverse'] = self.model.reverseparams
        return d

    def load_config(self, config):
        ''' Load configuration into project '''
        super().load_config(config)  # UncertProp and Sweeps
        self.model.reverseparams = config.get('reverse', {})
        if 'targetunits' not in self.model.reverseparams:
            if 'func' in self.model.reverseparams:
                self.model.reverseparams['funcname'] = self.model.model.functionnames[self.model.reverseparams.get('func')]
                self.model.reverseparams.pop('func')
            funcname = self.model.reverseparams.get('funcname')
            self.model.reverseparams['targetunits'] = self.outunits.get(funcname)
        return
