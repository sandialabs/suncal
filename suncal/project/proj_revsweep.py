''' Reverse sweep project component '''

import numpy as np

from .proj_sweep import ProjectSweep
from ..sweep import UncertSweepReverse
from ..common import unitmgr
from ..datasets.dataset_model import DataSet


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
        self.result = self.model.calculate(mc=mc, samples=self.nsamples)
        return self.result

    def get_dataset(self, name=None):
        ''' Get DataSet object from sweep output with the given name. If name is None, return a list
            of array names available.
        '''
        names = []
        for n in self.model.model.functionnames:
            if self.result is not None and self.result.gum is not None:
                names.append(f'{n} (GUM)')
            if self.result is not None and self.result.montecarlo is not None:
                names.append(f'{n} (MC)')

        if name is None:
            return names

        if name in names:
            name, method = name.split(' ')
            dset = self.to_array(gum=(method == '(GUM)'), funcname=name)
        else:
            raise ValueError(f'{name} not found in output')
        return dset

    def to_array(self, gum=True, funcname=None):
        ''' Return DataSet object of swept data and uncertainties

            Args:
                gum (bool): Use gum (True) or monte carlo (False) values
                funcidx (int): Index of function in calculator as y values

            Returns:
                dset: DataSet containing mean and uncertainties of each sweep point
        '''
        xvals = [unitmgr.strip_units(x) for x in self.result.report._sweepvals]
        names = self.result.report._header_strs.copy()

        if gum:
            yvals = [unitmgr.strip_units(g.solvefor_value) for g in self.result.gum.resultlist]
            uyvals = [unitmgr.strip_units(g.u_solvefor_value) for g in self.result.gum.resultlist]
        else:
            yvals = [unitmgr.strip_units(g.solvefor_value) for g in self.result.montecarlo.resultlist]
            uyvals = [unitmgr.strip_units(g.u_solvefor_value) for g in self.result.montecarlo.resultlist]
        names.append(funcname)
        names.append(f'u({funcname})')
        return DataSet(np.vstack((xvals, yvals, uyvals)), colnames=names)

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
