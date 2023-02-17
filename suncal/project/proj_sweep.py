''' Uncertainty sweep project component '''

import numpy as np

from .component import ProjectComponent
from .proj_uncert import ProjectUncert
from ..sweep import UncertSweep
from ..uncertainty import Model
from ..uncertainty.report.units import units_report
from ..common import unitmgr
from ..datasets.dataset_model import DataSet


class ProjectSweep(ProjectComponent):
    ''' Uncertainty project component '''
    def __init__(self, model=None, name='sweep'):
        super().__init__()
        self.name = name
        if model is not None:
            self.model = model
        else:
            self.model = UncertSweep(Model('f=x'))
            self.model.model.var('x').typeb(dist='normal', unc=1, k=2, name='u(x)')
        self.nsamples = 1000000
        self.seed = None
        self.outunits = {}
        self.result = None
        self.longdescription = None
        self.project = None  # Parent project

    def calculate(self):
        ''' Run the calculation '''
        if self.seed:
            np.random.seed(self.seed)
        self.result = self.model.calculate(samples=self.nsamples)
        return self.result

    def units_report(self, **kwargs):
        ''' Create report showing units of all parameters '''
        return units_report(self.model, self.outunits)

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
            yvals = unitmgr.strip_units(self.result.gum.expected()[funcname])
            uyvals = unitmgr.strip_units(self.result.gum.uncertainties()[funcname])
        else:
            yvals = unitmgr.strip_units(self.result.montecarlo.expected()[funcname])
            uyvals = unitmgr.strip_units(self.result.montecarlo.uncertainties()[funcname])
        names.append(funcname)
        names.append(f'u({funcname})')
        return DataSet(np.vstack((xvals, yvals, uyvals)), colnames=names)

    def load_config(self, config):
        ''' Load configuration into this project '''
        puncert = ProjectUncert()
        puncert.load_config(config)
        self.model.model = puncert.model
        self.name = puncert.name
        self.outunits = puncert.outunits
        self.nsamples = puncert.nsamples
        self.longdescription = puncert.longdescription
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
        puncert.longdescription = self.longdescription
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
