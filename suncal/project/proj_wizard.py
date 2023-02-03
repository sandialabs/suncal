''' Guided/wizard uncertainty calculation interface '''

from enum import IntEnum, auto

import numpy as np

from .component import ProjectComponent
from .proj_uncert import ProjectUncert
from ..uncertainty import Model
from ..common import unitmgr


class VariableDataType(IntEnum):
    ''' Type of measured data for variable '''
    SINGLE = auto()
    REPEATABILITY = auto()
    REPRODUCIBILITY = auto()


class ProjectUncertWizard(ProjectComponent):
    ''' Project component for Guided Uncertainty Wizard. '''
    def __init__(self, model=None, name='wizard'):
        if model is None:
            self.model = Model()
        else:
            self.model = model
        self.variablesdone = []
        self.outunits = {}
        self.name = name
        self.longdescription = ''
        self.report = None
        self.project = None  # Parent project

    def set_function(self, func):
        ''' Change the model function, without changing variable definitions '''
        if len(self.model.exprs) == 0 or self.model.exprs[0] != func:
            definedvars = self.model.variables.variables
            self.model = Model(func)

            # Restore any variables already defined
            varnames = self.model.variables.names
            for varname, var in definedvars.items():
                if varname in varnames:
                    self.model.variables.variables[varname] = var

    @property
    def missingvars(self):
        ''' Get list of variable names that have not yet been defined '''
        return sorted(list(set(self.model.variables.names).difference(set(self.variablesdone))))

    def inpt_type(self, name):
        ''' Get the VariableDataType of the variablename '''
        data = self.model.var(name).value
        dimension = len(data.shape)
        if dimension == 1 and data.shape[0] > 1:
            res = VariableDataType.REPEATABILITY
        elif dimension == 2:
            res = VariableDataType.REPRODUCIBILITY
        else:
            res = VariableDataType.SINGLE
        return res

    def set_inputval(self, name, data, units=None, num_newmeas=None, autocor=True):
        ''' Set Input value only '''
        data = np.asarray(data)
        dimension = len(data.shape)
        if dimension == 0:
            mean = unitmgr.make_quantity(float(data), units)
            self.model.var(name).measure(mean)
        elif dimension == 1:
            if units:
                data = unitmgr.make_quantity(data, units)
            self.model.var(name).measure(data, num_new_meas=num_newmeas, autocor=autocor)
        elif dimension == 2:
            if units:
                data = unitmgr.make_quantity(data, units)
            self.model.var(name).measure(data, num_new_meas=num_newmeas)
        else:
            raise NotImplementedError
        self.variablesdone.append(name)

    def set_uncert(self, varname, **params):
        ''' Set the Type B uncertianty defined by params '''
        self.model.var(varname).typeb(**params)

    def calculate(self):
        ''' Run the calculation '''
        self.result = self.model.calculate().units(**self.outunits)
        return self.result

    def get_config(self):
        ''' Get project configuration '''
        d = ProjectUncert(self.model).get_config()
        d['mode'] = 'wizard'
        d['name'] = self.name
        d['desc'] = self.longdescription
        return d

    def load_config(self, config):
        ''' Load project configuration '''
        uproj = ProjectUncert()
        uproj.load_config(config)
        self.model = uproj.model
        self.name = config.get('name', '')
        self.longdescription = config.get('desc', '')
        self.variablesdone = self.model.variables.names
