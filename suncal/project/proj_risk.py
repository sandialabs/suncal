''' Risk Calculation project component '''

from .component import ProjectComponent
from ..common import distributions

from ..risk.risk_model import RiskModel


class ProjectRisk(ProjectComponent):
    ''' Risk project component '''
    def __init__(self, model=None, name='risk'):
        super().__init__()
        self.name = name
        if model is None:
            self.model = RiskModel()
        else:
            self.model = model
        self.description = ''
        self.project = None  # Parent project
        self.result = self.model

    def calculate(self):
        ''' "Calculate" values, returning the model/results object '''
        return self.model

    def get_config(self):
        ''' Get configuration dictionary '''
        d = {}
        d['mode'] = 'risk'
        d['name'] = self.name
        d['desc'] = self.description
        d['bias'] = self.model.testbias

        if self.model.procdist is not None:
            d['distproc'] = self.model.procdist.get_config()

        if self.model.testdist is not None:
            d['disttest'] = self.model.testdist.get_config()

        d['GBL'] = self.model.gbofsts[0]
        d['GBU'] = self.model.gbofsts[1]
        d['LL'] = self.model.speclimits[0]
        d['UL'] = self.model.speclimits[1]
        return d

    def load_config(self, config):
        ''' Load config into this project instance '''
        self.description = config.get('desc', '')
        self.model.speclimits = (config.get('LL', 0), config.get('UL', 0))
        self.model.gbofsts = (config.get('GBL', 0), config.get('GBU', 0))
        self.model.testbias = config.get('bias', 0)

        dproc = config.get('distproc', None)
        if dproc is not None:
            self.model.procdist = distributions.from_config(dproc)
        else:
            self.model.procdist = None

        dtest = config.get('disttest', None)
        if dtest is not None:
            self.model.testdist = distributions.from_config(dtest)
        else:
            self.model.testdist = None
