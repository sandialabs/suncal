''' Risk Calculation project component '''

from .component import ProjectComponent
from ..common import distributions

from ..risk.risk_model import RiskModel, RiskResults


class ProjectRisk(ProjectComponent):
    ''' Risk project component '''
    def __init__(self, model=None, name='risk'):
        super().__init__(name=name)
        if model is None:
            self.model = RiskModel()
        else:
            self.model = model

    def calculate(self) -> RiskResults:
        ''' Calculate values, returning the results object '''
        self._result = self.model.calculate()
        return self._result

    def get_config(self):
        ''' Get configuration dictionary '''
        d = {}
        d['mode'] = 'risk'
        d['name'] = self.name
        d['desc'] = self.description
        d['bias'] = self.model.testbias

        if self.model.process_dist is not None:
            d['distproc'] = self.model.process_dist.get_config()

        if self.model.measure_dist is not None:
            d['disttest'] = self.model.measure_dist.get_config()

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
            self.model.process_dist = distributions.from_config(dproc)
        else:
            self.model.process_dist = None

        dtest = config.get('disttest', None)
        if dtest is not None:
            self.model.measure_dist = distributions.from_config(dtest)
        else:
            self.model.measure_dist = None
