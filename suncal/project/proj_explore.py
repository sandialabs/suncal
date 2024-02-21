''' Distribution Explorer project component '''

from .component import ProjectComponent
from ..common import distributions
from ..distexplore import DistExplore


class ProjectDistExplore(ProjectComponent):
    ''' Distribution Explorer project component '''
    def __init__(self, model=None, name='distributions'):
        super().__init__(name=name)
        self.nsamples = 10000
        self.seed = None
        if model is None:
            self.model = DistExplore()
        else:
            self.model = model
        self._result = self.model

    def calculate(self):
        ''' Run calculation '''
        return self._result

    def get_config(self):
        ''' Get configuration '''
        d = {}
        d['mode'] = 'distributions'
        d['name'] = self.name
        d['desc'] = self.description
        d['seed'] = self.model.seed
        d['distnames'] = [str(x) for x in self.model.dists]
        d['distributions'] = [x.get_config() if x is not None else None for x in self.model.dists.values()]
        return d

    def load_config(self, config):
        ''' Load config into this project '''
        self.name = config.get('name', 'distributions')
        self.description = config.get('desc', '')
        self.seed = config.get('seed', None)
        exprs = config.get('distnames', [])
        dists = [distributions.from_config(x) if x is not None else None for x in config.get('distributions', [])]
        self.model.dists = dict(zip(exprs, dists))
