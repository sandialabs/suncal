''' DataSet calculation project component '''

import numpy as np

from .component import ProjectComponent
from ..datasets.dataset_model import DataSet, DataSetSummary


class ProjectDataSet(ProjectComponent):
    ''' DataSet project component '''
    def __init__(self, model=None, name='dataset'):
        super().__init__(name=name)
        if model is None:
            self.model = DataSet()
        else:
            self.model = model

    def issummary(self):
        ''' Is the data in DataSetSummary form? '''
        return isinstance(self.model, DataSetSummary)

    def setdata(self, data):
        self.model.data = data
        self._result = None

    def setcolnames(self, names):
        self.model.colnames = names
        self._result = None

    def clear(self):
        self._result = None
        self.model.clear()

    def calculate(self):
        ''' Run calculation '''
        self._result = self.model.calculate()
        return self._result

    def get_dists(self):
        ''' Get dictionary of distributions in this dataset '''
        d = {}
        colnames = self.model.colnames

        # Pooled stats returned as mean/std/df dictionary
        if len(colnames) > 1:
            # Get pooled statistics
            pstats = self.model.result.pooled
            stderr = self.model.result.uncertainty
            d['Stdev of Mean'] = {
                'median': pstats.mean,
                'std': stderr.stderr,
                'df': stderr.stderr_degf}
            d['Repeatability Stdev'] = {
                'median': pstats.mean,
                'std': pstats.repeatability,
                'df': pstats.repeat_degf}
            d['Reproducibility Stdev'] = {
                'median': pstats.mean,
                'std': pstats.reproducibility,
                'df': pstats.reprod_degf}

        # Individual columns are returned as sampled data
        for col in colnames:
            d[f'Column {col}'] = {'samples': self.model.get_column(col)}

        return d

    def get_arrays(self):
        d = {'Group Means': self.model.to_array()}
        return d

    def get_config(self):
        ''' Get the dataset configuration dictionary '''
        d = {}
        d['mode'] = 'data'
        d['name'] = self.name
        d['colnames'] = self.model.colnames
        d['data'] = self.model.data.astype('float').tolist()
        d['desc'] = self.description

        if isinstance(self.model, DataSetSummary):
            d['nmeas'] = self.model.nmeas.astype('float').tolist()
            d['means'] = self.model.means.astype('float').tolist()
            d['stds'] = self.model.stdevs.astype('float').tolist()
            d['summary'] = True
        return d

    def load_config(self, config):
        ''' Load config into this DataSet '''
        self.name = config.get('name', 'data')
        self.description = config.get('desc', '')
        colnames = config.get('colnames')
        if 'nmeas' in config:
            means = config.get('means')
            stds = config.get('stds')
            nmeas = config.get('nmeas')
            self.model = DataSetSummary(means, stds, nmeas, column_names=colnames)
        else:
            self.model = DataSet(np.array(config['data']), column_names=colnames)
        self.calculate()
