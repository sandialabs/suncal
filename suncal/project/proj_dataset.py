''' DataSet calculation project component '''

import numpy as np

from .component import ProjectComponent
from ..datasets.dataset_model import DataSet, DataSetSummary


class ProjectDataSet(ProjectComponent):
    ''' DataSet project component '''
    def __init__(self, model=None, name='dataset'):
        super().__init__()
        self.name = name
        if model is None:
            self.model = DataSet()
        else:
            self.model = model
        self.description = ''
        self.project = None  # Parent project
        self.result = self.model

    def issummary(self):
        ''' Is the data in DataSetSummary form? '''
        return isinstance(self.model, DataSetSummary)

    def calculate(self):
        ''' Run calculation '''
        return self.model

    def get_dists(self):
        ''' Get dictionary of distributions in this dataset '''
        d = {}
        colnames = self.model.colnames

        # Pooled stats returned as mean/std/df dictionary
        if len(colnames) > 1:
            # Get pooled statistics
            pstats = self.model.pooled_stats()
            stderr = self.model.standarderror()
            d['Stdev of Mean'] = {
                'mean': pstats.mean,
                'sem': stderr.standarderror,
                'std': stderr.standarddeviation,
                'df': stderr.degf}
            d['Repeatability Stdev'] = {
                'mean': pstats.mean,
                'std': pstats.repeatability,
                'df': pstats.repeatability_degf}
            d['Reproducibility Stdev'] = {
                'mean': pstats.mean,
                'std': pstats.reproducibility,
                'df': pstats.reproducibility_degf}

        # Individual columns are returned as sampled data
        for col in colnames:
            d[f'Column {col}'] = {'samples': self.model.get_column(col)}

        return d

    def get_dataset(self, name=None):
        ''' Get a DataSet from this calculation. If name is None, return list of available datasets. '''
        names = ['Columns']
        if len(self.model.colnames) > 1:
            names.append('Summarized Array')

        if name is None:
            return names

        if name in names:
            if name == 'Summarized Array':
                return self.model.to_array()
            return self.model

        else:
            raise ValueError(f'{name} not found in output')
        return names

    def get_config(self):
        ''' Get the dataset configuration dictionary '''
        d = {}
        d['mode'] = 'data'
        d['name'] = self.name
        d['colnames'] = self.model.colnames
        d['data'] = self.model.data.astype('float').tolist()
        d['desc'] = self.description

        if isinstance(self.model, DataSetSummary):
            d['nmeas'] = self.model._nmeas().astype('float').tolist()
            d['means'] = self.model._means().astype('float').tolist()
            d['stds'] = self.model._stds().astype('float').tolist()
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
            self.model = DataSetSummary(colnames, means, stds, nmeas)
        else:
            self.model = DataSet(np.array(config['data']), colnames=colnames)
        self.result = self.model
