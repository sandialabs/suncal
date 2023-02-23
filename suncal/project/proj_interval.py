''' Project components for interval calculations '''

from .component import ProjectComponent
from ..intervals import (TestInterval, TestIntervalAssets, BinomialInterval, BinomialIntervalAssets,
                         VariablesInterval, VariablesIntervalAssets)


class ProjectIntervalTest(ProjectComponent):  # A3
    ''' A3 Interval project component '''
    def __init__(self, model=None, name='intervaltest'):
        super().__init__()
        self.name = name
        if model is not None:
            self.model = model
        else:
            self.model = TestInterval()
        self.result = None
        self.longdescription = None
        self.project = None  # Parent project

    def calculate(self):
        ''' Calculate the interval '''
        self.result = self.model.calculate()
        return self.result

    def get_config(self):
        ''' Get configuration dictionary '''
        d = {}
        d['mode'] = 'intervaltest'
        d['name'] = self.name
        d['desc'] = self.longdescription
        d['intol'] = self.model.intol
        d['total'] = self.model.n
        d['I0'] = self.model.I0
        d['target'] = self.model.Rtarget
        d['maxchange'] = self.model.maxchange
        d['conf'] = self.model.conf
        d['mindelta'] = self.model.mindelta
        d['minint'] = self.model.minint
        d['maxint'] = self.model.maxint
        return d

    def load_config(self, config):
        ''' Load interval object from config dictionary '''
        self.name = config.get('name', 'interval')
        self.longdescription = config.get('desc', '')
        self.model.intol = config.get('intol', 0)
        self.model.n = config.get('total', 0)
        self.model.I0 = config.get('I0', 365)
        self.model.Rtarget = config.get('Rtarget', .95)
        self.model.maxchange = config.get('maxchange', 2)
        self.model.conf = config.get('conf', .5)
        self.model.mindelta = config.get('mindelta', 5)
        self.model.minint = config.get('minint', 14)
        self.model.maxint = config.get('maxint', 1826)


class ProjectIntervalTestAssets(ProjectComponent):
    ''' A3 Project Component - from Asset data '''
    def __init__(self, model=None, name='intervaltestasset'):
        super().__init__()
        self.name = name
        if model is not None:
            self.model = model
        else:
            self.model = TestIntervalAssets()
        self.result = None
        self.longdescription = None
        self.project = None  # Parent project

    def calculate(self):
        ''' Calculate the interval '''
        self.result = self.model.calculate()
        return self.result

    def get_config(self):
        ''' Get configuration dictionary '''
        d = {}
        d['mode'] = 'intervaltestasset'
        d['name'] = self.name
        d['desc'] = self.longdescription
        d['I0'] = self.model.I0
        d['target'] = self.model.Rtarget
        d['tol'] = self.model.tol
        d['thresh'] = self.model.thresh
        d['maxchange'] = self.model.maxchange
        d['conf'] = self.model.conf
        d['mindelta'] = self.model.mindelta
        d['minint'] = self.model.minint
        d['maxint'] = self.model.maxint
        d['assets'] = {}
        for a, vals in self.model.assets.items():
            d['assets'][a] = {
                'startdates': list(vals['startdates']) if vals['startdates'] is not None else None,
                'enddates': list(vals['enddates']) if vals['enddates'] is not None else None,
                'passfail': list(vals['passfail']) if vals['passfail'] is not None else None}
        return d

    def load_config(self, config):
        ''' Load interval object from config dictionary '''
        self.name = config.get('name', 'interval')
        self.longdescription = config.get('desc', '')
        self.model.assets = config.get('assets', {})
        self.model.I0 = config.get('I0', 365)
        self.model.Rtarget = config.get('Rtarget', .95)
        self.model.tol = config.get('tol', 56)
        self.model.thresh = config.get('thresh', .5)
        self.model.maxchange = config.get('maxchange', 2)
        self.model.conf = config.get('conf', .5)
        self.model.mindelta = config.get('mindelta', 5)
        self.model.minint = config.get('minint', 14)
        self.model.maxint = config.get('maxint', 1826)


class ProjectIntervalBinom(ProjectComponent):
    ''' S2 Interval project component '''
    def __init__(self, model=None, name='intervalbinom'):
        super().__init__()
        self.name = name
        if model is not None:
            self.model = model
        else:
            self.model = BinomialInterval()
        self.result = None
        self.longdescription = None
        self.project = None  # Parent project

    def calculate(self):
        ''' Calculate the interval '''
        self.result = self.model.calculate()
        return self.result

    def get_config(self):
        ''' Get configuration dictionary '''
        d = {}
        d['mode'] = 'intervalbinom'
        d['name'] = self.name
        d['desc'] = self.longdescription
        d['Rt'] = self.model.Rtarget
        d['ti'] = list(self.model.ti)
        d['ri'] = list(self.model.Ri)
        d['ni'] = list(self.model.ni)
        return d

    def load_config(self, config):
        ''' Load interval object from config dictionary '''
        self.name = config.get('name', 'interval')
        self.longdescription = config.get('desc', '')
        self.model.Rtarget = config.get('Rt', .95)
        self.model.Ri = config.get('ri', [])
        self.model.ni = config.get('ni', [])
        self.model.ti = config.get('ti', [])


class ProjectIntervalBinomAssets(ProjectComponent):
    ''' S2 interval project component - from asset data '''
    def __init__(self, model=None, name='intervalbinomasset'):
        super().__init__()
        self.name = name
        if model is not None:
            self.model = model
        else:
            self.model = BinomialIntervalAssets()
        self.result = None
        self.longdescription = None
        self.project = None  # Parent project

    def calculate(self):
        ''' Calculate the interval '''
        self.result = self.model.calculate()
        return self.result

    def get_config(self):
        ''' Get configuration dictionary '''
        d = {}
        d['mode'] = 'intervalbinomasset'
        d['name'] = self.name
        d['desc'] = self.longdescription
        d['target'] = self.model.Rtarget
        d['bins'] = self.model.bins
        d['binlefts'] = self.model.binlefts
        d['binwidth'] = self.model.binwidth
        d['assets'] = {}
        for a, vals in self.model.assets.items():
            d['assets'][a] = {
                'startdates': list(vals['startdates']) if vals['startdates'] is not None else None,
                'enddates': list(vals['enddates']) if vals['enddates'] is not None else None,
                'passfail': list(vals['passfail']) if vals['passfail'] is not None else None}
        return d

    def load_config(self, config):
        ''' Load interval object from config dictionary '''
        self.name = config.get('name', 'interval')
        self.longdescription = config.get('desc', '')
        self.model.Rtarget = config.get('target', .9)
        self.model.bins = config.get('bins', 10)
        self.model.binlefts = config.get('binlefts', None)
        self.model.binwidth = config.get('binwidth', None)
        self.model.assets = config.get('assets', {})


class ProjectIntervalVariables(ProjectComponent):
    ''' Variables interval project component '''
    def __init__(self, model=None, name='intervalvariables'):
        super().__init__()
        self.name = name
        if model is not None:
            self.model = model
        else:
            self.model = VariablesInterval()
        self.result = None
        self.longdescription = None
        self.project = None  # Parent project

    def calculate(self):
        ''' Calculate the interval '''
        self.result = self.model.calculate()
        return self.result

    def get_config(self):
        ''' Get configuration dictionary '''
        d = {}
        d['mode'] = 'intervalvariables'
        d['name'] = self.name
        d['desc'] = self.longdescription
        d['u0'] = self.model.u0
        d['y0'] = self.model.y0
        d['kvalue'] = self.model.kvalue
        d['maxm'] = self.model.maxm
        d['m'] = self.model.m
        d['utarget'] = self.model.utarget
        d['rlimits'] = list(self.model.rlimits)
        d['rconf'] = self.model.rconf
        d['dt'] = list(self.model.t)
        d['deltas'] = list(self.model.deltas)
        return d

    def load_config(self, config):
        ''' Load the configuration '''
        self.name = config.get('name', 'interval')
        self.longdescription = config.get('desc', '')
        self.model.t = config.get('dt', [])
        self.model.deltas = config.get('deltas', [])
        self.model.u0 = config.get('u0', 0)
        self.model.y0 = config.get('y0', 0)
        self.model.m = config.get('m', None)
        self.model.maxm = config.get('maxm', 1)
        self.model.utarget = config.get('utarget', 0.5)
        self.model.rlimits = config.get('rlimits', (-1, 1))
        self.model.rconf = config.get('rconf', 0.95)
        self.model.kvalue = config.get('kvalue', 1)


class ProjectIntervalVariablesAssets(ProjectComponent):
    ''' Variables interval project component - from asset data '''
    def __init__(self, model=None, name='intervalvariablesasset'):
        super().__init__()
        self.name = name
        if model is not None:
            self.model = model
        else:
            self.model = VariablesIntervalAssets()
        self.result = None
        self.longdescription = None
        self.project = None  # Parent project

    def calculate(self):
        ''' Calculate the interval '''
        self.result = self.model.calculate()
        return self.result

    def get_config(self):
        ''' Get configuration dictionary '''
        d = {}
        d['mode'] = 'intervalvariablesasset'
        d['name'] = self.name
        d['desc'] = self.longdescription
        d['u0'] = self.model.u0
        d['kvalue'] = self.model.kvalue
        d['y0'] = self.model.y0
        d['maxm'] = self.model.maxm
        d['m'] = self.model.m
        d['utarget'] = self.model.utarget
        d['rlimits'] = list(self.model.rlimits)
        d['rconf'] = self.model.rconf
        d['assets'] = {}
        for a, vals in self.model.assets.items():
            d['assets'][a] = {
                'startdates': list(vals['startdates']) if vals['startdates'] is not None else None,
                'enddates': list(vals['enddates']) if vals['enddates'] is not None else None,
                'asleft': list(vals['asleft']) if vals['asleft'] is not None else None,
                'asfound': list(vals['asfound']) if vals['asfound'] is not None else None}
        return d

    def load_config(self, config):
        ''' Load the configuration '''
        self.name = config.get('name', 'interval')
        self.longdescription = config.get('desc', '')
        self.model.u0 = config.get('u0', 0)
        self.model.y0 = config.get('y0', 0)
        self.model.m = config.get('m', None)
        self.model.maxm = config.get('maxm', 1)
        self.model.utarget = config.get('utarget', 0.5)
        self.model.rlimits = config.get('rlimits', (-1, 1))
        self.model.rconf = config.get('rconf', 0.95)
        self.model.kvalue = config.get('kvalue', 1)
        self.model.assets = config.get('assets', {})
