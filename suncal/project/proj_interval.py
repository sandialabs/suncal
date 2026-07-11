''' Project components for interval calculations '''

from dataclasses import asdict
from .component import ProjectComponent
from ..intervals import (A3Params, a3_testinterval,
                         S2Params, s2_binom_interval,
                         VariablesData,
                         VariablesReliabilityTarget,
                         VariablesUncertaintyTarget)
from ..intervals.binoms2 import get_passfails


class ProjectIntervalTest(ProjectComponent):  # A3
    ''' A3 Interval project component '''
    def __init__(self, name='intervaltest'):
        super().__init__(name=name)
        self.params: A3Params = A3Params()

    def get_config(self):
        config = super().get_config()
        config['mode'] = 'intervaltest'
        config['params'] = asdict(self.params)
        return config

    def load_config(self, config):
        ''' Load configration into Component '''
        self.description = config.get('description', config.get('desc'))
        self.params = A3Params(**config.get('params'))
        self.parent = config.get('parent')

    @classmethod
    def from_config(cls, config):
        new = cls()
        new.load_config(config)
        return new

    def calculate(self):
        ''' Calculate the interval '''
        self._result = a3_testinterval(self.params)
        return self._result


class ProjectIntervalTestAssets(ProjectComponent):
    ''' A3 Project Component - from Asset data '''
    def __init__(self, name='intervaltestasset'):
        super().__init__(name=name)
        self.assets: dict[str, dict[str, float]] = {'A': {'startdates': [],
                                                          'enddates': [],
                                                          'passfail': []}}
        # assetname: {startdates: x, enddates: x, passfail: x}
        self.params: A3Params = A3Params()
        self.tolerance = 56
        self.thresh = 0.5

    def get_config(self):
        config = super().get_config()
        config['mode'] = 'intervaltestasset'
        config['params'] = asdict(self.params)
        config['assets'] = self.assets
        config['tolerance'] = self.tolerance
        config['thresh'] = self.thresh
        return config

    def load_config(self, config):
        ''' Load configration into Component '''
        self.description = config.get('description', config.get('desc'))
        self.assets = config.get('assets', {})
        self.tolerance = config.get('tolerance', 56)
        self.thresh = config.get('thresh', 0.5)
        params = config.get('params', {})
        params.pop('intol', None)  # These are calculated from the assets list
        params.pop('n', None)
        self.params = A3Params.from_assets(self.assets.values(),
                                           tolerance=self.tolerance,
                                           threshold=self.thresh,
                                           **params)

    @classmethod
    def from_config(cls, config):
        new = cls(name=config.get('name'))
        new.load_config(config)
        return new

    def calculate(self):
        ''' Calculate the interval '''
        self._result = a3_testinterval(self.params)
        return self._result


class ProjectIntervalBinom(ProjectComponent):
    ''' S2 Interval project component '''
    def __init__(self, name='intervalbinom'):
        super().__init__(name=name)
        self.params: S2Params = S2Params()
        self.conf = 0.95  # Confidence for reporting uncertainty in interval

    def get_config(self):
        config = super().get_config()
        config['mode'] = 'intervalbinom'
        config['Rt'] = self.params.target
        config['ti'] = self.params.ti
        config['ti0'] = self.params.ti0
        config['ri'] = self.params.ri
        config['ni'] = self.params.ni
        config['conf'] = self.conf
        return config

    def load_config(self, config):
        ''' Load configration into Component '''
        self.params = S2Params(
            target=config.get('Rt', .95),
            ti=config.get('ti', []),
            ri=config.get('ri', []),
            ni=config.get('ni', []),
            ti0=config.get('ti0', []))
        self.conf = config.get('conf', 0.95)
        self.description = config.get('description', config.get('desc'))
        self.parent = config.get('parent')

    @classmethod
    def from_config(cls, config):
        new = cls(name=config.get('name'))
        new.load_config(config)
        return new

    def calculate(self):
        ''' Calculate the interval '''
        self._result = s2_binom_interval(self.params, conf=self.conf)
        return self._result


class ProjectIntervalBinomAssets(ProjectComponent):
    ''' S2 interval project component - from asset data '''
    def __init__(self, name='intervalbinomasset'):
        super().__init__(name=name)
        self.assets: dict[str, dict[str, float]] = {'A': {'startdates': [],
                                                          'enddates': [],
                                                          'passfail': []}}
            # assetname: {startdates: x, enddates: x, passfail: x}
        self.target = 0.95
        self.bins = 10
        self.binlefts: list[float] = None
        self.binwidth: float = None
        self.conf = 0.95  # Confidence for reporting uncertainty in interval

    def get_config(self):
        config = super().get_config()
        config['mode'] = 'intervalbinomasset'
        config['target'] = self.target
        config['assets'] = self.assets
        config['bins'] = self.bins
        config['binlefts'] = self.binlefts
        config['binwidth'] = self.binwidth
        return config

    def load_config(self, config):
        ''' Load configration into Component '''
        self.description = config.get('description', config.get('desc'))
        self.target = config.get('target', 0.95)
        self.assets = config.get('assets', {})
        self.bins = config.get('bins', 10)
        self.binlefts = config.get('binlefts')
        self.binwidth = config.get('binwidth')
        self.parent = config.get('parent')

    @property
    def params(self):
        return S2Params.from_assets(self.assets.values(),
                                    self.target,
                                    self.bins,
                                    self.binlefts,
                                    self.binwidth)

    def passfails(self, asset: str):
        ''' Get list of t, pass/fail for one asset '''
        return get_passfails(self.assets.get(asset))

    @classmethod
    def from_config(cls, config):
        new = cls(name=config.get('name'))
        new.load_config(config)
        return new

    def calculate(self):
        ''' Calculate the interval '''
        self._result = s2_binom_interval(self.params)
        return self._result


class ProjectIntervalVariables(ProjectComponent):
    ''' Variables interval pro:ject component '''
    def __init__(self, name='intervalvariables'):
        super().__init__(name=name)
        self.data: VariablesData = VariablesData([], [])
        self._mode_reliability: bool = True  # Calculate reliability target or uncertainty target?
        self.utarget = 0.5
        self.tol_lo = -1
        self.tol_hi = 1
        self.reliability_target = 0.95
        self.order = 1
        self.maxorder = 1
        self._result = None

    @property
    def result(self):
        if self._result is None:
            if self._mode_reliability:
                return self._calculate_reliability_target()
            return self._calculate_uncertainty_target()
        return self._result

    def get_config(self):
        config = super().get_config()
        config['mode'] = 'intervalvariables'
        config['deltas'] = self.data.deltas.tolist()
        config['dt'] = self.data.dt.tolist()
        config['u0'] = self.data.u0
        config['u0_degf'] = self.data.u0_degf
        config['y0'] = self.data.y0
        config['m'] = self.order
        config['maxm'] = self.maxorder
        if self._mode_reliability:
            config['tolerance'] = [self.tol_lo, self.tol_hi]
            config['reliability_target'] = self.reliability_target
        else:
            config['utarget'] = self.utarget
        return config

    def load_config(self, config):
        ''' Load configration into Component '''
        self.description = config.get('description', config.get('desc'))
        self.data = VariablesData(
            dt=config.get('dt', []),
            deltas=config.get('deltas', []),
            u0=config.get('u0', 0),
            u0_degf=config.get('u0_degf', float('inf')),
            y0=config.get('y0', 0))

        self._mode_reliability = 'utarget' not in config
        self.utarget = config.get('utarget')
        self.tol_lo = config.get('tolerance', config.get('rlimits', (0, 0)))[0]
        self.tol_hi = config.get('tolerance', config.get('rlimits', (0, 0)))[1]
        self.reliability_target = config.get('reliability_target', config.get('rconf', .95))
        self.order = config.get('m')
        self.maxorder = config.get('maxm')

    @classmethod
    def from_config(cls, config):
        new = cls(name=config.get('name'))
        new.load_config(config)
        return new

    def _calculate_reliability_target(self):
        ''' Calculate the interval '''
        calc = VariablesReliabilityTarget(
            self.data.dt,
            self.data.deltas,
            self.data.u0,
            self.data.y0,
            self.data.u0_degf)
        self._result = calc.calculate(
            self.tol_lo, self.tol_hi, self.reliability_target,
            self.order, self.maxorder)
        return self._result

    def _calculate_uncertainty_target(self):
        ''' Calculate the interval '''
        calc = VariablesUncertaintyTarget(
            self.data.dt,
            self.data.deltas,
            self.data.u0,
            self.data.y0,
            self.data.u0_degf)
        self._result = calc.calculate(self.utarget, self.order, self.maxorder)
        return self._result

    def calculate(self):
        ''' Calculate '''
        if self._mode_reliability:
            return self._calculate_reliability_target()
        return self._calculate_uncertainty_target()


class ProjectIntervalVariablesAssets(ProjectComponent):
    ''' Variables interval project component - from asset data '''
    def __init__(self, name='intervalvariablesasset'):
        super().__init__(name=name)
        self.assets: dict[str, dict[str, float]] = {'A': {'startdates': [],
                                                          'enddates': [],
                                                          'passfail': []}}
        self.data = VariablesData(dt=[], deltas=[])
        self._mode_reliability: bool = True  # Calculate reliability target or uncertainty target?
        self.utarget = 0.5
        self.tol_lo = -1
        self.tol_hi = 1
        self.reliability_target = 0.95
        self.order = 1
        self.maxorder = 1
        self._result = None

    @property
    def result(self):
        if self._result is None:
            if self._mode_reliability:
                return self._calculate_reliability_target()
            return self._calculate_uncertainty_target()
        return self._result

    def get_config(self):
        config = super().get_config()
        config['mode'] = 'intervalvariablesasset'
        config['assets'] = self.assets
        config['m'] = self.order
        config['maxm'] = self.maxorder
        if self._mode_reliability:
            config['tolerance'] = [self.tol_lo, self.tol_hi]
            config['reliability_target'] = self.reliability_target
        else:
            config['utarget'] = self.utarget
        return config

    def load_config(self, config):
        ''' Load configration into Component '''
        self.description = config.get('description', config.get('desc'))
        self.assets = config.get('assets', {})
        self._mode_reliability = 'utarget' not in config
        self.utarget = config.get('utarget')
        self.tol_lo = config.get('tolerance', config.get('rlimits', (0, 0)))[0]
        self.tol_hi = config.get('tolerance', config.get('rlimits', (0, 0)))[1]
        self.reliability_target = config.get('reliability_target', config.get('rconf', .95))
        self.order = config.get('m')
        self.maxorder = config.get('maxm')
        self.data = VariablesData.from_assets(
            self.assets.values(),
            u0=config.get('u0', 0),
            y0=config.get('y0', 0),
        )

    @classmethod
    def from_config(cls, config):
        new = cls(name=config.get('name'))
        new.load_config(config)
        return new

    def _calculate_reliability_target(self):
        ''' Calculate the interval '''
        calc = VariablesReliabilityTarget(
            self.data.dt,
            self.data.deltas,
            self.data.u0,
            self.data.y0,
            self.data.u0_degf)
        self._result = calc.calculate(
            self.tol_lo, self.tol_hi, self.reliability_target,
            self.order, self.maxorder)
        return self._result

    def _calculate_uncertainty_target(self):
        ''' Calculate the interval '''
        calc = VariablesUncertaintyTarget(
            self.data.dt,
            self.data.deltas,
            self.data.u0,
            self.data.y0,
            self.data.u0_degf)
        self._result = calc.calculate(self.utarget, self.order, self.maxorder)
        return self._result

    def calculate(self):
        ''' Calculate '''
        if self._mode_reliability:
            return self._calculate_reliability_target()
        return self._calculate_uncertainty_target()
