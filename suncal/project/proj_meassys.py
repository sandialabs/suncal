''' Suncal Project Component for Data Evaluation '''
import numpy as np

from ..common import unitmgr
from ..common.limit import Limit
from .component import ProjectComponent
from ..uncertainty.variables import Typeb
from ..uncertainty.report.units import units_report
from ..meassys.meassys import MeasureSystem, SystemQuantity, SystemIndirectQuantity
from ..meassys.curve import SystemCurve


class ProjectMeasSys(ProjectComponent):
    ''' Measured Data Evaluation Project Component '''
    def __init__(self, name: str = 'system'):
        super().__init__(name=name)
        self.model = MeasureSystem()
        self._result = self.model

    def calculate(self) -> 'SystemResult':
        ''' Calculate values '''
        return self.model.calculate()

    def units_report(self, **kwargs):
        ''' Create report showing units of all parameters '''
        model, outunits = self.model.gummodel()
        return units_report(model, outunits, **kwargs)

    def _qty_config(self, qty):
        ''' Get config for one quantity in the system '''
        if isinstance(qty, SystemQuantity):
            qcfg = {
                'type': 'direct',
                'name': qty.name,
                'symbol': qty.symbol,
                'desc': qty.description,
                'testpoint': qty.testpoint,
                'typeb': [config_typeb(typeb) for typeb in qty.typebs],
                'typea': qty.typea.tolist() if qty.typea is not None else None,
                'tolerance': qty.tolerance.config() if qty.tolerance else None,
                'units': qty.units,
                'description': qty.description,
            }
        elif isinstance(qty, SystemIndirectQuantity):
            qcfg = {
                'type': 'indirect',
                'name': qty.name,
                'symbol': qty.symbol,
                'desc': qty.description,
                'equation': qty.equation,
                'tolerance': qty.tolerance.config() if qty.tolerance else None,
                'units': qty.outunits,
                'description': qty.description,
            }
        else:  # SystemCurve
            assert isinstance(qty, SystemCurve)
            qcfg = {
                'type': 'curve',
                'fitmodel': qty.fitmodel,
                'predictor': qty.predictor_var,
                'response': qty.response_var,
                'units': {name: str(u) for name, u in qty.units.items()},
                'odr': qty.odr,
                'guess': qty.guess,
                'tolerances': {name: tol.config() for name, tol in qty.tolerances.items()},
                'predictions': qty.predictions,
                'data': {name: values for name, values in qty.data},
                'description': qty.description,
                'descriptions': qty.descriptions,
            }
        return qcfg

    def get_config(self):
        d = {}
        d['mode'] = 'system'
        d['name'] = self.name
        d['desc'] = self.description
        d['seed'] = self.model.seed
        d['confidence'] = self.model.confidence
        d['correlate'] = self.model.correlate_typeas
        d['correlations'] = self.model.correlations
        d['samples'] = self.model.samples
        d['quantities'] = []
        for qty in self.model.quantities:
            d['quantities'].append(self._qty_config(qty))
        return d

    def _qty_from_config(self, qcfg: dict):
        ''' Create one quantity from config '''
        qtytype = qcfg.get('type')
        if qtytype == 'direct':
            qty = SystemQuantity()
            qty.name = qcfg.get('name')
            qty.symbol = qcfg.get('symbol')
            qty.description = qcfg.get('desc')
            qty.testpoint = qcfg.get('testpoint')
            qty.units = qcfg.get('units')
            qty.typebs = [typeb_from_config(bcfg, qty.testpoint) for bcfg in qcfg.get('typeb', []) if bcfg is not None]
            typea = qcfg.get('typea')
            if typea is not None:
                typea = np.asarray(typea)
            qty.typea = typea
            qty.tolerance = Limit.from_config(qcfg.get('tolerance'))

        elif qtytype == 'indirect':
            qty = SystemIndirectQuantity()
            qty.name = qcfg.get('name')
            qty.symbol = qcfg.get('symbol')
            qty.description = qcfg.get('desc')
            qty.equation = qcfg.get('equation')
            qty.tolerance = Limit.from_config(qcfg.get('tolerance'))
            qty.outunits = qcfg.get('units')

        else:
            assert qtytype == 'curve'
            qty = SystemCurve()
            qty.set_fitmodel(qcfg.get('fitmodel'))
            qty.predictor_var = qcfg.get('predictor', 'x')
            qty.response_var = qcfg.get('response', 'y')
            qty.odr = qcfg.get('odr', False)
            qty.guess = qcfg.get('guess')
            qty.tolerances = {name: Limit.from_config(tol) for name, tol in qcfg.get('tolerances')}
            qty.predictions = qcfg.get('predictions', {})
            qty.units = qcfg.get('units', {})
            data = qcfg.get('data')
            qty.data = [(name, data) for name, data in data.items()]
        return qty

    def load_config(self, config: dict):
        ''' Load configuation into the project '''
        self.name = config.get('name', '')
        self.description = config.get('desc', '')
        self.model = MeasureSystem()
        self.model.seed = config.get('seed', None)
        self.model.confidence = config.get('confidence', .95)
        self.model.samples = config.get('samples', 1000000)
        self.model.correlate_typeas = config.get('correlate', True)
        self.model.correlations = config.get('correlations', [])
        for qcfg in config.get('quantities', []):
            qty = self._qty_from_config(qcfg)
            qty.quantities = self.model.quantities
            self.model.quantities.append(qty)

    def duplicate(self, qty):
        ''' Duplicate the quantity '''
        cfg = self._qty_config(qty)
        new = self._qty_from_config(cfg)
        new.quantities = self.model.quantities
        self.model.quantities.append(new)


def config_typeb(typeb) -> dict:
    ''' Get configuration for TypeB '''
    cfg = {
        'name': typeb.name,
        'desc': typeb.description,
        'degf': typeb.degf,
        'units': str(typeb.units) if typeb.units else None,
        'dist': typeb.distname,
    }
    cfg.update({name: unitmgr.strip_units(value) for name, value in typeb.kwargs.items()})
    return cfg


def typeb_from_config(config: dict, nominal: float) -> Typeb:
    ''' Make an MqaUncertainty from config '''
    desc = config.pop('desc', None)
    dist = config.pop('dist', 'normal')
    return Typeb(dist, nominal=nominal, description=desc, **config)
