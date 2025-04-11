''' Suncal Project Component for Measurement Quality Assurance '''
from dataclasses import asdict
from decimal import Decimal
from scipy import stats

from .component import ProjectComponent
from ..common import unitmgr
from ..common.limit import Limit
from ..uncertainty.variables import Typeb

from ..mqa.system import MqaSystem
from ..mqa.mqa import MqaQuantity, MqaMeasurand
from ..mqa.pdf import Pdf
from ..mqa.guardband import MqaGuardbandRule, MqaGuardbandRuleset, MqaGuardband
from ..mqa.measure import MqaMeasurement, MTE
from ..mqa.equipment import EquipmentList, MqaEquipmentRange
from ..mqa.costs import MqaCosts, MqaItemCost, MqaAnnualCost, MqaDowntime


class ProjectMqa(ProjectComponent):
    ''' MQA Project Component '''
    def __init__(self, model: MqaSystem = None, name: str = 'mqa'):
        super().__init__(name=name)
        if model is None:
            self.model = MqaSystem()
        else:
            self.model = model
        self._result = self.model

    def calculate(self) -> 'MqaQuantityResult':
        ''' Calculate values '''
        return self.model.calculate()

    def get_config(self):
        ''' Get System Config '''
        d = {}
        d['mode'] = 'mqa'
        d['name'] = self.name
        d['desc'] = self.description
        d['equipment'] = self.model.equipment.config()
        d['gbrules'] = config_gbrules(self.model.gbrules)
        d['quantities'] = [config_quantity(q) for q in self.model.quantities]
        return d

    def load_config(self, config):
        ''' Load configuation into the project '''
        self.name = config.get('name', '')
        self.description = config.get('desc', '')
        self.model.equipment = EquipmentList.from_config(config.get('equipment'))
        self.model.gbrules = gbrules_from_config(config.get('gbrules'))

        quantities = config.get('quantities', [])
        qtys = []
        for cfg in quantities:
            qtys.append(quantity_fromconfig(cfg, self.model.equipment, self.model.gbrules))
        self.model.quantities = qtys


def config_pdf(pdf: Pdf) -> dict:
    ''' Get configuration for Pdf '''
    return pdf.setup


def config_gbrules(rules: MqaGuardbandRuleset):
    cfg = []
    for rule in rules.rules:
        cfg.append({
            'idn': rule.idn,
            'name': rule.name,
            'method': rule.method,
            'threshold': rule.threshold,
        })
    return cfg


def config_guardband(guardband: MqaGuardband) -> dict:
    return {
        'method': guardband.method,
        'accept_limit': guardband.accept_limit.config() if guardband.accept_limit else None,
        'rule': guardband.rule.idn if guardband.rule else None
    }


def config_measurand(measurand: MqaMeasurand) -> dict:
    return {
        'name': measurand.name,
        'units': str(measurand.units),
        'description': measurand.description,
        'testpoint': measurand.testpoint,
        'eopr_pct': measurand.eopr_pct,
        'eopr_pdf': config_pdf(measurand.eopr_pdf) if measurand.eopr_pdf else None,
        'eopr_true': measurand.eopr_true,
        'tolerance': measurand.tolerance.config(),
        'degrade': measurand.degrade_limit.config() if measurand.degrade_limit else None,
        'fail': measurand.fail_limit.config() if measurand.fail_limit else None,
        'psr': measurand.psr
    }


def config_typebs(typebs: list[Typeb]) -> dict:
    ''' Get configuration for MqaUncertainty '''
    bs = []
    for b in typebs:
        uncdict = {
            'name': b.name,
            'desc': b.description,
            'degf': b.degf,
            'units': str(b.units) if b.units else None,
            'dist': b.distname,
        }
        uncdict.update({name: unitmgr.strip_units(value) for name, value in b.kwargs.items()})
        bs.append(uncdict)
    return bs


def config_mte(mte: MTE) -> dict:
    return {
        'name': mte.name,
        'quantity': config_quantity(mte.quantity) if mte.quantity else None,
        'equipment': str(mte.equipment.equipment.idn) if mte.equipment else None,
        'range': str(mte.equipment.range.idn) if mte.equipment and mte.equipment.range else None,
        'accuracy_plusminus': mte.accuracy_plusminus,
        'accuracy_eopr': mte.accuracy_eopr,
    }


def config_measurement(measure: MqaMeasurement) -> dict:
    ''' Get configuration for MqaMeasureProcess '''
    return {
        'equation': measure.equation,
        'typeb': config_typebs(measure.typebs),
        'typea': unitmgr.strip_units(measure.typea).tolist() if measure.typea is not None else None,
        'testpoints': measure.testpoints,
        'indirect': {name: config_mte(mte) for name, mte in measure.mteindirect.items()},
        'mte': config_mte(measure.mte),
        'calibration': {
            'repair_limit': measure.calibration.repair_limit.config() if measure.calibration.repair_limit else None,
            'mte_adjust': config_pdf(measure.calibration.mte_adjust) if measure.calibration.mte_adjust else None,
            'mte_repair': config_pdf(measure.calibration.mte_repair) if measure.calibration.mte_repair else None,
            'stress_pre': config_pdf(measure.calibration.stress_pre) if measure.calibration.stress_pre else None,
            'stress_post': config_pdf(measure.calibration.stress_post) if measure.calibration.stress_post else None,
            'policy': measure.calibration.policy,
            'p_discard': measure.calibration.p_discard},
        'interval': {
            'years': measure.interval.years,
            'reliability': measure.interval.reliability_model,
            'test_interval': measure.interval.test_years,
            'test_eopr': measure.interval.test_eopr,
            },
    }


def config_quantity(qty: MqaQuantity) -> dict:
    ''' Get configuration for MqaQuantity '''
    cfg = {
        'measurand': config_measurand(qty.measurand),
        'measurement': config_measurement(qty.measurement),
        'guardband': config_guardband(qty.guardband),
        'costs': asdict(qty.costs),
        'enditem': qty.enditem
    }
    return cfg


def pdf_from_config(config: dict) -> Pdf:
    ''' Make a Pdf from config '''
    mode = config.pop('from')
    if mode == 'stdev':
        return Pdf.from_stdev(**config)
    if mode == 'scipy':
        dist = getattr(stats, config.get('shape'))
        return Pdf.from_dist(
            dist(*config.get('args', []), **config.get('kwds', {}))
        )
    if mode == 'itp':
        return Pdf.from_itp(**config)
    if mode == 'fit':
        return Pdf.from_fit(config.get('fitdata'))
    if mode == 'cosine':
        return Pdf.cosine_utility(**config)
    raise ValueError


def gbrules_from_config(cfg: list[dict]) -> MqaGuardbandRuleset:
    ''' Load Guardband Ruleset from config '''
    rules = []
    for rule in cfg:
        item = MqaGuardbandRule()
        item.idn = rule.get('idn')
        item.method = rule.get('method', 'rds')
        item.name = rule.get('name', '')
        item.threshold = rule.get('threshold', 1)
        rules.append(item)
    ruleset = MqaGuardbandRuleset()
    ruleset.rules = rules
    return ruleset


def guardband_from_config(config: dict, rules: MqaGuardbandRuleset) -> MqaGuardband:
    ''' Load guardband from config '''
    ruleid = config.get('rule')
    new = MqaGuardband()
    new.method = config.get('method')
    limit_cfg = config.get('accept_limit')
    new.accept_limit = Limit.from_config(limit_cfg) if limit_cfg else None
    newrule = rules.locate(ruleid)
    if newrule:
        new.rule = newrule
    return new


def measurand_fromconfig(config: dict) -> MqaMeasurand:
    ''' Load MqaMeasurand from config '''
    new = MqaMeasurand()
    new.name = config.get('name')
    new.units = config.get('units')
    new.description = config.get('description')
    new.testpoint = config.get('testpoint', 0)
    new.eopr_pct = config.get('eopr_pct')
    new.eopr_true = config.get('eopr_true')
    new.psr = config.get('psr')
    if (c := config.get('eopr_pdf')):
        new.eopr_pdf = pdf_from_config(c)
    if (c := config.get('tolerance')):
        new.tolerance = Limit.from_config(c)
    if (c := config.get('degrade')):
        new.degrade_limit = Limit.from_config(c)
    if (c := config.get('fail')):
        new.fail_limit = Limit.from_config(c)
    return new


def typebs_fromconfig(config: list[dict]) -> list[Typeb]:
    ''' Load typeb uncerts from config '''
    new = []
    for typeb in config:
        desc = typeb.pop('desc', None)
        dist = typeb.pop('dist', 'normal')
        new.append(Typeb(dist, description=desc, **typeb))
    return new


def mte_fromconfig(config: dict, equipment: EquipmentList, gbrules: MqaGuardbandRuleset, childqty: MqaQuantity) -> MTE:
    ''' Load MTE from config '''
    new = MTE()
    if (qty := config.get('quantity')):
        new.quantity = quantity_fromconfig(qty, equipment, gbrules)
        new.quantity.child = childqty
    if (equipid := config.get('equipment')):
        rngid = config.get('range')
        new.equipment = MqaEquipmentRange()
        new.equipment.equipment = equipment.locate(equipid)
        if rngid:
            new.equipment.range, new.equipment.equipment = equipment.locate_range(rngid)
    new.accuracy_plusminus = config.get('accuracy_plusminus')
    new.accuracy_eopr = config.get('accuracy_eopr')
    new.name = config.get('name', '')
    return new


def meas_fromconfig(config: dict, equipment: EquipmentList, gbrules: MqaGuardbandRuleset, qty: MqaQuantity) -> MqaMeasurement:
    ''' Measurement from config '''
    new = MqaMeasurement()
    new.equation = config.get('equation')
    new.typebs = typebs_fromconfig(config.get('typeb'))
    new.typea = config.get('typea')
    new.testpoints = config.get('testpoints', {})
    new.mteindirect = {name: mte_fromconfig(mte, equipment, gbrules, qty) for name, mte in config.get('indirect', {}).items()}
    new.mte = mte_fromconfig(config.get('mte'), equipment, gbrules, qty)

    cal = config.get('calibration', {})
    if (c := cal.get('repair_limit')):
        new.calibration.repair_limit = Limit.from_config(c)
    if (c := cal.get('mte_adjust')):
        new.calibration.mte_adjust = pdf_from_config(c)
    if (c := cal.get('mte_repair')):
        new.calibration.mte_repair = pdf_from_config(c)
    if (c := cal.get('stress_pre')):
        new.calibration.stress_pre = pdf_from_config(c)
    if (c := cal.get('stress_post')):
        new.calibration.stress_post = pdf_from_config(c)
    new.calibration.policy = cal.get('policy')
    new.calibration.p_discard = cal.get('p_discard')

    intv = config.get('interval', {})
    new.interval.years = intv.get('years', 1)
    new.interval.test_years = intv.get('test_years', None)
    new.interval.test_eopr = intv.get('test_eopr', None)
    new.interval.reliability_model = intv.get('reliability', 'none')
    return new


def quantity_fromconfig(config: dict, equipment: EquipmentList, gbrules: MqaGuardbandRuleset) -> MqaQuantity:
    ''' MQA Quantity from config '''
    new = MqaQuantity()
    new.measurand = measurand_fromconfig(config.get('measurand'))
    new.measurement = meas_fromconfig(config.get('measurement'), equipment, gbrules, new)
    new.guardband = guardband_from_config(config.get('guardband'), gbrules)
    costcfg = config.get('costs', {})
    costannual = costcfg.get('annual', {})
    new.costs = MqaCosts()
    new.costs.item = MqaItemCost(**costcfg.get('item'))
    new.costs.annual = MqaAnnualCost(**costannual)
    new.costs.annual.downtime = MqaDowntime(**costannual.get('downtime'))
    new.enditem = config.get('enditem', True)
    return new
