''' Guardbanding Rules '''
from typing import Literal
from dataclasses import dataclass, field
from uuid import uuid4

from ..common.limit import Limit


GbOption = Literal['auto', 'manual', 'none']
GbMethod = Literal['rds', 'rp10', 'u95', 'dobbert', 'pfa', 'cpfa']


@dataclass
class MqaGuardbandRule:
    ''' A guardbanding rule, including method and thresholds for applicability '''
    idn: str = field(default_factory=lambda: uuid4().hex)
    name: str = '4:1 RDS'
    method: GbMethod = 'rds'
    threshold: float = None

    def __post_init__(self):
        if self.threshold is None:
            self.threshold = .02 if self.method in ['pfa', 'cpfa', 'specific'] else 4.


def default_rules() -> list[MqaGuardbandRule]:
    ''' Create default guardbanding rules '''
    return [
        MqaGuardbandRule(),
        MqaGuardbandRule(name='2% PFA', method='pfa', threshold=2)
    ]


@dataclass
class MqaGuardbandRuleset:
    ''' Guardbanding rules for different measurements '''
    rules: list[MqaGuardbandRule] = field(default_factory=default_rules)

    def locate(self, idn: str) -> MqaGuardbandRule:
        ''' Find a rule by ID '''
        for rule in self.rules:
            if rule.idn == idn:
                return rule
        return None

    @property
    def rule_names(self):
        ''' List of guardband rule names '''
        return [rule.name for rule in self.rules]

    def by_name(self, name: str) -> MqaGuardbandRule:
        ''' Locate a rule using its name '''
        idx = self.rule_names.index(name)
        return self.rules[idx]


@dataclass
class MqaGuardband:
    ''' Guardbanding method and limit '''
    method: GbOption = 'auto'
    rule: MqaGuardbandRule = field(default_factory=MqaGuardbandRule)
    accept_limit: Limit = None

    def __str__(self):
        if self.method == 'auto':
            return self.rule.name
        elif self.method == 'none':
            return 'No Guardband'
        return str(self.accept_limit)
