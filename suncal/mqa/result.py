''' Results of MQA calculation '''
from typing import Optional
from dataclasses import dataclass

from ..common import reporter
from ..common.ttable import k_factor
from ..common.limit import Limit

from ..uncertainty.results.gum import GumResults
from .pdf import Pdf
from .report.system import MqaSystemReport
from .report.quantity import MqaUncertaintyReport, MqaQuantityReport


@reporter.reporter(MqaUncertaintyReport)
@dataclass
class MqaUncertaintyResult:
    ''' Uncertainty of one quantity '''
    stdev: float  # k=1
    accuracy: float
    pdf: Pdf
    degrees_freedom: float
    parent: 'MqaQuantityResult'
    gum: GumResults = None
    gumparents: list['MqaQuantityResult'] = None

    def expanded(self, conf=.95):
        ''' Expanded uncertainty '''
        k = k_factor(conf, self.degrees_freedom)
        return self.stdev * k

    def total_costs(self) -> 'MqaCostResult':
        ''' Aggregate costs up the calibration chain '''
        if self.parent:
            return self.parent.total_costs()

        if self.gumparents:
            cost = MqaCostAnnualResult()
            for parent in self.gumparents:
                cost += parent.total_costs()
            return cost

        return MqaCostAnnualResult()


@dataclass
class MqaReliabilityResult:
    ''' A reliability as PDF and percent '''
    pdf: Pdf
    pct: float


@dataclass
class MqaCapabilityResult:
    ''' TAR and TUR '''
    tar: float
    tur: float


@dataclass
class MqaEoprResult:
    ''' Calculate end-of-period reliability '''
    true: MqaReliabilityResult
    observed: MqaReliabilityResult


@dataclass
class MqaRiskResult:
    ''' False decision risk probabilities '''
    pfa_true: float
    pfr_true: float
    cpfa_true: float
    pfa_observed: float
    pfr_observed: float
    cpfa_observed: float


@dataclass
class MqaPdfs:
    ''' Distributions calculated for the MqaItem

        Args:
            ap: Adjustment process pdf
            rp: Repair process pdf
            eop: True End-of-period reliability pdf
            eop_obs: Observed End-of-period reliability pdf
            bt: Before-test pdf
            t: Observed during test pdf
            accepted_x: accepted given x
            x_accepted: x given accepted
            x_rejected: x given rejected
            pt: post-test pdf
            x_adjust: x given adjusted
            adjust_x: adjusted given x
            notadjust_x: notadjusted given x
            x_notadjust: x given notadjusted
            x_renewed: x given renewed
    '''
    ap: Optional[Pdf] = None
    rp: Optional[Pdf] = None
    bt: Optional[Pdf] = None
    t: Optional[Pdf] = None
    accepted_x: Optional[Pdf] = None
    x_accepted: Optional[Pdf] = None
    x_rejected: Optional[Pdf] = None
    pt: Optional[Pdf] = None
    x_adjust: Optional[Pdf] = None
    adjust_x: Optional[Pdf] = None
    notadjust_x: Optional[Pdf] = None
    x_notadjust: Optional[Pdf] = None
    x_renewed: Optional[Pdf] = None


@dataclass
class MqaProbs:
    ''' Probabilities calculated for the MqaItem

        Args:
            accepted: probability a DUT will be accepted
            obs_oot: Probability DUT will be observed out of tolerance
            adjust: Probability DUT will be adjusted
            repair: Probability DUT will be repaired
            notadjust: Probability DUT will not be adjusted
            accepted: Probability DUT will be accepted by test
            renewed: Probability DUT will be adjusted or repaired
    '''
    # Defaults corrleate to an always-accepted DUT
    accepted: float = 1
    obs_oot: float = 0
    adjust: float = 0
    repair: float = 0
    notadjust: float = 1
    accepted: float = 1
    renewed: float = 0


@dataclass
class MqaPeriodReliabilityResult:
    ''' Result of MQA reliability PDFs and Probabilities '''
    pdfs: MqaPdfs
    probs: MqaProbs
    bop: MqaReliabilityResult  # reliability is bop.pct
    aop: MqaReliabilityResult
    success: float


@dataclass
class MqaCostItemResult:
    ''' Calculated cost of end item performance '''
    expected: float = 0


@dataclass
class MqaCostAnnualResult:
    ''' Calculated annual costs '''
    p_available: float = 0
    ns: float = 0
    spare_cost: float = 0
    spares_year: float = 0
    cal: float = 0
    adj: float = 0
    rep: float = 0
    support: float = 0
    total: float = 0
    performance: float = 0  # For end-items only

    def __add__(self, other: 'MqaCostAnnualResult'):
        return MqaCostAnnualResult(
            0,
            0,
            self.spare_cost + other.spare_cost,
            self.spares_year + other.spares_year,
            self.cal + other.cal,
            self.adj + other.adj,
            self.rep + other.rep,
            self.support + other.support,
            self.total + other.total,
            self.performance + other.performance
        )


@reporter.reporter(MqaQuantityReport)
@dataclass
class MqaQuantityResult:
    ''' Calculated results of one MQA quantity '''
    item: 'MqaQuantity'
    testpoint: float
    tolerance: Limit
    uncertainty: MqaUncertaintyResult
    capability: MqaCapabilityResult
    eopr: MqaEoprResult
    risk: MqaRiskResult
    guardband: Limit
    reliability: MqaPeriodReliabilityResult
    cost_item: MqaCostItemResult
    cost_annual: MqaCostAnnualResult

    def total_costs(self) -> 'MqaCostAnnualResult':
        ''' Chain all the costs '''
        costs = self.cost_annual
        costs += self.uncertainty.total_costs()
        return costs


@reporter.reporter(MqaSystemReport)
@dataclass
class MqaSystemResult:
    ''' Result for a system of quantities '''
    quantities: list[MqaQuantityResult] = None
