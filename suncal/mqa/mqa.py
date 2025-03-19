''' Measurement Quality Assurance '''
from typing import Sequence
from decimal import Decimal
from dataclasses import dataclass, field
from scipy.integrate import simpson
from scipy import stats
import numpy as np

from ..common.limit import Limit
from .pdf import Pdf
from .costs import MqaCosts
from .risks import pfa, pfr, cpfa, target_pfa, target_cpfa
from .measure import MqaMeasurement
from .equipment import EquipmentList
from .guardband import MqaGuardband
from .result import (
    MqaUncertaintyResult,
    MqaQuantityResult,
    MqaCapabilityResult,
    MqaEoprResult,
    MqaRiskResult,
    MqaPdfs,
    MqaProbs,
    MqaReliabilityResult,
    MqaPeriodReliabilityResult,
    MqaCostItemResult,
    MqaCostAnnualResult,
    )


@dataclass
class MqaMeasurand:
    ''' Common info about the measurand '''
    name: str = 'Measurand'
    units: str = ''
    description: str = ''
    testpoint: float = 0
    eopr_pct: float = 0.9067
    eopr_pdf: Pdf = None
    eopr_true: bool = False  # eopr is TRUE (True) or OBSERVED (False)
    tolerance: Limit = field(default_factory=lambda: Limit.from_plusminus(Decimal('0.0'), Decimal('1.0')))
    degrade_limit: Limit = None
    fail_limit: Limit = None
    psr: float = 1

    @property
    def eopr(self) -> tuple[Pdf, float, bool]:
        ''' Get observed EOPR as Pdf and Percent

            Returns:
                EOPR PDF
                EOPR percent
                Whether EOPR is True vs Observed (False)
        '''
        if self.eopr_pdf:
            obs_pdf = self.eopr_pdf
            obs_pct = obs_pdf.itp(self.tolerance)
        else:
            obs_pdf = Pdf.from_itp(self.testpoint, self.eopr_pct, self.tolerance)
            obs_pct = self.eopr_pct
        return obs_pdf, obs_pct, self.eopr_true

    def utility(self) -> Pdf:
        ''' Get Utility PDF from degrade and fail points as cosine distribution '''
        if self.degrade_limit is not None and self.fail_limit is not None:
            return Pdf.cosine_utility(self.degrade_limit, self.fail_limit)
        return Pdf.step(self.tolerance.flow, self.tolerance.fhigh)


class MqaQuantity:
    ''' One Quantity in an MQA System '''

    class Mode:
        ''' What features to calculate '''
        BASIC = 0
        RELIABILITY = 1
        COSTS = 2

    def __init__(self):
        super().__init__()
        self.measurand: MqaMeasurand = MqaMeasurand()
        self.measurement: MqaMeasurement = MqaMeasurement()
        self.guardband: MqaGuardband = MqaGuardband()
        self.costs: MqaCosts = MqaCosts()
        self.enditem: bool = True
        self.result: MqaQuantityResult = None
        self.equipmentlist: EquipmentList = None
        self.child: 'MqaQuantity' = None  # This qty is used to calibrate another

    @property
    def mqa_mode(self) -> str:
        ''' Determine what features enabled (reliability decay or costs) '''
        if self.costs.enabled:
            return MqaQuantity.Mode.COSTS
        if self.measurement.interval.reliability_model != 'none' or self.measurement.calibration.policy != 'never':
            return MqaQuantity.Mode.RELIABILITY
        return MqaQuantity.Mode.BASIC

    @property
    def utility_enabled(self) -> bool:
        ''' Whether to show Probability of Success based on utility curve '''
        return self.enditem and (self.measurand.degrade_limit is not None or self.measurand.fail_limit is not None)

    def _calc_uncert(self) -> MqaUncertaintyResult:
        ''' Calculate uncertainty of measurement process '''
        assert self.result is not None
        if self.measurement.mte.quantity is not None:
            self.measurement.mte.quantity.calculate()
        self.result.uncertainty = self.measurement.uncertainty(self.measurand.testpoint)
        return self.result.uncertainty

    def _calc_tar(self) -> MqaCapabilityResult:
        ''' Calculate TAR and TUR '''
        assert self.result is not None
        assert self.result.uncertainty is not None
        tolerance = self.measurand.tolerance
        uncert = self.result.uncertainty

        if tolerance.onesided:
            limit = np.nanmin((tolerance.flow, tolerance.fhigh))
            tar = abs(self.measurand.testpoint - limit) / uncert.accuracy
            tur = abs(self.measurand.testpoint - limit) / uncert.expanded()
        else:
            tolerance = float(tolerance.plusminus)
            try:
                tar = tolerance / uncert.accuracy
                tur = tolerance / uncert.expanded()
            except ZeroDivisionError:
                tar = 0
                tur = 0
        self.result.capability = MqaCapabilityResult(tar, tur)
        return self.result.capability

    def _calc_guardband(self) -> Limit:
        ''' The acceptance/guardbanded limit. Measured values outside this limit
            are adjusted or rejected.
        '''
        assert self.result is not None
        assert self.result.uncertainty is not None
        tolerance = self.measurand.tolerance
        expanded = self.result.uncertainty.expanded()

        limit = tolerance  # Default, including when method == 'none'

        if self.guardband.method == 'none':
            limit = tolerance

        elif self.guardband.method == 'manual' and self.guardband.accept_limit:
            limit = self.guardband.accept_limit

        # One-sided
        elif tolerance.onesided:
            if np.isfinite(tolerance.flow):
                limit = Limit(
                    Decimal(tolerance.flow + expanded).quantize(tolerance.low*Decimal('0.1')),
                    float('inf')
                    )
            else:
                limit = Limit(
                    float('-inf'),
                    Decimal(tolerance.fhigh - expanded*2).quantize(tolerance.high*Decimal('0.1'))
                )

        elif self.guardband.rule.method in ['rds', 'rp10', 'u95', 'dobbert']:
            tur = self.result.capability.tur
            if tur is not None and round(tur, 3) < self.guardband.rule.threshold:
                if self.guardband.rule.method == 'rds':
                    try:
                        gbf = round(Decimal(np.sqrt(1 - 1/tur**2)), 4)
                    except ZeroDivisionError:
                        gbf = Decimal('0')
                elif self.guardband.rule.method == 'dobbert':
                    M = 1.04 - np.exp(0.38 * np.log(tur) - 0.54)
                    gbf = round(Decimal(1 - M / tur), 4)
                elif self.guardband.rule.method == 'rp10':
                    gbf = round(Decimal(1.25 - 1/tur), 4)
                elif self.guardband.rule.method == 'u95':
                    gbf = round(Decimal(1 - 1/tur), 4)
                else:
                    gbf = Decimal('1')

                plusminus = (tolerance.plusminus * gbf).quantize(tolerance.plusminus*Decimal('0.1'))  # Add one decimal place
                limit = Limit.from_plusminus(tolerance.center, plusminus)
            else:
                limit = self.measurand.tolerance

        elif self.guardband.rule.method == 'pfa':
            target = self.guardband.rule.threshold / 100
            pfa_nogb = pfa(self.result.uncertainty.pdf, self.result.eopr.true.pdf,
                           self.measurand.tolerance, self.measurand.tolerance)
            if pfa_nogb > target:
                limit = target_pfa(
                            self.result.uncertainty.pdf,
                            self.result.eopr.true.pdf,
                            self.measurand.tolerance,
                            target)
                limit = Limit.from_plusminus(limit.center.quantize(tolerance.center),
                                             limit.plusminus.quantize(tolerance.plusminus*Decimal('0.1')))

        elif self.guardband.rule.method == 'cpfa':
            target = self.guardband.rule.threshold / 100
            pfa_nogb = cpfa(self.result.uncertainty.pdf, self.result.eopr.true.pdf,
                            self.measurand.tolerance, self.measurand.tolerance)
            if pfa_nogb > target:
                limit = target_cpfa(
                            self.result.uncertainty.pdf,
                            self.result.eopr.true.pdf,
                            self.measurand.tolerance,
                            target)
                limit = Limit.from_plusminus(limit.center.quantize(tolerance.center),
                                             limit.plusminus.quantize(tolerance.plusminus*Decimal('0.1')))

        else:
            raise NotImplementedError(self.guardband.rule.method)

        self.result.guardband = limit
        return self.result.guardband

    def _calc_eopr(self) -> MqaEoprResult:
        ''' Calculate True EOPR '''
        assert self.result is not None
        pdf, pct, eopr_true = self.measurand.eopr
        if eopr_true:
            true = MqaReliabilityResult(pdf, pct)
            observed = true

        else:
            if pdf.std <= self.result.uncertainty.pdf.std:
                true_pdf = pdf
                print('fy_x > eop')  # This shouldn't happen, but may with user inputs
            else:
                # Calculate TRUE reliability PDF
                true_pdf = Pdf.from_stdev(
                    pdf.mean,
                    np.sqrt(pdf.std**2 - self.result.uncertainty.pdf.std**2)
                )
            true_pct = min(true_pdf.itp(self.measurand.tolerance), 1)
            true = MqaReliabilityResult(true_pdf, true_pct)
            observed = MqaReliabilityResult(pdf, pct)

        self.result.eopr = MqaEoprResult(true=true, observed=observed)
        return self.result.eopr

    def _calc_risks(self, guardband: bool = True) -> MqaRiskResult:
        ''' Calculate PFA, PFR '''
        assert self.result is not None
        acceptance = self.result.guardband if guardband else self.measurand.tolerance

        pfa_true = pfa(self.result.uncertainty.pdf, self.result.eopr.true.pdf,
                       self.measurand.tolerance, acceptance)

        pfr_true = pfr(self.result.uncertainty.pdf, self.result.eopr.true.pdf,
                       self.measurand.tolerance, acceptance)

        cpfa_true = cpfa(self.result.uncertainty.pdf, self.result.eopr.true.pdf,
                         self.measurand.tolerance, acceptance)

        pfa_obs = pfa(self.result.uncertainty.pdf, self.result.eopr.observed.pdf,
                      self.measurand.tolerance, acceptance)

        pfr_obs = pfr(self.result.uncertainty.pdf, self.result.eopr.observed.pdf,
                      self.measurand.tolerance, acceptance)

        cpfa_obs = cpfa(self.result.uncertainty.pdf, self.result.eopr.observed.pdf,
                        self.measurand.tolerance, acceptance)
        self.result.risk = MqaRiskResult(
            pfa_true=pfa_true,
            pfr_true=pfr_true,
            cpfa_true=cpfa_true,
            pfa_observed=pfa_obs,
            pfr_observed=pfr_obs,
            cpfa_observed=cpfa_obs,
        )
        return self.result.risk

    def _calc_reliability(self) -> MqaReliabilityResult:
        ''' Calculate reliability pdfs and probabilities, BOPR '''
        assert self.result is not None
        pdfs = MqaPdfs()
        probs = MqaProbs()

        # Pre-measurement stress
        if self.measurement.calibration.stress_pre is not None:
            pdfs.bt = self.result.eopr.true.pdf.convolve(self.measurement.calibration.stress_pre)
        else:
            pdfs.bt = self.result.eopr.true.pdf

        fy_x = self.result.uncertainty.pdf.given_y(self.measurand.testpoint)
        pdfs.t = pdfs.bt.convolve(fy_x)
        pdfs.accepted_x = fy_x.integrate_fgiveny(*self.result.guardband)
        probs.accepted = pdfs.t.integrate(*self.result.guardband)
        pdfs.x_accepted = pdfs.accepted_x * pdfs.bt / probs.accepted
        pdfs.x_rejected = (1 - pdfs.accepted_x) / (1 - probs.accepted) * pdfs.bt
        probs.obs_oot = 1 - pdfs.bt.convolve(fy_x).integrate(*self.measurand.tolerance)

        # Renewal policies
        if self.measurement.calibration.policy in ['always', 'asneeded']:
            if self.measurement.calibration.mte_adjust is not None:
                pdfs.ap = self.measurement.calibration.mte_adjust
            else:
                pdfs.ap = fy_x
            if self.measurement.calibration.mte_repair is not None:
                pdfs.rp = self.measurement.calibration.mte_repair
            else:
                pdfs.rp = fy_x

            repair_limit = self.measurement.calibration.repair_limit if self.measurement.calibration.repair_limit else self.result.guardband
            pdfs.x_adjust = pdfs.ap
            probs.repair = 1 - pdfs.t.integrate(*repair_limit)
            pdfs.notadjust_x = fy_x.integrate_fgiveny(*self.result.guardband)
            pdfs.adjust_x = 1 - pdfs.notadjust_x
            probs.adjust = (1 - pdfs.t.integrate(*self.result.guardband)) - probs.repair
            probs.repair = probs.repair * (1 - self.measurement.calibration.p_discard)

            probs.notadjust = 1 - probs.adjust
            pdfs.notadjust_x = 1 - pdfs.adjust_x
            pdfs.x_notadjust = pdfs.notadjust_x * pdfs.x_rejected / probs.notadjust
            probs.renewed = probs.adjust + probs.repair
            pdfs.x_renewed = (pdfs.ap * probs.adjust + pdfs.rp * probs.repair) / probs.renewed

            if self.measurement.calibration.policy == 'always':
                pdfs.pt = (
                    pdfs.ap * probs.adjust +
                    pdfs.rp * probs.notadjust * (1-self.measurement.calibration.p_discard))
            else:  # 'asneeded'
                pdfs.pt = (
                    pdfs.x_accepted * probs.accepted +
                    pdfs.x_renewed * probs.renewed)

        else:  # 'never' renewal
            # No adjustment Pdfs needed - skip them
            pdfs.pt = pdfs.x_accepted

        # Post-measurement stress
        if self.measurement.calibration.stress_post is not None:
            bop = pdfs.pt.convolve(self.measurement.calibration.stress_post)
        else:
            bop = pdfs.pt

        bop_pct = bop.integrate(*self.measurand.tolerance)
        self.measurement.interval.set_model(
            bop_pct, self.result.eopr.true.pct, bop,
            self.measurand.tolerance, self.measurand.testpoint)
        aop = self.measurement.interval.aop()
        aop_pct = aop.itp(self.measurand.tolerance)
        self.result.reliability = MqaPeriodReliabilityResult(
            pdfs,
            probs,
            MqaReliabilityResult(bop, bop_pct),
            MqaReliabilityResult(aop, aop_pct),
            None  # success (calculated later)
        )
        return self.result.reliability

    def _calc_success(self) -> float:
        ''' Calculate probability of success '''
        assert self.result is not None
        assert self.result.reliability is not None
        if self.measurement.interval.reliability_model == 'none':
            # No reliability model. AOP = BOP = EOPR(True)
            utility = self.measurand.utility()
            p_success = min(self.measurand.psr * (self.result.eopr.true.pdf * utility).integrate(), 1.0)

        else:
            interval = self.measurement.interval.years
            tt = np.linspace(0, interval, 100)
            success_t = self.success_time(tt)
            p_success = simpson(success_t, x=tt) / interval

        self.result.reliability.success = p_success
        return self.result.reliability.success

    def success_time(self, t: Sequence[float]) -> Sequence[float]:
        ''' Calculate probability of success over time range t (years) '''
        # Only run this if it is an end item
        psr = self.measurand.psr
        utility = self.measurand.utility()
        p_success_t = np.zeros_like(t, dtype=float)
        for i, tt in enumerate(t):
            if tt == 0:
                p_success_t[i] = min(psr * (self.result.reliability.bop.pdf*utility).integrate(), 1.0)
            else:
                p_success_t[i] = min(psr * (self.reliability_t(tt)*utility).integrate(), 1.0)
        return p_success_t

    def reliability_t(self, t: float) -> Pdf:
        ''' Calculate Reliability Pdf at time t (years)

            Note the reliability model normalizes the reliability PDF,
            so reliability_t(0) will not match bop.pdf exactly. See note in RP-19
            "It should be noted that, this expression [R(t)] becomes valid
            at the point between BOP and EOP where the attribute bias
            distribution becomes approximately normal."
        '''
        return self.measurement.interval.pdf_time(t)

    def interval_for_eopr(self, eopr: float, true_eopr: bool = False) -> float:
        ''' Calculate a new interval to acheive the input EOPR '''
        if true_eopr:
            true_eopr = eopr
            if self.measurand.tolerance.onesided:
                sigma_true = Pdf.from_itp(self.measurand.testpoint, true_eopr, self.measurand.tolerance).std
            else:
                sigma_true = float(self.measurand.tolerance.plusminus) / stats.norm.ppf((1+true_eopr)/2)
            sigma_obs = np.sqrt(sigma_true**2 - self.result.uncertainty.stdev**2)
            pdf_obs = Pdf.from_stdev(self.measurand.testpoint, sigma_obs)
            obs_eopr = pdf_obs.itp(self.measurand.tolerance)
        else:
            obs_eopr = eopr
            if self.measurand.tolerance.onesided:
                sigma_obs = Pdf.from_itp(self.measurand.testpoint, obs_eopr, self.measurand.tolerance).std
            else:
                sigma_obs = float(self.measurand.tolerance.plusminus) / stats.norm.ppf((1+obs_eopr)/2)
            sigma_true = np.sqrt(sigma_obs**2 - self.result.uncertainty.stdev**2)
            pdf_true = Pdf.from_stdev(self.measurand.testpoint, sigma_true)
            true_eopr = pdf_true.itp(self.measurand.tolerance)

        new_interval_yr = self.measurement.interval.model.find_interval(true_eopr)
        self.measurement.interval.years = new_interval_yr
        self.measurand.eopr_pct = obs_eopr
        return new_interval_yr

    def eopr_for_interval(self, interval: float) -> float:
        ''' Change the interval and predict the new OBSERVED reliability '''
        sigma_true = self.reliability_t(interval).std
        obs_pdf = Pdf.from_stdev(self.measurand.testpoint, np.sqrt(sigma_true**2 + self.result.uncertainty.stdev**2))
        obs_eopr = obs_pdf.itp(self.measurand.tolerance)
        self.measurand.eopr_pct = obs_eopr
        self.measurement.interval.years = interval
        return obs_eopr

    def _calc_costs(self):
        ''' Calculate cost model '''
        assert self.result is not None
        if self.mqa_mode < MqaQuantity.Mode.COSTS:
            self.result.cost_item = MqaCostItemResult()
            self.result.cost_annual = MqaCostAnnualResult()
            return

        fa = self.costs.item.cfa * self.result.risk.pfa_true
        fr = self.costs.item.cfr * self.result.risk.pfr_true
        itemcost = MqaCostItemResult(expected=fa+fr)

        costs = MqaCostAnnualResult()

        downtime = (self.costs.annual.downtime.cal +
                    self.costs.annual.downtime.adj * self.result.reliability.probs.adjust +
                    self.costs.annual.downtime.rep * self.result.reliability.probs.repair)
        p_available = 1 / (1 + downtime / self.measurement.interval.days)
        ns = (downtime/self.measurement.interval.days) * self.costs.annual.nuut * self.costs.annual.suut
        spare_cost = ns * self.costs.annual.uut  # csa in RP19
        csyear = self.costs.annual.spare_startup * spare_cost

        # Annual costs
        dut_per_year = (self.costs.annual.nuut + ns) / self.measurement.interval.years
        ccal = self.costs.annual.cal * dut_per_year
        cadj = self.costs.annual.adjust * dut_per_year * self.result.reliability.probs.adjust
        crep = dut_per_year * self.costs.annual.repair * self.result.reliability.probs.repair
        support = ccal + cadj + crep
        total = support  # No performance cost for cal items

        costs.p_available = p_available
        costs.ns = ns
        costs.spare_cost = spare_cost
        costs.spares_year = csyear
        costs.cal = ccal
        costs.adj = cadj
        costs.rep = crep
        costs.support = support
        costs.total = total

        if self.enditem:
            costs.performance = self.costs.item.cfa * self.costs.annual.nuut * self.costs.annual.pe * (1 - self.result.reliability.success)
            # cpc = self.costs.cf * self.costs.nuut * self.costs.pe * self.result.reliability.success  #--> successful event
            costs.total = costs.support + costs.performance

        self.result.cost_item = itemcost
        self.result.cost_annual = costs

    def _init_result(self):
        ''' Initialize the MqaQuantityResult object '''
        self.result = MqaQuantityResult(
            self,
            self.measurand.testpoint,
            self.measurand.tolerance,
            None,  # Uncertainty
            None,  # capability
            None,  # EOPR
            None,  # risk
            None,  # guardband
            None,  # Reliability
            None,  # Cost_item
            None   # Cost Annual
        )

    def calculate(self, refresh: bool = False) -> MqaQuantityResult:
        ''' Calculate everything '''
        if refresh or self.result is None:
            self._init_result()
            self._calc_uncert()
            self._calc_tar()
            self._calc_eopr()
            self._calc_guardband()
            self._calc_risks()
            self._calc_reliability()
            self._calc_success()
            self._calc_costs()
        return self.result
