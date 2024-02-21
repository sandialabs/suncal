''' Risk calculation model consisting of process and test distributions and specification limits '''
from typing import Sequence
import warnings
from dataclasses import dataclass
from collections import namedtuple
import numpy as np
from scipy import stats

from ..common import distributions, reporter
from . import risk, guardband, guardband_tur
from .risk_montecarlo import PFAR_MC
from .report.risk import (RiskReport,
                          RiskReportMonteCarlo,
                          RiskReportGuardbandSweep,
                          RiskReportProbConform)


@reporter.reporter(RiskReport)
@dataclass
class RiskResults:
    ''' Risk calculation results

        Attributes:
            pfa: Global probability of false acceptance
            cpfa: Conditional global probability of false acceptance
            pfr: Global probability of false reject
            process_risk: Probability of process being out of tolerance
            process_lower: Probability of process being below lower tolerance
            process_upper: Probability of process being above upper tolerance
            Cpk: Process capability index. Cpk > 1.333 indicates
                process is capable of meeting specifications.
            tur: Test Uncertainty Ratio
            itp: In-tolerance Probability (1 - process_risk)
            specific: Probability of the specific test measurement being
                out of tolerance
            specific_accept: Whether the specific measurement is accepted
            specific_worst: Worst-case specific risk
            process_dist: Distribution of the process
            measure_dist: Distribution of the measurement
            measure_bias: Bias in measurement
            tolerance: The tolerance being tested to
            gbofsts: Guardband offsets as difference from the tolerance
            cost_fa: Cost of a false accept, for cost-based guardbands
            cost_fr: Cost of a false reject, for cost-based guardbands
    '''
    pfa: float
    cpfa: float
    pfr: float
    process_risk: float
    process_lower: float
    process_upper: float
    cpk: float
    tur: float
    itp: float
    specific: float
    specific_accept: bool
    specific_worst: float
    process_dist: distributions.Distribution
    measure_dist: distributions.Distribution
    measure_bias: float
    tolerance: tuple[float, float]
    gbofsts: tuple[float, float]
    cost_fa: float = None
    cost_fr: float = None


@reporter.reporter(RiskReportMonteCarlo)
@dataclass
class RiskMonteCarloResults:
    ''' Monte Carlo risk calculation results

        Attributes:
            pfa: Global probability of false acceptance
            cpfa: Conditional global probability of false acceptance
            pfr: Global probability of false reject
            process_dist: Distribution of the process
            measure_dist: Distribution of the measurement
            tolerance: The tolerance being tested to
            gbofsts: Guardband offsets as difference from the tolerance
            process_samples: Monte Carlo samples from process distribution
            measure_samples: Monte Carlo samples from measurement distribution
    '''
    pfa: float
    cpfa: float
    pfr: float
    tur: float
    process_dist: distributions.Distribution
    measure_dist: distributions.Distribution
    measure_bias: float
    tolerance: tuple[float, float]
    gbofsts: tuple[float, float]
    process_samples: Sequence[float] = None
    measure_samples: Sequence[float] = None


@reporter.reporter(RiskReportGuardbandSweep)
@dataclass
class RiskGuardbandSweepResult:
    ''' Guardband sweep result

        Attributes:
            guardband: Array of guardband values swept
            pfa: Probability of false accept at each guardband
            pfr: Probability of false reject at each guardband
    '''
    guardband: Sequence[float]
    pfa: Sequence[float]
    pfr: Sequence[float]


@reporter.reporter(RiskReportProbConform)
@dataclass
class RiskConformanceResult:
    ''' Probability of Conformance result

        Attributes:
            measured: Array of measurement values
            probconform: Probability of conformance at each measured value
            tolerance: The tolerance being tested to
            gbofsts: Guardband offsets as difference from the tolerance
    '''
    measured: Sequence[float]
    probconform: Sequence[float]
    tolerance: tuple[float, float]
    gbofsts: tuple[float, float]


class RiskModel:
    ''' Risk calculation model

        Args:
            procdist: Process distribution
            testdist: Test/Measurment distribution
            speclimist: (Lower, Upper) specification limits
            gbofsts: (lower, upper) acceptance offsets from specification limits
    '''
    def __init__(self, procdist=None, testdist=None, speclimits=None, gbofsts=None):
        self.measure_dist = testdist
        self.process_dist = procdist
        self.speclimits = speclimits
        self._gbofsts = gbofsts
        self.testbias = 0      # Offset between testdist median and measurement result
        self.cost_fa = None    # Cost of false accept and reject for cost-based guardbanding
        self.cost_fr = None

        if procdist is None and testdist is None:
            self.process_dist = distributions.get_distribution('normal', loc=0, std=.51)
            self.measure_dist = distributions.get_distribution('normal', loc=0, std=.125)
        if speclimits is None:
            self.speclimits = (-1, 1)

    @property
    def gbofsts(self):
        ''' Guardband offsets '''
        if self._gbofsts is not None:
            return self._gbofsts
        return (0, 0)

    @gbofsts.setter
    def gbofsts(self, gbofsts):
        ''' Set guardband offsets '''
        self._gbofsts = gbofsts

    @property
    def _tur(self) -> float:
        ''' Get test uncertainty ratio.
            Speclimit range / Expanded test measurement uncertainty.
        '''
        rng = (self.speclimits[1] - self.speclimits[0])/2   # Half the interval
        uncert = self.measure_dist.std() * 2   # k=2
        return rng/uncert

    @property
    def _itp(self):
        ''' Get in-tolerance probability '''
        process_risk = risk.specific_risk(self.process_dist, *self.speclimits).total
        return 1-process_risk

    @property
    def is_simple(self):
        ''' Check if simplified normal-only functions can be used '''
        if self.process_dist is None or self.measure_dist is None:
            return False
        if self.process_dist.median() != 0 or self.testbias != 0:
            return False
        if self.process_dist.name != 'normal' or self.measure_dist.name != 'normal':
            return False
        if self.speclimits[1] != 1 or self.speclimits[0] != -1:
            return False
        if self.gbofsts[1] != self.gbofsts[0]:
            return False
        return True

    def _specific_risk(self):
        ''' Calculate specific accept/reject risk given the test measurement defined by dist_test
            including its median shift. Does not consider process dist. If median(testdist)
            is within spec limits, PFA is returned. If median(testdist) is outside spec
            limits, PFR is returned.

            Args:
                risk (float): Probability of false accept or reject
                accept (bool): Accept or reject this measurement
        '''
        med = self.measure_dist.median() + self.testbias
        LL, UL = self.speclimits
        LL, UL = min(LL, UL), max(LL, UL)  # Make sure LL < UL
        accept = (LL + self.gbofsts[0] <= med <= UL - self.gbofsts[1])

        if LL + self.gbofsts[0] <= med <= UL - self.gbofsts[1]:
            pfx = self.measure_dist.cdf(LL) + (1 - self.measure_dist.cdf(UL))
        else:
            pfx = abs(self.measure_dist.cdf(LL) - self.measure_dist.cdf(UL))

        result = namedtuple('SpecificRisk', ['probability', 'accept'])
        return result(pfx, accept)

    def _specific_worstcase(self):
        ''' Calculate worst-case specific risk, when measured value = acceptance limit '''
        expected = self.measure_dist.median() + self.testbias
        args = distributions.get_distargs(self.measure_dist)
        locorig = args.pop('loc', 0)

        # Test both limits
        loc = self.speclimits[0]+self.gbofsts[0]
        measdist = self.measure_dist.dist(loc=loc-(expected-locorig), **args)
        sprisk_lower = risk.specific_risk(measdist, *self.speclimits).total
        sprisk_upper = 0

        if self.testbias != 0 or not self.is_simple:
            loc = self.speclimits[1]-self.gbofsts[1]
            measdist = self.measure_dist.dist(loc=loc-(expected-locorig), **args)
            sprisk_upper = risk.specific_risk(measdist, *self.speclimits).total

        return np.nanmax([sprisk_lower, sprisk_upper])

    def get_procdist_args(self):
        ''' Get dictionary of arguments for process distribution '''
        args = self.process_dist.kwds.copy()
        args.update({'dist': self.process_dist.name})
        return args

    def get_testdist_args(self):
        ''' Get dictionary of arguments for test distribution '''
        args = self.measure_dist.kwds.copy()
        args.update({'dist': self.measure_dist.name})
        return args

    def calculate(self) -> RiskResults:
        ''' Calculate (just returns self as RiskModel is same as RiskResults) '''
        process_risk = risk.specific_risk(self.process_dist, *self.speclimits)
        specific_risk = self._specific_risk()
        sp_worst = self._specific_worstcase()
        pfa = risk.PFA(
            self.process_dist, self.measure_dist, *self.speclimits,
            *self.gbofsts, self.testbias)
        cpfa = risk.PFA_conditional(
            self.process_dist, self.measure_dist, *self.speclimits,
            *self.gbofsts, self.testbias)
        pfr = risk.PFR(
            self.process_dist, self.measure_dist, *self.speclimits,
            *self.gbofsts, self.testbias)

        return RiskResults(
            pfa=pfa,
            cpfa=cpfa,
            pfr=pfr,
            process_risk=process_risk.total,
            process_lower=process_risk.lower,
            process_upper=process_risk.upper,
            cpk=process_risk.cpk,
            tur=self._tur,
            itp=1-process_risk.total,
            specific=specific_risk.probability,
            specific_accept=specific_risk.accept,
            specific_worst=sp_worst,
            process_dist=self.process_dist,
            measure_dist=self.measure_dist,
            measure_bias=self.testbias,
            tolerance=self.speclimits,
            gbofsts=self.gbofsts,
            cost_fa=self.cost_fa,
            cost_fr=self.cost_fr)

    def calculate_montecarlo(self, nsamples: int = 10000) -> RiskMonteCarloResults:
        pfa, pfr, psamples, msamples, cpfa = PFAR_MC(
            self.process_dist,
            self.measure_dist,
            *self.speclimits,
            *self.gbofsts,
            N=nsamples,
            testbias=self.testbias)

        return RiskMonteCarloResults(
            pfa=pfa,
            cpfa=cpfa,
            pfr=pfr,
            tur=self._tur,
            process_dist=self.process_dist,
            measure_dist=self.measure_dist,
            measure_bias=self.testbias,
            tolerance=self.speclimits,
            gbofsts=self.gbofsts,
            process_samples=psamples,
            measure_samples=msamples)

    def calc_guardband_sweep(self, rng: Sequence[float] = None,
                             num: int = 26) -> RiskGuardbandSweepResult:
        ''' Calculate PFA/PFR as function of guardband

            Args:
                rng: Limits of guardband range (0, 1)
                num: Number of points in sweep
        '''
        if rng is None:
            rng = (0, self.measure_dist.std()*3)
        guardbands = np.linspace(rng[0], rng[1], num=num)
        pfa = np.empty(len(guardbands))
        pfr = np.empty(len(guardbands))
        for i, gb in enumerate(guardbands):
            pfa[i] = risk.PFA(
                self.process_dist, self.measure_dist, *self.speclimits,
                gb, gb, self.testbias)
            pfr[i] = risk.PFR(
                self.process_dist, self.measure_dist, *self.speclimits,
                gb, gb, self.testbias)
        return RiskGuardbandSweepResult(
            guardband=guardbands,
            pfa=pfa,
            pfr=pfr)

    def calc_probability_conformance(self, num=500) -> RiskConformanceResult:
        ''' Calculate probability of conformance curve across
            range of measurement values
        '''
        kwds = distributions.get_distargs(self.measure_dist)
        LL, UL = self.speclimits
        w = (UL-LL)
        xx = np.linspace(LL-w/2, UL+w/2, num=num)
        if not np.isfinite(w):
            w = self.measure_dist.std() * 4
            xx = np.linspace(self.measure_dist.mean()-w if not np.isfinite(LL) else LL-w/2,
                             self.measure_dist.mean()+w if not np.isfinite(UL) else UL+w/2,
                             num=num)
        fa_lower = np.empty(len(xx))
        fa_upper = np.empty(len(xx))
        for i, loc in enumerate(xx):
            self.measure_dist.set_median(loc-self.testbias)
            kwds = distributions.get_distargs(self.measure_dist)
            dtestswp = self.measure_dist.dist(**kwds)
            fa_lower[i] = risk.specific_risk(dtestswp, LL=LL, UL=np.inf).total
            fa_upper[i] = risk.specific_risk(dtestswp, LL=-np.inf, UL=UL).total
        probconform = 1-(fa_lower + fa_upper)
        return RiskConformanceResult(xx, probconform, self.speclimits, self.gbofsts)

    def guardband_tur(self, method: str):
        ''' Apply TUR-based guardband
                Args:
            method: Guardband method to apply. One of: 'dobbert', 'rss',
              'rp10', 'test', '4:1', 'pfa', 'cpfa', 'minimax', 'mincost', 'specific'.

        Notes:
            Dobbert's method maintains <2% PFA for ANY itp at the TUR.
            RSS method: GB = sqrt(1-1/TUR**2)
            test method: GB = 1 - 1/TUR  (subtract the 95% test uncertainty)
            rp10 method: GB = 1.25 - 1/TUR (similar to test, but less conservative)
            4:1 method: Solve for GB that results in same PFA as 4:1 at this itp
        '''
        if not self.is_simple:
            warnings.warn('Using TUR-based guardband when TUR assumptions may not apply. '
                          'Results may not acheive desired PFA.')

        if method == 'rss':
            gbf = guardband_tur.rss(self._tur)
        elif method == 'dobbert':
            gbf = guardband_tur.dobbert(self._tur)
        elif method == 'rp10':
            gbf = guardband_tur.rp10(self._tur)
        elif method == 'test':
            gbf = guardband_tur.test95(self._tur)
        elif method == '4:1':
            gbf = guardband_tur.four_to_1(self._tur, itp=self._itp)

        rng = (self.speclimits[1] - self.speclimits[0])/2
        gb = rng * (1 - gbf)
        self.gbofsts = gb, gb

    def guardband_pfa(self, pfa=0.08, conditional=False,
                      optimizepfr=False, allow_negative=False):
        ''' Guardband to hit the desired PFA.

            Args:
                pfa: Target PFA (for PFA and specific methods.)
                conditional (bool): Guardband for conditional PFA
                optimizepfr (bool): Find possibly asymmetric guardbands that minimize PFR.
                    Applies to pfa and cpfa methods.
                allow_negative (bool): Allow negative guardbands (accepting OOT DUTs)
                    Applies to pfa and cpfa methods.
        '''
        if optimizepfr:
            gbl, gbu = guardband.optimize(self.process_dist,
                                          self.measure_dist,
                                          *self.speclimits, target=pfa,
                                          allow_negative=allow_negative,
                                          conditional=conditional)
            self.gbofsts = gbl, gbu

        elif not conditional:
            gb = guardband.target(self.process_dist,
                                  self.measure_dist,
                                  *self.speclimits, target_PFA=pfa,
                                  testbias=self.testbias)
            self.gbofsts = gb, gb

        else:
            gb = guardband.target_conditional(self.process_dist,
                                              self.measure_dist,
                                              *self.speclimits, target_PFA=pfa,
                                              testbias=self.testbias)
            self.gbofsts = (gb, gb)

    def guardband_pfr(self, pfr=2):
        ''' Guardband to hit the desired PFR.

            Args:
                pfr: Target PFR
        '''
        gb = guardband.target_pfr(self.process_dist,
                                  self.measure_dist,
                                  *self.speclimits, target_pfr=pfr,
                                  testbias=self.testbias)
        self.gbofsts = gb, gb

    def guardband_cost(self, method='mincost', costfa=100, costfr=10):
        ''' Guardband using cost-based method

            Notes:
                mincost method: Minimize the total expected cost due to false decisions (Ref Easterling 1991)
                minimax method: Minimize the maximum expected cost due to false decisions (Ref Easterling 1991)
        '''
        self.cost_fa = costfa   # Save these for reporting
        self.cost_fr = costfr

        if self.is_simple:
            cc_over_cp = costfa/costfr
            if method == 'minimax':
                gbf = guardband_tur.minimax(self._tur, cc_over_cp=cc_over_cp)
            else:
                gbf = guardband_tur.mincost(self._tur, itp=self._itp, cc_over_cp=cc_over_cp)
            rng = (self.speclimits[1] - self.speclimits[0])/2
            gb = rng * (1 - gbf)
            self.gbofsts = gb, gb
        else:
            if method == 'minimax':
                self.gbofsts = guardband.minimax(self.measure_dist, costfa, costfr, costfr, *self.speclimits)
            else:
                self.gbofsts = guardband.mincost(self.process_dist, self.measure_dist,
                                                 costfa, costfr, *self.speclimits)

    def guardband_specific(self, target: float = 0.08):
        ''' Guardband to reach maximum specific risk of target value '''
        self.gbofsts = guardband.specific(
            self.measure_dist, *self.speclimits, target, testbias=self.testbias)


class RiskModelSimple:
    ''' Risk calculation model only using TUR, ITP, and GBF

        Args:
            tur: Test uncertainty rato
            itp: in-tolerance probability
            gbf: guardband factor
    '''
    def __init__(self, tur: float = 4, itp: float = .95, gbf: float = 1):
        self.tur = tur
        self.itp = itp
        self.gbf = gbf

    def dist_measure(self, loc=0):
        sigmatest = 1/self.tur/2
        return stats.norm(loc=loc, scale=sigmatest)

    def dist_process(self):
        sigma0 = risk.get_sigmaproc_from_itp(self.itp)
        return stats.norm(loc=0, scale=sigma0)

    def calculate(self) -> RiskResults:
        ''' Calculate risk '''
        gb = 1-self.gbf
        pfa = risk.PFA_norm(self.itp, self.tur, self.gbf)
        pfr = risk.PFR_norm(self.itp, self.tur, self.gbf)
        cpfa = risk.PFA_norm(self.itp, self.tur, self.gbf)
        process = risk.specific_risk(self.dist_process(), -1, 1)
        specific_worst = risk.specific_risk(self.dist_measure(self.gbf), -1, 1).total
        return RiskResults(
            pfa=pfa,
            cpfa=cpfa,
            pfr=pfr,
            process_risk=process.total,
            process_lower=process.lower,
            process_upper=process.upper,
            cpk=process.cpk,
            tur=self.tur,
            itp=self.itp,
            specific=None,
            specific_accept=True,
            specific_worst=specific_worst,
            process_dist=self.dist_process(),
            measure_dist=self.dist_measure(),
            measure_bias=0,
            tolerance=(-1, 1),
            gbofsts=(gb, gb))

    def guardband_pfa(self, pfa=0.08, conditional=False):
        ''' Calculate guardband FACTOR to achieve target PFA '''
        if not conditional:
            gb = guardband.target(self.dist_process(),
                                  self.dist_measure(),
                                  -1, 1,
                                  target_PFA=pfa)

        else:
            gb = guardband.target_conditional(self.dist_process(),
                                              self.dist_measure(),
                                              -1, 1,
                                              target_PFA=pfa)
        self.gbf = 1 - gb
        return self.gbf

    def guardband_tur(self, method: str):
        ''' Apply TUR-based guardband
                Args:
            method: Guardband method to apply. One of: 'dobbert', 'rss',
              'rp10', 'test', '4:1', 'pfa', 'cpfa', 'minimax', 'mincost', 'specific'.

        Notes:
            Dobbert's method maintains <2% PFA for ANY itp at the TUR.
            RSS method: GB = sqrt(1-1/TUR**2)
            test method: GB = 1 - 1/TUR  (subtract the 95% test uncertainty)
            rp10 method: GB = 1.25 - 1/TUR (similar to test, but less conservative)
            4:1 method: Solve for GB that results in same PFA as 4:1 at this itp
        '''
        if method == 'rss':
            gbf = guardband_tur.rss(self.tur)
        elif method == 'dobbert':
            gbf = guardband_tur.dobbert(self.tur)
        elif method == 'rp10':
            gbf = guardband_tur.rp10(self.tur)
        elif method == 'test':
            gbf = guardband_tur.test95(self.tur)
        elif method == '4:1':
            gbf = guardband_tur.four_to_1(self.tur, itp=self.itp)
        self.gbf = gbf
        return self.gbf

    def guardband_specific(self, target: float = 0.08):
        ''' Guardband to reach maximum specific risk of target value '''
        gbofsts = guardband.specific(self.dist_measure(), -1, 1, target)
        self.gbf = 1-gbofsts[0]
        return self.gbf

    def guardband_cost(self, method='mincost', costfa=100, costfr=10):
        ''' Guardband using cost-based method

            Notes:
                mincost method: Minimize the total expected cost due to false decisions (Ref Easterling 1991)
                minimax method: Minimize the maximum expected cost due to false decisions (Ref Easterling 1991)
        '''
        self.cost_fa = costfa   # Save these for reporting
        self.cost_fr = costfr
        cc_over_cp = costfa/costfr
        if method == 'minimax':
            self.gbf = guardband_tur.minimax(self.tur, cc_over_cp=cc_over_cp)
        else:
            self.gbf = guardband_tur.mincost(self.tur, itp=self.itp, cc_over_cp=cc_over_cp)
        return self.gbf
