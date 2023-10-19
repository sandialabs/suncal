''' Risk calculation model consisting of process and test distributions and specification limits '''

from collections import namedtuple
import warnings
from scipy import stats

from ..common import distributions
from . import risk
from . import guardband
from . import guardband_tur
from .report.risk import RiskReport


class RiskModel:
    ''' Risk calculation model

        Args:
            procdist: Process distribution
            testdist: Test/Measurment distribution
            speclimist: (Lower, Upper) specification limits
            gbofsts: (lower, upper) acceptance offsets from specification limits
    '''
    def __init__(self, procdist=None, testdist=None, speclimits=None, gbofsts=None):
        self.testdist = testdist
        self.procdist = procdist
        self.speclimits = speclimits
        self._gbofsts = gbofsts
        self.testbias = 0      # Offset between testdist median and measurement result
        self.cost_FA = None    # Cost of false accept and reject for cost-based guardbanding
        self.cost_FR = None
        self.report = RiskReport(self)

        if procdist is None and testdist is None:
            self.procdist = distributions.get_distribution('normal', loc=0, std=.51)
            self.testdist = distributions.get_distribution('normal', loc=0, std=.125)
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

    def calculate(self):
        ''' Calculate (just returns self as RiskModel is same as RiskResults) '''
        return self  # RiskModel is same as RiskResults

    def is_simple(self):
        ''' Check if simplified normal-only functions can be used '''
        if self.procdist is None or self.testdist is None:
            return False
        if self.procdist.median() != 0 or self.testbias != 0:
            return False
        if self.procdist.name != 'normal' or self.testdist.name != 'normal':
            return False
        if self.speclimits[1] != 1 or self.speclimits[0] != -1:
            return False
        if self.gbofsts[1] != self.gbofsts[0]:
            return False
        return True

    def to_simple(self):
        ''' Convert to simple, normal-only form. '''
        if self.is_simple():
            return  # Already in simple form

        # Get existing parameters
        tur = self.get_tur() if self.testdist is not None else 4
        itp = self.get_itp() if self.procdist is not None else 0.95
        median = self.testdist.median() if self.testdist is not None else 0
        gbf = self.get_gbf()

        # Convert to normal/symmetric
        self.speclimits = (-1, 1)
        sigma0 = self.speclimits[1] / stats.norm.ppf((1+itp)/2)
        self.procdist = distributions.get_distribution('normal', loc=0, std=sigma0)
        sigmat = 1/tur/2
        self.testdist = distributions.get_distribution('normal', loc=median, std=sigmat)
        self.set_gbf(gbf)

    def set_gbf(self, gbf):
        ''' Set guardband factor

            Args
                gbf: Guardband factor as multiplier of specification
                  limit. Acceptance limit A = T * gbf.
        '''
        rng = (self.speclimits[1] - self.speclimits[0])/2
        gb = rng * (1 - gbf)
        self._gbofsts = gb, gb

    def get_gbf(self):
        ''' Get guardband as multiplier GB where A = T * GB '''
        gb = self.gbofsts[1] - (self.gbofsts[1] - self.gbofsts[0])/2
        rng = (self.speclimits[1] - self.speclimits[0])/2
        gbf = 1 - gb / rng
        return gbf

    def get_tur(self):
        ''' Get test uncertainty ratio.
            Speclimit range / Expanded test measurement uncertainty.
        '''
        rng = (self.speclimits[1] - self.speclimits[0])/2   # Half the interval
        TL = self.testdist.std() * 2   # k=2
        return rng/TL

    def get_itp(self):
        ''' Get in-tolerance probability '''
        return 1 - self.process_risk()

    def specific_risk(self):
        ''' Calculate specific accept/reject risk given the test measurement defined by dist_test
            including its median shift. Does not consider process dist. If median(testdist)
            is within spec limits, PFA is returned. If median(testdist) is outside spec
            limits, PFR is returned.

            Args:
                risk (float): Probability of false accept or reject
            accept (bool): Accept or reject this measurement
        '''
        med = self.testdist.median() + self.testbias
        LL, UL = self.speclimits
        LL, UL = min(LL, UL), max(LL, UL)  # Make sure LL < UL
        accept = (LL + self.gbofsts[0] <= med <= UL - self.gbofsts[1])

        if LL + self.gbofsts[0] <= med <= UL - self.gbofsts[1]:
            pfx = self.testdist.cdf(LL) + (1 - self.testdist.cdf(UL))
        else:
            pfx = abs(self.testdist.cdf(LL) - self.testdist.cdf(UL))

        result = namedtuple('SpecificRisk', ['probability', 'accept'])
        return result(pfx, accept)

    def process_risk(self):
        ''' Calculate total process risk, risk of process distribution being outside
            specification limits
        '''
        return risk.specific_risk(self.procdist, *self.speclimits)[1]

    def cpk(self):
        ''' Get process risk and CPK values

        Args:
            Cpk (float): Process capability index. Cpk > 1.333 indicates process is
              capable of meeting specifications.
            risk_total (float): Total risk (0-1 range) of nonconformance
            risk_lower (float): Risk of nonconformance below LL
            risk_upper (float): Risk of nonconformance above UL
        '''
        return risk.specific_risk(self.procdist, *self.speclimits)

    def PFA(self, conditional=False):
        ''' Calculate probability of false acceptance (consumer risk).

            Returns:
                PFA (float): Probability of false accept (0-1)
        '''
        if conditional:
            return risk.PFA_conditional(self.procdist, self.testdist, *self.speclimits,
                        *self.gbofsts, self.testbias)
        return risk.PFA(self.procdist, self.testdist, *self.speclimits,
                        *self.gbofsts, self.testbias)

    def PFR(self):
        ''' Calculate probability of false reject (producer risk).

            Returns:
                PFR (float): Probability of false reject (0-1)
        '''
        return risk.PFR(self.procdist, self.testdist, *self.speclimits,
                        *self.gbofsts, self.testbias)

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
            gbl, gbu = guardband.optimize(self.procdist, self.testdist,
                                          *self.speclimits, target=pfa,
                                          allow_negative=allow_negative,
                                          conditional=conditional)
            self.gbofsts = gbl, gbu

        elif not conditional:
            gb = guardband.target(self.procdist, self.testdist,
                                    *self.speclimits, target_PFA=pfa,
                                    testbias=self.testbias)
            self.gbofsts = gb, gb

        else:
            gb = guardband.target_conditional(self.procdist, self.testdist,
                                              *self.speclimits, target_PFA=pfa,
                                              testbias=self.testbias)
            self.gbofsts = (gb, gb)

    def guardband_tur(self, method):
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
        tur = self.get_tur()
        if not self.is_simple():
            warnings.warn('Using TUR-based guardband when TUR assumptions may not apply. '
                            'Results may not acheive desired PFA.')

        if method == 'rss':
            gbf = guardband_tur.rss(tur)
        elif method == 'dobbert':
            gbf = guardband_tur.dobbert(tur)
        elif method == 'rp10':
            gbf = guardband_tur.rp10(tur)
        elif method == 'test':
            gbf = guardband_tur.test95(tur)
        elif method == '4:1':
            gbf = guardband_tur.four_to_1(tur, itp=self.get_itp())
        self.set_gbf(gbf)

    def guardband_cost(self, method='mincost', costfa=100, costfr=10):
        ''' Guardband using cost-based method

            Notes:
                mincost method: Minimize the total expected cost due to false decisions (Ref Easterling 1991)
                minimax method: Minimize the maximum expected cost due to false decisions (Ref Easterling 1991)
        '''
        tur = self.get_tur()
        if not self.is_simple():
            warnings.warn('Using TUR-based guardband when TUR assumptions may not apply. '
                            'Results may not acheive desired PFA.')

        cc_over_cp = costfa/costfr
        if method == 'minimax':
            gbf = guardband_tur.minimax(tur, cc_over_cp=cc_over_cp)
        else: # method == 'mincost':
            itp = self.get_itp()
            gbf = guardband_tur.mincost(tur, itp=itp, cc_over_cp=cc_over_cp)
        self.set_gbf(gbf)
        self.cost_FA = costfa   # Save these for reporting
        self.cost_FR = costfr

    def guardband_specific(self, pfa):
        ''' Guardband for worst-case specific risk '''
        gbl, gbu = guardband.specific(self.testdist, *self.speclimits, pfa)
        self.gbofsts = (gbl, gbu)

    def set_itp(self, itp):
        ''' Set in-tolerance probability by adjusting process distribution
            with specification limits of +/-

            Args:
                itp (float): In-tolerance probability (0-1)
        '''
        self.to_simple()
        sigma = self.speclimits[1] / stats.norm.ppf((1+itp)/2)
        self.procdist = distributions.get_distribution('normal', loc=0, std=sigma)

    def set_tur(self, tur):
        ''' Set test uncertainty ratio by adjusting test distribution

            Args:
                tur (float): Test uncertainty ratio (> 0)
        '''
        self.to_simple()
        sigma = 1/tur/2
        median = self.testdist.median()
        self.testdist = distributions.get_distribution('normal', loc=median, std=sigma)

    def set_testmedian(self, median):
        ''' Set median of test measurement

            Args:
                median (float): Median value of a particular test measurement result
        '''
        sigma = self.testdist.std()
        self.testdist = distributions.get_distribution('normal', loc=median, std=sigma)

    def get_procdist_args(self):
        ''' Get dictionary of arguments for process distribution '''
        args = self.procdist.kwds.copy()
        args.update({'dist': self.procdist.name})
        return args

    def get_testdist_args(self):
        ''' Get dictionary of arguments for test distribution '''
        args = self.testdist.kwds.copy()
        args.update({'dist': self.testdist.name})
        return args
