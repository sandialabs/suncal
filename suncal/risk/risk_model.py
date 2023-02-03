''' Risk calculation model consisting of process and test distributions and specification limits '''

from collections import namedtuple
from scipy import stats

from ..common import distributions
from . import risk
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

    def PFA(self, approx=True):
        ''' Calculate probability of false acceptance (consumer risk).

            Args:
                approx (bool): Use trapezoidal integration approximation for speed

            Returns:
                PFA (float): Probability of false accept (0-1)
        '''
        return risk.PFA(self.procdist, self.testdist, *self.speclimits,
                        *self.gbofsts, self.testbias, approx)

    def PFR(self, approx=True):
        ''' Calculate probability of false reject (producer risk).

            Args:
                approx (bool): Use trapezoidal integration approximation for speed

            Returns:
                PFR (float): Probability of false reject (0-1)
        '''
        return risk.PFR(self.procdist, self.testdist, *self.speclimits,
                        *self.gbofsts, self.testbias, approx)

    def calc_guardband(self, method, pfa=None):
        ''' Set guardband using a predefined method

        Args:
            method: Guardband method to apply. One of: 'dobbert', 'rss',
              'rp10', 'test', '4:1', 'pfa', 'minimax', 'mincost', 'specific'.
            pfa: Target PFA (for method 'pfa' and 'specific'. Defaults to 0.008)

        Notes:
            Dobbert's method maintains <2% PFA for ANY itp at the TUR.
            RSS method: GB = sqrt(1-1/TUR**2)
            test method: GB = 1 - 1/TUR  (subtract the 95% test uncertainty)
            rp10 method: GB = 1.25 - 1/TUR (similar to test, but less conservative)
            pfa method: Solve for GB to produce desired PFA
            4:1 method: Solve for GB that results in same PFA as 4:1 at this itp
            mincost method: Minimize the total expected cost due to false decisions (Ref Easterling 1991)
            minimax method: Minimize the maximum expected cost due to false decisions (Ref Easterling 1991)
        '''
        CcCp = None
        if method in ['minimax', 'mincost']:
            if (self.cost_FA is None or self.cost_FR is None):
                raise ValueError('Minimax and Mincost methods must set costs of false accept/reject using `set_costs`.')
            CcCp = self.cost_FA/self.cost_FR

        if method == 'specific':
            gbl, gbu = risk.guardband_specific(self.testdist, *self.speclimits, pfa)
            self.gbofsts = (gbl, gbu)
        elif self.is_simple() or method != 'pfa':
            itp = self.get_itp() if self.procdist is not None else None
            gbf = risk.guardband_norm(method, self.get_tur(), pfa=pfa, itp=itp, CcCp=CcCp)
            self.set_gbf(gbf)
        else:
            gb = risk.guardband(self.procdist, self.testdist,
                                *self.speclimits, pfa,
                                self.testbias, approx=True)
            self.gbofsts = (gb, gb)   # Returns guardband as offset

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

    def set_costs(self, FA, FR):
        ''' Set cost of false accept and reject for cost-minimization techniques '''
        self.cost_FA = FA
        self.cost_FR = FR

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
