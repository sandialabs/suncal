''' Calculations for risk, including probability of false accept (PFA) or consumer risk,
    and probability of false reject (PFR) or producer risk.

    The functions PFA_tur and PFR_tur assume normal distributions and symmetric specification
    limits (in terms of standard deviations). These implement the formulas in
    Deaver's "How to Maintain Confidence" paper. They will be the fastest options if the
    distributions can be written in this form.

    The PFA and PFR functions take arbitrary distributions and perform the false accept/
    false reject double integrals numerically. Distributions can be either frozen instances
    of scipy.stats or random samples (e.g. Monte Carlo output of a forward uncertainty
    propagation calculation). PFAR_MC will find both PFA and PFR using a Monte Carlo method.

    find_guardband can be used to determine the guardband required to meet a specified PFA.
'''

import numpy as np
import yaml
from scipy import stats
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.integrate import dblquad
from scipy.optimize import brentq

from . import output
from . import customdists


def process_risk(dist, LL, UL):
    ''' Calculate process risk and process capability index for the distribution.

        Parameters
        ----------
        dist: stats.rv_frozen
            Distribution of possible unit under test values
        LL: float
            Lower specification limit
        UL: float
            Upper specification limit

        Returns
        -------
        Cpk: float
            Process capability index. Cpk > 1.333 indicates process is capable of
            meeting specifications.
        risk_total: float
            Total risk (0-1 range) of nonconformance
        risk_lower: float
            Risk of nonconformance below LL
        risk_upper: float
            Risk of nonconformance above UL

        Notes
        -----
        Normal distributions use the standard definition for cpk:

            min( (UL - x)/(3 sigma), (x - LL)/(3 sigma) )

        Non-normal distributions use the proportion nonconforming:

            min( norm.ppf(risk_lower)/3, norm.ppf(risk_upper)/3 )

        (See https://www.qualitydigest.com/inside/quality-insider-article/process-performance-indices-nonnormal-distributions.html)
    '''
    LL, UL = min(LL, UL), max(LL, UL)  # make sure LL < UL
    risk_lower = dist.cdf(LL)
    risk_upper = 1 - dist.cdf(UL)
    risk_total = risk_lower + risk_upper
    if dist.dist.name == 'norm':
        # Normal distributions can use the standard definition of cpk, process capability index
        cpk = min((UL-dist.mean())/(3*dist.std()), (dist.mean()-LL)/(3*dist.std()))
    else:
        # Non-normal distributions use fractions out.
        # See https://www.qualitydigest.com/inside/quality-insider-article/process-performance-indices-nonnormal-distributions.html
        cpk = max(0, min(abs(stats.norm.ppf(risk_lower))/3, abs(stats.norm.ppf(risk_upper))/3))
        if risk_lower > .5 or risk_upper > .5:
            cpk = -cpk
    return cpk, risk_total, risk_lower, risk_upper


def PFA_tur(sigma, TUR, GB=1):
    ''' Calculate Probability of False Accept (Consumer Risk) for normal
        distributions given spec limit and TUR.

        Parameters
        ----------
        sigma: float
            Specification Limit in terms of standard deviations, symmetric on
            each side of the mean
        TUR: float
            Test Uncertainty Ratio
        GB: float (optional)
            Guard Band factor (0-1) with 1 being no guard band

        Returns
        -------
        PFA: float
            Probability of False Accept

        Reference
        ---------
        Equation 6 in Deaver - How to Maintain Confidence
    '''
    c, _ = dblquad(lambda y, t: np.exp(-(y*y + t*t)/2) / np.pi, sigma, np.inf, gfun=lambda t: -TUR*(t+sigma*GB), hfun=lambda t: -TUR*(t-sigma*GB))
    return c


def PFR_tur(sigma, TUR, GB=1):
    ''' Calculate Probability of False Reject (Producer Risk) for normal
        distributions given spec limit and TUR.

        Parameters
        ----------
        sigma: float
            Specification Limit in terms of standard deviations, symmetric on
            each side of the mean
        TUR: float
            Test Uncertainty Ratio
        GB: float (optional)
            Guard Band factor (0-1) with 1 being no guard band

        Returns
        -------
        PFR: float
            Probability of False Reject

        Reference
        ---------
        Equation 7 in Deaver - How to Maintain Confidence
    '''
    p, _ = dblquad(lambda y, t: np.exp(-(y*y + t*t)/2) / np.pi, -sigma, sigma, gfun=lambda t: TUR*(GB*sigma-t), hfun=lambda t: np.inf)
    return p


def PFA(dist_proc, dist_test, LL, UL, GBL=0, GBU=0, approx=False):
    ''' Calculate Probability of False Accept (Consumer Risk) for arbitrary
        product and test distributions.

        Parameters
        ----------
        dist_proc: stats.rv_frozen
            Distribution of possible unit under test values from process
        dist_test: stats.rv_frozen
            Distribution of possible test measurement values
        LL: float
            Lower specification limit (absolute)
        UL: float
            Upper specification limit (absolute)
        GBL: float
            Lower guard band, as offset. Test limit is LL + GBL.
        GBU: float
            Upper guard band, as offset. Test limit is UL - GBU.
        approx: bool
            Approximate using discrete probability distribution. This
            uses trapz integration so it may be faster than letting
            scipy integrate the actual pdf function.

        Returns
        -------
        PFA: float
            Probability of False Accept
    '''
    if approx:
        xx = np.linspace(dist_proc.median() - dist_proc.std()*8, dist_proc.median() + dist_proc.std()*8, num=1000)
        xx2 = np.linspace(dist_test.median() - dist_test.std()*8,  dist_test.median() + dist_test.std()*8, num=1000)
        return _PFA_discrete((xx, dist_proc.pdf(xx)), (xx2, dist_test.pdf(xx2)), LL, UL, GBL=GBL, GBU=GBU)

    else:
        # Strip loc keyword from test distribution so it can be changed,
        # but shift loc so the MEDIAN (expected) value starts at the spec limit.
        median = dist_test.median()
        kwds = dist_test.kwds.copy()
        locorig = kwds.pop('loc', 0)

        def integrand(y, t):
            return dist_test.dist.pdf(y, loc=t-(median-locorig), **kwds) * dist_proc.pdf(y)

        c1, _ = dblquad(integrand, LL+GBL, UL-GBU, gfun=lambda t: UL, hfun=lambda t: np.inf)
        c2, _ = dblquad(integrand, LL+GBL, UL-GBU, gfun=lambda t: -np.inf, hfun=lambda t: LL)
        return c1 + c2


def _PFA_discrete(dist_proc, dist_test, LL, UL, GBL=0, GBU=0):
    ''' Calculate Probability of False Accept (Consumer Risk) using
        sampled distributions.

        Parameters
        ----------
        dist_proc: array
            Sampled values from process distribution
        dist_test: array
            Sampled values from test measurement distribution
        LL: float
            Lower specification limit (absolute)
        UL: float
            Upper specification limit (absolute)
        GBL: float
            Lower guard band, as offset. Test limit is LL + GBL.
        GBU: float
            Upper guard band, as offset. Test limit is UL - GBU.

        Returns
        -------
        PFA: float
            Probability of False Accept
    '''
    if isinstance(dist_proc, tuple):
        procx, procy = dist_proc
        dy = procx[1]-procx[0]
    else:
        procy, procx = np.histogram(dist_proc, bins='auto', density=True)
        dy = procx[1]-procx[0]
        procx = procx[1:] - dy/2

    if isinstance(dist_test, tuple):
        testx, testy = dist_test
        dx = testx[1]-testx[0]
    else:
        testy, testx = np.histogram(dist_test, bins='auto', density=True)
        dx = testx[1]-testx[0]
        testx = testx[1:] - dx/2

    testmed = np.median(testx)
    c = 0
    for t, ut in zip(procx[np.where(procx > UL)], procy[np.where(procx > UL)]):
        idx = np.where(testx+t-testmed < UL-GBU)
        c += np.trapz(ut*testy[idx], dx=dx)

    for t, ut in zip(procx[np.where(procx < LL)], procy[np.where(procx < LL)]):
        idx = np.where(testx+t-testmed > LL+GBL)
        c += np.trapz(ut*testy[idx], dx=dx)

    c *= dy
    return c


def PFR(dist_proc, dist_test, LL, UL, GBL=0, GBU=0, approx=False):
    ''' Calculate Probability of False Reject (Producer Risk) for arbitrary
        product and test distributions.

        Parameters
        ----------
        dist_proc: stats.rv_frozen
            Distribution of possible unit under test values from process
        dist_test: stats.rv_frozen
            Distribution of possible test measurement values
        LL: float
            Lower specification limit (absolute)
        UL: float
            Upper specification limit (absolute)
        GBL: float
            Lower guard band, as offset. Test limit is LL + GBL.
        GBU: float
            Upper guard band, as offset. Test limit is UL - GBU.
        approx: bool
            Approximate using discrete probability distribution. This
            uses trapz integration so it may be faster than letting
            scipy integrate the actual pdf function.

        Returns
        -------
        PFR: float
            Probability of False Reject

    '''
    if approx:
        xx = np.linspace(dist_proc.median() - dist_proc.std()*8, dist_proc.median() + dist_proc.std()*8, num=1000)
        xx2 = np.linspace(dist_test.median() - dist_test.std()*8,  dist_test.median() + dist_test.std()*8, num=1000)
        return _PFR_discrete((xx, dist_proc.pdf(xx)), (xx2, dist_test.pdf(xx2)), LL, UL, GBL=GBL, GBU=GBU)

    else:
        # Strip loc keyword from test distribution so it can be changed,
        # but shift loc so the MEDIAN value starts at the spec limit.
        median = dist_test.median()
        kwds = dist_test.kwds.copy()
        locorig = kwds.pop('loc', 0)

        def integrand(y, t):
            return dist_test.dist.pdf(y, loc=t-(median-locorig), **kwds) * dist_proc.pdf(y)

        p1, _ = dblquad(integrand, UL-GBU, np.inf, gfun=lambda t: LL, hfun=lambda t: UL)
        p2, _ = dblquad(integrand, -np.inf, LL+GBL, gfun=lambda t: LL, hfun=lambda t: UL)
        return p1 + p2


def _PFR_discrete(dist_proc, dist_test, LL, UL, GBL=0, GBU=0):
    ''' Calculate Probability of False Reject (Producer Risk) using
        sampled distributions.

        Parameters
        ----------
        dist_proc: array
            Sampled values from process distribution
        dist_test: array
            Sampled values from test measurement distribution
        LL: float
            Lower specification limit (absolute)
        UL: float
            Upper specification limit (absolute)
        GBL: float
            Lower guard band, as offset. Test limit is LL + GBL.
        GBU: float
            Upper guard band, as offset. Test limit is UL - GBU.

        Returns
        -------
        PFR: float
            Probability of False Reject
    '''
    if isinstance(dist_proc, tuple):
        procx, procy = dist_proc
        dy = procx[1]-procx[0]
    else:
        procy, procx = np.histogram(dist_proc, bins='auto', density=True)
        dy = procx[1]-procx[0]
        procx = procx[1:] - dy/2

    if isinstance(dist_test, tuple):
        testx, testy = dist_test
        dx = testx[1]-testx[0]
    else:
        testy, testx = np.histogram(dist_test, bins='auto', density=True)
        dx = testx[1]-testx[0]
        testx = testx[1:] - dx/2

    testmed = np.median(testx)
    c = 0
    for t, ut in zip(procx[np.where((procx > LL) & (procx < UL))], procy[np.where((procx > LL) & (procx < UL))]):
        idx = np.where(testx+t-testmed > UL-GBU)
        c += np.trapz(ut*testy[idx], dx=dx)
        idx = np.where(testx+t-testmed < LL+GBL)
        c += np.trapz(ut*testy[idx], dx=dx)

    c *= dy
    return c


def PFAR_MC(dist_proc, dist_test, LL, UL, GBL=0, GBU=0, N=100000, retsamples=False):
    ''' Probability of False Accept/Reject using Monte Carlo Method

        dist_proc: array or stats.rv_frozen
            Distribution of possible unit under test values from process
        dist_test: array or stats.rv_frozen
            Distribution of possible test measurement values
        LL: float
            Lower specification limit (absolute)
        UL: float
            Upper specification limit (absolute)
        GBL: float
            Lower guard band, as offset. Test limit is LL + GBL.
        GBU: float
            Upper guard band, as offset. Test limit is UL - GBU.
        N: int
            Number of Monte Carlo samples
        retsamples: bool
            Return samples along with probabilities

        Returns
        -------
        PFA: float
            False accept probability
        PFR: float
            False reject probability
        proc_samples: array (optional)
            Monte Carlo samples for uut (if retsamples==True)
        test_samples: array (optional)
            Monte Carlo samples for test measurement (if retsamples==True)
    '''
    proc_samples = dist_proc.rvs(size=N)
    median = dist_test.median()
    kwds = dist_test.kwds.copy()
    locorig = kwds.pop('loc', 0)
    test_samples = dist_test.dist.rvs(loc=proc_samples-(median-locorig), **kwds)

    FA = np.count_nonzero(((proc_samples > UL) & (test_samples < UL-GBU)) | ((proc_samples < LL) & (test_samples > LL+GBL)))/N
    FR = np.count_nonzero(((proc_samples < UL) & (test_samples > UL-GBU)) | ((proc_samples > LL) & (test_samples < LL+GBL)))/N

    if retsamples:
        return FA, FR, proc_samples, test_samples
    else:
        return FA, FR


def find_guardband(dist_proc, dist_test, LL, UL, target_PFA, approx=False):
    ''' Calculate (symmetric) guard band required to meet a target PFA value.

        Parameters
        ----------
        dist_proc: stats.rv_frozen
            Distribution of possible unit under test values from process
        dist_test: stats.rv_frozen
            Distribution of possible test measurement values
        LL: float
            Lower specification limit (absolute)
        UL: float
            Upper specification limit (absolute)
        target_PFA: float
            Probability of false accept required
        approx: bool
            Approximate the integral using discrete probability distribution.
            Faster than using scipy.integrate.

        Returns
        -------
        GB: float
            Guardband required to meet target PFA. Symmetric on upper and
            lower limits, such that lower test limit is LL+GB and upper
            test limit is UL-GB.

        Notes
        -----
        Uses Brent's Method to find zero of PFA(dist_proc, dist_test, LL, UL, GBU=x, GBL=x)-target_PFA.
    '''
    # NOTE: This can be slow (several minutes) especially for non-normals. Any way to speed up?
    w = UL-(LL+UL)/2
    try:
        gb, r = brentq(lambda x: PFA(dist_proc, dist_test, LL, UL, GBU=x, GBL=x, approx=approx)-target_PFA, a=-w/2, b=w/2, full_output=True)
    except ValueError:
        return np.nan  # Problem solving

    if r.converged:
        return gb
    else:
        return np.nan


class UncertRisk(object):
    ''' Class incorporating the risk calculations, which can be saved in a Project object. '''
    def __init__(self, name='risk', dproc=customdists.normal(1), dtest=customdists.normal(0.25)):
        self.name = name
        self.description = ''
        self.dist_proc = dproc
        self.dist_test = dtest
        self.speclimits = (-2.0, 2.0)
        self.guardband = (0, 0)
        self.approx = True   # Use numerical approximation for probability distribution integrations

    def set_process_dist(self, dist):
        ''' Set the process distribution

            Parameters
            ----------
            dist: stats.rv_continuous
                Stats distribution for process
        '''
        self.dist_proc = dist

    def set_test_dist(self, dist):
        ''' Set the test distribution

            Parameters
            ----------
            dist: stats.rv_continuous
                Stats distribution for test
        '''
        self.dist_test = dist

    def set_speclimits(self, LL, UL):
        ''' Set specification limits

        Parameters
        ----------
        LL: float
            Lower specification limit
        UL: float
            Upper specification limit
        '''
        self.speclimits = (LL, UL)

    def set_guardband(self, GBL, GBU):
        ''' Set guardband, relative to spec limits

        Parameters
        ----------
        GBL: float
            Lower guard band. Absolute limit will be LL + GBL
        GBU: float
            Upper guard band. Absolute limit will be UL - GBU
        '''
        self.guardband = (GBL, GBU)

    def process_risk(self):
        ''' Calculate process risk and process capability index for the distribution.

        Returns
        -------
        Cpk: float
            Process capability index. Cpk > 1.333 indicates process is capable of
            meeting specifications.
        risk_total: float
            Total risk (0-1 range) of nonconformance
        risk_lower: float
            Risk of nonconformance below LL
        risk_upper: float
            Risk of nonconformance above UL

        Notes
        -----
        Normal distributions use the standard definition for cpk:

            min( (UL - x)/(3 sigma), (x - LL)/(3 sigma) )

        Non-normal distributions use the proportion nonconforming:

            min( norm.ppf(risk_lower)/3, norm.ppf(risk_upper)/3 )

        (See https://www.qualitydigest.com/inside/quality-insider-article/process-performance-indices-nonnormal-distributions.html)
        '''
        return process_risk(self.dist_proc, self.speclimits[0], self.speclimits[1])

    def PFA(self):
        ''' Calculate total probability of false accept (consumer risk) '''
        if self.dist_test is None:
            raise ValueError('Test distribution is not defined')

        return PFA(self.dist_proc, self.dist_test, self.speclimits[0], self.speclimits[1],
                   self.guardband[0], self.guardband[1], self.approx)

    def PFR(self):
        ''' Calculate total probability of false reject (producer risk) '''
        if self.dist_test is None:
            raise ValueError('Test distribution is not defined')

        return PFR(self.dist_proc, self.dist_test, self.speclimits[0], self.speclimits[1],
                   self.guardband[0], self.guardband[1], self.approx)

    def PFAR_meas(self):
        ''' Calculate PFA or PFR of the specific test measurement defined by dist_test
            including its shift. Does not consider dist_proc. If median(dist_test) is within spec
            limits, PFA is returned. If median(dist_test) is outside spec limits, PFR is returned.

            Returns
            PFx: float
                Probability of false accept or reject
            accept: bool
                Accept or reject this measurement
        '''
        med = self.dist_test.median()
        LL, UL = self.speclimits
        LL, UL = min(LL, UL), max(LL, UL)  # Make sure LL < UL
        accept = (med >= LL + self.guardband[0] and med <= UL - self.guardband[0])

        if med >= LL + self.guardband[0] and med <= UL - self.guardband[0]:
            PFx = self.dist_test.cdf(LL) + (1 - self.dist_test.cdf(UL))
        else:
            PFx = abs(self.dist_test.cdf(LL) - self.dist_test.cdf(UL))

        return PFx, accept

    def calculate(self):
        ''' "Calculate" values, returning RiskOutput object '''
        self.out = RiskOutput(self)
        return self.out

    def get_output(self):
        ''' Get output object (or None if not calculated yet) '''
        return self.out

    def get_procdist_args(self):
        ''' Get dictionary of arguments for process distribution '''
        return self.get_config()['distproc']

    def get_testdist_args(self):
        ''' Get dictionary of arguments for test distribution '''
        return self.get_config()['disttest']

    def get_config(self):
        ''' Get configuration dictionary '''
        d = {}
        d['mode'] = 'risk'
        d['name'] = self.name
        d['desc'] = self.description

        if self.dist_proc is not None:
            d['distproc'] = customdists.get_config(self.dist_proc)

        if self.dist_test is not None:
            d['disttest'] = customdists.get_config(self.dist_test)

        d['GBL'] = self.guardband[0]
        d['GBU'] = self.guardband[1]
        d['LL'] = self.speclimits[0]
        d['UL'] = self.speclimits[1]
        return d

    def save_config(self, fname):
        ''' Save configuration to file.

            Parameters
            ----------
            fname: string or file
                File name or file object to save to
        '''
        d = self.get_config()
        out = yaml.dump([d], default_flow_style=False)
        try:
            fname.write(out)
        except AttributeError:
            with open(fname, 'w') as f:
                f.write(out)

    @classmethod
    def from_config(cls, config):
        ''' Create new UncertRisk from configuration dictionary '''
        newrisk = cls(name=config.get('name', 'risk'))
        newrisk.description = config.get('desc', '')
        newrisk.set_speclimits(config.get('LL', 0), config.get('UL', 0))
        newrisk.set_guardband(config.get('GBL', 0), config.get('GBU', 0))

        dproc = config.get('distproc', None)
        if dproc is not None:
            dist_proc = customdists.from_config(dproc)
            newrisk.set_process_dist(dist_proc)
        else:
            newrisk.dist_proc = None

        dtest = config.get('disttest', None)
        if dtest is not None:
            dist_test = customdists.from_config(dtest)
            newrisk.set_test_dist(dist_test)
        else:
            newrisk.dist_test = None
        return newrisk

    @classmethod
    def from_configfile(cls, fname):
        ''' Read and parse the configuration file. Returns a new UncertRisk
            instance.

            Parameters
            ----------
            fname: string or file
                File name or open file object to read configuration from
        '''
        try:
            try:
                yml = fname.read()  # fname is file object
            except AttributeError:
                with open(fname, 'r') as fobj:  # fname is string
                    yml = fobj.read()
        except UnicodeDecodeError:
            # file is binary, can't be read as yaml
            return None

        try:
            config = yaml.safe_load(yml)
        except yaml.scanner.ScannerError:
            return None  # Can't read YAML

        u = cls.from_config(config[0])  # config yaml is always a list
        return u


class RiskOutput(output.Output):
    ''' Output object for risk calculation. Just a reporting wrapper around
        UncertRisk for parallelism with other calculator modes.
    '''
    def __init__(self, risk):
        self.risk = risk

    def report(self, **kwargs):
        ''' Generate report of risk calculation '''
        hdr = []
        cols = []

        if self.risk.dist_proc is not None:
            cpk, risk_total, risk_lower, risk_upper = self.risk.process_risk()
            hdr.extend(['Process Risk'])   # No way to span columns at this point...
            cols.append(['Process Risk: {:.2f}%'.format(risk_total*100),
                         'Upper limit risk: {:.2f}%'.format(risk_upper*100),
                         'Lower limit risk: {:.2f}%'.format(risk_lower*100),
                         'Process capability index (Cpk): {:.6f}'.format(cpk)])

        if self.risk.dist_test is not None:
            val = self.risk.dist_test.median()
            PFx, accept = self.risk.PFAR_meas()  # Get PFA/PFR of specific measurement

            hdr.extend(['Test Measurement Risk'])
            cols.append([
                'Measured value: {:.4g}'.format(val),
                'Result: {}'.format('ACCEPT' if accept else 'REJECT'),
                'PF{} of measurement: {:.2f}%'.format('A' if accept else 'R', PFx*100),
                ''])

        if self.risk.dist_test is not None and self.risk.dist_proc is not None:
            hdr.extend(['Combined Risk'])
            cols.append([
                'Total PFA: {:.2f}%'.format(self.risk.PFA()*100),
                'Total PFR: {:.2f}%'.format(self.risk.PFR()*100), '', ''])

        if len(hdr) > 0:
            rows = list(map(list, zip(*cols)))  # Transpose cols->rows
            return output.md_table(rows=rows, hdr=hdr)
        else:
            return output.MDstring()

    def report_all(self, **kwargs):
        ''' Report with table and plots '''
        with mpl.style.context(output.mplcontext):
            plt.ioff()
            fig = plt.figure()
            self.plot_dists(fig)
        r = output.MDstring()
        r.add_fig(fig)
        r += self.report(**kwargs)
        return r

    def plot_dists(self, fig=None):
        ''' Plot risk distributions '''
        with mpl.style.context(output.mplcontext):
            plt.ioff()
            if fig is None:
                fig = plt.figure()
            fig.clf()

            nrows = (self.risk.dist_proc is not None) + (self.risk.dist_test is not None)
            plotnum = 0
            LL, UL = self.risk.speclimits
            GBL, GBU = self.risk.guardband

            # Add some room on either side of distributions
            pad = 0
            if self.risk.dist_proc is not None:
                pad = max(pad, self.risk.dist_proc.std() * 3)
            if self.risk.dist_test is not None:
                pad = max(pad, self.risk.dist_test.std() * 3)

            x = np.linspace(LL - pad, UL + pad, 300)
            if self.risk.dist_proc is not None:
                yproc = self.risk.dist_proc.pdf(x)
                ax = fig.add_subplot(nrows, 1, plotnum+1)
                ax.plot(x, yproc, label='Process Distribution', color='C0')
                ax.axvline(LL, ls='--', label='Specification Limits', color='C2')
                ax.axvline(UL, ls='--', color='C2')
                ax.fill_between(x, yproc, where=((x <= LL) | (x >= UL)), alpha=.5, color='C0')
                ax.set_ylabel('Probability Density')
                ax.set_xlabel('Value')
                ax.legend(loc='upper left')
                plotnum += 1

            if self.risk.dist_test is not None:
                ytest = self.risk.dist_test.pdf(x)
                median = self.risk.dist_test.median()
                ax = fig.add_subplot(nrows, 1, plotnum+1)
                ax.plot(x, ytest, label='Test Distribution', color='C1')
                ax.axvline(median, ls='--', color='C1')
                ax.axvline(LL, ls='--', label='Specification Limits', color='C2')
                ax.axvline(UL, ls='--', color='C2')
                if GBL != 0 or GBU != 0:
                    ax.axvline(LL+GBL, ls='--', label='Guard Band', color='C3')
                    ax.axvline(UL-GBU, ls='--', color='C3')

                if median > UL-GBU or median < LL+GBL:   # Shade PFR
                    ax.fill_between(x, ytest, where=((x >= LL) & (x <= UL)), alpha=.5, color='C1')
                else:  # Shade PFA
                    ax.fill_between(x, ytest, where=((x <= LL) | (x >= UL)), alpha=.5, color='C1')

                ax.set_ylabel('Probability Density')
                ax.set_xlabel('Value')
                ax.legend(loc='upper left')
            fig.tight_layout()
        return fig
