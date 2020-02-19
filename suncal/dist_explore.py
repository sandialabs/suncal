''' Backend for distribution explorer. This is mostly an educational/training function.

    DistExplore: Class for calculation distributions and running Monte-Carlos "by hand"
    DistOutput: Output report for distribution explorer
'''
import numpy as np
import yaml
import matplotlib as mpl
import matplotlib.pyplot as plt

from . import distributions
from . import output
from . import uparser


class DistExplore(object):
    ''' Distribution Explorer

        For setting up stats distributions, sampling them, and calculating Monte-Carlos
        using the sampled values.

        Parameters
        ----------
        name: string
            Name for the distribution explorer object
        samples: int
            Number of random samples to run
        seed: int or None
            Random number seed
    '''
    def __init__(self, name='distributions', samples=10000, seed=None):
        self.dists = {}         # Dictionary of name: stats.rv_continuous
        self.samples = samples  # Number of samples
        self.samplevalues = {}  # Dictionary of name: sample array
        self.seed = seed
        self.name = name
        self.description = ''
        self.out = DistOutput(self.samplevalues)

    def set_numsamples(self, N):
        ''' Set number of samples '''
        self.samples = N
        self.samplevalues = {}
        self.out = DistOutput(self.samplevalues)

    def sample(self, name):
        ''' Sample input with given name '''
        dist = self.dists.get(name, None)
        expr = uparser.get_expr(name)
        if expr is None:
            raise ValueError('Invalid expression {}'.format(name))

        if expr.is_symbol:
            # This is a base distribution, just sample it
            assert dist is not None
            self.samplevalues[name] = dist.rvs(self.samples)

            # But check for downstream Monte Carlos that use this variable and sample them too
            for mcexpr in [uparser.get_expr(n) for n in self.dists.keys()]:
                if mcexpr is not None and str(mcexpr) != name and name in [str(x) for x in mcexpr.free_symbols]:
                    self.sample(str(mcexpr))

        else:
            # This is an expression. Sample all the input variables if not sampled already.
            inputs = {}
            for i in [str(x) for x in expr.free_symbols]:
                if i not in self.samplevalues and i in self.dists.keys():
                    self.sample(i)
                elif i not in self.dists.keys():
                    raise ValueError('Variable {} has not been defined'.format(i))

                inputs[i] = self.samplevalues[i]
            self.samplevalues[name] = uparser.docalc(inputs, name)
        self.out = DistOutput(self.samplevalues)
        return self.samplevalues[name]

    def calculate(self):
        ''' Sample all distributions and return report '''
        if self.seed is not None:
            np.random.seed(self.seed)
        for name in self.dists.keys():
            self.sample(name)
        self.out = DistOutput(self.samplevalues)
        return self.out

    def get_output(self):
        ''' Get output object (or None if not calculated yet) '''
        return self.out

    def get_config(self):
        d = {}
        d['mode'] = 'distributions'
        d['name'] = self.name
        d['desc'] = self.description
        d['seed'] = self.seed
        d['distnames'] = [str(x) for x in self.dists.keys()]
        d['distributions'] = [x.get_config() if x is not None else None for x in self.dists.values()]
        return d

    @classmethod
    def from_config(cls, config):
        newdist = cls(name=config.get('name', 'distributions'))
        newdist.name = config.get('name', 'distributions')
        newdist.description = config.get('desc', '')
        newdist.seed = config.get('seed', None)
        exprs = config.get('distnames', [])
        dists = [distributions.from_config(x) if x is not None else None for x in config.get('distributions', [])]
        newdist.dists = dict(zip(exprs, dists))
        return newdist

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


class DistOutput(output.Output):
    ''' Output for distribution explorer '''
    def __init__(self, samples):
        self.samples = samples  # Dict
        self.fitparams = None   # Dictionary of fit parameters if a fit is plotted with plot_hist

    def report(self, **kwargs):
        ''' Generate report of distributions and Monte Carlos '''
        hdr = ['Parameter', 'Mean', 'Median', 'Standard Deviation']
        rows = []
        for name, samples in self.samples.items():
            if samples is not None:
                rows.append([output.format_math(name),
                             output.formatter.f(np.mean(samples), fmin=3, **kwargs),
                             output.formatter.f(np.median(samples), fmin=3, **kwargs),
                             output.formatter.f(np.std(samples), ddof=1, fmin=3, **kwargs)])
            else:
                rows.append([output.format_math(name), 'N/A', 'N/A', 'N/A'])
        r = output.md_table(rows, hdr)
        return r

    def report_single(self, name, **kwargs):
        ''' Report stats on a single distribution '''
        samples = np.atleast_1d(self.samples.get(name))
        stdev = samples.std(ddof=1)
        q025, q25, q75, q975 = np.quantile(samples, (.025, .25, .75, .975))

        hdr = ['Parameter', 'Value']
        rows = [['Mean', output.formatter.f(samples.mean(), fmin=3, **kwargs)],
                ['Standard Deviation', output.formatter.f(stdev, fmin=3, **kwargs)],
                ['Standard Uncertainty', output.formatter.f(stdev/np.sqrt(len(samples)), fmin=3, **kwargs)],
                ['N', format(len(samples), 'd')],
                ['Minimum', output.formatter.f(samples.min(), fmin=3, **kwargs)],
                ['First Quartile', output.formatter.f(q25, fmin=3, **kwargs)],
                ['Median', output.formatter.f(np.median(samples), fmin=3, **kwargs)],
                ['Third Quartile', output.formatter.f(q75, fmin=3, **kwargs)],
                ['Maximum', output.formatter.f(samples.max(), fmin=3, **kwargs)],
                ['95% Coverage Interval', '{}, {}'.format(output.formatter.f(q025, fmin=3, **kwargs), output.formatter.f(q975, fmin=3, **kwargs))],
                ]
        r = output.md_table(rows, hdr)

        if self.fitparams is not None:
            rows = list(zip(self.fitparams.keys(), [output.formatter.f(x, fmin=3) for x in self.fitparams.values()]))
            r += '### Fit Parameters'
            r += output.md_table(rows, hdr)

        return r

    def plot_hist(self, name, plot=None, **kwargs):
        ''' Plot histogram of the sampled values

            Parameters
            ----------
            name: string
                Name of distribution to plot
            plot: maptlotlib figure or axis
                Figure or axis to plot on. Will be created if not provided.

            Keyword Arguments
            -----------------
            fitdist: string
                Plot a fit of the named distribution to the data
            qqplot: bool
                Show a Q-Q probability plot in a second axis
            coverage: bool
                Show 95% coverage (symmetric) interval as vertical lines
        '''
        samples = self.samples.get(name)
        fitdist = kwargs.get('fitdist', None)
        qqplot = kwargs.get('qqplot', False)
        coverage = None
        if kwargs.get('coverage', False):
            coverage = np.quantile(samples, (0.025, 0.975))

        fig, ax = output.initplot(plot)
        if len(np.atleast_1d(samples)) > 1:
            params = output.fitdist(samples, distname=fitdist, plot=fig, qqplot=qqplot, coverage=coverage)
        else:
            fig.clf()
            ax = fig.add_subplot(1, 1, 1)
            ax.axvline(samples, label='Sample')
            params = None
        self.fitparams = params

    def report_all(self, **kwargs):
        ''' Report all values '''
        fitdist = kwargs.get('fitdist', None)
        qqplot = kwargs.get('qqplot', False)
        coverage = kwargs.get('coverage', False)
        r = output.MDstring()
        for name in self.samples.keys():

            with mpl.style.context(output.mplcontext):
                plt.ioff()
                fig = plt.figure()
                self.plot_hist(name, plot=fig, fitdist=fitdist, qqplot=qqplot, coverage=coverage)
                fig.suptitle(name)
            r.add_fig(fig)
            r += self.report_single(name, **kwargs)
        return r
