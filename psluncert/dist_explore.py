''' Backend for distribution explorer. This is mostly an educational/training function.

    DistExplore: Class for calculation distributions and running Monte-Carlos "by hand"
    DistOutput: Output report for distribution explorer
'''
import numpy as np
import sympy
import yaml
import matplotlib.pyplot as plt

from . import customdists
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
    def __init__(self, name='distributions', samples=100, seed=None):
        self.distlist = []   # List of distributions
        self.distexpr = []   # List of distribution formulas/names, sympy expressions
        self.samples = samples  # Number of samples
        self.seed = seed
        self.name = name
        self.description = ''

    def add_dist(self, name, dist=None):
        ''' Add a distribution.

            Parameters
            ----------
            name: string
                Name or formula for the distribution
            dist: rv_continuous
                Stats distribution. Omit if name is a formula.
        '''
        self.distlist.append(dist)
        self.distexpr.append(uparser.get_expr(name))

    def update_dist(self, index, name, dist=None):
        ''' Update the distribution at the given index '''
        self.distexpr[index] = uparser.get_expr(name)
        self.distlist[index] = dist

    def set_numsamples(self, N):
        ''' Set number of samples '''
        self.samples = N

    def calculate(self):
        ''' Sample all distributions '''
        samplelist = []
        for dist, expr in zip(self.distlist, self.distexpr):
            if expr is None:
                samplelist.append(None)

            elif expr.is_symbol and dist is not None:
                try:
                    samplelist.append(dist.rvs(self.samples))
                except ValueError:
                    samplelist.append(None)

            else:
                # MUST go in correct order
                names = [str(x) for x in self.distexpr]
                inputs = [str(x) for x in expr.free_symbols]
                try:
                    inputsamples = [samplelist[names.index(i)] for i in inputs]
                    samplelist.append(uparser.docalc(dict(zip(inputs, inputsamples)), str(expr)))
                except (TypeError, IndexError, ValueError):  # An input is not defined or NaN
                    samplelist.append(None)

        self.out = DistOutput(self.distexpr, samplelist)
        return self.out

    def get_output(self):
        ''' Get output object (or None if not calculated yet) '''
        return self.out

    def get_config(self):
        d = {}
        d['mode'] = 'distributions'
        d['name'] = self.name
        d['desc'] = self.description
        d['distnames'] = [str(x) for x in self.distexpr]
        d['distributions'] = [customdists.get_config(x) if x is not None else None for x in self.distlist]
        return d

    @classmethod
    def from_config(cls, config):
        newdist = cls(name=config.get('name', 'distributions'))
        newdist.name = config.get('name', 'distributions')
        newdist.description = config.get('desc', '')
        newdist.distexpr = [uparser.get_expr(x) for x in config.get('distnames', [])]
        newdist.distlist = [customdists.from_config(x) if x is not None else None for x in config.get('distributions', [])]
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
    def __init__(self, exprs, samples):
        self.exprs = exprs
        self.samples = samples

    def report(self, **kwargs):
        ''' Generate report of distributions and Monte Carlos '''
        hdr = ['Parameter', 'Mean', 'Median', 'Standard Deviation']
        rows = []
        for expr, sample in zip(self.exprs, self.samples):
            if sample is not None:
                rows.append([output.sympyeqn(expr),
                             output.formatter.f(np.mean(sample), fmin=3, **kwargs),
                             output.formatter.f(np.median(sample), fmin=3, **kwargs),
                             output.formatter.f(np.std(sample), ddof=1, fmin=3, **kwargs)])
            else:
                rows.append([output.sympyeqn(expr), 'N/A', 'N/A', 'N/A'])
        r = output.md_table(rows, hdr)
        return r

    def report_all(self, **kwargs):
        ''' Generate full report '''
        fitdist = kwargs.get('fitdist', None)
        qqplot = kwargs.get('qqplot', False)

        r = output.MDstring('### Sampled statistics\n\n')
        r += self.report(**kwargs)

        r += '### Histograms\n\n'
        paramslist = []
        for expr, sample in zip(self.exprs, self.samples):
            fig = plt.figure()
            try:
                paramslist.append(output.fitdist(sample, dist=fitdist, fig=fig, qqplot=qqplot))
            except TypeError:
                pass
            r.add_fig(fig)
            fig.gca().set_xlabel('$' + sympy.latex(expr) + '$')

        if fitdist:
            r += '### Fit Distribution ({}) Parameters\n\n'.format(fitdist)
            for expr, params in zip(self.exprs, paramslist):
                r += '#### {}\n\n'.format(output.sympyeqn(expr))
                if params:
                    rows = [[name, '{:.4g}'.format(val)] for name, val in params.items()]
                    r += output.md_table(rows, hdr=['Parameter', 'Value']) + '\n\n'
                else:
                    r += 'No fit found\n\n'
        return r
