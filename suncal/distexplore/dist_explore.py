''' Backend for distribution explorer. This is mostly an educational/training function. '''
import numpy as np

from ..common import uparser
from .report.dist_explore import ReportDistExplore


class DistExplore:
    ''' Distribution Explorer (same as DistExploreResults)

        For setting up stats distributions, sampling them, and calculating Monte-Carlos
        using the sampled values.

        Args:
            name (string): Name for the distribution explorer object
            samples (int): Number of random samples to run
            seed (int): Random number seed
    '''
    def __init__(self, samples=10000, seed=None):
        self.dists = {}         # Dictionary of name (or expr): stats distribution
        self.nsamples = samples  # Number of samples
        self.samplevalues = {}  # Dictionary of name: sample array
        self.seed = seed
        self.report = ReportDistExplore(self)

    def set_numsamples(self, N):
        ''' Set number of samples '''
        self.nsamples = N
        self.samplevalues = {}

    def sample(self, name):
        ''' Sample input with given name

            Args:
                name (str): Name of variable to sample

            Returns:
                samples (array): Array of random samples
        '''
        dist = self.dists.get(name, None)
        expr = uparser.parse_math(name, raiseonerr=False)
        if expr is None:
            raise ValueError(f'Invalid expression {name}')

        if expr.is_symbol:
            # This is a base distribution, just sample it
            assert dist is not None
            self.samplevalues[name] = dist.rvs(self.nsamples)

            # But check for downstream Monte Carlos that use this variable and sample them too
            for mcexpr in [uparser.parse_math(n, raiseonerr=False) for n in self.dists]:
                if mcexpr is not None and str(mcexpr) != name and name in [str(x) for x in mcexpr.free_symbols]:
                    self.sample(str(mcexpr))

        else:
            # This is an expression. Sample all the input variables if not sampled already.
            inputs = {}
            for i in [str(x) for x in expr.free_symbols]:
                if i not in self.samplevalues and i in self.dists:
                    self.sample(i)
                elif i not in self.dists:
                    raise ValueError(f'Variable {i} has not been defined')

                inputs[i] = self.samplevalues[i]
            self.samplevalues[name] = uparser.callf(name, inputs)
        return self.samplevalues[name]

    def calculate(self):
        ''' Sample all distributions and return report '''
        if self.seed is not None:
            np.random.seed(self.seed)
        for name in self.dists:
            self.sample(name)
        return self
