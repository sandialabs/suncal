'''
PSL Uncertainty Calculator
Sandia Primary Standards Lab

Compute combined uncertainty of a system
y = f(x1, x2... xn)

given N inputs each with a given probability distribution. Inputs can have any
distribution defined in scipy.stats, or a custom distribution by subclassing
scipy.stats.rv_continuous or scipy.stats.rv_discrete.

Example usage:
>>> function = 'a * b'  # Function can be callable, string expression, or sympy expr
>>> u = UncertCalc(function)
>>> u.set_input('a', nom=10, dist='gaussian', std=.5)
>>> u.set_input('b', nom=5, dist='uniform', scale=1)
>>> u.calculate()


Copyright 2019 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government
retains certain rights in this software.
'''

from .uncertainty import UncertCalc, UncertaintyCalc, InputUncert, InputVar
from .unc_complex import UncertComplex
from .version import __version__, __date__
from . import risk, curvefit, dataset, reverse, sweeper, ttable, unitmgr


