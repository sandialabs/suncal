'''
Suncal Uncertainty Module - Compute combined uncertainty of a system

    y = f(x1, x2... xn)

given N inputs each with a given probability distribution. Inputs can have any
distribution defined in scipy.stats, or a custom distribution by subclassing
scipy.stats.rv_continuous or scipy.stats.rv_discrete.

Example usage:
>>> model = Model('f = a + b')
>>> model.var('a').measure(10).typeb(dist='normal', std=0.5)
>>> model.var('b').measure(5).typeb(dist='uniform', a=1)
>>> model.calculate()
'''

from .model import Model, ModelCallable
from .model_cplx import ModelComplex, ModelComplexCallable

__all__ = ['Model', 'ModelCallable', 'ModelComplex', 'ModelComplexCallable']
