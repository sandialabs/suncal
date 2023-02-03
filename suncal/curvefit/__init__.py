''' Curve Fitting with Uncertainties on the inputs and outputs '''

from .curvefit import linefit, linefit_lsq, linefitYork, odrfit
from .curvefit_model import CurveFit
from .uncertarray import Array

__all__ = ['CurveFit', 'Array']
