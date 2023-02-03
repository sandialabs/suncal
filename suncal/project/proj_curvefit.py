''' CurveFit project component '''

import numpy as np

from .component import ProjectComponent

from ..curvefit import CurveFit, Array
from ..curvefit.results.curvefit import CurveFitResultsCombined


class ProjectCurveFit(ProjectComponent):
    ''' Uncertainty project component '''
    def __init__(self, model=None, name='curvefit'):
        super().__init__()
        self.name = name
        if model is not None:
            self.model = model
        else:
            arr = Array(x=[], y=[])
            self.model = CurveFit(arr)
        self.nsamples = 5000
        self.seed = None
        self.outunits = {}
        self.result = None
        self.longdescription = None
        self.odr = False
        self.p0 = None
        self.polyorder = 2
        self.bounds = None
        self.project = None  # Parent project

    def set_fitfunc(self, func, polyorder=2, bounds=None, odr=None, p0=None):
        ''' Set the fit function '''
        self.polyorder = polyorder
        self.bounds = bounds
        self.odr = odr
        self.p0 = p0
        self.model.set_fitfunc(func, polyorder, bounds, odr, p0)

    def calculate(self, lsq=True, monte=False, markov=False, gum=False):
        ''' Calculate the curve fit '''
        if self.seed:
            np.random.seed(self.seed)

        lsqresult = monteresult = markovresult = gumresult = None
        if lsq:
            lsqresult = self.model.calculate()
        if monte:
            monteresult = self.model.monte_carlo(self.nsamples)
        if markov:
            markovresult = self.model.markov_chain_monte_carlo(self.nsamples)
        if gum:
            gumresult = self.model.calculate_gum()
        self.result = CurveFitResultsCombined(lsq=lsqresult, montecarlo=monteresult, markov=markovresult, gum=gumresult)
        return self.result

    def get_dists(self):
        ''' Get distributions in this output. '''
        dists = {}
        for method in ['lsq', 'gum', 'montecarlo', 'markov']:
            if getattr(self.result, method) is not None:
                result = getattr(self.result, method)
                for pidx, param in enumerate(self.model.pnames):
                    if method in ['montecarlo', 'markov']:
                        dists[f'{param} ({method})'] = {'samples': result.samples[:, pidx]}
                    else:
                        dists[f'{param} ({method.upper()})'] = {'mean': result.coeffs[pidx],
                                                                'std': result.uncerts[pidx],
                                                                'df': result.degf}
        for method in ['lsq', 'gum', 'montecarlo', 'markov']:
            if getattr(self.result, method) is not None:
                result = getattr(self.result, method)
                dists[f'Confidence ({method.upper()})'] = {
                    'xdates': self.model.arr.xdate,
                    'function': lambda x, result=result: {'mean': result.y(x),
                                                          'std': result.confidence_band(x),
                                                          'df': result.degf}}

                dists[f'Prediction ({method.upper()})'] = {
                    'xdates': self.model.arr.xdate,
                    'function': lambda x, result=result: {'mean': result.y(x),
                                                          'std': result.prediction_band(x),
                                                          'df': result.degf}}
        return dists

    def get_config(self):
        setup = self.model.fitsetup()
        d = {}
        d['mode'] = 'curvefit'
        d['curve'] = setup.modelname
        d['name'] = self.name
        d['desc'] = self.longdescription
        d['odr'] = self.odr
        d['xname'] = setup.xname
        d['yname'] = setup.yname
        d['xdates'] = setup.points.xdate
        d['abssigma'] = self.model.absolute_sigma
        if setup.modelname == 'poly':
            d['order'] = self.polyorder
        if self.p0 is not None:
            d['p0'] = self.p0
        if self.bounds is not None:
            d['bound0'] = self.bounds[0]
            d['bound1'] = self.bounds[1]

        d['arrx'] = setup.points.x.astype('float').tolist()  # Can't yaml numpy arrays, use list
        d['arry'] = setup.points.y.astype('float').tolist()
        if setup.points.has_ux():
            d['arrux'] = list(setup.points.ux)
        if setup.points.has_uy():
            d['arruy'] = list(setup.points.uy)
        return d

    def load_config(self, config):
        ''' Load configuration into the model '''
        self.name = config.get('name', '')
        self.seed = config.get('seed', None)
        self.longdescription = config.get('desc', '')
        arr = Array(np.asarray(config.get('arrx'), dtype=float),
                    np.asarray(config.get('arry'), dtype=float),
                    ux=config.get('arrux', 0.),
                    uy=config.get('arruy', 0.),
                    xdate=config.get('xdates', False))
        self.model.arr = arr
        self.set_fitfunc(
            config['curve'],
            polyorder=config.get('order', 2),
            p0=config.get('p0', None),
            odr=config.get('odr', None))

        self.model.xname = config.get('xname', 'x')
        self.model.yname = config.get('yname', 'y')
        self.model.absolute_sigma = config.get('abssigma', True)
