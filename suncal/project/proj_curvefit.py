''' CurveFit project component '''

from dataclasses import dataclass
import numpy as np

from .component import ProjectComponent
from ..common.limit import Limit
from ..curvefit import CurveFit, Array, WaveCalc
from ..curvefit.results.curvefit import CurveFitResultsCombined


@dataclass
class FitOptions:
    ''' Options for curve fit

        Attribuites:
            func: The function to fit
            polyorder: Order of polynomial for func='poly'
            bounds: Fit bounds
            odr: Use Orthogonal Distance Regression
            absolute_sigma: Treat sigma as absolute (not relative) value
            p0: Initial guess of parameters
    '''
    func: str
    polyorder: int = 1
    bounds: tuple[float, float] = None
    odr: bool = False
    absolute_sigma: bool = False
    p0: list[float] = None


class ProjectCurveFit(ProjectComponent):
    ''' Uncertainty project component '''
    def __init__(self, model: CurveFit = None, name='curvefit'):
        super().__init__(name=name)
        if model is not None:
            self.model = model
        else:
            arr = Array(x=[], y=[])
            self.model = CurveFit(arr)
        self.nsamples = 5000
        self.seed = None
        self.outunits = {}
        self.fitoptions = FitOptions('line')

    def set_fitfunc(self, func, polyorder=2, bounds=None, odr=None, p0=None):
        ''' Set the fit function '''
        self.fitoptions = FitOptions(func, polyorder=polyorder, bounds=bounds, odr=odr, p0=p0)
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
        outwave = self.model.calculate_wave()
        self._result = CurveFitResultsCombined(
            lsq=lsqresult, montecarlo=monteresult, markov=markovresult,
            gum=gumresult, waveform=outwave)
        return self._result

    def get_dists(self):
        ''' Get distributions in this output. '''
        dists = {}
        for m in ['lsq', 'markov', 'montecarlo', 'gum']:
            # Only return one calculation method, defaulting to LSQ
            if self.result.method(m) is not None:
                method = m
                break

        if getattr(self.result, method) is not None:
            result = getattr(self.result, method)
            for pidx, param in enumerate(self.model.pnames):
                dists[f'{param}'] = {'median': result.coeffs[pidx],
                                     'std': result.uncerts[pidx],
                                     'df': result.degf}

            for name, (x, _) in result.predictions.items():
                y = result.y(x)
                uy = result.prediction_band(x, k=1)
                dists[name] = {
                    'median': y,
                    'std': uy,
                    'df': result.degf
                }

        for wname, wave in self.result.waveform.features.items():
            dists[wname] = {
                'median': wave.uncert.nominal,
                'std': wave.uncert.uncert,
                'df': result.degf
            }

        return dists

    def get_config(self):
        setup = self.model.fitsetup()
        d = {}
        d['mode'] = 'curvefit'
        d['curve'] = setup.modelname
        d['name'] = self.name
        d['desc'] = self.description
        d['odr'] = self.fitoptions.odr
        d['xname'] = setup.xname
        d['yname'] = setup.yname
        d['xdates'] = setup.points.xdate
        d['abssigma'] = self.model.absolute_sigma
        d['tolerances'] = {name: tol.config() for name, tol in self.model.tolerances.items()}
        d['predictions'] = {name: {'value': val, 'tolerance': tol.config() if tol else None} for name, (val, tol) in self.model.predictions.items()}
        d['waveform'] = {name: {'calc': w.calc, 'clip': list(w.clip) if w.clip else None, 'thresh': w.thresh,
                                'tolerance': w.tolerance.config() if w.tolerance else None} for name, w in self.model.wavecalcs.items()}

        if setup.modelname == 'poly':
            d['order'] = self.fitoptions.polyorder
        if self.fitoptions.p0 is not None:
            d['p0'] = self.fitoptions.p0
        if self.fitoptions.bounds is not None:
            d['bound0'] = self.fitoptions.bounds[0]
            d['bound1'] = self.fitoptions.bounds[1]

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
        self.description = config.get('desc', '')
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
        self.model.tolerances = {
            name: Limit.from_config(cfg)
            for name, cfg in config.get('tolerances', {}).items()
        }

        self.model.predictions = {
            name: (c.get('value'), Limit.from_config(c.get('tolerance')) if c.get('tolerance') else None)
            for name, c in config.get('predictions', {}).items()
        }

        self.model.wavecalcs = {
            name: WaveCalc(calc=c.get('calc'), clip=c.get('clip'), thresh=c.get('thresh'), tolerance=Limit.from_config(c.get('tolerance') if c.get('tolerance') else None))
            for name, c in config.get('waveform', {}).items()
        }
