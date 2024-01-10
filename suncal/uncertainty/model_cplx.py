''' Uncertainty Calculator wrapper for processing complex numbers. '''
import inspect
from collections import namedtuple
import sympy
import numpy as np

from ..common import uparser
from .model import Model, ModelCallable
from .variables import RandomVariable
from .results.gum import GumResultsCplx
from .results.monte import McResultsCplx
from .results.uncertainty import UncertaintyCplxResults


def _wrap_callable(func, informats=None, outfmt='ri'):
    ''' Wrap callable function by splitting out real/imaginary components
        of each input and output

        Args:
        ----------
        func (callalbe): Function to wrap
        informats: Dictionary of {'argument': 'ri, 'ma', 'madegrees'} specifying
            whether each argument is in real/imaginary or magnitude/angle format
        outfmt (str): Desired format for output quantity: 'ri' or 'ma'
    '''
    funcname = func.__name__

    if informats is None:
        informats = {}

    innames = list(inspect.signature(func).parameters.keys())  # Complex argument names (a, b, etc.)
    kwnames = []   # Split argument names (a_real, a_imag, b_mag, b_deg, etc.)
    for name in innames:
        fmt = informats.get(name, 'ri')
        assert fmt in ['ma', 'ri', 'madegrees']
        if fmt == 'ma':
            kwnames.extend([f'{name}_mag', f'{name}_rad'])
        elif fmt == 'madegrees':
            kwnames.extend([f'{name}_mag', f'{name}_deg'])
        else:
            kwnames.extend([f'{name}_real', f'{name}_imag'])

    try:  # Make a test call to see return type
        kargs = {k: np.random.random() for k in innames}
        out = func(**kargs)
    except (ValueError, TypeError, IndexError, NameError) as exc:
        raise ValueError('Cannot determine output structure of callable function.') from exc

    try:
        noutputs = len(out)
        if hasattr(out, '_fields'):  # Namedtuple
            outnames = out._fields
        else:
            outnames = [f'{funcname}_{i+1}' for i in range(noutputs)]
    except TypeError:
        noutputs = 1
        outnames = [funcname]

    outcomps = []
    for name in outnames:
        if outfmt == 'ma':
            outcomps.extend([f'{name}_mag', f'{name}_rad'])
        else:
            outcomps.extend([f'{name}_real', f'{name}_imag'])

    def wrapfunc(**kwargs):
        ''' The wrapped function, to be called by ModelCallable '''
        fargs = {}
        for name in innames:
            # Convert kwargs into complex numbers
            fmt = informats.get(name, 'ri')
            if fmt == 'ma':
                mag = kwargs[f'{name}_mag']
                rad = kwargs[f'{name}_rad']
                fargs[name] = mag * np.cos(rad) + 1j * mag * np.sin(rad)
            elif fmt == 'madegrees':
                mag = kwargs[f'{name}_mag']
                rad = np.pi / 180 * kwargs[f'{name}_deg']  # Note: np.deg2rad does weird things to Pint Quantities here
                fargs[name] = mag * np.cos(rad) + 1j * mag * np.sin(rad)
            else:
                fargs[name] = kwargs[f'{name}_real'] + 1j * kwargs[f'{name}_imag']
        ret = func(**fargs)

        outvals = []
        for i, name in enumerate(outnames):
            if noutputs == 1:
                retval = ret
            else:
                retval = ret[i]
            if outfmt == 'ma':
                # note: np.angle() doesn't preserve units. np.arctan2 returns Quantity(radians)
                outvals.extend([abs(retval), np.arctan2(retval.imag, retval.real)])
            else:
                outvals.extend([retval.real, retval.imag])

        outtuple = namedtuple('CplxOut', outcomps)
        result = outtuple(*outvals)
        return result

    return kwnames, outcomps, wrapfunc


class RandomVariableCplx:
    ''' A Pair of RandomVariables, one real one imaginary '''
    def __init__(self, name):
        self.type = 'ri'  # or ma or madegrees
        self.name = name
        self.value = None
        self.uncertainty = None
        self.corr = None
        self.units = None

    def measure(self, value, uncertainty, k=2, correlation=None, units=None):
        ''' Define measured value and uncertainty of the RandomVariable.
            Assumes normal distribution for real and imaginary components.

            Args:
                value (complex): Complex-value of measurand
                uncertainty (complex): Complex-value of uncertainty
                k (float): Coverage factor for real and imaginary uncertainties
                correlation (float): Correlation between real and imaginary components
                units (str): Units to apply to RandomVariable
        '''
        self.type = 'ri'
        self.value = value
        self.uncertainty = uncertainty/k
        self.units = units
        self.corr = correlation

    def measure_magphase(self, magnitude, phase, umagnitude, uphase, degrees=False, k=2,
                         correlation=None, units=None):
        ''' Define measured value and uncertainty of the RandomVariable in
            magnitude/phase format. Assumes normal distribution for magnitude
            and phase components.

            Args:
                magnitude (float): Magnitude of measuand
                phase (float): Phase of measurand (radians or degrees)
                umagnitude (float): Uncertainty in magnitude of measuand
                uphase (float): Uncertainty in phase of measurand (radians or degrees)
                degrees (bool): Phase and Phase uncertainty are in degrees
                k (float): Coverage factor uncertainty components
                correlation (float): Correlation between magnitude and phase components
                units (str): Units to apply to RandomVariable
        '''
        self.type = 'ma' if not degrees else 'madegrees'
        self.value = magnitude, phase
        self.uncertainty = umagnitude/k, uphase/k
        self.units = units
        self.corr = correlation

    def get_randvars(self):
        ''' Get split RandomVariables dictionary as
            {name_real: RandomVariable, name_imag: RandomVariable}, etc.
        '''
        if self.type == 'ri':
            real = RandomVariable()
            real.measure(np.real(self.value))
            real.typeb(std=np.real(self.uncertainty))
            imag = RandomVariable()
            imag.measure(np.imag(self.value))
            imag.typeb(std=np.imag(self.uncertainty))
            randvars = {f'{self.name}_real': real, f'{self.name}_imag': imag}

        elif self.type == 'ma':
            mag = RandomVariable()
            mag.measure(np.real(self.value[0]))
            mag.typeb(std=np.real(self.uncertainty[0]))
            rad = RandomVariable()
            rad.measure(np.real(self.value[1]))
            rad.typeb(std=np.real(self.uncertainty[1]))
            randvars = {f'{self.name}_mag': mag, f'{self.name}_rad': rad}

        elif self.type == 'madegrees':
            mag = RandomVariable()
            mag.measure(np.real(self.value[0]))
            mag.typeb(std=np.real(self.uncertainty[0]))
            deg = RandomVariable()
            deg.measure(np.deg2rad(np.real(self.value[1])))
            deg.typeb(std=np.deg2rad(np.real(self.uncertainty[1])))
            randvars = {f'{self.name}_mag': mag, f'{self.name}_deg': deg}

        return randvars

    def get_correlation(self):
        ''' Get correlation between components as list of
            (name_real, name_imag, correlation) tuples, to pass directly
            in to Variables.correlate(*corr)
        '''
        if self.corr:
            if self.type == 'ri':
                return f'{self.name}_real', f'{self.name}_imag', self.corr
            if self.type == 'ma':
                return f'{self.name}_mag', f'{self.name}_rad', self.corr
            if self.type == 'madegrees':
                return f'{self.name}_mag', f'{self.name}_deg', self.corr
        return None


class ModelComplex:
    ''' Complex Value measurement model

        Args:
            *exprs (string): Functions to compute
            magphase (bool): Calculate output in magnitude/phase format
    '''
    def __init__(self, *exprs, magphase=False):
        self.exprs = exprs
        self.magphase = magphase
        self.cplx_vars = {}

    def var(self, name):
        ''' Get one complex random variable in the model '''
        if name in self.cplx_vars:
            return self.cplx_vars[name]

        v = RandomVariableCplx(name)
        self.cplx_vars[name] = v
        return v

    def _build_model_sympy(self):
        ''' Set up a Model with real-valued components only '''
        # 1) Split out equal signs in equations
        funcnames = []
        exprs = []
        for i, expr in enumerate(self.exprs):
            if '=' in expr:
                funcname, expr = expr.split('=')
                funcnames.append(funcname.strip())
                exprs.append(expr.strip())
            else:
                funcnames.append(f'f{i}')
                exprs.append(expr.strip())

        # 2) Sympyfy
        exprs = [uparser.parse_math(expr, allowcomplex=True) for expr in exprs]

        # 3) Substitute components
        for name, var in self.cplx_vars.items():
            for i, expr in enumerate(exprs):
                if var.type == 'ri':
                    exprs[i] = expr.subs({name: (sympy.Symbol(f'{name}_real', real=True) +
                                                 sympy.I * sympy.Symbol(f'{name}_imag', real=True))})
                elif var.type == 'ma':
                    mag = sympy.Symbol(f'{name}_mag', real=True)
                    rad = sympy.Symbol(f'{name}_rad', real=True)
                    exprs[i] = expr.subs({name: mag * sympy.cos(rad) + sympy.I * mag * sympy.sin(rad)})

                else:  # var.type == 'madegrees'
                    mag = sympy.Symbol(f'{name}_mag', real=True)
                    deg = sympy.Symbol(f'{name}_deg', real=True)
                    exprs[i] = expr.subs(
                        {name: mag * sympy.cos(sympy.pi/180*deg) + sympy.I * mag * sympy.sin(sympy.pi/180*deg)})

        # 4) Split exprs into real/imag, or mag/phase
        splitnames = []
        splitexprs = []
        for expr, fname in zip(exprs, funcnames):
            f_real, f_imag = expr.as_real_imag()
            f_real = f_real.simplify()
            f_imag = f_imag.simplify()

            # The above as_real_imag() must have the symbols defined with real=True to avoid
            # having an expression with re(x) and im(x) in it. But with real=True, any case of
            # sqrt(x**2) is simplified into Abs(x), which breaks the Derivatives later on.
            # These two lines are a stupid way to convert the expressions to standard symbols
            # (without real=True) so that sqrt(x**2) remains sqrt(x**2) which is differentiable.
            f_real = sympy.sympify(str(f_real))
            f_imag = sympy.sympify(str(f_imag))

            if self.magphase:
                f_mag = sympy.sqrt(f_real**2 + f_imag**2)
                f_ph = sympy.atan2(f_imag, f_real)
                splitnames.extend([f'{fname}_mag', f'{fname}_rad'])
                splitexprs.extend([f_mag, f_ph])
            else:
                splitexprs.extend([f_real, f_imag])
                splitnames.extend([f'{fname}_real', f'{fname}_imag'])

        # 5) Simplify expressions and generate the model
        riexprs = []
        for expr, name in zip(splitexprs, splitnames):
            riexprs.append(f'{name} = {expr.simplify()}')

        self.model = Model(*riexprs)

        # 6) Set values and correlations
        for name, varcplx in self.cplx_vars.items():
            randvars = varcplx.get_randvars()
            self.model.variables.variables.update(randvars)
            if varcplx.corr:
                self.model.variables.correlate(*varcplx.get_correlation())

        return self.model

    def calculate_gum(self):
        ''' Run GUM calculation '''
        model = self._build_model_sympy()
        results = model.calculate_gum()
        return GumResultsCplx(results)

    def monte_carlo(self, samples=1000000):
        ''' Run Monte Carlo calculation '''
        model = self._build_model_sympy()
        results = model.monte_carlo(samples=samples)
        return McResultsCplx(results)

    def calculate(self, samples=1000000):
        ''' Run GUM and Monte Carlo calculation and generate a report '''
        gumresults = self.calculate_gum()
        mcresults = self.monte_carlo(samples=samples)
        return UncertaintyCplxResults(gumresults, mcresults)


class ModelComplexCallable:
    ''' Complex Value measurement model using callable function

        Args:
            function (callable): Function to compute. May return multiple parameters
                as a namedtuple.
            magphase (bool): Calculate output in magnitude/phase format
    '''
    def __init__(self, function, magphase=False):
        self.function = function
        self.magphase = magphase
        self.cplx_vars = {}

    def var(self, name):
        ''' Get one complex random variable in the model '''
        if name in self.cplx_vars:
            return self.cplx_vars[name]

        v = RandomVariableCplx(name)
        self.cplx_vars[name] = v
        return v

    def _build_model(self):
        ''' Set up a Model with real-valued components only. Wraps the self.function to split
            real/imaginary component arguments.
        '''
        informats = {name: var.type for name, var in self.cplx_vars.items()}
        argnames, funcnames, _functionwrapped = _wrap_callable(
            self.function, informats, outfmt='ma' if self.magphase else 'ri')
        model = ModelCallable(_functionwrapped, names=funcnames, argnames=argnames)

        # Set values and correlations
        for varcplx in self.cplx_vars.values():
            randvars = varcplx.get_randvars()
            model.variables.variables.update(randvars)
            if varcplx.corr:
                model.variables.correlate(*varcplx.get_correlation())
        self.model = model
        return self.model

    def calculate_gum(self):
        ''' Run GUM calculation '''
        model = self._build_model()
        results = model.calculate_gum()
        return GumResultsCplx(results)

    def monte_carlo(self, samples=1000000):
        ''' Run Monte Carlo calculation '''
        model = self._build_model()
        results = model.monte_carlo(samples=samples)
        return McResultsCplx(results)

    def calculate(self, samples=1000000):
        ''' Run GUM and Monte Carlo calculation and generate a report '''
        gumresults = self.calculate_gum()
        mcresults = self.monte_carlo(samples=samples)
        return UncertaintyCplxResults(gumresults, mcresults)
