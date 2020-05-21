''' PSL Uncertainty Calculator - Sandia National Labs

Main module for computing uncertainties.
'''
from contextlib import suppress
from collections import namedtuple
import sympy
import numpy as np
import scipy.stats as stat
import inspect
import warnings
import yaml
import logging
from pint import DimensionalityError, UndefinedUnitError, OffsetUnitCalculusError

from . import ureg
from . import uparser
from . import distributions
from . import report
from . import out_uncert

np.seterr(all='warn', divide='ignore', invalid='ignore', under='ignore')  # Don't show div/0 warnings. We usually just want to happily return Inf.


_COPULAS = ['gaussian', 't']  # Supported copulas

# pyYaml can't serialize numpy float64 for some reason. Add custom representer.
def np64_representer(dumper, data):
    return dumper.represent_float(float(data))  # Just convert to regular float.
def npi64_representer(dumper, data):
    return dumper.represent_int(int(data))
yaml.add_representer(np.float64, np64_representer)
yaml.add_representer(np.int64, npi64_representer)


def multivariate_t_rvs(mean, corr, df=np.inf, size=1):
    ''' Generate random variables from multivariate Student t distribution.
        Not implemented in Scipy. Code taken from scikits package.

        Parameters
        ----------
        mean:   array_like, shape (M,)
                Mean of random variables
        corr:   array_like, shape (M,M)
                Correlation matrix
        df:     int, optional
                degrees of freedom
        size:   int, optional
                Number of samples for output array

        Returns
        -------
        rvs:    array_like shape (size, M)
                Correlated random variables
    '''
    mean = np.asarray(mean)
    d = len(mean)
    if df == np.inf:
        x = 1.
    else:
        x = np.random.chisquare(df, size)/df
    z = np.random.multivariate_normal(np.zeros(d), corr, (size,))
    return mean + z / np.sqrt(x)[:, None]


class InputVar(object):
    ''' Input variable class. Contains a mean/expected value and a list of uncertainty components.

        Parameters
        ----------
        name: string
            Name of the variable
        nom: float
            Nominal value for the variable
        desc: string
            Description of the variable
        units: string
            Unit name as string. Must be parsable by pint.UnitRegistry.parse_units.

        Attributes
        ----------
        sampledvalues: array
            Monte Carlo samples of this input variable
        '''
    def __init__(self, name, nom=1, units=None, desc=''):
        self.name = name
        self.nom = nom
        self.desc = desc
        self.uncerts = []
        self.normsamples = None
        self.sampledvalues = None
        self.units = uparser.parse_unit(units)

    def __repr__(self):
        return '<InputVar> {} = {}{}'.format(self.name, self.nom, report.Unit(self.units).plaintext())

    def clear(self):
        ''' Clear sampled values '''
        self.normsamples = None
        self.sampledvalues = None

    def stdunc(self):
        ''' Get combined standard uncertainty of this variable. Multiple components
            are RSS'd together.

            Returns
            -------
            Combined standard uncertainty: float
        '''
        if str(self.units) in ['degC', 'degF', 'celsius', 'fahrenheit']:
            dfltunits = getattr(ureg, 'delta_'+str(self.units))   # stdev of temperature units must be delta_
        else:
            dfltunits = self.units

        if len(self.uncerts) == 0:
            return 0 * dfltunits
        var = [u.var() for u in self.uncerts]
        var = sum(u for u in var if np.isfinite(u))   # squaring and np.nansum wont work if units are different. Loop it.
        if var == 0:
            return 0 * dfltunits
        return np.sqrt(var)

    def mean(self):
        ''' Get nominal value of variable

            Returns
            -------
            Mean value: float
        '''
        return self.nom * self.units

    def degf(self):
        ''' Get effective degrees of freedom, combining all uncertainty components
            using Welch-Satterthwaite equation.

            Returns
            -------
            Degrees of freedom: float
        '''
        if len(self.uncerts) == 0:
            return np.inf

        num2 = np.nansum([u.var().magnitude for u in self.uncerts])
        denom = np.nansum([u.var().magnitude**2 / u.degf for u in self.uncerts])
        return np.float64(num2)**2 / denom  # Use numpy float so div/0 returns inf.

    def add_comp(self, name, dist='normal', degf=np.inf, desc='', units=None, **args):
        ''' Add uncertainty component to this variable. If uncertainty name
            already exists, will be overwritten.

            Parameters
            ----------
            name: string
                Uncertainty name
            dist: string or scipy.stats.rv_continuous
                Distribution definition. Can be name of distribution in scipy.stats,
                name of distribution in distributions, or an instance
                of scipy.stats.rv_continuous or scipy.stats_rv_discrete.
            degf: float
                Degrees of freedom
            desc: string, optional
                Description of uncertainty component

            Keyword Arguments
            -----------------
            variable:
                Keyword arguments passed to the distribution in scipy.stats.
                Special cases for normal or t distributions can use 'std' or ('unc' with 'k')
        '''
        unames = [u.name for u in self.uncerts]
        if name in unames:   # Existing component, update it
            idx = unames.index(name)
            if units is None:
                units = str(self.uncerts[idx].units)
            self.uncerts[idx] = InputUncert(name, dist=dist, degf=degf, desc=desc, nom=self.nom, units=units, parentunits=str(self.units), **args)
        else:   # New component

            if units is None:
                units = str(self.units)  # Default to nominal variable units
                if units in ['degC', 'degF', 'celsius', 'fahrenheit']:   # But uncertainty of temperature units should default to delta_temperature units.
                    units = 'delta_{}'.format(units)

            self.uncerts.append(InputUncert(name, dist=dist, degf=degf, desc=desc, nom=self.nom, units=units, parentunits=str(self.units), **args))
        self.normsamples = None
        self.sampledvalues = None
        return self.uncerts[-1]

    def get_comp(self, name=None):
        ''' Get uncertainty component by name '''
        if name is None:
            return self.uncerts[0]
        else:
            unames = [u.name for u in self.uncerts]
            return self.uncerts[unames.index(name)]

    def rem_comp(self, index):
        ''' Remove uncertainty component at index '''
        self.uncerts.pop(index)
        self.normsamples = None
        self.sampledvalues = None

    def set_units(self, units):
        ''' Set the units parameter (does not change nominal value). Return True if successful. '''
        try:
            self.units = uparser.parse_unit(units)
        except ValueError:
            return False

        # Units trickle down to uncertainty components if not set already
        # OR if uncert has incompatible units
        for unc in self.uncerts:
            unc.parentunits = self.units
            if unc.units is None or unc.units.dimensionality != self.units.dimensionality:
                if units in ['degC', 'degF', 'celsius', 'fahrenheit']:   # Temperature units are always delta_ type
                    units = 'delta_'+units
                unc.set_units(units)

        self.normsamples = None
        self.sampledvalues = None
        return True

    def set_nom(self, mean):
        ''' Set the expected (nominal) value for this variable '''
        if isinstance(mean, str):
            try:
                self.nom = float(uparser.callf(mean))
            except (AttributeError, ValueError, TypeError, OverflowError):
                return False  # Don't change - error
        else:
            self.nom = mean

        for u in self.uncerts:
            u.nom = self.nom
        self.normsamples = None
        self.sampledvalues = None
        return True

    def _set_normsamples(self, normsamples):
        ''' Set normalized samples used for multivariate normals. '''
        self.normsamples = normsamples
        for u in self.uncerts:
            u.normsamples = normsamples

    def sample(self, samples=1000000):
        ''' Generate random samples for this variable, stored in self.sampledvalues. '''
        if self.sampledvalues is None:
            self.sampledvalues = np.ones(samples) * self.nom * self.units
            for u in self.uncerts:
                self.sampledvalues += u.sample(samples=samples, inc_nom=False)
        return self.sampledvalues

    def get_latex(self):
        ''' Get variable representation in latex format

            Returns
            -------
            latex expression: str
        '''
        return sympy.latex(sympy.Symbol(self.name))

    def get_unitstr(self):
        ''' Get units as string '''
        return report.Unit(self.units).prettytext()


class InputUncert(object):
    ''' Input variable class. Stores the variable name, distribution, and random
        samples.

        Parameters
        ----------
        name: string
            Name for the variable, to be used as argument to y function
        nom: float
            Nominal value. Will be an offset to loc param if needed. If multiple uncerts
            in a system, can be left out to add in at the end.
        dist: string or scipy.stats.rv_continuous
            Distribution definition. Can be name of distribution in scipy.stats,
            name of distribution in distributions, or an instance
            of scipy.stats.rv_continuous or scipy.stats_rv_discrete.
        degf: float
            Degrees of freedom

        Keyword Arguments
        -----------------
        variable:
            Keyword arguments passed to the distribution in scipy.stats.
            Special cases for normal or t distributions can use 'std' or ('unc' with 'k')

        Attributes
        ----------
        sampledvalues: array
            Sampled values for this uncertainty. May be centered about 0 or about the
            nominal value of parent InputVar.
    '''
    def __init__(self, name, nom=0, dist='normal', degf=np.inf, units=None, parentunits=None, desc='', **args):
        self.name = name
        self.distname = dist
        self.degf = degf
        self.nom = nom
        self.parentunits = uparser.parse_unit(parentunits)
        self.args = args.copy()  # User-entered arguments   (e.g. '5%' with nom=100)
        self.savedargs = {}      # Saved argument entries keep if distribution changes then changes back.
        self.required_args = []
        self.sampledvalues = None
        self.normsamples = None  # Normalized gaussian multivariate samples..
        self.desc = desc
        self.units = uparser.parse_unit(units)
        self.set_dist(dist)

    def __str__(self):
        return self.name

    def __repr__(self):
        return '<InputUnc> ' + self.name

    def set_dist(self, distname):
        ''' Set distribution.

            Parameters
            ----------
            dist: string or scipy.stats.rv_continuous
                Distribution definition. Can be name of distribution in scipy.stats,
                name of distribution in distributions, or an instance
                of scipy.stats.rv_continuous or scipy.stats_rv_discrete.
        '''
        self.distname = distname
        self.updateparams()

    def updateparams(self):
        ''' Update the distribution function from the provided arguments. '''
        self.savedargs.update(self.args)

        if self.distname == 'gaussian':
            self.distname = 'normal'  # Gaussian, normal, whatever

        distargs = self.args.copy()
        self.required_args = distributions.get_argnames(self.distname)
        for aname, aval in distargs.items():
            if isinstance(aval, str):  # Convert string arguments to float
                nom = (self.nom*self.parentunits).to(self.units).magnitude  # Convert nominal to same units as uncertainty

                # Allow entering % as % of nominal
                # or %range(X) as percent of X range
                # or ppm as ppm of nominal
                # or ppmrange(X) as ppm of X range
                aval = aval.replace('%range(', '/100*(')
                aval = aval.replace('ppmrange(', '/1E6*(')
                aval = aval.replace('ppbrange(', '/1E9*(')
                aval = aval.replace('ppm', '/1E6*{}'.format(nom))
                aval = aval.replace('ppb', '/1E9*{}'.format(nom))
                val = aval.replace('%', '/100*{}'.format(nom))
                try:
                    val = float(uparser.callf(val))
                except (AttributeError, ValueError, TypeError):
                    val = np.nan
                except OverflowError:
                    val = np.inf
                distargs[aname] = val

        if 'df' in self.required_args and 'df' not in distargs:
            distargs['df'] = self.degf
        elif 'df' in self.args:
            self.degf = float(distargs['df'])

        self.distribution = distributions.get_distribution(self.distname, **distargs)
        self.sampledvalues = None
        self.normsamples = None

        return self.check_args()   # True if args are all ok.

    def set_units(self, units):
        ''' Change units (without changing any values) '''
        # Uncertainties always use differential temperature units
        if units in ['degC', 'celsius']:
            units = 'delta_degC'
        elif units in ['degF', 'fahrenheit']:
            units = 'delta_degF'

        try:
            units = uparser.parse_unit(units)
        except ValueError:
            return False

        if self.parentunits and self.parentunits.dimensionality != units.dimensionality:
            return False

        self.units = units
        return True

    def pdf(self, stds=4, inc_nom=True):
        ''' Get probability density function as array (x, y)

            Parameters
            ----------
            stds: float
                Number of standard deviations to include in x range
            inc_nom: boolean
                Shift distribution by nominal value of input

            Returns
            x: 1D array
                X values, in terms of parent variable's units
            y: 1D array
                probability density values

            Note
            ----
            Discrete distribution types (i.e. scipy.stats.rv_discrete) do not
            have PDF functions, in this case a Monte Carlo approximation of the PDF
            will be returned.
        '''
        s = self.distribution.std()
        med = self.distribution.median()
        x = np.linspace(med-s*stds, med+s*stds, num=100)*self.units
        try:
            y = self.distribution.pdf(x.magnitude)
        except AttributeError:
            # Discrete dists don't have PDF
            try:
                samples = self.distribution.rvs(1000000).astype(float) * self.units
                if inc_nom:
                    samples += (self.nom*self.parentunits)
                    samples.to(self.parentunits).magnitude
                y, x = np.histogram(samples, bins=200)
                x = x[1:]
            except ValueError:
                x, y = [], []
        else:
            if inc_nom:
                x = x + (self.nom*self.parentunits)
                x.to(self.parentunits).magnitude
        return x, y

    def sample(self, samples=1000000, inc_nom=True):
        ''' Generate random samples from the distribution. Stored in self.sampledvalues.

            Parameters
            ----------
            samples: int
                Number of Monte-Carlo samples
            inc_nom:
                Sample values about nominal value instead of 0
        '''
        if self.normsamples is None:  # Uncorrelated variables
            self.sampledvalues = self.distribution.rvs(size=samples)
        else:  # Input has been correlated using correlate_inputs()
            self.sampledvalues = self.distribution.ppf(self.normsamples)
        if inc_nom:
            self.sampledvalues += self.nom
        return self.sampledvalues * self.units

    def clear(self):
        ''' Clear the Monte Carlo samples '''
        self.sampledvalues = None
        self.normsamples = None

    def std(self):
        ''' Return the standard deviation of the distribution function '''
        return self.distribution.std() * self.units

    def var(self):
        ''' Return the variance of the distribution function'''
        return self.distribution.var() * self.units**2

    def check_args(self):
        ''' Return true if all arguments are valid. '''
        try:
            self.distribution.rvs()
        except (AttributeError, ValueError):
            return False
        return True

    def get_latex(self):
        ''' Get latex representation of the uncertainty name

            Returns
            -------
            latex expression: str
        '''
        return sympy.latex(sympy.Symbol(self.name))

    def get_unitstr(self):
        ''' Get units as string '''
        return report.Unit(self.units).prettytext()


class InputFunc(object):
    ''' Class for an input function. Can be used as an InputVar.

        Parameters
        ----------
        function: string, sympy expression, or callable
            The function to evaluate
        variables: list of InputVar objects
            The variables (and any other InputFunc objects) used to
            compute the value of this function
        name: string
            Name for the function. Required if this function is used
            as an input to other functions.
        desc: string
            Description

        Attributes
        ----------
        function: sympy or callable
            The function as callable expression
        ftype: str
            Function type 'sympy' or 'callable'
        sampledvalues: array
            Monte Carlo samples of function value
        '''
    def __init__(self, function=None, variables=None, outunits=None, name=None, desc=''):
        self.variables = variables if variables is not None else []
        self.output = {}  # Dictionary of output objects for various methods
        self.sampledvalues = None
        self.desc = desc
        self.outunits = outunits   # CAN be None to leave units alone
        self.show = True

        if isinstance(function, (sympy.Basic, str)):
            self.origfunction = function  # Keep original, un-sympyfied string
            if not isinstance(function, sympy.Basic):
                # Not sympy expression, convert it
                function = uparser.parse_math(function, name=name)
            self.ftype = 'sympy'
            self.function = function

            self.name = name if name is not None else ''
            if name is not None:
                self.origfunction = '{} = {}'.format(name, self.origfunction)

        elif callable(function):
            self.ftype = 'callable'
            self.function = function
            self.origfunction = function
            self.kwnames = None   # Names of keyword arguments if function is callable
            if name is None:
                if hasattr(function, '__name__'):
                    self.name = function.__name__
                elif isinstance(function, np.vectorize):
                    # Get here using UncertCalc(np.vectorize(mufunc)), for example.
                    self.name = function.pyfunc.__name__
                else:
                    self.name = 'func'   # Shouldn't get here?
            else:
                self.name = name
        else:
            raise ValueError('Unknown function type')

    def __str__(self):
        ''' Get string representation of the function. Could be name parameter,
            or sympy string
        '''
        if self.ftype == 'sympy' and self.name != '':
            return '{} = {}'.format(self.name, str(self.function))
        if self.name == '' or self.name is None:
            return str(self.function)
        return self.name

    def __repr__(self):
        return '<InputFunc> ' + self.__str__()

    def clear(self):
        ''' Clear the output data and sampled values '''
        self.output = {}
        self.out = None
        self.sampledvalues = None
        for i in self.variables:
            if i.sampledvalues is not None:
                i.clear()

    def sample(self, samples=1000000):
        ''' Pull random samples from all inputs and combine. Returns array of samples. '''
        self.varsamples = {}
        for i in self.get_basevars():
            self.varsamples[i.name] = i.sample(samples)

        basefunc = self.get_basefunc()
        try:
            self.sampledvalues = uparser.callf(basefunc, self.varsamples)
            # Must have N samples in, N samples out... but only if there's actually variables in the equation.
            if len(self.varsamples) > 0 and len(self.sampledvalues) != samples: raise ValueError
        except DimensionalityError:
            # Hack around Pint bug/inconsistency (see https://github.com/hgrecco/pint/issues/670, closed without solution)
            #   with x = np.arange(5) * ureg.dimensionless
            #   np.exp(x) --> returns dimensionless array
            #   2**x --> raises DimensionalityError
            # Since units/dimensionality has already been verified, this fix strips units and adds them back.
            varsamples = {name: val.magnitude for name, val in self.varsamples.items()}
            self.sampledvalues = uparser.callf(basefunc, varsamples) * ureg.dimensionless

        except (TypeError, ValueError):
            # Call might have failed if basefunc is not vectorizable. Use numpy vectorize
            # to broadcast over array and try again.
            logging.info('Vectorizing function {}...'.format(str(basefunc)))

            # Vectorize will strip units - see https://github.com/hgrecco/pint/issues/828.
            # First, run a single sample through the function to determine what units come out
            outsingle = uparser.callf(basefunc, {k: v[0] for k, v in self.varsamples.items()})
            mcoutunits = outsingle.units if hasattr(outsingle, 'units') else ureg.dimensionless

            # Then apply those units to whole array of sampled values.
            # vectorize() will issue a UnitStripped warning, but we're handling it outside Pint, so ignore it.
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                self.sampledvalues = uparser.callf(np.vectorize(basefunc), self.varsamples) * mcoutunits

            # Convert to desired output units if necessary
            if self.outunits is not None and mcoutunits != self.outunits:
                self.sampledvalues.ito(self.outunits)

        return self.sampledvalues

    def mean(self):
        ''' Evaluate the function at the point specified by input nominal values.

            Returns
            -------
            Mean value: float
        '''
        return uparser.callf(self.get_basefunc(), self.get_basemeans())

    def stdunc(self):
        ''' Evaluate standard deviation of the function. Uses GUM value if available, or MC if not.

            Returns
            -------
            Standard Uncertainty: float
        '''
        if 'gum' in self.output:
            return self.output['gum'].uncert
        elif 'mc' in self.output:
            return self.output['mc'].uncert
        self.calc_GUM()
        return self.output['gum'].uncert

    def degf(self):
        ''' Get effective degrees of freedom '''
        return self._degf

    def _gradient(self):
        ''' Compute the gradient at the point specified by inputs.

            Returns
            -------
            grad: array
                Nx1 array of gradient values wrt each variable
            method: string
                Gradient method, either 'symbolic' or 'numeric'
        '''
        if self.ftype == 'sympy':
            with suppress(ZeroDivisionError, ValueError):
                return self._symgradient(), 'symbolic'
        # Fall back to numeric
        return self._numgradient(), 'numeric'

    def _numgradient(self):
        ''' Calculate gradient of function at the mean point numerically, using dx = 1ppm.
            Units are stripped, returning only magnitudes.
        '''
        basemeans = self.get_basemeans()
        basenames = self.get_basenames()
        basefunc = self.get_basefunc()
        baseuncerts = {k: v.to(basemeans[k]).magnitude for k, v in self.get_baseuncerts().items()}
        grad = []
        for name in basenames:
            d1 = basemeans.copy()
            dx = np.float64(baseuncerts[name] / 1E6) * d1[name].units # Float64 won't have div/0 errors, will go to inf
            if dx == 0:
                dx = 1E-6 * d1[name].units
            d1[name] = d1[name] + dx
            d2 = basemeans.copy()
            d2[name] = d2[name] - dx
            val = (uparser.callf(basefunc, d1) - uparser.callf(basefunc, d2))/(2*dx)
            grad.append(val.to_reduced_units())
        return grad

    def _symgradient(self):
        ''' Calculate gradient of function at the mean point using symbolic derivative.
            Units are stripped, returning only magnitudes.
        '''
        basemeans = self.get_basemeans()
        basenames = self.get_basenames()
        basefunc = self.get_basefunc()

        grad = []
        for x in basenames:
            try:
                dfunc = sympy.Derivative(basefunc, sympy.Symbol(x), evaluate=True)
                f = sympy.lambdify(basenames, dfunc, 'numpy')
                grad.append(f(**basemeans))
            except OverflowError:
                grad.append(None)

        if None in grad:
            raise ValueError('Unable to compute symbolic gradient')
        return grad

    def get_basefunc(self):
        ''' Get the base function. If function is sympy expression,
            it will be reduced to it's simplest form with only variables
            (no chaining of functions).

            Returns
            -------
            sympy or callable:
                Sympy expression or python callable, depending on function type
        '''
        func = self.function
        if self.ftype == 'callable':
            return func

        # Recursively substitute each function in until the function doesn't change anymore
        oldfunc = None
        count = 0
        while oldfunc != func and count < 100:
            oldfunc = func
            for v in reversed(self.variables):
                if isinstance(v, InputFunc) and v.ftype == 'sympy':
                    func = func.subs(v.name, v.function)
            count += 1
        if count >= 100:
            raise RecursionError('Circular reference in function set')
        return func

    def get_basevars(self):
        ''' Get base variables used by this function.

            Returns
            -------
            list
                List of InputVar objects.
        '''
        if self.ftype == 'sympy':
            free = [s.name for s in self.get_basefunc().free_symbols]
            return [v for v in self.variables if v.name in free]
        else:
            if self.kwnames:
                names = self.kwnames
            else:
                names = inspect.signature(self.function).parameters.keys()
            return [v for v in self.variables if v.name in names]

    def get_basesymbols(self):
        ''' Get symbols for each base variable

            Returns
            -------
            dict:
                Dictionary of {variablename: sympy symbol} for each input variable
        '''
        return [sympy.Symbol(v.name) for v in self.get_basevars()]

    def get_basemeans(self):
        ''' Get mean values of base variables

            Returns
            -------
            dict:
                Dictionary of {variablename: mean} for each input variable
        '''
        return dict((i.name, i.mean()) for i in self.get_basevars())

    def get_baseuncerts(self):
        ''' Get standard uncertainty values of base variables

            Returns
            -------
            dict:
                Dictionary of {variablename: standard uncertainty} for each input variable
        '''
        return dict((i.name, i.stdunc()) for i in self.get_basevars())

    def get_basenames(self):
        ''' Get names of base variables

            Returns
            -------
            list of string
                List of names of each base variable
        '''
        return [v.name for v in self.get_basevars()]

    def get_latex(self):
        ''' Get latex representation of function

            Returns
            -------
            latex expression: str
        '''
        if self.name == '':
            return sympy.latex(self.function)
        else:
            return sympy.latex(uparser.parse_math(self.name, raiseonerr=False))

    def full_func(self):
        ''' Build function as "f = x + y" sympy expression (instead of just "x + y") using InputFunc object '''
        if self.ftype == 'callable' or self.ftype == 'array':
            func = self.name
        elif self.name != '':
            func = sympy.Eq(sympy.Symbol(self.name), self.function)
        else:
            func = self.function
        return func

    def calculate(self, **kwargs):
        ''' Calculate all methods (gum, mc) '''
        samples = kwargs.get('samples', 1000000)
        gum = kwargs.get('gum', True)
        mc = kwargs.get('mc', True)
        lsq = kwargs.get('lsq', True)
        self.correlation = kwargs.get('correlation', None)
        sens = kwargs.get('sensitivity', True)
        outputs = []
        if gum and hasattr(self, 'calc_GUM'):
            gumout = self.calc_GUM(correlation=self.correlation)
            outputs.append(gumout)
        if mc and hasattr(self, 'calc_MC'):
            mcout = self.calc_MC(samples=samples, sensitivity=sens)
            outputs.append(mcout)
        if lsq and hasattr(self, 'calc_LSQ'):
            lsqout = self.calc_LSQ()
            outputs.append(lsqout)

        self.out = out_uncert.FuncOutput(outputs, self)
        return self.out

    def calc_GUM(self, correlation=None, calc_sym=True):
        ''' Calculate uncertainty using GUM method.

            Parameters
            ----------
            correlation: array
                Correlation matrix to apply
            calc_sym: boolean
                Calculate symbolic solution (turn off to speed up calculation)
        '''
        warns = []
        basevars = self.get_basevars()
        grad, gradtype = self._gradient()
        if gradtype == 'numeric' and self.ftype == 'sympy':
            warns.append('Symbolic gradient failed. Using numeric.')
        u = [i.stdunc() for i in basevars]
        try:
            mean = self.mean()
            origunits = mean.units if hasattr(mean, 'units') else ureg.dimensionless
            units = origunits

            # If Pint supported numpy arrays with different units/dimensions in a single array
            # We could compute uncertainty (GUM eq 13) in a single line:
            #     corr = np.triu(correlation) + np.triu(correlation).T - np.diag(correlation.diagonal())
            #     uncert = np.sqrt(grad.dot(np.outer(u, u) * corr).dot(grad))
            # But it doesn't so we have to loop it
            uncert = 0
            for dfdx, uval in zip(grad, u):
                uncert += dfdx**2 * uval**2

            if correlation is not None:
                for i in range(len(basevars)):
                    for j in range(i+1, len(basevars)):
                        uncert += 2 * grad[i] * grad[j] * u[i] * u[j] * correlation[i, j]
            uncert = np.sqrt(uncert)
            if not hasattr(uncert, 'units'):
                uncert = uncert * ureg.dimensionless

            # Convert to desired units
            if self.outunits is not None:
                units = ureg.parse_units(self.outunits)
                mean.ito(units)
                uncertunits = units if str(units) not in ['degF', 'degC', 'celsius', 'fahrenheit'] else f'delta_{units}'
                uncert.ito(uncertunits)
            else:
                try:
                    mean.ito_reduced_units()
                except AttributeError:   # Some callable function types won't pass through units
                    mean *= ureg.dimensionless
                units = mean.units
                uncertunits = units if str(units) not in ['degF', 'degC', 'celsius', 'fahrenheit'] else f'delta_{units}'
                uncert.ito(uncertunits)
        except OverflowError:
            warns.append('Overflow in GUM uncertainty calculation')
            units = ureg.dimensionless
            uncert = np.nan * units
            mean = np.nan * units

        # Pint/Numpy cant do arrays with differing units. Must do them the hard way.
        gradu = []
        for x, y in zip(grad, u):
            try:
                gradu.append((x*y).to_reduced_units())
            except DimensionalityError:
                # Pint can crash on to_reduced_units() when units get complicated
                # See https://github.com/hgrecco/pint/issues/774
                # and https://github.com/hgrecco/pint/issues/771
                gradu.append((x*y).to_compact())
        gradu2 = [g*g for g in gradu]           # numpy: gradu2 = gradu * gradu
        uncert2 = uncert*uncert
        g2overuncert2 = [(x/uncert2).to_reduced_units() for x in gradu2]  # numpy: g2overuncert2 = gradu2 / uncert2

        if not np.isfinite(uncert) or uncert == 0:
            residual = np.nan
            props = np.zeros(len(u)) * np.nan
        else:
            residual = 1 - sum(g2overuncert2).magnitude
            if abs(residual) < 1E-7: residual = 0
            residual = residual * 100
            #props = gradu2 / (uncert*uncert) * 100
            props = [g.magnitude*100 for g in g2overuncert2]

        idegf = np.array([i.degf() for i in basevars])
        gidegf = [x**4/i for x, i in zip(gradu, idegf)]
        gidegf = [g for g in gidegf if np.isfinite(g)]
        try:
            self._degf = (uncert**4 / sum(gidegf)).magnitude   # numpy: uncert**4 / np.nansum(gradu**4 / idegf)
        except AttributeError:
            self._degf = np.inf

        params = {'function': str(self),
                  'latex': self.get_latex(),
                  'mean': mean,
                  'uncert': uncert,
                  'props': props,
                  'residual': residual,
                  'sensitivity': grad,
                  'ui': u,
                  'degf': self._degf,
                  'units': units,
                  'origunits': origunits,   # original "natural" units, result of gum equation before converting to desired output units
                  'name': self.name,
                  'desc': self.desc,
                  'warns': warns}

        if self.ftype == 'sympy' and calc_sym:
            symout = self.calc_SYM(correlation=correlation, uncert=uncert, degf=self._degf)
            params['symbolic'] = symout

        self.output['gum'] = out_uncert.create_output(method='gum', **params)
        return self.output['gum']

    def calc_MC(self, samples=1000000, sensitivity=True):
        ''' Calculate Monte Carlo uncertainty for the function.

            Parameters
            ----------
            samples: int
                Number of samples
            sensitivity: boolean
                Calculate Monte Carlo sensitivity coefficients (requires running a MC for each variable)
        '''
        warns = []
        self.sample(samples)

        y = self.sampledvalues
        try:
            if self.outunits is not None:
                y.ito(ureg.parse_units(self.outunits))
            else:
                y.ito_reduced_units()
        except AttributeError:      # Callable types may not pass through units
            y *= ureg.dimensionless
        units = y.units
        uncertunits = units if str(units) not in ['degF', 'degC', 'celsius', 'fahrenheit'] else ureg.parse_units(f'delta_{units}')
        y = np.array(y.magnitude, ndmin=1, copy=False) * y.units  # MC constants may come through as scalars

        if not all(np.isfinite(y.magnitude)):
            warns.append('Some Monte-Carlo samples are NaN. Ignoring in statistics.')

        stdY = np.nanstd(y.magnitude, ddof=1) * uncertunits
        meanY = np.nanmean(y.magnitude) * units
        medY = np.nanmedian(y.magnitude) * units

        sens = []
        props = []
        if sensitivity:
            # Calculate non-linear sensitivity coefficients
            xvals = self.get_basemeans()
            basefunc = self.get_basefunc()

            # Keep all inputs the same shape
            for x in xvals.keys():
                xvals[x] = np.full(samples, xvals[x].magnitude) * xvals[x].units

            for v in self.get_basevars():  # Hold all inputs fixed but one, compare output stddev.
                x = xvals.copy()
                x[v.name] = v.sampledvalues
                try:
                    ustd = uparser.callf(basefunc, x).std()
                except DimensionalityError:
                    x = {name: val.magnitude for name, val in x.items()}
                    ustd = uparser.callf(basefunc, x).std()
                except AttributeError:
                    ustd = 0   # callf returns single value.. no std
                except (TypeError, ValueError):
                    # Call might have failed if basefunc is not vectorizable. Use numpy vectorize
                    # to broadcast over array and try again.
                    ustd = uparser.callf(np.vectorize(basefunc), x).std()
                sens.append((ustd/v.stdunc()).to_base_units().to_reduced_units())   # to_reduced_units() will take care of prefix multipliers and dimensionless values
                props.append((ustd/stdY).to_base_units().to_reduced_units().magnitude**2 * 100)

        params = {'mean': meanY,
                  'expected': self.mean(),   # expected/measured value may not match mean or median of MC distribution
                  'uncert': stdY,
                  'median': medY,
                  'samples': y,
                  'units': units,
                  'sensitivity': sens if len(sens) else None,
                  'props': props if len(props) else None,
                  'function': str(self),
                  'latex': self.get_latex(),
                  'desc': self.desc,
                  'inputs': self.get_basenames(),
                  'warns': warns
                  }
        self.output['mc'] = out_uncert.create_output(method='mc', **params)
        return self.output['mc']

    def calc_SYM(self, correlation=None, uncert=None, degf=None):
        ''' Calculate uncertainty symbolically.

            Parameters
            ----------
            correlation: array, optional
                Correlation matrix
            uncert: float
                GUM calculated uncertainty
            degf: float
                Degrees of freedom
        '''
        if self.ftype == 'callable':
            raise ValueError('Cant symbolically compute a callable input function')

        use_corr = correlation is not None and not np.allclose(np.identity(len(correlation)), correlation)
        basesymbols = self.get_basesymbols()
        basefunc = self.get_basefunc()
        Xmeans = self.get_basemeans()
        Xuncerts = [var.stdunc() for var in self.get_basevars()]

        if self.name != '':
            fsymbol = sympy.Symbol(self.name)
            uncfsymbol = sympy.Symbol('u_{'+self.name+'}')
        else:
            fsymbol = self.function
            uncfsymbol = sympy.Symbol('u_c')

        partials = []        # d/dy reduced
        partials_raw = []    # will look like [d/dy f]
        for x in basesymbols:
            p = sympy.Derivative(fsymbol, x)
            partials_raw.append(p)
            # SYMPY 1.2 CHANGE - derivative.subs(x, y) will substitute AFTER derivative is taken, leading to 0! Use replace() instead.
            partials.append(sympy.Derivative(basefunc, x).doit())

        uncsymbols = [sympy.Symbol('u_{}'.format(str(x))) for x in basesymbols]
        senssymbols = [sympy.Symbol('c_{}'.format(str(x))) for x in basesymbols]
        terms = [p**2 * u**2 for p, u in zip(partials_raw, uncsymbols)]
        cx_terms = [c**2 * u**2 for c, u in zip(senssymbols, uncsymbols)]
        uncform = sympy.Add(*terms)        # Formula in terms of partial derivatives
        uncform_cx = sympy.Add(*cx_terms)  # Formula in terms of sensitivity coefficients c_x
        if use_corr:
            covterms = []
            covsymbols = []
            covvals = []
            for i in range(len(basesymbols)):
                for j in range(i+1, len(basesymbols)):
                    covsymbols.append(sympy.Symbol('sigma_{},{}'.format(basesymbols[i], basesymbols[j])))
                    covvals.append(correlation[i, j])
                    covterms.append(2 * sympy.Derivative(fsymbol, basesymbols[i]) * sympy.Derivative(fsymbol, basesymbols[j]) * uncsymbols[i] * uncsymbols[j] * covsymbols[-1])
            uncform = uncform + sympy.Add(*covterms)
            uncform_cx = uncform_cx + sympy.Add(*covterms)
        uncform = sympy.root(uncform, 2)
        uncform_cx = sympy.root(uncform_cx, 2)

        degfsymbols = [sympy.Symbol('nu_{}'.format(str(x))) for x in basesymbols]
        dterms = [(u*c)**4/v for u, c, v in zip(uncsymbols, senssymbols, degfsymbols)]
        degfform = uncfsymbol**4 / sympy.Add(*dterms)

        partials_solved = []
        for p in partials:
            try:
                partials_solved.append(sympy.lambdify(Xmeans.keys(), p, 'numpy')(**Xmeans))
            except (TypeError, ValueError, OverflowError, ZeroDivisionError):
                partials_solved.append(sympy.nan)

        symout = {'function': self.full_func(),
                  'partials': partials,
                  'partials_raw': partials_raw,  # Unsimplified partial derivatives
                  'partials_solved': partials_solved,
                  'uc_symbol': uncfsymbol,
                  'unc_formula': uncform,        # Unsimplified uncertainty formula
                  'unc_formula_sens': uncform_cx,  # Uncertainty formula in terms of sensitivity coefficients
                  'uncertainty': uncform.replace(fsymbol, basefunc).doit(),  # Simplified uncertainty formula
                  'unc_val': uncert,
                  'var_symbols': basesymbols,  # Symbols for each variable
                  'var_means': [Xmeans[str(k)] for k in basesymbols],
                  'var_uncerts': Xuncerts,
                  'unc_symbols': uncsymbols,   # Symbols for uncertainty of each variable
                  'degf': degfform,  # Formula for degrees of freedom
                  'degf_val': degf}
        if use_corr:
            symout['covsymbols'] = covsymbols
            symout['covvals'] = covvals
        return symout


class UncertCalc(object):
    ''' Main Uncertainty Calculator Class.

        Parameters
        ----------
        function: string, sympy, or callable, or list
            Function or list of functions to calculate. Each function may be a string,
            callable, or sympy expression.
        samples: int, optional
            Number of Monte-Carlo samples. Default to 1E6.
        seed: int, optional
            Random number seed. Seed will be randomized if None.
        units: string or list
            Unit names for each function
        inputs: list of dict, optional
            Input variable definitions. If omitted, inputs must be defined using set_input().
            Dictionary contains:

            =================   ===============================================
            Key                 Description
            =================   ===============================================
            name                string, name of variable
            nom                 float, nominal value of variable
            desc                string, description for variable
            uncerts             list of dictionaries. See below
            =================   ===============================================

            Where each entry in the 'uncerts' list is a dictionary with:

            =================   ===============================================
            Key                 Description
            =================   ===============================================
            name                string (optional), name of uncertainty component
            desc                string (optional), description of uncertainty component
            dist                string (optional, default='normal') distribution name
            degf                float (optional, default=Inf) degrees of freedom
            **args             Keyword arguments defining the distribution, passed to stats instance.
            =================   ===============================================

        Attributes
        ----------
        functions: list
            List of InputFunc instances to calculate
        variables: list
            List of InputVar and InputFunc objects defined in the system. InputFunc are included
            in case a function is used as an input to another function.
        samples: int
            Number of Monte Carlo samples to calculate
        '''
    def __init__(self, function=None, inputs=None, units=None, samples=1000000, seed=None, name='uncertainty'):
        self.variables = []  # List of InputVar objects AND named InputFunc objects
        # CAUTION: variables list is passed "by reference" to InputFuncs. Do not reassign anywhere or link will be broken!
        # Instead, reassign using list mutation (with [:]) like this:  self.variables[:] = XYZ.
        self.functions = []  # List of InputFunc objects
        self.samples = int(samples)
        self._corr = None
        self.copula = 'gaussian'
        self.seed = seed
        self.longdescription = ''
        self.name = name
        self.out = None   # Output object

        if isinstance(function, list) or isinstance(function, tuple):
            if units is None:
                units = [None]*len(function)
            assert len(units) == len(function)
            [self.set_function(f, outunits=u) for f, u in zip(function, units)]
        elif function is not None:
            self.set_function(function, outunits=units)

        if inputs is not None:
            for inpt in inputs:
                uncerts = inpt.get('uncerts', [])
                iname = inpt.get('name', '')
                units = inpt.get('units', None)
                self.set_input(iname, nom=inpt.get('nom', 0), desc=inpt.get('desc', ''), units=units)
                for udict in uncerts:
                    self.set_uncert(var=iname, **udict)

    def clearall(self):
        ''' Clear all inputs, resetting to blank calculator '''
        self.functions[:] = []
        self._corr = None
        self.longdescription = ''
        self.out = None
        self.variables[:] = []

    def set_function(self, func, idx=None, name=None, outunits=None, desc='', show=True, kwnames=None):
        ''' Add or update a function in the system.

            Parameters
            ----------
            func: string, callable, or sympy expression
                The function to add
            idx: int, optional
                Index of existing function. If not provided, the function names will be searched,
                or a new function will be added if not in the function names already.
            name: name of the function
            desc: description for the function
            show: boolean, show this function in the reports?
            kwnames: list
                List of names of keyword arguments to function if func is callable. Required
                only if func is callable and called with kwargs or args instead of named arguments.
        '''
        if name is None and isinstance(func, str) and '=' in func:
            name, func = func.split('=')
            name = name.strip()
            func = func.strip()
        elif name is None and hasattr(func, 'name') and func.name is not None:
            name = func.name
        elif name is not None and hasattr(func, 'name'):
            func.name = name

        fnames = [f.name for f in self.functions]
        if idx is None and name is not None and name in fnames:
            idx = fnames.index(name)
        elif idx is None and func in self.functions:
            idx = self.functions.index(func)

        if idx is not None and idx >= len(self.functions):
            idx = None

        if not hasattr(func, 'calc_GUM'):  # isinstance(func, InputFunc)  # Doesn't seem to work across uarray module
            func = InputFunc(func, variables=self.variables, outunits=outunits, name=name, desc=desc)  # self.variables list passed as object

        if func.ftype == 'callable':
            params = inspect.signature(func.function).parameters
            if any(p.kind != inspect.Parameter.POSITIONAL_OR_KEYWORD for p in params.values()):
                if kwnames is None:
                    raise ValueError('Callable function uses keyword arguments. kwnames must be supplied to set_function.')
                else:
                    func.kwnames = kwnames

        func.show = show
        if idx is None:  # Not an existing function, add it
            self.functions.append(func)
        else:
            self.functions[idx] = func

        inptnames = self.get_inputnames()
        if name is not None and name not in inptnames:  # It has a name, add it to variable list to be used in later functions
            self.variables.append(func)
        elif name in inptnames:  # Already in there, update it
            i = inptnames.index(name)
            self.variables[i] = func

    def reorder(self, names):
        ''' Reorder the function list using order of names array.

            Parameters
            ----------
            names: list of strings
                List of function names in new order
        '''
        fnames = [f.name for f in self.functions]
        newidx = [fnames.index(f) for f in names]
        self.functions = [self.functions[f] for f in newidx]
        self.variables[:] = self.functions + [v for v in self.variables if not isinstance(v, InputFunc)]

    def remove_function(self, idx):
        ''' Remove function at idx '''
        with suppress(IndexError):
            self.functions.pop(idx)

    def set_input(self, inpt, nom=1, units=None, desc='', **kwargs):
        ''' Add or update an input nominal value and description in the system.

            Parameters
            ----------
            inpt: string or InputVar
                Variable to update, either existing InputVar instance or
                string name of the variable
            nom: float
                Nominal value
            desc: string
                Description

            Keyword Arguments
            -----------------
            uname: str
                Name of an uncertainty component to add to this input
            dist: str
                Name of a distribution for the uname uncertainty component
            degf: float
                Degrees of freedom for the uncertainty component
            ...:
                Other keyword arguments passed to the distribution (e.g. 'std', 'unc', 'k', etc.)

            Note
            ----
            The Keyword Arguments will be passed to set_uncert() and may be used to set
            one uncertainty component on this input variable. Additional components
            may be added with explicit calls to set_uncert().
        '''
        idx = None
        names = self.get_inputnames()
        if inpt in self.variables:
            idx = self.variables.index(inpt)
        elif inpt in names:
            idx = names.index(inpt)

        if idx is None:  # Not an existing InputVar. Add it
            if not isinstance(inpt, InputVar):
                inpt = InputVar(name=inpt, nom=nom, desc=desc, units=units)
            self.variables.append(inpt)
            idx = -1
        else:  # Existing variable, update it
            self.variables[idx].nom = nom
            self.variables[idx].desc = desc
            if units is not None:
                self.variables[idx].units = uparser.parse_unit(units)

        if len(kwargs) > 0:
            self.set_uncert(var=self.variables[idx].name,
                            name=kwargs.pop('uname', 'u({})'.format(self.variables[idx].name)),
                            **kwargs)

        return self.variables[idx]

    def set_uncert(self, var, name=None, dist='normal', degf=np.inf, units=None, desc='', **args):
        ''' Add or update an uncertainty component.

            Parameters
            ----------
            var: string
                Name of input variable to add uncertainty to
            name: string
                Name for uncertainty component
            dist: string or rv_continuous instance
                Distribution for uncertainty component
            degf: float
                Degrees of freedom for this component
            desc: string
                Description of uncertainty

            Keyword Arguments
            -----------------
            variable:
                Keyword arguments defining distribution will be passed to the
                rv_continuous instance in scipy.stats.
        '''
        names = [v.name for v in self.variables]
        if var not in names:
            raise ValueError('Variable {} not defined'.format(var))

        idx = names.index(var)

        if name is None:
            name = '{}{}'.format(self.variables[idx].name, len(self.variables[idx].uncerts))

        self.variables[idx].add_comp(name=name, dist=dist, degf=degf, desc=desc, units=units, **args)

    def get_input(self, name):
        ''' Return input with the given name

            Parameters
            ----------
            name: str
                Name of the input to return

            Returns
            -------
            Input variable: InputVar
        '''
        inpts = [x for x in self.variables if x.name == name]
        if len(inpts) > 0:
            return inpts[0]
        else:
            return None

    def get_inputnames(self):
        ''' Get names of the input variables

            Returns
            -------
            Input names: list of str
        '''
        return [i.name for i in self.variables]

    def get_functionnames(self):
        ''' Get names of the functions

            Returns
            -------
            Function names: list of str
        '''
        return [i.name for i in self.functions]

    def get_baseinputs(self):
        ''' Get the base variables (only inputs, not dependent functions)

            Returns
            -------
            Input names: list of InputVar
        '''
        inpts = []
        for v in self.variables:
            if (isinstance(v, InputVar) or (len(self.functions) > 1 and isinstance(v, InputFunc) and v.ftype == 'callable')):
                inpts.append(v)
        return inpts

    def get_baseinputnames(self):
        ''' Get the base variable names (only inputs, not dependent functions)

            Returns
            -------
            Input names: list of str
        '''
        return [i.name for i in self.get_baseinputs()]

    def get_reqd_inputs(self):
        ''' Return a list of input names required to define the function(s). '''
        funcnames = [f.name for f in self.functions]

        inputs = set()
        for f in self.functions:
            if f.ftype == 'callable' and f.kwnames:
                inputs = inputs | set(f.kwnames)
            elif f.ftype == 'callable':
                keys = set(inspect.signature(f.function).parameters.keys())
                if 'kwargs' in keys:
                    keys.remove('kwargs')
                inputs = inputs | set(k for k in keys if k not in funcnames)
            elif f.ftype == 'array':
                pass
            else:
                inputs = inputs | set(str(s) for s in f.function.free_symbols if str(s) not in funcnames)

        return [str(i) for i in inputs]

    def add_required_inputs(self):
        ''' Add and remove inputs so that all free variables in functions are
            defined. New variables will default to normal, mean=0, std=1.
        '''
        inputs = self.get_reqd_inputs()
        self.variables[:] = [i for i in self.variables if i.name in inputs and isinstance(i, InputVar)]  # And remove ones that aren't there anymore
        [self.set_input(i, unc=1, k=2) for i in inputs if i not in [x.name for x in self.variables]]  # Add new variables
        self.variables.extend(f for f in self.functions if f.name != '')   # but keep named functions

    def set_correlation(self, corr, copula='gaussian', **args):
        ''' Set correlation of inputs as a matrix.

            Parameters
            ----------
            corr : array_like, shape (M,M)
                correlation matrix. Must be square where M is number of inputs.
                Only upper triangle is considered.
            copula : string
                Copula for correlating the inputs. Supported copulas:
                ['gaussian', 't']
            **args : variable
                copula arguments. Only df: degrees of freedom (t copula) is supported.
        '''
        assert copula in _COPULAS
        self._corr = corr
        self.copula = copula

    def set_copula(self, copula='gaussian'):
        ''' Set the copula for correlating inputs.

            Parameters
            ----------
            copula: string
                Name of coupla to use. Must be 'gaussian' or 't'.
        '''
        assert copula in _COPULAS
        self.copula = copula

    def clear_corr(self):
        ''' Clear correlation table '''
        self._corr = None

    def correlate_vars(self, var1, var2, correlation):
        ''' Set correlation between two inputs.

            Parameters
            ----------
            var1: string
            var2: string
                Names of the variables correlate
            correlation: float
                Correlation coefficient for the two variables
        '''
        if var1 == var2:
            return

        baseinputs = self.get_baseinputs()
        if self._corr is None:
            self._corr = np.eye(len(baseinputs))
        Xnames = [i.name for i in baseinputs]
        idx1 = Xnames.index(var1)
        idx2 = Xnames.index(var2)
        try:
            float(correlation)
        except ValueError:
            return  # Fail

        self._corr[idx1, idx2] = correlation
        self._corr[idx2, idx1] = correlation

    def get_corr_matrix(self, varnames=None):
        ''' Get correlation matrix for the given variable names '''
        basenames = self.get_baseinputnames()
        if varnames is None:
            varnames = basenames

        if self._corr is None:
            return np.eye(len(varnames))
        elif len(varnames) == len(basenames):
            return self._corr
        else:
            # Only rows/cols for varnames
            i = [v for v in range(len(varnames)) if varnames[v] in basenames]
            return self._corr[:, i][i, :]

    def _gen_corr_samples(self, **args):
        ''' Generate random samples for each of the inputs '''
        covok = True
        if self._corr is not None:
            # Matrix be symmetric. Copy upper triangle to lower triangle
            fullcorr = np.triu(self._corr) + np.triu(self._corr).T - np.diag(self._corr.diagonal())

            # Generate correlated random samples, save to each input
            if self.copula == 'gaussian':
                # Note: correlation==covariance since all std's are 1 right now.
                with warnings.catch_warnings(record=True) as w:
                    # Roundabout way of catching a numpy warning that should really be an exception
                    warnings.simplefilter('always')
                    normsamples = stat.multivariate_normal.rvs(cov=fullcorr, size=self.samples)
                    if len(w) > 0:
                        covok = False  # Not positive semi-definite
                    normsamples = stat.norm.cdf(normsamples)

            elif self.copula == 't':
                df = args.get('df', np.inf)
                mean = np.zeros(fullcorr.shape[0])
                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter('always')
                    normsamples = multivariate_t_rvs(mean=mean, corr=fullcorr, size=self.samples, df=df)
                    normsamples = stat.t.cdf(normsamples, df=df)
                    if len(w) > 0:
                        covok = False  # Not positive semi-definite

            normsamples[np.where(normsamples == 1.0)] = 1 - 1E-9  # If rounded to 1 or 0, then we get infinity.
            normsamples[np.where(normsamples == 0.0)] = 1E-9
            for idx, i in enumerate(self.get_baseinputs()):
                i._set_normsamples(normsamples[:, idx])
        return covok

    def clear_output(self):
        ''' Clear the output calculation results '''
        for f in self.functions:
            f.clear()

    def check_inputs(self):
        ''' Check that all inputs are defined.

            Raises
            ------
            ValueError
                if required inputs are not defined.
        '''
        definputs = [x.name for x in self.variables]
        reqinputs = self.get_reqd_inputs()
        for i in reqinputs:
            if i not in definputs:
                raise ValueError('Required input "{}" is not defined.'.format(i))

    def check_dimensionality(self):
        ''' Check that units/dimensionality are correct.

            Raises
            ------
            pint.DimensionalityError if units are incompatible
        '''
        for func in self.functions:
            mean = func.mean()
            if func.outunits is not None:
                if not hasattr(mean, 'units'):
                    mean = mean * ureg.dimensionless
                mean.to(func.outunits)
        for inpt in self.get_baseinputs():
            inpt.stdunc()

    def check_circular(self):
        ''' Check the functions list to ensure no circular dependencies

            Returns
            -------
            True if the function set has a circular reference
        '''
        if len(self.functions) > 1:
            for f in self.functions:
                try:
                    f.get_basefunc()
                except RecursionError:
                    return True
        return False

    def get_output(self):
        return self.out

    def calculate(self, **kwargs):
        ''' Run calculation and return output report.

            Parameters
            ----------
            gum: boolean
                Calculate using GUM method
            mc: boolean
                Calculate using Monte-Carlo method
            lsq: boolean
                Calculate using Least-Squares method, if available. InputFunc object
                must have calc_LSQ method.
            sensitivity: boolean
                Calculate Monte Carlo sensitivity coefficients (requires running a MC for each variable)
        '''
        mc = kwargs.get('mc', True)
        samples = kwargs.pop('samples', self.samples)
        warns = None
        self.clear_output()
        self.check_inputs()
        if mc:
            if self.seed is not None:
                np.random.seed(self.seed)
            if self._corr is not None:
                covok = self._gen_corr_samples()
                if not covok:
                    warns = ['Covariance matrix is not positive semi-definite.']

        outputs = []
        for f in self.functions:
            correlation = self.get_corr_matrix(f.get_basenames())
            funcout = f.calculate(samples=samples, correlation=correlation, **kwargs)
            outputs.append(funcout)
        self.out = out_uncert.CalcOutput(outputs, self, warns=warns)
        return self.out

    def calcGUM(self):
        ''' Calculate GUM results only. '''
        return self.calculate(mc=False, lsq=False)

    def calcMC(self, sensitivity=True):
        ''' Calculate Monte-Carlo Results Only.

            Parameters
            ----------
            sensitivity: boolean
                Calculate Monte Carlo sensitivity coefficients (requires running a MC for each variable)
        '''
        return self.calculate(gum=False, lsq=False, sensitivity=sensitivity)

    def get_corr_list(self):
        ''' Get correlations

            Returns
            -------
            list of (str, str, float) tuples:
                Each tuple lists one correlation coefficient as (input1, input2, correlation)
        '''
        s = []
        basenames = self.get_baseinputs()
        for idx, iname in enumerate(basenames):
            for idx2, iname2 in enumerate(basenames[idx+1:]):
                s.append((iname.name, iname2.name, self._corr[idx, idx+idx2+1]))
        return s

    def get_contour(self, func1, func2, getcorr=False):
        ''' Get contour grid for plotting of func1 vs func2

        Parameters
        ----------
        func1: int
            Index of first function
        func2: int
            Index of second function

        Returns
        -------
        x: array
            2D array of x points
        y: array
            2D array of y points
        pdf: array
            2D array of joint PDF values

        Note
        ----
        The output of the function can be plotted directly using matplotlib contour().

        >>> x, y, p = u.get_contour(0, 1)
        >>> plt.contour(x, y, p)
        '''
        Contour = namedtuple('Contour', ['x', 'y', 'pdf'])
        # Get mean/uncertainty in original units so they're compatible with sens coefficients
        m0 = self.functions[func1].out.gum.mean.to(self.functions[func1].out.gum.properties['origunits']).magnitude
        m1 = self.functions[func2].out.gum.mean.to(self.functions[func2].out.gum.properties['origunits']).magnitude
        u0 = self.functions[func1].out.gum.uncert.to(self.functions[func1].out.gum.properties['origunits']).magnitude
        u1 = self.functions[func2].out.gum.uncert.to(self.functions[func2].out.gum.properties['origunits']).magnitude

        # Make sensitivity tables the same shape/order - basevars may be different for each function - pad with 0's.
        s1dict = dict([v, c] for v, c in zip(self.functions[func1].get_basevars(), self.functions[func1].out.gum.sensitivity))
        s2dict = dict([v, c] for v, c in zip(self.functions[func2].get_basevars(), self.functions[func2].out.gum.sensitivity))
        s1dict = dict([v, c.magnitude if hasattr(c, 'magnitude') else c] for v, c in s1dict.items())
        s2dict = dict([v, c.magnitude if hasattr(c, 'magnitude') else c] for v, c in s2dict.items())
        baseinputs = self.get_baseinputs()
        s1 = []
        s2 = []
        for i, v in enumerate(baseinputs):
            s1.append(s1dict.get(v, 0))
        for i, v in enumerate(baseinputs):
            s2.append(s2dict.get(v, 0))

        # See 6.2 in GUM-sup-2
        Cx = np.vstack((s1, s2))
        corr = self._corr if self._corr is not None else np.eye(len(baseinputs))
        S = np.diag([i.stdunc().magnitude for i in self.get_baseinputs()])
        Ux = S @ corr @ S   # Convert correlation to covariance
        Uy = Cx @ Ux @ Cx.T

        if getcorr:
            return Uy[0, 1] / (u0*u1)

        try:
            rv = stat.multivariate_normal(np.array([m0, m1]), cov=Uy)
        except (ValueError, np.linalg.LinAlgError):
            return Contour(None, None, None)

        x, y = np.meshgrid(np.linspace(m0-3*u0, m0+3*u0), np.linspace(m1-3*u1, m1+3*u1))
        pos = np.dstack((x, y))
        x = (x*self.functions[func1].out.gum.properties['origunits']).to(self.functions[func1].out.gum.units)
        y = (y*self.functions[func2].out.gum.properties['origunits']).to(self.functions[func2].out.gum.units)
        return Contour(x, y, rv.pdf(pos))

    def get_config(self):
        ''' Get configuration dictionary describing this calculation '''
        if any([callable(f.origfunction) for f in self.functions]):
            raise ValueError('Cannot save callable function to config file.')

        d = {}
        d['name'] = self.name
        d['mode'] = 'uncertainty'
        d['samples'] = self.samples
        d['functions'] = []
        d['inputs'] = []

        for f in self.functions:
            fdict = {'name': f.name, 'expr': str(f.function), 'desc': f.desc, 'units': format(f.outunits) if f.outunits else None}
            d['functions'].append(fdict)

        for inpt in self.get_baseinputs():
            idict = {'name': inpt.name, 'mean': inpt.nom, 'desc': inpt.desc, 'units': format(inpt.units)}
            ulist = []
            if len(inpt.uncerts) > 0:
                for unc in inpt.uncerts:
                    udict = {'name': unc.name, 'desc': unc.desc, 'degf': unc.degf, 'units': format(unc.units) if unc.units else None}
                    if not isinstance(unc.distname, str):
                        udict.update(distributions.get_config(unc.distname))
                    else:
                        udict.update({'dist': unc.distname})
                        udict.update(dict(unc.args))
                    ulist.append(udict)
                idict['uncerts'] = ulist
            d['inputs'].append(idict)

        if self.seed is not None:
            d['seed'] = self.seed

        if self._corr is not None:
            d['correlations'] = []
            for v1, v2, cor in self.get_corr_list():
                d['correlations'].append({'var1': v1, 'var2': v2, 'cor': format(cor, '.4f')})

        if self.longdescription is not None and self.longdescription != '':
            d['description'] = self.longdescription
        return d

    def save_config(self, fname):
        ''' Save configuration to file.

            Parameters
            ----------
            fname: string or file object
                File name or open file object to write configuration to
        '''
        d = self.get_config()
        d = [d]  # Must go in list to support multi-calculation project structure
        out = yaml.dump(d, default_flow_style=False)  # Can't use safe_dump with our np.float64 representer. But we still safe_load.

        try:
            fname.write(out)
        except AttributeError:
            with open(fname, 'w') as f:
                f.write(out)

    def save_samples(self, fname, fmt='csv', inputs=True, outputs=True):
        ''' Save Monte-Carlo samples to file.

            Parameters
            ----------
            fname: string or file
                Filename or open file object to save
            fmt: string
                One of 'csv' or 'npz' for comma separated text or numpy binary format
            inputs: bool
                Save input samples?
            outputs: bool
                Save output samples?
        '''
        if inputs:
            variables = self.get_baseinputs()
            hdr = ['{}{}'.format(v.name, report.Unit(v.units).prettytext(bracket=True)) for v in variables]
            hdr.extend(['{}{}'.format(str(f), report.Unit(ureg.parse_units(f.outunits)).prettytext(bracket=True)) for f in self.functions])
            csv = np.stack([v.sampledvalues.magnitude for v in variables], axis=1)
        if outputs:
            out = np.array([self.out.foutputs[f].mc.samples.magnitude for f in range(len(self.functions))]).T
            if inputs:
                csv = np.hstack([csv, out])
            else:
                csv = out
        if fmt == 'csv':
            np.savetxt(fname, csv, delimiter=', ', fmt='%.8e', header=', '.join(hdr))
        elif fmt == 'npz':
            np.savez_compressed(fname, samples=csv, hdr=hdr)
        else:
            raise ValueError('Unsupported file format {}'.format(fmt))

    @classmethod
    def from_config(cls, config):
        ''' Set up calculator using configuration dictionary. '''

        def get_float(d, key, default=0):
            ''' Get float from dictionary key, return default if not found or if value can't be
                cast to a float.
            '''
            if key not in d:
                return default
            try:
                val = float(d[key])
            except ValueError:
                logging.warning('Cannot convert "{}" to float.'.format(d[key]))
                return default
            else:
                return val

        u = cls()

        if 'samples' in config:
            u.samples = int(get_float(config, 'samples', 1E6))  # Use int(float()) to handle exponential notation

        if 'functions' in config:
            for func in config['functions']:
                if type(func) == dict:
                    try:
                        u.set_function(func.get('expr'), name=func.get('name'), desc=func.get('desc', ''), outunits=func.get('units', None))
                    except ValueError:
                        logging.warning('Function {} = {} is invalid'.format(func.get('name'), func.get('expr')))

        if 'inputs' in config:
            for inpt in config['inputs']:
                if type(inpt) == dict:
                    newinpt = u.set_input(inpt.get('name', ''), nom=get_float(inpt, 'mean'), desc=inpt.get('desc', ''), units=inpt.get('units', None))
                    if 'uncerts' in inpt:
                        for unc in inpt['uncerts']:
                            newinpt.add_comp(**unc)

        if 'seed' in config:
            u.seed = int(get_float(config, 'seed'))

        if 'correlations' in config:
            for cor in config['correlations']:
                u.correlate_vars(cor.get('var1'), cor.get('var2'), get_float(cor, 'cor'))

        if 'description' in config:
            u.longdescription = config['description']

        if 'name' in config:
            u.name = config['name']

        if u.functions is None or len(u.functions) == 0 or len(u.variables) == 0:
            # No functions/inputs defined in file, probably wrong file type!
            return None
        return u

    @classmethod
    def from_configfile(cls, fname):
        ''' Read and parse the configuration file. Returns a new UncertCalc
            instance. See Examples folder for sample config file.

            Parameters
            ----------
            fname: string or file object
                File name or open file object to read configuration from

            Returns
            -------
            UncertCalc:
                New uncertainty calculator object
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

        if isinstance(config, list):
            # To support old (1.0) and new (1.1+) style config files
            config = config[0]

        u = cls.from_config(config)
        return u

    def units_report(self, **kwargs):
        ''' Create report showing units of all parameters in UncertCalc '''
        hdr = ['Parameter', 'Units', 'Abbreviation', 'Dimensionality']
        rows = []
        for func in self.functions:
            msg = None
            name = report.Math(func.name) if func.name else report.Math(func.function)
            try:
                mean = func.mean()
            except DimensionalityError as e:
                msg = '<font color="red">' + str(e) + '</font>'
                mean = None
            except OffsetUnitCalculusError:
                msg = '<font color="red">Ambiguous offset (temerature) units. Try delta_degC.'
                mean = None

            if mean is not None and func.outunits is not None:
                try:
                    units = ureg.parse_units(func.outunits)
                except UndefinedUnitError:
                    msg = '<font color="red">Undefined Unit: {}</font>'.format(func.outunits)
                    units = mean.units
                else:
                    try:
                        mean.ito(func.outunits)
                    except DimensionalityError as e:
                        msg = '<font color="red">' + str(e) + '</font>'

            elif mean is not None:
                units = mean.units  # Units not specified, use native units

            if msg:
                rows.append([name, msg, '-', '-'])
            else:
                rows.append([report.Math(func.name),
                             report.Unit(units, abbr=False, dimensionless='-'),
                             report.Unit(units, abbr=True, dimensionless='-'),
                             report.Unit(units.dimensionality, abbr=False, dimensionless='-')])

        for inpt in self.get_baseinputs():
            rows.append([report.Math(inpt.name),
                         report.Unit(inpt.units, abbr=False, dimensionless='-'),
                         report.Unit(inpt.units, abbr=True, dimensionless='-'),
                         report.Unit(inpt.units.dimensionality, abbr=False, dimensionless='-')])
            for comp in inpt.uncerts:
                try:
                    comp.std().to(inpt.units)
                except OffsetUnitCalculusError:
                    rows.append([report.Math(comp.name), '<font color="red">Ambiguous unit {}. Try "delta_{}".</font>'.format(comp.units, comp.units), '-', '-'])
                except DimensionalityError:
                    rows.append([report.Math(comp.name), '<font color="red">Cannot convert {} to {}</font>'.format(comp.units, inpt.units), '-', '-'])
                else:
                    rows.append([report.Math(comp.name),
                                report.Unit(comp.units, abbr=False, dimensionless='-'),
                                report.Unit(comp.units, abbr=True, dimensionless='-'),
                                report.Unit(comp.units.dimensionality, abbr=False, dimensionless='-')])
        rpt = report.Report(**kwargs)
        rpt.table(rows, hdr)
        return rpt


UncertaintyCalc = UncertCalc  # Support Abbreviation
