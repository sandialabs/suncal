''' PSL Uncertainty Calculator - Sandia National Labs

Main module for computing uncertainties.
'''
from collections import namedtuple
import sympy
import numpy as np
from scipy import stats
import inspect
import warnings
import logging
import yaml
from pint import DimensionalityError, UndefinedUnitError, OffsetUnitCalculusError

from . import unitmgr
from . import uparser
from . import distributions
from . import report
from . import out_uncert


# Will get div/0 errors, for example, with degrees of freedom in W-S formula
# Can safely ignore them and let the result be inf.
np.seterr(divide='ignore', invalid='ignore', over='ignore')


_COPULAS = ['gaussian', 't']  # Supported copulas

# pyYaml can't serialize numpy float64 for some reason. Add custom representer.
def np64_representer(dumper, data):
    return dumper.represent_float(float(data))  # Just convert to regular float.
def npi64_representer(dumper, data):
    return dumper.represent_int(int(data))
yaml.add_representer(np.float64, np64_representer)
yaml.add_representer(np.int64, npi64_representer)


GUMout = namedtuple('GUMoutput', ['uncert', 'nom', 'degf', 'Uy', 'Ux', 'Cx', 'symbolic', 'warnings'], defaults=(None,)*8)
MCout = namedtuple('MCoutput', ['uncert', 'nom', 'samples', 'warnings'], defaults=(None,)*4)


def matmul(a, b):
    ''' Matrix multiply. Manually looped to preserve units since Pint
        doesn't allow matrices with different units on each element
    '''
    result = []
    for i in range(len(a)):
        row = []
        for j in range(len(b[0])):
            product = 0
            for v in range(len(a[i])):
                if a[i][v] == 1:   # Symbolic gets ugly when multiplying by 1
                    product += b[v][j]
                elif b[v][j] == 1:
                    product += a[i][v]
                else:
                    product += a[i][v] * b[v][j]
            row.append(product)
        result.append(row)
    return result


def diagonal(a):
    ''' Return diagonal of square matrix '''
    if len(a) > 0 and a[0]:
        return [a[i][i] for i in range(len(a))]
    else:
        return []


def transpose(a):
    ''' Transpose list-of-list matrix a (to preserve units) '''
    return list(map(list, zip(*a)))


def eval_matrix(U, values):
    ''' Evaluate matrix (list of lists) of sympy expressions U with values '''
    U_eval = []
    for row in U:
        U_row = []
        for expr in row:
            df = sympy.lambdify(values.keys(), expr, 'numpy')  # Can't subs() with pint Quantities
            U_row.append(df(**values))
        U_eval.append(U_row)
    return U_eval


def eval_list(U, values):
    ''' Evaluate a list of sympy expressions U with values '''
    U_eval = []
    for expr in U:
        df = sympy.lambdify(values.keys(), expr, 'numpy')  # Can't subs() with pint Quantities
        U_eval.append(df(**values))
    return U_eval


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
        self.nom = np.float64(nom)  # Keep float64 to avoid /0 errors
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
        # Take a delta to see if the uncertainty units should be converted to
        # delta_XXX units (ie temperature)
        test = 1*self.units
        dfltunits = (test - test).units  # may be delta_degC, etc.

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
        return unitmgr.Quantity(self.nom, self.units)

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
        return np.float64(num2)**2 / denom if denom != 0 else np.inf # Use numpy float so div/0 returns inf.

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
                self.nom = np.float64(uparser.callf(mean))
            except (AttributeError, ValueError, TypeError, OverflowError):
                return False  # Don't change - error
        else:
            self.nom = np.float64(mean)

        for u in self.uncerts:
            u.nom = np.float64(self.nom)
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
            self.sampledvalues = unitmgr.Quantity(np.full(samples, self.nom), self.units)
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
    ''' Input uncertainty class. Stores the uncertainty name, distribution, and random
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
        self.nom = np.float64(nom)
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
                nom = unitmgr.Quantity(self.nom, self.parentunits).to(self.units).magnitude  # Convert nominal to same units as uncertainty

                # Allow entering % as % of nominal
                # or %range(X) as percent of X range
                # or ppm as ppm of nominal
                # or ppmrange(X) as ppm of X range
                aval = aval.replace('%range(', '/100*(')
                aval = aval.replace('ppmrange(', '/1E6*(')
                aval = aval.replace('ppbrange(', '/1E9*(')
                aval = aval.replace('ppm', '/1E6*{}'.format(nom))
                aval = aval.replace('ppb', '/1E9*{}'.format(nom))
                aval = aval.replace('%', '/100*{}'.format(nom))
                try:
                    aval = np.float64(uparser.callf(aval))
                except (AttributeError, ValueError, TypeError):
                    aval = np.nan
                except OverflowError:
                    aval = np.inf
            if hasattr(aval, 'units'):
                assert aval.units == self.units
                aval = aval.magnitude
            distargs[aname] = aval

        if 'df' in self.required_args and 'df' not in distargs:
            if not np.isfinite(self.degf):
                distargs['df'] = 1E9  # Essentially inf, but works with scipy.stats
            else:
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
        x = unitmgr.Quantity(np.linspace(med-s*stds, med+s*stds, num=100), self.units)
        try:
            y = self.distribution.pdf(x.magnitude)
        except AttributeError:
            # Discrete dists don't have PDF
            try:
                samples = unitmgr.Quantity(self.distribution.rvs(1000000).astype(float), self.units)
                if inc_nom:
                    samples += unitmgr.Quantity(self.nom, self.parentunits)
                    samples.to(self.parentunits).magnitude
                y, x = np.histogram(samples, bins=200)
                x = x[1:]
            except ValueError:
                x, y = [], []
        else:
            if inc_nom:
                x = x + unitmgr.Quantity(self.nom, self.parentunits)
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
        return unitmgr.Quantity(self.sampledvalues, self.units)

    def clear(self):
        ''' Clear the Monte Carlo samples '''
        self.sampledvalues = None
        self.normsamples = None

    def std(self):
        ''' Return the standard deviation of the distribution function '''
        return unitmgr.Quantity(self.distribution.std(), self.units)

    def var(self):
        ''' Return the variance of the distribution function'''
        return unitmgr.Quantity(self.distribution.var(), self.units**2)

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


class Inputs():
    ''' All InputVars in the system '''
    def __init__(self):
        self.inputvars = []
        self.corr_list = []  # list of correlation pairs (v1, v2, corr)
        self.copula = 'gaussian'
        self.nsamples = 1000000
        self.seed = None

    def __len__(self):
        return len(self.inputvars)

    def __getitem__(self, i):
        return self.inputvars[i]

    @property
    def names(self):
        ''' Get list of input names '''
        return [v.name for v in self.inputvars]

    @property
    def units(self):
        return [v.units for v in self.inputvars]

    @property
    def symbols(self):
        ''' Get list of names as sympy.Symbols '''
        return [sympy.Symbol(n) for n in self.names]

    @property
    def unc_symbols(self):
        ''' Get list of uncertainty names as sympy.Symbols '''
        return [sympy.Symbol('u_{}'.format(n)) for n in self.names]

    def reorder(self, names):
        ''' Inputs must remain in same order as model. This lets model change order just before computing '''
        newidx = [self.names.index(f) for f in names]
        self.inputvars = [self.inputvars[i] for i in newidx]

    def means(self):
        ''' Get dictionary of mean/nominal values for each input variable '''
        return {v.name: v.mean() for v in self.inputvars}

    def stdunc(self):
        ''' Get dictionary of standard uncertainty values for each input variable '''
        return {'u_{}'.format(v.name): v.stdunc() for v in self.inputvars}

    def degfs(self):
        ''' Get degrees of freedom for each input variable '''
        return {'nu_{}'.format(v.name): v.degf() for v in self.inputvars}

    def clear_samples(self):
        ''' Clear MC samples on each variable '''
        for v in self.inputvars:
            v.clear()

    def sample(self):
        ''' Generate random samples for each variable '''
        if self.seed:
            np.random.seed(self.seed)

        self.clear_samples()

        covok = True
        if len(self.corr_list) > 0:
            covok = self._gen_corr_samples()

        self.sampledvalues = {}
        for v in self.inputvars:
            self.sampledvalues[v.name] = v.sample(self.nsamples)
        return self.sampledvalues, covok

    def correlation(self):
        ''' Get correlation matrix as entered '''
        corr = np.eye(len(self.inputvars))
        names = self.names
        for v1, v2, c in self.corr_list:
            idx1 = names.index(v1)
            idx2 = names.index(v2)
            corr[idx1, idx2] = c
            corr[idx2, idx1] = c
        return corr

    @property
    def corr_symbols(self):
        ''' Get list of symbols of nonzero correlation coefficients '''
        symb = []
        for v1, v2, c in self.corr_list:
            symb.append(sympy.Symbol('sigma_{}{}'.format(v1, v2)))
        return symb

    def correlation_sym(self):
        ''' Get correlation matrix as symbols (including symbols for 0 correlations) '''
        corr = []
        names = self.names
        for idx1, name1 in enumerate(names):
            row = []
            for idx2, name2 in enumerate(names):
                if name1 == name2:
                    row.append(1.0)
                else:
                    if idx1 < idx2:
                        row.append(sympy.Symbol(f'sigma_{names[idx1]}{names[idx2]}'))
                    else:
                        row.append(sympy.Symbol(f'sigma_{names[idx2]}{names[idx1]}'))
            corr.append(row)
        return corr

    def corr_values(self):
        ''' Get dictionary of correlation coefficient values '''
        corr = {}
        names = self.names
        for idx1, name1 in enumerate(names):
            for idx2, name2 in enumerate(names):
                if idx1 < idx2:
                    corr[f'sigma_{name1}{name2}'] = 0.0

        for v1, v2, c in self.corr_list:
            name1 = f'sigma_{v1}{v2}'
            name2 = f'sigma_{v2}{v1}'
            if name1 in corr:
                corr[name1] = c
            elif name2 in corr:
                corr[name2] = c
            else:
                assert False
        return corr

    def covariance(self):
        ''' Get covariance matrix [Ux] (numerical values) '''
        # Make sure ALL values in matrix, including 0's, have correct units
        # why we're looping instead of just subbing into covariance_sym.
        corr = self.correlation()
        Ux = []
        for i, v1 in enumerate(self.inputvars):
            row = []
            for j, v2 in enumerate(self.inputvars):
                if i == j:
                    row.append(v1.stdunc()**2)
                elif corr[i, j] != 0:
                    row.append(v1.stdunc() * v2.stdunc() * corr[i, j])
                else:
                    row.append(v1.stdunc() * v2.stdunc() * 0)  # *0 produces a 0 with correct units
            Ux.append(row)
        return Ux

    def covariance_sym(self):
        ''' Get covariance matrix [Ux] as list of sympy expressions '''
        S = []
        for i, v in enumerate(self.inputvars):
            row = [0]*i
            row.append(sympy.Symbol('u_{}'.format(v.name)))
            row.extend([0]*(len(self.inputvars) - i - 1))
            S.append(row)

        corr = self.correlation_sym()
        Ux = matmul(matmul(S, corr), S)
        return Ux

    def set_input(self, name, nom=1, desc='', units=None, **kwargs):
        ''' Add or update an input nominal value and description in the system.

            Parameters
            ----------
            name: string
                Name of variable to update
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
        names = self.names
        if hasattr(nom, 'units'):
            if units is not None and str(nom.units) != units:
                raise TypeError('Nominal value units do not match units argument')
            units = str(nom.units)
            nom = nom.magnitude

        if name not in names:
            inpt = InputVar(name=name, nom=nom, desc=desc, units=units)
            self.inputvars.append(inpt)
        else:  # Existing variable, update it
            idx = names.index(name)
            inpt = self.inputvars[idx]
            inpt.nom = np.float64(nom)
            inpt.desc = desc
            if units is not None:
                inpt.units = uparser.parse_unit(units)

        # Add the uncertainty
        if len(kwargs) > 0:
            self.set_uncert(var=name,
                            name=kwargs.pop('uname', 'u({})'.format(name)),
                            **kwargs)

        return inpt

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
            Keyword arguments defining distribution will be passed to the
            rv_continuous instance in scipy.stats.
        '''
        names = self.names
        if var not in names:
            raise ValueError('Variable {} not defined'.format(var))

        idx = names.index(var)

        if name is None:
            name = '{}{}'.format(self.inputvars[idx].name, len(self.inputvars[idx].uncerts))

        self.inputvars[idx].add_comp(name=name, dist=dist, degf=degf, desc=desc, units=units, **args)

    def remove_input(self, name):
        ''' Remove input from list '''
        idx = self.names.index(name)
        self.inputvars.pop(idx)

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
        try:
            self.corr_list.append((var1, var2, float(correlation)))
        except ValueError:
            return  # Failed to convert to float

    def set_copula(self, copula='gaussian'):
        ''' Set the copula for correlating inputs.

            Parameters
            ----------
            copula: string
                Name of coupla to use. Must be 'gaussian' or 't'.
        '''
        assert copula in _COPULAS
        self.copula = copula

    def _gen_corr_samples(self, **args):
        ''' Generate random samples for each of the inputs '''
        covok = True

        corr = self.correlation()
        # Note: correlation==covariance since all std's are 1 right now.

        # Generate correlated random samples, save to each input
        if self.copula == 'gaussian':
            with warnings.catch_warnings(record=True) as w:
                # Roundabout way of catching a numpy warning that should really be an exception
                warnings.simplefilter('always')
                normsamples = stats.multivariate_normal.rvs(cov=corr, size=self.nsamples)
                if len(w) > 0:
                    covok = False  # Not positive semi-definite
                normsamples = stats.norm.cdf(normsamples)

        elif self.copula == 't':
            df = args.get('df', np.inf)
            mean = np.zeros(corr.shape[0])
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter('always')
                normsamples = multivariate_t_rvs(mean=mean, corr=corr, size=self.nsamples, df=df)
                normsamples = stats.t.cdf(normsamples, df=df)
                if len(w) > 0:
                    covok = False  # Not positive semi-definite

        normsamples[np.where(normsamples == 1.0)] = 1 - 1E-9  # If rounded to 1 or 0, then we get infinity.
        normsamples[np.where(normsamples == 0.0)] = 1E-9
        for idx, inpt in enumerate(self.inputvars):
            inpt._set_normsamples(normsamples[:, idx])
        return covok


class Model():
    ''' Generic Measurement model with N outputs '''
    @property
    def symbols(self):
        ''' Get model function names as list of sympy.Symbols '''
        return [sympy.Symbol(n) for n in self.outnames]

    @property
    def unc_symbols(self):
        ''' Get list of uncertainty names as sympy.Symbols '''
        return [sympy.Symbol('u_{}'.format(n)) for n in self.outnames]

    @property
    def outnames(self):
        if self._outnames is None:
            self._setup_output()
        return self._outnames

    @outnames.setter
    def outnames(self, value):
        self._outnames = value

    def set_inputs(self, inputs):
        ''' Set Inputs object '''
        self.inputs = inputs

    def check_dimensionality(self):
        ''' Check that units are compatible. Raises DimentionalityError or UndefinedUnitsError '''
        # Check units compatibility
        f = self.eval().values()

        for out, units in zip(f, self.outunits):
            if units is not None:
                if not hasattr(out, 'units'):
                    out = unitmgr.Quantity(out, unitmgr.dimensionless)
                out.to(units) # will raise if output units are incompatible

        self.inputs.stdunc()  # will raise if uncert components are incompatible

    def check_circular(self):
        ''' Check functions list to ensure no circular dependencies. Returns True
            if circular reference is found.
        '''
        try:
            self.get_baseexprs()
        except RecursionError:
            return True
        return False

    def get_baseexprs(self):
        ''' Does nothing on callables '''
        return

    def sensitivity(self):
        ''' Cx - (6.2.1.3 GS2) '''
        raise NotImplementedError  # Subclass

    def GUMcovariance(self):
        ''' Calculate GUM uncertainty/covariance matrix - Uy (6.2.1.3 GS2) '''
        raise NotImplementedError  # Subclass

    def _eval_vectorized(self, inputsamples):
        ''' Evaluate MC samples by vectorizing the model '''
        try:
            samples = self.eval(inputsamples)
        except DimensionalityError:
            # Hack around Pint bug/inconsistency (see https://github.com/hgrecco/pint/issues/670, closed without solution)
            #   with x = np.arange(5) * units.dimensionless
            #   np.exp(x) --> returns dimensionless array
            #   2**x --> raises DimensionalityError
            # Since units/dimensionality has already been verified, this fix strips units and adds them back.
            inputsamples = {k: v.magnitude for k, v in inputsamples.items()}
            samples = self.eval(inputsamples)
        except (TypeError, ValueError):
            # Call might have failed if function is not vectorizable. Use numpy vectorize
            # to broadcast over array and try again.
            logging.info('Vectorizing function {}...'.format(str(self.function)))

            # Vectorize will strip units - see https://github.com/hgrecco/pint/issues/828.
            # First, run a single sample through the function to determine what units come out
            outsingle = uparser.callf(self.function, {k: v[0] for k, v in inputsamples.items()})
            mcoutunits = outsingle.units if hasattr(outsingle, 'units') else unitmgr.dimensionless

            # Then apply those units to whole array of sampled values.
            # vectorize() will issue a UnitStripped warning, but we're handling it outside Pint, so ignore it.
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                out = unitmgr.Quantity(uparser.callf(np.vectorize(self.function), inputsamples), mcoutunits)
                if len(self.outnames) > 1:
                    samples = dict(zip(self.outnames, out))
                else:
                    samples = {self.outnames[0]: out}

        # Fix weird cases where function returns array of objects instead of floats
        for i, (k, v) in enumerate(samples.items()):
            if not hasattr(v, 'magnitude'):
                # Some functions may strip units
                v = unitmgr.Quantity(v, 'dimensionless')

            try:
                len(v)
            except TypeError:
                v = unitmgr.Quantity(np.full(self.inputs.nsamples, v.magnitude), v.units)
            samples[k] = v.astype(np.float)

            if self.outunits[i]:
                samples[k] = samples[k].to(self.outunits[i])
            return samples

    def MCsample(self):
        ''' Generate Monte Carlo samples, NxM

            Returns
            -------
            uncert: array of standard uncertainties
            nom: array of nominal values
            samples: dict of Monte Carlo samples for each function output
            warnings: list of warning strings
        '''
        # sample each input M times, correlated if necessary
        # Calculate model M times, resulting in MxN
        warns = []
        inputsamples, covok = self.inputs.sample()
        if not covok:
            warns.append('Covariance matrix is not positive semi-definite.')

        self.sampledvalues = self._eval_vectorized(inputsamples)
        nom = [v.mean() for v in self.sampledvalues.values()]
        uncert = []
        for i, (out, vals) in enumerate(self.sampledvalues.items()):
            try:
                uncert.append(vals.std())
            except AttributeError:  # Single/constant value returned from MC
                uncert.append(0 * nom[i].units)

            uncert[-1].ito(nom[i].units)

            if not all(np.isfinite(np.atleast_1d(np.float64(vals.magnitude)))):
                warns.append('Some Monte-Carlo samples in {} are NaN. Ignoring in statistics.'.format(out))

        out = MCout(uncert, nom, self.sampledvalues, warns)
        return out

    def MCsensitivity(self):
        ''' Calculate Monte Carlo non-linear sensitivity coefficients. Assumes MCsample has been run.

            Returns
            -------
            sensitivity: matrix of sensitivity coefficients (rows = outputs, columns = inputs)
            proportions: matrix of uncertainty contribution proportions (rows = outputs, columns = inputs)
        '''
        if self.sampledvalues is None:
            self.MCsample()

        nom = self.inputs.means()
        uncert = [v.std() for v in self.sampledvalues.values()]

        # Set all input values to the nominal
        for x in nom.keys():
            nom[x] = unitmgr.Quantity(np.full(self.inputs.nsamples, nom[x].magnitude), nom[x].units)

        CxT = []     # Transposed sensitivity matrix
        propsT = []  # Transposed proportions
        for i, name in enumerate(self.inputs.names):
            x = nom.copy()
            # Then set each input to sampled values, one at a time
            x[name] = self.inputs[i].sampledvalues
            ustd = self._eval_vectorized(x)
            ustd = [v.std() for v in ustd.values()]  # dict to list of stds for each function out

            # to_reduced_units() will take care of prefix multipliers and dimensionless values
            CxT.append([(u/self.inputs[i].stdunc()).to_reduced_units() for u in ustd])
            propsT.append([(u/uncert[i]).to_reduced_units().magnitude**2 for i, u in enumerate(ustd)])

        MCprops = namedtuple('MCproportions', ['sensitivity', 'proportions'])
        return MCprops(transpose(CxT), transpose(propsT))


class ModelSympy(Model):
    ''' Measurement model consisting of list of N Sympy expressions to evaluate

        Parameters
        ----------
        exprs: list
            String expressions (sympyfiable) or Sympy expressions to add to the model
        outunits: list
            List of desired units on each model function (strings, Pint-recognizable)
    '''
    def __init__(self, exprs=None, outunits=None):
        self.exprs = []   # String expressions
        self.sympyexprs = [] # Sympy expressions
        self.outnames = []
        self.inputnames = []
        self.outunits = []
        self.descriptions = []
        self.show = []
        self.noutputs = len(self.exprs)
        self.sampledvalues = None

        self.inputs = Inputs()
        if exprs is not None:
            if outunits is None:
                outunits = [None]*self.noutputs

            for exp, unit in zip(exprs, outunits):
                self.add_expr(exp, outunits=unit)

    @property
    def expr_symbols(self):
        ''' Get functions as sympy expressions '''
        return [sympy.Eq(f, exp) for f, exp in zip(self.symbols, self.sympyexprs)]

    def add_expr(self, expr, name=None, idx=None, outunits=None, desc='', show=True):
        ''' Add a function expression to the model

            Parameters
            ----------
            expr: string or Sympy
                The expression to add
            name: string
                A name for the expression. Will be extracted from expr if expr contains an equals
            idx: int
                Use idx to replace/overwrite an existing expression in the model
            outunits: string
                Desired units for output of function
            desc: string
                Description of the function
            show: bool
                Whether to show the result in output reports
        '''
        if isinstance(expr, sympy.Basic):
            symexpr = expr
            expr = str(expr)

        if name is None:
            if '=' in expr:
                name, expr = expr.split('=')
            else:
                name = 'f{}'.format(len(self.exprs)+1)

        name = name.strip()
        expr = expr.strip()
        symexpr = uparser.parse_math(expr, name=name)

        if idx is None or idx >= len(self.exprs):
            idx = len(self.exprs)
            self.exprs.append(None)  # To be replaced later
            self.sympyexprs.append(None)
            self.outnames.append(None)
            self.outunits.append(None)
            self.descriptions.append(None)
            self.show.append(None)

        self.exprs[idx] = expr
        self.sympyexprs[idx] = symexpr
        self.outnames[idx] = name
        self.outunits[idx] = outunits
        self.descriptions[idx] = desc
        self.show[idx] = show
        self.get_baseexprs()  # Build inputnames
        self.noutputs = len(self.exprs)

    def remove_expr(self, idx):
        ''' Remove expression at index '''
        try:
            self.exprs.pop(idx)
            self.sympyexprs.pop(idx)
            self.outnames.pop(idx)
            self.outunits.pop(idx)
            self.descriptions.pop(idx)
            self.show.pop(idx)
            self.inputnames = []
        except IndexError:
            pass
        self.get_baseexprs()  # Build inputnames

    def reorder(self, names):
        ''' Reorder the function list using order of names array.

            Parameters
            ----------
            names: list of strings
                List of function names in new order
        '''
        fnames = self.outnames
        newidx = [fnames.index(f) for f in names]

        def order(A, newidx):
            return [A[i] for i in newidx]

        self.exprs = order(self.exprs, newidx)
        self.sympyexprs = order(self.sympyexprs, newidx)
        self.outnames = order(self.outnames, newidx)
        self.outunits = order(self.outunits, newidx)
        self.descriptions = order(self.descriptions, newidx)
        self.show = order(self.show, newidx)

    def get_baseexprs(self):
        ''' Get model functions in terms of base inputs only (substitute any dependencies
            on other functions in the model). Also extracts self.inputnames.
        '''
        # Recursively substitute each function in until the function doesn't change anymore
        baseexprs = []
        self.inputnames = []
        for exp in self.sympyexprs:
            oldfunc = None
            count = 0
            while oldfunc != exp and count < 100:
                oldfunc = exp
                for vname in exp.free_symbols:
                    if str(vname) in self.outnames:
                        exp = exp.subs(vname, self.sympyexprs[self.outnames.index(str(vname))])
                count += 1
            if count >= 100:
                raise RecursionError('Circular reference in function set')
            baseexprs.append(exp)
            self.inputnames.extend([str(s) for s in exp.free_symbols if str(s) not in self.outnames])

        # inputnames will be alpha sorted for sympy models, but not callables
        self.inputnames = sorted(list(set(self.inputnames)))
        return baseexprs

    def sensitivity_sym(self):
        ''' Symbolic sensitivity matrix (Cx) '''
        Cx = []
        for i, exp in enumerate(self.get_baseexprs()):
            Cx_row = []
            for j, var in enumerate(self.inputnames):
                df = sympy.Derivative(exp, sympy.Symbol(var), evaluate=True).simplify()
                Cx_row.append(df)
            Cx.append(Cx_row)
        return Cx

    def sensitivity(self):
        ''' Sensitivity Matrix Cx evaluated at the measured values '''
        Cx = self.sensitivity_sym()
        return eval_matrix(Cx, self.inputs.means())

    def GUMcovariance_sym(self):
        ''' Calculate GUM uncertainty/covariance matrix

            Returns
            -------
            uncert: list
                List of uncertainties for each function in the model
            nom: list
                List of nominal values for each function
            degf: list
                Degrees of freedom for each function
            Uy: matrix (list of lists)
                Covariance matrix of output
            Ux: matrix
                Covariance matrix of inputs
            Cx: matrix
                Sensitivity coefficients

            Note
            ----
            See GUM Supplement 2, section 6.2.1.3
        '''
        self.inputs.reorder(self.inputnames)  # Ensure Ux, Cx, Uy all in same variable order
        Cx = self.sensitivity_sym()
        CxT = transpose(Cx)
        Ux = self.inputs.covariance_sym()

        if len(Ux) == 0:
            uncerts = [0 * self.outunits[i] if self.outunits[i] else 0 for i in range(self.noutputs)]
            degf = [np.inf] * self.noutputs
            Uy = [[]]
            return GUMout(uncerts, self.sympyexprs, degf, Uy, Ux, Cx)

        Uy = matmul(matmul(Cx, Ux), CxT)

        # Just uncertainties (sqrt of diagonal of Uy)
        uncerts = [sympy.sqrt(u) for u in diagonal(Uy)]

        varnames = self.inputs.names
        degfsymbols = [sympy.Symbol('nu_{}'.format(str(x))) for x in varnames]
        uncertsymbols = [sympy.Symbol('u_{}'.format(str(x))) for x in varnames]

        degf = []
        for i, out in enumerate(self.outnames):
            uncfsymbol = sympy.Symbol('u_{}'.format(out))
            denom = [(u*c)**4/v for u, c, v in zip(uncertsymbols, Cx[i], degfsymbols)]
            denom = sympy.Add(*denom)
            if denom == 0:
                degf.append(np.inf)
            else:
                degf.append(uncfsymbol**4 / denom)
        return GUMout(uncerts, self.sympyexprs, degf, Uy, Ux, Cx)

    def GUMcovariance(self, symboliconly=False):
        ''' Use GUM method to compute covariance in the output functions

            Returns
            -------
            uncert: list
                List of uncertainties for each function in the model
            nom: list
                List of nominal values for each function
            degf: list
                Degrees of freedom for each function
            Uy: matrix (list of lists)
                Covariance matrix of output
            Ux: matrix
                Covariance matrix of inputs
            Cx: matrix
                Sensitivity coefficients
            symbolic: GUMout
                Symbolic solution of the above return parameters
            warnings: list
                List of any warnings encountered during computation
        '''
        warns = []
        sym = self.GUMcovariance_sym()
        if not symboliconly:
            values = self.inputs.means()
            uncs = self.inputs.stdunc()
            corrs = self.inputs.corr_values()
            values.update(uncs)
            values.update(corrs)
            Ux = eval_matrix(sym.Ux, values)
            Uy = eval_matrix(sym.Uy, values)
            Cx = eval_matrix(sym.Cx, values)

            if not all(all(np.isfinite(u) for u in k) for k in Uy):
                warns.append('Overflow in GUM uncertainty calculation')

            # Nominal value
            nom = self.eval().values()
            nom = [n.to(self.outunits[i]) if self.outunits[i] is not None else n for i, n in enumerate(nom)]

            # Just output uncertainties (sqrt of diagonal of Uy)
            uncerts = eval_list(sym.uncert, values)
            # Convert to desired units
            uncerts = [u.to(self.outunits[i]) if self.outunits[i] is not None else u.to(nom[i].units) if hasattr(u, 'to') else u*unitmgr.dimensionless for i, u in enumerate(uncerts)]

            # Deg freedom
            nu = self.inputs.degfs()
            outunc = {'u_{}'.format(outname): out for outname, out in zip(self.outnames, uncerts)}
            values.update(outunc)
            values.update(nu)
            degf = eval_list(sym.degf, values)
            degf = [d.to_reduced_units().magnitude if hasattr(d, 'to_reduced_units') else d for d in degf]  # Degf is dimensionless, drop units

            return GUMout(uncerts, nom, degf, Uy, Ux, Cx, symbolic=sym, warnings=warns)
        else:
            return GUMout(symbolic=sym)

    def eval(self, values=None):
        ''' Evaluate the model at the values dict '''
        if values is None:
            values = self.inputs.means()

        lambdifys = [sympy.lambdify(self.inputnames, expr, 'numpy') for expr in self.get_baseexprs()]
        result = {}
        for i, func in enumerate(lambdifys):
            try:
                res = func(**values)
            except ZeroDivisionError:
                res = np.inf
                if self.outunits[i] is not None:
                    res = unitmgr.Quantity(res, self.outunits[i])
            else:
                if self.outunits[i] is not None and hasattr(res, 'to'):
                    res = res.to(self.outunits[i])
            if not hasattr(res, 'units'):
                res = unitmgr.Quantity(res, unitmgr.dimensionless)  # Can end up with no units for constant models, e.g.
            result[self.outnames[i]] = res
        return result


class ModelCallable(Model):
    ''' Measurement model consisting of a single callable function, with N outputs

        Parameters
        ----------
        func: callable
            The function to evaluate. May return tuple of multiple quantities.
        outunits: list
            List of desired units for each output of the function.
        finnames: list (optional)
            List of parameters to func. Required if func uses keyword arguments
        foutnames: list (optional)
            List of output parameter names from func. Can be determined automatically
            if func returns a namedtuple, otherwise if foutnames is not provided
            the outputs will be named func_1, _2, etc.
        finunits, foutunits: list (optional)
            If the function cannot directly handle Pint Quantities, use these parameters
            to specify a list of units for each input and the resulting units for each output
            of the function.
    '''
    def __init__(self, func, outunits=None, finnames=None, foutnames=None, finunits=None, foutunits=None):
        self.function = func
        self.inputnames = finnames
        self._outnames = foutnames
        self.outunits = outunits
        self.exprs = foutnames  # No expressions, just function names
        self.expr_symbols = foutnames
        self.descriptions = []
        self.show = []
        self.inputs = None
        self.sampledvalues = None
        if hasattr(self.function, '__name__'):
            self.name = self.function.__name__
        elif isinstance(self.function, np.vectorize):
            self.name = self.function.pyfunc.__name__
        else:
            self.name = 'f'   # Shouldn't get here?

        if self.inputnames is None:
            params = inspect.signature(self.function).parameters
            if any(p.kind != inspect.Parameter.POSITIONAL_OR_KEYWORD for p in params.values()):
                raise ValueError('Callable function uses keyword arguments. Please specify finnames parameter.')
            self.inputnames = sorted(list(params.keys()))

        # Wrap function with in/out units if specified
        if finunits and foutunits:
            if not isinstance(finunits, (list, tuple)):
                raise TypeError('finunits parameter must be list')
            if not isinstance(foutunits, (list, tuple)):
                raise TypeError('foutunits parameter must be list')

            finunits = [uparser.parse_unit(u) for u in finunits]
            foutunits = [uparser.parse_unit(u) for u in foutunits]
            foutunits = foutunits[0] if len(foutunits) == 1 else foutunits
            self.function = unitmgr.ureg.wraps(foutunits, finunits)(func)

    def _setup_output(self):
        ''' Attempt to determine return value names from function '''
        # By delaying this until AFTER inputs are defined, units can properly propagate through
        # the function call when determining output structure
        if self._outnames is None:
            out = uparser.callf(self.function, self.inputs.means())
            try:
                if hasattr(out, '_fields'):  # Namedtuple
                    self._outnames = out._fields
                else:
                    self._outnames = ['{}_{}'.format(self.name, str(i+1)) for i in range(len(out))]
            except TypeError:
                self._outnames = [self.name]

        if self.outunits is None:
            self.outunits = [None]*len(self._outnames)
        if not isinstance(self.outunits, (tuple, list)):
            self.outunits = [self.outunits]
        if self.show is None or len(self.show) == 0:
            self.show = [True] * len(self._outnames)
        self.noutputs = len(self._outnames)

    def sensitivity(self):
        ''' Calculate sensitivity matrix (numerical gradient) '''
        means = self.inputs.means()
        uncerts = self.inputs.stdunc()

        CxT = []
        for name in self.inputnames:
            uname = 'u_{}'.format(name)
            d1 = means.copy()
            dx = np.float64(uncerts[uname].to(d1[name].units).magnitude / 1E6) * d1[name].units # Float64 won't have div/0 errors, will go to inf
            if dx == 0:
                dx = unitmgr.Quantity(1E-6, d1[name].units)
            d1[name] = d1[name] + dx
            d2 = means.copy()
            d2[name] = d2[name] - dx
            delta = 2*dx
            result1 = self.eval(d1)
            result2 = self.eval(d2)
            CxT.append([((result1[out]-result2[out])/delta).to_reduced_units() for out in self.outnames])
        return transpose(CxT)

    def GUMcovariance(self, symboliconly=False):
        ''' Calculate GUM uncertainty/covariance matrix Uy

            Returns
            -------
            uncert: list
                List of uncertainties for each function in the model
            nom: list
                List of nominal values for each function
            degf: list
                Degrees of freedom for each function
            Uy: matrix (list of lists)
                Covariance matrix of output
            Ux: matrix
                Covariance matrix of inputs
            Cx: matrix
                Sensitivity coefficients

            Note
            ----
            See GUM Supplement 2, section 6.2.1.3
        '''
        self.inputs.reorder(self.inputnames)  # Ensure Ux, Cx, Uy all in same variable order
        Cx = self.sensitivity()
        CxT = transpose(Cx)
        Ux = self.inputs.covariance()
        Uy = matmul(matmul(Cx, Ux), CxT)

        nom = list(self.eval().values())
        nom = [n.to(self.outunits[i]) if self.outunits[i] is not None else n for i, n in enumerate(nom)]
        uncerts = [np.sqrt(Uy[i][i]) for i in range(len(Uy))]
        uncerts = [u.to(self.outunits[i]) if self.outunits[i] is not None else u for i, u in enumerate(uncerts)]

        # Deg Freedom - Welch-Satterthwaite
        nu = self.inputs.degfs().values()
        us = [np.sqrt(Ux[i][i]) for i in range(len(Ux))]
        degf = []
        for cs, uc in zip(Cx, uncerts):  # Each output in model
            denom = sum([(ci*ui)**4 / vi for ci, ui, vi in zip(cs, us, nu)])
            df = (uc**4 / denom).to_reduced_units().magnitude  # reduce to fix things like m**4 / mm**4
            degf.append(df)
        return GUMout(uncerts, nom=nom, degf=degf, Uy=Uy, Ux=Ux, Cx=Cx)

    def eval(self, values=None):
        ''' Evaluate the model at the values dict '''
        self._setup_output()
        if values is None:
            values = self.inputs.means()

        out = uparser.callf(self.function, values)

        if len(self.outnames) > 1:
            return dict(zip(self.outnames, out))
        else:
            return {self.outnames[0]: out}


class UncertCalc():
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

        Keyword Arguments
        -----------------
        finnames: list (optional)
            List of arguments to callable function. Required when function is callable and
            takes keyword arguments.
        foutnames: list (optional)
            List of names of return values from callable function. Can be deduced automatically
            if function returns a namedtuple
        finunits: list (optional)
        foutunits: list (optional)
            "Natural" input and output units (as strings) from callable function. Informs
            suncal that function() called with finunits on the inputs will result in
            foutunits on the outputs. Specify these parameters if callable function cannot
            operate on Pint Quantities.
        '''
    def __init__(self, function=None, inputs=None, units=None, samples=1000000, seed=None, name='uncertainty',
                 finnames=None, foutnames=None, finunits=None, foutunits=None):
        self.model = None
        self.inputs = Inputs()
        self.inputs.seed = seed
        self.inputs.nsamples = int(samples)
        self.longdescription = ''
        self.name = name
        self.out = None   # Output object

        if isinstance(function, list) or isinstance(function, tuple):
            if units is None:
                units = [None]*len(function)
            assert len(units) == len(function)
            [self.set_function(f, outunits=u) for f, u in zip(function, units)]

        elif isinstance(function, sympy.Basic) or isinstance(function, str):
            assert units is None or isinstance(units, str)
            self.set_function(function, outunits=units)

        elif callable(function):
            self.model = ModelCallable(function, outunits=units, finnames=finnames, foutnames=foutnames,
                                       finunits=finunits, foutunits=foutunits)
            self.model.inputs = self.inputs

        elif function is not None:
            raise TypeError('Unknown function format - {}'.format(type(function)))

        if inputs is not None:
            for inpt in inputs:
                uncerts = inpt.get('uncerts', [])
                iname = inpt.get('name', '')
                units = inpt.get('units', None)
                self.set_input(iname, nom=inpt.get('nom', 0), desc=inpt.get('desc', ''), units=units)
                for udict in uncerts:
                    self.set_uncert(var=iname, **udict)

    def get_output(self):
        ''' Get output object (for GUI) '''
        return self.out

    def set_function(self, func, idx=None, name=None, outunits=None, desc='', show=True):
        ''' Add or update a sympy function in the system.

            Parameters
            ----------
            func: string, or sympy expression
                The function to add
            idx: int, optional
                Index of existing function. If not provided, the function names will be searched,
                or a new function will be added if not in the function names already.
            name: string
                Name of the function
            outunits: string
                Convert the output to these units
            desc: string
                Description for the function
            show: boolean
                Show this function in the reports
        '''
        # Callables must use UncertCalc(callalbefunc) since only one is supported
        assert self.model is None or isinstance(self.model, ModelSympy)

        if self.model is None:
            self.model = ModelSympy()
            self.model.set_inputs(self.inputs)

        if idx is None and name is not None and name in self.model.outnames:
            idx = self.model.outnames.index(name)

        self.model.add_expr(func, idx=idx, name=name, outunits=outunits, desc=desc, show=show)

    def set_input(self, name, nom=1, units=None, desc='', **kwargs):
        ''' Add or update an input nominal value and description in the system.

            Parameters
            ----------
            name: string
                Name of variable to update
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
        newinpt = self.inputs.set_input(name, nom=nom, desc=desc, units=units)
        if len(kwargs) > 0:
            self.inputs.set_uncert(name, name=kwargs.pop('uname', 'u({})'.format(name)), **kwargs)
        return newinpt

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
        self.inputs.set_uncert(var=var, name=name, dist=dist, degf=degf, units=units, desc=desc, **args)

    def clearall(self):
        ''' Clear all inputs, resetting to blank calculator '''
        self.model = None
        self.inputs = Inputs()
        self.longdescription = ''
        self.out = None

    def clearout(self):
        ''' Clear output (ie back button pressed) '''
        self.out = None

    def reorder(self, names):
        ''' Reorder the function list using order of names array.

            Parameters
            ----------
            names: list of strings
                List of function names in new order
        '''
        assert isinstance(self.model, ModelSympy)
        self.model.reorder(names)

    def remove_function(self, idx):
        ''' Remove function at idx '''
        assert isinstance(self.model, ModelSympy)
        self.model.remove_expr(idx)

    def get_inputvar(self, name):
        ''' Return InputVar object with the given name

            Parameters
            ----------
            name: str
                Name of the input to return

            Returns
            -------
            Input variable: InputVar
        '''
        try:
            idx = self.inputs.names.index(name)
        except IndexError:
            return None
        else:
            return self.inputs[idx]

    def get_functionnames(self):
        ''' Get the name of each function in the model '''
        return self.model.outnames

    @property
    def required_inputs(self):
        ''' Return a list of input names required to define the function(s). '''
        return self.model.inputnames

    def add_required_inputs(self):
        ''' Add and remove inputs so that all free variables in functions are
            defined. New variables will default to normal, mean=0, std=1.
        '''
        inputs = self.model.inputnames

        # Add generic input variable if not in list
        for inpt in inputs:
            if inpt not in self.inputs.names:
                self.inputs.set_input(inpt, unc=1, k=2)

        # Remove inputs not in current model
        for name in self.inputs.names:
            if name not in inputs:
                self.inputs.remove_input(name)

    def set_correlation(self, corr, names, copula='gaussian', **args):
        ''' Set correlation of inputs as a matrix.

            Parameters
            ----------
            corr : array_like, shape (M,M)
                correlation matrix. Must be square where M is number of inputs.
                Only upper triangle is considered.
            names: list
                List of variable names corresponding to the rows/columns of cor
            copula : string
                Copula for correlating the inputs. Supported copulas:
                ['gaussian', 't']
            **args : variable
                copula arguments. Only df: degrees of freedom (t copula) is supported.
        '''
        assert copula in _COPULAS
        self.inputs.copula = copula
        for idx1, name1 in enumerate(names):
            for idx2, name2 in enumerate(names):
                if idx1 < idx2:
                    self.inputs.correlate_vars(name1, name2, corr[idx1, idx2])

    def clear_corr(self):
        ''' Clear correlation table '''
        self.inputs._corr = None

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
        self.inputs.correlate_vars(var1, var2, correlation)

    def units_report(self, **kwargs):
        ''' Create report showing units of all parameters in UncertCalc '''
        hdr = ['Parameter', 'Units', 'Abbreviation', 'Dimensionality']
        rows = []
        values = self.inputs.means()
        lambdifys = [sympy.lambdify(self.model.inputnames, expr, 'numpy') for expr in self.model.get_baseexprs()]
        for fidx, func in enumerate(lambdifys):
            msg = None
            name = report.Math(self.model.outnames[fidx])
            try:
                result = func(**values)
                if self.model.outunits[fidx] is not None:
                    result = result.to(self.model.outunits[fidx])
                    units = unitmgr.parse_units(self.model.outunits[fidx])
                else:
                    units = result.units

            except ZeroDivisionError:
                result = np.inf
                if self.model.outunits[fidx] is not None:
                    result = unitmgr.Quantity(result, self.model.outunits[fidx])
            except DimensionalityError as e:
                msg = '<font color="red">' + str(e) + '</font>'
                result = None
            except OffsetUnitCalculusError:
                msg = '<font color="red">Ambiguous offset (temerature) units. Try delta_degC.'
                result = None
            except UndefinedUnitError:
                msg = '<font color="red">Undefined Unit: {}</font>'.format(self.model.outunits[fidx])
            
            if msg:
                rows.append([name, msg, '-', '-'])
            else:
                rows.append([report.Math(self.model.outnames[fidx]),
                             report.Unit(units, abbr=False, dimensionless='-'),
                             report.Unit(units, abbr=True, dimensionless='-'),
                             report.Unit(units.dimensionality, abbr=False, dimensionless='-')])

        for inpt in self.inputs:
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

    def calculate(self, **kwargs):
        ''' Calculate uncertainty of the model

            Keyword Arguments
            ----------------
            gum: bool
                Calculate GUM method
            mc: bool
                Calculate Monte Carlo method
            samples: int
                Number of Monte Carlo Samples

            Returns
            -------
            out_uncert.UncertOutput
        '''
        gum = kwargs.get('gum', True)
        mc = kwargs.get('mc', True)
        if 'samples' in kwargs:
            self.inputs.nsamples = int(kwargs.get('samples'))
        self.out = out_uncert.UncertOutput(self.model, self.inputs, gum=gum, mc=mc)
        return self.out

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
        if not (outputs or inputs):
            return

        if self.model.sampledvalues is None:
            self.model.MCsample()
        hdr = []
        if inputs:
            variables = self.inputs.inputvars
            hdr = ['{}{}'.format(v.name, report.Unit(v.units).prettytext(bracket=True)) for v in variables]
            csv = np.stack([v.sampledvalues.magnitude for v in variables], axis=1)
        if outputs:
            outnames = self.model.sampledvalues.keys()
            outvals = self.model.sampledvalues.values()
            outunits = self.model.outunits
            out = np.array([v.magnitude for v in outvals]).T
            hdr.extend(['{}{}'.format(str(f), report.Unit(unitmgr.parse_units(u)).prettytext(bracket=True)) for f, u in zip(outnames, outunits)])
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

    def get_config(self):
        ''' Get configuration dictionary describing this calculation '''
        if isinstance(self.model, ModelCallable):
            raise ValueError('Cannot save callable function to config file.')

        d = {}
        d['name'] = self.name
        d['mode'] = 'uncertainty'
        d['samples'] = self.inputs.nsamples
        d['functions'] = []
        d['inputs'] = []

        customunits = unitmgr.get_customunits()
        if customunits:
            d['unitdefs'] = customunits

        for func, name, desc, units in zip(self.model.sympyexprs, self.model.outnames, self.model.descriptions, self.model.outunits):
            fdict = {'name': name, 'expr': str(func), 'desc': desc, 'units': format(units) if units else None}
            d['functions'].append(fdict)

        for inpt in self.inputs.inputvars:
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

        if self.inputs.seed is not None:
            d['seed'] = self.inputs.seed

        if len(self.inputs.corr_list) > 0:
            d['correlations'] = []
            for v1, v2, cor in self.inputs.corr_list:
                d['correlations'].append({'var1': v1, 'var2': v2, 'cor': format(cor, '.4f')})

        if self.longdescription is not None and self.longdescription != '':
            d['description'] = self.longdescription
        return d

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

        if 'unitdefs' in config:
            unitmgr.register_units(config['unitdefs'])

        u = cls()

        if 'samples' in config:
            u.inputs.nsamples = int(get_float(config, 'samples', 1E6))  # Use int(float()) to handle exponential notation

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
            u.inputs.seed = int(get_float(config, 'seed'))

        if 'correlations' in config:
            for cor in config['correlations']:
                u.inputs.correlate_vars(cor.get('var1'), cor.get('var2'), get_float(cor, 'cor'))

        if 'description' in config:
            u.longdescription = config['description']

        if 'name' in config:
            u.name = config['name']

        if u.model is None or len(u.model.exprs) == 0 or len(u.inputs) == 0:
            # No functions/inputs defined in file, probably wrong file type!
            return None
        return u

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
        except yaml.YAMLError:
            return None  # Can't read YAML

        if isinstance(config, list):
            # To support old (1.0) and new (1.1+) style config files
            config = config[0]

        u = cls.from_config(config)
        return u


UncertaintyCalc = UncertCalc
