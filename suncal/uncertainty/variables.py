''' Manage RandomVariables assigned to an uncertainty propagation model '''

import warnings
from collections import namedtuple
import numpy as np
import sympy
from scipy import stats

from ..common import matrix, unitmgr, uparser
from ..common.multivariate_t import multivariate_t_rvs
from ..common.distributions import get_distribution, get_argnames
from ..datasets.dataset_model import DataSet
from ..datasets.dataset import uncert_autocorrelated


VariableInfo = namedtuple('VariableInfo', ['expected', 'uncertainty', 'degf',
                                           'correlation', 'descriptions', 'components'])


class RandomVariable:
    ''' A random variable with one Type A and 0+ Type B uncertainties. Note
        a Type A in this context means a set of measurements is stored and
        evaluated for the uncertainty.

        Args:
            value (float or ndarray): scalar or vector measured values for expected
              and type A uncertainty. May also be a Pint Quantity with units.
    '''
    def __init__(self, value=0):
        self.value = np.atleast_1d(value)
        self._typea = None  # Explicit Type A uncertainty when value is scalar
        self._typeb = []  # List of Type B components
        self.description = ''
        self.num_new_meas = self.value.size
        self._autocor = True

    def __repr__(self):
        return f'<RandomVariable {self.expected} ± {self.uncertainty} (k = 1)>'

    def measure(self, values, units=None, typea=None, num_new_meas=None,
                autocor=True, description=None):
        ''' Add measurement to the random variable.

            Args:
                values: (float or ndarray) Measured value(s). If vector,
                  this RandomVariable will have a Type A uncertainty.
                  May also be a Pint Quantity with units.
                units (str): units to apply to value
                typea (float): Type A uncertainty to apply if values is
                  scalar and an uncertainty is known. Ignored when values
                  is array.
                num_new_meas (int): Number of new measurements to apply
                  this uncertainty to. N in stddev/sqrt(N).
                autocor (bool): Adjust for autocorrelation in 1D data (if
                  significant and length>50)
                description (str): description of the variable

            Returns:
                The same RandomVariable object (use for chaining function calls)
        '''
        self.value = np.atleast_1d(values)
        self._typea = typea

        self._autocor = autocor
        if units:
            self.value = unitmgr.make_quantity(self.value, units)
        if num_new_meas is not None:
            self.num_new_meas = num_new_meas
        else:
            self.num_new_meas = self.value.size
        if description is not None:
            self.description = description
        return self   # for chaining

    def clear_typeb(self):
        ''' Remove all type b components '''
        self._typeb = []
        return self

    def typeb(self, dist='normal', description='', **kwargs):
        ''' Add a type b uncertainty

            Args:
                dist (str): name of probability distribution
                description (str): description for the uncertainty
                **kwargs: arguments passed to suncal.common.distribution.Distribution

            Returns:
                The same RandomVariable object (use for chaining multiple typeb function calls)
        '''
        kwargs.pop('nominal', None)
        self._typeb.append(Typeb(dist=dist, nominal=self.expected, description=description, **kwargs))
        return self   # for chaining

    def _typea_variance_ofmean(self):
        ''' Calculate Type A variance of the mean '''
        values, units = unitmgr.split_units(self.value)
        varunits = str(units**2) if units else None
        if values.size < 2:
            unc = self._typea**2 if self._typea is not None else 0
            if units:
                return unitmgr.make_quantity(unc, varunits)
            return unc

        if len(self.value.shape) == 1:  # 1D, use regular variance
            autocor_factor = 1  # Autocorrelation multiplier
            if len(self.value) > 50 and self._autocor:
                unc = uncert_autocorrelated(values)
                if unc.r > 1.3:
                    autocor_factor = unc.r
            return unitmgr.make_quantity(autocor_factor * values.var(ddof=1) / self.num_new_meas, varunits)

        if self.num_new_meas != self.value.size:
            unc = DataSet(values).result.uncertainty.stdev**2 / self.num_new_meas
        else:
            unc = DataSet(values).result.uncertainty.stderr**2
        if units:
            return unitmgr.make_quantity(unc, varunits)
        return unc

    @property
    def units(self):
        ''' Units of measured value '''
        return unitmgr.get_units(self.value)

    @property
    def expected(self):
        ''' Expected value of the RandomVariable '''
        return np.nanmean(self.value)

    @property
    def variance(self):
        ''' Combined variance of the RandomVariable '''
        variance = self._typea_variance_ofmean()
        for typeb in self._typeb:
            typeb_variance = typeb.variance
            if unitmgr.has_units(variance) and not unitmgr.has_units(typeb_variance):
                typeb_variance = unitmgr.Quantity(typeb_variance, variance.units)
            variance += typeb_variance
        return variance

    @property
    def uncertainty(self):
        ''' Combined standard uncertainty of the RandomVariable '''
        return np.sqrt(self.variance)

    @property
    def typea(self):
        ''' Type A uncertainty '''
        return np.sqrt(self._typea_variance_ofmean())

    @property
    def degrees_freedom(self):
        ''' Effective degrees of freedom of the RandomVariable '''
        nu_a = np.inf
        if self.value.size > 1:
            nu_a = self.value.size - 1

        denom = self._typea_variance_ofmean()**2 / nu_a
        for typeb in self._typeb:
            denom += (typeb.variance**2 / typeb.degf)

        if denom == 0:
            return np.inf

        return unitmgr.strip_units(self.variance**2 / denom)

    @property
    def typeb_names(self):
        ''' Names of all Type B components defined in the RandomVariable '''
        return [b.name for b in self._typeb]

    @property
    def info(self):
        ''' Information about the RandomVariable uncertainty components '''
        uncerts = []
        typeaunc = self.typea
        if typeaunc != 0:
            uncerts.append({
                'name': 'Type A',
                'description': f'Type A uncertainty from {self.value.size} measurements',
                'data': self.value,
                'num_newmeas': self.num_new_meas,
                'uncertainty': typeaunc,
                'degf': self.value.size - 1
            })

        for typeb in self._typeb:
            uncerts.append({
                'name': typeb.name,
                'description': typeb.description,
                'dist': typeb.distname,
                'uncertainty': typeb.uncertainty,
                'degf': typeb.degf
            })
        return uncerts

    def get_typeb(self, name):
        ''' Get a Type B uncertainty by name

            Args:
                name (str): Name of the type b component

            Returns:
                Typeb instance
         '''
        names = [b.name for b in self._typeb]
        idx = names.index(name)
        return self._typeb[idx]

    def sample(self, nsamples):
        ''' Generate random samples (uncorrelated with other variables)

            Args:
                nsamples (int): Number of random samples

            Returns:
                Array of sampled values
        '''
        samples = 0
        units = None
        if self.value.size > 1 or self._typea is not None:
            mean = np.nanmean(self.value)
            unc = self.uncertainty
            if unitmgr.has_units(mean):
                units = mean.units
                mean = mean.magnitude
                uncunits = unc.units
                if (('delta_' in str(unc.units) or 'Δ' in str(unc.units)) and
                    ('delta_' not in str(units) and 'Δ' not in str(units))):
                        uncunits = unitmgr.parse_units(f'delta_{units}')
                unc = unc.to(uncunits).magnitude
            samples = stats.norm.rvs(mean, unc, nsamples)
            if units:
                samples = samples * units
        else:
            samples = np.nanmean(self.value)
            if unitmgr.has_units(samples):
                units = samples.units

        for typeb in self._typeb:
            b_samples = typeb.sample(nsamples)
            if units and not unitmgr.has_units(b_samples):
                b_samples = b_samples * units
            samples += b_samples
        return samples

    def sample_correlated(self, norm_samples):
        ''' Generate random samples, correlated with other RandomVariables

            Args:
                norm_samples (array): array of random normal samples generated
                with multivariate_normal, correlated with other
                norm_samples used for other RandomVariables.

            Returns:
                Array of random samples
         '''
        samples = 0
        units = None
        if self.value.size > 1 or self._typea is not None:
            mean = np.nanmean(self.value)
            unc = self.uncertainty
            if unitmgr.has_units(mean):
                units = mean.units
                mean = mean.magnitude
                unc = unc.to(units).magnitude
            samples += stats.norm.ppf(norm_samples, loc=mean, scale=unc)
            if units:
                samples = samples * units
        else:
            samples = np.nanmean(self.value)

        for typeb in self._typeb:
            b_samples = typeb.sample_correlated(norm_samples)
            if units and not unitmgr.has_units(b_samples):
                b_samples = b_samples * units
            samples += b_samples
        return samples


class Typeb:
    ''' Type B unceratinty component.

        User typically adds type b components via RandomVariable.typeb method.

        Args:
            dist (str): name of probability distribution
            name (str): name of Type B component
            nominal (float): Nominal value associated with the RandomVariable
            description (str): Description of the Type B component
            **kwargs: Arguments passed to suncal.common.distributions.Distribution
     '''
    def __init__(self, dist='normal', name=None, nominal=0, description='', **kwargs):
        if dist in ['gaussian', 'norm']:
            dist = 'normal'

        self.distname = dist
        self.description = description
        self.name = 'Type B' if name is None else name
        self.nominal = nominal
        self.degf = np.inf
        self.kwargs = kwargs
        self.units = kwargs.pop('units', None)
        if self.units and isinstance(self.units, str):
            self.units = unitmgr.parse_units(self.units)

        kwargs_magnitude = self._parse_kwds(kwargs)
        self.distribution = get_distribution(dist, **kwargs_magnitude)

    def set_kwargs(self, **newkwargs):
        ''' Set new kwargs to the Distribution instance '''
        self.kwargs.update(newkwargs)
        kwargs_magnitude = self._parse_kwds(self.kwargs)
        self.distribution = get_distribution(self.distname, **kwargs_magnitude)

    def set_nominal(self, nominal):
        ''' Change the nominal value '''
        self.nominal = nominal
        kwargs_magnitude = self._parse_kwds(self.kwargs)
        self.distribution = get_distribution(self.distname, **kwargs_magnitude)

    def _parse_kwds(self, kwargs):
        ''' Parse the keyword args, converting string uncertainties such as "1%" into values '''
        self.distname = kwargs.get('dist', self.distname)
        self.required_args = get_argnames(self.distname)
        newargs = {}
        for name, value in kwargs.items():
            value, units = unitmgr.split_units(value)
            if self.units is None:
                self.units = units

            if isinstance(value, str):
                # Allow entering % as % of nominal
                # or %range(X) as percent of X range
                # or ppm as ppm of nominal
                # or ppmrange(X) as ppm of X range
                nominal = unitmgr.strip_units(unitmgr.match_units(self.nominal, self.units))
                value = value.replace('%range(', '/100*(')
                value = value.replace('ppmrange(', '/1E6*(')
                value = value.replace('ppbrange(', '/1E9*(')
                value = value.replace('ppm', f'/1E6*{nominal}')
                value = value.replace('ppb', f'/1E9*{nominal}')
                value = value.replace('%', f'/100*{nominal}')
                try:
                    value = np.float64(uparser.callf(value))
                except (AttributeError, ValueError, TypeError):
                    value = np.nan
                except OverflowError:
                    value = np.inf
            newargs[name] = value  # Magnitude only

        if 'degf' in newargs:
            newargs['df'] = newargs.pop('degf')
        self.degf = newargs.get('df', np.inf)
        if 'df' in self.required_args and 'df' not in newargs:
            newargs['df'] = 1E9  # inf doesn't work with scipy.stats. Use large number.

        if unitmgr.has_units(self.nominal) and self.units is None:
            self.units = self.nominal.units
            # Uncertainties always use delta units
            if str(self.units) in ['degC', 'celsius', 'degree_Celsius']:
                self.units = unitmgr.ureg.delta_degC
            elif str(self.units) in ['degF', 'fahrenheit', 'degree_Fahrenheit']:
                self.units = unitmgr.ureg.delta_degF

        return newargs

    @property
    def variance(self):
        ''' Variance of the component '''
        variance = self.distribution.var()
        if self.units:
            variance = variance * self.units**2
        return variance

    @property
    def uncertainty(self):
        ''' Standard uncertainty of the component '''
        return np.sqrt(self.variance)

    def isvalid(self):
        ''' Check whether the distribution parameters are valid '''
        try:
            self.distribution.rvs()
        except (ValueError, TypeError):
            return False
        return True

    def sample(self, nsamples=1000000):
        ''' Generate random samples

            Args:
                nsamples (int): Number of random samples

            Returns:
                1D Array of random samples
        '''
        samples = self.distribution.rvs(nsamples)
        if self.units:
            samples = samples * self.units
        return samples

    def sample_correlated(self, norm_samples):
        ''' Generate random samples, correlated with other RandomVariables

            Args:
                norm_samples (array): array of random normal samples generated
                with multivariate_normal, correlated with other
                norm_samples used for other RandomVariables.

            Returns:
                1D Array of random samples
         '''
        samples = self.distribution.ppf(norm_samples)
        if self.units:
            samples = samples * self.units
        return samples

    def pdf(self, stds=6, num=200):
        ''' Get X and Y of probability Density Function

            Args:
                stds (float): Number of standard deviations to include
                  on each side of nominal
                num (int): Number of points in array

            Returns:
                X and Y arrays for plotting PDF of the uncertainty component
        '''
        plusminus = self.distribution.std() * stds
        mid = self.distribution.median()
        x = np.linspace(mid - plusminus, mid + plusminus, num=num)
        y = self.distribution.pdf(x)
        nom = unitmgr.strip_units(self.nominal)
        return x + nom, y


class Variables:
    ''' A collection of RandomVariables

        Args:
            *names (str): names of all new RandomVariables to define
    '''
    def __init__(self, *names):
        self.variables = {}
        for name in names:
            self.variables[name] = RandomVariable()
        self._correlation = np.eye(len(self.variables))

    @property
    def names(self):
        ''' List of names of RandomVariables '''
        return list(self.variables.keys())

    @property
    def expected(self):
        ''' Dictionary expected values of RandomVariables '''
        return {name: v.expected for name, v in self.variables.items()}

    @property
    def uncertainties(self):
        ''' Dictionary uncertainties values of RandomVariables '''
        return {name: v.uncertainty for name, v in self.variables.items()}

    @property
    def degrees_freedom(self):
        ''' Dictionary degrees of freedom of RandomVariables '''
        return {name: v.degrees_freedom for name, v in self.variables.items()}

    @property
    def units(self):
        ''' Dictionary Pint units of RandomVariables '''
        return {name: unitmgr.get_units(v.value) for name, v in self.variables.items()}

    @property
    def correlation_coefficients(self):
        ''' Dictionary of correlation coefficients between RandomVariables, including
            0 for uncorrelated values
        '''
        symbols = {}
        for idx1, name1 in enumerate(self.names):
            for idx2, name2 in enumerate(self.names):
                if name1 < name2:
                    symbols[f'sigma_{name1}{name2}'] = self._correlation[idx1, idx2]
                    symbols[f'sigma_{name2}{name1}'] = self._correlation[idx2, idx1]
        return symbols

    @property
    def info(self):
        ''' Info about all the RandomVariables '''
        components = {name: v.info for name, v in self.variables.items()}
        descriptions = {name: v.description for name, v in self.variables.items()}
        return VariableInfo(self.expected, self.uncertainties, self.degrees_freedom,
                            self.correlation_coefficients, descriptions, components)

    def symbol_values(self):
        ''' Return dictionary of ALL variable values, uncertainties, correlations
            (use for substituting in sympy expressions)
         '''
        # Rename uncertainties to u_X, degf to nu_X, for substitution into sympy expressions
        symbols = self.expected
        uncs = {f'u_{name}': unc for name, unc in self.uncertainties.items()}
        degf = {f'nu_{name}': df for name, df in self.degrees_freedom.items()}
        symbols.update(uncs)
        symbols.update(degf)
        symbols.update(self.correlation_coefficients)
        return symbols

    def get(self, name):
        ''' Get random variable by name '''
        return self.variables.get(name)

    def add(self, name, variable):
        ''' Add a random variable

            Args:
                name (str): Name of the variable
                variable (RandomVariable): Variable instance to add
         '''
        assert isinstance(variable, RandomVariable)
        self.variables[name] = variable

    def correlate(self, var1, var2, correlation):
        ''' Set correlation coefficient between two inputs

            Args:
                var1 (str): name of variable
                var2 (str): name of variable
                correlation (float): correlation coefficient between
                  the two variables
        '''
        idx1 = self.names.index(var1)
        idx2 = self.names.index(var2)
        self._correlation[idx1, idx2] = correlation
        self._correlation[idx2, idx1] = correlation

    def set_correlation(self, corr, names):
        ''' Set correlation of inputs as a matrix.

            Args:
                corr (array): (M,M) correlation matrix. Must be square where M
                    is number of inputs. Only upper triangle is considered.
                names (list): List of variable names corresponding to the rows/columns of cor
        '''
        for idx1, name1 in enumerate(names):
            for idx2, name2 in enumerate(names):
                if idx1 < idx2:
                    self.correlate(name1, name2, corr[idx1, idx2])

    def get_correlation_coeff(self, var1, var2):
        ''' Get correlation coefficient between two variables

            Args:
                var1 (str): name of variable
                var2 (str): name of variable
        '''
        idx1 = self.names.index(var1)
        idx2 = self.names.index(var2)
        return self._correlation[idx1, idx2]

    def correlation_symbolic(self):
        ''' Get correlation matrix as symbols

            Returns:
                List of lists of sympy expressions
        '''
        corr = []
        for idx1, name1 in enumerate(self.names):
            row = []
            for idx2, name2 in enumerate(self.names):
                if name1 == name2:
                    row.append(1.0)
                elif self._correlation[idx1][idx2] == 0:
                    row.append(0.0)
                else:
                    if idx1 < idx2:
                        row.append(sympy.Symbol(f'sigma_{name1}{name2}'))
                    else:
                        row.append(sympy.Symbol(f'sigma_{name2}{name1}'))
            corr.append(row)
        return corr

    def correlation_matrix(self):
        ''' Correlation matrix between variables '''
        return self._correlation

    def covariance_symbolic(self):
        ''' covariance matrix [Ux] as sympy expressions

            Returns:
                List of lists of sympy expressions
        '''
        S = []
        for i, name in enumerate(self.variables.keys()):
            row = [0]*i
            row.append(sympy.Symbol(f'u_{name}'))
            row.extend([0]*(len(self.variables) - i - 1))
            S.append(row)

        corr = self.correlation_symbolic()
        Ux = matrix.matmul(matrix.matmul(S, corr), S)
        return Ux

    def covariance(self):
        ''' Get covariance matrix [Ux] (numerical)

            Returns:
                List of lists of float
        '''
        Ux = []
        for i, v1 in enumerate(self.variables.values()):
            row = []
            for j, v2 in enumerate(self.variables.values()):
                if i == j:
                    row.append(v1.uncertainty**2)
                elif self._correlation[i, j] != 0:
                    row.append(v1.uncertainty * v2.uncertainty * self._correlation[i, j])
                else:
                    row.append(v1.uncertainty * v2.uncertainty * 0)  # *0 produces a 0 with correct units
            Ux.append(row)
        return Ux

    def has_correlation(self):
        ''' Determine whether any inputs are correlated '''
        for i in range(len(self.names)):
            for j in range(len(self.names)):
                if i < j and self._correlation[i, j] != 0:
                    return True
        return False

    def sample(self, nsamples=1000000, copula='gaussian'):
        ''' Generate random samples

            Args:
                nsamples (int): number of random samples
                copula (str): 'gaussian' or 't'

            Returns:
                Dictionary of arrays of random samples
         '''
        if self.has_correlation():
            norm_samples = self._correlated_samples(nsamples=nsamples, copula=copula)
            samples = {name: var.sample_correlated(norm_samples[name]) for name, var in self.variables.items()}
        else:
            samples = {name: var.sample(nsamples) for name, var in self.variables.items()}
        return samples

    def _correlated_samples(self, nsamples=1000000, copula='gaussian', degf=np.inf):
        ''' Generate correlated NORMAL random samples for each of the inputs

            Args:
                nsamples (int): number of random samples
                copula (str): 'gaussian' or 't'
                degf (float): Degrees of freedom for 't' copula

            Returns:
                Dictionary of arrays of random samples
         '''
        # Note: correlation==covariance since all std's are 1 right now.
        # Generate correlated random samples
        if copula == 'gaussian':
            with warnings.catch_warnings(record=True) as w:
                # Roundabout way of catching a numpy warning that should really be an exception
                warnings.simplefilter('always')
                normsamples = stats.multivariate_normal.rvs(cov=self._correlation, size=nsamples)
                if len(w) > 0:
                    warnings.warn('Correlation Matrix is not positive semi-definite')
                normsamples = stats.norm.cdf(normsamples)

        elif copula == 't':
            mean = np.zeros(self._correlation.shape[0])
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter('always')
                # Note: Scipy now has multivariate_t, but it does not account for covariance.
                normsamples = multivariate_t_rvs(mean=mean, corr=self._correlation, size=nsamples, df=degf)
                normsamples = stats.t.cdf(normsamples, df=degf)
                if len(w) > 0:
                    raise ValueError('Correlation Matrix is not positive semi-definite')
        else:
            raise ValueError(f'Unimplemented copula {copula}. Must be `gaussian` or `t`.')

        normsamples[np.where(normsamples == 1.0)] = 1 - 1E-9  # If rounded to 1 or 0, then we get infinity.
        normsamples[np.where(normsamples == 0.0)] = 1E-9
        samples = {}
        for idx, name in enumerate(self.names):
            samples[name] = normsamples[:, idx]
        return samples
