''' Legacy UncertCalc API. Deprecated in suncal 1.6. '''

import warnings
from collections import namedtuple
import numpy as np

from ..common import unitmgr
from ..uncertainty import Model, ModelCallable
from ..project import ProjectUncert


def make_qty(value):
    ''' Convert value into a quantity if not already. Old interface always returned
        Pint Quantity, even when dimensionless
    '''
    if unitmgr.has_units(value):
        return value
    return unitmgr.make_quantity(value, 'dimensionless')


class UncertCalc(ProjectUncert):
    def __init__(self, function=None, inputs=None, units=None, samples=1000000, seed=None, name='uncertainty',
                 finnames=None, foutnames=None, finunits=None, foutunits=None):

        warnings.warn('UncertCalc class is deprecated. Use suncal.Model.', DeprecationWarning)

        if function is None:
            model = Model()

        else:
            if not isinstance(function, (list, tuple)):
                function = [function]

            if callable(function[0]):
                if foutunits is not None and isinstance(foutunits, str):
                    foutunits = [foutunits]
                model = ModelCallable(*function, names=foutnames, unitsin=finunits, unitsout=foutunits)
            else:
                model = Model(*function)
        super().__init__(model, name=name)
        self.inputs = self.model.variables
        self.longdescription = ''
        self.name = name
        self.out = None   # Output object
        self.seed = seed
        self.nsamples = int(samples)
        self._copula = 'gaussian'
        if units is not None:
            if isinstance(units, str):
                units = [units]

            if isinstance(model, ModelCallable):
                names = foutnames
            else:
                names = self.model.functionnames
            self.outunits = {name: unit for name, unit in zip(names, units)}

        if inputs is not None:
            config = self.get_config()
            for inpt in inputs:
                # "nom" vs "mean" allowed in config..
                inpt['mean'] = inpt.get('nom', inpt.get('mean', 0))
            config['inputs'] = inputs
            self.load_config(config)

    @property
    def required_inputs(self):
        return self.model.variables.names

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
        assert isinstance(self.model, Model)  # Not ModelCallable - only one function supported
        if idx is None and name is not None and name in self.model.functionnames:
            idx = self.model.functionnames.index(name)
        elif idx is None:
            idx = 0

        config = self.get_config()
        config['functions'].insert(idx, {
            'name': name,
            'expr': func,
            'desc': desc,
            'units': outunits})
        self.load_config(config)

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
        config = self.get_config()
        inputs = config.get('inputs')
        names = [inpt['name'] for inpt in inputs]
        idx = names.index(name)
        inputdict = inputs[idx]
        inputdict['mean'] = nom
        if units is not None:
            inputdict['units'] = units
        inputdict['desc'] = desc
        uncertlist = inputdict.get('uncerts')

        if len(kwargs) > 0:
            uncertnames = [unc['name'] for unc in uncertlist]
            try:
                uncidx = uncertnames.index(name)
                uncert = uncertlist[uncidx]
            except ValueError:
                uncertlist.append({})
                uncert = uncertlist[-1]

            uncert['name'] = kwargs.pop('uname', f'u({name})')
            uncert['desc'] = ''
            uncert.update(**kwargs)
        self.load_config(config)

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
        config = self.get_config()
        inputs = config.get('inputs')
        names = [inpt['name'] for inpt in inputs]
        idx = names.index(var)
        inputdict = inputs[idx]
        uncertlist = inputdict.get('uncerts')

        uncertnames = [unc['name'] for unc in uncertlist]
        try:
            uncidx = uncertnames.index(name)
            uncert = uncertlist[uncidx]
        except ValueError:
            uncertlist.append({})
            uncert = uncertlist[-1]

        uncert['name'] = name
        uncert['dist'] = dist
        if units is not None:
            uncert['units'] = str(units)
        uncert['desc'] = ''
        uncert['degf'] = degf
        uncert.update(**args)
        self.load_config(config)

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
        self.model.variables.correlate(var1, var2, correlation)

    def set_correlation(self, corr, names, copula='gaussian'):
        ''' Set correlation of inputs as a matrix.

            Args:
                corr (array): (M,M) correlation matrix. Must be square where M
                    is number of inputs. Only upper triangle is considered.
                names (list): List of variable names corresponding to the rows/columns of cor
                copula: (str): Copula for correlating the inputs, either 'gaussian' or 't'
        '''
        self._copula = copula
        for idx1, name1 in enumerate(names):
            for idx2, name2 in enumerate(names):
                if idx1 < idx2:
                    self.model.variables.correlate(name1, name2, corr[idx1, idx2])

    def calculate(self, GUM=True, MC=True, samples=1000000):
        self.nsamples = samples
        report = super().calculate(mc=MC)
        self.out = UncertOutput(report)
        return self.out


class GUMOutput:
    def __init__(self, gumresult):
        self._gumresult = gumresult

    def _repr_markdown_(self):
        ''' Markdown representation for display in Jupyter '''
        return self.report().get_md()

    def uncert(self, idx=0):
        fname = self._gumresult.functionnames[idx]
        return make_qty(self._gumresult.uncertainty[fname])

    def nom(self, idx=0):
        fname = self._gumresult.functionnames[idx]
        return make_qty(self._gumresult.expected[fname])

    def degf(self, idx=0):
        fname = self._gumresult.functionnames[idx]
        return self._gumresult.degf[fname]

    def expanded(self, fidx=0, cov=0.95, **kwargs):
        fname = self._gumresult.functionnames[fidx]
        Expanded = namedtuple('Expanded', ['uncertainty', 'k'])
        exp = self._gumresult.expanded(conf=cov)
        return Expanded(make_qty(exp[fname].uncertainty), make_qty(exp[fname].k))

    def report(self, **kwargs):
        return self._gumresult.summary()

    def report_correlation(self, **kwargs):
        return self._gumresult.correlation(**kwargs)

    def report_derivation(self, solve=False, **kwargs):
        return self._gumresult.derivation(solve=solve, **kwargs)

    def report_sens(self, **kwargs):
        return self._gumresult.sensitivity(**kwargs)

    def plot_pdf(self, plot=None, funcidx=None, **kwargs):
        if funcidx is not None:
            if not isinstance(funcidx, (list, tuple)):
                funcidx = [funcidx]
            kwargs['functions'] = [self._gumresult.functionnames[f] for f in funcidx]
        self._gumresult.plot.pdf(fig=plot, **kwargs)

    def plot_correlation(self, plot=None, **kwargs):
        return self._gumresult.plot.joint_pdf(fig=plot, **kwargs)


class MCOutput:
    def __init__(self, mcresult):
        self._mcresult = mcresult

    def _repr_markdown_(self):
        ''' Markdown representation for display in Jupyter '''
        return self.report().get_md()

    def samples(self, idx=0):
        fname = self._mcresult.functionnames[idx]
        return self._mcresult.samples[fname]

    def uncert(self, idx=0):
        fname = self._mcresult.functionnames[idx]
        return make_qty(self._mcresult.uncertainty[fname])

    def nom(self, idx=0):
        fname = self._mcresult.functionnames[idx]
        return make_qty(self._mcresult.expected[fname])

    def expanded(self, fidx=0, cov=0.95, **kwargs):
        fname = self._mcresult.functionnames[fidx]
        Expanded = namedtuple('Expanded', ['minimum', 'maximum', 'k'])
        exp = self._mcresult.expanded(conf=cov, shortest=kwargs.get('shortest', False))
        return Expanded(make_qty(exp[fname].low), make_qty(exp[fname].high), exp[fname].k)

    def report(self, **kwargs):
        return self._mcresult.report.summary(**kwargs)

    def plot_pdf(self, plot=None, funcidx=None, **kwargs):
        if funcidx is not None:
            if not isinstance(funcidx, (list, tuple)):
                funcidx = [funcidx]
            kwargs['functions'] = [self._mcresult.functionnames[f] for f in funcidx]
        kwargs.pop('histtype', None)  # Removed kwarg
        self._mcresult.report.plot.pdf(fig=plot, **kwargs)

    def plot_xscatter(self, plot=None, contour=False, **kwargs):
        inpts = kwargs.pop('inpts', None)
        if inpts is not None:
            kwargs['variables'] = [self._mcresult.variablenames[f] for f in inpts]

        if contour:
            return self._mcresult.report.plot.variable_contour(fig=plot, **kwargs)
        return self._mcresult.report.plot.variable_scatter(fig=plot, **kwargs)

    def plot_xhists(self, fig=None, **kwargs):
        inpts = kwargs.pop('inpts', None)
        if inpts is not None:
            kwargs['variables'] = [self._mcresult.variablenames[f] for f in inpts]
        return self._mcresult.report.plot.variable_hist(fig=fig, **kwargs)

    def plot_correlation(self, plot=None, funcs=None, contour=True, **kwargs):
        # Old used indexes, new uses names
        functions = kwargs.pop('funcs', None)
        if functions is not None:
            kwargs['functions'] = [self._mcresult.functionnames[f] for f in functions]

        # Can only do one interval now
        intervals = kwargs.pop('intervals', None)
        if intervals is not None:
            kwargs['interval'] = intervals[-1]

        if contour:
            self._mcresult.report.plot.joint_pdf(**kwargs)
        else:
            kwargs.pop('bins', None)
            self._mcresult.report.plot.scatter(**kwargs)

    def plot_converge(self, fig=None, div=25, relative=False, **kwargs):
        return self._mcresult.report.plot.converge(fig=fig, div=div, relative=relative, **kwargs)


class UncertOutput:
    def __init__(self, uncreport):
        self._result = uncreport
        self.gum = GUMOutput(self._result.gum)
        self.mc = MCOutput(self._result.montecarlo)

    def _repr_markdown_(self):
        ''' Markdown representation for display in Jupyter '''
        return self.report().get_md()

    def report(self, **kwargs):
        return self._result.report.summary(**kwargs)

    def report_summary(self, **kwargs):
        return self._result.report.summary_withplots(**kwargs)

    def report_expanded(self, conf=.95, shortest=False, **kwargs):
        return self._result.report.expanded(conf=conf, shortest=shortest, **kwargs)

    def report_sens(self, **kwargs):
        return self._result.report.sensitivity(**kwargs)

    def report_inputs(self, **kwargs):
        return self._result.report.variables.summary(**kwargs)

    def report_components(self, **kwargs):
        return self._result.report.variables.components(**kwargs)

    def report_all(self, **kwargs):
        return self._result.report.all(setup={}, **kwargs)

    def plot_pdf(self, plot=None, **kwargs):
        # Old used indexes, new uses names
        functions = kwargs.pop('funcs', None)
        if functions is not None:
            kwargs['functions'] = [self._result.report.functionnames[f] for f in functions]

        # Can only do one interval now
        intervals = kwargs.pop('intervals', None)
        if intervals is not None:
            kwargs['interval'] = intervals[-1]

        kwargs.pop('label', None)  # Label-by-description not implemented
        return self._result.report.plot.pdf(fig=plot, **kwargs)

    def plot_correlation(self, plot=None, funcs=None, contour=True, bins=35, **kwargs):
        # Seemed to use plot and fig args interchangeably
        if 'fig' in kwargs:
            plot = kwargs.pop('fig', None)

        # Old used indexes, new uses names
        functions = kwargs.pop('funcs', None)
        if functions is not None:
            kwargs['functions'] = [self._result.report.functionnames[f] for f in functions]

        kwargs.pop('label', None)  # Label-by-description not implemented
        return self._result.report.plot.joint_pdf(fig=plot, **kwargs)
