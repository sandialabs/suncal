''' Uncertainty propagation project component '''
from enum import IntEnum, auto
from decimal import Decimal
import numpy as np

from .component import ProjectComponent
from ..common import unitmgr, report
from ..common.limit import Limit

from ..uncertainty.model import Model
from ..uncertainty.report.units import units_report
from ..uncertainty.results.uncertainty import UncertaintyResults


class MeasuredDataType(IntEnum):
    ''' Type of measured data for variable '''
    SINGLE = auto()
    REPEATABILITY = auto()
    REPRODUCIBILITY = auto()


class ProjectUncert(ProjectComponent):
    ''' Uncertainty project component '''
    def __init__(self, model=None, name='uncertainty'):
        super().__init__(name=name)
        if model is not None:
            self.model = model
        else:
            self.model = Model()
        self.nsamples = 1000000
        self.seed = None
        self.outunits: dict[str, str] = {}  # {varname: unitstring}
        self.variablesdone: list[str] = []  # List of variable names defined in the wizard
        self.iswizard = False  # Use wizard GUI interface

    def setdefault_model(self):
        ''' Set a default measurement model '''
        self.model = Model('f=x')
        self.model.var('x').typeb(dist='normal', unc=1, k=2, name='u(x)')

    @property
    def missingvars(self):
        ''' Get list of variable names that have not yet been defined '''
        return sorted(list(set(self.model.varnames).difference(self.variablesdone)))

    def variable_type(self, varname: str) -> MeasuredDataType:
        ''' Get measured type of the variable '''
        variable = self.model.var(varname)
        value = np.asarray(variable.value)
        if value.ndim > 1:
            return MeasuredDataType.REPRODUCIBILITY
        if len(value) > 1:
            return MeasuredDataType.REPEATABILITY
        return MeasuredDataType.SINGLE

    def set_function(self, *funcs):
        ''' Change the model function, without changing variable definitions '''
        if len(self.model.exprs) == 0 or self.model.exprs[0] != funcs[0]:
            definedvars = self.model.variables.variables
            self.model = Model(*funcs)

            # Restore any variables already defined
            varnames = self.model.variables.names
            for varname, var in definedvars.items():
                if varname in varnames:
                    self.model.variables.variables[varname] = var

    def measure_variable(self, name, data, units=None, num_newmeas=None, autocor=True):
        ''' Set measured value of the variable '''
        data = np.asarray(data)
        dimension = len(data.shape)
        if dimension == 0:
            data = unitmgr.make_quantity(float(data), units)
            self.model.var(name).measure(data)
        elif dimension == 1:
            if units:
                data = unitmgr.make_quantity(data, units)
            self.model.var(name).measure(data, num_new_meas=num_newmeas, autocor=autocor)
        elif dimension == 2:
            if units:
                data = unitmgr.make_quantity(data, units)
            self.model.var(name).measure(data, num_new_meas=num_newmeas)
        else:
            raise NotImplementedError

        if units is not None:
            for typeb in self.model.var(name)._typeb:
                if data.dimensionality != typeb.units.dimensionality:
                    typeb.units = data.units

        self.variablesdone.append(name)

    def calculate(self, mc=True):
        ''' Run the calculation '''
        if self.seed:
            np.random.seed(self.seed)
        gumresult = self.model.calculate_gum()
        if mc:
            mcresult = self.model.monte_carlo(samples=self.nsamples)
            self._result = UncertaintyResults(gumresult, mcresult)
        else:
            self._result = UncertaintyResults(gumresult, None)
        if self.outunits is not None:
            self._result.units(**self.outunits)
        return self._result

    def get_dists(self):
        ''' Get distributions resulting from this calculation. If name is none,
            return a list of available distribution names.
        '''
        dists = {}
        for funcname in self.model.functionnames:
            if self.result.montecarlo is not None:
                name = f'{funcname} (MC)'
                samples = self.result.montecarlo.samples[funcname]
                dists[name] = {
                    'samples': samples,
                    'median': np.median(samples),
                    'expected': self.result.montecarlo.expected[funcname]}

            if self.result.gum is not None:
                name = f'{funcname} (GUM)'
                dists[name] = {
                    'median': unitmgr.strip_units(self.result.gum.expected[funcname]),
                    'std': unitmgr.strip_units(self.result.gum.uncertainty[funcname]),
                    'df': self.result.gum.degf[funcname]}
        return dists

    def units_report(self, **kwargs):
        ''' Create report showing units of all parameters '''
        return units_report(self.model, self.outunits, **kwargs)

    def clear_uncertainties(self):
        ''' Clear uncertainties from the project '''
        for varname in self.model.varnames:
            variable = self.model.var(varname)
            variable.clear_typeb()
            variable._typea = None

    def save_samples_csv(self, fname):
        ''' Save Monte Carlo samples to CSV file '''
        hdr, csv = self._build_samples_table()
        np.savetxt(fname, csv, delimiter=', ', fmt='%.8e', header=', '.join(hdr))

    def save_samples_npz(self, fname):
        ''' Save Monte Carlo samples to numpy NPZ file '''
        hdr, csv = self._build_samples_table()
        np.savez_compressed(fname, samples=csv, hdr=hdr)

    def _build_samples_table(self, inputs=True, outputs=True):
        if not (outputs or inputs):
            return None, None

        hdr = []
        if inputs:
            varnames = self.result.montecarlo.variablenames
            vsamples = self.result.montecarlo.varsamples
            units = [report.Unit(unitmgr.get_units(vsamples[name])).prettytext(bracket=True)
                     for name in varnames]
            hdr = [f'{name}{unit}' for name, unit in zip(varnames, units)]
            csv = np.stack([unitmgr.strip_units(vsamples[vname]) for vname in varnames], axis=1)

        if outputs:
            funcnames = self.result.montecarlo.functionnames
            fsamples = self.result.montecarlo.samples
            out = np.array([unitmgr.strip_units(fsamples[name]) for name in funcnames]).T
            units = [report.Unit(unitmgr.parse_units(self.result.montecarlo._units.get(fname))).prettytext(bracket=True)
                     for fname in funcnames]
            hdr.extend([f'{str(f)}{u}' for f, u in zip(funcnames, units)])
            if inputs:
                csv = np.hstack([csv, out])
            else:
                csv = out
        return hdr, csv

    def get_config(self):
        ''' Get configuration dictionary describing this calculation '''
        d = {}
        d['name'] = self.name
        d['mode'] = 'uncertainty'
        d['samples'] = self.nsamples
        d['functions'] = []
        d['inputs'] = []

        customunits = unitmgr.get_customunits()
        if customunits:
            d['unitdefs'] = customunits

        if self.model.functionnames:  # Callable models won't have this
            for i, funcname in enumerate(self.model.functionnames):
                funcsetup = {
                    'name': funcname,
                    'expr': str(self.model.exprs[i]),
                    'desc': self.model.descriptions.get(funcname, ''),
                    'tolerance': self.model.tolerances.get(funcname).config() if self.model.tolerances.get(funcname) else None,
                    'units': str(self.outunits.get(funcname)) if self.outunits.get(funcname) else None}
                d['functions'].append(funcsetup)

        for varname in self.model.varnames:
            variable = self.model.var(varname)
            units = unitmgr.get_units(variable.expected)
            typea = variable.typea
            vardict = {
                'name': varname,
                'mean': unitmgr.strip_units(variable.expected),
                'numnewmeas': variable.num_new_meas if variable._typea else None,
                'autocorrelate': variable._autocor,
                'desc': variable.description,
                'units': str(units) if units else None,
            }
            if typea is not None and typea > 0:
                vardict['typea'] = unitmgr.strip_units(variable.value).tolist()

            if variable._typea is not None:
                vardict['typea_uncert'] = variable._typea

            uncertlist = []
            for typeb in variable._typeb:
                uncdict = {
                    'name': typeb.name,
                    'desc': typeb.description,
                    'degf': typeb.degf,
                    'units': str(typeb.units) if typeb.units else None,
                    'dist': typeb.distname,
                }
                kwargs = {name: unitmgr.strip_units(value) for name, value in typeb.kwargs.items()}
                uncdict.update(kwargs)
                uncertlist.append(uncdict)
            vardict['uncerts'] = uncertlist
            d['inputs'].append(vardict)

        if self.seed is not None:
            d['seed'] = self.seed

        if self.model.variables.has_correlation():
            d['correlations'] = []
            for i, var1 in enumerate(self.model.variables.names):
                for j, var2 in enumerate(self.model.variables.names):
                    if i < j:
                        corr = self.model.variables.get_correlation_coeff(var1, var2)
                        if corr != 0:
                            d['correlations'].append({'var1': var1, 'var2': var2, 'cor': f'{corr:.4f}'})

        if self.description is not None and self.description != '':
            d['description'] = self.description
        return d

    def load_config(self, config):
        ''' Load config into this project instance '''
        if 'unitdefs' in config:
            unitmgr.register_units(config['unitdefs'])

        if len(config.get('functions', [])) > 0 and config.get('functions')[0]['expr'] != '':  # Callable functions will have none
            names = [func.get('name') for i, func in enumerate(config.get('functions'))]
            names = [name if name else f'f_{i}' for i, name in enumerate(names)]
            exprs = [f'{names[i]}={func["expr"]}' for i, func in enumerate(config.get('functions'))]
            self.model = Model(*exprs)

        self.name = config.get('name', 'uncertainty')
        self.outunits = {func['name']: func.get('units') for func in config.get('functions')}
        self.model.descriptions = {func['name']: func.get('desc') for func in config.get('functions')}
        for func in config.get('functions'):
            if (limcfg := func.get('tolerance')):
                self.model.tolerances[func['name']] = Limit.from_config(limcfg)
        self.nsamples = config.get('samples', 1000000)
        self.description = config.get('description', config.get('desc'))
        self.seed = config.get('seed')

        for variable in config.get('inputs', []):
            modelvar = self.model.var(variable['name'])
            modelvar._typeb = []
            modelvar.description = variable.get('desc', '')
            units = variable.get('units')
            typea = variable.get('typea', None)
            if typea is not None:
                value = unitmgr.make_quantity(typea, units)
            else:
                value = unitmgr.make_quantity(variable.get('mean'), units)
            modelvar.measure(value, num_new_meas=variable.get('numnewmeas'),
                             typea=variable.get('typea_uncert'),
                             autocor=variable.get('autocorrelate', True),
                             description=variable.get('desc', ''))
            for uncert in variable.get('uncerts', []):
                desc = uncert.pop('desc', None)
                dist = uncert.pop('dist', 'normal')
                modelvar.typeb(dist, description=desc, **uncert)

        for cor in config.get('correlations', []):
            self.model.variables.correlate(cor['var1'], cor['var2'], cor['cor'])
