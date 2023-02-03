''' Uncertainty propagation project component '''

import numpy as np

from .component import ProjectComponent
from ..common import unitmgr, report

from ..uncertainty.model import Model
from ..uncertainty.report.units import units_report
from ..uncertainty.results.uncertainty import UncertaintyResults


class ProjectUncert(ProjectComponent):
    ''' Uncertainty project component '''
    def __init__(self, model=None, name='uncertainty'):
        super().__init__()
        self.name = name
        if model is not None:
            self.model = model
        else:
            self.model = Model('f=x')
            self.model.var('x').typeb(dist='normal', unc=1, k=2, name='u(x)')
        self.nsamples = 1000000
        self.seed = None
        self.outunits = {}
        self.result = None
        self.longdescription = None
        self.project = None  # Parent project

    def clear_sweeps(self):
        ''' Clear sweeps from model '''
        self.model.sweeplist = []

    def calculate(self, mc=True):
        ''' Run the calculation '''
        if self.seed:
            np.random.seed(self.seed)
        gumresult = self.model.calculate_gum()
        if mc:
            mcresult = self.model.monte_carlo(samples=self.nsamples)
            self.result = UncertaintyResults(gumresult, mcresult)
        else:
            self.result = UncertaintyResults(gumresult, None)
        if self.outunits is not None:
            self.result.units(**self.outunits)
        return self.result

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
                    'mean': unitmgr.strip_units(self.result.gum.expected[funcname]),
                    'std': unitmgr.strip_units(self.result.gum.uncertainty[funcname]),
                    'df': self.result.gum.degf[funcname]}
        return dists

    def units_report(self, **kwargs):
        ''' Create report showing units of all parameters '''
        return units_report(self.model, self.outunits)

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
            return

        hdr = []
        if inputs:
            varnames = self.result.montecarlo.variablenames
            vsamples = self.result.montecarlo.varsamples
            units = [report.Unit(unitmgr.split_units(vsamples[name])[1]).prettytext(bracket=True)
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
            for funcname in self.model.functionnames:
                funcsetup = {
                    'name': funcname,
                    'expr': str(self.model.basesympys.get(funcname, '')),
                    'desc': self.model.descriptions.get(funcname, ''),
                    'units': str(self.outunits.get(funcname)) if self.outunits.get(funcname) else None}
                d['functions'].append(funcsetup)

        for varname in self.model.varnames:
            variable = self.model.var(varname)
            _, units = unitmgr.split_units(variable.expected)
            vardict = {
                'name': varname,
                'mean': unitmgr.strip_units(variable.expected),
                'numnewmeas': variable.num_new_meas,
                'desc': variable.description,
                'units': str(units) if units else None,
                'typea': unitmgr.strip_units(variable.value).tolist()
            }
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

        if self.longdescription is not None and self.longdescription != '':
            d['description'] = self.longdescription
        return d

    def load_config(self, config):
        ''' Load config into this project instance '''
        if 'unitdefs' in config:
            unitmgr.register_units(config['unitdefs'])

        if len(config.get('functions', [])) > 0 and config.get('functions')[0]['expr'] != '':  # Callable functions will have none
            names = [func.get('name') for i, func in enumerate(config.get('functions'))]
            names = [name if name else f'f_{i}' for i, name in enumerate(names)]
            exprs = [f'{names[i]}={func["expr"]}' for i, func in enumerate(config.get('functions'))]
            model = Model(*exprs)
            self.model = model

        self.name = config.get('name', 'uncertainty')
        self.outunits = {func['name']: func.get('units') for func in config.get('functions')}
        self.model.descriptions = {func['name']: func.get('desc') for func in config.get('functions')}
        self.nsamples = config.get('samples', 1000000)
        self.longdescription = config.get('description')
        self.seed = config.get('seed')

        for variable in config.get('inputs', []):
            modelvar = self.model.var(variable['name'])
            modelvar._typeb = []
            modelvar.description = variable.get('desc', '')
            units = variable.get('units')
            value = unitmgr.make_quantity(variable.get('mean'), units)
            modelvar.measure(value, num_new_meas=variable.get('numnewmeas'),
                             description=variable.get('desc', ''))
            for uncert in variable.get('uncerts', []):
                desc = uncert.pop('desc', None)
                dist = uncert.pop('dist', 'normal')
                modelvar.typeb(dist, description=desc, **uncert)

        for cor in config.get('correlations', []):
            self.model.variables.correlate(cor['var1'], cor['var2'], cor['cor'])
