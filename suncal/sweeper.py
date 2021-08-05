''' Methods for running multiple uncertainty calculations sweeping over input arrays.

    Use UncertSweep to set up a sweep on an existing UncertCalc object.

    The SweepOutput* classes hold the output of an UncertSweep calculation.
'''
from collections import namedtuple
import numpy as np
import sympy
import yaml
import matplotlib.pyplot as plt

from . import dataset
from . import reverse
from . import uncertainty
from . import output
from . import uparser
from . import report
from . import plotting
from .unitmgr import ureg


class UncertSweep(object):
    ''' Class for running multiple uncertainty calculations over a range of input values

        Parameters
        ----------
        unccalc: Uncertainty Propagation Object
            Uncertainty Calculator to sweep
    '''
    def __init__(self, unccalc, name='sweep'):
        self.unccalc = unccalc
        self.sweeplist = []
        self.name = name
        self.out = None

    def clear_sweeps(self):
        ''' Remove all sweeps from the sweeplist '''
        self.sweeplist = []

    def add_sweep_nom(self, varname, values, units=None):
        ''' Add sweep of nominal value.

            Parameters
            ----------
            varname: string
                Name of variable to sweep nominal value
            values: array
                Values for sweep
        '''
        d = {'var': varname, 'comp': 'nom', 'values': values, 'units': units}
        self.sweeplist.append(d)

    def add_sweep_df(self, varname, values, comp=None):
        ''' Add sweep of degrees of freedom.

            Parameters
            ----------
            varname: string
                Name of variable to sweep deg.freedom value
            values: array
                Values for sweep
        '''
        if comp is None:
            comp = 'u({})'.format(varname)
        d = {'var': varname, 'comp': comp, 'param': 'df', 'values': values}
        self.sweeplist.append(d)

    def add_sweep_corr(self, var1, var2, values):
        ''' Add sweep of correlation coefficient between var1 and var2 '''
        d = {'var': 'corr', 'var1': var1, 'var2': var2, 'values': values}
        self.sweeplist.append(d)

    def add_sweep_unc(self, varname, values, comp=None, param='std', units=None):
        ''' Add sweep of uncertainty component parameter.

            Parameters
            ----------
            varname: string
                Name of variable to sweep uncertainty component
            values: array
                Values for sweep
            comp: string
                Name of uncertainty component. Defaults to u_{varname}.
            param: string
                Name of uncertainty parameter in distribution.
            index: int
                Index of sweep item in sweeplist. Use None to add a new item
                to the list.
        '''
        if comp is None:
            comp = 'u({})'.format(varname)
        d = {'var': varname, 'comp': comp, 'param': param, 'values': values, 'units': units}
        self.sweeplist.append(d)

    def calcGUM(self):
        ''' Calculate sweep using GUM method '''
        return self.calculate(gum=True, mc=False)

    def calcMC(self, samples=1000000):
        ''' Calculate sweep using MC method '''
        return self.calculate(samples=samples, gum=False, mc=True)

    def calculate(self, samples=1000000, gum=True, mc=True):
        ''' Calculate using both GUM and MC methods '''
        if len(self.sweeplist) == 0:
            raise ValueError('No sweeps defined.')

        N = 0
        for sweepparams in self.sweeplist:
            # Note: all N's should be the same...
            N = max(N, len(sweepparams.get('values', [])))

        reportlist = []
        for sweepidx in range(N):
            # Make a copy (using config dictionary) so we don't destroy the original uncertcalc object and overwrite inputs
            # Note: dont use deepcopy() or we'll be copying output data too
            ucalccopy = uncertainty.UncertCalc.from_config(self.unccalc.get_config())
            for sweepparams in self.sweeplist:
                inptname = sweepparams.get('var', None)
                comp = sweepparams.get('comp', 'nom')
                param = sweepparams.get('param', None)
                values = sweepparams.get('values', [])
                units = sweepparams.get('units', None)

                if inptname == 'corr':
                    ucalccopy.correlate_vars(sweepparams['var1'], sweepparams['var2'], values[sweepidx])
                elif comp == 'nom':
                    inptvar = ucalccopy.get_inputvar(inptname)
                    inptvar.set_nom(values[sweepidx])
                    if units:
                        inptvar.set_units(units)
                elif param == 'df':
                    inptvar = ucalccopy.get_inputvar(inptname)
                    inptvar.get_comp(comp).degf = values[sweepidx]
                else:
                    inptvar = ucalccopy.get_inputvar(inptname)
                    distname = inptvar.get_comp(comp).distname
                    ucalccopy.set_uncert(var=inptname, name=comp, dist=distname, units=units, **{param: values[sweepidx]})

            reportlist.append(ucalccopy.calculate(gum=gum, mc=mc, samples=samples))
        self.out = SweepOutput(reportlist, self.sweeplist)
        return self.out

    def get_output(self):
        ''' Get output object (or None if not calculated yet) '''
        return self.out

    @classmethod
    def from_configfile(cls, fname):
        ''' Read and parse the configuration file. Returns a new UncertSweep
            instance.

            Parameters
            ----------
            fname: string or file object
                File name or open file object to read configuration from
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

        u = cls.from_config(config[0])  # config yaml is always a list
        return u

    @classmethod
    def from_config(cls, config):
        unccalc = uncertainty.UncertCalc.from_config(config)
        newsweep = cls(unccalc)

        sweeps = config.get('sweeps', [])
        for sweep in sweeps:
            var = sweep['var']
            comp = sweep.get('comp', None)
            param = sweep.get('param', None)
            units = sweep.get('units', None)
            values = sweep['values']
            if var == 'corr':
                newsweep.add_sweep_corr(sweep.get('var1', None), sweep.get('var2', None), values)
            elif comp == 'nom':
                newsweep.add_sweep_nom(var, values, units=units)
            elif param == 'df':
                newsweep.add_sweep_df(var, values, comp)
            else:
                newsweep.add_sweep_unc(var, values, comp, param, units=units)
        return newsweep

    def get_config(self):
        ''' Get configuration dictionary '''
        d = self.unccalc.get_config()
        d['mode'] = 'sweep'
        sweeps = []
        for sweep in self.sweeplist:
            sweep['values'] = list(sweep.get('values', []))
            sweeps.append(sweep)
        d['sweeps'] = sweeps
        return d

    def save_config(self, fname):
        ''' Save configuration to file. '''
        d = self.get_config()
        out = yaml.dump([d], default_flow_style=False)  # Can't use safe_dump with our np.float64 representer. But we still safe_load.
        try:
            fname.write(out)
        except AttributeError:
            with open(fname, 'w') as f:
                f.write(out)


class UncertSweepReverse(UncertSweep):
    ''' Sweep a reverse propagation calculator '''
    def __init__(self, unccalc, name='reversesweep'):
        super().__init__(unccalc, name=name)   # Just override the default name

    def calculate(self, samples=1000000, gum=True, mc=True):
        ''' Calculate reverse propagation sweep.

            Parameters
            ----------
            gum: boolean
                Compute by solving GUM uc expression for ui
            mc: boolean
                Compute using Monte Carlo of reverse expression
            samples: int
                Number of Monte Carlo samples
        '''
        N = 0
        for sweepparams in self.sweeplist:
            # Note: all N's should be the same...
            N = max(N, len(sweepparams.get('values', [])))

        reportlist = []
        for sweepidx in range(N):
            for sweepparams in self.sweeplist:
                inptname = sweepparams.get('var', None)
                comp = sweepparams.get('comp', 'nom')
                param = sweepparams.get('param', None)
                values = sweepparams.get('values', [])
                # Make a full copy (using config dictionary) so we don't destroy the original uncertcalc object and overwrite inputs
                ucalccopy = reverse.UncertReverse.from_config(self.unccalc.get_config())
                if inptname == 'corr':
                    ucalccopy.correlate_vars(sweepparams['var1'], sweepparams['var2'], values[sweepidx])
                elif comp == 'nom':
                    inptvar = ucalccopy.get_inputvar(inptname)
                    inptvar.set_nom(values[sweepidx])
                elif param == 'df':
                    inptvar = ucalccopy.get_inputvar(inptname)
                    inptvar.degf = lambda: values[sweepidx]  # Clobbers the InputVar.degf() method!
                else:
                    ucalccopy.set_uncert(var=inptname, name=comp, **{param: values[sweepidx]})
            reportlist.append(ucalccopy.calculate(gum=gum, mc=mc))

        funcname = self.unccalc.model.outnames[self.unccalc.reverseparams['func']]
        self.out = SweepOutputReverse(reportlist, self.sweeplist, varname=self.unccalc.reverseparams['solvefor'], funcname=funcname)
        return self.out

    def get_config(self):
        ''' Get configuration dictionary '''
        d = super().get_config()
        d['mode'] = 'reversesweep'
        return d

    @classmethod
    def from_config(cls, config):
        unccalc = reverse.UncertReverse.from_config(config)
        newsweep = cls(unccalc)

        sweeps = config.get('sweeps', [])
        for sweep in sweeps:
            var = sweep['var']
            comp = sweep.get('comp', None)
            param = sweep.get('param', None)
            units = sweep.get('units', None)
            values = sweep['values']
            if var == 'corr':
                newsweep.add_sweep_corr(sweep.get('var1', None), sweep.get('var2', None), values)
            elif comp == 'nom':
                newsweep.add_sweep_nom(var, values, units=units)
            elif param == 'df':
                newsweep.add_sweep_df(var, values, comp)
            else:
                newsweep.add_sweep_unc(var, values, comp, param, units=units)
        return newsweep


class SweepOutput(output.Output):
    ''' This class holds the output of a multi-point sweep uncertainty calculation.

        Parameters
        ----------
        outputlist: list
            Individual UncertOutput objects in this sweep
        sweeplist: list
            List of sweep parameters
    '''
    def __init__(self, outputlist, sweeplist):
        self.outputlist = outputlist
        self.outnames = self.outputlist[0].names

        # Generate column headers for input values that are changing
        self.inpthdr = []
        for sweepparams in sweeplist:
            inptname = sweepparams.get('var', None)
            comp = sweepparams.get('comp', 'nom')
            param = sweepparams.get('param', None)
            if comp == 'nom':
                r = report.Math(inptname)
            elif param == 'df':
                r = (report.Math(inptname), ' deg.f')
            elif param in ['unc', 'std']:
                r = report.Math(comp.replace('(', '_').replace(')', ''))
            else:
                r = (report.Math(comp.replace('(', '_').replace(')', '')), ', ', param)
            self.inpthdr.append(r)

        self.inpthdr_strs = []
        for hdr in self.inpthdr:
            r = report.Report()
            try:
                r.add(*hdr, end='')
            except TypeError:
                r.add(hdr, end='')
            self.inpthdr_strs.append(r.get_md(mathfmt='ascii'))

        self.inptvals = [v['values']*uparser.parse_unit(v.get('units', '')) for v in sweeplist]
        self.N = len(self.inptvals[0])

        try:
            self.outpvalsgum = [[r.gum.nom(i) for r in outputlist] for i in range(outputlist[0].nouts)]
            self.outpuncsgum = [[r.gum.uncert(i) for r in outputlist] for i in range(outputlist[0].nouts)]
            self.outpvalsgum = [ureg.Quantity(np.array([r.magnitude for r in self.outpvalsgum[i]]), self.outpvalsgum[i][0].units) for i in range(len(self.outpvalsgum))]
            self.outpuncsgum = [ureg.Quantity(np.array([r.magnitude for r in self.outpuncsgum[i]]), self.outpuncsgum[i][0].units) for i in range(len(self.outpvalsgum))]
        except AttributeError:
            self.outpvalsgum = None
            self.outpuncsgum = None

        try:
            self.outpvalsmc = [[r.mc.nom(i) for r in outputlist] for i in range(outputlist[0].nouts)]
            self.outpuncsmc = [[r.mc.uncert(i) for r in outputlist] for i in range(outputlist[0].nouts)]
            self.outpvalsmc = [ureg.Quantity(np.array([r.magnitude for r in self.outpvalsmc[i]]), self.outpvalsmc[i][0].units) for i in range(len(self.outpvalsmc))]
            self.outpuncsmc = [ureg.Quantity(np.array([r.magnitude for r in self.outpuncsmc[i]]), self.outpuncsmc[i][0].units) for i in range(len(self.outpuncsmc))]
        except AttributeError:
            self.outpvalsmc = None
            self.outpuncsmc = None

    def get_dataset(self, name=None):
        ''' Get DataSet object from sweep output with the given name. If name is None, return a list
            of array names available.
        '''
        names = []
        for n in self.outnames:
            if self.outpvalsgum is not None:
                names.append('{} (GUM)'.format(n))
            if self.outpvalsmc is not None:
                names.append('{} (MC)'.format(n))

        if name is None:
            return names

        elif name in names:
            name, method = name.split(' ')
            funcidx = self.outnames.index(name)
            dset = self.to_array(gum=(method == '(GUM)'), funcidx=funcidx)
        else:
            raise ValueError('{} not found in output'.format(name))
        return dset

    def get_single_desc(self, idx):
        ''' Get description for a single index in the sweep '''
        slist = []
        for i in range(len(self.inpthdr)):
            valstrs = report.Number.number_array(self.inptvals[i])
            slist.append('{} = {}'.format(self.inpthdr_strs[i], valstrs[idx].string()))
        return '; '.join(slist).strip()

    def report(self, **kwargs):
        ''' Show results (table) of sweep calculation

            Keyword Arguments
            -----------------
            gum: bool
            mc: bool

            Other kwargs passed to report.Report.
        '''
        rpt = report.Report(**kwargs)
        if kwargs.get('gum', True) and self.outpvalsgum is not None:
            rpt.hdr('GUM Results', level=2)
            inptvalstrs = [report.Number.number_array(a) for a in self.inptvals]
            outvalstrs = [report.Number.number_array(a) for a in self.outpvalsgum]
            uncstrs = [[report.Number(a) for a in x] for x in self.outpuncsgum]
            rows = []
            for inpts, means, uncs in zip(list(zip(*inptvalstrs)), list(zip(*outvalstrs)), list(zip(*uncstrs))):
                rows.append(list(inpts) + [k for j in list(zip(means, uncs)) for k in j])   # i.e. transpose
            hdr = self.inpthdr.copy()
            for n in self.outnames:
                hdr.append(report.Math(n))
                hdr.append(report.Math('u_{}'.format(n)))
            rpt.table(rows, hdr=hdr)

        if kwargs.get('mc', True) and self.outpvalsmc is not None:
            rpt.hdr('Monte Carlo results', level=2)
            inptvalstrs = [report.Number.number_array(a) for a in self.inptvals]
            outvalstrs = [report.Number.number_array(a) for a in self.outpvalsmc]
            uncstrs = [[report.Number(a) for a in x] for x in self.outpuncsmc]
            rows = []
            for inpts, means, uncs in zip(list(zip(*inptvalstrs)), list(zip(*outvalstrs)), list(zip(*uncstrs))):
                rows.append(list(inpts) + [k for j in list(zip(means, uncs)) for k in j])   # i.e. transpose
            hdr = self.inpthdr.copy()
            for n in self.outnames:
                hdr.append(report.Math(n))
                hdr.append(report.Math('u_{}'.format(n)))
            rpt.table(rows, hdr=hdr)
        return rpt

    def expanded(self, cov=0.95, fidx=0, normal=False, shortest=False, method='gum'):
        ''' Get array of expanded uncertainties.

            Parameters
            ----------
            cov: float
                Coverage probability for uncertainties (0-1 range)
            fidx: int
                Index of function in calculator
            normal: bool
                For GUM uncertainties, use normal instead of t-distribution
            shortest: bool
                For Monte Carlo uncertainties, use shortest interval instead of
                symmetric interval

            Returns
            -------
            expanded: array
                Array of expanded uncertainties at each sweep point (GUM method)
            umin: array
                Array of bottom of coverage interval for each sweep point (MC method)
            umax: array
                Array of top of coverage interval for each sweep point (MC method)
            k: array
                Array of k-values for each sweep point.
        '''
        if method == 'gum':
            Expanded = namedtuple('Expanded', ['uncertainty', 'k'])
            expanded = [getattr(r, method).expanded(fidx=fidx, cov=cov, normal=normal) for r in self.outputlist]
            # Last index is always 0 because there's only one parameter in UncertCalc Output.
            uncert = np.array([x[0].magnitude for x in expanded]) * expanded[0][0].units
            k = np.array([x[1] for x in expanded])
            return Expanded(uncert, k)
        elif method == 'mc':
            Expanded = namedtuple('Expanded', ['minimum', 'maximum', 'k'])
            expanded = [getattr(r, method).expanded(fidx=fidx, cov=cov, shortest=shortest) for r in self.outputlist]
            umin = [x[0].magnitude for x in expanded] * expanded[0][0].units
            umax = [x[1].magnitude for x in expanded] * expanded[0][1].units
            k = [x[2] for x in expanded]
            return Expanded(umin, umax, k)

    def report_expanded(self, cov=0.95, fidx=0, normal=False, shortest=False, **kwargs):
        ''' Report table of expanded uncertainties

            Parameters
            ----------
            cov: float
                Coverage probability for uncertainties (0-1 range)
            fidx: int
                Index of function in calculator
            normal: bool
                For GUM uncertainties, use normal instead of t-distribution
            shortest: bool
                For Monte Carlo uncertainties, use shortest interval instead of
                symmetric interval

            Keyword Arguments
            -----------------
            Passed to report.Report
        '''
        rpt = report.Report(**kwargs)
        inptvalstrs = [report.Number.number_array(a) for a in self.inptvals]
        if kwargs.get('gum', True) and self.outpvalsgum is not None:
            uncvals, kvals = self.expanded(cov=cov, fidx=fidx, normal=normal, method='gum')
            hdr = self.inpthdr + ['Expanded Uncertainty', 'k']
            rows = []
            for inpts, unc, k in zip(list(zip(*inptvalstrs)), uncvals, kvals):
                rows.append(list(inpts) + [report.Number(unc)] + [report.Number(k, n=2)])
            rpt.hdr('GUM', level=3)
            rpt.table(rows, hdr)

        if kwargs.get('mc', True) and self.outpvalsmc is not None:
            umins, umaxs, kvals = self.expanded(cov=cov, fidx=fidx, shortest=shortest, method='mc')
            hdr = self.inpthdr + ['Min', 'Max', 'k']
            rows = []
            for inpts, umin, umax, k in zip(list(zip(*inptvalstrs)), umins, umaxs, kvals):
                rows.append(list(inpts) + [report.Number(umin)] + [report.Number(umax)] + [report.Number(k, n=2)])
            rpt.hdr('Monte Carlo', level=3)
            rpt.table(rows, hdr)
        return rpt

    def plot(self, plot=None, inptidx=0, funcidx=0, uy='errorbar', expanded=False, cov=.95, gum=True, mc=True):
        ''' Show plot of output value vs. input[inptidx] (for ONE function)

            Parameters
            ----------
            plot: matplotlib figure or axis
                If omitted, a new axis will be created. If both gum and mc are True,
                two axes in figure will be created.
            inptidx: int
                Index of input variable to use as x axis
            funcidx: int
                Index of function in calculator to plot
            uy: string
                'errorbar' for errorbar plot, None for no uncertainties, or
                linestyle (e.g. ':', '--', etc) for line uncertainties.
            expanded: bool
                Show uncertainty lines as expanded values
            cov: float
                Coverage probability for expanded uncertainties (0-1 range)
            gum: bool
                Plot GUM results
            mc: bool
                Plot MC results
        '''
        fig, ax = plotting.initplot(plot)
        fig.clf()
        if gum and mc:
            axgum = fig.add_subplot(1, 2, 1)
            axmc = fig.add_subplot(1, 2, 2)
        else:
            ax = fig.add_subplot(1, 1, 1)
            axgum = axmc = ax

        def doplot(ax, xvals, yvals, uyvals, label, xunits=None, yunits=None):
            ''' Function for plotting '''
            if uy == 'errorbar':
                ax.errorbar(xvals, yvals, yerr=uyvals, marker='o', label=label)
            elif uy is not None:
                ax.plot(xvals, yvals, ls='-', marker='o', label=label)
                if uyvals.ndim > 1:
                    p, = ax.plot(xvals, yvals - uyvals[0, :], ls=':')
                    ax.plot(xvals, yvals + uyvals[1, :], ls=':', color=p.get_color())
                else:
                    p, = ax.plot(xvals, yvals+uyvals, ls=':')
                    ax.plot(xvals, yvals-uyvals, ls=':', color=p.get_color())
            else:
                ax.plot(xvals, yvals, ls=uy, marker='o', label=label)
            ax.set_xlabel(self.inpthdr_strs[inptidx])
            ax.set_ylabel(self.outnames[funcidx])
            if xunits:
                ax.set_xlabel(ax.get_xlabel() + report.Unit(xunits).latex(bracket=True))
            if yunits:
                ax.set_ylabel(ax.get_ylabel() + report.Unit(yunits).latex(bracket=True))

        xvals = self.inptvals[inptidx].magnitude
        xunits = self.inptvals[inptidx].units
        if gum:
            yvals = self.outpvalsgum[funcidx].magnitude
            yunits = self.outpvalsgum[funcidx].units
            uyvals = self.outpuncsgum[funcidx].magnitude
            if expanded:
                uyvals, _ = self.expanded(cov=cov, fidx=funcidx, method='gum')
            doplot(axgum, xvals, yvals, uyvals, label='GUM', xunits=xunits, yunits=yunits)

        if mc:
            yvals = self.outpvalsmc[funcidx].magnitude
            yunits = self.outpvalsmc[funcidx].units
            uyvals = self.outpuncsmc[funcidx].magnitude
            if expanded:
                umin, umax, _ = self.expanded(cov=cov, fidx=funcidx, method='mc')  # First two cols define errorbars
                uyvals = abs(umax.magnitude - umin.magnitude)
            doplot(axmc, xvals, yvals, uyvals, label='Monte Carlo', xunits=xunits, yunits=yunits)

        if gum and mc and axmc is not axgum:
            axgum.set_title('GUM')
            axmc.set_title('Monte Carlo')
        elif gum and mc:
            axgum.legend(loc='best')

    def report_summary(self, **kwargs):
        ''' Report summary, including table AND plot '''
        rpt = report.Report(**kwargs)
        with plt.style.context(plotting.plotstyle):
            rpt.hdr('Sweep Results', level=2)
            rpt.append(self.report(**kwargs))
            for i in range(len(self.outnames)):
                fig = plt.figure()
                self.plot(plot=fig, inptidx=0, funcidx=i)
                rpt.plot(fig)
                plt.close(fig)
        return rpt

    def report_all(self, **kwargs):
        ''' Report full output '''
        r = self.report_summary(**kwargs)
        r.hdr('Expanded Uncertainties', level=3)
        r.append(self.report_expanded(**kwargs))
        return r

    def get_rptsingle(self, idx=0):
        ''' Get output object from single run in sweep '''
        return self.outputlist[idx]

    def to_array(self, gum=True, funcidx=0):
        ''' Return DataSet object of swept data and uncertainties

            Parameters
            ----------
            gum: bool
                Use gum (True) or monte carlo (False) values
            funcidx: int
                Index of function in calculator as y values

            Returns
            -------
            dset: DataSet object
                DataSet containing mean and uncertainties of each sweep point
        '''
        xvals = [x.magnitude for x in self.inptvals]
        names = self.inpthdr_strs.copy()

        if gum:
            yvals = self.outpvalsgum[funcidx].magnitude
            uyvals = self.outpuncsgum[funcidx].magnitude
        else:
            yvals = self.outpvalsmc[funcidx].magnitude
            uyvals = self.outpuncsmc[funcidx].magnitude
        names.append(self.outnames[funcidx])
        names.append(f'u({self.outnames[funcidx]})')
        return dataset.DataSet(np.vstack((xvals, yvals, uyvals)), colnames=names)


class SweepOutputReverse(output.Output):
    ''' This class holds the output of multi-point sweep reverse uncertainty propagation.

        Parameters
        ----------
        outputlist: list
            Individual output objects in this sweep
        sweeplist: list
            List of sweep parameters
        varname: str
            Name of variable solved for
    '''
    def __init__(self, outputlist, sweeplist, varname, funcname):
        self.outputlist = outputlist
        self.varname = varname
        self.name = funcname

        # Generate column headers for input values that are changing
        self.inpthdr = []
        for sweepparams in sweeplist:
            inptname = sweepparams.get('var', None)
            comp = sweepparams.get('comp', 'nom')
            param = sweepparams.get('param', None)
            if comp == 'nom':
                self.inpthdr.append(report.Math(inptname))
            elif param == 'df':
                self.inpthdr.append((report.Math(inptname), ' deg.f'))
            elif param in ['unc', 'std']:
                self.inpthdr.append(report.Math(comp.replace('(', '_').replace(')', '')))
            else:
                self.inpthdr.append((report.Math(comp.replace('(', '_').replace(')', '')), '\n'+param))

        self.inpthdr_strs = []
        for hdr in self.inpthdr:
            r = report.Report()
            try:
                r.add(*hdr, end='')
            except TypeError:
                r.add(hdr, end='')
            self.inpthdr_strs.append(r)

        self.inptvals = [v['values']*uparser.parse_unit(v.get('units', '')) for v in sweeplist]
        self.N = len(self.inptvals[0])

        if self.outputlist[0].mcdata:
            self.mcoutvals = np.array([r.mcdata['i'].magnitude for r in self.outputlist]) * self.outputlist[0].mcdata['i'].units
            self.mcoutuncs = np.array([r.mcdata['u_i'].magnitude for r in self.outputlist]) * self.outputlist[0].mcdata['u_i'].units
        else:
            self.mcoutvals = self.mcoutuncs = None

        if self.outputlist[0].gumdata:
            self.gumoutvals = np.array([r.gumdata['i'].magnitude for r in self.outputlist]) * self.outputlist[0].gumdata['i'].units
            self.gumoutuncs = np.array([r.gumdata['u_i'].magnitude for r in self.outputlist]) * self.outputlist[0].gumdata['u_i'].units
        else:
            self.gumoutvals = self.gumoutuncs = None

    def get_dataset(self, name=None):
        ''' Get DataSet object from sweep output with the given name. If name is None, return a list
            of array names available.
        '''
        names = []
        if self.gumoutvals is not None:
            names.append('{} (GUM)'.format(self.name))
        if self.mcoutvals is not None:
            names.append('{} (MC)'.format(self.name))

        if name is None:
            return names

        elif name in names:
            name, method = name.split(' ')
            dset = self.to_array(gum=(method == '(GUM)'))
        else:
            raise ValueError('{} not found in output'.format(name))
        return dset

    def report(self, **kwargs):
        ''' Report table of results of reverse-sweep '''
        r = report.Report(**kwargs)
        inptvalstrs = [report.Number.number_array(a) for a in self.inptvals]
        varname = report.Math(self.varname)
        uvarname = report.Math(f'u_{self.varname}')
        if self.gumoutvals is not None:
            rows = []
            r.hdr('GUM', level=3)
            hdr = self.inpthdr + [varname, uvarname]
            for inpts, val, unc in zip(list(zip(*inptvalstrs)), self.gumoutvals, self.gumoutuncs):
                row = list(inpts) + [report.Number(val), report.Number(unc)]
                rows.append(row)
            r.table(rows, hdr)

        if self.mcoutvals is not None:
            rows = []
            r.hdr('Monte Carlo', level=3)
            hdr = self.inpthdr + [varname, uvarname]
            for inpts, val, unc in zip(list(zip(*inptvalstrs)), self.mcoutvals, self.mcoutuncs):
                row = list(inpts) + [report.Number(val), report.Number(unc)]
                rows.append(row)
            r.table(rows, hdr)
        return r

    def report_summary(self, **kwargs):
        f_req = self.outputlist[0].gumdata['f_required']
        uf_req = self.outputlist[0].gumdata['uf_required']
        fname = self.outputlist[0].gumdata['fname']
        eqn = sympy.Eq(fname, self.outputlist[0].gumdata['f'])
        r = report.Report(**kwargs)
        r.hdr('Reverse Sweep Results', level=2)
        r.sympy(eqn, end='\n\n')
        r.add('Target: ', report.Math.from_sympy(fname), ' = ', report.Number(f_req, matchto=uf_req),
              ' ± ', report.Number(uf_req), '\n\n')
        r.append(self.report(**kwargs))
        with plt.style.context(plotting.plotstyle):
            fig, ax = plt.subplots()
            self.plot(plot=ax)
            r.plot(fig)
            plt.close(fig)
        return r

    def plot(self, plot=None, xidx=0, GUM=True, MC=True):
        ''' Plot results of reverse-sweep.

            Parameters
            ----------
            plot: matplotlib axis or figure
                If omitted, a new axis will be created
            xidx: int
                Index of input sweep variable to use as x-axis
            GUM: bool
                Show GUM calculation result
            MC: bool
                Show MC calculation result
        '''
        fig, ax = plotting.initplot(plot)
        xunits = self.inptvals[xidx].units
        if GUM and self.gumoutuncs is not None:
            ax.plot(self.inptvals[xidx].magnitude, self.gumoutuncs.magnitude, marker='o', label='GUM')
            yunits = self.gumoutuncs[0].units
        if MC and self.mcoutuncs is not None:
            ax.plot(self.inptvals[xidx].magnitude, self.mcoutuncs.magnitude, marker='^', label='Monte Carlo')
            yunits = self.mcoutuncs[0].units

        xunitstr = report.Unit(xunits).latex(bracket=True)
        yunitstr = report.Unit(yunits).latex(bracket=True)
        ax.set_xlabel(self.inpthdr_strs[xidx].get_md(mathfmt='latex') + xunitstr)
        ax.set_ylabel('Required $u_{{{}}}$'.format(self.varname) + yunitstr)

        if GUM and MC and self.gumoutuncs is not None and self.mcoutuncs is not None:
            ax.legend(loc='best')

    def to_array(self, gum=True):
        ''' Return DataSet object of swept data and uncertainties

            Parameters
            ----------
            gum: bool
                Use GUM values (True) or MC values (False)

            Returns
            -------
            dset: DataSet object
                DataSet containing x, y, ux, and uy values
        '''
        xvals = [x.magnitude for x in self.inptvals]
        names = [r.get_md(mathfmt='ascii') for r in self.inpthdr_strs]

        if gum:
            yvals = self.gumoutvals.magnitude
            uyvals = self.gumoutuncs.magnitude
        else:
            yvals = self.mcoutvals.magnitude
            uyvals = self.mcoutuncs.magnitude
        names.extend([self.varname, f'u({self.varname})'])
        return dataset.DataSet(np.vstack((xvals, yvals, uyvals)), colnames=names)
