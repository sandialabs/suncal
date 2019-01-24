''' Methods for running multiple uncertainty calculations sweeping over input arrays.

    Use UncertSweep to set up a sweep on an existing UncertCalc object.

    The SweepOutput* classes hold the output of an UncertSweep calculation.
'''
import numpy as np
import sympy
import yaml

from . import uarray
from . import reverse
from . import uncertainty
from . import output

try:
    import matplotlib as mpl
    import matplotlib.pyplot as plt
except ImportError:
    pass
else:
    mpl.style.use('bmh')


class UncertSweep(object):
    ''' Class for running multiple uncertainty calculations over a range of input values

        Parameters
        ----------
        unccalc: Uncertainty Propagation Object
            Uncertianty Calculator to sweep
    '''
    def __init__(self, unccalc, name='sweep'):
        self.unccalc = unccalc
        self.sweeplist = []
        self.name = name
        self.out = None

    def clear_sweeps(self):
        ''' Remove all sweeps from the sweeplist '''
        self.sweeplist = []

    def add_sweep_nom(self, varname, values):
        ''' Add sweep of nominal value.

            Parameters
            ----------
            varname: string
                Name of variable to sweep nominal value
            values: array
                Values for sweep
        '''
        d = {'var': varname, 'comp': 'nom', 'values': values}
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

    def add_sweep_unc(self, varname, values, comp=None, param='std'):
        ''' Add sweep of uncertainty component parameter.

            Parameters
            ----------
            varname: string
                Name of variable to sweep uncertainty component
            values: array
                Values for sweep
            comp: string
                Name of uncertainty component. Defaults to u(varname).
            param: string
                Name of uncertainty parameter in distribution.
            index: int
                Index of sweep item in sweeplist. Use None to add a new item
                to the list.
        '''
        if comp is None:
            comp = 'u({})'.format(varname)
        d = {'var': varname, 'comp': comp, 'param': param, 'values': values}
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
            for sweepparams in self.sweeplist:
                inptname = sweepparams.get('var', None)
                comp = sweepparams.get('comp', 'nom')
                param = sweepparams.get('param', None)
                values = sweepparams.get('values', [])
                # Make a copy (using config dictionary) so we don't destroy the original uncertcalc object and overwrite inputs
                # Note: dont use deepcopy() or we'll be copying output data too
                ucalccopy = uncertainty.UncertCalc.from_config(self.unccalc.get_config())
                inptvar = ucalccopy.get_input(inptname)
                if inptname == 'corr':
                    ucalccopy.correlate_vars(sweepparams['var1'], sweepparams['var2'], values[sweepidx])
                elif comp == 'nom':
                    inptvar.set_nom(values[sweepidx])
                elif param == 'df':
                    inptvar.get_comp(comp).degf = values[sweepidx]
                else:
                    ucalccopy.set_uncert(var=inptname, name=comp, **{param: values[sweepidx]})
            reportlist.append(ucalccopy.calculate(gum=gum, mc=mc))
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
        except yaml.scanner.ScannerError:
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
            values = sweep['values']
            if var == 'corr':
                newsweep.add_sweep_corr(comp, param, values)
            elif comp == 'nom':
                newsweep.add_sweep_nom(var, values)
            elif param == 'df':
                newsweep.add_sweep_df(var, values, comp)
            else:
                newsweep.add_sweep_unc(var, values, comp, param)
        return newsweep

    def get_config(self):
        ''' Get configuration dictionary '''
        d = self.unccalc.get_config()
        d['mode'] = 'sweep'
        sweeps = []
        for sweep in self.sweeplist:
            var = sweep.get('var', None)
            comp = sweep.get('comp', 'nom')
            param = sweep.get('param', None)
            values = list(sweep.get('values', []))
            sweeps.append({'var': var, 'comp': comp, 'param': param, 'values': values})
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
        super(UncertSweepReverse, self).__init__(unccalc, name=name)   # Just override the default name

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
                inptvar = ucalccopy.get_input(inptname)
                if inptname == 'corr':
                    ucalccopy.correlate_vars(sweepparams['var1'], sweepparams['var2'], values[sweepidx])
                elif comp == 'nom':
                    inptvar.set_nom(values[sweepidx])
                elif param == 'df':
                    inptvar.degf = values[sweepidx]
                else:
                    ucalccopy.set_uncert(var=inptname, name=comp, **{param: values[sweepidx]})
            reportlist.append(ucalccopy.calculate())

        funcname = self.unccalc.functions[self.unccalc.reverseparams['func']].name
        self.out = SweepOutputReverse(reportlist, self.sweeplist, varname=self.unccalc.reverseparams['solvefor'], funcname=funcname)
        return self.out

    def get_config(self):
        ''' Get configuration dictionary '''
        d = super(UncertSweepReverse, self).get_config()
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
            values = sweep['values']
            if var == 'corr':
                newsweep.add_sweep_corr(comp, param, values)
            elif comp == 'nom':
                newsweep.add_sweep_nom(var, values)
            elif param == 'df':
                newsweep.add_sweep_df(var, values, comp)
            else:
                newsweep.add_sweep_unc(var, values, comp, param)
        return newsweep


class SweepOutput(output.Output):
    ''' This class holds the output of a multi-point sweep uncertainty calculation.

        Parameters
        ----------
        outputlist: list
            Individual CalcOutput objects in this sweep
        sweeplist: list
            List of sweep parameters
    '''
    def __init__(self, outputlist, sweeplist):
        self.outputlist = outputlist
        self.funcnames = self.outputlist[0].get_funcnames()

        # Generate column headers for input values that are changing
        self.inpthdr = []
        for sweepparams in sweeplist:
            inptname = sweepparams.get('var', None)
            comp = sweepparams.get('comp', 'nom')
            param = sweepparams.get('param', None)
            if comp == 'nom':
                self.inpthdr.append(inptname)
            elif comp == 'df':
                self.inpthdr.append(inptname + ' deg.f')
            elif param in ['unc', 'std']:
                self.inpthdr.append(comp)
            else:
                self.inpthdr.append(comp + ', ' + param)
        self.inptvals = np.vstack([v['values'] for v in sweeplist]).transpose()
        self.N = self.inptvals.shape[0]

        self.outnames = [f.name for f in outputlist[0].foutputs]
        try:
            self.outpvalsgum = np.vstack([[r.get_output(fidx=i, method='gum').mean[0] for i in range(len(outputlist[0].foutputs))] for r in outputlist])
            self.outpuncsgum = np.vstack([[r.get_output(fidx=i, method='gum').uncert[0] for i in range(len(outputlist[0].foutputs))] for r in outputlist])
        except AttributeError:
            self.outpvalsgum = None
            self.outpuncsgum = None

        try:
            self.outpvalsmc = np.vstack([[r.get_output(fidx=i, method='mc').mean[0] for i in range(len(outputlist[0].foutputs))] for r in outputlist])
            self.outpuncsmc = np.vstack([[r.get_output(fidx=i, method='mc').uncert[0] for i in range(len(outputlist[0].foutputs))] for r in outputlist])
        except AttributeError:
            self.outpvalsmc = None
            self.outpuncsmc = None

    def get_array(self, name=None):
        ''' Get Array object from sweep output with the given name. If name==None, return a list
            of array names available.

            See Also: to_array for getting array by function index
        '''
        names = []
        for n in self.funcnames:
            if self.outpvalsgum is not None:
                names.append('{} (GUM)'.format(n))
            if self.outpvalsmc is not None:
                names.append('{} (MC)'.format(n))

        if name is None:
            return names

        elif name in names:
            name, method = name.split(' ')
            funcidx = self.funcnames.index(name)
            # NOTE: always returning FIRST sweep column as x value here
            arr = self.to_array(gum=(method == '(GUM)'), funcidx=funcidx, inptidx=0)
        else:
            raise ValueError('{} not found in output'.format(name))
        return arr

    def get_single_desc(self, idx):
        ''' Get description for a single index in the sweep '''
        slist = []
        for i in range(len(self.inpthdr)):
            valstrs = output.formatter.f_array(self.inptvals[:, i])
            slist.append('{} = {}'.format(self.inpthdr[i], valstrs[idx]))
        return '; '.join(slist)

    def report(self, **kwargs):
        ''' Show results (table) of sweep calculation

            Keyword Arguments
            -----------------
            gum: bool
            mc: bool

            Passed to number formatter. E.g. 'n=3' for 3 significant digits.
        '''
        rpt = output.MDstring()
        if kwargs.get('gum', True) and self.outpvalsgum is not None:
            rpt += '## GUM Results\n\n'
            rows = []
            inptvalstrs = np.apply_along_axis(output.formatter.f_array, 0, self.inptvals)
            outvalstrs = np.apply_along_axis(output.formatter.f_array, 0, self.outpvalsgum)
            for inpts, means, uncs in zip(inptvalstrs, outvalstrs, self.outpuncsgum):
                row = inpts.tolist()
                for mean, unc in zip(means, uncs):
                    row.extend([mean, output.formatter.f(unc, **kwargs)])
                rows.append(row)

            hdr = self.inpthdr.copy()
            for n in self.outnames:
                hdr.append(n)
                hdr.append('u({})'.format(n))
            rpt += output.md_table(rows, hdr=hdr)

        if kwargs.get('mc', True) and self.outpvalsmc is not None:
            rpt += '## Monte Carlo results\n\n'
            rows = []
            outvalstrmc = np.apply_along_axis(output.formatter.f_array, 0, self.outpvalsmc)
            for inpts, means, uncs in zip(inptvalstrs, outvalstrmc, self.outpuncsmc):
                row = inpts.tolist()
                for mean, unc in zip(means, uncs):
                    row.extend([mean, output.formatter.f(unc, **kwargs)])
                rows.append(row)

            hdr = self.inpthdr.copy()
            for n in self.outnames:
                hdr.append(n)
                hdr.append('u({})'.format(n))
            rpt += output.md_table(rows, hdr=hdr)
        return rpt

    def expanded(self, cov=0.95, fidx=0, normal=False, shortest=False, method='gum'):
        ''' Get array of expanded uncertainties.

            Parameters
            ----------
            cov: float
                Coverage interval for uncertainties (0-1 range)
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
                For GUM uncertainties, array is Nx2. First column is expanded uncertainty,
                second column is k-value. For MC uncertainties, array is Nx3, with columns
                of minimum, maximum, and k-value for range.
        '''
        if method == 'gum':
            expanded = [r.get_output(fidx=fidx, method=method).expanded(cov=cov, normal=normal) for r in self.outputlist]
            expanded = np.array([np.array([x[0], y]) for x, y in expanded])
        elif method == 'mc':
            expanded = [r.get_output(fidx=fidx, method=method).expanded(cov=cov, shortest=shortest) for r in self.outputlist]
            expanded = np.asarray(expanded)[:, :, 0]
        return expanded

    def report_expanded(self, cov=0.95, fidx=0, normal=False, shortest=False, **kwargs):
        ''' Report table of expanded uncertainties

            Parameters
            ----------
            cov: float
                Coverage interval for uncertainties (0-1 range)
            fidx: int
                Index of function in calculator
            normal: bool
                For GUM uncertainties, use normal instead of t-distribution
            shortest: bool
                For Monte Carlo uncertainties, use shortest interval instead of
                symmetric interval

            Keyword Arguments
            -----------------
            Passed to number formatter. E.g. 'n=3' for 3 significant digits.
        '''
        rpt = output.MDstring()
        if kwargs.get('gum', True) and self.outpvalsgum is not None:
            expanded = self.expanded(cov=cov, fidx=fidx, normal=normal, method='gum')  # expanded returns uncertainty, k
            hdr = self.inpthdr + ['Expanded Uncertainty', 'k']
            rows = []
            for inpt, (unc, k) in zip(self.inptvals, expanded):
                row = [output.formatter.f(i, **kwargs) for i in inpt] + [output.formatter.f(unc, **kwargs), output.formatter.f(k, n=2)]
                rows.append(row)
            rpt += '### GUM\n\n'
            rpt += output.md_table(rows, hdr)

        if kwargs.get('mc', True) and self.outpvalsmc is not None:
            expanded = self.expanded(cov=cov, fidx=fidx, shortest=shortest, method='mc')
            hdr = self.inpthdr + ['Min', 'Max', 'k']
            rows = []
            for inpt, (mn, mx, k) in zip(self.inptvals, expanded):
                row = [output.formatter.f(i, **kwargs) for i in inpt] + [output.formatter.f(mn, **kwargs), output.formatter.f(mx, **kwargs), output.formatter.f(k, n=2)]
                rows.append(row)
            rpt += '### Monte Carlo\n\n'
            rpt += output.md_table(rows, hdr)
        return rpt

    def plot(self, ax=None, inptidx=0, funcidx=0, uy='errorbar', expanded=False, cov=.95, gum=True, mc=True):
        ''' Show plot of output value vs. input[inptidx] (for ONE function)

            Parameters
            ----------
            ax: matplotlib axis
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
                Coverage interval for expanded uncertainties (0-1 range)
            gum: bool
                Plot GUM results
            mc: bool
                Plot MC results
        '''
        try:
            axgum, axmc = ax
        except TypeError:
            if ax is None:
                if gum and mc:
                    fig, (axgum, axmc) = plt.subplots(ncols=2)
                else:
                    fig, ax = plt.subplots()
                    axgum = ax
                    axmc = ax
            else:
                axgum = ax
                axmc = ax

        def doplot(ax, xvals, yvals, uyvals, label):
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
            ax.set_xlabel(self.inpthdr[inptidx])
            ax.set_ylabel(self.outnames[funcidx])

        xvals = self.inptvals[:, inptidx]
        if gum:
            yvals = self.outpvalsgum[:, funcidx]
            uyvals = self.outpuncsgum[:, funcidx]
            if expanded:
                uyvals = self.expanded(cov=cov, fidx=funcidx, method='gum')[:, 0]  # First col is uncert.
            doplot(axgum, xvals, yvals, uyvals, label='GUM')

        if mc:
            yvals = self.outpvalsmc[:, funcidx]
            uyvals = self.outpuncsmc[:, funcidx]
            if expanded:
                uyvals = self.expanded(cov=cov, fidx=funcidx, method='mc')  # First two cols define errorbars
                uyvals = np.array([abs(yvals - uyvals[:, 0]), abs(yvals - uyvals[:, 1])])
            doplot(axmc, xvals, yvals, uyvals, label='Monte Carlo')

        if gum and mc and axmc is not axgum:
            axgum.set_title('GUM')
            axmc.set_title('Monte Carlo')
        elif gum and mc:
            axgum.legend(loc='best')

    def report_summary(self, **kwargs):
        ''' Report summary, including table AND plot '''
        with mpl.style.context(output.mplcontext):
            r = output.MDstring('## Sweep Results\n\n')
            r += self.report(**kwargs)
            with mpl.style.context(output.mplcontext):
                plt.ioff()
                fig, axs = plt.subplots(ncols=2)
                self.plot(ax=axs, inptidx=0, funcidx=0)
                r.add_fig(fig)
        return r

    def report_all(self, **kwargs):
        ''' Report full output '''
        r = self.report_summary(**kwargs)
        r += '### Expanded Uncertainties\n\n'
        r += self.report_expanded(**kwargs)
        return r

    def get_rptsingle(self, idx=0):
        ''' Get output object from single run in sweep '''
        return self.outputlist[idx]

    def to_array(self, gum=True, inptidx=0, funcidx=0):
        ''' Return Array object of swept data and uncertainties

            Parameters
            ----------
            gum: bool
                Use gum (True) or monte carlo (False) values
            inptidx: int
                Index of input variable to use as x values
            funcidx: int
                Index of function in calculator as y values

            Returns
            -------
            arr: Array object
                Array of x, y, and uy values

            See Also: get_array for getting an array by name (used in GUI)
        '''
        xvals = self.inptvals[:, inptidx]
        if gum:
            yvals = self.outpvalsgum[:, funcidx]
            uyvals = self.outpuncsgum[:, funcidx]
        else:
            yvals = self.outpvalsmc[:, funcidx]
            uyvals = self.outpuncsmc[:, funcidx]
        return uarray.Array(xvals, yvals, uy=uyvals)


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
        self.outnames = [out.name for out in outputlist]
        self.varname = varname
        self.name = funcname

        # Generate column headers for input values that are changing
        self.inpthdr = []
        for sweepparams in sweeplist:
            inptname = sweepparams.get('var', None)
            comp = sweepparams.get('comp', 'nom')
            param = sweepparams.get('param', None)
            if comp == 'nom':
                self.inpthdr.append(inptname)
            elif comp == 'df':
                self.inpthdr.append(inptname + ' deg.f')
            elif param in ['unc', 'std']:
                self.inpthdr.append(comp)
            else:
                self.inpthdr.append(comp + '\n' + param)
        self.inptvals = np.vstack([v['values'] for v in sweeplist]).transpose()
        if self.outputlist[0].mcdata:
            self.mcoutvals = np.array([r.mcdata['i'] for r in self.outputlist])
            self.mcoutuncs = np.array([r.mcdata['u_i'] for r in self.outputlist])
        else:
            self.mcoutvals = self.mcoutuncs = None

        if self.outputlist[0].gumdata:
            self.gumoutvals = np.array([r.gumdata['i'] for r in self.outputlist])
            self.gumoutuncs = np.array([r.gumdata['u_i'] for r in self.outputlist])
        else:
            self.gumoutvals = self.gumoutuncs = None

    def get_array(self, name=None):
        ''' Get array from sweep output with the given name. If name==None, return a list
            of array names available.

            Returned array is 3-columns. X column (currently) is np.arange(len(y)), Y column
            is sweep value, and UY column standard uncertainty.
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
            arr = self.to_array(gum=(method == '(GUM)'), inptidx=0)  # NOTE: always returning first sweep column as x
        else:
            raise ValueError('{} not found in output'.format(name))
        return arr

    def report(self, **kwargs):
        ''' Report table of results of reverse-sweep '''
        r = output.MDstring()
        inptvalstrs = np.apply_along_axis(output.formatter.f_array, 0, self.inptvals)
        rows = []
        if self.gumoutvals is not None and self.mcoutvals is not None:
            hdr = self.inpthdr + ['GUM Uncertainty (k=1)', 'MC Uncertainty (k=1)']
            for inpts, uncgum, uncmc in zip(inptvalstrs, self.gumoutuncs, self.mcoutuncs):
                row = inpts.tolist() + [output.formatter.f(uncgum, **kwargs), output.formatter.f(uncmc, **kwargs)]
                rows.append(row)
        elif self.gumoutvals is not None:
            hdr = self.inpthdr + ['Uncertainty']
            for inpts, unc in zip(inptvalstrs, self.gumoutuncs):
                row = inpts.tolist() + [output.formatter.f(unc, **kwargs)]
                rows.append(row)
        elif self.mcoutvals is not None:
            hdr = self.inpthdr + ['Uncertainty']
            for inpts, unc in zip(inptvalstrs, self.mcoutuncs):
                row = inpts.tolist() + [output.formatter.f(unc, **kwargs)]
                rows.append(row)
        r += output.md_table(rows, hdr)
        return r

    def report_summary(self, **kwargs):
        f_req = self.outputlist[0].gumdata['f_required']
        uf_req = self.outputlist[0].gumdata['uf_required']
        eqn = sympy.Eq(self.outputlist[0].gumdata['fname'], self.outputlist[0].gumdata['f'])
        r = output.MDstring('## Reverse Sweep Results\n\n')
        r += output.sympyeqn(eqn) + '\n\n'
        r += 'Target: {} = {} {} {}'.format(self.varname, output.formatter.f(f_req, matchto=uf_req, **kwargs), output.UPLUSMINUS, output.formatter.f(uf_req, **kwargs)) + '\n\n'
        r += self.report(**kwargs)
        with mpl.style.context(output.mplcontext):
            plt.ioff()
            fig, ax = plt.subplots()
            self.plot(ax=ax)
            r.add_fig(fig)
        return r

    def plot(self, ax=None, xidx=0, GUM=True, MC=True):
        ''' Plot results of reverse-sweep.

            Parameters
            ----------
            ax: matplotlib axis
                If omitted, a new axis will be created
            xidx: int
                Index of input sweep variable to use as x-axis
            GUM: bool
                Show GUM calculation result
            MC: bool
                Show MC calculation result
        '''
        if ax is None:
            fig, ax = plt.subplots()

        if GUM and self.gumoutuncs is not None:
            ax.plot(self.inptvals[:, xidx], self.gumoutuncs, marker='o', label='GUM')
        if MC and self.mcoutuncs is not None:
            ax.plot(self.inptvals[:, xidx], self.mcoutuncs, marker='^', label='Monte Carlo')

        ax.set_xlabel(self.inpthdr[xidx])
        ax.set_ylabel('Required $u_{}$'.format(self.varname))

        if GUM and MC and self.gumoutuncs is not None and self.mcoutuncs is not None:
            ax.legend(loc='best')

    def to_array(self, gum=True, inptidx=0):
        ''' Return Array object of swept data and uncertainties

            Parameters
            ----------
            gum: bool
                Use GUM values (True) or MC values (False)
            inptidx: int
                Index of input variable to use as x values
            funcidx: int
                Index of function in calculator as y values

            Returns
            -------
            arr: Array object
                Array of x, y, and uy values
        '''
        xvals = self.inptvals[:, inptidx]
        if gum:
            yvals = np.array(self.gumoutvals)
            uyvals = np.array(self.gumoutuncs)
        else:
            yvals = np.array(self.mcoutvals)
            uyvals = np.array(self.mcoutuncs)
        return uarray.Array(xvals, yvals, uy=uyvals)
