''' Reverse uncertainty propagation class '''

import sympy
from scipy import stats
import numpy as np
import logging

from . import uncertainty
from . import output

import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.style.use('bmh')


class UncertReverse(uncertainty.UncertCalc):
    ''' Reverse uncertainty propagation.

        Parameters
        ----------
        function: string, sympy, or callable, or list
            Function or list of functions to calculate. Each function may be a string,
            callable, or sympy expression.
        samples: int, optional
            Number of Monte-Carlo samples. Default to 1E6.
        seed: int, optional
            Random number seed. Seed will be randomized if None.
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
            \**args             Keyword arguments defining the distribution, passed to stats instance.
            =================   ===============================================

        solvefor: string
            Name of input variable to solve for
        targetnom: float
            Target nominal value for variable
        targetunc: float
            Target standard uncertainty for variable
        fidx: int
            Index of target function in system.

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
    def __init__(self, function=None, inputs=None, samples=1000000, seed=None, name='reverse', solvefor=None,
                 targetnom=None, targetunc=None, fidx=-1):
        super(UncertReverse, self).__init__(function, inputs, samples, seed, name)
        self.reverseparams = {}
        if solvefor is not None:
            self.set_reverse(solvefor, targetnom, targetunc, fidx)

    def set_reverse(self, solvefor, targetnom, targetunc, fidx=-1):
        ''' Set up reverse calculation parameters.

            Parameters
            ----------
            solvefor: string
                Name of input variable to solve for
            targetnom: float
                Target nominal value for variable
            targetunc: float
                Target standard uncertainty for variable
            fidx: int
                Index of target function in system.
        '''
        self.reverseparams = {'solvefor': solvefor,
                              'targetnom': targetnom,
                              'targetunc': targetunc,
                              'func': fidx}

    def calculate(self, **kwargs):
        ''' Calculate reverse uncertainty propagation.
            Parameters
            ----------
            gum: boolean
                Calculate using GUM method
            mc: boolean
                Calculate using Monte-Carlo method
        '''
        self.add_required_inputs()
        fidx = self.reverseparams.get('func', -1)
        var = self.reverseparams['solvefor']
        req_nom = self.reverseparams['targetnom']
        req_uncert = self.reverseparams['targetunc']

        function = self.functions[fidx]  # Function object
        if function.ftype != 'sympy':
            raise ValueError('Reverse calculation requires sympy function')

        if var not in function.get_basenames():
            raise ValueError('Undefined solvefor variable {}'.format(var))

        # Calculate GUM symbolically then solve for uncertainty component
        self.add_required_inputs()
        symout = function.calc_SYM()
        fname = 'f' if function.name is None else function.name

        # Solve function for variable of interest
        var_f = sympy.solve(sympy.Eq(sympy.Symbol(fname), function.get_basefunc()), sympy.Symbol(var))[0]
        ucombined = symout['uncertainty']  # Symbolic expression for combined uncertainty

        u_ivar = sympy.Symbol('u_'+var)  # Symbol for unknown uncertainty we're solving for
        u_iexpr = sympy.solve(sympy.Eq(symout['uc_symbol'], ucombined), u_ivar)[1]  # Solve for u_i, keep positive solution
        u_iexpr = u_iexpr.subs({var: var_f})  # Replace var with var_f
        u_ival = u_iexpr.subs(function.get_basemeans())
        i_val = var_f.subs(function.get_basemeans()).subs({fname: req_nom})

        # Rename values in baseuncerts
        uncertvals = function.get_baseuncerts()
        for k, v in uncertvals.copy().items():
            uncertvals.pop(k)
            uncertvals['u_'+k] = v

        reverse_GUM = None
        if kwargs.get('gum', True):
            # Plug everything in
            u_ival = u_ival.subs(uncertvals)
            u_ival = u_ival.subs({fname: req_nom, symout['uc_symbol']: req_uncert})
            try:
                u_ival = float(u_ival)
                i_val = abs(float(i_val))
            except TypeError:
                logging.warning('No real solution for reverse calculation.')
                u_ival = None

            reverse_GUM = {'u_iexpr': u_iexpr,
                           'u_i': u_ival,
                           'u_iname': u_ivar,
                           'u_c': ucombined,
                           'i': i_val,
                           'f': function.get_basefunc(),
                           'fname': sympy.Symbol(fname),
                           'ucname': symout['uc_symbol'],
                           'f_required': req_nom,
                           'uf_required': req_uncert,
                           }

        reverse_MC = None
        if kwargs.get('mc', True):
            # Use Monte Carlo Method - Must add correlation between f and input variables
            ucalc = uncertainty.UncertCalc(var + '=' + str(var_f), samples=self.samples)  # Set up UncertCalc with reversed expression
            for origvar in function.get_basevars():
                if origvar.name == var: continue
                origunc = origvar.stdunc()
                if np.isfinite(origunc) and origunc != 0:
                    ucalc.set_input(origvar.name, nom=origvar.nom, std=origunc)
                else:
                    ucalc.set_input(origvar.name, nom=origvar.nom)  # Doesn't like std=0
            ucalc.set_input(fname, nom=req_nom, std=req_uncert)

            # Correlate variables: see GUM C.3.6 NOTE 3 - Estimate correlation from partials
            for vname, part in zip(symout['var_symbols'], symout['partials']):
                if str(vname) == var: continue
                ci = float(part.subs({var: var_f}).subs(function.get_basemeans()).subs(uncertvals).subs({fname: req_nom}))
                corr = ucalc.get_input(str(vname)).stdunc() / ucalc.get_input(fname).stdunc() * ci
                if np.isfinite(corr):
                    ucalc.correlate_vars(str(vname), fname, corr)

            # Include existing correlations between inputs
            if self._corr is not None:
                for v1, v2, corr in self.get_corr_list():
                    if v1 == var or v2 == var: continue
                    ucalc.correlate_vars(v1, v2, corr)

            mcout = ucalc.calcMC()
            reverse_MC = {'u_i': mcout.foutputs[0].mc.uncert[0],
                          'i': mcout.foutputs[0].mc.mean[0],
                          'f_required': req_nom,
                          'uf_required': req_uncert,
                          'rev_ucalc': ucalc
                          }
            if np.count_nonzero(np.isfinite(mcout.foutputs[0].mc.samples)) < self.samples * .95:
                # less than 95% of trials resulted in real number, consider this a no-solution
                reverse_MC['u_i'] = None
                reverse_MC['i'] = None

        self.out = ReverseOutput(reverse_GUM, reverse_MC)
        return self.out

    @classmethod
    def from_config(cls, config):
        ''' Load configuration from dictionary '''
        newrev = uncertainty.UncertCalc.from_config(config)
        newrev.__class__ = cls   # Convert UncertCalc into UncertReverse!
        newrev.reverseparams = config.get('reverse', {})
        return newrev

    def get_config(self):
        ''' Get configuration dictionary for calculation '''
        d = super(UncertReverse, self).get_config()
        d['mode'] = 'reverse'
        d['reverse'] = self.reverseparams
        return d


class ReverseOutput(output.Output):
    ''' Class for reporting results of reverse uncertainty calculation

        Parameters
        ----------
        gumdata: dict
            Dictionary of values from GUM calculation
        mcdata: dict
            Dictionary of values from MC calculation
    '''
    def __init__(self, gumdata, mcdata):
        self.name = None
        self.gumdata = gumdata
        self.mcdata = mcdata

    def report(self, gum=True, mc=True, **kwargs):
        ''' Report summary of reverse calculation.

            Parameters
            gum: bool
                Show results of GUM calculation
            mc: bool
                Show results of Monte Carlo calculation

            Keyword Arguments
            -----------------
            See NumFormatter()
        '''
        s = output.MDstring()
        if self.gumdata and gum:
            ui = self.gumdata['u_i']
            i = self.gumdata['i']

            s += '## GUM reverse uncertainty\n\n'
            s += output.sympyeqn(sympy.Eq(self.gumdata['fname'], self.gumdata['f']))
            s += '\n\n Combined uncertainty:\n\n'
            s += output.sympyeqn(sympy.Eq(self.gumdata['ucname'], self.gumdata['u_c']))
            s += '\n\nsolved for uncertainty of input:\n\n'
            s += output.sympyeqn(sympy.Eq(self.gumdata['u_iname'], self.gumdata['u_iexpr']))
            s += u'\n\n For output value of {} {} {} (k=1),\n'.format(output.formatter.f(self.gumdata['f_required'], matchto=self.gumdata['uf_required'], **kwargs), output.UPLUSMINUS, output.formatter.f(self.gumdata['uf_required'], **kwargs))
            if i is None or ui is None:
                s += u'No real solution found\n\n'
            else:
                s += u'required input value is {} {} {} (k=1).\n\n'.format(output.formatter.f(i, matchto=ui, **kwargs), output.UPLUSMINUS, output.formatter.f(ui, **kwargs))

        if self.mcdata and mc:
            if self.gumdata and gum:
                s += '\n\n----\n\n'
            s += '## Monte Carlo reverse uncertainty\n\n'
            i = self.mcdata['i']
            ui = self.mcdata['u_i']
            if i is None or ui is None:
                s += u'No real solution found\n\n'
            else:
                s += u'For output value of {} {} {} (k=1), '.format(output.formatter.f(self.mcdata['f_required'], matchto=self.mcdata['uf_required'], **kwargs), output.UPLUSMINUS, output.formatter.f(self.mcdata['uf_required'], **kwargs))
                s += u'required input value is: {} {} {} (k=1).\n\n'.format(output.formatter.f(i, matchto=ui, **kwargs), output.UPLUSMINUS, output.formatter.f(ui, **kwargs))

        return s

    def report_summary(self, **kwargs):
        r = self.report(**kwargs)
        with mpl.style.context(output.mplcontext):
            plt.ioff()
            fig = plt.figure()
            self.plot_pdf(fig)
            r.add_fig(fig)
        return r

    def plot_pdf(self, fig=None, **kwargs):
        ''' Plot PDF/Histogram of the reverse calculation '''
        if fig is None:
            fig = plt.gcf()

        mccolor = kwargs.pop('mccolor', 'C0')
        gumcolor = kwargs.pop('gumcolor', 'C1')

        fig.clf()
        ax = fig.add_subplot(1, 1, 1)
        if self.gumdata and self.gumdata['u_i'] is not None:
            mean = self.gumdata['i']
            std = self.gumdata['u_i']
            xx = np.linspace(mean-4*std, mean+4*std, num=200)
            yy = stats.norm.pdf(xx, loc=mean, scale=std)
            kwargs['label'] = 'GUM PDF'
            kwargs['color'] = gumcolor
            ax.plot(xx, yy, **kwargs)

        if self.mcdata:
            kwargs['color'] = mccolor
            kwargs['label'] = 'Monte Carlo'
            self.mcdata['rev_ucalc'].out.foutputs[0].plot_pdf(ax=ax, **kwargs)
