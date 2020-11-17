''' Reverse uncertainty propagation class '''

import sympy
from scipy import stats
import numpy as np
import logging

from . import uncertainty
from . import output
from . import report
from . import uparser
from . import plotting

import matplotlib.pyplot as plt


class UncertReverse(uncertainty.UncertCalc):
    r''' Reverse uncertainty propagation.

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
                 units=None, targetnom=None, targetunc=None, fidx=-1):
        super().__init__(function=function, inputs=inputs, samples=samples, seed=seed, name=name, units=units)
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

        model = self.model
        if not isinstance(model, uncertainty.ModelSympy):
            raise ValueError('Reverse calculation requires sympy function')

        if var not in self.inputs.names:
            raise ValueError('Undefined solvefor variable {}'.format(var))

        req_nom *= uparser.parse_unit(model.outunits[fidx])
        req_uncert *= uparser.parse_unit(model.outunits[fidx])
        solvefor_units = self.inputs.means()[var].units

        # Calculate GUM symbolically then solve for uncertainty component
        self.add_required_inputs()
        symout = model.GUMcovariance().symbolic  # uncerts, self.sympyexprs, degf, Uy, Ux, Cx
        fname = model.outnames[fidx]

        # Solve function for variable of interest
        var_f = sympy.solve(sympy.Eq(sympy.Symbol(fname), model.get_baseexprs()[fidx]), sympy.Symbol(var))[0]
        ucombined = symout.uncert[0]  # Symbolic expression for combined uncertainty

        # ucombined will have sigma/correlation values in it, likely 0 - sub them out
        corrvals = {k: v for k, v in self.inputs.corr_values().items() if v == 0}
        ucombined = ucombined.subs(corrvals).simplify()

        u_ivar = sympy.Symbol('u_'+var)  # Symbol for unknown uncertainty we're solving for
        u_ovar = sympy.Symbol('u_'+fname)

        try:
            u_iexpr = sympy.solve(sympy.Eq(u_ovar, ucombined), u_ivar)[1]  # Solve for u_i, keep positive solution
        except IndexError:
            # Will fail with no solution for model f = x due to sqrt(x**2) not simplifying.
            u_iexpr = u_ovar
        else:
            u_iexpr = u_iexpr.subs({var: var_f})  # Replace var with var_f
        u_ival = u_iexpr.subs(self.inputs.means())
        inpts = self.inputs.means()
        inpts.pop(var)
        inpts.update({fname: req_nom})
        i_val = sympy.lambdify(inpts.keys(), var_f, 'numpy')(**inpts)
        i_val.ito(solvefor_units)

        uncertvals = self.inputs.stdunc()

        reverse_GUM = None
        if kwargs.get('gum', True):
            # Plug everything in
            inpts.update(uncertvals)
            inpts.update(self.inputs.corr_values())
            inpts.update({str(u_ovar): req_uncert})
            u_ival = sympy.lambdify(inpts.keys(), u_ival, 'numpy')(**inpts)
            u_ival.ito(i_val.units)
            if not np.isreal(u_ival.magnitude) or not np.isreal(i_val.magnitude):
                logging.warning('No real solution for reverse calculation.')
                u_ival = None

            reverse_GUM = {'u_iexpr': u_iexpr,
                           'u_i': u_ival,
                           'u_iname': u_ivar,
                           'u_c': ucombined,
                           'i': i_val,
                           'f': self.model.get_baseexprs()[fidx],
                           'fname': sympy.Symbol(fname),
                           'ucname': self.model.unc_symbols[fidx],
                           'f_required': req_nom,
                           'uf_required': req_uncert,
                           }

        reverse_MC = None
        if kwargs.get('mc', True):
            # Use Monte Carlo Method - Must add correlation between f and input variables
            ucalc = uncertainty.UncertCalc(var + '=' + str(var_f), samples=self.inputs.nsamples, units=str(solvefor_units))  # Set up UncertCalc with reversed expression
            for origvar in self.inputs:
                if origvar.name == var: continue
                origunc = origvar.stdunc()
                if np.isfinite(origunc.magnitude) and origunc != 0:
                    ucalc.set_input(origvar.name, nom=origvar.nom, std=origunc.to(origvar.units).magnitude, units=str(origvar.units))
                else:
                    ucalc.set_input(origvar.name, nom=origvar.nom, units=str(origvar.units))  # Doesn't like std=0
            ucalc.set_input(fname, nom=req_nom.magnitude, std=req_uncert.magnitude, units=str(self.model.outunits[fidx]))

            # Correlate variables: see GUM C.3.6 NOTE 3 - Estimate correlation from partials
            for vname, part in zip(self.inputs.symbols, symout.Cx[fidx]):
                if str(vname) == var: continue
                inpts = self.inputs.means()
                inpts.update(uncertvals)
                inpts.update({fname: req_nom})
                ci = sympy.lambdify(inpts.keys(), part.subs({var: var_f}), 'numpy')(**inpts)
                corr = (ucalc.get_inputvar(str(vname)).stdunc() / ucalc.get_inputvar(fname).stdunc() * ci).magnitude  # dimensionless
                if np.isfinite(corr):
                    ucalc.correlate_vars(str(vname), fname, corr)

            # Include existing correlations between inputs
            if len(self.inputs.corr_list) > 0:
                for v1, v2, corr in self.inputs.corr_list:
                    if v1 == var or v2 == var: continue
                    ucalc.correlate_vars(v1, v2, corr)

            mcout = ucalc.calculate(gum=False).mc
            reverse_MC = {'u_i': mcout.uncert(0),
                          'i': mcout.nom(0),
                          'f_required': req_nom,
                          'uf_required': req_uncert,
                          'rev_ucalc': ucalc
                          }
            if np.count_nonzero(np.isfinite(list(mcout.samples().magnitude))) < self.inputs.nsamples * .95:
                # less than 95% of trials resulted in real number, consider this a no-solution
                reverse_MC['u_i'] = None
                reverse_MC['i'] = None

        self.out = ReverseOutput(reverse_GUM, reverse_MC, self.model.outnames[fidx])
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
        d = super().get_config()
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
    def __init__(self, gumdata, mcdata, funcname=None):
        self.name = funcname
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
            Passed to report.Report
        '''
        rpt = report.Report(**kwargs)
        if self.gumdata and gum:
            ui = self.gumdata['u_i']
            i = self.gumdata['i']

            rpt.hdr('GUM reverse uncertainty', level=2)
            rpt.sympy(sympy.Eq(self.gumdata['fname'], self.gumdata['f']))
            rpt.txt('\n\n Combined uncertainty:\n\n')
            rpt.sympy(sympy.Eq(self.gumdata['ucname'], self.gumdata['u_c']))
            rpt.txt('\n\nsolved for uncertainty of input:\n\n')
            rpt.sympy(sympy.Eq(self.gumdata['u_iname'], self.gumdata['u_iexpr']))
            rpt.add(u'\n\n For output value of ',
                    report.Number(self.gumdata['f_required'], matchto=self.gumdata['uf_required']),
                    ' ± ',
                    report.Number(self.gumdata['uf_required']),
                    ' (k=1),\n')
            if i is None or ui is None:
                rpt.txt(u'No real solution found\n\n')
            else:
                rpt.add('required input value is ',
                        report.Number(i, matchto=ui),
                        ' ± ',
                        report.Number(ui),
                        ' (k=1).\n\n')

        if self.mcdata and mc:
            if self.gumdata and gum:
                rpt.div()
            rpt.hdr('Monte Carlo reverse uncertainty', level=2)
            i = self.mcdata['i']
            ui = self.mcdata['u_i']
            if i is None or ui is None:
                rpt.txt('No real solution found\n\n')
            else:
                rpt.add('For output value of ',
                        report.Number(self.mcdata['f_required'], matchto=self.mcdata['uf_required']),
                        ' ± ',
                        report.Number(self.mcdata['uf_required']),
                        ' (k=1), required input value is: ',
                        report.Number(i, matchto=ui),
                        ' ± ',
                        report.Number(ui),
                        ' (k=1).\n\n')
        return rpt

    def report_summary(self, **kwargs):
        r = self.report(**kwargs)
        with plt.style.context(plotting.plotstyle):
            fig = plt.figure()
            self.plot_pdf(fig)
            r.plot(fig)
            plt.close(fig)
        return r

    def plot_pdf(self, plot=None, **kwargs):
        ''' Plot PDF/Histogram of the reverse calculation '''
        fig, ax = plotting.initplot(plot)
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
            self.mcdata['rev_ucalc'].out.mc.plot_pdf(plot=ax, **kwargs)
