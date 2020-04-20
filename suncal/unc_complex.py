''' Uncertainty Calculator wrapper for processing complex numbers. '''
import sympy
import numpy as np
import matplotlib.pyplot as plt

from . import output
from . import uparser
from . import uncertainty
from . import report


def _expr_to_complex(expr):
    ''' Convert the expression into real and imaginary parts

        Each variable is replaced with real and imaginary components:
            a -> a_r + I * a_i
        Then the expression is simplified and split into real and imaginary
        components, returning f_re, f_im sympy expressions.

        Parameters
        ----------
        expr: string
            String expression to parse

        Returns
        -------
        f_re, f_im: sympy expressions of real and imaginary components
    '''
    expr = uparser.parse_math(expr, allowcomplex=True)

    var = list(expr.free_symbols)
    comps = [sympy.Symbol('{}_r'.format(v), real=True) for v in var]
    comps += [sympy.Symbol('{}_i'.format(v), real=True) for v in var]

    subdict = {}
    for v in var:
        subdict[v] = sympy.Symbol('{}_r'.format(v), real=True) + sympy.I * sympy.Symbol('{}_i'.format(v), real=True)
    expr_real, expr_imag = expr.subs(subdict).as_real_imag()
    return expr_real, expr_imag


class UncertComplex(object):
    ''' Wrap UncertCalc by first splitting each the function and inputs into
        real and imaginary components.

        Parameters
        ----------
        funcs: string or list of strings
            Function expressions to calculate.
        magphase: bool or list of bool
            Show results in magnitude/phase format
        degrees: bool or list of bool
            Show magphase format in degrees instead of radians
        samples: int
            Number of Monte Carlo samples to run
    '''
    def __init__(self, funcs, magphase=False, degrees=True, samples=100000):
        if not isinstance(funcs, list) and not isinstance(funcs, tuple):
            funcs = [funcs]

        if not isinstance(magphase, list) and not isinstance(magphase, tuple):
            magphase = [magphase] * len(funcs)

        if not isinstance(magphase, list) and not isinstance(magphase, tuple):
            degrees = [degrees] * len(funcs)

        self.functions = funcs
        self.magphase = magphase
        self.degrees = degrees
        self.samples = samples
        self.inputs = {}
        self.corrs = []

    def set_input(self, name, nom, unc, k=2, corr=None):
        ''' Set the input value and uncertainty, measured in real and imaginary components.

            Parameters
            ----------
            name: string
                Name of input variable
            nom: complex
                Nominal/measured value of input, as complex type
            unc: complex
                Uncertainty of input. Assumes normal distribution.
            k: float
                Coverage factor of entered uncertainty. k=2 specifies ~95% coverage probability.
            corr: float
                Correlation between real and imaginary components of uncertainty.
                Must be between 0 and 1.
        '''
        # TODO: allow arbitrary distributions. Currently assumes normal
        self.inputs['{}_r'.format(name)] = (np.real(nom), np.real(unc)/k)
        self.inputs['{}_i'.format(name)] = (np.imag(nom), np.imag(unc)/k)
        if corr is not None:
            self.corrs.append(('{}_r'.format(name), '{}_i'.format(name), corr))

    def set_input_magph(self, name, mag, phase, u_mag, u_phase, k=2, degrees=True, corr=None):
        ''' Set the input value and uncertainty, measured as a magnitude and phase.

            Parameters
            ----------
            name: string
                Name of input variable
            mag: float
                Nominal/measured magnitude of variable
            phase: float
                Nominal/measured phase of variable. May be in degrees or radians
                depending on degrees parameter.
            u_mag: float
                Uncertainty of magnitude
            u_phase: float
                Uncertainty of phase
            k: float
                Coverage factor of entered uncertainty. k=2 specifies ~95% coverage probability.
            degrees: bool
                Whether the phase and u_phase parameters are entered in degrees
            corr: float
                Correlation between magnitude and phase components of uncertainty.
                Must be between 0 and 1.
        '''
        self.inputs['{}_m'.format(name)] = (mag, u_mag/k)
        if degrees:
            u_phase = np.deg2rad(u_phase)
        self.inputs['{}_p'.format(name)] = (phase, u_phase/k)
        if corr is not None:
            self.corrs.append(('{}_m'.format(name), '{}_p'.format(name), corr))

    def build_ucalc(self):
        ''' Set up UncertCalc object from the input definitions '''
        self.ucalc = uncertainty.UncertCalc(samples=self.samples)

        # Split each function into real and imaginary functions
        fnames = []
        for i, func in enumerate(self.functions):
            if '=' in func:
                fname, func = func.split('=')
                fname = fname.strip()
                fnames.append('{}_r'.format(fname))
                fnames.append('{}_i'.format(fname))
            else:
                fnames.append('f{}_r'.format(i))
                fnames.append('f{}_i'.format(i))

        for i, func in enumerate(self.functions):
            if '=' in func:
                fname, func = func.split('=')
                fname = fname.strip()
            else:
                fname = 'f_{}'.format(i)

            # Substitute variables as a -> (a_r+b_i*I)
            f_re, f_im = _expr_to_complex(func.strip())

            # Remove components that are 0 to simplify
            zeros = {v: 0 for v in list((f_re+f_im).free_symbols) if self.inputs.get(str(v)) == (0, 0)}
            f_re = f_re.subs(zeros)
            f_im = f_im.subs(zeros)

            # Set up the calculator
            self.ucalc.set_function(str(f_re), name='{}_r'.format(fname))
            self.ucalc.set_function(str(f_im), name='{}_i'.format(fname))

            if self.magphase:
                if ((isinstance(self.magphase, list) or isinstance(self.magphase, tuple)) and not self.magphase[i]):
                    continue

                f_mag = sympy.sqrt(f_re**2 + f_im**2)
                self.ucalc.set_function(str(f_mag), name='{}_m'.format(fname))
                f_ph = sympy.atan2(f_im, f_re)  # Radians

                if self.degrees:
                    f_ph = sympy.deg(f_ph)

                self.ucalc.set_function(str(f_ph), name='{}_p'.format(fname))

        for inpt, val in self.inputs.items():
            if val != (0, 0):
                if inpt.endswith('m') or inpt.endswith('p'):
                    basename = inpt.split('_')[0]
                    self.ucalc.set_function('{}_m * cos({}_p)'.format(basename, basename), name='{}_r'.format(basename))
                    self.ucalc.set_function('{}_m * sin({}_p)'.format(basename, basename), name='{}_i'.format(basename))
                self.ucalc.set_input(inpt, nom=val[0], std=val[1])

        for var1, var2, corr in self.corrs:
            self.ucalc.correlate_vars(var1, var2, corr)

    def calculate(self):
        ''' Run the calculation '''
        self.build_ucalc()
        self.ucalc_out = self.ucalc.calculate()  # Full calc output, with f_r and f_i functions
        self.out = CplxCalcOutput(self.ucalc_out, self.magphase, self.degrees)
        return self.out


class CplxCalcOutput(output.Output):
    ''' Like output.FuncOutput, but combine f_r and f_i into a single table. '''
    def __init__(self, fullout, magphase=False, degrees=True):
        self.fullout = fullout
        self.magphase = magphase
        self.degrees = degrees

    def report(self, **kwargs):
        ''' Report the results '''
        hdr = ['Method', 'Mean', 'Standard Deviation', 'Correlation']
        rpt = report.Report(**kwargs)

        deg = '°' if self.degrees else ' rad'

        fidx = 0
        for i in range(len(self.magphase)):
            if self.magphase[i]:
                # Display Mag∠Deg for this function
                outmag = self.fullout.foutputs[fidx+2]
                outph = self.fullout.foutputs[fidx+3]
                gumcor = self.fullout.ucalc.get_contour(fidx+2, fidx+3, getcorr=True)
                mccor = np.corrcoef(self.fullout.foutputs[fidx+2].get_output(method='mc').properties['samples'].magnitude,
                                    self.fullout.foutputs[fidx+3].get_output(method='mc').properties['samples'].magnitude)
                mccor = mccor[0, 1]
                rows = []
                for methodmag, methodph in zip(outmag._baseoutputs, outph._baseoutputs):
                    rows.append([methodmag.method,
                                 # TODO: incorporate mag/phase and real/imag formatting into report.Number formatter
                                 '{} ∠{}{}'.format(report.Number(methodmag.mean, matchto=methodmag.uncert),
                                                   report.Number(methodph.mean, matchto=methodph.uncert), deg),
                                 '{} ∠{}{}'.format(report.Number(methodmag.uncert),
                                                   report.Number(methodph.uncert), deg),
                                 format(gumcor if methodmag.method == 'GUM Approximation' else mccor, '.4f')])
                rpt.hdr('${}$'.format(sympy.latex(sympy.Symbol(outmag.name[:-2]))), level=3)
                fidx += 4  # [real, imag, mag, phase]
            else:
                # Display real + j*imaginary for this function
                outre = self.fullout.foutputs[fidx]
                outim = self.fullout.foutputs[fidx+1]
                gumcor = self.fullout.ucalc.get_contour(fidx, fidx+1, getcorr=True)
                mccor = np.corrcoef(self.fullout.foutputs[fidx].get_output(method='mc').properties['samples'].magnitude,
                                    self.fullout.foutputs[fidx+1].get_output(method='mc').properties['samples'].magnitude)[0, 1]
                rows = []
                for methodre, methodim in zip(outre._baseoutputs, outim._baseoutputs):
                    rows.append([methodre.method,
                                 report.Number(methodre.mean + 1j*methodim.mean, matchto=methodre.uncert),
                                 report.Number(methodre.uncert + 1j*methodim.uncert),
                                 format(gumcor if methodre.method == 'GUM Approximation' else mccor, '.4f')
                                ])
                rpt.hdr('${}$'.format(sympy.latex(sympy.Symbol(outre.name[:-2]))), level=3)
                fidx += 2  # [real, imag]
            rpt.table(rows, hdr)
        return rpt

    def plot(self, fig=None, fidx=0, **kwargs):
        ''' Plot uncertainty region in polar form

            Parameters
            ----------
            fig: matplotlib figure
                Figure to plot on. If not provided, a new figure will be created
            fidx: int
                Index of function to plot

            Keyword Arguments
            -----------------
            showgum: bool
                Show the GUM solution
            showmc: bool
                Show the Monte Carlo solution
            gumcontour: bool
                Plot GUM as contour plots. If false, an approximate 95% region will be shaded
                without considering correlation between components.
            mccontour: bool
                Plot Monte Carlo as contours. If false, a scatter plot of the first samples
                will be plotted.
            points: int
                Number of samples to plot in Monte Carlo scatter plot if mccontour is True.
            bins: int
                Number of bins for calculating 2D histogram used to estimate contour plots
            cmapmc: string
                Name of matplotlib colormap for Monte Carlo contour
            cmapgum: string
                Name of matplotlib colormap for GUM contour
            color: string
                Name of maptlotlib color for GUM shaded region
            colormc: string
                Name of matplotlib color for Monte Carlo scatter plot
            polar: bool
                Show results as polar plot (if magphase parameter is True in the calculation)

            Returns
            -------
            ax: matplotlib axis
        '''
        showgum = kwargs.get('showgum', True)
        showmc = kwargs.get('showmc', True)
        gumcontour = kwargs.get('gumcontour', True)
        mccontour = kwargs.get('mccontour', False)
        points = kwargs.get('points', 5000)
        bins = kwargs.get('bins', 35)
        cmapmc = kwargs.get('cmapmc', 'viridis')
        cmapgum = kwargs.get('cmapgum', 'hot')
        colormc = kwargs.get('colormc', 'C2')  # MC Scatter color
        color = kwargs.get('color', 'C3')      # GUM color
        polar = kwargs.get('polar', True)      # Use polar if mag/phase mode. If False, plots Phase vs Mag.

        if fig is None:
            fig = plt.gcf()
        fig.clf()

        if self.magphase[fidx]:
            # mag/phase polar
            ax = fig.add_subplot(111, polar=self.magphase[fidx] and polar)
            outy = self.fullout.foutputs[fidx+2]  # Mag/Phase components for this function index
            outx = self.fullout.foutputs[fidx+3]  # Phase is plotted as x-value in polar plots
            y = outy.get_output(method='gum').mean.magnitude   # Imaginary or Magnitude
            x = outx.get_output(method='gum').mean.magnitude   # Real or Phase
            yunc, k = outy.get_output(method='gum').expanded(.95)
            xunc, k = outx.get_output(method='gum').expanded(.95)
            xunc = xunc.magnitude
            yunc = yunc.magnitude
            samplesy = outy.get_output(method='mc').properties['samples'].magnitude
            samplesx = outx.get_output(method='mc').properties['samples'].magnitude
            if self.degrees and polar:
                x = np.deg2rad(x)
                xunc = np.deg2rad(xunc)
                samplesx = np.deg2rad(samplesx)
            if not polar:
                y, x = x, y
                yunc, xunc = xunc, yunc
                samplesy, samplesx = samplesx, samplesy

        else:  # Real/Imaginary
            ax = fig.add_subplot(111)
            outx = self.fullout.foutputs[fidx]  # Real/Imaginary components for this function index
            outy = self.fullout.foutputs[fidx+1]
            y = outy.get_output(method='gum').mean.magnitude   # Imaginary or Magnitude
            x = outx.get_output(method='gum').mean.magnitude   # Real or Phase
            yunc, _ = outy.get_output(method='gum').expanded(.95)
            xunc, _ = outx.get_output(method='gum').expanded(.95)
            xunc = xunc.magnitude
            yunc = yunc.magnitude
            samplesy = outy.get_output(method='mc').properties['samples'].magnitude
            samplesx = outx.get_output(method='mc').properties['samples'].magnitude

        if showgum:
            if gumcontour:
                re, im, p = self.fullout.ucalc.get_contour(fidx, fidx+1)   # Contour based on re/im
                re = re.magnitude
                im = im.magnitude
                if self.magphase[fidx]:
                    mg = np.sqrt(re**2+im**2)
                    ph = np.arctan2(im, re)  # Already radians
                    if polar:
                        ax.contour(ph, mg, p, 5, cmap=cmapgum)
                    else:
                        if self.degrees:
                            ph = np.rad2deg(ph)
                        ax.contour(mg, ph, p, 5, cmap=cmapgum)
                else:
                    ax.contour(re, im, p, 5, cmap=cmapgum)

            else:
                ax.plot(x, y, marker='x', color=color)
                xx = np.linspace(x-xunc, x+xunc)
                ax.fill_between(xx, y1=np.full(len(xx), y-yunc), y2=np.full(len(xx), y+yunc), color=color, alpha=.2)

        if showmc:
            if mccontour:
                y95 = np.percentile(samplesy, [2.5, 97.5])  # Cut off histogram at 95%.
                x95 = np.percentile(samplesx, [2.5, 97.5])
                counts, xbins, ybins = np.histogram2d(samplesx, samplesy, bins=bins, range=(x95, y95))
                ax.contour(counts, 5, extent=[xbins.min(), xbins.max(), ybins.min(), ybins.max()], cmap=cmapmc)
            else:
                ax.plot(samplesx[:points], samplesy[:points], marker='.', ls='', markersize=1, color=colormc, zorder=0)

        name = outy.name[:-2]
        name = sympy.latex(sympy.Symbol(name))
        if self.magphase[fidx] and not polar:
            ax.set_xlabel('$|{}|$'.format(name))
            ax.set_ylabel('Arg(${}$) ({})'.format(name, 'deg' if self.degrees else 'rad'))
        elif self.magphase[fidx]:
            ax.set_title('${}$'.format(name))
        else:
            ax.set_xlabel('Re(${}$)'.format(name))
            ax.set_ylabel('Im(${}$)'.format(name))
        fig.tight_layout()
        return ax
