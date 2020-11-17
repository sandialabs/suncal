''' Uncertainty Calculator wrapper for processing complex numbers. '''
import inspect
from collections import namedtuple
import sympy
import numpy as np
import matplotlib.pyplot as plt

from . import output
from . import out_uncert
from . import uparser
from . import uncertainty
from . import report

ValueRI = namedtuple('ValueRI', ['real', 'imag', 'ureal', 'uimag', 'correlation', 'units'])
ValueMA = namedtuple('ValueMA', ['mag', 'rad', 'umag', 'urad', 'correlation', 'units'])
ValueMAdeg = namedtuple('ValueMAdeg', ['mag', 'deg', 'umag', 'udeg', 'correlation', 'units'])


def _wrap_callable(func, informats=None, innames=None, outnames=None, outfmt='RI'):
    ''' Wrap callable function by splitting out real/imaginary components
        of each input and output

        Parameters
        ----------
        func: callalbe
            Function to wrap
        informats: dict
            Dictionary of {'argument': 'RI, 'MA', 'MAdeg'} specifying whether each
            input is in real/imaginary or magnitude/angle format
        innames: list
            List of arguments to func
        outnames: list
            List of return values from func. Can be determined automatically if
            return is a namedtuple
        outfmt: string
            Desired format for output quantity: 'RI' or 'MA'
    '''
    funcname = func.__name__

    if informats is None:
        informats = {}

    if innames is None:
        innames = list(inspect.signature(func).parameters.keys())  # Complex argument names (a, b, etc.)
    kwnames = []   # Split argument names (a_real, a_imag, b_mag, b_deg, etc.)
    for name in innames:
        fmt = informats.get(name, 'RI')
        if fmt == 'MA':
            kwnames.extend([f'{name}_mag', f'{name}_rad'])
        elif fmt == 'MAdeg':
            kwnames.extend([f'{name}_mag', f'{name}_deg'])
        else:
            kwnames.extend([f'{name}_real', f'{name}_imag'])

    if outnames is None:
        try:  # Make a test call to see return type
            kargs = {k: np.random.random() for k in innames}
            out = func(**kargs)
        except (ValueError, TypeError, IndexError, NameError):
            raise ValueError('Cannot determine output structure of callable function. Please specify foutnames parameter.')

        try:
            noutputs = len(out)
            if hasattr(out, '_fields'):  # Namedtuple
                outnames = out._fields
            else:
                outnames = ['{}_{}'.format(funcname, str(i+1)) for i in range(noutputs)]
        except TypeError:
            noutputs = 1
            outnames = [funcname]

    outcomps = []
    for i, name in enumerate(outnames):
        if outfmt == 'MA':
            outcomps.extend([f'{name}_mag', f'{name}_rad'])
        else:
            outcomps.extend([f'{name}_real', f'{name}_imag'])

    def wrapfunc(**kwargs):
        ''' The wrapped function, to be called by UncertCalc '''
        fargs = {}
        for name in innames:
            # Convert kwargs into complex numbers
            fmt = informats.get(name, 'RI')
            if fmt == 'MA':
                mag = kwargs[f'{name}_mag']
                rad = kwargs[f'{name}_rad']
                fargs[name] = mag * np.cos(rad) + 1j * mag * np.sin(rad)
            elif fmt == 'MAdeg':
                mag = kwargs[f'{name}_mag']
                rad = np.pi / 180 * kwargs[f'{name}_deg']  # Note: np.deg2rad does weird things to Pint Quantities here
                fargs[name] = mag * np.cos(rad) + 1j * mag * np.sin(rad)
            else:
                fargs[name] = kwargs[f'{name}_real'] + 1j * kwargs[f'{name}_imag']
        ret = func(**fargs)

        outvals = []
        for i, name in enumerate(outnames):
            if noutputs == 1:
                retval = ret
            else:
                retval = ret[i]
            if outfmt == 'MA':
                # note: np.angle() doesn't preserve units. np.arctan2 returns Quantity(radians)
                outvals.extend([abs(retval), np.arctan2(retval.imag, retval.real)])
            else:
                outvals.extend([retval.real, retval.imag])

        outtuple = namedtuple('CplxOut', outcomps)
        result = outtuple(*outvals)
        return result

    return kwnames, outcomps, wrapfunc


class UncertComplex(object):
    ''' Uncertainty calculation with complex numbers. Wraps UncertCalc class
        by splitting input and output values into their real/imaginary, or
        magnitude/phase components.

        Parameters
        ----------
        funcs: string or list of strings
            Function expressions to calculate.
        magphase: bool
            Show results in magnitude/phase format
        degrees: bool
            Show magphase format in degrees instead of radians
        samples: int
            Number of Monte Carlo samples to run
    '''
    def __init__(self, funcs, magphase=False, degrees=True, samples=100000):
        if not isinstance(funcs, list) and not isinstance(funcs, tuple):
            funcs = [funcs]

        self.functions = funcs
        self.magphase = magphase
        self.degrees = degrees
        self.samples = samples
        self.inputs = {}
        self.corrs = []

    def set_input(self, name, nom, unc=0, k=2, corr=None, units=None):
        ''' Set the input value and uncertainty, measured in real and imaginary components.

            Parameters
            ----------
            name: string
                Name of input variable
            nom: complex
                Nominal/measured value of input, as complex type
            unc: complex
                Uncertainty of input. Assumes normal distribution for both components.
            k: float
                Coverage factor of entered uncertainty
            corr: float
                Correlation between real and imaginary components of uncertainty.
                Must be between 0 and 1.
        '''
        # TODO: allow arbitrary distributions. Currently assumes normal
        self.inputs[name] = ValueRI(nom.real, nom.imag, unc.real/k, unc.imag/k, corr, units)
        if corr is not None:
            self.corrs.append((f'{name}_real', f'{name}_imag', corr))

    def set_input_magph(self, name, mag, phase, u_mag, u_phase, k=2, degrees=True, corr=None, units=None):
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
                Coverage factor of entered uncertainty.
            degrees: bool
                Whether the phase and u_phase parameters are entered in degrees
            corr: float
                Correlation between magnitude and phase components of uncertainty.
                Must be between 0 and 1.
        '''
        if degrees:
            self.inputs[name] = ValueMAdeg(mag, phase, u_mag/k, u_phase/k, corr, units)
        else:
            self.inputs[name] = ValueMA(mag, phase, u_mag/k, u_phase/k, corr, units)

        if corr is not None:
            if degrees:
                self.corrs.append((f'{name}_mag', f'{name}_deg', corr))
            else:
                self.corrs.append((f'{name}_mag', f'{name}_rad', corr))

    def _build_ucalc(self):
        ''' Set up UncertCalc object from the input definitions '''

        # Decide what format (RI or MA) each input is defined as
        informats = {}
        for inpt, val in self.inputs.items():
            informats[inpt] = 'MA' if hasattr(val, 'rad') else 'MAdeg' if hasattr(val, 'deg') else 'RI'

        if callable(self.functions[0]):
            # Callables, use _wrap_callable to make a new function
            outfmt = 'MA' if self.magphase else 'RI'
            innames, outnames, func = _wrap_callable(self.functions[0], informats=informats, outfmt=outfmt)
            self.ucalc = uncertainty.UncertCalc(func, finnames=innames, foutnames=outnames)
        else:
            # Sympy expressions, substitute real/imaginary variables and split the output
            self.ucalc = uncertainty.UncertCalc(samples=self.samples)

            # Split out equal signs in equations
            fnames = []
            funcs = []
            for i, func in enumerate(self.functions):
                if '=' in func:
                    fname, func = func.split('=')
                    fnames.append(fname.strip())
                    funcs.append(func.strip())
                else:
                    fnames.append(f'f{i}')
                    funcs.append(func.strip())

            # Convert strings to sympy expressions
            funcs = [uparser.parse_math(expr, allowcomplex=True) for expr in funcs]

            # Substitute inputs a -> a_real, a_imag
            for inpt, fmt in informats.items():
                for i, func in enumerate(funcs):
                    if fmt == 'MA':
                        mag = sympy.Symbol(f'{inpt}_mag', real=True)
                        rad = sympy.Symbol(f'{inpt}_rad', real=True)
                        funcs[i] = func.subs({inpt: mag * sympy.cos(rad) + sympy.I * mag * sympy.sin(rad)})
                    else: # fmt == 'RI':
                        funcs[i] = func.subs({inpt: sympy.Symbol(f'{inpt}_real', real=True) + sympy.I * sympy.Symbol(f'{inpt}_imag', real=True)})

            # Split each function into real/imag (or mag/phase)
            splitnames = []
            splitfuncs = []
            for func, fname in zip(funcs, fnames):
                f_real, f_imag = func.as_real_imag()
                f_real = f_real.simplify()
                f_imag = f_imag.simplify()

                # The above as_real_imag() must have the symbols defined with real=True to avoid
                # having an expression with re(x) and im(x) in it. But with real=True, any case of
                # sqrt(x**2) is simplified into Abs(x), which breaks the Derivatives later on.
                # These two lines are a stupid way to convert the expressions to standard symbols
                # (without real=True) so that sqrt(x**2) remains sqrt(x**2) which is differentiable.
                f_real = sympy.sympify(str(f_real))
                f_imag = sympy.sympify(str(f_imag))

                if self.magphase:
                    f_mag = sympy.sqrt(f_real**2 + f_imag**2)
                    f_ph = sympy.atan2(f_imag, f_real)
                    splitnames.extend([f'{fname}_mag', f'{fname}_rad'])
                    splitfuncs.extend([f_mag, f_ph])
                else:
                    splitfuncs.extend([f_real, f_imag])
                    splitnames.extend([f'{fname}_real', f'{fname}_imag'])

            # Simplify the expressions and add to calc
            for func, name in zip(splitfuncs, splitnames):
                func = func.simplify()
                self.ucalc.set_function(str(func), name=name)

        for inpt, val in self.inputs.items():
            # Keep angular units as dimensionless so Uy doesn't end up with degrees
            # (which is also dimensionless) in the output
            if hasattr(val, 'deg'):
                self.ucalc.set_input(f'{inpt}_mag', val.mag, std=val.umag, units=val.units)
                self.ucalc.set_input(f'{inpt}_deg', val.deg, std=val.udeg, units='dimensionless')
            elif hasattr(val, 'rad'):
                self.ucalc.set_input(f'{inpt}_mag', val.mag, std=val.umag, units=val.units)
                self.ucalc.set_input(f'{inpt}_rad', val.rad, std=val.urad, units='dimensionless')
            else:
                self.ucalc.set_input(f'{inpt}_real', val.real, std=val.ureal, units=val.units)
                self.ucalc.set_input(f'{inpt}_imag', val.imag, std=val.uimag, units=val.units)

        for var1, var2, corr in self.corrs:
            self.ucalc.correlate_vars(var1, var2, corr)

    def calculate(self):
        ''' Run the calculation '''
        self._build_ucalc()
        self.ucalc_out = self.ucalc.calculate()  # Full calc output, with f_real and f_imag functions
        self.out = CplxCalcOutput(self.ucalc_out, self.magphase, self.degrees)
        return self.out


class GUMOutputCplx(output.Output):
    ''' Output of a Complex GUM calculation

        Parameters
        ----------
        fullgum: out_uncert.GUMOutput
            Output object of full GUM calculation (split into real/imag components)
        magphase: bool
            Results were calculated in magnitude/phase format
        degrees: bool
            Phases were calculated in degrees
    '''
    def __init__(self, fullgum, magphase=False, degrees=True):
        self.full = fullgum
        self.magphase = magphase
        self.degrees = degrees
        self.names = [f.rsplit('_')[0] for f in self.full.names[::2]]
        self.nouts = len(self.names)

    def _index(self, fname):
        ''' Get index if function name '''
        if fname is None:
            return 0
        else:
            return self.names.index(fname) if isinstance(fname, str) else fname

    def _basenames(self, fname=None):
        ''' Get base names into self.full for both components '''
        fidx = self._index(fname)
        fname = self.names[fidx]
        if self.magphase:
            mag = f'{fname}_mag'
            ph = f'{fname}_rad'
            return mag, ph
        else:
            real = f'{fname}_real'
            imag = f'{fname}_imag'
            return real, imag

    def nom(self, fname=None):
        ''' Return nominal value as tuple of (real, imag) or (mag, phase) '''
        x, y = self._basenames(fname)
        return self.full.nom(x), self.full.nom(y)

    def uncert(self, fname=None):
        ''' Return uncertainty as tuple of (real, imag) or (mag, phase) '''
        x, y = self._basenames(fname)
        return self.full.uncert(x), self.full.uncert(y)

    def degf(self, fname=None):
        ''' Get degrees of freedom for the function '''
        x, y = self._basenames(fname)
        return self.full.degf(x)

    def report(self, **kwargs):
        ''' Generate report of complex values and uncertainties '''
        hdr = ['Function', 'Nominal', 'Std. Uncertainty', 'Correlation']
        deg = '°' if self.degrees else ' rad'
        rows = []
        for i, fname in enumerate(self.names):
            cor = self.correlation(i)
            if self.magphase:
                mag, ph = self.nom(i)
                umag, uph = self.uncert(i)
                if self.degrees:
                    ph, uph = np.rad2deg(ph), np.rad2deg(uph)
                mag, ph, umag, uph = mag.magnitude, ph.magnitude, umag.magnitude, uph.magnitude
                rows.append([fname,
                             '{} ∠{}{}'.format(report.Number(mag, matchto=umag), report.Number(ph, matchto=uph), deg),
                             '± {} ∠{}{}'.format(report.Number(umag), report.Number(uph), deg),
                             '{:.4f}'.format(cor)])
            else:
                real, imag = self.nom(i)
                ureal, uimag = self.uncert(i)
                rows.append([fname,
                             '{}'.format(report.Number(real + 1j*imag, matchto=ureal)),
                             '± {}'.format(report.Number(ureal + 1j*uimag)),
                             '{:.4f}'.format(cor)])

        r = report.Report(**kwargs)
        r.table(rows, hdr=hdr)
        return r

    def plot(self, fidx=0, ax=None, polar=True, **kwargs):
        ''' Plot the uncertainty region on polar or rectangular axis

            Parameters
            ----------
            fidx: int or string
                Function index or name
            ax: matplotlib axis
                Axis to plot on
            polar: bool
                Show plot in polar format

            Keyword Arguments
            -----------------
            contour: bool
                Draw uncertainty region with contour lines
            cmap: string
                Name of Matplotlib colormap for contour lines
            color: string
                Name of color for shaded region (when contour = False)
        '''
        contour = kwargs.get('contour', True)  # Contour lines vs fill
        cmap = kwargs.get('cmap', 'hot')
        color = kwargs.get('color', 'C3')

        fidx = self._index(fidx)
        fname = self.names[fidx]

        if ax is None:
            fig = plt.gcf()
            ax = fig.add_subplot(111, polar=polar)

        xname, yname = self._basenames(fidx)

        if contour:
            # Draw contoured uncert region
            fidx = fidx*2
            x, y, p = out_uncert._contour(self.full._nom, self.full.Uy, fidx, fidx+1)
            xunc, yunc = 0, 0
        else:
            x, y = self.nom(fidx)
            xunc, yunc, k = self.expanded(fidx)
            xunc, yunc = xunc.magnitude, yunc.magnitude
            x, y = x.magnitude, y.magnitude

        # Convert internal RI or MA into whatever's needed for plot
        if not self.magphase:
            # xy are RI
            if polar:
                # Polar plots theta first
                y, x = np.sqrt(x**2 + y**2), np.arctan2(y, x)
                yunc, xunc = np.sqrt(xunc**2 + yunc**2), np.arctan2(yunc, xunc)
        else:
            # xy are MA
            if not polar:
                x, y = x * np.cos(y), x * np.sin(y)
                xunc, yunc = xunc * np.cos(yunc), xunc * np.sin(yunc)
            else:
                x, y = y, x
                xunc, yunc = yunc, xunc

        # Plot it
        if contour:
            ax.contour(x, y, p, 10, cmap=cmap)
        else:
            # Shade 95% region (without considering correlation)
            ax.plot(x, y, marker='x', color=color)
            xx = np.linspace(x-xunc, x+xunc)
            ax.fill_between(xx, y1=np.full(len(xx), y-yunc), y2=np.full(len(xx), y+yunc), color=color, alpha=.2)

        if not polar:
            ax.set_xlabel(f'Re({fname})')
            ax.set_ylabel(f'Im({fname})')

    def correlation(self, fname=None):
        ''' Get correlation between real/imaginary (or magnitude/phase) components '''
        fidx = self._index(fname)
        idx1 = fidx * 2
        idx2 = idx1 + 1
        return self.full.correlation(idx1, idx2)

    def expanded(self, fname=None, cov=0.95, **kwargs):
        ''' Get expanded uncertainty of the function result

            Parameters
            ----------
            fname: string or int
                Function name or index
            cov: float
                Confidence level, 0-1 range

            Returns
            -------
            x, y: float
                Expanded uncertainties for real/imaginary, or magnitude/phase values
            k: float
                K-value required to reach the level of confidence
        '''
        x, y = self._basenames(fname)
        x, k = self.full.expanded(x, cov=cov, **kwargs)
        y, _ = self.full.expanded(y, cov=cov, **kwargs)
        return x, y, k


class MCOutputCplx(GUMOutputCplx):
    ''' Output of a Complex Monte Carlo calculation

        Parameters
        ----------
        fullmc: out_uncert.MCOutput
            Output object of full MC calculation (split into real/imag components)
        magphase: bool
            Results were calculated in magnitude/phase format
        degrees: bool
            Phases were calculated in degrees
    '''
    def __init__(self, fullmc, magphase=False, degrees=True):
        super().__init__(fullmc, magphase, degrees)

    def degf(self):
        raise NotImplementedError  # MC doesn't have degf

    def expanded(self, fname=None, cov=0.95, **kwargs):
        ''' Get expanded uncertainty of the function result

            Parameters
            ----------
            fname: string or int
                Function name or index
            cov: float
                Confidence level, 0-1 range

            Returns
            -------
            xlo, xhi: float
                Lower and upper bounds for real or magnitude component
            ylo, yhi: float
                Lower and upper bounds for imaginary or phase component
            k: float
                K-value required to reach the level of confidence
        '''
        x, y = self._basenames(fname)
        xlo, xhi, k = self.full.expanded(x, cov=cov, **kwargs)
        ylo, yhi, _ = self.full.expanded(y, cov=cov, **kwargs)
        return (xlo, xhi), (ylo, yhi), k

    def plot(self, fidx=0, ax=None, polar=True, **kwargs):
        ''' Plot scatterplot of samples or contours of uncertainty region
            on polar or rectangular axis

            Parameters
            ----------
            fidx: int or string
                Function index or name
            ax: matplotlib axis
                Axis to plot on
            polar: bool
                Show plot in polar format

            Keyword Arguments
            -----------------
            points: int
                Number of samples to include (default = 5000)
            bins: int
                Number of bins for forming contour plot
            contour: bool
                Draw uncertainty region with contour lines
            cmap: string
                Name of Matplotlib colormap for contour lines
            color: string
                Name of color for shaded region (when contour = False)
        '''

        points = kwargs.get('points', 5000)
        bins = kwargs.get('bins', 35)
        cmap = kwargs.get('cmap', 'viridis')
        color = kwargs.get('color', 'C2')
        contour = kwargs.get('contour', False)

        fidx = self._index(fidx)
        fname = self.names[fidx]

        if ax is None:
            fig = plt.gcf()
            ax = fig.add_subplot(111, polar=polar)

        samplesx = self.full.samples(fidx).magnitude
        samplesy = self.full.samples(fidx+1).magnitude

        if self.magphase:
            if not polar:
                samplesx, samplesy = samplesx * np.cos(samplesy), samplesx * np.sin(samplesy)
            else:
                samplesx, samplesy = samplesy, samplesx
        elif polar:
            samplesy, samplesx = np.sqrt(samplesx**2 + samplesy**2), np.arctan2(samplesy, samplesx)

        if contour:
            y95 = np.percentile(samplesy, [2.5, 97.5])  # Cut off histogram at 95%.
            x95 = np.percentile(samplesx, [2.5, 97.5])
            counts, xbins, ybins = np.histogram2d(samplesx, samplesy, bins=bins, range=(x95, y95))
            ax.contour(counts, 5, extent=[xbins.min(), xbins.max(), ybins.min(), ybins.max()], cmap=cmap)
        else:
            ax.plot(samplesx[:points], samplesy[:points], marker='.', ls='', markersize=1, color=color, zorder=0)

        if not polar:
            ax.set_xlabel(f'Re({fname})')
            ax.set_ylabel(f'Im({fname})')


class CplxCalcOutput(output.Output):
    ''' Output of a complex uncertainty calculation

        Parameters
        ----------
        fullout: out_uncert.UncertOutput
            Output object of full calculation (split into real/imag components)
        magphase: bool
            Results were calculated in magnitude/phase format
        degrees: bool
            Phases were calculated in degrees
    '''
    def __init__(self, fullout, magphase=False, degrees=True):
        self.fullout = fullout
        self.gum = GUMOutputCplx(fullout.gum, magphase, degrees)
        self.mc = MCOutputCplx(fullout.mc, magphase, degrees)
        self.magphase = magphase
        self.degrees = degrees
        self.names = [f.rsplit('_')[0] for f in self.fullout.names[::2]]
        self.nouts = len(self.names)

    def _correlation(self, out, fidx):
        ''' Get correlation between real/imaginary components

            Parameters
            ----------
            out: GUMOutputCplx or MCOutputCplx
                The output to get correlation from
            fidx: int or str
                Function index or name
        '''
        idx1 = fidx*2
        idx2 = idx1+1
        return out.correlation(idx1, idx2)

    def report(self, **kwargs):
        ''' Report the results '''
        hdr = ['Function', 'Method', 'Nominal', 'Standard Uncertainty', 'Correlation']
        rpt = report.Report(**kwargs)
        deg = '°' if self.degrees else ' rad'

        def _addcol(out, fname):
            fidx = self.names.index(fname)
            cor = self._correlation(out, fidx)
            if self.magphase:
                mag = out.nom(f'{fname}_mag')
                umag = out.uncert(f'{fname}_mag')
                ph = out.nom(f'{fname}_rad').magnitude
                uph = out.uncert(f'{fname}_rad').magnitude
                if self.degrees:
                    ph, uph = np.rad2deg(ph), np.rad2deg(uph)

                cols = ['{} ∠{}{}'.format(report.Number(mag, matchto=umag), report.Number(ph, matchto=uph), deg),
                        '± {} ∠{}{}'.format(report.Number(umag), report.Number(uph), deg),
                        '{:.4f}'.format(cor)]
            else:
                real = out.nom(f'{fname}_real')
                ureal = out.uncert(f'{fname}_real')
                imag = out.nom(f'{fname}_imag')
                uimag = out.uncert(f'{fname}_imag')
                cols = ['{}'.format(report.Number(real + 1j*imag, matchto=ureal)),
                        '± {}'.format(report.Number(ureal + 1j*uimag)),
                        '{:.4f}'.format(cor)]
            return cols

        rows = []
        for fname in self.names:
            if self.fullout.gum:
                row = [fname, 'GUM']
                row.extend(_addcol(self.fullout.gum, fname))
                rows.append(row)
            if self.fullout.mc:
                row = [fname, 'Monte Carlo']
                row.extend(_addcol(self.fullout.mc, fname))
                rows.append(row)
        rpt.table(rows, hdr)
        return rpt

    def plot(self, fig=None, fidx=0, **kwargs):
        ''' Plot uncertainty region in polar form

            Parameters
            ----------
            fig: matplotlib figure
                Figure to plot on. If not provided, a new figure will be created
            fidx: int or str
                Index or name of function to plot

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

        ax = fig.add_subplot(111, polar=polar)

        if showgum:
            self.gum.plot(fidx=fidx, ax=ax, cmap=cmapgum, color=color, polar=polar, contour=gumcontour)

        if showmc:
            self.mc.plot(fidx=fidx, ax=ax, cmap=cmapmc, color=colormc, polar=polar, contour=mccontour, points=points, bins=bins)

        fig.tight_layout()
        return ax
