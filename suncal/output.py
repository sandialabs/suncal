''' Generic functions for reporting output of calculations.

    All output reports are stored in markdown format. The MDstring class
    provides methods for maintaining the string data and adding images
    and equations. The NumFormatter class provides functionality for
    converting numbers into strings beyond the built-in string formatters,
    handling things such as significant figures, scientific/engineering
    notation, and SI prefixes. The Output class should be subclassed
    for each calculation type with specific reports. This class has
    representer methods that convert the markdown into HTML or plain
    text for use in Jupyter notebooks or interactive terminal sessions.
'''
import os
import re
import shutil
import sympy
import numpy as np
from scipy import stats
from io import BytesIO
import base64
import markdown
import subprocess
from contextlib import contextmanager
import matplotlib as mpl
import matplotlib.pyplot as plt

from . import ureg
from . import css
from . import uparser
from . import distributions
from . import latexchars


mpl.style.use('bmh')
sympy.printing.latex.__globals__['requires_partial'] = lambda x: True   # Render all derivatives as partial
sympy.printing.pretty.__globals__['requires_partial'] = lambda x: True

latex_path = shutil.which('pdflatex')
pandoc_path = shutil.which('pandoc')

if os.name == 'posix':
    # Mac APPs run from Finder won't pick up the path from .profile.
    # Default pandoc install is /usr/bin/local, try that one.
    # Default MacTex install has symlinks in /Library/Tex/texbin
    if pandoc_path is None:
        pandoc_path = '/usr/local/bin/pandoc' if os.path.exists('/usr/local/bin/pandoc') else None
    if latex_path is None:
        latex_path = '/Library/TeX/texbin/pdflatex' if os.path.exists('/Library/TeX/texbin/pdflatex') else None

_math_mode = 'mpl'  # How to render math: 'mpl', 'mathjax', or 'text'
_fig_mode = 'svg'   # or 'png'
_mathjaxurl = 'https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.4/MathJax.js'

mplcontext = {'figure.figsize': (8, 6), 'font.size': 14}   # Common plot parameters, usage: "with mpl.style.context(mplcontext):"


@contextmanager
def report_format(math='mpl', fig='svg', mjurl=None):
    ''' Context manager for changing report format dynamically.

        Parameters
        ----------
        math: string
            Mode for displaying math. One of 'mpl', 'mathjax', 'text'
        fig: string
            Figure mode. 'svg', 'png', 'text'
        mjurl: string
            URL for mathjax css

        Usage
        -----
            >>> with(report_format(math='mathjax', fig='svg')):
            >>>     r.get_md()
    '''
    global _math_mode
    global _fig_mode
    global _mathjaxurl
    m, f, mj = _math_mode, _fig_mode, _mathjaxurl  # Save
    _math_mode, _fig_mode = math, fig  # Override
    if mjurl:
        _mathjaxurl = mjurl
    yield
    _math_mode, _fig_mode, _mathjaxurl = m, f, mj  # Restore


def format_math(expr):
    ''' Format the math expr as a string, either using sympy or
        the plain string if sympy fails.
    '''
    if isinstance(expr, str):
        sympyexpr = uparser.get_expr(expr)
        if sympyexpr is None:
            return expr    # Can't sympify. Use string as-is.
    else:
        sympyexpr = expr
    return sympyeqn(sympyexpr)


def sympyeqn(sym_expr):
    ''' Convert sympy to $latex$ or plain-text '''
    if _math_mode != 'text':
        return '${}$'.format(sympy.latex(sym_expr))
    else:
        return sympy.pretty(sym_expr)


def sympyeqn_units(sym_expr, unit, **kwargs):
    ''' Convert sympy equation and units into $latex$ or plain-text '''
    eqstr = sympyeqn(sym_expr)
    if unit is not None:
        if _math_mode != 'text':
            if kwargs.get('fullunit', False):
                unitstr = '{:L}'.format(unit)
            else:
                unitstr = '{:~L}'.format(unit)
            unitstr.encode('ascii', 'latex').decode()   # Convert other special chars to latex codes
            eqstr = eqstr.rstrip('$') + r'\,{}$'.format(unitstr)
        else:
            unitstr = formatunit(unit, **kwargs)  # Text unit
            eqstr = eqstr + ' {:P}'
    return eqstr


def formatunittex(units, fullunit=False, bracket=False):
    ''' Convert Pint units to latex.

        Parameters
        ----------
        units: pint Unit
            Units to format
        fullunit: bool
            Format the full name (e.g. 'centimeters' instead of 'cm')
        bracket: bool
            Include square brackets around unit, useful for plot axes (e.g. [$cm^2$])
    '''
    if units == ureg.dimensionless:
        return ''

    if fullunit:
        ustr = '{:L}'.format(units)
    else:
        ustr = '{:~L}'.format(units)
    if bracket:
        ustr = r' $\left[ {} \right]$'.format(ustr)
    else:
        ustr = r' ${}$'.format(ustr)

    # Convert any remaining unicode symbols into latex, which Pint's L format doesn't do.
    ustr = ustr.encode('ascii', 'latex').decode()
    return ustr


def formatunit(units, fullunit=False, dimensionlessabbr=''):
    ''' Convert Pint units to plain-text (UTF8)

        Parameters
        ----------
        fullunit: bool
            Show full, unabbreviated unit name (e.g. 'centimeters'
            instead of 'cm')
        dimensionlessabbr: string
            String to return for dimensionless abbreviations. Useful
            for putting value in markdown table where empty string
            doesn't work.
    '''
    fmt = 'P'
    fmt = fmt if fullunit else '~' + fmt
    if units is None or units == ureg.dimensionless:
        return dimensionlessabbr
    unitstr = f'  {units:{fmt}}'
    return unitstr


def eqn_to_mdimg(expr, color=None, fontsize=18):
    ''' Convert tex equation into <img> tag using matplotlib and context figure mode (svg or png). '''
    if not expr.startswith('$'):
        expr = '${}$'.format(expr)
    buf = BytesIO()
    style = {'savefig.facecolor': (0, 0, 0, 0), 'savefig.edgecolor': (0, 0, 0, 0), 'text.color': mpl.rcParams['text.color']}
    if color:
        style['text.color'] = color

    if _fig_mode == 'svg':
        with mpl.style.context(style):
            prop = mpl.font_manager.FontProperties(size=fontsize)
            mpl.mathtext.math_to_image(expr, buf, prop=prop, format='svg')
        svg = buf.getvalue().decode('utf-8')
        svg = svg[svg.find('<svg'):]  # Strip HTML header stuff
        s = r"![](data:image/svg+xml;base64,{})".format(base64.b64encode(svg.encode('utf-8')).decode('utf-8'))

    elif _fig_mode == 'png':
        with mpl.style.context(style):
            mpl.mathtext.MathTextParser('bitmap').to_png(buf, expr, color=style.get('text.color'), dpi=120, fontsize=fontsize)
        buf.seek(0)
        s = r"![](data:image/png;base64,{})".format(base64.b64encode(buf.read()).decode('utf-8'))

    else:
        raise ValueError('Unsupported figure mode {}'.format(_fig_mode))
    return s


def sympy_to_svg(expr, color='black', fontsize=18):
    ''' Convert sympy expression into SVG (string) '''
    tex = '$' + sympy.latex(expr) + '$'
    buf = BytesIO()
    with mpl.style.context({'text.color': color, 'savefig.facecolor': (0, 0, 0, 0), 'savefig.edgecolor': (0, 0, 0, 0)}):
        prop = mpl.font_manager.FontProperties(size=fontsize)
        mpl.mathtext.math_to_image(tex, buf, prop=prop, format='svg')
    svg = buf.getvalue().decode('utf-8')
    svg = svg[svg.find('<svg'):]  # Strip HTML header stuff
    return svg


def tex_to_html(expr):
    ''' Convert sympy expression to HTML <img> tag. '''
    md = eqn_to_mdimg(expr)
    return markdown.markdown(md)


def initplot(plot):
    ''' Initialize figure and axis to plot on.

        If plot is None, new figure and axis will be created.
        If plot is Figure, the current axis will be used (or possibly cleared depending on function)
        If plot is Axis, it will be plotted on

        fig, ax tuple is always returned.
    '''
    if plot is None:
        fig = plt.gcf()
        ax = plt.gca()
    elif hasattr(plot, 'gca'):
        fig, ax = plot, plot.gca()
    elif hasattr(plot, 'figure'):
        fig, ax = plot.figure, plot
    else:
        raise ValueError('Undefined plot type')
    return fig, ax


def axes_grid(n, fig, maxcols=4):
    ''' Make a grid of n matplotlib axes in the figure with at most maxcols columns. '''
    rows = int(np.ceil(n/maxcols))
    try:
        cols = int(np.ceil(n/rows))
    except ZeroDivisionError:
        return []
    axs = [fig.add_subplot(rows, cols, idx+1) for idx in range(n)]
    return axs


def setup_mplparams():
    ''' Set some default matplotlib parameters for things like fonts '''
    try:
        mpl.rcParams['figure.subplot.left'] = .12
        mpl.rcParams['figure.subplot.right'] = .95
        mpl.rcParams['figure.subplot.top'] = .95
        mpl.rcParams['figure.subplot.bottom'] = .12
        mpl.rcParams['figure.subplot.hspace'] = .4
        mpl.rcParams['figure.subplot.wspace'] = .4
        mpl.rcParams['axes.formatter.use_mathtext'] = True
        mpl.rcParams['axes.formatter.limits'] = [-4, 4]
        mpl.rcParams['mathtext.fontset'] = 'stixsans'
        mpl.rcParams['mathtext.default'] = 'regular'
        mpl.rcParams['figure.max_open_warning'] = 0
        mpl.rcParams['axes.formatter.useoffset'] = False
    except AttributeError:
        pass
setup_mplparams()


def probplot(y, ax, sparams=(), dist='norm', conf=.95):
    ''' Plot quantile probability plot. If data falls on straight line,
        data is normally distributed.

        Parameters
        ----------
        y: array
            Sampled data to fit
        ax: matplotlib axis
            Axis to plot on
        sparams: dictionary
            Shape parameters for distribution.
        dist: string
            Name of distribution to fit.
        conf: float
            Level of confidence for confidence bands
    '''
    _dist = distributions.get_distribution(dist, **sparams)
    y = y.copy()
    y.sort()
    n = len(y)
    p = (np.arange(n) + 0.5)/n   # like R's ppoints function assuming n > 10
    z = _dist.ppf(p)
    try:
        coef = np.polyfit(z, y, deg=1)  # Find fit line
    except np.linalg.LinAlgError:
        coef = (0, 0)
    fitval = np.poly1d(coef)(z)

    zz = stats.norm.ppf(1-(1-conf)/2)
    SE = (coef[0]/_dist.pdf(z)) * np.sqrt(p*(1-p)/n)

    ax.plot(z, y, marker='o', ls='', color='C0')
    ax.plot(z, fitval, ls='-', color='C1')
    ax.fill_between(z, fitval-zz*SE, fitval+zz*SE, alpha=.4, color='C2')
    ax.set_xlabel('Theoretical Quantiles ({})'.format(dist))
    ax.set_ylabel('Ordered Sample Values')


def fitdist(y, distname='norm', plot=None, qqplot=False, bins='sqrt', points=None, coverage=None, xlabel='Parameter'):
    ''' Fit a distribution to the data and plot comparison.

        Parameters
        ----------
        y: array
            1D Data to fit
        dist: string to rv_continuous
            Distribution to fit to
        plot: matplotlib figure or axis
            Figure or axis to plot comparison. Will be cleared.
        qqplot: boolean
            Plot a Q-Q normal probability plot
        bins: int
            Number of bins for histogram (see numpy.histogram_bin_edges).
            Defaults to square root of data size.
        points: int
            Number of points to show in Q-Q plot
        coverage: array
            List of coverage probabilities to plot as vertical lines
    '''
    fig, ax = initplot(plot)
    fig.clf()

    # Takes a long time with all 1E6 points.. thin them out
    if points is not None:
        ythin = y[::len(y)//points]
    else:
        ythin = y

    if distname:
        dist = distributions.get_distribution(distname)
        fitparams = dist.fit(ythin)

    ax = fig.add_subplot(1, qqplot+1, 1)
    if not np.isfinite(y).any():
        return
    y = y[np.isfinite(y)]

    ax.hist(y, density=True, bins=bins)

    if distname:
        xx = np.linspace(y.min(), y.max(), num=100)
        yy = dist.pdf(xx)
        ax.plot(xx, yy, color='C1', label='{} Fit'.format(distname))

    if coverage is not None:
        for c in coverage:
            ax.axvline(c, ls='--', color='C2')

    ax.set_xlabel(xlabel)
    ax.set_ylabel('Probability Density')

    if distname:
        ax.legend(loc='best')
        if qqplot:
            ax2 = fig.add_subplot(1, 2, 2)
            params = fitparams.copy()
            params.pop('loc', 0)     # Omit loc/scale to get quantiles
            params.pop('scale', 0)
            probplot(ythin, ax2, sparams=params, dist=distname)
        return fitparams


class NumFormatter(object):
    ''' Class for configuring and formatting numbers.

        Parameters
        ----------
        n: int
            Number significant figures for uncertainty values
        fmt: string
            Number format, can be: ['auto', 'decimal', 'scientific', 'engineering', 'si']
        thresh: int
            Threshold for changing to scientific notation when in auto mode (10^thresh)
        elower: bool
            Show exponential with lower case 'e'
    '''
    def __init__(self, n=2, fmt='auto', thresh=5, elower=True):
        self.formatspec = {'n': n, 'fmt': fmt, 'thresh': thresh, 'elower': elower}
        numfmts = ['auto', 'decimal', 'scientific', 'sci', 'engineering', 'eng', 'si']
        if fmt not in numfmts:
            raise ValueError('Number Format must be one of {}'.format(', '.join(numfmts)))

        if n < 1:
            raise ValueError('Significant Figures must be >= 1')

    @property
    def n(self):
        return self.formatspec['n']

    def f(self, num, **kwargs):
        ''' Format the number with appropriate significant figures.

            Parameters
            ----------
            num: float
                number to format

            Keyword Arguments
            -----------------
            n: int
                number of significant figures to show
            fmin: int
                minimum number of decimal places (e.g. num=123, n=2 --> '120' but with fmin=1 --> '120.0')
            fmt: string
                format specifier ('auto', 'decimal', 'scientific'/'sci', 'engineering'/'eng', 'si')
            thresh: int
                threshold for switching to scientific notation in 'auto' format
            elower: bool
                use lower case 'e' in exponent (True) or upper case 'E' (False)
            matchto: float
                determine number of decimal places when matchto is formatted to n sigfigs, then
                match num to that many decimal places.
            matchtolim: int
                maximum decimal places allowed for matchto. default = 10

            Notes
            -----
            If any kwargs are not specified, the class-wide default will be used.

            Examples
            --------
            For num = 0.000123 with figs = 2:

            >>> decimal: 0.00012
            >>> scientific: 1.2e-04
            >>> engineering: 120e-06
            >>> si: 120u

            Returns
            -------
            string:
                Formatted string representation of number with correct number of significant digits.
        '''
        figs = kwargs.get('n', self.formatspec.get('n', 2))
        fmin = kwargs.get('fmin', None)
        fmt = kwargs.get('fmt', self.formatspec.get('fmt', 'auto'))
        thresh = kwargs.get('thresh', self.formatspec.get('thresh', 5))
        elower = kwargs.get('elower', self.formatspec.get('elower', True))
        matchtolim = kwargs.get('matchtolim', 10)
        fullunit = kwargs.get('fullunit', False)
        echr = 'e' if elower else 'E'

        if hasattr(num, 'units'):
            num, unit = num.magnitude, num.units
        else:
            unit = None

        if 'matchto' in kwargs:
            figs = min(matchtolim, self.matchprecision(num, kwargs.get('matchto'), figs))

        if num is None:
            numstr = 'nan'

        elif not np.isfinite(num):
            numstr = '{}'.format(num)  # Will format into 'nan' or 'inf'

        elif num == 0:
            if figs == 1:
                numstr = '0'
            else:
                numstr = '0.' + '0'*(figs-1)

            if fmt in ['sci', 'scientific', 'eng', 'engineering']:
                numstr = numstr + 'e+00'

        else:
            if fmt == 'auto':
                if abs(num) > 10**thresh or abs(num) < 10**-thresh:
                    fmt = 'sci'
                else:
                    fmt = 'decimal'

            exp = int(np.floor(np.log10(abs(num))))   # Exponent if written in exp. notation.
            roundto = -(exp - (figs-1))
            if fmin is not None:
                roundto = max(fmin, roundto)
                figs = roundto + exp + 1

            if fmt == 'decimal':
                numstr = '{{:.{}f}}'.format(max(0, roundto)).format(np.round(num, roundto))

            elif fmt == 'sci' or fmt == 'scientific':
                numstr = '{{:.{}{}}}'.format(figs-1, echr).format(num)

            elif fmt == 'eng' or fmt == 'engineering' or fmt == 'si':
                # REF: https://stackoverflow.com/a/40691220
                exp3 = exp - (exp % 3)  # Exponent as multiple of 3, for engineering notation
                num = num/(10**exp3)
                roundto = -int((np.floor(np.log10(abs(num)))) - (figs-1))
                num = np.round(num, roundto)
                if num == int(num):
                    num = int(num)  # Gets rid of extra .0

                if fmt == 'si' and exp3 >= -24 and exp3 <= 24:
                    suffix = 'yzafpnum kMGTPEZY'[exp3 // 3 + 8]
                    numstr = '{{:.{}f}}{{}}'.format(max(0, roundto)).format(num, suffix).rstrip()
                else:
                    numstr = '{{:.{}f}}{}{{:+03d}}'.format(max(0, roundto), echr).format(num, exp3)
            else:
                raise ValueError('Unknown format {}'.format(fmt))

        if unit:
            numstr += '{}'.format(formatunit(unit, fullunit=fullunit))

        return numstr

    def matchprecision(self, num, num2, n=2):
        ''' Return number of sigfigs required to match num2 precision when printed to n figs.
            Can pass output of this to sympy's .n() method.

            Example: matchprecision(num=100, num2=0.123, n=2) == 5
            Need to print 100.00 (5 sigfigs) to match the 2 decimal places of 0.12 (num2 with 2 sigfigs)
            This takes care of any zeros at the end.
        '''
        if hasattr(num, 'units'):
            num = num.magnitude
        if hasattr(num2, 'units'):
            num2 = num2.magnitude

        try:
            roundto = -int(self.f(num2, n=n, fmt='sci')[-3:]) + n  # Will always be X.Xe+YY
            figs = int((np.floor(np.log10(abs(num)))))+roundto
        except (TypeError, ValueError, OverflowError):  # Something is inf or nan or None
            figs = 2
        else:
            figs = max(2, figs)
        return figs

    def f_array(self, arr, **kwargs):
        ''' Format all numbers in the 1D array with enough precision so they are distinct.

            Parameters
            ----------
            arr: array-like
                Numbers to convert to strings

            Returns
            -------
            List of strings

            Notes
            -----
            For example, the numbers 1232 and 1233, printed to the default 2 significant figs,
            would look identical ('1200'). Using f_array produces distinct output:

                >>> f(1232) == 1232
                >>> f(1233) == 1233
                >>> f_array([1232, 1233]) == ['1232', '1233']
        '''
        kwfmin = kwargs.pop('fmin', 0)
        try:
            arrmag = arr.magnitude
        except AttributeError:
            arrmag = arr  # not a pint quantity

        xdiff = abs(np.diff(sorted(np.asarray(arrmag, dtype=np.float64))))
        try:
            diffmin = xdiff[np.nonzero(xdiff)].min()
        except ValueError:
            fmin = 0   # All numbers are the same. Just use default sigfigs.
        else:
            try:
                fmin = max(kwfmin, -int((np.floor(np.log10(diffmin)))))
            except (OverflowError, ValueError):
                fmin = 0
        strings = [self.f(x, fmin=fmin, **kwargs) for x in arr]
        return strings


# Set up default format specs, override with output.formatter = NumFormatter(xxx)
# This is mostly for use in Jupyter so user doesn't have to specify **kwargs on every report() call.
# GUI will always pass args from GUI config manager.
formatter = NumFormatter()


def sympy_to_buf(expr, fname=None, color='black', fontsize=10, dpi=120):
    ''' Convert sympy expression to a png file buffer.

        Parameters
        ----------
        expr: sympy expression
            equation to convert
        fname: string (optional)
            If provided, filename to save the image to
        color: string
            Color for text foreground
        fontsize: float
            Font size

        Returns
        -------
        BytesIO object representing the image
    '''
    tex = sympy.latex(expr, mode='inline')
    return tex_to_buf(tex, fname=fname, color=color, fontsize=fontsize, dpi=dpi)


def tex_to_buf(tex, fname=None, color='black', fontsize=10, dpi=120):
    ''' Convert latex string expression to a png file buffer.

        Parameters
        ----------
        tex: latex expression
            equation to convert
        fname: string (optional)
            If provided, filename to save the image to
        color: string
            Color for text foreground
        fontsize: float
            Font size

        Returns
        -------
        BytesIO object representing the image
    '''
    if tex == '':
        return BytesIO()
    if not tex.startswith('$'):
        tex = '${}$'.format(tex)
    buf = BytesIO()
    with mpl.style.context({'savefig.edgecolor': (0, 0, 0, 0), 'savefig.facecolor': (0, 0, 0, 0)}):
        mpl.mathtext.MathTextParser('bitmap').to_png(buf, tex, color=color, dpi=dpi, fontsize=fontsize)
    buf.seek(0)
    if fname:
        with open(fname, 'wb') as f:
            f.write(buf.read())
            buf.seek(0)
    return buf


def fig_to_svg(fig, fname=None):
    ''' Convert matplotlib figure to an embedded html image tag using base64 encoding

        Parameters
        ----------
        fig: matplotlib figure
            The figure to embed
        fname: string (optional)
            Filename to save to

        Returns
        -------
        svg: string
            SVG format image, contained in <svg>...</svg> tags for use in html
    '''
    buf = BytesIO()
    fig.savefig(buf, bbox_inches='tight', format='svg')
    svg = buf.getvalue().decode('utf-8')
    svg = svg[svg.find('<svg'):]
    if fname:
        with open(fname, 'w') as f:
            f.write(svg)
    return svg


def fig_to_png(fig, fname=None):
    ''' Convert matplotlib figure to a png in base64.

        Parameters
        ----------
        fig: matplotlib figure
            Figure to convert
        fname: string, optional
            If provided, filename to save the image to

        Returns
        -------
        BytesIO, representing the image
    '''
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=150)
    buf.seek(0)
    if fname:
        with open(fname, 'wb') as f:
            f.write(buf.read())
            buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


def fig_to_txt(fig, char='.', H=18, W=55):
    ''' Extract data from matplotlib axis and plot as text. Each line
        in a figure will be plotted in a new text axis. Histogram
        plots will be deconstructed into text separately.

        Parameters
        ----------
        fig: matplotlib figure
            Figure to convert
        char: string character
            Character to use for point in plot

        Returns
        -------
        string: string representation of figure
    '''
    allplotstrs = []
    for ax in fig.axes:
        for line in ax.lines:
            x, y = line.get_data()
            label = line.get_label()
            plotstr = textplot(x, y, char=char)
            if not label.startswith('_'):
                plotstr = format(label, '^{}'.format(W+7)) + '\n' + plotstr
            allplotstrs.append(plotstr)

        if len(ax.patches) > 0:
            xmin, xmax = ax.get_xlim()
            ymin, ymax = ax.get_ylim()

            s = np.full((H, W), ' ')
            for x, p in enumerate(ax.patches):
                try:
                    ynorm = int((p.get_height())/ymax * (H-1))
                    xnorm1 = int((p.get_x()-xmin)/(xmax-xmin) * (W-1))
                    xnorm2 = xnorm1 + max(1, int(np.round(p.get_width()/xmax * (W-1))))
                except ValueError:
                    pass  # patch height is nan or ymax is nan...
                else:
                    for y in range(ynorm):
                        s[y][xnorm1:xnorm2] = char

            lines = [''.join(line) for line in s[::-1]]
            lines.append('-'*W)
            bottom = format(xmin, '.4g').ljust(W//2 - 3)
            bottom += format((xmin+xmax)/2, '.4g').ljust(W//2 - 3)
            bottom += format(xmax, '.4g')
            lines.append(bottom)
            label = ax.patches[0].get_label()
            plotstr = '\n'.join(lines)
            if not label.startswith('_'):
                plotstr = format(label, '^{}'.format(W)) + '\n' + plotstr

            allplotstrs.append(plotstr)

    return '\n\n\n'.join(allplotstrs) + '\n\n'


def textplot(xvals, yvals, W=55, H=18, char='.'):
    ''' Plot x, y data in plain-text format as scatter plot

        Parameters
        ----------
        xvals, yvals: arrays
            Arrays of x, y data to plot
        W: int
            Plot width in characters
        H: int
            Plot height in characters
        char: string character
            Character to use for a point in the plot

        Returns
        -------
        string: string representation of plot
    '''
    xvals = np.asarray(xvals)
    yvals = np.asarray(yvals)
    xmin, xmax = xvals.min(), xvals.max()
    ymin, ymax = yvals.min(), yvals.max()

    if not (np.isfinite(xmin) and np.isfinite(xmax) and np.isfinite(ymin) and np.isfinite(ymax)):
        xnorm = []
        ynorm = []
        xmin = ymin = 0
        xmax = ymax = 1
    else:
        xnorm = ((xvals-xmin)/(xmax-xmin) * (W-1)).astype(int)
        ynorm = ((yvals-ymin)/(ymax-ymin) * (H-1)).astype(int)

    if ymin == ymax:
        ynorm = np.ones(len(yvals), dtype=int)
    if xmin == xmax:
        xnorm = np.ones(len(xvals), dtype=int)

    s = np.full((H, W), ' ')
    for x, y in zip(xnorm, ynorm):
        s[y][x] = char

    margin = 7

    lines = []
    for h, line in enumerate(s[::-1]):
        if h == 0:  # Top
            prefix = format(ymax, '.4g').rjust(margin)[:margin]
        elif h == H//2:
            prefix = format((ymax+ymin)/2, '.4g').rjust(margin)[:margin]
        elif h == H-1:
            prefix = format(ymin, '.4g').rjust(margin)[:margin]
        else:
            prefix = ' ' * margin

        lines.append(prefix + '|' + ''.join(line))

    lines.append(' ' * margin + '-'*W)
    bottom = ' ' * (margin + 1)
    bottom += format(xmin, '.4g').ljust(W//2 - 3)
    bottom += format((xmin+xmax)/2, '.4g').ljust(W//2 - 3)
    bottom += format(xmax, '.4g')
    lines.append(bottom)
    return '\n'.join(lines)


def mpl_rendermath(md):
    ''' Use Matplotlib to render all equations (contained in $$) to markdown embedded images '''
    def repl(s):
        s = s.groups()[0]
        if s.startswith('$$'):
            s = s[1:-1]
        return eqn_to_mdimg(s)
    return re.sub(r'(\$.+?\$)', repl, md)


class MDstring(object):
    ''' String, but rendered as markdown in Jupyter, and convertible to html. '''
    extensions = ['markdown.extensions.tables']

    def __init__(self, s=''):
        self.string = str(s)
        self.figcnt = 0
        self.figlist = []

    def __repr__(self):
        return self.string

    def __str__(self):
        return self.__repr__()

    def __len__(self):
        return len(self.string)

    def _repr_markdown_(self):
        ''' Get Markdown representation, with embedded math and figures rendered using context '''
        return self.get_md()

    def get_md(self, unicode=True):
        ''' Get Markdown, with embedded math and figures rendered using context '''
        md = self.raw_md()
        if not unicode:
            md = md.encode('ascii', 'latex').decode()  # Convert unicode to latex-escaped symbols
        if _math_mode != 'mathjax' and _fig_mode != 'text':
            md = mpl_rendermath(md)   # Use MPL to render equations
        return md

    def raw_md(self):
        ''' Get Markdown, rendering figures appropriately, but math remains in $..$ '''
        md = self.string
        if self.figcnt > 0:
            # Convert MPL figures to appropriate format as referenced images
            md += '\n\n'
            for i, fig in enumerate(self.figlist):
                if _fig_mode == 'text':
                    md = md.replace('![][FIG{}]'.format(i), fig_to_txt(fig))
                else:
                    md += '\n[FIG{}]: '.format(i)
                    if _fig_mode == 'svg':
                        svg = fig_to_svg(fig)
                        md += 'data:image/svg+xml;base64,{}'.format(base64.b64encode(svg.encode('utf-8')).decode('utf-8'))
                    elif _fig_mode == 'png':
                        png = fig_to_png(fig)
                        md += 'data:image/png;base64,{}'.format(png)
                    else:
                        raise ValueError('Unknown fig_mode {}'.format(_fig_mode))
        return md

    def __add__(self, other):
        new = self.__class__()
        new.string = self.string
        new.figcnt = self.figcnt
        new.figlist = self.figlist.copy()
        if self.__class__ == other.__class__:
            # Concatenate MDString with MDString
            appendstring = other.string
            for i in range(other.figcnt):
                appendstring = appendstring.replace('[FIG{}]'.format(i), '[FIG{}]'.format(i+new.figcnt))
            new.string += appendstring
            new.figlist.extend(other.figlist)
            new.figcnt = len(new.figlist)
        else:
            # Concatenate plain string
            new.string += str(other)
        return new

    def __radd__(self, other):
        new = self.__class__(other)
        new += self
        return new

    def add_fig(self, fig, eol='\n\n'):
        ''' Add Matplotlib figure as reference, to be rendered in desired format later '''
        if _math_mode == 'text':
            self.string += fig_to_txt(fig)
        else:
            self.string += '![][FIG{}]'.format(self.figcnt) + eol
            self.figcnt += 1
            self.figlist.append(fig)

    def get_html(self):
        ''' Generate full HTML, including <head>, mathjax script, etc.

            Returns
            -------
            html: HTML as string
        '''
        s = self.get_md()
        CSS = '<style type="text/css">' + css.css + '</style>'

        if _math_mode == 'mathjax':
            # Use Mathjax header that includes $..$, not just default of only $$..$$
            CSS += r'''<script type="text/x-mathjax-config"> MathJax.Hub.Config({tex2jax: {inlineMath: [['$','$'], ['\\(','\\)']]}});</script>'''
            CSS += '\n' + r'''<script type="text/javascript" async src="''' + _mathjaxurl + '?config=TeX-AMS_CHTML"></script>'

        html = markdown.markdown(s, extensions=self.extensions)
        html = html.encode('ascii', 'xmlcharrefreplace').decode('utf-8')

        # Some table styles can't go in CSS, at least as rendered by QTextWidget, so must go in table tags
        html = html.replace('<table>', '<table border="0.5" cellpadding="0" cellspacing="0">')
        html = html.replace('<th>', '<th align="center" bgcolor="lightgray">')
        return CSS + '\n' + html

    def save_html(self, fname):
        ''' Save report to HTML. '''
        html = self.get_html()
        with open(fname, 'w') as f:
            f.write(html)
        return None  # No error

    def save_odt(self, fname):
        ''' Save report to Open Document (ODT) format. Requires Pandoc. '''
        if pandoc_path is None:
            raise ValueError('ODT format requires Pandoc.')

        if not fname.lower().endswith('.odt'):
            fname += '.odt'

        # ODT can nicely handle SVG images
        with report_format(math='mathjax', fig='svg'):
            p = subprocess.Popen([pandoc_path, '-o', fname], stdin=subprocess.PIPE, stderr=subprocess.PIPE, stdout=subprocess.PIPE)
            out, err = p.communicate(self.get_md().encode('utf-8'))
        return err.decode('utf-8')

    def save_docx(self, fname):
        ''' Save report to Word (docx) format. Requires Pandoc. '''
        if pandoc_path is None:
            raise ValueError('DOCX format requires Pandoc.')

        if not fname.lower().endswith('.docx'):
            fname += '.docx'

        # MSWord can't do SVG... gotta rasterize it
        with report_format(math='mathjax', fig='png'):
            p = subprocess.Popen([pandoc_path, '-o', fname], stdin=subprocess.PIPE, stderr=subprocess.PIPE, stdout=subprocess.PIPE)
            out, err = p.communicate(self.get_md().encode('utf-8'))
        return err.decode('utf-8')

    def save_tex(self, fname):
        ''' Save to raw LaTeX source '''
        if not fname.lower().endswith('.tex'):
            fname += '.tex'

        if pandoc_path is None:
            raise ValueError('TEX format requires Pandoc.')

        with report_format(math='mathjax', fig='png'):
            # NOTE: EPS format would be better but seems to be broken in Pandoc 2.7.
            # It extracts the eps file but omits the extension so pdflatex won't run.
            # SVG is not supported by pdflatex.

            # Convert utf8 markdown into plain ascii with latex codes for special characters
            # 'latex' handler was installed by importing latexchars.py
            md = self.get_md().encode('ascii', 'latex')

            fname = os.path.realpath(fname)
            filepath = os.path.dirname(fname)
            p = subprocess.Popen([pandoc_path, '--extract-media', 'images', '-s', '-o', fname], cwd=filepath, stdin=subprocess.PIPE, stderr=subprocess.PIPE, stdout=subprocess.PIPE)
            out, err = p.communicate(md)
        return err.decode('utf-8')

    def save_pdf(self, fname):
        ''' Save report to Word (docx) format. Requires Pandoc. '''
        if pandoc_path is None:
            raise ValueError('PDF format requires Pandoc.')
        if latex_path is None:
            raise ValueError('PDF format requires LaTeX')

        if not fname.lower().endswith('.pdf'):
            fname += '.pdf'

        with report_format(math='mathjax', fig='png'):
            # Convert utf8 markdown into plain ascii with latex codes for special characters
            # 'latex' handler was installed by importing latexchars.py
            md = self.get_md().encode('ascii', 'latex')

            # Must specify working dir for Popen because pandoc creates temp folder there.
            # Without it we get permission denied errors when running from app.
            fname = os.path.realpath(fname)
            filepath = os.path.dirname(fname)
            p = subprocess.Popen([pandoc_path, '-o', fname, '--pdf-engine={}'.format(latex_path)], cwd=filepath, stdin=subprocess.PIPE, stderr=subprocess.PIPE, stdout=subprocess.PIPE)
            out, err = p.communicate(md)
        return err.decode('utf-8')


def md_table(rows, hdr=None, **kwargs):
    ''' Convert list of rows into markdown-format table '''
    s = '\n'
    if not hdr:
        hdr = ['-'] * len(rows[0])  # PyMarkdown must have a header row, with non-empty strings

    widths = np.array([len(h) for h in hdr], dtype=int)
    for row in rows:
        widths = np.maximum(widths, np.array([len(c) for c in row]))
    widths = widths + 1

    s += '|'.join(['{{:{}}}'.format(w).format(h) for w, h in zip(widths, hdr)]) + '\n'
    s += '|'.join(['{}'.format(w*'-') for w in widths]) + '\n'
    for row in rows:
        s += '|'.join(['{{:{}}}'.format(w).format(r) for w, r in zip(widths, row)]) + '\n'

    return MDstring(s + '\n')


class Output(object):
    ''' Generic output object. Each calculation type (uncertainty, reverse, curvefit, etc)
        subclasses this and must implement the report() method. Subclasses may also implement
        get_dists() and get_dataset() to provide raw output data.
    '''
    def __str__(self):
        ''' String representation of output '''
        return str(self.report(figmode='text', mathmode='text'))

    def __repr__(self):
        return self.__str__()

    def _repr_markdown_(self):
        ''' Markdown representation for display in Jupyter '''
        return self.report().get_md()

    def report(self, **kwargs):
        ''' Generate markdown report. Wraps the local report output in
            MDString.

            Should return the essential numerical representation of the
            calculation results.

            Keyword Arguments
            -----------------
            figmode: string
                Mode for adding matplotlib figures to report: 'svg', 'png', 'text'
            mathmode: string
                Mode for adding math to report: 'latex', 'text', 'svg', 'png'

            Returns
            -------
            markdown string: MDstring
                Markdown in MDstring format (same as string with a Jupyter representation as
                formatted output)
        '''
        return MDstring(self._report(**kwargs))

    def report_summary(self, **kwargs):
        ''' Generate a summary report. Includes the typical information and plots. '''
        return MDstring(self.report(**kwargs))   # Local

    def report_all(self, **kwargs):
        ''' Generate a full report, including all results and plots. '''
        return MDstring(self.report(**kwargs))

    def get_dists(self, name=None, **kwargs):
        ''' Return a distribution from the output '''
        return None

    def get_dataset(self, name=None, **kwargs):
        ''' Return a DataSet from the output '''
        return None
