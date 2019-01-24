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

from . import css
from . import customdists


mpl.style.use('bmh')
sympy.printing.latex.__globals__['requires_partial'] = lambda x: True
sympy.printing.pretty.__globals__['requires_partial'] = lambda x: True

UDELTA = u'\u03B4'
UPLUSMINUS = u'\u00B1'

isWindows = os.name == 'nt'
use_unicode = not isWindows

pandoc_path = shutil.which('pandoc')
latex_path = shutil.which('pdflatex')

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


def sympyeqn(sym_expr):
    ''' Convert sympy to $latex$ or plain-text '''
    if _math_mode != 'text':
        return '${}$'.format(sympy.latex(sym_expr))
    else:
        return sympy.pretty(sym_expr, use_unicode=use_unicode)


def eqn_to_mdimg(expr, color='black', fontsize=18):
    ''' Convert tex equation into <img> tag using matplotlib and context figure mode (svg or png). '''
    if not expr.startswith('$'):
        expr = '${}$'.format(expr)
    buf = BytesIO()

    if _fig_mode == 'svg':
        prop = mpl.font_manager.FontProperties(size=fontsize)
        mpl.mathtext.math_to_image(expr, buf, prop=prop, format='svg')
        svg = buf.getvalue().decode('utf-8')
        svg = svg[svg.find('<svg'):]  # Strip HTML header stuff
        s = r"![](data:image/svg+xml;base64,{})".format(base64.b64encode(svg.encode('utf-8')).decode('utf-8'))
    elif _fig_mode == 'png':
        with mpl.style.context({'savefig.edgecolor': (0, 0, 0, 0), 'savefig.facecolor': (0, 0, 0, 0)}):
            mpl.mathtext.MathTextParser('bitmap').to_png(buf, expr, color=color, dpi=120, fontsize=fontsize)
        buf.seek(0)
        s = r"![](data:image/svg+xml;base64,{})".format(base64.b64encode(buf.read()).decode('utf-8'))
    return s


def sympy_to_svg(expr, color='black', fontsize=18):
    ''' Convert sympy expression into SVG (string) '''
    tex = '$' + sympy.latex(expr) + '$'
    buf = BytesIO()
    prop = mpl.font_manager.FontProperties(size=fontsize)
    mpl.mathtext.math_to_image(tex, buf, prop=prop, format='svg')
    svg = buf.getvalue().decode('utf-8')
    svg = svg[svg.find('<svg'):]  # Strip HTML header stuff
    return svg


def tex_to_html(expr):
    ''' Convert sympy expression to HTML <img> tag. '''
    md = eqn_to_mdimg(expr)
    return markdown.markdown(md)


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
    except AttributeError:
        pass
setup_mplparams()


def probplot(y, ax, sparams=(), dist='norm'):
    ''' Plot quantile probability plot. If data falls on straight line,
        data is normally distributed.

        Parameters
        ----------
        y: array
            Sampled data to fit
        ax: matplotlib axis
            Axis to plot on
        sparams: tuple
            Shape parameters for distribution. Omit for normal.
        dist: string
            Name of distribution to fit.
    '''
    (osm, osr), (slope, intercept, r) = stats.probplot(y, sparams=sparams, dist=dist)
    ax.plot(osm, osr, marker='o', ls='', label='Samples')
    xx = np.linspace(osm.min(), osm.max())
    ax.plot(xx, np.poly1d((slope, intercept))(xx), color='C1', label='Line Fit')
    ax.set_xlabel('Theoretical Quantiles ({})'.format(dist.name))
    ax.set_ylabel('Ordered Sample Values')
    ax.legend(loc='upper left')


def fitdist(y, dist='norm', fig=None, qqplot=False, bins='sqrt', points=None):
    ''' Fit a distribution to the data and plot comparison.

        Parameters
        ----------
        y: array
            1D Data to fit
        dist: string to rv_continuous
            Distribution to fit to
        fig: matplotlib figure
            Figure to plot comparison
        qqplot: boolean
            Plot a Q-Q normal probability plot
        bins: int
            Number of bins for histogram (see numpy.histogram_bin_edges).
            Defaults to square root of data size.
        points: int
            Number of points to show in Q-Q plot
    '''
    if fig is None:
        fig = plt.gcf()
    fig.clf()

    if dist:
        rv = customdists.get_dist(dist)
        fitparams = rv.fit(y)

    ax = fig.add_subplot(1, qqplot+1, 1)
    if not np.isfinite(y).any():
        return
    y = y[np.isfinite(y)]

    ax.hist(y, density=True, bins=bins, label='Samples')

    if dist:
        xx = np.linspace(y.min(), y.max(), num=100)
        yy = rv.pdf(xx, *fitparams)
        ax.plot(xx, yy, color='C1', label='{} Fit'.format(dist.title()))

    ax.set_xlabel('Parameter')
    ax.set_ylabel('Probability Density')
    ax.legend(loc='best')

    if dist:
        if qqplot:
            ax2 = fig.add_subplot(1, 2, 2)
            if points is None:
                points = min(100, len(y))
            thin = len(y)//points
            probplot(y[::thin], ax2, sparams=fitparams[:-2], dist=rv)  # Omit loc/scale to get quantiles

        paramnames = rv.shapes
        if paramnames is not None:
            paramnames = [r.strip() for r in paramnames.split(',')]  # .shapes returns a comma-sep string
            paramnames += ['loc', 'scale']
        else:
            paramnames = ['loc', 'scale']
        return dict(zip(paramnames, fitparams))


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
        echr = 'e' if elower else 'E'

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
        return numstr

    def matchprecision(self, num, num2, n=2):
        ''' Return number of sigfigs required to match num2 precision when printed to n figs.
            Can pass output of this to sympy's .n() method.

            Example: matchprecision(num=100, num2=0.123, n=2) == 5
            Need to print 100.00 (5 sigfigs) to match the 2 decimal places of 0.12 (num2 with 2 sigfigs)
            This takes care of any zeros at the end.
        '''
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
        xdiff = abs(np.diff(sorted(np.asarray(arr, dtype=np.float64))))
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


def sympy_to_buf(expr, fname=None, color='black', fontsize=12, dpi=120):
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
    buf = BytesIO()
    with mpl.style.context({'savefig.edgecolor': (0, 0, 0, 0), 'savefig.facecolor': (0, 0, 0, 0)}):
        mpl.mathtext.MathTextParser('bitmap').to_png(buf, sympy.latex(expr, mode='inline'),
                                                     color=color, dpi=dpi, fontsize=fontsize)
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
        s = self.string
        if not use_unicode:    # Stupid Windows can't handle unicode
            s = s.replace(UDELTA, 'd').replace(UPLUSMINUS, '+/-')
        return s

    def __str__(self):
        return self.__repr__()

    def __len__(self):
        return len(self.string)

    def _repr_markdown_(self):
        ''' Get Markdown representation, with embedded math and figures rendered using context '''
        return self.get_md()

    def get_md(self):
        ''' Get Markdown, with embedded math and figures rendered using context '''
        md = self.raw_md()
        if not use_unicode:  # Stupid Windows can't handle unicode
            md = md.replace(UDELTA, 'd')
            md = md.replace(UPLUSMINUS, '+/-')
        if _math_mode != 'mathjax':
            md = mpl_rendermath(md)   # Use MPL to render equations
        return md

    def raw_md(self):
        ''' Get Markdown, rendering figures appropriately, but math remains in $..$ '''
        md = self.string
        if self.figcnt > 0:
            # Convert MPL figures to appropriate format as referenced images
            md += '\n\n'
            for i, fig in enumerate(self.figlist):
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
        html = html.replace(UPLUSMINUS, '&plusmn;')
        html = html.replace(UDELTA, '&delta;')
        html = html.replace(' < ', '&lt;')
        html = html.replace(' > ', '&gt;')

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

    def save_pdf(self, fname):
        ''' Save report to Word (docx) format. Requires Pandoc. '''
        if pandoc_path is None:
            raise ValueError('PDF format requires Pandoc.')
        if latex_path is None:
            raise ValueError('PDF format requires LaTeX')

        if not fname.lower().endswith('.pdf'):
            fname += '.pdf'

        with report_format(math='mathjax', fig='png'):
            md = self.get_md()
            md = md.replace(UDELTA, r'$\delta$')  # Latex doesn't like these unicode chars
            md = md.replace(UPLUSMINUS, r'$\pm$')
            # Must specify working dir for Popen because pandoc creates temp folder there.
            # Without it we get permission denied errors when running from app.
            fname = os.path.realpath(fname)
            filepath = os.path.dirname(fname)
            p = subprocess.Popen([pandoc_path, '-o', fname, '--pdf-engine={}'.format(latex_path)], cwd=filepath, stdin=subprocess.PIPE, stderr=subprocess.PIPE, stdout=subprocess.PIPE)
            out, err = p.communicate(md.encode('utf-8'))
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
        get_distribution() and get_array() to provide raw output data.
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

    def get_array(self, name=None, **kwargs):
        ''' Return an array from the output '''
        return None
