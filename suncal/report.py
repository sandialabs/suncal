import os
import re
import subprocess
from io import BytesIO
import base64
import shutil
from collections import ChainMap
from contextlib import suppress
import numpy as np
import sympy
import markdown
import matplotlib as mpl
import matplotlib.pyplot as plt

from . import unitmgr
from . import uparser
from . import css
from . import latexchars


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

# Default mathjax script for embedding in HTML header
_mathjaxurl = 'https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.4/MathJax.js'


# Defaults if kwargs aren't provided
default_sigfigs = 2
default_numformat = 'auto'
default_thresh = 5
default_E = True


def _matchprecision(num, num2, n=2):
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
        roundto = -int(str(Number(num2, n=n, fmt='sci'))[-3:]) + n  # Will always be X.Xe+YY
        figs = int((np.floor(np.log10(abs(num)))))+roundto
    except (TypeError, ValueError, OverflowError):  # Something is inf or nan or None
        figs = default_sigfigs
    else:
        figs = max(2, figs)
    return figs


def _scale_svg(svg, scale=1):
    ''' Change "width" and "height" parameters of an SVG string '''
    def _repl(match):
        ''' Regex replacement function '''
        size = match.group(2)
        num = ''.join(c for c in size if c in '0123456789.')
        unit = ''.join(c for c in size if c not in num)
        return match.group(1) + f'{float(num)*scale}{unit}"'
    svg = re.sub(r'(<svg.*width=")(\w.*?)"', _repl, svg, count=1)
    svg = re.sub(r'(<svg.*height=")(\w.*?)"', _repl, svg, count=1)
    return svg


class Number(object):
    ''' A formatted numeric value for use in a report

        Parameters
        ----------
        value: float
            The value to report

        Keyword Arguments
        -----------------
        n: int
            Number of significant figures
        fmt: string
            Format for the number - auto, decimal, scientific, engineering, si
        fmin: int
            Minimum number of decimal places, as override to n to prevent
            rounding too much.
        matchto: float
            First format the matchto number to n sig. figures, then return the
            value formatted to the same number of decimals. Typically matchto is
            an uncertainty and value is the measured value, per GUM guidance.
        matchtolim: int
            Maximum number of decimal places when matchto value is set
        thresh: int
            Exponent threshold for converting to scientific notation when in
            "auto" format. Numbers above 10**thresh will be printed in scientific
            notation.
        elower: bool
            Dispaly scientific notation with lowercase "e"
        unitfmt: string
            Format for printing a unit if value is a Pint Quantity - pretty,
            latex, html, plain
        abbr: bool
            Abbreviate the unit
        dimensionless: string
            Value to print when unit is dimensionless
        unitpad: string
            String to print between value and unit
    '''
    numfmts = ['auto', 'decimal', 'scientific', 'sci', 'engineering', 'eng', 'si']

    def __init__(self, value, **kwargs):
        self.value = value
        self.kwargs = kwargs

    def __str__(self):
        return self.string()

    def _repr_markdown_(self):
        ''' Markdown representation for Jupyter '''
        return self.string()

    def string(self, **kwargs):
        ''' Get string representation of the number.

            Keyword Arguments
            -----------------
            See Number keyword arguments. Anything defined in
            Number __init__ kwargs override the string() kwargs
        '''
        kargs = ChainMap(self.kwargs, kwargs)
        matchtolim = kargs.get('matchtolim', 10)
        figs = kargs.get('n', default_sigfigs)
        fmin = kargs.get('fmin', None)
        fmt = kargs.get('fmt', default_numformat).lower()
        thresh = kargs.get('thresh', default_thresh)
        elower = kargs.get('elower', default_E)
        matchtolim = kargs.get('matchtolim', 10)
        echr = 'e' if elower else 'E'

        if fmt not in self.numfmts:
            raise ValueError('Number Format must be one of {}'.format(', '.join(self.numfmts)))

        if figs < 1:
            raise ValueError('Significant Figures must be >= 1')

        if hasattr(self.value, 'units'):
            value, unit = self.value.magnitude, self.value.units
        else:
            value = self.value
            unit = None

        if 'matchto' in kargs:
            figs = min(matchtolim, _matchprecision(value, kargs['matchto'], figs))

        if value is None:
            numstr = 'nan'

        elif not np.isfinite(value):
            numstr = '{}'.format(value)  # Will format into 'nan' or 'inf'

        elif value == 0:
            if figs == 1:
                numstr = '0'
            else:
                numstr = '0.' + '0'*(figs-1)

            if fmt in ['sci', 'scientific', 'eng', 'engineering']:
                numstr = numstr + 'e+00'

        else:
            if fmt == 'auto':
                if abs(value) > 10**thresh or abs(value) < 10**-thresh:
                    fmt = 'sci'
                else:
                    fmt = 'decimal'

            exp = int(np.floor(np.log10(abs(value))))   # Exponent if written in exp. notation.
            roundto = -(exp - (figs-1))
            if fmin is not None:
                roundto = max(fmin, roundto)
                figs = roundto + exp + 1

            if fmt == 'decimal':
                numstr = '{{:.{}f}}'.format(max(0, roundto)).format(np.round(value, roundto))

            elif fmt == 'sci' or fmt == 'scientific':
                numstr = '{{:.{}{}}}'.format(figs-1, echr).format(value)

            elif fmt == 'eng' or fmt == 'engineering' or fmt == 'si':
                # REF: https://stackoverflow.com/a/40691220
                exp3 = exp - (exp % 3)  # Exponent as multiple of 3, for engineering notation
                value = value/(10**exp3)
                roundto = -int((np.floor(np.log10(abs(value)))) - (figs-1))
                value = np.round(value, roundto)
                if value == int(value):
                    value = int(value)  # Gets rid of extra .0

                if fmt == 'si' and exp3 >= -24 and exp3 <= 24:
                    suffix = 'yzafpnum kMGTPEZY'[exp3 // 3 + 8]
                    numstr = '{{:.{}f}}{{}}'.format(max(0, roundto)).format(value, suffix).rstrip()
                else:
                    numstr = '{{:.{}f}}{}{{:+03d}}'.format(max(0, roundto), echr).format(value, exp3)
            else:
                raise ValueError('Unknown format {}'.format(fmt))

        if unit:
            unitfmt = kargs.get('unitfmt', 'pretty')
            abbr = kargs.get('abbr', True)
            dimensionless = kargs.get('dimensionless', '')
            unitpad = kargs.get('unitpad', ' ')
            numstr += unitpad
            numstr += Unit(unit).string(unitfmt=unitfmt, abbr=abbr, dimensionless=dimensionless)
        return numstr

    @classmethod
    def number_array(cls, arr, **kwargs):
        ''' Return a list of Number objects with enough precision that they will
            print uniquely.

            Parameters
            ----------
            arr: array or list
                Array of numeric values

            Keyword Arguments
            -----------------
            Same as Number class
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
        numbers = [cls(x, fmin=fmin, **kwargs) for x in arr]
        return numbers


class Unit(object):
    ''' A formatted unit value to use in a report

        Parameters
        ----------
        unit: Pint unit
            A Pint unit to report

        Keyword Arguments
        -----------------
        abbr: bool
            Abbreviate the unit
        bracket: bool
            Enclose the unit in square brackets
            (typically for use in plot labels)
        dimensionless: string
            String to report when unit is dimensionless
        escape: bool
            Enclose the latex representation in $..$
    '''
    def __init__(self, unit, **kwargs):
        self.unit = unit
        self.kwargs = kwargs

    def __str__(self):
        return self.string()

    def _repr_markdown_(self):
        ''' Markdown representation for Jupyter '''
        return self.string()

    def string(self, **kwargs):
        ''' Get string representation of the unit

            Keyword Arguments
            -----------------
            unitfmt: string
                Format to return - latex, html, plain, pretty
        '''
        fmt = kwargs.get('unitfmt', kwargs.get('fmt', 'pretty'))
        if fmt == 'latex':
            return self.latex(**kwargs)
        elif fmt == 'html':
            return self.html(**kwargs)
        elif fmt == 'plain':
            return self.plaintext(**kwargs)
        else:
            return self.prettytext(**kwargs)

    def html(self, **kwargs):
        ''' Get HTML representation of the unit

            Keyword Arguments
            -----------------
            Same as Unit class. Arguments provided at class instantiation
            override arguments given here.
        '''
        kargs = ChainMap(self.kwargs, kwargs)
        abbr = kargs.get('abbr', True)
        bracket = kargs.get('bracket', False)  # Enclose in ' [ ]'
        dimensionless = kargs.get('dimensionless', '')

        if self.unit is None or unitmgr.is_dimensionless(self.unit):
            return dimensionless

        fmt = 'H'
        if abbr:
            fmt = '~' + fmt
        unitstr = format(self.unit, fmt)
        if bracket:
            unitstr = ' [{}]'.format(unitstr)
        return unitstr

    def latex(self, **kwargs):
        ''' Get Latex representation of the unit

            Keyword Arguments
            -----------------
            Same as Unit class. Arguments provided at class instantiation
            override arguments given here.
        '''
        kargs = ChainMap(self.kwargs, kwargs)
        abbr = kargs.get('abbr', True)
        bracket = kargs.get('bracket', False)  # Enclose in ' [ ]'
        dimensionless = kargs.get('dimensionless', '')
        escape = kargs.get('escape', True)  # Enclose in $..$

        if self.unit is None or unitmgr.is_dimensionless(self.unit):
            return dimensionless

        fmt = 'L'
        if abbr:
            fmt = '~' + fmt
        unitstr = format(self.unit, fmt).encode('ascii', 'latex').decode()
        # encode can leave $ in the string if units contain, e.g. /mu symbol
        # if the whole string is not already enclosed in $$.
        unitstr = unitstr.replace('$', ' ')
        if escape:
            unitstr = '${}$'.format(unitstr)
        if bracket:
            unitstr = ' [{}]'.format(unitstr)
        return unitstr

    def plaintext(self, **kwargs):
        ''' Get plain text (ascii) representation of the unit

            Keyword Arguments
            -----------------
            Same as Unit class. Arguments provided at class instantiation
            override arguments given here.
        '''
        kargs = ChainMap(self.kwargs, kwargs)
        abbr = kargs.get('abbr', True)
        bracket = kargs.get('bracket', False)  # Enclose in ' [ ]'
        dimensionless = kargs.get('dimensionless', '')
        if self.unit is None or unitmgr.is_dimensionless(self.unit):
            return dimensionless

        fmt = '~' if abbr else ''
        unitstr = format(self.unit, fmt)
        if bracket:
            unitstr = ' [{}]'.format(unitstr)
        return unitstr

    def prettytext(self, **kwargs):
        ''' Get pretty text representation of the unit (may contain
            unicode characters)

            Keyword Arguments
            -----------------
            Same as Unit class. Arguments provided at class instantiation
            override arguments given here.
        '''
        kargs = ChainMap(self.kwargs, kwargs)
        abbr = kargs.get('abbr', True)
        bracket = kargs.get('bracket', False)  # Enclose in ' [ ]'
        dimensionless = kargs.get('dimensionless', '')

        if self.unit is None or unitmgr.is_dimensionless(self.unit):
            return dimensionless

        fmt = 'P'
        if abbr:
            fmt = '~' + fmt
        unitstr = format(self.unit, fmt)
        if bracket:
            unitstr = ' [{}]'.format(unitstr)
        return unitstr


class Math(object):
    ''' A formatted mathematical expression for use in a report

        Parameters
        ----------
        expr: string
            The expression to print. Must be sympify-able string. Use
            `from_latex` or `from_sympy` to instantiate from other formats.
        unit: Pint unit (optional)
            Units to include at the end of the expression
    '''
    def __init__(self, expr=None, unit=None):
        # Use init when expr is string math expression (ie sympify-able)
        self.unit = unit
        if expr is not None:
            sympyexpr = uparser.parse_math(expr, raiseonerr=False)
            if sympyexpr:
                self.sympyexpr = sympyexpr
                self.latexexpr = sympy.latex(self.sympyexpr).encode('ascii', 'latex').decode()
                self.latexexpr = self.latexexpr.replace(r'\frac{d}{d ', r'\frac{\partial}{\partial ')  # All derivatives are partial
                self.prettytextexpr = sympy.pretty(self.sympyexpr)  # May use multiple lines for fractions, etc.
                self.simpletextexpr = str(self.sympyexpr)  # Typically the same as expr string
            else:
                self.sympyexpr = None
                self.latexexpr = expr
                self.prettytextexpr = expr
                self.simpletextexpr = expr

            if unit is not None:
                self.latexexpr = self.latexexpr + r'\,' + Unit(unit).latex(escape=False)
                self.prettytextexpr += Unit(unit).prettytext()
                self.simpletextexpr += Unit(unit).plaintext()
        else:
            self.sympyexpr = None
            self.latexexpr = None
            self.prettytextexpr = None
            self.simpletextexpr = None

    @classmethod
    def from_latex(cls, tex):
        ''' Create Math object from a latex expression. Pretty and Plain text outputs
            will not be available.
        '''
        math = cls()
        math.latexexpr = tex.lstrip('$').rstrip('$').encode('ascii', 'latex').decode()
        math.prettytextexpr = math.latexexpr
        math.simpletextexpr = math.latexexpr
        return math

    @classmethod
    def from_sympy(cls, expr, unit=None):
        ''' Create Math object from a sympy object. '''
        math = cls()
        math.sympyexpr = expr
        math.latexexpr = sympy.latex(expr).encode('ascii', 'latex').decode()
        math.latexexpr = math.latexexpr.replace(r'\frac{d}{d ', r'\frac{\partial}{\partial ')  # All derivatives are partial
        math.latexexpr = math.latexexpr.replace(r'\limits', '')  # MPL doesn't understand proper integration limits
        math.prettytextexpr = sympy.pretty(expr)
        math.simpletextexpr = str(expr)
        if unit is not None:
            math.latexexpr = math.latexexpr + r'\,' + Unit(unit).latex(escape=False)
            math.prettytextexpr += Unit(unit).prettytext()
            math.simpletextexpr += Unit(unit).plaintext()
        return math

    def __str__(self):
        ''' String representation of the math expression '''
        if self.simpletextexpr is not None:
            return self.simpletextexpr
        else:
            return str(self.latexexpr)

    def _repr_markdown_(self):
        ''' Markdown representation for Jupyter '''
        return self.latex()

    def png_buf(self, color='black', fontsize=16, dpi=120):
        ''' Render math to BytesIO buffer in PNG format using matplotlib

            Parameters
            ----------
            color: string
                Matplotlib-compatible color for text
            fontsize: int
                Font size
            dpi: int
                Dots per inch for PNG
        '''
        style = {'savefig.facecolor': (0, 0, 0, 0), 'savefig.edgecolor': (0, 0, 0, 0), 'text.color': mpl.rcParams['text.color']}
        if color:
            style['text.color'] = color
        buf = BytesIO()
        with mpl.style.context(style):
            mpl.mathtext.MathTextParser('bitmap').to_png(buf, self.latex(), color=style.get('text.color'), dpi=dpi, fontsize=fontsize)
        buf.seek(0)
        return buf

    def png_b64(self, color='black', fontsize=16, dpi=120):
        ''' Render math to base-64 encoded PNG

            Parameters
            ----------
            color: string
                Matplotlib-compatible color for text
            fontsize: int
                Font size
            dpi: int
                Dots per inch for PNG
        '''
        buf = self.png_buf(color=color, fontsize=fontsize, dpi=dpi)
        return 'data:image/png;base64,{}'.format(base64.b64encode(buf.read()).decode('utf-8'))

    def svg_buf(self, color='black', fontsize=16):
        ''' Render math to BytesIO buffer in SVG format using matplotlib

            Parameters
            ----------
            color: string
                Matplotlib-compatible color for text
            fontsize: int
                Font size
        '''
        buf = BytesIO()
        with mpl.style.context({'text.color': color, 'savefig.facecolor': (0, 0, 0, 0), 'savefig.edgecolor': (0, 0, 0, 0)}):
            prop = mpl.font_manager.FontProperties(size=fontsize)
            with suppress(ValueError):  # Double subscripts can raise here
                mpl.mathtext.math_to_image(self.latex(), buf, prop=prop, format='svg')
        buf.seek(0)
        return buf

    def svg_str(self, color='black', fontsize=16):
        ''' Render to SVG as string

            Parameters
            ----------
            color: string
                Matplotlib-compatible color for text
            fontsize: int
                Font size
        '''
        buf = self.svg_buf(color=color, fontsize=fontsize)
        svg = buf.getvalue().decode('utf-8')
        svg = svg[svg.find('<svg'):]  # Strip HTML header stuff
        return svg

    def svg_b64(self, color='black', fontsize=16):
        ''' Render base-64 encoded SVG, prefixed for use in Markdown or HTML
            image tag.

            Parameters
            ----------
            color: string
                Matplotlib-compatible color for text
            fontsize: int
                Font size
        '''
        buf = self.svg_buf(color=color, fontsize=fontsize)
        return 'data:image/svg+xml;base64,{}'.format(base64.b64encode(buf.read()).decode('utf-8'))

    def sympy(self):
        ''' Get sympy expression '''
        return self.sympyexpr

    def latex(self, enclose=('$', '$')):
        ''' Return math as latex-compatible string.

            Parameters
            ----------
            enclose: tuple
                Escape characters to enclose latex string
        '''
        if self.latexexpr:
            return enclose[0] + self.latexexpr + enclose[1]
        return ''

    def prettytext(self):
        ''' Return math as pretty-printed plain text. May contain unicode characters '''
        if self.prettytextexpr is None:
            # Math class created from latex directly can't convert back to sympy to plaintext.
            raise ValueError('No plain-text representation available for latex math "{}"'.format(self.latexexpr))
        return self.prettytextexpr

    def simpletext(self):
        ''' Return math as simple/plain text, ascii only characters '''
        if self.simpletextexpr is None:
            # Math class created from latex directly can't convert back to sympy to plaintext.
            raise ValueError('No plain-text representation available for latex math "{}"'.format(self.latexexpr))
        return self.simpletextexpr


class Plot(object):
    ''' A matplotlib figure for use in a report.

        Parameters
        ----------
        fig: matplotlib Figure
            The figure to format
    '''
    def __init__(self, fig=None):
        self.fig = fig  # MPL figure

    def __del__(self):
        plt.close(self.fig)

    def _repr_markdown_(self):
        ''' Markdown representation for Jupyter '''
        md = '![]({})'.format(self.svg_b64())
        return md

    def png_buf(self, dpi=120):
        ''' Render figure as BytesIO buffer in PNG format

            Parameters
            ----------
            dpi: int
                Dots per inch for PNG
        '''
        buf = BytesIO()
        self.fig.savefig(buf, format='png', dpi=dpi)
        buf.seek(0)
        return buf

    def png_b64(self, dpi=120):
        ''' Render to base-64 encoded PNG

            Parameters
            ----------
            dpi: int
                Dots per inch for PNG
        '''
        buf = self.png_buf(dpi=dpi)
        return 'data:image/png;base64,{}'.format(base64.b64encode(buf.read()).decode('utf-8'))

    def svg_buf(self, scale=1):
        ''' Render to BytesIO buffer in SVG format '''
        buf = BytesIO()
        self.fig.savefig(buf, bbox_inches='tight', format='svg')
        svg = buf.getvalue().decode('utf-8')
        svg = svg[svg.find('<svg'):]  # Strip HTML header stuff
        if scale != 1:
            svg = _scale_svg(svg, scale)
        buf = BytesIO(svg.encode())
        buf.seek(0)
        return buf

    def svg_str(self):
        ''' Render to SVG string '''
        buf = self.svg_buf()
        svg = buf.getvalue().decode('utf-8')
        return svg

    def svg_b64(self):
        ''' Render base-64 encoded SVG string, prefixed for use in markdown or html image tag '''
        buf = self.svg_buf()
        return 'data:image/svg+xml;base64,{}'.format(base64.b64encode(buf.read()).decode('utf-8'))

    def textplot(self, char='.', H=18, W=55):
        ''' Plot the figure as plain text/ascii

            Parameters
            ----------
            char: string
                Character to print for each data point
            H: int
                Character height of plot
            W: int
                Character width of plot
        '''
        allplotstrs = []
        for ax in self.fig.axes:
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
                    with suppress(ValueError):
                        # ValueError when patch height is nan or ymax is nan, ignore
                        ynorm = int((p.get_height())/ymax * (H-1))
                        xnorm1 = int((p.get_x()-xmin)/(xmax-xmin) * (W-1))
                        xnorm2 = xnorm1 + max(1, int(np.round(p.get_width()/xmax * (W-1))))
                        for y in range(ynorm):
                            s[y][xnorm1:xnorm2] = char

                lines = [''.join(line) for line in reversed(s)]
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
    for h, line in enumerate(reversed(s)):
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


class Report(object):
    ''' A Report consisting of text, plots, math equations, and values for
        formatting in different formats.

        Keyword Arguments
        -----------------
        mathfmt: string
            Format for math expressions - latex, pretty, text, png, svg.
        mathdelim: tuple
            Delimiter/escape characters for math expressions, typically ('$', '$').
        figfmt: string
            Format for matplotlib figures - svg or png
        pngdpi: int
            Dots per inch for PNG images
        inline: bool
            Render Mardkown images inline (True) or as references in footer
        unicode: bool
            Allow unicode characters (True) or only ascii (False)
    '''
    def __init__(self, **kwargs):
        self._s = ''
        self._plots = []
        self._values = []
        self._eqns = []
        self._units = []
        self.kwargs = kwargs

    def __str__(self):
        return self.get_md()

    def _repr_markdown_(self):
        ''' Markdown representation for Jupyter '''
        return self.get_md()

    def hdr(self, text, level=1):
        ''' Add a header to the report

            Parameters
            ----------
            text: string
                Text of the header
            level: int
                Header level. 1 is top level (# HEADER) in markdown.
        '''
        self._s += '{} {}\n\n'.format('#'*level, text)

    def txt(self, text):
        ''' Add text to the report '''
        self._s += text

    def div(self):
        ''' Add a horizontal divider to the report '''
        self._s += '\n---\n\n'

    def sympy(self, sympyexpr, end=''):
        ''' Add a sympy expression to the report

            Parameters
            ----------
            sympyexpr: Sympy
                Sympy equation to add
            end: string
                Characters to print (such as newline) after the expression
        '''
        self._s += self._insert_obj(Math.from_sympy(sympyexpr), end=end)

    def mathexpr(self, mathstr, end=''):
        ''' Add a math expression to the report

            Parameters
            ----------
            mathstr: string
                Math expression, must be sympify-able
            end: string
                Characters to print (such as newline) after the expression
        '''
        self._s += self._insert_obj(Math(mathstr), end=end)

    def plot(self, fig, end='\n\n'):
        ''' Add matplotlib figure to the report

            Parameters
            ----------
            fig: matplotlib Figure
                Figure to add
            end: string
                Characters to print (such as newline) after the expression
        '''
        self._s += self._insert_obj(Plot(fig), end=end)

    def unit(self, unit, end='', **kwargs):
        ''' Add a Pint unit to the report

            Parameters
            ----------
            unit: Pint object
                Unit to represent
            end: string
                Characters to print after the unit

            Keyword Arguments
            -----------------
            See Unit class
        '''
        self._s += self._insert_obj(Unit(unit, **kwargs), end=end)

    def num(self, value, end='', **kwargs):
        ''' Add a Numeric value to the report

            Parameters
            ----------
            value: float
                Value to represent
            end: string
                Characters to print after the unit

            Keyword Arguments
            -----------------
            See Number class
        '''
        self._s += self._insert_obj(Number(value, **kwargs), end=end)

    def _insert_obj(self, obj, end=''):
        ''' Insert an object and return the string to add to _s '''
        if isinstance(obj, Number):
            # NOTE: putting a Number inside brackets won't work...
            s = '[[VAL{}]]{}'.format(len(self._values), end)
            self._values.append(obj)
        elif isinstance(obj, Math):
            s = '[[EQN{}]]{}'.format(len(self._eqns), end)
            self._eqns.append(obj)
        elif isinstance(obj, sympy.Basic):
            s = '[[EQN{}]]{}'.format(len(self._eqns), end)
            self._eqns.append(Math.from_sympy(obj))
        elif isinstance(obj, Unit):
            s = '[[UNT{}]]{}'.format(len(self._units), end)
            self._units.append(obj)
        elif isinstance(obj, Plot):
            s = '[[PLT{}]]{}'.format(len(self._plots), end)
            self._plots.append(obj)
        elif isinstance(obj, (list, tuple)):
            s = ''
            for childobject in obj:
                s += self._insert_obj(childobject)
            s += end
        else:  # Text
            s = '{}{}'.format(obj, end)
        return s

    def table(self, rows, hdr):
        ''' Add a table to the report

            Parameters
            ----------
            rows: list
                List of lists for each row. Each list item may be a
                string, Number, Math, or a tuple containing multiple string,
                Number, and Math objects for the table cell.
            hdr: list
                List of items for the table header. Each item may be
                string, Number, or Math, or a tuple of these for each
                header cell.
        '''
        s = '\n'
        if hdr is None:
            hdr = ['-'] * len(rows[0])  # PyMarkdown must have a header row, with non-empty strings

        widths = np.array([len(str(h))+1 for h in hdr], dtype=int)
        for row in rows:
            widths = np.maximum(widths, np.array([len(c) if hasattr(c, '__len__') else 1 for c in row]))
        widths = widths + 1
        widths = np.maximum(widths, 9)

        # Header
        line = []
        for col in hdr:
            line.append(self._insert_obj(col))
        s += ' | '.join('{{:{}}}'.format(w).format(val) for w, val in zip(widths, line)) + '\n'
        s += '|'.join('{}'.format(w*'-') for w in widths) + '\n'
        for row in rows:
            line = []
            for col in row:
                line.append(self._insert_obj(col))
            s += ' | '.join('{{:{}}}'.format(w).format(val) for w, val in zip(widths, line)) + '\n'

        # Add | at beginning and end
        lines = ''
        for line in s.splitlines():
            lines += (('|' + line + '|\n') if len(line) > 0 else '\n')
        s = lines

        self._s += s + '\n\n'

    def add(self, *args, end='\n'):
        ''' Add multiple items to the report

            Parameters
            ----------
            args: objects
                Each arg may be a string, Number, Math, Plot, or sympy expression
        '''
        for arg in args:
            self._s += self._insert_obj(arg, end=end)

    def append(self, report, end=''):
        ''' Append another report onto this one '''
        appendstring = report._s

        # Go backwards through the tagged objects to renumber them
        for i in range(len(report._plots)-1, -1, -1):
            appendstring = appendstring.replace('[[PLT{}]]'.format(i), '[[PLT{}]]'.format(i+len(self._plots)))
        for i in range(len(report._eqns)-1, -1, -1):
            appendstring = appendstring.replace('[[EQN{}]]'.format(i), '[[EQN{}]]'.format(i+len(self._eqns)))
        for i in range(len(report._values)-1, -1, -1):
            appendstring = appendstring.replace('[[VAL{}]]'.format(i), '[[VAL{}]]'.format(i+len(self._values)))
        for i in range(len(report._units)-1, -1, -1):
            appendstring = appendstring.replace('[[UNT{}]]'.format(i), '[[UNT{}]]'.format(i+len(self._units)))
        self._plots.extend(report._plots)
        self._eqns.extend(report._eqns)
        self._values.extend(report._values)
        self._units.extend(report._units)
        self._s += appendstring
        self._s += end

    def get_md(self, **kwargs):
        ''' Get the report in markdown format.

            Keyword Arguments
            -----------------
            See Report class. Arguments specified at Report instantiation
            override arguments given here.
        '''
        kargs = ChainMap(self.kwargs, kwargs)
        mathfmt = kargs.get('mathfmt', 'latex')  # Latex, pretty, text, png, svg (png and svg are matplotlib-rendered)
        mathdelim = kargs.get('mathdelim', ('$', '$'))   # Math delimiter
        figfmt = kargs.get('figfmt', 'svg')      # svg, png
        pngdpi = kargs.get('pngdpi', 120)        # dpi for png images
        inline = kargs.get('inline', False)   # Show images inline or as references in footer
        allowunicode = kargs.get('unicode', True)

        footer = '\n\n'
        imagecnt = 0
        s = self._s
        start = s.find('[[')
        while start > -1:
            end = s.find(']]') + 2
            tag = s[start:end]

            obj = tag[2:5]  # Skip [[ and 3-letter type designator
            idx = int(tag[5:-2])  # omit ]]

            if obj == 'EQN':
                eqn = self._eqns[idx]

                if mathfmt == 'latex':
                    formattedmath = eqn.latex(enclose=mathdelim)
                elif mathfmt == 'svg':
                    if inline:
                        formattedmath = '![]({})'.format(eqn.svg_b64())
                    else:
                        formattedmath = '![IMG{}][]'.format(imagecnt)
                        footer += '[IMG{}]: {}\n'.format(imagecnt, eqn.svg_b64())
                        imagecnt += 1
                elif mathfmt == 'png':
                    if inline:
                        formattedmath = '![]({})'.format(eqn.png_b64(dpi=pngdpi))
                    else:
                        formattedmath = '![IMG{}][]'.format(imagecnt)
                        footer += '[IMG{}]: {}\n'.format(imagecnt, eqn.png_b64(dpi=pngdpi))
                        imagecnt += 1
                elif mathfmt == 'ascii':
                    formattedmath = eqn.simpletext()
                else: # mathfmt in ['text', 'txt']:
                    formattedmath = eqn.prettytext()

                s = s.replace(tag, formattedmath)

            elif obj == 'VAL':
                val = self._values[idx]
                s = s.replace(tag, val.string(**kargs))

            elif obj == 'UNT':
                unit = self._units[idx]
                s = s.replace(tag, unit.string(**kargs))

            elif obj == 'PLT':
                p = self._plots[idx]
                if figfmt in ['text', 'txt']:
                    pstr = p.textplot(char='o')
                elif inline:
                    if figfmt == 'svg':
                        pstr = '![]({})'.format(p.svg_b64())
                    else:  # png
                        pstr = '![]({})'.format(p.png_b64(dpi=pngdpi))
                else:
                    pstr = '![IMG{}][]\n\n'.format(imagecnt)
                    if figfmt == 'svg':
                        footer += '[IMG{}]: {}\n'.format(imagecnt, p.svg_b64())
                    else:
                        footer += '[IMG{}]: {}\n'.format(imagecnt, p.png_b64(dpi=pngdpi))

                s = s.replace(tag, pstr)
                imagecnt += 1

            else:
                raise

            start = s.find('[[')
        s += footer

        if not allowunicode:
            # Unicode not allowed, convert to latex-escaped symbols
            s = s.encode('ascii', 'latex').decode('utf-8')

        return s.strip()

    def get_html(self, **kwargs):
        ''' Get report in HTML format, including CSS and mathjax header if needed.

            Keyword Arguments
            -----------------
            See Report class. Arguments specified at Report instantiation
            override arguments given here.
        '''
        CSS = '<style type="text/css">' + css.css + '</style>'

        if kwargs.get('mathfmt', 'latex') == 'latex':
            # Include Mathjax script
            # Use Mathjax header that includes $..$, not just default of only $$..$$
            CSS += r'''<script type="text/x-mathjax-config"> MathJax.Hub.Config({tex2jax: {inlineMath: [['$','$'], ['\\(','\\)']]}});</script>'''
            CSS += '\n' + r'''<script type="text/javascript" async src="''' + _mathjaxurl + '?config=TeX-AMS_CHTML"></script>'

        # Convert markdown to HTML
        html = markdown.markdown(self.get_md(**kwargs), extensions=['markdown.extensions.tables'])
        html = html.encode('ascii', 'xmlcharrefreplace').decode('utf-8')

        # Some table styles can't go in CSS, at least as rendered by QTextWidget, so must go in table tags
        html = html.replace('<table>', '<table border="0.5" cellpadding="0" cellspacing="0">')
        html = html.replace('<th>', '<th align="center" bgcolor="lightgray">')
        return CSS + '\n' + html

    def save_html(self, fname, **kwargs):
        ''' Get report in HTML format and save to file.

            Parameters
            ----------
            fname: string
                File name to save

            Keyword Arguments
            -----------------
            See Report class. Arguments specified at Report instantiation
            override arguments given here.
        '''
        if not fname.lower().endswith('.html') or fname.endswith('.htm'):
            fname += '.html'
        html = self.get_html(**kwargs)
        with open(fname, 'w') as f:
            f.write(html)
        return None  # No error

    def save_odt(self, fname, **kwargs):
        ''' Save report to Open Document (ODT) format. Requires Pandoc. '''
        if pandoc_path is None:
            raise ValueError('ODT format requires Pandoc.')

        if not fname.lower().endswith('.odt'):
            fname += '.odt'

        md = self.get_md(inline=True, mathfmt='latex', figfmt='svg', **kwargs)
        # ODT can nicely handle SVG images
        p = subprocess.Popen([pandoc_path, '-o', fname], stdin=subprocess.PIPE, stderr=subprocess.PIPE, stdout=subprocess.PIPE)
        out, err = p.communicate(md.encode('utf-8'))
        return err.decode('utf-8')

    def save_docx(self, fname, **kwargs):
        ''' Save report to Word (docx) format. Requires Pandoc. '''
        if pandoc_path is None:
            raise ValueError('DOCX format requires Pandoc.')

        if not fname.lower().endswith('.docx'):
            fname += '.docx'

        md = self.get_md(mathfmt='latex', figfmt='png', inline=True, **kwargs)

        # MSWord can't do SVG... gotta rasterize it
        p = subprocess.Popen([pandoc_path, '-o', fname], stdin=subprocess.PIPE, stderr=subprocess.PIPE, stdout=subprocess.PIPE)
        out, err = p.communicate(md.encode('utf-8'))
        return err.decode('utf-8')

    def save_tex(self, fname, **kwargs):
        ''' Save report as tex file for later processing with Latex. Requires Pandoc. '''
        if not fname.lower().endswith('.tex'):
            fname += '.tex'

        if pandoc_path is None:
            raise ValueError('TEX format requires Pandoc.')

        # NOTE: EPS format would be better but seems to be broken in Pandoc 2.7.
        # It extracts the eps file but omits the extension so pdflatex won't run.
        # SVG is not supported by pdflatex.

        # Convert utf8 markdown into plain ascii with latex codes for special characters
        # 'latex' handler was installed by importing latexchars.py
        md = self.get_md(mathfmt='latex', figfmt='png', inline=True, **kwargs).encode('ascii', 'latex')
        fname = os.path.realpath(fname)
        filepath = os.path.dirname(fname)
        p = subprocess.Popen([pandoc_path, '--extract-media', 'images', '-s', '-o', fname], cwd=filepath, stdin=subprocess.PIPE, stderr=subprocess.PIPE, stdout=subprocess.PIPE)
        out, err = p.communicate(md)
        return err.decode('utf-8')

    def save_pdf(self, fname, **kwargs):
        ''' Save report to PDF format. Requires Pandoc and LaTeX. '''
        if pandoc_path is None:
            raise ValueError('PDF format requires Pandoc.')
        if latex_path is None:
            raise ValueError('PDF format requires LaTeX')

        if not fname.lower().endswith('.pdf'):
            fname += '.pdf'

        # Convert utf8 markdown into plain ascii with latex codes for special characters
        # 'latex' handler was installed by importing latexchars.py
        md = self.get_md(mathfmt='latex', figfmt='png', inline=True, **kwargs).encode('ascii', 'latex')

        # Must specify working dir for Popen because pandoc creates temp folder there.
        # Without it we get permission denied errors when running from app.
        fname = os.path.realpath(fname)
        filepath = os.path.dirname(fname)
        p = subprocess.Popen([pandoc_path, '-o', fname, '--pdf-engine={}'.format(latex_path)], cwd=filepath, stdin=subprocess.PIPE, stderr=subprocess.PIPE, stdout=subprocess.PIPE)
        out, err = p.communicate(md)
        return err.decode('utf-8')
