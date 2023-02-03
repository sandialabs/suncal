''' Common functions for plotting results '''

from contextlib import suppress, contextmanager
import numpy as np
from scipy import stats
import matplotlib as mpl
import matplotlib.pyplot as plt

from ..common import distributions


# Common plot parameters, usage: "with mpl.style.context(plotstyle):"
plotstyle = {'figure.figsize': (8, 6), 'font.size': 12}
dfltsubplots = {'wspace': .1, 'hspace': .1, 'left': .05, 'right': .95, 'top': .95, 'bottom': .05}


# This unfortunately overwrites the user's MPL context, so user notebooks will end up with these params
# after importing suncal. But matplotlib contexts are broken such that certain parameters won't stay
# with the plot - see the "won't fix" bug report: https://github.com/matplotlib/matplotlib/issues/11376/
# meaning things like color='C0' and mathtext.fontset won't actually plot with their context-defined values.
mpl.style.use('bmh')


def setup_mplparams():
    ''' Set some default matplotlib parameters for things like fonts '''
    with suppress(AttributeError, KeyError):
        mpl.rcParams['figure.subplot.left'] = .12
        mpl.rcParams['figure.subplot.right'] = .95
        mpl.rcParams['figure.subplot.top'] = .95
        mpl.rcParams['figure.subplot.bottom'] = .12
        mpl.rcParams['figure.subplot.hspace'] = .4
        mpl.rcParams['figure.subplot.wspace'] = .4
        mpl.rcParams['figure.facecolor'] = 'none'
        mpl.rcParams['axes.formatter.use_mathtext'] = True
        mpl.rcParams['axes.formatter.limits'] = [-4, 4]
        mpl.rcParams['mathtext.fontset'] = 'stixsans'
        mpl.rcParams['mathtext.default'] = 'regular'
        mpl.rcParams['figure.max_open_warning'] = 0
        mpl.rcParams['axes.formatter.useoffset'] = False

        # v3.3 of Matplotlib made a really bad decision to change the epoch used by num2date
        # for plotting so that datetime.toordinal() does not give the same float date as num2date()
        # unless epoch is overridden like this. But it could be broken by the user if they
        # change rcParams after importing suncal. MPL < 3.3 will raise KeyError here, but it should
        # be suppressed above
        mpl.rcParams['date.epoch'] = '0000-12-31T00:00:00'


setup_mplparams()


class ReportPlot:
    ''' Context manager for adding figures to report. Ensures figure is closed
        so it doesn't display twice in Jupyter and is properly garbage collected.

        Use via plot_figure() function to chain with plt.style.context.
    '''
    def __enter__(self):
        self._fig = plt.figure()
        return self._fig

    def __exit__(self, exc_type, exc_val, exc_trace):
        plt.close(self._fig)


@contextmanager
def plot_figure():
    ''' Context manager for adding plots to Reports with the defined style.

        Usage:
            with plot_figure() as fig:
                ... # Plot stuff to figure
    '''
    with plt.style.context(plotstyle), ReportPlot() as fig:
        yield fig


def initplot(plot=None):
    ''' Initialize a Figure and Axis to plot on.

        Args:
            plot: plt.Figure, plt.Axis, or None. If None, new figure and
                axis will be created. If Figure or Axis, the Figure AND Axis
                will be returned.

        Returns:
            fig: plt.Figure instance
            ax: plt.Axis instance
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


def probplot(y, ax, sparams=None, dist='norm', conf=.95):
    ''' Plot quantile probability plot. If data falls on straight line,
        data is normally distributed.

        Args:
            y (array): Sampled data to fit
            ax (plt.Axis): Axis to plot on
            sparams (dictionary): Shape parameters for distribution.
            dist (string): Name of distribution to fit.
            conf (float): Level of confidence for confidence bands
    '''
    sparams = {} if sparams is None else sparams
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
    ax.set_xlabel(f'Theoretical Quantiles ({dist})')
    ax.set_ylabel('Ordered Sample Values')


def fitdist(y, distname='norm', fig=None, qqplot=False, bins='sqrt', points=None, coverage=None, xlabel='Parameter'):
    ''' Fit a distribution to the data and plot comparison.

        Args:
            y (array): 1D Data to fit
            distname (string): Name of distribution to fit to
            fig (plt.Figure): Figure to plot on (will be cleared)
            qqplot (boolean): Plot a Q-Q normal probability plot
            bins (int or string): Number of bins for histogram (see numpy.histogram_bin_edges).
                Defaults to square root of data size.
            points (int): Number of points to show in Q-Q plot
            coverage (array): List of coverage probabilities to plot as vertical lines
            xlabel (string): Label for x-axis
    '''
    fig, _ = initplot(fig)
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
        return None
    y = y[np.isfinite(y)]

    ax.hist(y, density=True, bins=bins)

    if distname:
        xx = np.linspace(y.min(), y.max(), num=100)
        yy = dist.pdf(xx)
        ax.plot(xx, yy, color='C1', label=f'{distname} Fit')

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


def equalize_scales(axis1, axis2):
    ''' Set axis scales equal

        Args:
            axis1: plt.Axis instance
            axis2: plt.Axis instance
    '''
    xlim1, xlim2 = axis1.get_xlim(), axis2.get_xlim()
    ylim1, ylim2 = axis1.get_ylim(), axis2.get_ylim()
    xlim = (min(xlim1[0], xlim2[0]), max(xlim1[1], xlim2[1]))
    ylim = (min(ylim1[0], ylim2[0]), max(ylim1[1], ylim2[1]))
    if np.isfinite(xlim[0]) and np.isfinite(xlim[1]):
        axis1.set_xlim(xlim)
        axis2.set_xlim(xlim)
    if np.isfinite(ylim[0]) and np.isfinite(ylim[1]):
        axis1.set_ylim(ylim)
        axis2.set_ylim(ylim)
