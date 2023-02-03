''' Report of complex-number uncertainty calculations '''

import numpy as np
import matplotlib.pyplot as plt

from ...common import report, unitmgr
from .uncertainty import ReportUncertainty
from .gum import _contour


class ReportComplexGum:
    ''' Reports of Complex-value GUM calculation '''
    def __init__(self, gumresults):
        self._results = gumresults
        self.magphase = any(['_mag' in x for x in self._results.functionnames])
        self._functionnames = list(set(
            [name.removesuffix('_real').removesuffix('_imag').removesuffix('_mag')
             .removesuffix('_rad') for name in self._results.functionnames]))
        self._variablenames = list(set(
            [name.removesuffix('_real').removesuffix('_imag').removesuffix('_mag').removesuffix('_rad')
             .removesuffix('_deg') for name in self._results.variablenames]))

    def summary(self, **kwargs):
        ''' Generate report of complex values and uncertainties '''
        hdr = ['Function', 'Nominal', 'Std. Uncertainty', 'Correlation']
        deg = '°' if self._results._degrees else ' rad'
        rows = []
        for fname in self._functionnames:
            if self.magphase:
                cor = self._results.correlation()[f'{fname}_mag'][f'{fname}_rad']
                mag = self._results.expected[f'{fname}_mag']
                ph = self._results.expected[f'{fname}_rad']
                umag = self._results.uncertainty[f'{fname}_mag']
                uph = self._results.uncertainty[f'{fname}_rad']
                if self._results._degrees:
                    ph, uph = np.rad2deg(ph), np.rad2deg(uph)
                mag, ph = unitmgr.strip_units(mag), unitmgr.strip_units(ph)
                umag, uph = unitmgr.strip_units(umag), unitmgr.strip_units(uph)
                rows.append([fname,
                             f'{report.Number(mag, matchto=umag)} ∠{report.Number(ph, matchto=uph)}{deg}',
                             f'± {report.Number(umag)} ∠{report.Number(uph)}{deg}',
                             f'{cor:.4f}'])
            else:
                cor = self._results.correlation()[f'{fname}_real'][f'{fname}_imag']
                real = self._results.expected[f'{fname}_real']
                imag = self._results.expected[f'{fname}_imag']
                ureal = self._results.uncertainty[f'{fname}_real']
                uimag = self._results.uncertainty[f'{fname}_imag']
                rows.append([fname,
                             f'{report.Number(real + 1j*imag, matchto=ureal)}',
                             f'± {report.Number(ureal + 1j*uimag)}',
                             f'{cor:.4f}'])

        r = report.Report(**kwargs)
        r.table(rows, hdr=hdr)
        return r

    def plot(self, funcname=None, ax=None, polar=True, contour=True, cmap='hot', color='C3'):
        ''' Plot the uncertainty region on polar or rectangular axis

            Args:
                funcname: Name of function to plot
                ax (plt.axis): matplotlib axis to plot on
                polar (bool): Show plot in polar format
                contour (bool): Draw uncertainty region with contour lines
                cmap (string): Name of Matplotlib colormap for contour lines
                color (string): Name of color for shaded region (when contour = False)
        '''
        if funcname is None:
            funcname = self._functionnames[0]

        if ax is None:
            fig = plt.gcf()
            ax = fig.add_subplot(111, polar=polar)

        if self.magphase:
            fname1 = f'{funcname}_mag'
            fname2 = f'{funcname}_rad'
        else:
            fname1 = f'{funcname}_real'
            fname2 = f'{funcname}_imag'

        if contour:
            # Draw contoured uncert region
            x, y, p = _contour(self._results.expected, self._results.covariance(), fname1, fname2)
            xunc, yunc = 0, 0
        else:
            x, y = self._results.expected[fname1], self._results.expected[fname2]
            expand1 = self._results.expand(fname1)
            expand2 = self._results.expand(fname2)
            xunc, yunc = unitmgr.strip_units(expand1), unitmgr.strip_units(expand2)
            x, y = unitmgr.strip_units(x), unitmgr.strip_units(y)

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
            ax.set_xlabel(f'Re({funcname})')
            ax.set_ylabel(f'Im({funcname})')


class ReportComplexMc:
    ''' Reports of Complex-value Monte Carlo calculation '''
    def __init__(self, mcresults):
        self._results = mcresults
        self.magphase = any(['_mag' in x for x in self._results.functionnames])
        self._functionnames = list(set(
            name.removesuffix('_real').removesuffix('_imag').removesuffix('_mag')
            .removesuffix('_rad') for name in self._results.functionnames))
        self._variablenames = list(set(
            name.removesuffix('_real').removesuffix('_imag').removesuffix('_mag').removesuffix('_rad')
            .removesuffix('_deg') for name in self._results.variablenames))

    def summary(self, **kwargs):
        ''' Generate report of complex values and uncertainties '''
        hdr = ['Function', 'Nominal', 'Std. Uncertainty', 'Correlation']
        deg = '°' if self._results._degrees else ' rad'
        rows = []
        for fname in self._functionnames:
            if self.magphase:
                cor = self._results.correlation()[f'{fname}_mag'][f'{fname}_rad']
                mag = self._results.expected[f'{fname}_mag']
                ph = self._results.expected[f'{fname}_rad']
                umag = self._results.uncertainty[f'{fname}_mag']
                uph = self._results.uncertainty[f'{fname}_rad']
                if self._results._degrees:
                    ph, uph = np.rad2deg(ph), np.rad2deg(uph)
                mag, ph = unitmgr.strip_units(mag), unitmgr.strip_units(ph)
                umag, uph = unitmgr.strip_units(umag), unitmgr.strip_units(uph)
                rows.append([fname,
                             f'{report.Number(mag, matchto=umag)} ∠{report.Number(ph, matchto=uph)}{deg}',
                             f'± {report.Number(umag)} ∠{report.Number(uph)}{deg}',
                             f'{cor:.4f}'])
            else:
                cor = self._results.correlation()[f'{fname}_real'][f'{fname}_imag']
                real = self._results.expected[f'{fname}_real']
                imag = self._results.expected[f'{fname}_imag']
                ureal = self._results.uncertainty[f'{fname}_real']
                uimag = self._results.uncertainty[f'{fname}_imag']
                rows.append([fname,
                             f'{report.Number(real + 1j*imag, matchto=ureal)}',
                             f'± {report.Number(ureal + 1j*uimag)}',
                             f'{cor:.4f}'])

        r = report.Report(**kwargs)
        r.table(rows, hdr=hdr)
        return r

    def plot(self, funcname=None, ax=None, polar=True, points=5000, bins=35, contour=False, cmap='viridis', color='C2'):
        ''' Plot scatterplot of samples or contours of uncertainty region
            on polar or rectangular axis

            Args:
                funcname: Name of function to plot
                ax (plt.axis): matplotlib axis to plot on
                polar (bool): Show plot in polar format
                points: Number of samples to include
                bins (int): Number of bins for forming contour plot
                contour (bool): Draw uncertainty region with contour lines
                cmap (string): Name of Matplotlib colormap for contour lines
                color (string): Name of color for shaded region (when contour = False)
        '''
        if funcname is None:
            funcname = self._functionnames[0]

        if self.magphase:
            fname1 = f'{funcname}_mag'
            fname2 = f'{funcname}_rad'
        else:
            fname1 = f'{funcname}_real'
            fname2 = f'{funcname}_imag'

        if ax is None:
            fig = plt.gcf()
            ax = fig.add_subplot(111, polar=polar)

        samplesx = unitmgr.strip_units(self._results.samples[fname1])
        samplesy = unitmgr.strip_units(self._results.samples[fname2])

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
            ax.set_xlabel(f'Re({funcname})')
            ax.set_ylabel(f'Im({funcname})')


class ReportComplex:
    ''' Reports of Complex-value GUM and Monte Carlo calculation '''
    def __init__(self, results):
        self._results = results
        self.fullreport = self._results.componentresults.report
        self.magphase = any(['_mag' in x for x in self._results.gum.functionnames])
        self._functionnames = list(set(
            name.removesuffix('_real').removesuffix('_imag').removesuffix('_mag')
            .removesuffix('_rad') for name in self._results.gum.functionnames))
        self._variablenames = list(set(
            name.removesuffix('_real').removesuffix('_imag').removesuffix('_mag').removesuffix('_rad')
            .removesuffix('_deg') for name in self._results.gum.variablenames))

    def _repr_markdown_(self):
        return self.summary().get_md()

    def summary(self, **kwargs):
        ''' Report the results '''
        hdr = ['Function', 'Method', 'Nominal', 'Standard Uncertainty', 'Correlation']
        rpt = report.Report(**kwargs)
        deg = '°' if self._results._degrees else ' rad'

        def _addcol(out, fname):
            if self.magphase:
                fname1 = f'{fname}_mag'
                fname2 = f'{fname}_rad'
                cor = out.correlation()[fname1][fname2]
                if hasattr(out, 'numeric'):
                    out = out.numeric
                mag = out.expected[fname1]
                umag = out.uncertainty[fname1]
                ph = unitmgr.strip_units(out.expected[fname2])
                uph = unitmgr.strip_units(out.uncertainty[fname2])
                if self._results.degrees:
                    ph, uph = np.rad2deg(ph), np.rad2deg(uph)

                cols = [f'{report.Number(mag, matchto=umag)} ∠{report.Number(ph, matchto=uph)}{deg}',
                        f'± {report.Number(umag)} ∠{report.Number(uph)}{deg}',
                        f'{cor:.4f}']
            else:
                fname1 = f'{fname}_real'
                fname2 = f'{fname}_imag'
                cor = out.correlation()[fname1][fname2]
                if hasattr(out, 'numeric'):
                    out = out.numeric
                real = out.expected[fname1]
                ureal = out.uncertainty[fname2]
                imag = out.expected[fname2]
                uimag = out.uncertainty[fname2]
                cols = [f'{report.Number(real + 1j*imag, matchto=ureal)}',
                        f'± {report.Number(ureal + 1j*uimag)}',
                        f'{cor:.4f}']
            return cols

        rows = []
        for fname in self._functionnames:
            row = [fname, 'GUM']
            row.extend(_addcol(self._results.gum, fname))
            rows.append(row)
            row = [fname, 'Monte Carlo']
            row.extend(_addcol(self._results.montecarlo, fname))
            rows.append(row)
        rpt.table(rows, hdr)
        return rpt

    def plot(self, fig=None, funcname=None, **kwargs):
        ''' Plot uncertainty region in polar form

            Args:
                fig (plt.Figure): Figure to plot on. If not provided, a new figure will be created
                funcname (str): Name of function to plot
                showgum (bool): Show the GUM solution
                showmc (bool): Show the Monte Carlo solution
                gumcontour (bool): Plot GUM as contour plots. If false, an approximate 95%
                    region will be shaded without considering correlation between components.
                mccontour (bool): Plot Monte Carlo as contours. If false, a scatter plot of
                    the first samples will be plotted.
                points (int): Number of samples to plot in Monte Carlo scatter plot if mccontour is True.
                bins (int): Number of bins for calculating 2D histogram used to estimate contour plots
                cmapmc (string): Name of matplotlib colormap for Monte Carlo contour
                cmapgum (string): Name of matplotlib colormap for GUM contour
                color (string): Name of maptlotlib color for GUM shaded region
                colormc (string): Name of matplotlib color for Monte Carlo scatter plot
                polar (bool): Show results as polar plot (if magphase parameter is True in the calculation)
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
            self._results.gum.report.plot(funcname=funcname, ax=ax, cmap=cmapgum, color=color,
                                          polar=polar, contour=gumcontour)

        if showmc:
            self._results.montecarlo.report.plot(funcname=funcname, ax=ax, cmap=cmapmc, color=colormc,
                                                 polar=polar, contour=mccontour, points=points, bins=bins)

        fig.tight_layout()
