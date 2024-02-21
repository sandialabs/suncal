''' Report of risk calculation '''

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from .. import risk_sweep
from ...common import report, plotting, distributions


class RiskReport:
    ''' Generate risk calculation reports '''
    def __init__(self, riskmodel):
        self.result = riskmodel
        self.plot = RiskPlot(self.result)

    def summary(self, **kwargs):
        ''' Generate report of risk calculation '''
        hdr = []
        cols = []
        cost = None
        conditional = kwargs.get('conditional', False)

        if self.result.process_dist is not None:
            hdr.extend(['Process Risk'])   # No way to span columns at this point...
            cols.append([('Process Risk: ', report.Number(self.result.process_risk*100, fmt='auto'), '%'),
                         ('Upper limit risk: ', report.Number(self.result.process_upper*100, fmt='auto'), '%'),
                         ('Lower limit risk: ', report.Number(self.result.process_lower*100, fmt='auto'), '%'),
                         ('Process capability index (Cpk): ', report.Number(self.result.cpk)),
                        ])
            if self.result.cost_fa is not None:
                cost = self.result.cost_fa * self.result.process_risk  # Everything accepted - no false rejects

        if self.result.measure_dist is not None:
            val = self.result.measure_dist.median() + self.result.measure_bias
            hdr.extend(['Specific Measurement Risk'])
            if self.result.specific is not None:
                cols.append([
                    ('TUR: ', report.Number(self.result.tur, fmt='auto')),
                    ('Measured value: ', report.Number(val)),
                    (f'Specific F{"A" if self.result.specific_accept else "R"} Risk: ',
                     report.Number(self.result.specific*100, fmt='auto'), '%'),
                    ('Worst-Case Specific Risk: ',
                     report.Number(self.result.specific_worst*100, fmt='auto'), '%'),
                    ])
            else:
                cols.append([
                    ('TUR: ', report.Number(self.result.tur, fmt='auto')),
                    ('Worst-Case Specific Risk: ',
                     report.Number(self.result.specific_worst*100, fmt='auto'), '%'),
                    '&nbsp;'
                    ])

        if self.result.measure_dist is not None and self.result.process_dist is not None:
            hdr.extend(['Global Risk'])
            pfa = self.result.cpfa if conditional else self.result.pfa
            pfr = self.result.pfr
            cols.append([
                (f'Total PFA{" (conditional)" if conditional else ""}: ', report.Number(pfa*100, fmt='auto'), '%'),
                ('Total PFR: ', report.Number(pfr*100, fmt='auto'), '%'), '&nbsp;', '&nbsp;'])
            if self.result.cost_fa is not None and self.result.cost_fr is not None:
                cost = self.result.cost_fa * pfa + self.result.cost_fr * pfr

        rpt = report.Report()
        if len(hdr) > 0:
            rows = list(map(list, zip(*cols)))  # Transpose cols->rows
            rpt.table(rows=rows, hdr=hdr)

        if cost is not None:
            costrows = [['Cost of false accept', report.Number(self.result.cost_fa)],
                        ['Cost of false reject', report.Number(self.result.cost_fr)],
                        ['Expected cost', report.Number(cost)]]
            rpt.table(costrows, hdr=['Cost', 'Value'])
        return rpt

    def all(self, **kwargs):
        ''' Report with table and plots '''
        r = report.Report(**kwargs)
        with plotting.plot_figure() as fig:
            self.plot.joint(fig)
            r.plot(fig)
        r.txt('\n\n')
        r.append(self.summary(**kwargs))
        return r


class RiskPlot:
    ''' Generate plots of risk calculation '''
    def __init__(self, riskresults):
        self.result = riskresults

    def distributions(self, fig=None):
        ''' Plot risk distributions '''
        with plt.style.context(plotting.plotstyle):
            fig, _ = plotting.initplot(fig)
            fig.clf()

            process_dist = self.result.process_dist
            measure_dist = self.result.measure_dist

            nrows = (process_dist is not None) + (measure_dist is not None)
            plotnum = 0
            LL, UL = self.result.tolerance
            GBL, GBU = self.result.gbofsts

            # Add some room on either side of distributions
            pad = 0
            if process_dist is not None:
                pad = max(pad, process_dist.std() * 3)
            if measure_dist is not None:
                pad = max(pad, measure_dist.std() * 3)

            xmin = xmax = pad
            if np.isfinite(LL):
                xmin = LL-pad
            elif process_dist:
                xmin = process_dist.mean() - pad*2
            elif measure_dist:
                xmin = measure_dist.mean() - pad*2
            if np.isfinite(UL):
                xmax = UL+pad
            elif process_dist:
                xmax = process_dist.mean() + pad*2
            elif measure_dist:
                xmax = measure_dist.mean() + pad*2

            x = np.linspace(xmin, xmax, 300)
            if process_dist is not None:
                yproc = process_dist.pdf(x)
                ax = fig.add_subplot(nrows, 1, plotnum+1)
                ax.plot(x, yproc, label='Process Distribution', color='C0')
                ax.axvline(LL, ls='--', label='Specification Limits', color='C2')
                ax.axvline(UL, ls='--', color='C2')
                ax.fill_between(x, yproc, where=((x <= LL) | (x >= UL)), alpha=.5, color='C0')
                ax.set_ylabel('Probability Density')
                ax.set_xlabel('Value')
                ax.legend(loc='upper left', fontsize=10)
                plotnum += 1

            if measure_dist is not None:
                ytest = measure_dist.pdf(x)
                median = self.result.measure_dist.median()
                measured = median + self.result.measure_bias
                ax = fig.add_subplot(nrows, 1, plotnum+1)
                ax.plot(x, ytest, label='Test Distribution', color='C1')
                ax.axvline(measured, ls='--', color='C1')
                if measured != median:
                    ax.axvline(median, ls=':', color='gray')
                ax.axvline(LL, ls='--', label='Specification Limits', color='C2')
                ax.axvline(UL, ls='--', color='C2')
                if GBL != 0 or GBU != 0:
                    ax.axvline(LL+GBL, ls='--', label='Guardband', color='C3')
                    ax.axvline(UL-GBU, ls='--', color='C3')

                if measured > UL-GBU or measured < LL+GBL:   # Shade PFR
                    ax.fill_between(x, ytest, where=((x >= LL) & (x <= UL)), alpha=.5, color='C1')
                else:  # Shade PFA
                    ax.fill_between(x, ytest, where=((x <= LL) | (x >= UL)), alpha=.5, color='C1')

                ax.set_ylabel('Probability Density')
                ax.set_xlabel('Value')
                ax.legend(loc='upper left', fontsize=10)
            fig.tight_layout()
        return fig

    def joint(self, fig=None):
        ''' Plot risk distributions '''
        with plt.style.context(plotting.plotstyle):
            fig, _ = plotting.initplot(fig)

            # Creating the axes grid is slow. Reuse if possible.
            if len(fig.axes) != 3:
                fig.clf()
                ax1 = plt.subplot2grid((5, 5), loc=(1, 0), colspan=4, rowspan=4, fig=fig)
                ax2 = plt.subplot2grid((5, 5), loc=(0, 0), colspan=4, sharex=ax1, fig=fig)
                ax3 = plt.subplot2grid((5, 5), loc=(1, 4), rowspan=4, sharey=ax1, fig=fig)
            else:
                ax1, ax2, ax3 = fig.axes
                ax1.clear()
                ax2.clear()
                ax3.clear()

            process_dist = self.result.process_dist
            measure_dist = self.result.measure_dist

            pad = 0
            pad = max(pad, process_dist.std() * 3)
            pad = max(pad, measure_dist.std() * 3)

            LL, UL = self.result.tolerance
            GBL, GBU = self.result.gbofsts

            xmin = LL-pad if np.isfinite(LL) else process_dist.mean() - pad*2
            xmax = UL+pad if np.isfinite(UL) else process_dist.mean() + pad*2

            x = y = np.linspace(xmin, xmax, 300)
            xx, yy = np.meshgrid(x, y)
            pdf1 = process_dist.pdf(xx)
            expected = measure_dist.median()
            kwds = distributions.get_distargs(measure_dist)
            locorig = kwds.pop('loc', 0)
            pdf2 = measure_dist.dist.pdf(yy, loc=xx-(expected-locorig), **kwds)

            cmap = mpl.colors.LinearSegmentedColormap.from_list('suncalblues', ['#007a8611', '#007a86FF'])
            ax1.contourf(xx, yy, (pdf1*pdf2)**.5, levels=8, cmap=cmap)
            ax1.axvline(LL, color='black', ls='--', lw=1)
            ax1.axhline(LL, color='black', ls='--', lw=1)
            ax1.axvline(UL, color='black', ls='--', lw=1)
            ax1.axhline(UL, color='black', ls='--', lw=1)
            ax1.axhline(LL+GBL, color='gray', ls='--', lw=1)
            ax1.axhline(UL-GBU, color='gray', ls='--', lw=1)

            procpdf = process_dist.pdf(x)
            testpdf = measure_dist.pdf(y)

            ax2.plot(x, procpdf)
            ax2.fill_between(x, procpdf, where=((x > UL) | (x < LL)), color='C0', alpha=.5)
            ax2.axvline(LL, color='black', ls='--', lw=1)
            ax2.axvline(UL, color='black', ls='--', lw=1)

            ax3.plot(testpdf, y, color='C1')
            ax3.fill_betweenx(y, testpdf, where=((y > UL) | (y < LL)), color='C1', alpha=.5)
            median = self.result.measure_dist.median()
            measured = median + self.result.measure_bias
            ax3.axhline(measured, ls='--', color='C1')
            if measured != median:
                ax3.axhline(median, ls=':', color='gray')
            ax3.axhline(LL, color='black', ls='--', lw=1)
            ax3.axhline(UL, color='black', ls='--', lw=1)

            fig.subplots_adjust(hspace=0.05, wspace=.05)
            ax2.axes.get_xaxis().set_visible(False)
            ax2.axes.get_yaxis().set_visible(False)
            ax3.axes.get_xaxis().set_visible(False)
            ax3.axes.get_yaxis().set_visible(False)

            ax1.set_xlabel('Actual Product')
            ax1.set_ylabel('Test Result')
            ax1.set_ylim(y.min(), y.max())

            LLGB = LL+GBL if np.isfinite(LL+GBL) else xmin
            ULGB = UL-GBU if np.isfinite(UL-GBU) else xmax
            LL1 = LL if np.isfinite(LL) else xmin
            UL1 = UL if np.isfinite(UL) else xmax
            ax1.fill_between(x, LLGB, ULGB, where=(x < LL), color='C1', alpha=.15, label='False Accept')
            ax1.fill_between(x, LLGB, ULGB, where=(x > UL), color='C1', alpha=.15)
            ax1.fill_betweenx(y, LL1, UL1, where=(x < LL+GBL), color='C3', alpha=.15, label='False Reject')
            ax1.fill_betweenx(y, LL1, UL1, where=(x > UL-GBU), color='C3', alpha=.15)
            ax1.legend(loc='upper left', fontsize=10)
            ax1.set_xlim(xmin, xmax)
            ax1.set_ylim(xmin, xmax)

        return fig


class RiskReportMonteCarlo:
    ''' Generate risk calculation reports for Monte Carlo calculation '''
    def __init__(self, result):
        self.result = result

    def summary(self, **kwargs):
        rpt = report.Report(**kwargs)
        rpt.txt('- TUR: ')
        rpt.num(self.result.tur, fmt='auto', end='\n')
        rpt.txt('- Total PFA: ')
        rpt.num(self.result.pfa*100, fmt='auto', end='%\n')
        rpt.txt('- Total PFR: ')
        rpt.num(self.result.pfr*100, fmt='auto', end='%\n')
        return rpt

    def plot(self, fig=None, **kwargs):
        ''' Run Monte-Carlo risk and return report. If fig is provided, plot it. '''
        LL, UL = self.result.tolerance
        GB = self.result.gbofsts
        LLplot = np.nan if not np.isfinite(LL) else LL
        ULplot = np.nan if not np.isfinite(UL) else UL
        psamples = self.result.process_samples
        tsamples = self.result.measure_samples

        with plt.style.context(plotting.plotstyle):
            fig, _ = plotting.initplot(fig)
            fig.clf()
            ax = fig.add_subplot(1, 1, 1)
            if psamples is not None:
                ifa1 = (tsamples > LL+GB[0]) & (tsamples < UL-GB[1]) & ((psamples < LL) | (psamples > UL))
                ifr1 = ((tsamples < LL+GB[0]) | (tsamples > UL-GB[1])) & ((psamples > LL) & (psamples < UL))
                good = np.logical_not(ifa1 | ifr1)
                ax.plot(psamples[good], tsamples[good], marker='o', ls='',
                        markersize=2, color='C0', label='Correct Decision', rasterized=True)
                ax.plot(psamples[ifa1], tsamples[ifa1], marker='o', ls='',
                        markersize=2, color='C1', label='False Accept', rasterized=True)
                ax.plot(psamples[ifr1], tsamples[ifr1], marker='o', ls='',
                        markersize=2, color='C3', label='False Reject', rasterized=True)
                ax.axvline(LLplot, ls='--', lw=1, color='black')
                ax.axvline(ULplot, ls='--', lw=1, color='black')
                ax.axhline(LLplot+GB[0], lw=1, ls='--', color='gray')
                ax.axhline(ULplot-GB[1], lw=1, ls='--', color='gray')
                ax.axhline(LLplot, ls='--', lw=1, color='black')
                ax.axhline(ULplot, ls='--', lw=1, color='black')
                ax.legend(loc='upper left', fontsize=10)
                ax.set_xlabel('Actual Product')
                ax.set_ylabel('Test Result')
                pdist = self.result.process_dist
                tdist = self.result.measure_dist
                ax.set_xlim(np.nanmin([LLplot, pdist.mean()-5*pdist.std()]),
                            np.nanmax([ULplot, pdist.mean()+5*pdist.std()]))
                ax.set_ylim(np.nanmin([LLplot, pdist.mean()-5*pdist.std()-tdist.std()]),
                            np.nanmax([ULplot, pdist.mean()+5*pdist.std()+tdist.std()]))
                fig.tight_layout()
        return fig


class RiskReportGuardbandSweep:
    ''' Report Guardband Sweep '''
    def __init__(self, result):
        self.result = result

    def summary(self, fig=None, **kwargs):
        ''' Plot PFA/R vs guardband '''
        with plt.style.context(plotting.plotstyle):
            fig, _ = plotting.initplot(fig)
            fig.clf()

            ax1 = fig.add_subplot(2, 1, 1)
            ax1.plot(self.result.guardband, self.result.pfa*100)
            ax1.set_ylabel('False Accept %')
            ax2 = fig.add_subplot(2, 1, 2)
            ax2.plot(self.result.guardband, self.result.pfr*100)
            ax2.set_ylabel('False Reject %')
            ax2.set_xlabel('Guardband')
            ax1.sharex(ax2)
            fig.tight_layout()
            plt.close(fig)

            rpt = report.Report(**kwargs)
            rpt.hdr('Global Risk vs Guardband', level=2)
            hdr = ['Guardband', 'False Accept %', 'False Reject %']
            gbrange = [report.Number(g) for g in self.result.guardband]
            pfa = [report.Number(p*100) for p in self.result.pfa]
            pfr = [report.Number(p*100) for p in self.result.pfr]
            rows = list(map(list, zip(gbrange, pfa, pfr)))
            rpt.table(rows=rows, hdr=hdr)
        return rpt


class RiskReportProbConform:
    ''' Report probability of conformance '''
    def __init__(self, result):
        self.result = result

    def summary(self, fig=None, **kwargs):
        ''' Plot Probability of Conformance plot '''
        with plt.style.context(plotting.plotstyle):
            fig, _ = plotting.initplot(fig)
            fig.clf()

            LL, UL = self.result.tolerance
            ax = fig.add_subplot(1, 1, 1)
            ax.plot(self.result.measured,
                    self.result.probconform*100,
                    color='C1')
            ax.set_ylabel('Probability of Conformance %')
            ax.set_xlabel('Measurement Result')
            ax.axvline(LL, ls='--', label='Specification Limits', color='C3')
            ax.axvline(UL, ls='--', color='C3')
            GBL, GBU = self.result.gbofsts
            if GBL != 0 or GBU != 0:
                ax.axvline(LL+GBL, ls='--', label='Guardband', color='C0')
                ax.axvline(UL-GBU, ls='--', color='C0')
            plt.close(fig)

            rpt = report.Report(**kwargs)
            rpt.hdr('Probability of Conformance', level=2)
            hdr = ['Measurement Result', 'Probability of Conformance %']
            xx = report.Number.number_array(self.result.measured[::10])
            pc = [report.Number(p) for p in self.result.probconform[::10]*100]
            rows = list(map(list, zip(xx, pc)))
            rpt.table(rows, hdr=hdr)
        return rpt


def risk_sweeper(fig=None, **kwargs):
    ''' Plot PFA(R) sweep (simple mode only for now) '''
    xvar = kwargs.get('xvar', 'itp')
    zvar = kwargs.get('zvar', 'tur')
    xvals = kwargs.get('xvals', np.linspace(.55, .95, num=10))
    zvals = kwargs.get('zvals', [1.5, 2, 3, 4])
    yvar = kwargs.get('yvar', 'PFA')
    threed = kwargs.get('threed', False)
    logy = kwargs.get('logy', False)
    gbmode = kwargs.get('gbmode', None)
    sig0 = kwargs.get('sig0', None)
    tbias = kwargs.get('tbias', 0)
    pbias = kwargs.get('pbias', 0)

    # Default values
    gbf_dflt = 1 if gbmode is None else gbmode
    itp_dflt = 0.95
    tur_dflt = 4

    labels = {'tur': 'TUR',
              'itp': 'In-Tolerance Probability %',
              'tbias': 'Test Measurement Bias',
              'pbias': 'Process Distribution Bias',
              'gbf': 'GBF',
              'sig0': r'$SL/\sigma_0$'}

    rpt = report.Report(**kwargs)

    with plt.style.context(plotting.plotstyle):
        fig, _ = plotting.initplot(fig)
        fig.clf()

        yvars = [yvar.lower()] if yvar.lower() != 'both' else ['pfa', 'pfr']
        for k, yvar in enumerate(yvars):
            curves = risk_sweep.PFA_sweep_simple(
                xvar, zvar, xvals, zvals,
                GBFdflt=gbf_dflt, itpdflt=itp_dflt,
                TURdflt=tur_dflt, risk=yvar,
                sig0=sig0,
                tbias=tbias, pbias=pbias) * 100

            xlabel = labels.get(xvar, 'x')
            zlabel = labels.get(zvar, 'z')
            ylabel = f'{yvar.upper()} %'

            xplot = xvals if xvar.lower() not in ['itp', 'tbias', 'pbias'] else xvals * 100
            zplot = (np.zeros(len(xvals)) if xvar == 'none' else
                     zvals if zvar.lower() not in ['itp', 'tbias', 'pbias'] else
                     zvals * 100)

            if threed:
                ax = fig.add_subplot(1, len(yvars), k+1, projection='3d')
                xx, zz = np.meshgrid(xplot, zplot)
                ax.plot_surface(xx, zz, curves, cmap='coolwarm')
                ax.set_zlabel(ylabel)
                ax.set_ylabel(zlabel)
            else:
                ax = fig.add_subplot(1, len(yvars), k+1)
                for i in range(len(zvals)):
                    ax.plot(xplot, curves[i, :], label=str(zplot[i]))
                ax.set_ylabel(ylabel)

            ax.set_xlabel(xlabel)
            if zvar != 'none' and not threed:
                ax.legend(title=zlabel, fontsize=10)

            if logy:
                ax.set_yscale('log')

            rpt.hdr(ylabel, level=2)
            if zvar == 'none':
                hdr = [xlabel, ylabel]
            else:
                hdr = [xlabel] + [f'{zlabel}={z}' for z in zvals]
            rows = [[report.Number(xvals[i])] + [report.Number(p)
                    for p in row] for i, row in enumerate(curves.transpose())]
            rpt.table(rows, hdr)

        fig.tight_layout()
        plt.close(fig)
    return rpt
