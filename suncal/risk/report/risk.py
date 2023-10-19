''' Report of risk calculation '''

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

from ..risk import specific_risk
from ..risk_montecarlo import PFAR_MC
from .. import risk_sweep
from ...common import report, plotting, distributions


class RiskReport:
    ''' Generate risk calculation reports '''
    def __init__(self, riskmodel):
        self.model = riskmodel
        self.plot = RiskPlot(self.model)

    def summary(self, **kwargs):
        ''' Generate report of risk calculation '''
        hdr = []
        cols = []
        cost = None
        conditional = kwargs.get('conditional', False)

        if self.model.procdist is not None:
            cpk, risk_total, risk_lower, risk_upper = self.model.cpk()
            hdr.extend(['Process Risk'])   # No way to span columns at this point...
            cols.append([('Process Risk: ', report.Number(risk_total*100, fmt='auto'), '%'),
                         ('Upper limit risk: ', report.Number(risk_upper*100, fmt='auto'), '%'),
                         ('Lower limit risk: ', report.Number(risk_lower*100, fmt='auto'), '%'),
                         ('Process capability index (Cpk): ', report.Number(cpk))])
            if self.model.cost_FA is not None:
                cost = self.model.cost_FA * risk_total  # Everything accepted - no false rejects

        if self.model.testdist is not None:
            val = self.model.testdist.median() + self.model.testbias
            PFx, accept = self.model.specific_risk()  # Get PFA/PFR of specific measurement

            hdr.extend(['Specific Measurement Risk'])
            cols.append([
                ('TUR: ', report.Number(self.model.get_tur(), fmt='auto')),
                ('Measured value: ', report.Number(val)),
                f'Result: {"ACCEPT" if accept else "REJECT"}',
                (f'Specific F{"A" if accept else "R"} Risk: ', report.Number(PFx*100, fmt='auto'), '%'),
                ])

        if self.model.testdist is not None and self.model.procdist is not None:
            hdr.extend(['Global Risk'])
            pfa = self.model.PFA(conditional=conditional)
            pfr = self.model.PFR()
            cols.append([
                (f'Total PFA{" (conditional)" if conditional else ""}: ', report.Number(pfa*100, fmt='auto'), '%'),
                ('Total PFR: ', report.Number(pfr*100, fmt='auto'), '%'), '-', '-'])
            if self.model.cost_FA is not None and self.model.cost_FR is not None:
                cost = self.model.cost_FA * pfa + self.model.cost_FR * pfr

        rpt = report.Report()
        if len(hdr) > 0:
            rows = list(map(list, zip(*cols)))  # Transpose cols->rows
            rpt.table(rows=rows, hdr=hdr)

        if cost is not None:
            costrows = [['Cost of false accept', report.Number(self.model.cost_FA)],
                        ['Cost of false reject', report.Number(self.model.cost_FR)],
                        ['Expected cost', report.Number(cost)]]
            rpt.table(costrows, hdr=['Cost', 'Value'])
        return rpt

    def all(self, **kwargs):
        ''' Report with table and plots '''
        if kwargs.get('mc', False):
            with plotting.plot_figure() as fig:
                r = self.montecarlo(fig=fig, **kwargs)
                r.plot(fig)
        else:
            r = report.Report(**kwargs)
            with plotting.plot_figure() as fig:
                self.plot.joint(fig)
                r.plot(fig)
            r.txt('\n\n')
            r.append(self.summary(**kwargs))
        return r

    def montecarlo(self, fig=None, **kwargs):
        ''' Run Monte-Carlo risk and return report. If fig is provided, plot it. '''
        N = kwargs.get('samples', 100000)
        LL, UL = self.model.speclimits
        GB = self.model.gbofsts
        pfa, pfr, psamples, tsamples, *_ = PFAR_MC(
            self.model.procdist, self.model.testdist, LL, UL, *GB, N=N, testbias=self.model.testbias)

        LLplot = np.nan if not np.isfinite(LL) else LL
        ULplot = np.nan if not np.isfinite(UL) else UL

        if fig is not None:
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
                        markersize=2, color='C2', label='False Reject', rasterized=True)
                ax.axvline(LLplot, ls='--', lw=1, color='black')
                ax.axvline(ULplot, ls='--', lw=1, color='black')
                ax.axhline(LLplot+GB[0], lw=1, ls='--', color='gray')
                ax.axhline(ULplot-GB[1], lw=1, ls='--', color='gray')
                ax.axhline(LLplot, ls='--', lw=1, color='black')
                ax.axhline(ULplot, ls='--', lw=1, color='black')
                ax.legend(loc='upper left', fontsize=10)
                ax.set_xlabel('Actual Product')
                ax.set_ylabel('Test Result')
                pdist = self.model.procdist
                tdist = self.model.testdist
                xmin = np.nanmin([LLplot, pdist.mean()-5*pdist.std()])
                xmax = np.nanmax([ULplot, pdist.mean()+5*pdist.std()])
                ymin = np.nanmin([LLplot, pdist.mean()-5*pdist.std()-tdist.std()])
                ymax = np.nanmax([ULplot, pdist.mean()+5*pdist.std()+tdist.std()])
                ax.set_xlim(xmin, xmax)
                ax.set_ylim(ymin, ymax)
                fig.tight_layout()

        rpt = report.Report(**kwargs)
        rpt.txt('- TUR: ')
        rpt.num(self.model.get_tur(), fmt='auto', end='\n')
        rpt.txt('- Total PFA: ')
        rpt.num(pfa*100, fmt='auto', end='%\n')
        rpt.txt('- Total PFR: ')
        rpt.num(pfr*100, fmt='auto', end='%\n')
        return rpt

    def guardband_sweep(self, fig=None, **kwargs):
        ''' Plot PFA/R vs guardband '''
        with plt.style.context(plotting.plotstyle):
            fig, _ = plotting.initplot(fig)
            fig.clf()
            LL, UL = self.model.speclimits
            simple = self.model.is_simple()
            dtest = self.model.testdist
            dproc = self.model.procdist

            if dproc is not None and dtest is not None:
                if simple:
                    gbrange = np.linspace(0, 1, num=26)
                    xlabel = 'Guardband Factor'
                else:
                    gbrange = np.linspace(0, dtest.std()*3, num=26)
                    xlabel = 'Guardband'

                pfa = np.empty(len(gbrange))
                pfr = np.empty(len(gbrange))
                for i, gb in enumerate(gbrange):
                    if simple:
                        self.model.set_gbf(gb)
                    else:
                        self.model.gbofsts = (gb, gb)  # Always symmetric
                    pfa[i] = self.model.PFA()
                    pfr[i] = self.model.PFR()

                ax1 = fig.add_subplot(2, 1, 1)
                ax1.plot(gbrange, pfa*100)
                ax1.set_ylabel('False Accept %')
                ax2 = fig.add_subplot(2, 1, 2)
                ax2.plot(gbrange, pfr*100)
                ax2.set_ylabel('False Reject %')
                ax2.set_xlabel(xlabel)
                ax1.get_shared_x_axes().join(ax1, ax2)
                fig.tight_layout()
                if simple:
                    ax1.set_xlim(1, 0)  # Go backwards
                    ax2.set_xlim(1, 0)
                plt.close(fig)

                rpt = report.Report(**kwargs)
                rpt.hdr(f'Global Risk vs {xlabel}', level=2)
                hdr = [xlabel, 'False Accept %', 'False Reject %']
                gbrange = [report.Number(g) for g in gbrange]
                pfa = [report.Number(p*100) for p in pfa]
                pfr = [report.Number(p*100) for p in pfr]
                rows = list(map(list, zip(gbrange, pfa, pfr)))
                if simple:
                    rows = list(reversed(rows))
                rpt.table(rows=rows, hdr=hdr)
                return rpt

            rpt = report.Report(**kwargs)
            rpt.txt('No test distribution')
            return rpt

    def probconform(self, fig=None, **kwargs):
        ''' Plot Probability of Conformance plot '''
        with plt.style.context(plotting.plotstyle):
            fig, _ = plotting.initplot(fig)
            fig.clf()
            LL, UL = self.model.speclimits
            dtest = self.model.testdist

            if dtest is not None:
                kwds = distributions.get_distargs(dtest)
                bias = self.model.testbias
                w = (UL-LL)
                xx = np.linspace(LL-w/2, UL+w/2, num=500)
                if not np.isfinite(w):
                    w = dtest.std() * 4
                    xx = np.linspace(dtest.mean()-w if not np.isfinite(LL) else LL-w/2,
                                     dtest.mean()+w if not np.isfinite(UL) else UL+w/2,
                                     num=500)
                fa_lower = np.empty(len(xx))
                fa_upper = np.empty(len(xx))
                for i, loc in enumerate(xx):
                    dtest.set_median(loc-bias)
                    kwds = distributions.get_distargs(dtest)
                    dtestswp = dtest.dist(**kwds)  # Pass in rv_continuous, could be faster than Distribution object
                    fa_lower[i] = specific_risk(dtestswp, LL=LL, UL=np.inf).total
                    fa_upper[i] = specific_risk(dtestswp, LL=-np.inf, UL=UL).total
                fa = 1-(fa_lower + fa_upper)

                ax = fig.add_subplot(1, 1, 1)
                ax.plot(xx, fa*100, color='C1')
                ax.set_ylabel('Probability of Conformance %')
                ax.set_xlabel('Measurement Result')
                ax.axvline(LL, ls='--', label='Specification Limits', color='C2')
                ax.axvline(UL, ls='--', color='C2')
                GBL, GBU = self.model.gbofsts
                if GBL != 0 or GBU != 0:
                    ax.axvline(LL+GBL, ls='--', label='Guardband', color='C3')
                    ax.axvline(UL-GBU, ls='--', color='C3')
                fig.tight_layout()
                plt.close(fig)

                rpt = report.Report(**kwargs)
                rpt.hdr('Probability of Conformance', level=2)
                hdr = ['Measurement Result', 'Probability of Conformance %']
                xx = report.Number.number_array(xx[::10])
                pc = [report.Number(p) for p in fa[::10]*100]
                rows = list(map(list, zip(xx, pc)))
                rpt.table(rows, hdr=hdr)
                return rpt

            rpt = report.Report(**kwargs)
            rpt.txt('No test distribution')
            return rpt

    def sweep(self, fig=None, **kwargs):
        ''' Plot PFA(R) sweep (simple mode only for now) '''
        xvar = kwargs.get('xvar', 'itp')
        zvar = kwargs.get('zvar', 'tur')
        xvals = kwargs.get('xvals', np.linspace(.55, .95, num=10))
        zvals = kwargs.get('zvals', [1.5, 2, 3, 4])
        yvar = kwargs.get('yvar', 'PFA')
        threed = kwargs.get('threed', False)
        logy = kwargs.get('logy', False)
        gbmode = kwargs.get('gbf', None)
        sig0 = kwargs.get('sig0', None)
        tbias = kwargs.get('tbias', 0)
        pbias = kwargs.get('pbias', 0)

        gbf = self.model.get_gbf() if gbmode is None else gbmode

        labels = {'tur': 'TUR', 'itp': 'In-Tolerance Probability %', 'tbias': 'Test Measurement Bias',
                  'pbias': 'Process Distribution Bias', 'gbf': 'GBF', 'sig0': r'$SL/\sigma_0$'}

        rpt = report.Report(**kwargs)

        with plt.style.context(plotting.plotstyle):
            fig, _ = plotting.initplot(fig)
            fig.clf()

            yvars = [yvar.lower()] if yvar.lower() != 'both' else ['pfa', 'pfr']
            for k, yvar in enumerate(yvars):
                curves = risk_sweep.PFA_sweep_simple(
                    xvar, zvar, xvals, zvals,
                    GBFdflt=gbf, itpdflt=self.model.get_itp(),
                    TURdflt=self.model.get_tur(), risk=yvar,
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


class RiskPlot:
    ''' Generate plots of risk calculation '''
    def __init__(self, riskresults):
        self.model = riskresults

    def distributions(self, fig=None):
        ''' Plot risk distributions '''
        with plt.style.context(plotting.plotstyle):
            fig, _ = plotting.initplot(fig)
            fig.clf()

            procdist = self.model.procdist
            testdist = self.model.testdist

            nrows = (procdist is not None) + (testdist is not None)
            plotnum = 0
            LL, UL = self.model.speclimits
            GBL, GBU = self.model.gbofsts

            # Add some room on either side of distributions
            pad = 0
            if procdist is not None:
                pad = max(pad, procdist.std() * 3)
            if testdist is not None:
                pad = max(pad, testdist.std() * 3)

            xmin = xmax = pad
            if np.isfinite(LL):
                xmin = LL-pad
            elif procdist:
                xmin = procdist.mean() - pad*2
            elif testdist:
                xmin = testdist.mean() - pad*2
            if np.isfinite(UL):
                xmax = UL+pad
            elif procdist:
                xmax = procdist.mean() + pad*2
            elif testdist:
                xmax = testdist.mean() + pad*2

            x = np.linspace(xmin, xmax, 300)
            if procdist is not None:
                yproc = procdist.pdf(x)
                ax = fig.add_subplot(nrows, 1, plotnum+1)
                ax.plot(x, yproc, label='Process Distribution', color='C0')
                ax.axvline(LL, ls='--', label='Specification Limits', color='C2')
                ax.axvline(UL, ls='--', color='C2')
                ax.fill_between(x, yproc, where=((x <= LL) | (x >= UL)), alpha=.5, color='C0')
                ax.set_ylabel('Probability Density')
                ax.set_xlabel('Value')
                ax.legend(loc='upper left', fontsize=10)
                if self.model.is_simple():
                    ax.xaxis.set_major_formatter(FormatStrFormatter(r'%.1fSL'))
                plotnum += 1

            if testdist is not None:
                ytest = testdist.pdf(x)
                median = self.model.testdist.median()
                measured = median + self.model.testbias
                ax = fig.add_subplot(nrows, 1, plotnum+1)
                ax.plot(x, ytest, label='Test Distribution', color='C1')
                ax.axvline(measured, ls='--', color='C1')
                ax.axvline(median, ls=':', lw=.5, color='lightgray')
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
                if self.model.is_simple():
                    ax.xaxis.set_major_formatter(FormatStrFormatter(r'%.1fSL'))
            fig.tight_layout()
        return fig

    def joint(self, fig=None):
        ''' Plot risk distributions '''
        with plt.style.context(plotting.plotstyle):
            fig, _ = plotting.initplot(fig)
            fig.clf()

            procdist = self.model.procdist
            testdist = self.model.testdist

            pad = 0
            pad = max(pad, procdist.std() * 3)
            pad = max(pad, testdist.std() * 3)

            LL, UL = self.model.speclimits
            GBL, GBU = self.model.gbofsts

            xmin = LL-pad if np.isfinite(LL) else procdist.mean() - pad*2
            xmax = UL+pad if np.isfinite(UL) else procdist.mean() + pad*2

            x = y = np.linspace(xmin, xmax, 300)
            xx, yy = np.meshgrid(x, y)
            pdf1 = procdist.pdf(xx)
            expected = testdist.median()
            kwds = distributions.get_distargs(testdist)
            locorig = kwds.pop('loc', 0)
            pdf2 = testdist.dist.pdf(yy, loc=xx-(expected-locorig), **kwds)

            ax1 = plt.subplot2grid((5, 5), loc=(1, 0), colspan=4, rowspan=4, fig=fig)
            ax1.contourf(xx, yy, (pdf1*pdf2)**.5, levels=20, cmap='Blues')
            ax1.contour(xx, yy, (pdf1*pdf2)**.5, levels=20, colors='blue', linewidths=.1)
            ax1.axvline(LL, color='black', ls='--', lw=1)
            ax1.axhline(LL, color='black', ls='--', lw=1)
            ax1.axvline(UL, color='black', ls='--', lw=1)
            ax1.axhline(UL, color='black', ls='--', lw=1)
            ax1.axhline(LL+GBL, color='gray', ls='--', lw=1)
            ax1.axhline(UL-GBU, color='gray', ls='--', lw=1)

            procpdf = procdist.pdf(x)
            testpdf = testdist.pdf(y)

            ax2 = plt.subplot2grid((5, 5), loc=(0, 0), colspan=4, sharex=ax1, fig=fig)
            ax2.plot(x, procpdf)
            ax2.fill_between(x, procpdf, where=x > UL, color='C0', alpha=.25)
            ax2.fill_between(x, procpdf, where=x < LL, color='C0', alpha=.25)
            ax2.axvline(LL, color='black', ls='--', lw=1)
            ax2.axvline(UL, color='black', ls='--', lw=1)

            ax3 = plt.subplot2grid((5, 5), loc=(1, 4), rowspan=4, sharey=ax1, fig=fig)
            ax3.plot(testpdf, y, color='C1')
            ax3.fill_betweenx(y, testpdf, where=y > UL, color='C1', alpha=.25)
            ax3.fill_betweenx(y, testpdf, where=y < LL, color='C1', alpha=.25)
            ax3.axhline(testdist.mean(), ls='--', lw=1, color='C1')
            ax3.axhline(LL, color='black', ls='--', lw=1)
            ax3.axhline(UL, color='black', ls='--', lw=1)

            fig.subplots_adjust(hspace=0.05, wspace=.05)
            plt.setp(ax2.get_xticklabels(), visible=False)
            plt.setp(ax2.get_yticklabels(), visible=False)
            plt.setp(ax3.get_xticklabels(), visible=False)
            plt.setp(ax3.get_yticklabels(), visible=False)

            ax1.set_xlabel('Actual Product')
            ax1.set_ylabel('Test Result')
            ax1.set_ylim(y.min(), y.max())

            LLGB = LL+GBL if np.isfinite(LL+GBL) else xmin
            ULGB = UL-GBU if np.isfinite(UL-GBU) else xmax
            LL1 = LL if np.isfinite(LL) else xmin
            UL1 = UL if np.isfinite(UL) else xmax
            ax1.fill_between(x, LLGB, ULGB, where=(x < LL), color='C1', alpha=.15, label='False Accept')
            ax1.fill_between(x, LLGB, ULGB, where=(x > UL), color='C1', alpha=.15)
            ax1.fill_betweenx(y, LL1, UL1, where=(x < LL+GBL), color='C2', alpha=.15, label='False Reject')
            ax1.fill_betweenx(y, LL1, UL1, where=(x > UL-GBU), color='C2', alpha=.15)
            ax1.legend(loc='upper left', fontsize=10)
            ax1.set_xlim(xmin, xmax)
            ax1.set_ylim(xmin, xmax)

            if self.model.is_simple():
                ax1.xaxis.set_major_formatter(FormatStrFormatter(r'%.1fSL'))
                ax1.yaxis.set_major_formatter(FormatStrFormatter(r'%.1fSL'))
        return fig
