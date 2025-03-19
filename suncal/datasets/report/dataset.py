''' Reports for datasets and analysis of variance '''

import numpy as np

from ...common import ttable, report, plotting
from ..dataset import sigma_rhok


class ReportDataSet:
    ''' Report data set/ANOVA results

        Args:
            datasetresult: DataSetResult instance
    '''
    def __init__(self, datasetresult):
        self.result = datasetresult
        self.plot = DataSetPlot(self.result)

    def summary(self, **kwargs):
        ''' Group stats '''
        rows = []
        # data is None when Summarized Data was entered
        if self.result.data is None or len(self.result.data) > 0:
            meanstrs = report.Number.number_array(self.result.groups.means, fmin=0)
            groups = self.result.groups
            for i in range(len(groups.means)):
                rows.append([str(self.result.colnames[i]),
                            meanstrs[i],
                            report.Number(groups.variances[i], fmin=0),
                            report.Number(groups.std_devs[i], fmin=0),
                            report.Number(groups.std_errs[i], fmin=0),
                            report.Number(groups.degfs[i], fmin=0)])

        rpt = report.Report(**kwargs)
        rpt.table(rows, hdr=['Group', 'Mean', 'Variance', 'Std. Dev.', 'Std. Error', 'Deg. Freedom'])

        if self.result.tolerance is not None:
            rpt.append(self.conformance(**kwargs))
        return rpt

    def all(self, **kwargs):
        ''' Report summary, pooled statistics, anova '''
        rpt = self.summary(**kwargs)
        rpt.append(self.pooled(**kwargs))
        rpt.append(self.anova(**kwargs))
        return rpt

    def column(self, column_name=None, **kwargs):
        ''' Stats for one column/group '''
        group = self.result.group(column_name)
        data = group.values
        rpt = report.Report(**kwargs)
        if len(data) > 0:
            q025, q25, q75, q975 = np.quantile(data, (.025, .25, .75, .975))
            rows = [['Mean', report.Number(group.mean, fmin=0)],
                    ['Standard Deviation', report.Number(group.std_dev, fmin=3)],
                    ['Std. Error of the Mean', report.Number(group.std_err, fmin=3)],
                    ['Deg. Freedom', f'{group.degf:.2f}'],
                    ['Minimum',  report.Number(data.min(), fmin=3)],
                    ['First Quartile', report.Number(q25, fmin=3)],
                    ['Median', report.Number(np.median(data), fmin=3)],
                    ['Third Quartile', report.Number(q75, fmin=3)],
                    ['Maximum', report.Number(data.max(), fmin=3)],
                    ['95% Coverage Interval', f'{report.Number(q025, fmin=3)}, {report.Number(q975, fmin=3)}']
                    ]
            rpt.table(rows, hdr=['Parameter', 'Value'])
        return rpt

    def pooled(self, **kwargs):
        ''' Pooled statistics and Grand Mean '''
        pstats = self.result.pooled
        stderr = self.result.uncertainty
        rows = []
        rows.append(['Grand Mean', report.Number(pstats.mean, matchto=pstats.repeatability, fmin=0), '&nbsp;'])
        rows.append(['Repeatability (Pooled Standard Deviation)', report.Number(pstats.repeatability),
                     report.Number(pstats.repeat_degf, fmin=0)])
        rows.append(['Reproducibility Standard Deviation', report.Number(pstats.reproducibility),
                     report.Number(pstats.reprod_degf, fmin=0)])
        rows.append(['Reproducibility Significant?', str(stderr.reprod_significant), '&nbsp;'])
        rows.append(['Estimate of Standard Deviation of the Mean', report.Number(stderr.stderr),
                     report.Number(stderr.stderr_degf, fmin=0)])
        rpt = report.Report(**kwargs)
        rpt.table(rows, hdr=['Statistic', 'Value', 'Degrees of Freedom'])
        return rpt

    def anova(self, **kwargs):
        ''' Analysis of variance '''
        aresult = self.result.anova
        hdr = ['Source', 'SS', 'MS', 'F', 'F crit (95%)', 'p-value']
        rows = [['Between Groups', report.Number(aresult.sumsq_between), report.Number(aresult.mean_sumsq_between),
                report.Number(aresult.f, fmt='decimal'), report.Number(aresult.fcrit, fmt='decimal'),
                report.Number(aresult.p, fmt='decimal')],
                ['Within Groups', report.Number(aresult.sumsq_within), report.Number(aresult.mean_sumsq_within), '&nbsp;', '&nbsp;', '&nbsp;'],
                ['Total', report.Number(aresult.sumsq_between+aresult.sumsq_within), '&nbsp;', '&nbsp;', '&nbsp;', '&nbsp;']]
        rpt = report.Report(**kwargs)
        rpt.table(rows, hdr=hdr)
        rpt.table([['F < Fcrit?', str(aresult.f < aresult.fcrit)], ['p > 0.05?', str(aresult.p > 0.05)]],
                  hdr=['Test', 'Statistically equivalent (95%)?'])
        return rpt

    def conformance(self, **kwargs):
        ''' Probability of conformance '''
        rpt = report.Report(**kwargs)
        if self.result.uncertainty.poc is not None:

            rpt.hdr('Probability of Conformance', level=2)
            rpt.add('Tolerance: ', str(self.result.tolerance), end='\n\n')
            rpt.add(
                'Total Probability of Conformance: ', report.Number(self.result.uncertainty.poc*100, fmin=1, postfix=' %')
            )
            rpt.newline()

            rows = []
            for i, poc in enumerate(self.result.groups.pocs):
                rows.append([str(self.result.colnames[i]),
                             report.Number(poc*100, fmin=1, postfix=' %')])
            rpt.table(rows, ['Group', 'Probability of Conformance'])

        else:
            rpt.txt('No tolerance defined\n\n')
        return rpt

    def correlation(self, **kwargs):
        ''' Correlation between columns '''
        rpt = report.Report(**kwargs)
        if len(self.result.colnames) < 2:
            rpt.txt('Add columns to compute correlation.')
            return rpt

        corr = self.result.correlation
        names = self.result.colnames
        rows = []
        for name, corrow in zip(names, corr):
            rows.append([name] + [report.Number(f) for f in corrow])
        rpt.table(rows, hdr=['&nbsp;'] + names)
        return rpt

    def autocorrelation(self, column_name=None, **kwargs):
        ''' Report of autocorrelation for one column '''
        idx = self.result.groupidx(column_name)
        acor = self.result.autocorrelation[idx]
        rows = [['r (variance)', report.Number(acor.r, fmin=0)],
                ['r (uncertainty)', report.Number(acor.r_unc, fmin=0)],
                ['nc', str(acor.nc)],
                ['uncertainty', report.Number(acor.uncert)]]
        rpt = report.Report(**kwargs)
        rpt.table(rows, hdr=['Parameter', 'Value'])
        return rpt


class DataSetPlot:
    ''' Plot data set/ANOVA results

        Args:
            datasetmodel: DataSet instance
    '''
    def __init__(self, datasetmodel):
        self.result = datasetmodel

    def groups(self, fig=None):
        ''' Plot each group with errorbars.

            Args:
                fig: Matplotlib figure to plot on
        '''
        _, ax = plotting.initplot(fig)
        gstats = self.result.groups
        x = self.result.colvals
        y = gstats.means
        if len(x) != len(y):
            return  # Nothing to plot
        ks = np.array([ttable.k_factor(0.95, df) for df in gstats.degfs])
        uy = gstats.std_devs * ks

        ax.errorbar(x, y, yerr=uy, marker='o', ls='', capsize=4)
        ax.set_xlabel('Group')
        ax.set_ylabel('Value')

        if self.result.tolerance is not None:
            if np.isfinite(self.result.tolerance.flow):
                ax.axhline(self.result.tolerance.flow, ls='--', color='C3')
            if np.isfinite(self.result.tolerance.fhigh):
                ax.axhline(self.result.tolerance.fhigh, ls='--', color='C3')

    def histogram(self, colname=None, fig=None, fit=None, qqplot=False, bins='sqrt', points=None, interval=None):
        ''' Plot a histogram, with optional distribution fit and qq-plot, of one column

            Args:
                colname (string): Name of column to plot
                fig (plt.Figure): Matplotlib figure to plot on
                fit (string): Name of distribution to fit
                qqplot (bool): Show a Q-Q normal probability plot
                bins (int or string): Number of bins for histogram (see numpy.histogram_bin_edges)
                points (int): Number of points to show in Q-Q plot (reduce for speed)
                interval (float): Confidence interval to plot as vertical lines
        '''
        fig, _ = plotting.initplot(fig)
        data = self.result.group(colname).values
        if colname is None and len(self.result.colnames) > 0:
            colname = self.result.colnames[0]
        plotting.fitdist(data, fit, fig=fig, qqplot=qqplot, bins=bins,
                         points=points, coverage=interval, xlabel=colname,
                         tolerance=self.result.tolerance)

    def autocorrelation(self, colname=None, fig=None, nmax=None, conf=.95):
        ''' Plot autocorrelation vs lag for one column

            Args:
                colname (string): Name of column to plot
                fig (plt.Figure): Matplotlib figure to plot on
                nmax (int): Maximum lag (upper x limit to plot)
                conf (float): Confidence level (0-1) for confidence bands
        '''
        _, ax = plotting.initplot(fig)

        acorr = self.result.group_acorr(colname)
        x = self.result.group(colname).values
        if nmax is None:
            nmax = len(x)

        k = ttable.k_factor(conf, np.inf)
        ax.plot(acorr.rho[:nmax+1], marker='o')
        z = k/np.sqrt(len(x))
        ax.axhline(0, ls='-', color='black')
        ax.axhline(z, ls=':', color='black')
        ax.axhline(-z, ls=':', color='black')
        sig = sigma_rhok(acorr.rho)
        ax.plot(k*sig[:nmax+1], ls='--', color='red')
        ax.plot(-k*sig[:nmax+1], ls='--', color='red')

    def lag(self, colname=None, lag=1, fig=None):
        ''' Plot lag-plot for column

            Args:
                colname (string): Name of column to plot
                fig (plt.Figure): Matplotlib figure to plot on
                lag (int): Lag value to plot
        '''
        _, ax = plotting.initplot(fig)
        x = self.result.group(colname).values
        ax.plot(x[lag:], x[:len(x)-lag], ls='', marker='o')

    def scatter(self, col1, col2, fig=None):
        ''' Scatter plot between two columns

            Args:
                col1 (string): Name of column 1 data
                col2 (string): Name of column 2 data
                fig (plt.Figure): Matplotlib figure to plot on
        '''
        _, ax = plotting.initplot(fig)
        x = self.result.group(col1).values
        y = self.result.group(col2).values
        ax.scatter(x, y, marker='.')
        ax.set_xlabel(col1)
        ax.set_ylabel(col2)
