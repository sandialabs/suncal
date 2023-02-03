''' Reports for datasets and analysis of variance '''

import numpy as np

from ...common import ttable, report, plotting
from ..dataset import sigma_rhok


class ReportDataSet:
    ''' Report data set/ANOVA results

        Args:
            datasetmodel: DataSet instance
    '''
    def __init__(self, datasetmodel):
        self.model = datasetmodel
        self.plot = DataSetPlot(self.model)

    def summary(self, **kwargs):
        ''' Group stats '''
        rows = []
        names = self.model.colnames
        gstats = self.model.group_stats()
        meanstrs = report.Number.number_array(gstats.mean, fmin=0)
        for g, gmean, gvar, gstd, gsem, df in zip(names, meanstrs, gstats.variance,
                                                  gstats.standarddev, gstats.standarderror, gstats.degf):
            rows.append([str(g), gmean, report.Number(gvar, fmin=0),
                         report.Number(gstd, fmin=0), report.Number(gsem, fmin=0), report.Number(df, fmin=0)])

        rpt = report.Report(**kwargs)
        rpt.table(rows, hdr=['Group', 'Mean', 'Variance', 'Std. Dev.', 'Std. Error', 'Deg. Freedom'])
        return rpt

    def all(self, **kwargs):
        ''' Report summary, pooled statistics, anova '''
        rpt = self.summary(**kwargs)
        rpt.append(self.pooled(**kwargs))
        rpt.append(self.anova(**kwargs))
        return rpt

    def column(self, column_name=None, **kwargs):
        ''' Stats for one column/group '''
        st = self.model.column_stats(column_name)
        dat = self.model.get_column(column_name)
        q025, q25, q75, q975 = np.quantile(dat, (.025, .25, .75, .975))
        rows = [['Mean', report.Number(st.mean, fmin=0)],
                ['Standard Deviation', report.Number(st.standarddev, fmin=3)],
                ['Std. Error of the Mean', report.Number(st.standarderr, fmin=3)],
                ['Deg. Freedom', f'{st.degf:.2f}'],
                ['Minimum',  report.Number(dat.min(), fmin=3)],
                ['First Quartile', report.Number(q25, fmin=3)],
                ['Median', report.Number(np.median(dat), fmin=3)],
                ['Third Quartile', report.Number(q75, fmin=3)],
                ['Maximum', report.Number(dat.max(), fmin=3)],
                ['95% Coverage Interval', f'{report.Number(q025, fmin=3)}, {report.Number(q975, fmin=3)}']
                ]
        rpt = report.Report(**kwargs)
        rpt.table(rows, hdr=['Parameter', 'Value'])
        return rpt

    def pooled(self, **kwargs):
        ''' Pooled statistics and Grand Mean '''
        pstats = self.model.pooled_stats()
        stderr = self.model.standarderror()
        rows = []
        rows.append(['Grand Mean', report.Number(pstats.mean, matchto=pstats.repeatability, fmin=0), '-'])
        rows.append(['Repeatability (Pooled Standard Deviation)', report.Number(pstats.repeatability),
                     report.Number(pstats.repeatability_degf, fmin=0)])
        rows.append(['Reproducibility Standard Deviation', report.Number(pstats.reproducibility),
                     report.Number(pstats.reproducibility_degf, fmin=0)])
        rows.append(['Reproducibility Significant?', str(stderr.reprod_significant), '-'])
        rows.append(['Estimate of Standard Deviation of the Mean', report.Number(stderr.standarderror),
                     report.Number(stderr.degf, fmin=0)])
        rpt = report.Report(**kwargs)
        rpt.table(rows, hdr=['Statistic', 'Value', 'Degrees of Freedom'])
        return rpt

    def anova(self, **kwargs):
        ''' Analysis of variance '''
        aresult = self.model.anova()
        hdr = ['Source', 'SS', 'MS', 'F', 'F crit (95%)', 'p-value']
        rows = [['Between Groups', report.Number(aresult.SSbet), report.Number(aresult.MSbet),
                report.Number(aresult.F, fmt='decimal'), report.Number(aresult.Fcrit, fmt='decimal'),
                report.Number(aresult.P, fmt='decimal')],
                ['Within Groups', report.Number(aresult.SSwit), report.Number(aresult.MSwit), '-', '-', '-'],
                ['Total', report.Number(aresult.SSbet+aresult.SSwit), '-', '-', '-', '-']]
        rpt = report.Report(**kwargs)
        rpt.table(rows, hdr=hdr)
        rpt.table([['F < Fcrit?', str(aresult.F < aresult.Fcrit)], ['p > 0.05?', str(aresult.P > 0.05)]],
                  hdr=['Test', 'Statistically equivalent (95%)?'])
        return rpt

    def correlation(self, **kwargs):
        ''' Correlation between columns '''
        rpt = report.Report(**kwargs)
        if len(self.model.colnames) < 2:
            rpt.txt('Add columns to compute correlation.')
            return rpt

        corr = self.model.correlation()
        names = self.model.colnames
        rows = []
        for name, corrow in zip(names, corr):
            rows.append([name] + [report.Number(f) for f in corrow])
        rpt.table(rows, hdr=['-'] + names)
        return rpt

    def autocorrelation(self, column_name=None, **kwargs):
        ''' Report of autocorrelation for one column '''
        acor = self.model.autocorrelation_uncert(colname=column_name)
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
        self.model = datasetmodel

    def groups(self, fig=None):
        ''' Plot each group with errorbars.

            Args:
                fig: Matplotlib figure to plot on
        '''
        _, ax = plotting.initplot(fig)
        summary = self.model.summarize()
        gstats = summary.group_stats()
        x = summary.colnames_parsed()
        y = gstats.mean
        if len(x) != len(y):
            return  # Nothing to plot
        uy = gstats.standarddev

        ax.errorbar(x, y, yerr=uy, marker='o', ls='', capsize=4)
        if self.model.coltype == 'str':
            ax.set_xticks(x)
            ax.set_xticklabels(self.model.colnames)
        ax.set_xlabel('Group')
        ax.set_ylabel('Value')

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
        data = self.model.get_column(colname)
        if colname is None and len(self.model.colnames) > 0:
            colname = self.model.colnames[0]
        plotting.fitdist(data, fit, fig=fig, qqplot=qqplot, bins=bins,
                         points=points, coverage=interval, xlabel=colname)

    def autocorrelation(self, colname=None, fig=None, nmax=None, conf=.95):
        ''' Plot autocorrelation vs lag for one column

            Args:
                colname (string): Name of column to plot
                fig (plt.Figure): Matplotlib figure to plot on
                nmax (int): Maximum lag (upper x limit to plot)
                conf (float): Confidence level (0-1) for confidence bands
        '''
        _, ax = plotting.initplot(fig)

        x = self.model.get_column(colname)
        rho = self.model.autocorrelation(colname)

        if nmax is None:
            nmax = len(x)

        k = ttable.k_factor(conf, np.inf)
        ax.plot(rho[:nmax+1], marker='o')
        z = k/np.sqrt(len(x))
        ax.axhline(0, ls='-', color='black')
        ax.axhline(z, ls=':', color='black')
        ax.axhline(-z, ls=':', color='black')
        sig = sigma_rhok(rho)
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
        x = self.model.get_column(colname)
        ax.plot(x[lag:], x[:len(x)-lag], ls='', marker='o')

    def scatter(self, col1, col2, fig=None):
        ''' Scatter plot between two columns

            Args:
                col1 (string): Name of column 1 data
                col2 (string): Name of column 2 data
                fig (plt.Figure): Matplotlib figure to plot on
        '''
        _, ax = plotting.initplot(fig)
        x = self.model.get_column(col1)
        y = self.model.get_column(col2)
        ax.scatter(x, y, marker='.')
        ax.set_xlabel(col1)
        ax.set_ylabel(col2)
