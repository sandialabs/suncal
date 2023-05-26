''' Report distribution explorer Monte Carlo results '''

import numpy as np

from ...common import report, plotting


class ReportDistExplore:
    ''' Output for distribution explorer

        Args:
            model: DistExplore instance
    '''
    def __init__(self, model):
        self.model = model
        self.plot = PlotDistExplore(self.model)

    @property
    def samples(self):
        ''' Dictionary of name/expr : sample array '''
        return self.model.samplevalues

    def summary(self, **kwargs):
        ''' Generate report of distributions and Monte Carlos '''
        hdr = ['Parameter', 'Mean', 'Median', 'Standard Deviation']
        rows = []
        for name, samples in self.samples.items():
            if samples is not None:
                rows.append([report.Math(name),
                             report.Number(np.mean(samples), fmin=3),
                             report.Number(np.median(samples), fmin=3),
                             report.Number(np.std(samples), ddof=1, fmin=3)])
            else:
                rows.append([report.Math(name), 'N/A', 'N/A', 'N/A'])
        r = report.Report(**kwargs)
        r.table(rows, hdr)
        return r

    def single(self, name, fitparams=None, **kwargs):
        ''' Report stats on a single distribution '''
        samples = np.atleast_1d(self.samples.get(name))
        stdev = samples.std(ddof=1)
        q025, q25, q75, q975 = np.quantile(samples, (.025, .25, .75, .975))

        hdr = ['Parameter', 'Value']
        rows = [['Mean', report.Number(samples.mean(), fmin=3)],
                ['Standard Deviation', report.Number(stdev, fmin=3)],
                ['Standard Uncertainty', report.Number(stdev/np.sqrt(len(samples)), fmin=3)],
                ['N', f'{len(samples):d}'],
                ['Minimum', report.Number(samples.min(), fmin=3)],
                ['First Quartile', report.Number(q25, fmin=3)],
                ['Median', report.Number(np.median(samples), fmin=3)],
                ['Third Quartile', report.Number(q75, fmin=3)],
                ['Maximum', report.Number(samples.max(), fmin=3)],
                ['95% Coverage Interval', f'{report.Number(q025, fmin=3)}, {report.Number(q975, fmin=3)}'],
                ]
        r = report.Report(**kwargs)
        r.table(rows, hdr)

        if fitparams is not None:
            rows = list(zip(fitparams.keys(), [report.Number(x, fmin=3) for x in fitparams.values()]))
            r.hdr('Fit Parameters', level=3)
            r.table(rows, hdr)

        return r

    def all(self, **kwargs):
        ''' Report all values '''
        fitdist = kwargs.get('fitdist', None)
        qqplot = kwargs.get('qqplot', False)
        coverage = kwargs.get('coverage', False)
        r = report.Report(**kwargs)
        for name in self.samples:
            with plotting.plot_figure() as fig:
                self.plot.hist(name, fig=fig, fitdist=fitdist, qqplot=qqplot, coverage=coverage)
                fig.suptitle(f'${name}$')
                r.plot(fig)
            r.add('\n\n')
            r.append(self.single(name, **kwargs))
        return r


class PlotDistExplore:
    ''' Plot monte carlo results from distribution explorer

        Args:
            model: DistExplore instance
    '''
    def __init__(self, model):
        self.model = model

    def hist(self, name, fig=None, **kwargs):
        ''' Plot histogram of the sampled values

            Args:
                name (string): Name of distribution to plot
                fig (plt.Figure): maptlotlib figure to plot on
                fitdist (string): Plot a fit of this distribution to the data
                qqplot (bool): Show a Q-Q probability plot in a second axis
                interval (bool): Show a 95% coverage interval (symmetric) as vertical lines
        '''
        samples = self.model.samplevalues.get(name)
        fitdist = kwargs.get('fitdist', None)
        qqplot = kwargs.get('qqplot', False)
        interval = None
        if kwargs.get('interval', False):
            interval = np.quantile(samples, (0.025, 0.975))

        fig, _ = plotting.initplot(fig)
        if len(np.atleast_1d(samples)) > 1:
            params = plotting.fitdist(samples, distname=fitdist, fig=fig, qqplot=qqplot, coverage=interval)
        else:
            fig.clf()
            ax = fig.add_subplot(1, 1, 1)
            ax.axvline(samples, label='Sample')
            params = None
        return params
