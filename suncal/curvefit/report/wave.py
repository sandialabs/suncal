from ...common import report, plotting


class ReportWaveform:
    ''' Report waveform calculations '''
    def __init__(self, results):
        self._results = results
        self.plot = PlotWave(self._results)

    def _repr_markdown_(self):
        return self.summary().get_md()

    def summary(self, **kwargs):
        ''' Report a summary of results '''
        hdr = ['Name', 'Function', 'Value', 'Standard Uncertainty',
               'Uncertainty Region (95%)', 'Tolerance', 'Prob. Conformance']
        rows = []

        for name, result in self._results.features.items():

            funcname = {'max': 'Maximum Value',
                        'min': 'Minimum Value',
                        'pkpk': 'Peak-to-peak',
                        'thresh_rise': 'Rising Threshold Crossing',
                        'thresh_fall': 'Falling Threshold Crossing',
                        'rise': 'Rise Time',
                        'fall': 'Fall Time',
                        'fwhm': 'Full-width Half-Max'}.get(result.calc, '--')

            rows.append([
                name,
                funcname,
                report.Number(result.uncert.nominal, fmin=0, matchto=result.uncert.uncert),
                report.Number(result.uncert.uncert),
                ('(', report.Number(result.uncert.low, matchto=result.uncert.uncert), ', ',
                 report.Number(result.uncert.high, matchto=result.uncert.uncert), ')'),
                str(result.tolerance) if result.tolerance else '&nbsp;',
                report.Number(result.poc*100, fmin=1, postfix=' %') if result.poc is not None else '&nbsp;'
            ])

            if result.calc in ['max', 'min']:
                time = result.uncert.components.get('time')
                rows.append([
                    '&nbsp;',
                    f'Time of {result.calc.title()}imum Value',
                    report.Number(time.nominal, fmin=0, matchto=result.uncert),
                    report.Number(time.uncert),
                    ('(', report.Number(time.low, fmin=0, matchto=result.uncert.uncert), ', ',
                     report.Number(time.high, fmin=0, matchto=result.uncert.uncert), ')'),
                    '&nbsp;', '&nbsp;'
                ])

        rpt = report.Report(**kwargs)
        rpt.table(rows, hdr)
        return rpt


class PlotWave:
    ''' Plot results of waveform features '''
    def __init__(self, results):
        self._results = results
        self.arr = self._results.waveform

    def plot_feature(self, name: str, ax=None, **kwargs):
        ''' Plot a waveform feature by name '''
        prop = self._results.features.get(name)
        if not prop:
            return None
        if prop.calc in ['max', 'min']:
            self.plot_maxmin(name, ax=ax, **kwargs)
        elif prop.calc == 'pkpk':
            self.plot_pkpk(name, ax=ax, **kwargs)
        elif prop.calc in ['rise', 'fall']:
            self.plot_risefall(name, ax=ax, **kwargs)
        elif prop.calc in ['thresh_fall', 'thresh_rise']:
            self.plot_threshold(name, ax=ax, **kwargs)
        elif prop.calc == 'fwhm':
            self.plot_fwhm(name, ax=ax, **kwargs)
        else:
            assert False

    def plot_array(self, ax=None, **kwargs):
        ''' Plot the array '''
        fig, ax = plotting.initplot(ax)
        ax.errorbar(self.arr.x, self.arr.y, self.arr.uy*2, **kwargs)

    def plot_maxmin(self, name: str, value: bool = True,
                    time: bool = True, ax=None, **kwargs):
        ''' Plot Max or Min value '''
        prop = self._results.features.get(name)
        if prop.calc not in ['max', 'min']:
            raise ValueError(f'Cannot plot max/min for type `{prop.calc}`')

        fig, ax = plotting.initplot(ax)
        self.plot_array(ax=ax, **kwargs)

        if value:
            ax.axhline(prop.uncert.nominal, ls=':', color='black', lw=1)
            ax.axhline(prop.uncert.low, ls='--', color='C1')
            ax.axhline(prop.uncert.high, ls='--', color='C1')

        if time:
            t = prop.uncert.components.get('time')
            ax.axvline(t.nominal, ls=':', color='black', lw=1)
            ax.axvline(t.low, ls='--', color='C2')
            ax.axvline(t.high, ls='--', color='C2')

    def plot_pkpk(self, name: str, ax=None, **kwargs):
        ''' Plot Peak-to-Peak value '''
        prop = self._results.features.get(name)
        if prop.calc not in ['pkpk']:
            raise ValueError(f'Cannot plot peak-peak for type `{prop.calc}`')

        fig, ax = plotting.initplot(ax)
        self.plot_array(ax=ax, **kwargs)

        top = prop.uncert.components.get('min')
        bot = prop.uncert.components.get('max')
        ax.axhline(top.nominal, ls=':', color='black', lw=1)
        ax.axhline(top.low, ls='--', color='C1')
        ax.axhline(top.high, ls='--', color='C1')
        ax.axhline(bot.nominal, ls=':', color='black', lw=1)
        ax.axhline(bot.low, ls='--', color='C1')
        ax.axhline(bot.high, ls='--', color='C1')

    def plot_risefall(self, name: str, levels: bool = False, ax=None, **kwargs):
        ''' Plot Max or Min value '''
        prop = self._results.features.get(name)
        if prop.calc not in ['rise', 'fall']:
            raise ValueError(f'Cannot plot rise/fall for type `{prop.calc}`')

        fig, ax = plotting.initplot(ax)
        self.plot_array(ax=ax, **kwargs)

        if levels:
            y10 = prop.uncert.components.get('min')
            y90 = prop.uncert.components.get('max')
            ax.axhline(y10.nominal, ls=':', color='black', lw=1)
            ax.axhline(y10.low, ls='--', color='C2')
            ax.axhline(y10.high, ls='--', color='C2')
            ax.axhline(y90.nominal, ls=':', color='black', lw=1)
            ax.axhline(y90.low, ls='--', color='C2')
            ax.axhline(y90.high, ls='--', color='C2')

        start = prop.uncert.components.get('start')
        end = prop.uncert.components.get('end')
        ax.axvline(start.nominal, ls=':', color='black', lw=1)
        ax.axvline(start.low, ls='--', color='C1')
        ax.axvline(start.high, ls='--', color='C1')
        ax.axvline(end.nominal, ls=':', color='black', lw=1)
        ax.axvline(end.low, ls='--', color='C1')
        ax.axvline(end.high, ls='--', color='C1')

    def plot_threshold(self, name: str, ax=None, **kwargs):
        ''' Plot threshold crossing '''
        prop = self._results.features.get(name)
        if prop.calc not in ['thresh_fall', 'thresh_rise']:
            raise ValueError(f'Cannot plot threshold rise/fall for type `{prop.calc}`')

        fig, ax = plotting.initplot(ax)
        self.plot_array(ax=ax, **kwargs)

        ax.axhline(prop.thresh, ls='--', color='black')
        ax.axvline(prop.uncert.nominal, ls=':', color='black', lw=1)
        ax.axvline(prop.uncert.low, ls='--', color='C1')
        ax.axvline(prop.uncert.high, ls='--', color='C1')

    def plot_fwhm(self, name: str, levels: bool = True, times: bool = True,
                  ax=None, **kwargs):
        ''' Plot full-width-half-maximum '''
        prop = self._results.features.get(name)
        if prop.calc not in ['fwhm']:
            raise ValueError(f'Cannot plot FWHM for type `{prop.calc}`')

        fig, ax = plotting.initplot(ax)
        self.plot_array(ax=ax, **kwargs)

        h = prop.uncert.components.get('h')
        ax.axhline(h, ls='--', color='black')

        if levels:
            top = prop.uncert.components.get('min')
            bot = prop.uncert.components.get('max')
            ax.axhline(top.nominal, ls=':', color='black', lw=1)
            ax.axhline(top.low, ls='--', color='C2')
            ax.axhline(top.high, ls='--', color='C2')
            ax.axhline(bot.nominal, ls=':', color='black', lw=1)
            ax.axhline(bot.low, ls='--', color='C2')
            ax.axhline(bot.high, ls='--', color='C2')

        if times:
            start = prop.uncert.components.get('start')
            end = prop.uncert.components.get('end')
            ax.axvline(start.nominal, ls=':', color='black', lw=1)
            ax.axvline(start.low, ls='--', color='C1')
            ax.axvline(start.high, ls='--', color='C1')
            ax.axvline(end.nominal, ls=':', color='black', lw=1)
            ax.axvline(end.low, ls='--', color='C1')
            ax.axvline(end.high, ls='--', color='C1')
