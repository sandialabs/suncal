''' Inline help reports for ANOVA tools '''
from ...common import report

def inputs():
    rpt = report.Report()
    rpt.hdr('Data Entry', level=3)
    rpt.txt('Data is grouped by columns. Each column typically represents '
            'a reproducibility condition, such as all measurements made on a single day, '
            'or by a single operator. The values in the column represent repeatability '
            'conditions, such as repeated measurements made on that day or by that operator. '
            'Use the + and - buttons to add or remove data columns, or load data from a '
            'file using the menu. Double-click a column header to change its name, '
            'which could be a date.')
    return rpt


class AnovaHelp:
    @staticmethod
    def nohelp():
        rpt = report.Report()
        rpt.txt('No help available')
        return rpt

    @staticmethod
    def summary():
        rpt = report.Report()
        rpt.hdr('Data Set - Summary')
        rpt.txt('Summary statistics of the data are shown. The first table lists '
                'the statistics for each column/group in the table, including: \n\n'
                '- **Group**: The name of each group, as entered in the table header row\n'
                '- **Mean**: The mean (average) value of each column\n'
                '- **Variance**: The sample variance of each column\n'
                '- **Std. Dev.**: The sample standard deviation of each column\n'
                '- **Std. Error.**: The standard error of each column\n'
                '- **Deg. Freedom**: Degrees of freedom (number of measurements minus one) for each column.\n\n')
    
        rpt.txt('The Pooled Statistics shows statistics over all the groups:\n\n'
                '- **Grand Mean**: Mean of all the individual measurements\n'
                '- **Repeatability**: The pooled standard deviation. Square root of '
                'the average group variances weighted by their degrees of freedom. \n'
                '- **Reproducibility**: The standard deviation of the means of each group\n'
                '- **Reproducibility Significant?**: Whether the reproducibility is statistically '
                'significant compared to repeatability, based on an F-Test.\n'
                '- **Standard Deviation of the Mean**: Standard deviation of the mean '
                'computed using either repeatability or reproducibility, depending on significance.\n\n'
                'Refer to GUM Appendix H.5 for details of the calculations.\n\n')
        return rpt

    @staticmethod
    def histogram():
        rpt = report.Report()
        rpt.hdr('Data Set - Histogram')
        rpt.txt('Use the `Column` dropdown to select which column/group to display in '
                'the histogram. Summary statistics for the column are also displayed. '
                'The `Distribution Fit` dropdown is used to overlay a probability distribution '
                'on the histogram, and the `Probability Plot` checkbox enables a probability plot '
                'showing how well the data fits the selected distribution. For a good fit, '
                'all the data points should fall within the shaded uncertainty region.\n\n')
        return rpt

    @staticmethod
    def correlation():
        rpt = report.Report()
        rpt.hdr('Data Set - Correlation')
        rpt.txt('Displays the correlation coefficients between each column. Use '
                'the `Column` dropdowns to plot one column versus another.')
        return rpt

    @staticmethod
    def autocorrelation():
        rpt = report.Report()
        rpt.hdr('Data Set - Autocorrelation')
        rpt.txt('Displays the autocorrelation (correlation in time) of a column '
                'in the data set.\n\n'
                '- **r (variance)**: Multiplier for converting variance into '
                'autocorrelation-corrected variance\n'
                '- **r (uncertainty)**: Multiplier for converting uncertainty into '
                'autocorrelation-corrected uncertainty\n'
                '- **nc**: Cut-off lag, above which autocorrelation becomes insignificant\n'
                '- **uncertainty**: The autocorrelation-corrected uncertainty of the data\n\n')
        return rpt

    @staticmethod
    def anova():
        rpt = report.Report()
        rpt.hdr('Data Set - Analysis of Variance')
        rpt.txt('One-way analysis of variance of the data table. Shows the '
                'Between-Group and Within-Group Sum of Squares (SS) and variation (MS), '
                'The F-statistic and Critical F value for 95% confidence, and p-statistic. '
                'If F is less than the critical F, and p is greater than 0.05, the groups '
                'may be considered statistically equivalent (generated from the same population).')
        return rpt
