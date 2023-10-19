''' Inline help reports for Main Project Component Selction window '''
from ...common import report


class MainHelp:
    @staticmethod
    def project_types():
        rpt = report.Report()
        rpt.hdr('Sandia Uncertainty Calculator')
        rpt.hdr('Calculation Types', level=2)
        rpt.txt('- **Uncertainty Propagation**: Calculate uncertainty of a '
                'measurement model using GUM and Monte Carlo methods.\n'
                '- **Reverse Propagation**: Calculate uncertainty required for '
                'one input variable to achieve desired uncertainty of the model. Useful '
                'for selecting equipment capabilities.\n'
                '- **Uncertainty Sweep**: Compute GUM and Monte Carlo uncertainty '
                'over a range of input values\n'
                '- **Reverse Sweep**: Perform reverse uncertainty calculation over a '
                'range of input values\n'
                '- **Data Sets & ANOVA**: Import 1- or 2- dimensional sets of data '
                'for computing repeatability, reproducibility, and analysis of variance (ANOVA).\n'
                '- **Curve Fit**: Find the best-fitting curve through measured data points, '
                'including uncertainty. Works for arbitrary curve models.\n'
                '- **Risk Analysis**: Compute probability of false accept and false reject '
                'for a measurement\n'
                '- **Calibration Intervals**: Find the optimal calibration interval to achieve '
                'the desired end-of-period reliability\n'
                '- **Distribution Explorer**: Generate random samples from probability distributions '
                'and combine using Monte Carlo methods.\n'
                '- **Uncertainty Wizard**: Perform Uncertainty Propagation calculation by entering '
                'the measurement details using a step-by-step interview process.\n\n')
        return rpt