''' Inline help reports for Main Project Component Selction window '''
from ...common import report


class MainHelp:
    @staticmethod
    def project_types():
        rpt = report.Report()
        rpt.hdr('Sandia Uncertainty Calculator')
        rpt.hdr('Uncertainty Calculation Types', level=2)
        rpt.txt('- **All-in-one Measurement System**: Define a complete measurement system consisting '
                'of one or more measurands, indirect measurement equations, and curve fit calculations.\n'
                '- **Uncertainty Propagation**: Calculate uncertainty of a '
                'measurement model using GUM and Monte Carlo methods.\n'
                'Tolerances may be entered to calculate probability of conformance.\n'
                '- **R&R Data**: Import 1- or 2- dimensional sets of data '
                'for computing repeatability, reproducibility (R&R), and analysis of variance (ANOVA).\n'
                '- **Curve Fit**: Find the best-fitting curve through measured data points, '
                'including uncertainty. Works for arbitrary curve models.\n')
        rpt.hdr('Statistical Tools', level=2)
        rpt.txt(
                '- **Global Risk**: Compute average probability of false accept and false reject '
                'for a measurement\n'
                '- **Calibration Intervals**: Find the optimal calibration interval to achieve '
                'the desired end-of-period reliability\n'
                '- **End-to-end Measurement Quality Assurance**: Assess the capability '
                'of a measurement setup, evaluating uncertainty ratios and global risks.\n')

        rpt.hdr('Other calculations', level=2)
        rpt.txt('Available in the Project > Insert menu.\n\n'
                '- **Reverse Propagation**: Calculate uncertainty required for '
                'one input variable to achieve desired uncertainty of the model. Useful '
                'for selecting equipment capabilities.\n'
                '- **Uncertainty Sweep**: Compute GUM uncertainty over a range of values.\n'
                '- **Distribution Explorer**: Generate random samples from probability distributions '
                'and combine using Monte Carlo methods.\n')
        return rpt
