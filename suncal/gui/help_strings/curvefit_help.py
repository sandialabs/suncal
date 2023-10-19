''' Inline help reports for CUrveFit tools '''
from ...common import report

class CurveHelp:
    @staticmethod
    def nohelp():
        rpt = report.Report()
        rpt.txt('No help available')
        return rpt

    @staticmethod
    def inputs():
        rpt = report.Report()
        rpt.hdr('Curve Fit - Input')
        rpt.txt('Enter x and y values, and optionally values for uncertainty in y '
                'and uncertainty in x, into the table. Data may be pasted into '
                'the table or loaded from a file. Uncertainty in X must be '
                'enabled from the menu.\n\n')
        rpt.txt('Use the `Model` tab to set form of the curve to fit. The '
                '`Custom` model may be selected to enter arbitrary functions of x, '
                'with the fit parameters given other variable names. Select between '
                'minimizing Vertical Distances (standard regression) and minimizing '
                'Orthogonal Distances (works better for some models, and results in '
                'the same fit whether fitting to y versus x or x versus y.) \n\n'
                'The `Settings` tab enables alternate methods for computing uncertainty '
                'of the fit. Typically only Least Squares is needed for good results.')
        return rpt

    @staticmethod
    def fit():
        rpt = report.Report()
        rpt.hdr('Curve Fit - Results')
        rpt.txt('The best-fit curve of the selected fit model is displayed, both '
                'in graphical form, and as the individual fit parameters with their '
                'uncertainties. The r-statistic and standard error are also computed.\n\n')
        rpt.txt('The Confidence Band shows the zone where the best fit line is expected '
                'at the entered level of confidence. The Prediction Band shows where new '
                'measurements (predictions) along this curve could be expected to the same '
                'level of confidence.')
        return rpt

    @staticmethod
    def prediction():
        rpt = report.Report()
        rpt.hdr('Curve Fit - Prediction')
        rpt.txt('Use this page to make predictions at various x values '
                'along the curve. The predicted y value, its confidence interval '
                'and prediction interval are displayed for each point.')
        return rpt

    @staticmethod
    def interval():
        rpt = report.Report()
        rpt.hdr('Curve Fit - Interval')
        rpt.txt('Use this page to condense the prediction band over a portion '
                'of the curve into a single uncertainty value that gives the '
                'entered level of confidence over the entire range of values. '
                'Refer to GUM Appendix F.2 for details on the calculation.')
        return rpt

    @staticmethod
    def residuals():
        rpt = report.Report()
        rpt.hdr('Curve Fit - Residuals')
        rpt.txt('This page helps the user analyze the goodness of fit by displaying '
                'information about the curve fit residuals. If the model fits the data, '
                'the histogram should be approximately normal, with the raw residuals '
                'appearing randomly above and below the center line. The dots on '
                'the Normal Probability plot should fall along the dotted line to '
                'indicate a good fit.')
        return rpt

    @staticmethod
    def correlations():
        rpt = report.Report()
        rpt.hdr('Curve Fit - Correlations')
        rpt.txt('This page displays the correlation coefficients between fit '
                'parameters in the model.')
        return rpt

    @staticmethod
    def montecarlo():
        rpt = report.Report()
        rpt.txt('Curve Fit Output - Monte Carlo')
        return rpt