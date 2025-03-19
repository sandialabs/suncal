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
                'enabled from the menu. X-values may be entered as numbers or calendar dates.\n\n')
        rpt.hdr('Model Tab', level=3)
        rpt.txt('Use the `Model` tab to set form of the curve to fit. The '
                '`Custom` model may be selected to enter arbitrary functions of x, '
                'with the fit parameters given other variable names. Select between '
                'minimizing Vertical Distances (standard regression) and minimizing '
                'Orthogonal Distances (works better for some models, and results in '
                'the same fit whether fitting to y versus x or x versus y.) \n\n')
        rpt.hdr('Tolerances Tab', level=3)
        rpt.txt('This tab allows entering tolerances for each curve fit coefficient.'
                'For enabled coefficients, the probability of conformance with the tolerance '
                'will be computed.\n\n')
        rpt.hdr('Predictions Tab', level=3)
        rpt.txt('Use this tab to enter x-values at which to predict the y-value and '
                'its uncertainty.\n\n')
        rpt.hdr('Waveform Tab', level=3)
        rpt.txt('Calculation of waveform features, such as maximum value, minimum value, '
                'and threshold crossing times, may be entered on this tab. '
                'Use the "Clip Low" and "Clip High" columns to limit the x-range over which '
                'to calculate each feature. An optional tolerance may be enabled for '
                'computing probability of conformance of the feature to the tolerance.\n\n')
        rpt.hdr('Settings Tab', level=3)
        rpt.txt('The `Settings` tab enables alternate methods for computing uncertainty '
                'of the fit. Typically only Least Squares is needed for good results.\n\n')
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
                'level of confidence.\n\n')
        return rpt

    @staticmethod
    def prediction():
        rpt = report.Report()
        rpt.hdr('Curve Fit - Prediction')
        rpt.txt('This page shows values entered on the "Predictions" tab predicted '
                'along the curve. The predicted y value, its confidence interval '
                'and prediction interval are displayed for each point.\n\n')
        return rpt

    @staticmethod
    def waveform():
        rpt = report.Report()
        rpt.hdr('Curve Fit - Waveform')
        rpt.txt('This page shows computed waveform features, entered on the "Waveforms" tab, '
                'such as maximum, minimum, and threshold crossing times. '
                'Uncertainty is calculated using analytical methods from the curve data points.\n\n')
        return rpt

    @staticmethod
    def interval():
        rpt = report.Report()
        rpt.hdr('Curve Fit - Interval')
        rpt.txt('Use this page to condense the prediction band over a portion '
                'of the curve into a single uncertainty value that gives the '
                'entered level of confidence over the entire range of values. '
                'Refer to GUM Appendix F.2 for details on the calculation.\n\n')
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
                'indicate a good fit.\n\n')
        return rpt

    @staticmethod
    def correlations():
        rpt = report.Report()
        rpt.hdr('Curve Fit - Correlations')
        rpt.txt('This page displays the correlation coefficients between fit '
                'parameters in the model.\n\n')
        return rpt

    @staticmethod
    def montecarlo():
        rpt = report.Report()
        rpt.txt('Curve Fit Output - Monte Carlo')
        return rpt