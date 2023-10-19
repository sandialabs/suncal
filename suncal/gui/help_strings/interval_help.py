''' Inline help reports for Interval tools '''
from ...common import report


def dataentry():
    rpt = report.Report()
    rpt.hdr('Data Entry', level=2)
    rpt.txt('When creating a new Interval calculation, use the dialog '
            'to select between data entry modes.\n\n')
    rpt.hdr('Individual Calibration Data Entry', level=3)
    rpt.txt('Enter the historical calibration data for one or more assets:\n'
            '- Interval End: End date of the interval (or date of calibration). '
            'Most common date formats are recognized, such as DD/MM/YYYY, or YYYY_MM_DD.\n'
            '- Interval Start: Optional start date of the interval\n'
            '- Pass/Fail: Whether the calibration passed or failed (exclude any guardbanding). '
            'May be entered as "Pass" or "Fail", "1" or "0", "p" or "f".\n\n'
            'Use the + and - buttons to add or remove additional assets\n\n')
    return rpt


class IntervalHelp:
    @staticmethod
    def nohelp():
        rpt = report.Report()
        rpt.txt('No help available')
        return rpt

    @staticmethod
    def test():
        rpt = report.Report()
        rpt.hdr('Test Interval Method (A3)')
        rpt.txt('The Test Interval method may be used when attributes (pass/fail) '
                'data is available and all historical calibration data was over equal '
                'or similar assigned intervals.\n')

        rpt.hdr('Options', level=2)
        rpt.txt('- **Current Assigned Interval**: The interval, in days, currently assigned to the asset.\n'
                '- **Reliability Target**: Desired end-of-period reliability\n'
                '- **Maximum Change Factor**: The new interval is limited to this factor times the current interval\n'
                '- **Minimum Change**: Do not suggest a new interval if the change would be less than this many days\n'
                '- **Minimum Allowed Interval**: Shortest interval that may be suggested\n'
                '- **Maximum Allowed Interval**: Longest interval that may be suggested\n'
                '- **Minimum Rejection Confidence**: Suggest changing the interval when there is at least this much '
                'confidence that the current interval should be rejected.\n'
                '- **Asset Interval Tolerance**: Include historical intervals that were the current interval '
                'plus-or-minus this many days.\n\n'
                )
        rpt.append(dataentry())
        rpt.hdr('Summarized Values Data Entry', level=3)
        rpt.txt('Enter the number of historical calibrations at the assigned interval that were '
                'in-tolerance, and the total number of calibrations.\n\n')

        rpt.hdr('Results', level=2)
        rpt.txt('- **Suggested Interval**: The calculated interval predicted to achieve the '
                'Reliability Target, plus any minimum or maximum constraints as entered in the options.\n'
                '- **Calculated Interval**: The interval calculated to achieve the desired Reliability Target\n'
                '- **Current Interval Rejection Confidence**: Level of confidence for rejecting the current '
                'assigned interval. A new interval is suggested when this is greater than the Minimum Rejection '
                'Confidence entered in the options.\n'
                '- **True Reliability Range**: Range of possible true end-of-period reliability at the current '
                'assigned interval.\n'
                '- **Observed Reliability**: Percent of historical calibrations that were observed in-tolerance\n'
                '- **Number of calibrations used**: Number of historical calibrations meeting the assigned '
                'interval criteria and used in the calculation\n'
                '- **Rejected calibrations**: Number of historical calibrations not meeting the assigned '
                'interval critera, therefore not used in the calculation.\n\n')
        rpt.txt('Refer to NCSLI Recommended Practice 1 for details on the calculation.')
        return rpt

    @staticmethod
    def binomial():
        rpt = report.Report()
        rpt.hdr('Binomial Method (S2)')
        rpt.txt('The Binomial Interval method may be used when attributes (pass/fail) '
                'data is available and historical calibration intervals were of varied length.\n')

        rpt.hdr('Options', level=2)
        rpt.txt('- **Reliability Target**: Desired end-of-period reliability\n'
                '- **Confidence for Interval Range**: Used to compute a range of intervals '
                'where the true interval that achieves the desired reliability is expected to lie.\n'
                '- **Bins**: Historical data must be grouped into bins to form a reliability vs interval '
                'curve. Historical data will be binned into this many points to form that curve.\n'
                '- **Set Bins Manually**: Use this button to manually group historical calibrations into '
                'bins to form the reliability curve\n\n'
                )
        rpt.append(dataentry())
        rpt.hdr('Summarized Values Data Entry', level=3)
        rpt.txt('Manually enter points on the reliability vs. interval curve, including the '
                'interval length, observed reliabilty at that interval length, and the '
                'total number of calibrations done at that interval.\n\n')

        rpt.hdr('Results', level=2)
        rpt.txt('The historical reliability data is fit to several reliability models. For each model, '
                'the following items are computed:\n\n'
                '- **Interval**: The time at which the reliability curve falls below the target\n'
                '- **Rejection Confidence**: Confidence at which this reliability model should be rejected '
                'as the true model. Low numbers are more confident the model is correct.\n'
                '- **F-Test**: Whether the model passes the statistical F-Test\n'
                '- **Figure of Merit**: A figure of merit computed for each model. The "best" model '
                'is the one with the highest figure of merit.\n\n')
        rpt.txt('Refer to NCSLI Recommended Practice 1 for details on the calculation.')
        return rpt

    @staticmethod
    def variables():
        rpt = report.Report()
        rpt.hdr('Variables Method')
        rpt.txt('The Variabls Interval method may be used when calibration values (not simply pass/fail) '
                'data was recorded. Historical calibration intervals of varied length are needed.\n')

        rpt.hdr('Options', level=2)
        rpt.txt('- **Measurement Uncertainty**: The time-of-test uncertainty in new measurements\n'
                '- **Uncertainty k**: Coverage factor for the entered measurement uncertainty\n'
                '- **Next interval as-left value**: The measured value at the beginning of the upcoming interval\n'
                '- **Fit polynomial order**: Polynomial order for the curve fit to asset deviation over time\n\n')
        
        rpt.hdr('Uncertainty Target Options', level=3)
        rpt.txt('- **Maximum Allowed Uncertainty**: Stop the interval when the uncertainty is predicted '
                'to exceed this value\n\n')
        rpt.hdr('Reliability Target Options', level=3)
        rpt.txt('- **Lower and Upper Tolerance Limits**: Stop the interval when the predicted deviation '
                'plus uncertainty exceeds these limits.\n'
                '- **Confidence**: Level of confidence for the uncertainty used to stop the interval\n\n')

        rpt.append(dataentry())
        rpt.hdr('Summarized Values Data Entry', level=3)
        rpt.txt('Manually enter points on the deviation vs. interval curve including the '
                'interval length, and deviation from prior calibration observed at that interval.\n\n')

        rpt.hdr('Results', level=2)
        rpt.txt('- **Interval**: The calculated interval for each method\n'
                '- **Predicted value**: The value predicted at the end of that interval.\n\n')
        rpt.txt('Refer to NASA Handbook 8739 for details on the calculation.')
        return rpt