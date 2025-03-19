''' Inline help reports for Measurement System function '''
from ...common import report


class SystemHelp:
    @staticmethod
    def nohelp():
        rpt = report.Report()
        rpt.txt('No help available')
        return rpt

    @staticmethod
    def main():
        rpt = report.Report()
        rpt.hdr('Measurement System')
        rpt.txt('A measurement system consists of measured quantities, calculations on measured quantities, '
                'and curve fit measurements and calculations. Each of these may be added to the quantities '
                'list using the buttons in the top left corner.\n\n')

        rpt.hdr('Measured Quantities', level=2)
        rpt.txt('This section provides entry for direct-measurement quantities.\n\n'
                '- **Symbol**: Mathematical symbol for the quantity (for use by other calculated quantities)\n'
                '- **Value**: The measured value of the quantity.\n'
                '- **Units**: Measurement units\n'
                '- **Tolerance**: If the measurement is used to verify conformance to a tolerance, enter it here. It may '
                'be entered as a plus-or-minus value, a maximum or minimum limit, or endpoints of an allowable range. '
                'Used for calculating probability of conformance.\n\n')
        rpt.txt('Use the *A* and *B* buttons to add Type A data '
                'and Type B uncertainty components to the quantity. Type A data is entered in a table of '
                'repeatability (1xN) or reproducibility (NxM) values. The average of the data is used to set the '
                'Value column. Type B components are entered in the form of probability distributions.\n\n'
                )

        rpt.hdr('Indirect Quantity', level=2)
        rpt.txt('An indirect quantity is calculated from one or more other quantities. Uncertainty '
                'is determined using the GUM approach.\n\n'
                '- **Symbol**: Mathematical symbol to assign this quantity (may be used in other calculated quantities).\n'
                '- **Equation**: The equation used to calculate the quantity value, referencing the symbols of other quantities.\n'
                '- **Units**: Measurement units for the result of the calculation. Must be compatible with the units of the input quantities.\n'
                '- **Tolerance**: If the measurement is used to verify conformance to a tolerance, enter it here. It may '
                'be entered as a plus-or-minus value, a maximum or minimum limit, or endpoints of an allowable range. '
                'Used for calculating probability of conformance.\n\n'
                )

        rpt.hdr('Curve Fit Quantity', level=2)
        rpt.txt('A curve fit calculation may result in multiple quantites calculated using the best-fit curve to the measured data. '
                'Each fitting coefficient (such as slope and intercept) are calculated quantities with uncertainty, '
                'and additional quantites may be added for predictions along the curve.\n\n')

        rpt.txt('Use the **Model** dropdown to enter the type of curve to fit. Coeffients for the model are shown below. ')
        rpt.txt('Use the **Curve Data** button to enter the measured data and units. '
                'Double-click the column headers to rename the variable.\n'
                'Additional columns may be used. If given the same name, additional columns will be used '
                'as repeatability data to add a Type A uncertainty to each row of the variable. '
                'Other uncertainties may be entered by naming the column "u(x)", where *x* is the name of another column. '
                'Alternatively, columns may be calculated from other columns by entering an expression with an equal '
                'sign. For example, "z = y^2" will add a column *z* calculated from another column *y*.\n\n'
                )

        rpt.txt('The fitting coefficients are shown below the model. Each coefficient may be given an initial guess (in the Data column),'
                'and an optional tolerance for calculating probability of conformance.\n\n'
                )

        rpt.txt('If the fit curve is used to predict a y-value at a specific x-value, predictions may be added from '
                'the right-click menu on the curve item in the table. Each prediction value has a name, x-value (Data column), '
                'and an optional tolerance.'
                )
        return rpt

