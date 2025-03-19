''' Inline help reports for MQA function '''
from ...common import report


class MqaHelp:
    @staticmethod
    def nohelp():
        rpt = report.Report()
        rpt.txt('No help available')
        return rpt

    @staticmethod
    def main():
        rpt = report.Report()
        rpt.hdr('Measurement Quality Assurance')
        rpt.txt('The table lists all the measured quantities. Use the button in the upper left to '
                'add a new quantity. A shared equipment list and set of guardbanding policies may be '
                'defined using the buttons in the lower left of the screen.')

        rpt.hdr('Columns', level=2)
        rpt.txt('- **Quantity**: A descriptive name for the quantity\n'
                '- **Testpoint**: The nominal or expected value of the quantity\n'
                '- **Units**: Measurement units of the quantity\n'
                '- **Tolerance**: The tolerance or specification being verified by this measurement. Use the toggle button to'
                ' switch between plus-or-minus value, a minimum or maximum limit, or the endpoints of an allowable range of values.\n'
                '- **Utility**: ENter the end-item utility/performance limits. See below for options.\n'
                '- **EOPR%**: The observed end-of-period reliability in percent. Same as 100 minus '
                'the out-of-tolerance rate. May be calculated from historical data on this quantity '
                ' or estimated from data on a set of like quantities. Use the dropdown menu to switch between a True and an Observed'
                ' reliability (where observed reliability was measured using the same equipment in the equipment column\n'
                '- **Equipment**: Enter the equipment accuracy specification. See below for options.\n'
                '- **Guardband**: Enter the guardbanded acceptance limit. Use the dropdown menu to apply a pre-defined guardbanding policy.\n'
                '- **Measurement**: Enter details about the calibration/measurement process. See below for options.\n'
                '- **Costs**: Enter cost model information in this column. See below for options.\n'
                '- **TAR**: The computed Test Accuracy Ratio (Tolerance divided by Equipment Accuracy).\n'
                '- **TUR**: The computed Test Uncertainty Ratio (Tolerance divided by Total Measurement Uncertainty).\n'
                '- **PFA**: The computed Global (Average) Probability of False Acceptance\n'
                )

        rpt.hdr('Utility', level=3)
        rpt.txt('Defines the end-item performance of the item.\n\n'
                '- **Degrade Onset**: The point at which the utility of the device starts to degrade. Leave unchecked '
                'if the point is the same as the tolerance. Used for calculating the probabiliy of success metric.\n'
                '- **Failure Onset**: The point at which the utility of the device hits zero. Leave unchecked '
                'if the point is the same as the tolerance. Used for calculating the probabiliy of success metric.\n'
                '- **Successful Outcome Probability**: The probability of a successful outcome, given the end item is functional.\n\n')

        rpt.hdr('Equipment', level=3)
        rpt.txt('The equipment accuracy may be entered in different forms.\n\n'
                '- **Tolerance**: Select this option for equipment accuracy specified as a tolerance with a known reliability. Note a reliability of 91.67% corresponds to a uniform distribution.\n'
                '- **Select Equipment**: Accuracy comes from an item in the equipment list.\n'
                '- **Another Quantity**: Accuracy defined by another quantity in the measurement chain. The parent quantity is shown below the item in the table.\n'
                '- **Indirect Measurement**: Accuracy is defined by a set of other measurements combined through a measurement equation.\n'
                )

        rpt.hdr('Measurement/Calibration and Interval', level=2)
        rpt.txt('Properties of the calibration process and interval are entered in this column.\n\n'
                '- **Renewal Policy**: Policy for when to make an adjustment. "Never" means the '
                'UUT is never adjusted or repaired (out-of-tolerance UUTs are discarded). '
                '"Always" readjusts every UUT to nominal at every calibration. "As-Needed" means an adjustment '
                'is made when the UUT is found out of tolerance.\n'
                '- **Repair Limit**: Limits above which a repair is needed.\n'
                '- **Probabiliy of discarding**: If repair limit is exceeded, this is the probability the UUT '
                'is not reparable and will be discarded.\n'
                '- **Pre-Measruement Stress**: Probability distribution representing stress encountered '
                'by the UUT between being sent for calibration and the calibration measurement.\n'
                '- **Post-Measruement Stress**: Probability distribution representing stress encountered '
                'by the UUT between completion of calibraiton and return to service.\n'
                '- **Other Uncertainties**: Enter other measurement uncertainty components as probability distributions.\n'
                '- **Reliability Model**: How the UUT reliability decays over time through its interval. '
                'A model of "None" means the reliability does not decay, and the end-of-period reliability '
                'is the same as beginning-of-period reliability.\n'
                '- **Calibration Interval**: The length, in years, of the calibration interval.\n\n'
                )
        rpt.txt('Use the dropdown to experiment with new intervals using the same reliability decay curve.')

        rpt.hdr('Costs', level=2)
        rpt.txt('Enter the costs of calibration and performance of end-item UUTs.\n\n'
                '- **Calibraiton Cost**: The total cost of performing one calibration/measurement on this item.\n'
                '- **Adjustment Cost**: The total cost of adjusting this item during calibration.\n'
                '- **Repair Cost**: The total cost to repair the item.\n'
                '- **Number of UUTs in Inventory**: The number of identical items in the inventory.\n'
                '- **Spare Readiness Factor**: Readiness (from 0 to 1) of spares needed to cover downtime of the item\n'
                '- **Cost of a UUT**: Cost to startup a spare item\n'
                '- **Spare Startup Cost**: Total cost of a new identical item\n'
                '- **Calibration Downtime**: Time in days required to perform a calibration. Include shipping or storage time.\n'
                '- **Adjustment Downtime**: Time in days required to perform an adjustment of the item.\n'
                '- **Repair Downtime**: Time in days required to perform a repair of the item. Include shipping or storage time.\n'
                '- **Cost of Outcome**: Cost of an unsuccessful outcome of the end item (only shown for end-items)\n'
                '- **Probability of Outcome**: Probability of an unsuccessful outcome given the item fails (only shown for end-items)\n'
                )
        return rpt

    @staticmethod
    def guardband_rules():
        rpt = report.Report()
        rpt.hdr('Guardband Rule Editor')
        rpt.txt("Define a set of guardbanding policies selectable from a Quantity's Guardbanding tab.\n\n"
                'Each policy has a name, a method, and a threshold (TUR or PFA) at which to apply the rule. Methods include:\n\n'
                '- **RDS**: Root difference of squares, ')
        rpt.mathtex(r'GBF = \sqrt{1-1/TUR^2}', end='\n')
        rpt.txt('- **Dobbert**: Risk-managed guardband that acheives < 2% global probably of false accept. '
                'Sometimes known as "Method 6".\n'
                '- **RP10**: Method defined in NCSLI Recommended Practice 10, ')
        rpt.mathtex(r'GBF = 1.25-1/TUR', end='\n')
        rpt.txt('- **U95**: Acceptance limit is tolerance minus uncertainty at 95% level of confidence\n'
                '- **PFA**: Solve for the guardband limits that acheive the desired global probability of false accept.\n'
                '- **CPFA**: Solve for the guardband limits that acheive the desired conditional global probability of false accept.\n'
                )
        return rpt
