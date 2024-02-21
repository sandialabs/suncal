''' Inline help reports for Risk tools '''
from ...common import report


def outputs():
    rpt = report.Report()
    rpt.hdr('Risk Integral Results', level=3)
    rpt.hdr('Process Risk', level=4)
    rpt.txt('- **Process Risk**: Probability of any DUT being out of tolerance, '
            'without considering a test measurement. Same as 1 - ITP.\n'
            '- **Upper and lower limit risk**: The amount of Process Risk due '
            'to falling above the upper limit or below the lower limit.\n'
            '- **Process capability index (Cpk)**: Metric for estimating quality of the '
            'production process.\n\n')
    rpt.hdr('Specific Risk', level=4)
    rpt.txt('- **TUR (Test Uncertainty Ratio)**: Ratio of tolerance to measurement '
            'uncertainty at 95% level of confidence.\n')
    rpt.txt('- **Measured Value**: Value set by the slider widget for visualizing '
            'results of a specific measurement. Used to determine the "Result" '
            'and "Specific FA Risk" values, but does not affect Global Risk.\n')
    rpt.txt('- **Result**: The Accept or Reject result given the measured value.\n')
    rpt.txt('- **Specific FA Risk**: The specific risk (either False Accept or False Reject) '
            'given the measurement result. Equal to the probability that the true '
            'value is outside the limit when the result was Accept, or inside the '
            'limit when the result was Reject.\n\n')
    rpt.hdr('Global Risk', level=4)
    rpt.txt('- **Total PFA**: Total probability of false accept for the system. '
            'Probability that any DUT is out of tolerance *and* accepted\n')
    rpt.txt('- **Total PFR**: Total probability of false reject for the system. '
            'Probability that any DUT is in tolerance *and* rejected\n')
    rpt.txt('- **Conditional PFA**: Conditional probability of false accept for '
            'the system. Probability that a DUT is out of tolerance *given* it was '
            'accepted. Use the menu *Risk > Conditional PFA* to enable.\n\n')
    rpt.hdr('Guardband Calculator', level=3)
    rpt.txt('Use the menu item *Risk > Calculate Guardband* to automatically '
            'calculate a guardband using different methods. The methods are: \n\n'
            '- Target PFA: Solve for the guardband that results in the entered PFA. '
            'May be conditional or unconditional PFA. Includes options to simultaneously '
            'minimize the PFR, which may result in asymmetric guardbands, and to '
            'allow negative guardbands in cases where the PFA already exceeds the target.\n'
            '- TUR Guardbands: Calculate guardband using a function of TUR, such as the '
            'common root-difference-of-squares method, or the Managed 2% PFA method '
            'sometimes referred to as "Method 6".\n'
            '- Cost Guardbands: Enter the cost of a false accept and false reject, '
            'then the guardband is computed to achieve the minimum total expected cost or '
            'the worst-case maximum cost.\n'
            '- Specific Risk: Set the guardband to achieve maximum worst-case specific risk.\n\n')
    rpt.hdr('Calculations', level=3)
    hdr = ['Value', 'Definition']
    rows = [
        [report.Math.from_latex('C_{pk}'), report.Math.from_latex(r'\min(\frac{T_U - \mu}{3\sigma_0}, \frac{\mu-T_L}{3\sigma_0})')],
        ['TUR', report.Math.from_latex(r'\frac{T_U - T_L}{2U_{95}} = \frac{\pm T}{\pm U_{95}}')],
        ['Specific Risk', report.Math.from_latex(r'Pr(t>T_U | y)')],
        ['PFA', report.Math.from_latex(r'Pr(T_L \leq y \leq T_U \, and \, t < T_L) + Pr(T_L \leq y \leq T_U \, and\,  t > T_U)')],
        ['PFR', report.Math.from_latex(r'Pr(y < T_L \, and \, T_L < t < T_U) + Pr(y > T_U \, and\,  T_L < t < T_U)')],
        [report.Math.from_latex('T_U'), 'Upper Tolerance Limit'],
        [report.Math.from_latex('T_L'), 'Lower Tolerance Limit'],
        [report.Math.from_latex('T'), 'Plus-or-minus tolerance limits'],
        [report.Math.from_latex('U_{95}'), 'Measurement uncertainty at 95% level of confidence'],
        [report.Math.from_latex('t'), 'Measurement result'],
        [report.Math.from_latex('y'), 'True DUT value'],
        [report.Math.from_latex(r'\mu'), 'Average DUT value'],
        [report.Math.from_latex(r'\sigma_0'), 'Standard deviation of DUT distribution'],
        ]
    rpt.table(rows, hdr)
    return rpt


class RiskHelp:
    @staticmethod
    def nohelp():
        rpt = report.Report()
        rpt.txt('No help available')
        return rpt

    @staticmethod
    def simple():
        rpt = report.Report()
        rpt.hdr('Measurement Decision Risk', level=2)
        rpt.txt('Calculate probabilities of false accept and reject given a '
                'measurement uncertainty and expected distribution of DUTs.\n\n')
        rpt.hdr('Simple Mode Inputs', level=3)
        rpt.txt('Simple Mode assumes the DUT and measurement uncertainty '
                'distributions are normal and centered. The parameters defining '
                'the distributions are:\n\n'
                '- **TUR**: Test Uncertainty Ratio (Tolerance divided by Uncertainty)\n'
                '- **ITP**: In-tolerance Probability. Percent of DUTs that are in tolerance '
                'with no test measurement. Typically based on historical data. '
                'Sometimes called End-of-period Reliability (EOPR).\n'
                '- **Guardband Factor**: Multiplier for setting acceptance limits. '
                'The plus/minus acceptance limit is the plus/minus tolerance times '
                'the guardband factor.\n'
                '- **Test Measurement**: Sets the value of a specific measurement. '
                'For calculating specific risks.\n\n')
        rpt.append(outputs())
        return rpt

    @staticmethod
    def full():
        rpt = report.Report()
        rpt.hdr('Measurement Decision Risk', level=2)
        rpt.txt('Calculate probabilities of false accept and reject given a '
                'measurement uncertainty and expected distribution of DUTs.\n\n')
        rpt.hdr('Full Mode Inputs', level=3)
        rpt.txt('Full mode allows arbitrary probability distributions for '
                'the process and measurement uncertainty.\n\n'
                '- **Upper and Lower Specification Limit**: Tolerances defining '
                'an acceptable DUT. Use `inf` or `-inf` for an infinite tolerance '
                'to define a single-sided (minimum or maximum) limit.\n'
                '- **Process Distribution**: Distribution of DUTs, typically based '
                'on historical data. Use the *ITP* button to set the distribution '
                'to achieve a specified ITP value.\n'
                '- **Test Distribution**: Measurement uncertiainty distribution\n'
                '- **Guardband**: Use the checkbox to enable/disable the guardband. '
                'The guardband is entered as a relative offset from the tolerance. '
                'For example, with a Upper Specification Limit of 10 and guardband of 0.5, '
                'the upper acceptance limit is 9.5.\n\n')
        rpt.append(outputs())
        return rpt

    @staticmethod
    def simple_mc():
        rpt = report.Report()
        rpt.hdr('Measurement Decision Risk', level=2)
        rpt.txt('Calculate probabilities of false accept and reject given a '
                'measurement uncertainty and expected distribution of DUTs.\n\n')
        rpt.hdr('Monte Carlo Mode', level=3)
        rpt.txt('Monte Carlo mode calculates the PFA and PFR using Monte Carlo '
                'integration. Inputs are the same as Integral mode.')
        return rpt

    @staticmethod
    def full_mc():
        return RiskHelp.simple_mc()

    @staticmethod
    def gb_sweep():
        rpt = report.Report()
        rpt.hdr('Guardband Sweep', level=2)
        rpt.txt('The Guardband Sweep computes the PFA and PFR over a full '
                'range of guardband factors for the given DUT and measurement '
                'distributions. It may be useful for finding an optimal tradeoff '
                'between false accepts and false rejects.')
        return rpt

    @staticmethod
    def prob_conform():
        rpt = report.Report()
        rpt.hdr('Probability of Conformance', level=2)
        rpt.txt('Given a measurement uncertainty distribution, this '
                'plot shows the probability of a measured DUT at any given value being '
                'in tolerance. This probability does not conisder the prior distrubution '
                'of products, and only requires knowledge of the measurement uncertainty. '
                'One guardband method is to set the acceptance limits that result in '
                'a minimum Probability of Conformance.')
        return rpt

    @staticmethod
    def curves():
        rpt = report.Report()
        rpt.hdr('Risk Curves', level=2)
        rpt.txt('The Risk Curves mode generates sweeps of PFA and PFR over '
                'values of TUR, ITP, guardband factor, and bias. One of these '
                'variables must be selected as the sweep (x) value, and another can '
                'optionally be selected as the step (z) variable. The remaining '
                'variables are set as constants. The sweep values can then be entered '
                'by defining a start value, stop value, and the number of sweep points. '
                'Step values are entered as a comma-separated list of individual values.')
        return rpt
