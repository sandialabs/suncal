''' Inline help reports for Uncertainty Propagation tools '''
from ...common import report


class UncertHelp:
    @staticmethod
    def nohelp():
        rpt = report.Report()
        rpt.txt('No help available')
        return rpt

    @staticmethod
    def inputs():
        rpt = report.Report()
        rpt.hdr('Uncertainty Propagation')
        rpt.hdr('Input Page', level=2)
        rpt.hdr('Measurement Model', level=3)
        rpt.txt('Enter the measurement model as one or more functions (use the + '
                'and - buttons to add or remove functions from the list). ')
        rpt.txt('Use "Excel-like" notation, with operators +, -, *, /, and powers using ^.\n\n')
        rpt.txt('Variables in the model expression:\n\n- Are case sensitive\n- Use underscores for subscripts (R_1)\n')
        rpt.txt('- May be the name of Greek letters ("theta" = θ, "omega" = ω, "Omega" = Ω)\n')
        rpt.txt('- Some variables are interpreted as constants (e, pi, inf)\n')
        rpt.txt('- Cannot be Python keywords (if, else, def, class, etc.)\n')
        rpt.txt('- Constant quantities with units are entered in brackets: [360 degrees]\n\n')
        rpt.txt('Recognized Functions:\n\n - sin, cos, tan\n- asin, acos, atan, atan2\n')
        rpt.txt('- sinh, cosh, tanh, coth\n- asinh, acosh, atanh, acoth\n')
        rpt.txt('- exp, log, log10\n- sqrt, root\n- rad, deg\n\n')
        rpt.txt('If the indicator turns red, Suncal cannot interpret the equation.\n\n')
        rpt.txt('Units: Most common units are recognized as their full name '
                '(such as `centimeter`, `kilogram`) or abbreviations (`cm`, `kg`). '
                'Use `degC` and `degF` for degrees Celsius and Fahrenheit '
                '(`C` and `F` stand for Coulombs and Farads.) If units are left '
                'blank, the values are treated as dimensionless.\n\n')

        rpt.hdr('Measured Values and Uncertainties', level=3)
        rpt.txt('Each variable found in the model equations will be filled '
                'in to this table. For each variable, enter:\n\n'
                '- Measured Value: The measured or expected/nominal value of the variable\n'
                '- Units: Units of measure. If omitted, will be treated as dimensionless.\n'
                '- One or more uncertainty components (add components by right-clicking '
                'the table row):\n'
                '    * Distribution: The probability distribution for the uncertainty component '
                '    * Units: Uncertainty units may be different, but must be compatible '
                "with, the variable's units (example cm and mm)\n"
                '    * Other parameters needed to define the distribution, as displayed.\n\n')

        rpt.hdr('Correlations', level=3)
        rpt.txt('If the variables are correlated, use the + button to add the '
                'correlation coefficient between two variables.\n\n')

        rpt.hdr('Settings', level=3)
        rpt.txt('- Monte Carlo Samples: Specify the number of samples to generate. '
                'Default is 1 million.\n'
                '- Random Seed: Specify a seed value for the pseudo-random number '
                'generator (Mersenne Twister). If None, the seed will be randomized.\n'
                '- Symbolic GUM Solution: Check this box to ignore quantity values and '
                'compute a symbolic solution to the GUM.\n')

        rpt.hdr('Other Tools', level=3)
        rpt.txt('- Use the "Type A Measurement data" option in the Uncertainty menu '
                ' or right-click menu to load measured values from a file or another '
                ' Suncal calculation and calculate the Type A uncertainty.'
                '- Use `Check Units` from the `Uncertainty` menu to analyze '
                'compatibility of measurement units and check for errors.')
        return rpt

    @staticmethod
    def summary():
        rpt = report.Report()
        rpt.hdr('Uncertainty Propagation')
        rpt.hdr('Summary', level=2)
        rpt.txt('The summarized results of the calculation are shown, including '
                'results of the GUM method and Monte Carlo method for comparison\n\n'
                '- Nominal: The expected value of the model function\n'
                '- Std. Uncertainty: Standard uncertainty (k=1) of the function '
                'computed using each method\n'
                '- 95% Coverage: The expanded uncertainty, at 95% coverage\n'
                '- k: The coverage factor used to expand the uncertainty to 95% coverage\n'
                '- Deg. Freedom: Effective degrees of freedom of the result\n\n')
        rpt.txt('Units of the output quantity may be changed by entering different, '
                'but compatible units here.')
        return rpt

    @staticmethod
    def plots():
        rpt = report.Report()
        rpt.hdr('Uncertainty Propagation')
        rpt.hdr('GUM and Monte Carlo Comparison Plot', level=2)
        rpt.txt('This page shows an interactive plot comparing the GUM '
                'and Monte Carlo results.')
        return rpt

    @staticmethod
    def expanded():
        rpt = report.Report()
        rpt.hdr('Uncertainty Propagation')
        rpt.hdr('Expanded Uncertainties', level=2)
        rpt.txt('Uncertainties computed to an expanded Level of Confidence.\n\n')
        rpt.txt('The Monte Carlo expanded interval may be computed in two ways:\n\n'
                '- Symmetric: Choose endpoints of the interval that result in equal '
                'probability above and below the limits.\n'
                '- Shortest: Find the endpoints that result in the shortest possible interval '
                'covering the level of confidence.')
        return rpt

    @staticmethod
    def budget():
        rpt = report.Report()
        rpt.hdr('Uncertainty Propagation')
        rpt.hdr('Uncertainty Budget', level=2)
        rpt.txt('The Input Measurements and Uncertainty Budget tables repeat the '
                'parameters entered into the calculation. The Sensitivity Coefficients '
                'table analyzes the relative contribution from each variable in the model. '
                'Sensitivity columns show the sensitivity coefficients computed for '
                'each method. The Proportion column shows the percent of total uncertainty '
                'due to each variable. Note if correlations are present, the proportions '
                'may not add to 100%.')
        return rpt

    @staticmethod
    def derivation():
        rpt = report.Report()
        rpt.hdr('Uncertainty Propagation')
        rpt.hdr('GUM Derivation', level=2)
        rpt.txt('This page shows the derivation of the GUM equation combined uncertainty '
                'expression. The variables are listed, sensitivity coefficients (partial '
                'derivatives) are shown, along with the combined uncertainty and '
                'an expression for effective degrees of freedom.\n\nCheck the '
                '`Show derivation with values` box to substitute numbers into the expressions.')
        return rpt

    @staticmethod
    def validity():
        rpt = report.Report()
        rpt.hdr('Uncertainty Propagation')
        rpt.hdr('GUM Validity Check', level=2)
        rpt.txt('The GUM Validity report compares the endpoints (at 95% coverage) '
                'of both GUM and Monte Carlo methods to determine if they match within '
                'a tolerance based on significant digits. Refer to Section 8 of GUM '
                'Supplement 1 for details.')
        return rpt

    @staticmethod
    def mcdistribution():
        rpt = report.Report()
        rpt.hdr('Uncertainty Propagation')
        rpt.hdr('Monte Carlo Histogram', level=2)
        rpt.txt('Plot of Monte Carlo results, with a distribution overlay. '
                'The distirbution is considered a good fit to the histogram '
                'when all the points in the Probability Plot fall within the '
                'shaded uncertainty area.')
        return rpt

    @staticmethod
    def montecarlo():
        rpt = report.Report()
        rpt.hdr('Uncertainty Propagation')
        rpt.hdr('Monte Carlo Input Samples', level=2)
        rpt.txt('This page shows the generated Monte Carlo samples for '
                'each input variable. It can be useful to check for '
                'correlations between input variables.')
        return rpt

    @staticmethod
    def converge():
        rpt = report.Report()
        rpt.hdr('Uncertainty Propagation')
        rpt.hdr('Monte Carlo Convergence Plot', level=2)
        rpt.txt('This plot shows the results of the Monte Carlo calculation '
                'as a function of number of samples. If the results '
                'are still changing significantly after the maximum samples, '
                'the number of samples should be increased to improve '
                'the result.')
        return rpt

    @staticmethod
    def reverse_input():
        rpt = UncertHelp.inputs()
        rpt.div()
        rpt.hdr('Reverse Uncertainty Inputs', level=2)
        rpt.txt('- Function: Which function of the model to compute\n'
                '- Target Value: The desired value of the function\n'
                '- Target Uncertainty: The desired uncertainty function\n'
                '- Solve For: The variable within the model to solve for to achieve the '
                'desired value and uncertainty, leaving all other variables fixed.')
        return rpt

    @staticmethod
    def reverse_output():
        rpt = report.Report()
        rpt.hdr('Reverse Uncertainty Propagation')
        rpt.hdr('Results', level=2)
        rpt.txt('Results of the reverse uncertainty calculation, stating the value of '
                'the input variable and its uncertainty required to achieve the target function value. '
                'For the GUM method, the combined uncertainty expression is algebraically sovlved for the '
                'input variable. The Monte Carlo method is performed in reverse, with consideration for '
                'correlation between the input variables and the function result.')
        return rpt

    @staticmethod
    def sweep():
        rpt = UncertHelp.inputs()
        rpt.div()
        rpt.hdr('Uncertainty Sweep Inputs')
        rpt.txt('Use the Sweep + and - buttons to add sweep variables. A sweep parameter may be the:\n\n'
                '- Measured/mean value of a variable\n'
                '- Uncertainty component of a variable (specify the component and parameter)\n'
                '- Degrees of Freedom of a variable\n'
                '- Correlation coefficient between two variables\n\n'
                'When the parameter has been selected, enter the sweep points into the table. Right-click '
                'in the table to allow filling an entire column by start, stop, and step values. '
                'Multiple sweep columns of the same length may be entered to change parameters in parallel. ')
        return rpt

    @staticmethod
    def reversesweep_input():
        rpt = UncertHelp.reverse_input()
        rpt.txt('\n\n')
        rpt.hdr('Uncertainty Sweep Inputs')
        rpt.txt('Use the Sweep + and - buttons to add sweep variables. A sweep parameter may be the:\n\n'
                '- Measured/mean value of a variable\n'
                '- Uncertainty component of a variable (specify the component and parameter)\n'
                '- Degrees of Freedom of a variable\n'
                '- Correlation coefficient between two variables\n\n'
                'When the parameter has been selected, enter the sweep points into the table. Right-click '
                'in the table to allow filling an entire column by start, stop, and step values. '
                'Multiple sweep columns of the same length may be entered to change parameters in parallel. ')
        return rpt
