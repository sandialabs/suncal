''' Inline help reports for Uncertainty Wizard tool '''
from ...common import report


class WizardHelp:
    @staticmethod
    def page_meas_model():
        rpt = report.Report()
        rpt.hdr('Measurement Model', level=2)
        rpt.txt('The Measurement Model defines the function to calculate, in the form of\n\n')
        rpt.mathtex('y = f(x_1, x_2, ...)', end='\n\n')
        rpt.txt('Enter an equation, including equal sign, in the form "f = x". ')
        rpt.txt('Use "Excel-like" notation, with operators +, -, *, /, and powers using ^.\n\n')
        rpt.txt('Variables:\n\n- Are case sensitive\n- Use underscores for subscripts (R_1)\n')
        rpt.txt('- May be the name of Greek letters ("theta" = θ, "omega" = ω, "Omega" = Ω)\n')
        rpt.txt('- Some variables are interpreted as constants (e, pi, inf)\n')
        rpt.txt('- Cannot be Python keywords (if, else, def, class, etc.)\n\n')
        rpt.txt('Constant quantities with units are entered in brackets: [360 degrees]\n\n')
        rpt.txt('Recognized Functions:\n\n - sin, cos, tan\n- asin, acos, atan, atan2\n')
        rpt.txt('- sinh, cosh, tanh, coth\n- asinh, acosh, atanh, acoth\n')
        rpt.txt('- exp, log, log10\n- sqrt, root\n- rad, deg\n\n')
        rpt.txt('If the indicator turns red, Suncal cannot interpret the equation.')
        return rpt

    @staticmethod
    def page_datatype():
        rpt = report.Report()
        rpt.hdr('Variable Data Type', level=2)
        rpt.txt('Each variable in the measurement model must be assigned a value. This page provides '
                'options for how that value is determined. Select **Single value** when the variable was '
                'a known constant, or provided as one number. Select **Repeatability** if the variable '
                'was determined from a series of K repeated measurements under the same conditions. '
                '**Reproducibility** may be used if the variable was determined using K measurements over '
                'J different conditions - often K measurements per day over J days.')
        return rpt

    @staticmethod
    def page_single():
        rpt = report.Report()
        rpt.hdr('Single Value Measurement', level=2)
        rpt.txt('Enter the value of the variable, along with measurement units if applicable. '
                'Check the **Value** and **Units** indicator below the entry to make sure '
                'the units are interpreted as desired. If the indicator is dashed out, '
                "Suncal doesn't understand the units.")
        return rpt

    @staticmethod
    def page_uncerts():
        rpt = report.Report()
        rpt.hdr('Variable Uncertainty Review', level=2)
        rpt.txt('This screen summarizes the information provided about the variable so far, '
                'including the expected (mean) value, combined standard uncertainty, and any '
                'uncertainty components.\n\nA variable may have one Type A uncertainty (calculated '
                'from repeatability and reproducibility values), and any number of Type B '
                'uncertainties. Select the **Add a Type B uncertainty** option to add more Type B '
                'uncertainty components, or **Modify something** to change the parameters of '
                'an already entered component.')
        return rpt

    @staticmethod
    def page_typeb():
        rpt = report.Report()
        rpt.hdr('Type B Uncertainty', level=2)
        rpt.txt('Type B uncertainties are defined using a probability distribution based '
                'on the information that is given about the uncertainty. '
                'On this screen, a distribution type and its parameters may be entered. '
                'The **Name** field may be used for your own housekeeping, but does not '
                'affect the calculation. Choice of distribution depends on the information '
                'available about the uncertainty:\n\n'
                '- **normal**: The default choice, use when given an uncertainty *with* a '
                'confidence and/or coverage factor\n- **uniform**: Use when given an uncertainty '
                'as a plus-or-minus value with no other information\n- **triangular**: Use when '
                'given an uncertainty as a plus-or-minus value, and the value is more likely '
                'to be near the center of the tolerance range\n- **arcsine**: Use for values '
                'that flucuate sinusoidally\n- **resolution**: Use for entering equipment '
                'resolution. Same shape as uniform.\n\n')
        rpt.txt('The **± Uncertainty** field can do basic calculations. Values may be entered '
                "as a number, as a percent (of the variable's expected value), or as a simple "
                'formula. It also understands "%range()" for entry of manufacturer specifications '
                'given as "%reading + %range" values. For example, enter "1% + 0.01%range(100)" for '
                'an un uncertainty on the 100V range of a meter.\n\n')
        rpt.txt('The **Units** field is automatically populated with the same units as the variable, '
                'however it may be changed to any compatible units.')
        rpt.txt('**Degrees of Freedom** may be entered if known, but is often assumed infinite '
                'for Type B uncertainties')
        return rpt

    @staticmethod
    def page_repeat():
        rpt = report.Report()
        rpt.hdr('Repeatability Data (Type A Uncertainty)', level=2)
        rpt.txt('The expected (mean) value and uncertainty of the variable are estimated '
                'using the repeated measurements. The plot shows a histogram of the repeated '
                'measurement values for reference. The uncertainty is calculated as the '
                'standard error of the mean (standard deviation divided by the square '
                'root of the number of measurements).\n\n'
                'The screen provides a field to enter any measurement units for the measurements. '
                'Check the **Use this data to estimate uncertainty** box if the repeated measurements '
                'were part of a repeatability study to apply the uncertainty to other future identical '
                'measurements. Leave it unchecked to compute the uncertainty in that specific set '
                'of repeated measurements. The **Edit Data** button provides an option to change '
                'the measurement data.\n\n')
        rpt.txt('When there are more than 30 measurements in the data, Suncal will check '
                'for autocorrelation between the measurements and apply an uncertainty '
                'correction if autocorrelation is significant. Uncheck **Autocorrelation appears '
                'significant** to disable this correction.')
        return rpt

    @staticmethod
    def page_reprod():
        rpt = report.Report()
        rpt.hdr('Reproducibility Data (Type A Uncertainty)', level=2)
        rpt.txt('The expected (mean) value and uncertainty of the variable are estimated '
                'using the reproducibility measurements. The plot shows a the mean and standard '
                'deviation of each measurement group for reference. The uncertainty is calculated '
                'using the reproducibility if significant, or repeatability otherwise. See GUM '
                'Appendix H.2 for an example.\n\n'
                'The screen provides a field to enter any measurement units for the measurements. '
                'Check the **Use this data to estimate uncertainty** box if the measurements '
                'were part of a reproducibility study to apply the uncertainty to other future identical '
                'measurements. Leave it unchecked to compute the uncertainty in that specific set '
                'of repeated measurements. The **Edit Data** button provides an option to change '
                'the measurement data.\n\n')
        return rpt
    
    @staticmethod
    def page_units():
        rpt = report.Report()
        rpt.hdr('Units Conversion', level=2)
        rpt.txt('Suncal determines the units of the measurment function automatically. '
                'This screen provides an entry to convert the output to different, but compatible, '
                'units. The unit converter has no way to unambiguously determine the units '
                'of the output, so for example, it may suggest units of "V/A" instead of "Ω" when '
                "computing resistance using Ohm's Law. This page allows you to change those units.")
        return rpt
    
    @staticmethod
    def page_summary():
        rpt = report.Report()
        rpt.hdr('Uncertainty Model Review', level=2)
        rpt.txt('At this point, everything is set up and ready to calculate. A summary of the '
                'measurement model and variables is provided here. Use the **Back** button '
                'if anything needs to be changed, otherwise go foward to run the calculation '
                'and see the results.')
        return rpt
    
    @staticmethod
    def page_varselect():
        rpt = report.Report()
        rpt.hdr('Modify a Value', level=2)
        rpt.txt('Select the variable to modify.')
        return rpt
