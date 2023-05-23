from pyscript import Element, display
from js import alert, window, Object, console
from pyodide.ffi import to_js

from js import katex, renderMathInElement, delimiters

from pint import PintError
import matplotlib.pyplot as plt

from suncal import Model
from suncal.common import uparser, ttable
from suncal.common.report import Report

Report.apply_css = False  # Use this page's CSS, not Suncal's built-in CSS


class Uncert:
    def __init__(self):
        self.model = None
        self.results = None
        self.update_model()

    def update_model(self):
        ''' Update the model in real-time, without commiting it yet '''
        Element('model-preview').element.innerHTML = ''
        mathinnerHTMLs = []
        value = Element('model').value
        exprs = value.split(';') if ';' in value else [value]

        # Check each expr for validity
        allvalid = True
        for expr in exprs:
            name = None
            if '=' in expr:
                name, expr, *_ = expr.split('=')
                name = name.strip()
            expr.strip()
            try:
                uparser.parse_math_with_quantities(expr, name)
            except ValueError as err:
                display(str(err), target='model-preview', append=True)
                Element('button-model').element.disabled = True
                allvalid = False
            else:
                mathtex = uparser.parse_math_with_quantities_to_tex(expr)
                if name:
                    mathtex = f'{name} = {mathtex}'
                mathinnerHTMLs.append(katex.renderToString(mathtex))
        Element('model-preview').element.innerHTML = '<br>'.join(mathinnerHTMLs)
        Element('button-model').element.disabled = not allvalid
        return exprs

    def set_model(self):
        ''' Commit the model and show variable entry '''
        # Model was already checked for validity
        exprs = self.update_model()
        if exprs:
            savedvars = self.model.variables.variables if self.model else {}
            savedcor = self.model.variables.correlation_list if self.model else {}
            self.model = Model(*exprs)
            for v in self.model.variables.names:
                if v in savedvars:
                    self.model.variables.variables[v] = savedvars[v]

            for (var1, var2), correlation in savedcor.items():
                self.model.variables.correlate(var1, var2, correlation)

            select = Element('variable-select')
            select.element.innerHTML = ''
            for v in self.model.variables.names:
                select.element.innerHTML += f'<option value="{v}">{v}</option>'
            self.select_variable()

            correlationlist = Element('correlation-vars')
            correlationlist.element.innerHTML = ''
            for i, v1 in enumerate(self.model.variables.names):
                for v2 in self.model.variables.names[i+1:]:
                    correlationlist.element.innerHTML += f'<option value="{v1},{v2}">{v1} &harr; {v2}</option>'

            Element('measured-values').remove_class('input-hidden')
            Element('div-calculate').remove_class('input-hidden')

    def fill_variable(self):
        ''' Fill the variable entries for selected variable '''
        varname = Element('variable-select').value
        var = self.model.var(varname)
        value = Element('varvalue')
        expected = str(var.expected)
        value.element.value = expected

        uc = Element('varuncert')
        uc.write(str(var.uncertainty))
        Element('var-degf').write(str(var.degrees_freedom))

    def set_var_value(self):
        ''' Set measured/expected value of the selected variable '''
        varname = Element('variable-select').value
        var = self.model.var(varname)
        value = Element('varvalue').value
        units = None
        if ' ' in value:
            value, units = value.split(' ')

        try:
            value = float(value)
        except (ValueError) as exc:
            alert(str(exc))
            return

        try:
            var.measure(value, units=units)
        except PintError:
            alert(f'Invalid unit `{units}`')

    def add_component(self):
        ''' Add a new uncertianty component '''
        varname = Element('variable-select').value
        var = self.model.var(varname)
        typebnames = var.typeb_names
        try:
            num = int(typebnames[-1][1:])  # uXXX
        except IndexError:
            num = 0
        uname = f'u{num+1}'

        Element('uncert-box').remove_class('input-hidden')  # Must be visible or chaning select won't work
        Element('component-select').element.innerHTML += f'<option value="{uname}">{uname}</option>'
        Element('component-select').element.value = uname
        Element('distribution').element.value = 'normal'
        Element('uncertvalue').element.value = '1'
        Element('coverage').element.value = '2'
        Element('confidence').element.value = '95.45%'
        Element('degf').element.value = 'inf'
        var.typeb(name=uname, unc=1, k=2)
        Element('uncertunits').element.value = str(var._typeb[-1].units) if var._typeb[-1].units else ''
        self.fill_variable()

    def select_variable(self):
        ''' Selected Variable was changed '''
        varname = Element('variable-select').value
        var = self.model.var(varname)
        Element('component-select').element.innerHTML = ''
        if len(var.typeb_names) == 0:
            Element('uncert-box').add_class('input-hidden')
        else:
            Element('uncert-box').remove_class('input-hidden')
            for uname in var.typeb_names:
                Element('component-select').element.innerHTML += f'<option value="{uname}">{uname}</option>'
            self.select_component()
        self.fill_variable()

    def select_component(self):
        ''' Selected uncertainty component was changed '''
        varname = Element('variable-select').value
        var = self.model.var(varname)
        uncname = Element('component-select').value
        unc = var.get_typeb(uncname)

        Element('distribution').element.value = unc.distname
        Element('uncertvalue').element.value = str(unc.kwargs.get('unc', unc.kwargs.get('a', 1)))
        Element('coverage').element.value = str(unc.kwargs.get('k', '-'))
        Element('confidence').element.value = str(unc.kwargs.get('conf', '-'))
        Element('degf').element.value = str(unc.degf)

    def set_comp_value(self, cov=None):
        ''' Store entered values of uncertainty component '''
        varname = Element('variable-select').value
        var = self.model.var(varname)
        uncname = Element('component-select').value
        oldunc = var.get_typeb(uncname)

        try:
            distname = Element('distribution').value

            degf = float(Element('degf').value)
            k = float(Element('coverage').value)
            if cov in ['coverage', 'degf']:
                conf = f'{ttable.confidence(k, degf)*100:.2f}%'
                Element('confidence').element.value = conf
            elif cov == 'confidence':
                conf = float(Element('confidence').value.rstrip('%')) / 100
                k = ttable.k_factor(conf, degf)
                Element('coverage').element.value = format(k, '.2f')

            kwargs = {'k': k, 'degf': degf, 'units': Element('uncertunits').value}
            if distname == 'normal':
                kwargs['unc'] = Element('uncertvalue').value
            else:
                kwargs['a'] = Element('uncertvalue').value

            var.typeb(name=uncname, dist=distname, **kwargs)
        except (ValueError, PintError) as exc:
            Element('uncert-entry-error').write(str(exc))
        else:
            Element('uncert-entry-error').write('')
            var._typeb.remove(oldunc)
            self.fill_variable()

        if distname == 'normal':
            Element('uncert-label').element.innerHTML = 'Uncertainty'
            Element('cov-input').remove_class('input-hidden')
            Element('cov-label').remove_class('input-hidden')
            Element('cov-help').remove_class('input-hidden')
            Element('conf-input').remove_class('input-hidden')
            Element('conf-label').remove_class('input-hidden')
            Element('conf-help').remove_class('input-hidden')
        else:
            Element('uncert-label').element.innerHTML = 'Half-width'
            Element('cov-input').add_class('input-hidden')
            Element('cov-label').add_class('input-hidden')
            Element('cov-help').add_class('input-hidden')
            Element('conf-input').add_class('input-hidden')
            Element('conf-label').add_class('input-hidden')
            Element('conf-help').add_class('input-hidden')

    def calculate(self):
        try:
            self.results = self.model.calculate()
        except ValueError as exc:
            if 'Domain error in arguments' in str(exc):
                alert('Invalid value for Uncertainty')
            else:
                alert(str(exc))
        except PintError as exc:
            alert(str(exc))
        else:
            Element('suncal-input').add_class('input-hidden')
            Element('suncal-output').remove_class('input-hidden')

            if len(self.results.functionnames) > 0:
                Element('mcfit-function-select').remove_class('input-hidden')
                functionselect = Element('mcfit-function')
                functionselect.element.innerHTML = ''
                for i, name in enumerate(self.results.functionnames):
                    functionselect.element.innerHTML += f'<option value="{i}">{name}</option>'
            else:
                Element('mcfit-function-select').add_class('input-hidden')

            self.output_change()

    def output_change(self):
        Element('output-units').add_class('input-hidden')
        Element('uncert-control-comparison').add_class('input-hidden')
        Element('uncert-control-expanded').add_class('input-hidden')
        Element('uncert-control-derivation').add_class('input-hidden')
        Element('uncert-control-validity').add_class('input-hidden')
        Element('uncert-control-mcfit').add_class('input-hidden')
        Element('uncert-control-mcinputs').add_class('input-hidden')
        reportname = Element('outputpage').value
        if reportname == 'summary':
            Element('output-units').remove_class('input-hidden')
            unitvals = [str(self.results.getunits().get(f)) for f in self.results.functionnames]
            Element('output-units-value').element.value = '; '.join(unitvals)
            rpt = self.results.report.summary_withplots()

        elif reportname == 'comparison':
            joint = False
            if len(self.model.functionnames) > 1:
                Element('uncert-control-comparison').remove_class('input-hidden')
                joint = Element('uncert-control-comparison-joint').element.checked
            rpt = plt.figure()
            if joint:
                self.results.report.plot.joint_pdf(fig=rpt)
            else:
                self.results.report.plot.pdf(fig=rpt)

        elif reportname == 'expanded':
            Element('uncert-control-expanded').remove_class('input-hidden')
            conf = float(Element('report-expanded-confs').value) / 100
            rpt = self.results.report.expanded(conf=conf)

        elif reportname == 'budget':
            rpt = self.results.report.allinputs()

        elif reportname == 'derivation':
            Element('uncert-control-derivation').remove_class('input-hidden')
            values = Element('uncert-control-derivation-values').element.checked
            rpt = self.results.report.gum.derivation(solve=values)

        elif reportname == 'validity':
            Element('uncert-control-validity').remove_class('input-hidden')
            figs = int(Element('uncert-control-validity-figs').value)
            rpt = self.results.report.validity(ndig=figs)

        elif reportname == 'mcdist':
            Element('uncert-control-mcfit').remove_class('input-hidden')
            functionid = 0
            if len(self.results.functionnames) > 0:
                functionid = int(Element('mcfit-function').value)

            distname = Element('uncert-control-mcfit-dist').value
            rpt = plt.figure()
            self.results.report.montecarlo.plot.probplot(
                function=self.results.functionnames[functionid],
                distname=distname,
                fig=rpt)

        elif reportname == 'mcinputs':
            Element('uncert-control-mcinputs').remove_class('input-hidden')
            rpt = plt.figure()
            if Element('uncert-control-mcinputs-joint').element.checked:
                self.results.report.montecarlo.plot.variable_scatter(fig=rpt)
            else:
                self.results.report.montecarlo.plot.variable_hist(fig=rpt)
        elif reportname == 'mcconverge':
            rpt = plt.figure()
            self.results.report.montecarlo.plot.converge(fig=rpt)
        else:
            rpt = Report()
            rpt.hdr('TODO')

        Element('report').element.innerHTML = ''
        display(rpt, target='report', append=False)

        # Katex render the equations.
        # note - `delimiters` argument is a js nested data structure that doesn't
        # convert well from Python list of dicts so it's set up in a <script> on the html page.
        renderMathInElement(Element('report').element, delimiters=delimiters)

    def enable_correlations(self):
        ''' Enable/disable correlation controls '''
        if Element('enable-correlations').element.checked:
            Element('correlations-entries').remove_class('input-hidden')
        else:
            Element('correlations-entries').add_class('input-hidden')

    def select_correlation(self):
        ''' Fill correlation input with stored value for the selected variable pair '''
        v1, v2 = Element('correlation-vars').value.split(',')
        corr = self.model.variables.get_correlation_coeff(v1, v2)
        Element('correlation-value').element.value = corr

    def set_correlation(self):
        ''' Store entered correlation coefficient to the model '''
        try:
            corr = float(Element('correlation-value').value)
            v1, v2 = Element('correlation-vars').value.split(',')
        except ValueError:
            pass
        else:
            self.model.variables.correlate(v1, v2, corr)

    def change_units(self):
        ''' Units changed '''
        unitstr = Element('output-units-value').value
        units = [u.strip() for u in unitstr.split(';')]
        try:
            self.results.units(**dict(zip(self.results.functionnames, units)))
            self.results.report.summary()
        except PintError as exc:
            alert(str(exc))
        else:
            self.output_change()

    def back(self):
        ''' Go back to input page '''
        Element('suncal-input').remove_class('input-hidden')
        Element('suncal-output').add_class('input-hidden')

    def restart(self):
        ''' Clear inputs and restart the calculation '''
        Element('model').element.value = 'f = x'
        self.model = None
        self.update_model()
        Element('measured-values').add_class('input-hidden')
        Element('div-calculate').add_class('input-hidden')
        self.back()


async def file_save(report_method):
    ''' Download a report. Ask for filename before generating report due to time

        Args:
            report_method: method for generating the suncal Report instance
    '''
    try:
        options = {"startIn": "downloads", "suggestedName": "report.html"}
        fileHandle = await window.showSaveFilePicker(Object.fromEntries(to_js(options)))
    except Exception as e:
        console.log('Exception: ' + str(e.args))
        return

    footer = '''<br><div>Uncertainty calculated by <a href="https://sandiapsl.github.io">Suncal</a></div>'''
    Report.apply_css = True
    html = report_method().get_html(footer=footer)
    Report.apply_css = False

    file = await fileHandle.createWritable()
    await file.write(html)
    await file.close()
    return
