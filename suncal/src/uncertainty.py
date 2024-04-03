''' Pyscript interface to Suncal Uncertainty Propagation '''
from pyscript import document, window, display
from js import alert, Object, console
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

    def update_model(self, event=None):
        ''' Update the model in real-time, without commiting it yet '''
        document.querySelector('#model-preview').innerHTML = ''
        mathinnerHTMLs = []
        value = document.querySelector('#model').value
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
                document.querySelector('#button-model').disabled = True
                allvalid = False
            else:
                mathtex = uparser.parse_math_with_quantities_to_tex(expr)
                if name:
                    mathtex = f'{name} = {mathtex}'
                mathinnerHTMLs.append(katex.renderToString(mathtex))
        document.querySelector('#model-preview').innerHTML = '<br>'.join(mathinnerHTMLs)
        document.querySelector('#button-model').disabled = not allvalid
        return exprs

    def set_model(self, event=None):
        ''' Commit the model and show variable entry '''
        # Model was already checked for validity
        exprs = self.update_model()
        if exprs:
            savedvars = self.model.variables.variables if self.model else {}
            savedcor = correlation_list(self.model.variables) if self.model else {}
            self.model = Model(*exprs)
            for v in self.model.variables.names:
                if v in savedvars:
                    self.model.variables.variables[v] = savedvars[v]

            for (var1, var2), correlation in savedcor.items():
                self.model.variables.correlate(var1, var2, correlation)

            select = document.querySelector('#variable-select')
            select.innerHTML = ''
            for v in self.model.variables.names:
                select.innerHTML += f'<option value="{v}">{v}</option>'
            self.select_variable()

            correlationlist = document.querySelector('#correlation-vars')
            correlationlist.innerHTML = ''
            for i, v1 in enumerate(self.model.variables.names):
                for v2 in self.model.variables.names[i+1:]:
                    correlationlist.innerHTML += f'<option value="{v1},{v2}">{v1} &harr; {v2}</option>'

            document.querySelector('#measured-values').classList.remove('input-hidden')
            document.querySelector('#div-calculate').classList.remove('input-hidden')

    def fill_variable(self):
        ''' Fill the variable entries for selected variable '''
        varname = document.querySelector('#variable-select').value
        var = self.model.var(varname)
        value = document.querySelector('#varvalue')
        expected = str(var.expected)
        value.value = expected

        display(str(var.uncertainty), target='varuncert', append=False)
        display(str(var.degrees_freedom), target='var-degf', append=False)

    def set_var_value(self, event=None):
        ''' Set measured/expected value of the selected variable '''
        varname = document.querySelector('#variable-select').value
        var = self.model.var(varname)
        value = document.querySelector('#varvalue').value
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

    def add_component(self, event=None):
        ''' Add a new uncertianty component '''
        varname = document.querySelector('#variable-select').value
        var = self.model.var(varname)
        typebnames = var.typeb_names
        try:
            num = int(typebnames[-1][1:])  # uXXX
        except IndexError:
            num = 0
        uname = f'u{num+1}'

        document.querySelector('#uncert-box').classList.remove('input-hidden')###.remove_class('input-hidden')  # Must be visible or chaning select won't work
        document.querySelector('#component-select').innerHTML += f'<option value="{uname}">{uname}</option>'
        document.querySelector('#component-select').value = uname
        document.querySelector('#distribution').value = 'normal'
        document.querySelector('#uncertvalue').value = '1'
        document.querySelector('#coverage').value = '2'
        document.querySelector('#confidence').value = '95.45%'
        document.querySelector('#degf').value = 'inf'
        var.typeb(name=uname, unc=1, k=2)
        document.querySelector('#uncertunits').value = str(var._typeb[-1].units) if var._typeb[-1].units else ''
        self.fill_variable()

    def select_variable(self, event=None):
        ''' Selected Variable was changed '''
        varname = document.querySelector('#variable-select').value
        var = self.model.var(varname)
        document.querySelector('#component-select').innerHTML = ''
        if len(var.typeb_names) == 0:
            document.querySelector('#uncert-box').classList.add('input-hidden')
        else:
            document.querySelector('#uncert-box').classList.remove('input-hidden')
            for uname in var.typeb_names:
                document.querySelector('#component-select').innerHTML += f'<option value="{uname}">{uname}</option>'
            self.select_component()
        self.fill_variable()

    def select_component(self, event=None):
        ''' Selected uncertainty component was changed '''
        varname = document.querySelector('#variable-select').value
        var = self.model.var(varname)
        uncname = document.querySelector('#component-select').value
        unc = var.get_typeb(uncname)

        document.querySelector('#distribution').value = unc.distname
        document.querySelector('#uncertvalue').value = str(unc.kwargs.get('unc', unc.kwargs.get('a', 1)))
        document.querySelector('#coverage').value = str(unc.kwargs.get('k', '-'))
        document.querySelector('#confidence').value = str(unc.kwargs.get('conf', '-'))
        document.querySelector('#degf').value = str(unc.degf)

    def set_comp_value(self, event):###cov=None):
        ''' Store entered values of uncertainty component '''
        cov = event.target.id
        varname = document.querySelector('#variable-select').value
        var = self.model.var(varname)
        uncname = document.querySelector('#component-select').value
        oldunc = var.get_typeb(uncname)

        try:
            distname = document.querySelector('#distribution').value

            degf = float(document.querySelector('#degf').value)
            k = float(document.querySelector('#coverage').value)
            if cov in ['coverage', 'degf']:
                conf = f'{ttable.confidence(k, degf)*100:.2f}%'
                document.querySelector('#confidence').value = conf
            elif cov == 'confidence':
                conf = float(document.querySelector('#confidence').value.rstrip('%')) / 100
                k = ttable.k_factor(conf, degf)
                document.querySelector('#coverage').value = format(k, '.2f')

            kwargs = {'k': k, 'degf': degf, 'units': document.querySelector('#uncertunits').value}
            if distname == 'normal':
                kwargs['unc'] = document.querySelector('#uncertvalue').value
            else:
                kwargs['a'] = document.querySelector('#uncertvalue').value

            var.typeb(name=uncname, dist=distname, **kwargs)
        except (ValueError, PintError) as exc:
            display(str(exc), target='uncert-entry-error', append=False)
        else:
            display('', target='uncert-entry-error', append=False)
            var._typeb.remove(oldunc)
            self.fill_variable()

        if distname == 'normal':
            document.querySelector('#uncert-label').innerHTML = 'Uncertainty'
            document.querySelector('#cov-input').classList.remove('input-hidden')
            document.querySelector('#cov-label').classList.remove('input-hidden')
            document.querySelector('#cov-help').classList.remove('input-hidden')
            document.querySelector('#conf-input').classList.remove('input-hidden')
            document.querySelector('#conf-label').classList.remove('input-hidden')
            document.querySelector('#conf-help').classList.remove('input-hidden')
        else:
            document.querySelector('#uncert-label').innerHTML = 'Half-width'
            document.querySelector('#cov-input').classList.add('input-hidden')
            document.querySelector('#cov-label').classList.add('input-hidden')
            document.querySelector('#cov-help').classList.add('input-hidden')
            document.querySelector('#conf-input').classList.add('input-hidden')
            document.querySelector('#conf-label').classList.add('input-hidden')
            document.querySelector('#conf-help').classList.add('input-hidden')

    def calculate(self, event=None):
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
            document.querySelector('#suncal-input').classList.add('input-hidden')
            document.querySelector('#suncal-output').classList.remove('input-hidden')

            if len(self.results.functionnames) > 0:
                document.querySelector('#mcfit-function-select').classList.remove('input-hidden')
                functionselect = document.querySelector('#mcfit-function')
                functionselect.innerHTML = ''
                for i, name in enumerate(self.results.functionnames):
                    functionselect.innerHTML += f'<option value="{i}">{name}</option>'
            else:
                document.querySelector('#mcfit-function-select').classList.add('input-hidden')

            self.output_change()

    def output_change(self, event=None):
        document.querySelector('#output-units').classList.add('input-hidden')
        document.querySelector('#uncert-control-comparison').classList.add('input-hidden')
        document.querySelector('#uncert-control-expanded').classList.add('input-hidden')
        document.querySelector('#uncert-control-derivation').classList.add('input-hidden')
        document.querySelector('#uncert-control-validity').classList.add('input-hidden')
        document.querySelector('#uncert-control-mcfit').classList.add('input-hidden')
        document.querySelector('#uncert-control-mcinputs').classList.add('input-hidden')
        reportname = document.querySelector('#outputpage').value
        if reportname == 'summary':
            document.querySelector('#output-units').classList.remove('input-hidden')
            unitvals = [str(self.results.getunits().get(f)) for f in self.results.functionnames]
            document.querySelector('#output-units-value').value = '; '.join(unitvals)
            rpt = self.results.report.summary_withplots()

        elif reportname == 'comparison':
            joint = False
            if len(self.model.functionnames) > 1:
                document.querySelector('#uncert-control-comparison').classList.remove('input-hidden')
                joint = document.querySelector('#uncert-control-comparison-joint').checked
            rpt = plt.figure()
            if joint:
                self.results.report.plot.joint_pdf(fig=rpt)
            else:
                self.results.report.plot.pdf(fig=rpt)

        elif reportname == 'expanded':
            document.querySelector('#uncert-control-expanded').classList.remove('input-hidden')
            conf = float(document.querySelector('#report-expanded-confs').value) / 100
            rpt = self.results.report.expanded(conf=conf)

        elif reportname == 'budget':
            rpt = self.results.report.allinputs()

        elif reportname == 'derivation':
            document.querySelector('#uncert-control-derivation').classList.remove('input-hidden')
            values = document.querySelector('#uncert-control-derivation-values').checked
            rpt = self.results.report.gum.derivation(solve=values)

        elif reportname == 'validity':
            document.querySelector('#uncert-control-validity').classList.remove('input-hidden')
            figs = int(document.querySelector('#uncert-control-validity-figs').value)
            rpt = self.results.report.validity(ndig=figs)

        elif reportname == 'mcdist':
            document.querySelector('#uncert-control-mcfit').classList.remove('input-hidden')
            functionid = 0
            if len(self.results.functionnames) > 0:
                functionid = int(document.querySelector('#mcfit-function').value)

            distname = document.querySelector('#uncert-control-mcfit-dist').value
            rpt = plt.figure()
            self.results.report.montecarlo.plot.probplot(
                function=self.results.functionnames[functionid],
                distname=distname,
                fig=rpt)

        elif reportname == 'mcinputs':
            document.querySelector('#uncert-control-mcinputs').classList.remove('input-hidden')
            rpt = plt.figure()
            if document.querySelector('#uncert-control-mcinputs-joint').checked:
                self.results.report.montecarlo.plot.variable_scatter(fig=rpt)
            else:
                self.results.report.montecarlo.plot.variable_hist(fig=rpt)
        elif reportname == 'mcconverge':
            rpt = plt.figure()
            self.results.report.montecarlo.plot.converge(fig=rpt)
        else:
            rpt = Report()
            rpt.hdr('TODO')

        display(rpt, target='report', append=False)

        # Katex render the equations.
        # note - `delimiters` argument is a js nested data structure that doesn't
        # convert well from Python list of dicts so it's set up in a <script> on the html page.
        renderMathInElement(document.querySelector('#report'), delimiters=delimiters)

    def enable_correlations(self, event=None):
        ''' Enable/disable correlation controls '''
        if document.querySelector('#enable-correlations').checked:
            document.querySelector('#correlations-entries').classList.remove('input-hidden')
        else:
            document.querySelector('#correlations-entries').classList.add('input-hidden')

    def select_correlation(self, event=None):
        ''' Fill correlation input with stored value for the selected variable pair '''
        v1, v2 = document.querySelector('#correlation-vars').value.split(',')
        corr = self.model.variables.get_correlation_coeff(v1, v2)
        document.querySelector('#correlation-value').value = corr

    def set_correlation(self, event=None):
        ''' Store entered correlation coefficient to the model '''
        try:
            corr = float(document.querySelector('#correlation-value').value)
            v1, v2 = document.querySelector('#correlation-vars').value.split(',')
        except ValueError:
            pass
        else:
            self.model.variables.correlate(v1, v2, corr)

    def change_units(self, event=None):
        ''' Units changed '''
        unitstr = document.querySelector('#output-units-value').value
        units = [u.strip() for u in unitstr.split(';')]
        try:
            self.results.units(**dict(zip(self.results.functionnames, units)))
            self.results.report.summary()
        except PintError as exc:
            alert(str(exc))
        else:
            self.output_change()

    def back(self, event=None):
        ''' Go back to input page '''
        document.querySelector('#suncal-input').classList.remove('input-hidden')
        document.querySelector('#suncal-output').classList.add('input-hidden')

    def restart(self, event=None):
        ''' Clear inputs and restart the calculation '''
        document.querySelector('#model').value = 'f = x'
        self.model = None
        self.update_model()
        document.querySelector('#measured-values').classList.add('input-hidden')
        document.querySelector('#div-calculate').classList.add('input-hidden')
        self.back()


async def file_save(event=None):
    ''' Download a report. Ask for filename before generating report due to time '''
    try:
        options = {"startIn": "downloads", "suggestedName": "report.html"}
        fileHandle = await window.showSaveFilePicker(Object.fromEntries(to_js(options)))
    except Exception as e:
        console.log('Exception: ' + str(e.args))
        return

    footer = '''<br><div>Uncertainty calculated by <a href="https://sandiapsl.github.io">Suncal</a></div>'''
    Report.apply_css = True
    html = uncert.results.report.all().get_html(footer=footer)
    Report.apply_css = False

    file = await fileHandle.createWritable()
    await file.write(html)
    await file.close()
    return


def correlation_list(variables):
    ''' Get parseable dictionary of correlation coefficients
        in the form {(v1, v2): correlation}

        This used to be a method of Variables class in suncal.
    '''
    coeffs = {}
    for idx1, name1 in enumerate(variables.names):
        for idx2, name2 in enumerate(variables.names):
            if name1 < name2:
                coeffs[(name1, name2)] = variables._correlation[idx1, idx2]
    return coeffs


uncert = Uncert()