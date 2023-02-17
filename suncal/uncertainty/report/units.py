''' Report the units compatibility of the Model '''

import numpy as np
import sympy
from pint import DimensionalityError, OffsetUnitCalculusError, UndefinedUnitError

from ...common import unitmgr, report


def _convert_unit(value, units):
    ''' Convert a single value to the units. Adds dimensionless if units is None '''
    if unitmgr.has_units(value) and units is not None:
        return value.to(units)
    return value * unitmgr.ureg.dimensionless


def _function_rows(model, outunits):
    ''' Bulid function rows of units report '''
    rows = []
    for funcname in model.functionnames:
        msg = None
        varnames = model.varnames + list(model.constants.keys())
        funccallable = sympy.lambdify(varnames, model.basesympys[funcname], 'numpy')

        try:
            result = funccallable(**model.variables.expected, **model.constants)
            result = _convert_unit(result, outunits.get(funcname))
        except ZeroDivisionError:
            result = np.inf
            if model.units[funcname]:
                result = result * model.units[funcname]
        except DimensionalityError as err:
            msg = '<font color="red">' + str(err) + '</font>'
            result = None
        except OffsetUnitCalculusError:
            msg = '<font color="red">Ambiguous offset (temerature) units. Try delta_degC.'
            result = None
        except UndefinedUnitError:
            msg = f'<font color="red">Undefined Unit: {model.units[funcname]}</font>'

        if msg:
            rows.append([funcname, msg, '-', '-'])
        else:
            rows.append([
                report.Math(funcname),
                report.Unit(result.units, abbr=False, dimensionless='-'),
                report.Unit(result.units, abbr=True, dimensionless='-'),
                report.Unit(result.units.dimensionality, abbr=False, dimensionless='-')
            ])
    return rows


def _variable_rows(model):
    ''' Build rows for variables in model '''
    rows = []
    for varname in model.varnames:
        variable = model.var(varname)
        _, units = unitmgr.split_units(variable.expected)
        units = unitmgr.ureg.dimensionless if units is None else units
        rows.append([
            report.Math(varname),
            report.Unit(units, abbr=False, dimensionless='-'),
            report.Unit(units, abbr=True, dimensionless='-'),
            report.Unit(units.dimensionality, abbr=False, dimensionless='-')])

        for typeb in variable._typeb:
            typebunits = typeb.units if typeb.units else unitmgr.ureg.dimensionless
            try:
                _convert_unit(typeb.uncertainty, units)
            except OffsetUnitCalculusError:
                rows.append(
                    [report.Math(typeb.name),
                        f'<font color="red">Ambiguous unit {typeb.units}. Try "delta_{typeb.units}".</font>', '-', '-'])
            except DimensionalityError:
                rows.append(
                    [report.Math(typeb.name),
                        f'<font color="red">Cannot convert {typeb.units} to {typeb.units}</font>', '-', '-'])
            else:
                rows.append([report.Math(typeb.name),
                            report.Unit(typebunits, abbr=False, dimensionless='-'),
                            report.Unit(typebunits, abbr=True, dimensionless='-'),
                            report.Unit(typebunits.dimensionality, abbr=False, dimensionless='-')])
    return rows


def units_report(model, outunits, **kwargs):
    ''' Generate a report of units compatibility defined in model

        Args:
            model (Model instance): Measurement Model
            outunits (dict): Dictionary of functionname: units to convert
            **kwargs: arguments passed to Report instance

        Returns:
            suncal.common.report.Report instance
    '''
    hdr = ['Parameter', 'Units', 'Abbreviation', 'Dimensionality']

    rows = _function_rows(model, outunits)
    rows.extend(_variable_rows(model))
    rpt = report.Report(**kwargs)
    rpt.table(rows, hdr)
    return rpt
