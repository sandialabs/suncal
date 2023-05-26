''' Functions for handling units registry with Pint.

    Entire application must use a common UnitRegistry instance.
'''

import pint

ureg = pint.UnitRegistry(autoconvert_offset_to_baseunit=True)
ureg.define('inch_H2O = inch * water * g_0 = inH2O = inch_water')
pint.set_application_registry(ureg)  # Allows loading pickles containing pint units
_uregcustom = []  # List of custom unit definitions


Quantity = ureg.Quantity
dimensionless = ureg.dimensionless
parse_units = ureg.parse_units
parse_expression = ureg.parse_expression


def register_units(unitdefs):
    ''' Register unit definitions with Pint's Unit Registry. Return error string
        if a unit definition can't be parsed.
    '''
    global _uregcustom
    errmsg = []
    if unitdefs:
        _uregcustom = []
        for u in unitdefs.splitlines():
            if u:  # Ignore blank lines
                try:
                    ureg.define(u)
                    _uregcustom.append(u)
                except (ValueError, TypeError):
                    errmsg.append(u)
        ureg._build_cache()
    errmsg = '\n\t'.join(errmsg)
    if errmsg:
        errmsg = 'Error parsing units:\n' + errmsg
    return errmsg


def get_customunits():
    ''' Get string of custom unit definitions '''
    return '\n'.join(_uregcustom)


def is_dimensionless(u):
    ''' Check if unit is dimensionless '''
    return u == ureg.dimensionless


def has_units(u):
    ''' Determine if the value has units '''
    return hasattr(u, 'units')


def strip_units(u, reduce=False):
    ''' Remove units if they exist

        Args:
            reduce (bool): Convert units to reduced form
    '''
    if has_units(u):
        if reduce:
            u = u.to_reduced_units()
            if u != 0 and not is_dimensionless(u.units):
                raise ValueError('Reducing a dimensioned quantity')
        u = u.magnitude
    return u


def split_units(u):
    ''' Split magnitude and units

        Returns:
            magnitude (float): The magnitude of the value
            units (Pint or None): Units of the value
    '''
    units = None
    if has_units(u):
        units = u.units
        u = u.magnitude
    return u, units


def make_quantity(value, unit):
    ''' Make a Pint Quantity with the value and (possibly None) unit string '''
    if unit is not None:
        return value * parse_units(unit)
    return value


def convert(value, units):
    ''' Convert value to units, if units are defined '''
    if units is None:
        return value

    if hasattr(value, 'units'):
        if value._get_delta_units() and value._has_compatible_delta(str(units)):
            # value.to(unit) loses delta_ on value.units if unit is not delta
            units = 'delta_' + str(units)
        return value.to(units)

    return make_quantity(value, units)


def convert_dict(values, units):
    ''' Convert dictionary of {name: quantity} to units defined by {name: units} '''
    if units is None:
        return values

    newvalues = {}
    for name, value in values.items():
        if has_units(value) and name in units and units[name] is not None:
            newvalues[name] = convert(value, units[name])
        else:
            newvalues[name] = value
    return newvalues


def match_units(value1, value2):
    ''' Return value1 converted to same units as value2 '''
    if not has_units(value1) or not has_units(value2):
        return value1
    return convert(value1, value2.units)
