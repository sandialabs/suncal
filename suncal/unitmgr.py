''' Functions for handling units registry with Pint '''

import pint

ureg = pint.UnitRegistry(autoconvert_offset_to_baseunit=True)
ureg.define('micro- = 1e-6 = µ-')  # Print micro as mu symbol.
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

