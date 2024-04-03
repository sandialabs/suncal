''' Units converter interface '''
from pyscript import document, display

from pint import (UnitRegistry, PintError,
                  UndefinedUnitError, DimensionalityError,
                  DefinitionSyntaxError)
from tokenize import TokenError
ureg = UnitRegistry()


def convert(event):
    display('', target='error', append=False)

    qtyfrom = document.querySelector('#qty-from').value
    tounit = document.querySelector('#unit-to').value
    qtyfrom = qtyfrom.replace('^', '**')
    tounit = tounit.replace('^', '**')

    fromvalid = tovalid = False
    try:
        qtyfrom = ureg.Quantity(qtyfrom)
    except (ValueError, DefinitionSyntaxError, UndefinedUnitError,
            DimensionalityError, TokenError, AssertionError):
        vals = qtyfrom.split(maxsplit=1)
        unit = qtyfrom
        if len(vals) > 1:
            unit = vals[1]
        display(f'Unit {unit} is not defined', target='error')
    else:
        fromvalid = True
        display(format(qtyfrom.units, 'P'),
                target='unitname-from', append=False)
        display(format(qtyfrom.dimensionality, 'P'),
                target='dimension-from', append=False)

    try:
        tounit = ureg.Quantity(tounit)
    except (ValueError, DefinitionSyntaxError, UndefinedUnitError,
            DimensionalityError, AssertionError):
        display(f'Unit {tounit} is not defined', target='error', append=True)
    else:
        tovalid = True
        display(format(tounit.units, 'P'), target='unitname-to', append=False)
        display(format(tounit.dimensionality, 'P'),
                target='dimension-to', append=False)

    if fromvalid and tovalid:
        try:
            qtyto = qtyfrom.to(tounit)
        except DimensionalityError:
            display('Dimensionality Mismatch', target='error')
        else:
            display('', target='error', append=False)
            display(format(qtyto.magnitude, 'g'),
                    target='qty-to', append=False)
