''' Test output.py and formatters '''

import os
import sympy
import numpy
import matplotlib.pyplot as plt

from psluncert import output
from psluncert import project


def test_format():
    ''' Test significant figure formatter '''
    r = output.NumFormatter()
    assert r.f(0, n=1, fmt='decimal') == '0'      # Zeros are handled separately
    assert r.f(0, n=2, fmt='decimal') == '0.0'
    assert r.f(0, n=2, fmt='sci') == '0.0e+00'
    assert r.f(1, n=1, fmt='decimal') == '1'
    assert r.f(1, n=2, fmt='decimal') == '1.0'
    assert r.f(1, n=3, fmt='decimal') == '1.00'
    assert r.f(1, n=4, fmt='decimal') == '1.000'
    assert r.f(1, n=1, fmt='sci') == '1e+00'      # Scientific notation
    assert r.f(1, n=2, fmt='sci') == '1.0e+00'
    assert r.f(1, n=3, fmt='sci') == '1.00e+00'
    assert r.f(1, n=4, fmt='sci') == '1.000e+00'
    assert r.f(1.23456E6, n=1, fmt='sci') == '1e+06'
    assert r.f(1.23456E6, n=2, fmt='sci') == '1.2e+06'
    assert r.f(1.23456E6, n=3, fmt='sci') == '1.23e+06'
    assert r.f(1.23456E6, n=4, fmt='sci') == '1.235e+06'   # note rounding
    assert r.f(1.23456E6, n=5, fmt='sci') == '1.2346e+06'
    assert r.f(1.23456E6, n=6, fmt='sci') == '1.23456e+06'
    assert r.f(1.2E0, n=2, fmt='eng') == '1.2e+00'  # Engineering notation
    assert r.f(1.2E1, n=2, fmt='eng') == '12e+00'
    assert r.f(1.2E2, n=2, fmt='eng') == '120e+00'
    assert r.f(1.2E3, n=2, fmt='eng') == '1.2e+03'
    assert r.f(1.2E4, n=2, fmt='eng') == '12e+03'
    assert r.f(1.2E5, n=2, fmt='eng') == '120e+03'
    assert r.f(1.2E0, n=2, fmt='si') == '1.2'       # SI suffix
    assert r.f(1.2E1, n=2, fmt='si') == '12'
    assert r.f(1.2E2, n=2, fmt='si') == '120'
    assert r.f(1.2E3, n=2, fmt='si') == '1.2k'
    assert r.f(1.2E4, n=2, fmt='si') == '12k'
    assert r.f(1.2E5, n=2, fmt='si') == '120k'
    assert r.f(1.2E6, n=2, fmt='si') == '1.2M'
    assert r.f(1.2E-6, n=2, fmt='si') == '1.2u'
    assert r.f(1.23456, n=3) == '1.23'              # Auto format
    assert r.f(123.456, n=3) == '123'
    assert r.f(12345.67, n=3) == '12300'
    assert r.f(123456.7, n=3) == '1.23e+05'         # Number > thresh
    assert r.f(-123456.7, n=3) == '-1.23e+05'       # Negative too
    assert r.f(123456.7, n=3, thresh=6) == '123000'
    assert r.f(1.234567E-6, n=3) == '1.23e-06'
    assert r.f(1.234567E-6, n=3, thresh=7) == '0.00000123'
    assert r.f(numpy.inf, n=3) == 'inf'
    assert r.f(numpy.nan, n=3) == 'nan'
    assert r.f(1234, n=3, fmt='sci', elower=False) == '1.23E+03'
    assert r.f(1234, matchto=1.23, n=3) == '1234.00'
    assert r.f(1234, matchto=1.13, n=2) == '1234.0'
    assert r.f(1E-9, matchto=1E-10, n=2) == '1.00e-09'
    assert r.f(1E-9, matchto=numpy.nan, n=2) == '1.0e-09'  # Default to 2
    assert r.f(.1, matchto=1E-16, n=2) == '0.1000000000'
    assert r.f(.1, matchto=1E-16, n=2, matchtolim=4) == '0.1000'

    # Also pull defaults from report when n, etc not specified
    assert r.f(1.23456E6) == '1.2e+06'
    r = output.NumFormatter(fmt='sci', n=4)
    assert r.f(1.23456E6) == '1.235e+06'

    # Test f_array method
    r = output.NumFormatter()
    assert r.f_array([1232, 1233]) == ['1232', '1233']
    assert r.f_array([1000, 1001]) == ['1000', '1001']
    assert r.f_array([120000, 120010], fmt='si') == ['120.000k', '120.010k']
    assert r.f_array([900.00001, 900.00002]) == ['900.000010', '900.000020']
    assert r.f_array([100, 100, 100]) == ['100', '100', '100']  # Should work with zero delta
    assert r.f_array([1.1, 2.2, 3.3, 4.4], n=3) == ['1.10', '2.20', '3.30', '4.40']
    assert r.f_array([2.2, 3.3, 4.4, 1.1], n=3) == ['2.20', '3.30', '4.40', '1.10']  # Algorithm sorts values, make sure original order is preserved


def test_runreports():
    ''' Generate reports of each type. Can be manually verified. '''
    u = project.Project.from_configfile('test/project.yaml')  # Project file with all calculation types in it
    u.calculate()

    os.makedirs('testreports', exist_ok=True)
    # Plain text
    with output.report_format(math='text', fig='test'):
        md = u.report_all().get_md()
    with open('testreports/report.txt', 'w') as f:
        f.write(md)

    # HTML/Mathjax/SVG
    with output.report_format(math='mathjax', fig='svg'):
        md = u.report_all().get_html()
    with open('testreports/html_mj_svg.html', 'w') as f:
        f.write(md)

    # HTML/Mathjax/PNG
    with output.report_format(math='mathjax', fig='png'):
        md = u.report_all().get_html()
    with open('testreports/html_mj_png.html', 'w') as f:
        f.write(md)

    # HTML/MPL/SVG
    with output.report_format(math='mpl', fig='svg'):
        md = u.report_all().get_html()
    with open('testreports/html_mpl_svg.html', 'w') as f:
        f.write(md)

    if output.pandoc_path:
        md = u.report_all()
        err = md.save_odt('testreports/report.odt')
        assert err is None or err == ''
        err = md.save_docx('testreports/report.docx')
        assert err is None or err == ''
        if output.latex_path:
            err = md.save_pdf('testreports/report.pdf')
            assert err is None or err == ''
