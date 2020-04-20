''' Test output.py and formatters '''

import os
import sympy
import numpy
import matplotlib.pyplot as plt

from suncal import report
from suncal import project


def test_format():
    ''' Test significant figure formatter '''
    assert report.Number(0, n=1).string(fmt='decimal') == '0'      # Zeros are handled separately
    assert report.Number(0, n=2).string(fmt='decimal') == '0.0'
    assert report.Number(0, n=2).string(fmt='sci') == '0.0e+00'
    assert report.Number(1, n=1).string(fmt='decimal') == '1'
    assert report.Number(1, n=2).string(fmt='decimal') == '1.0'
    assert report.Number(1, n=3).string(fmt='decimal') == '1.00'
    assert report.Number(1, n=4).string(fmt='decimal') == '1.000'
    assert report.Number(1, n=1).string(fmt='sci') == '1e+00'      # Scientific notation
    assert report.Number(1, n=2).string(fmt='sci') == '1.0e+00'
    assert report.Number(1, n=3).string(fmt='sci') == '1.00e+00'
    assert report.Number(1, n=4).string(fmt='sci') == '1.000e+00'
    assert report.Number(1.23456E6, n=1).string(fmt='sci') == '1e+06'
    assert report.Number(1.23456E6, n=2).string(fmt='sci') == '1.2e+06'
    assert report.Number(1.23456E6, n=3).string(fmt='sci') == '1.23e+06'
    assert report.Number(1.23456E6, n=4).string(fmt='sci') == '1.235e+06'   # note rounding
    assert report.Number(1.23456E6, n=5).string(fmt='sci') == '1.2346e+06'
    assert report.Number(1.23456E6, n=6).string(fmt='sci') == '1.23456e+06'
    assert report.Number(1.2E0, n=2).string(fmt='eng') == '1.2e+00'  # Engineering notation
    assert report.Number(1.2E1, n=2).string(fmt='eng') == '12e+00'
    assert report.Number(1.2E2, n=2).string(fmt='eng') == '120e+00'
    assert report.Number(1.2E3, n=2).string(fmt='eng') == '1.2e+03'
    assert report.Number(1.2E4, n=2).string(fmt='eng') == '12e+03'
    assert report.Number(1.2E5, n=2).string(fmt='eng') == '120e+03'
    assert report.Number(1.2E0, n=2).string(fmt='si') == '1.2'       # SI suffix
    assert report.Number(1.2E1, n=2).string(fmt='si') == '12'
    assert report.Number(1.2E2, n=2).string(fmt='si') == '120'
    assert report.Number(1.2E3, n=2).string(fmt='si') == '1.2k'
    assert report.Number(1.2E4, n=2).string(fmt='si') == '12k'
    assert report.Number(1.2E5, n=2).string(fmt='si') == '120k'
    assert report.Number(1.2E6, n=2).string(fmt='si') == '1.2M'
    assert report.Number(1.2E-6, n=2).string(fmt='si') == '1.2u'
    assert report.Number(1.23456, n=3).string() == '1.23'              # Auto format
    assert report.Number(123.456, n=3).string() == '123'
    assert report.Number(12345.67, n=3).string() == '12300'
    assert report.Number(123456.7, n=3).string() == '1.23e+05'         # Number > thresh
    assert report.Number(-123456.7, n=3).string() == '-1.23e+05'       # Negative too
    assert report.Number(123456.7).string(n=3, thresh=6) == '123000'
    assert report.Number(1.234567E-6).string(n=3) == '1.23e-06'
    assert report.Number(1.234567E-6).string(n=3, thresh=7) == '0.00000123'
    assert report.Number(numpy.inf).string(n=3) == 'inf'
    assert report.Number(numpy.nan).string(n=3) == 'nan'
    assert report.Number(1234).string(n=3, fmt='sci', elower=False) == '1.23E+03'
    assert report.Number(1234, matchto=1.23, n=3).string() == '1234.00'
    assert report.Number(1234, matchto=1.13, n=2).string() == '1234.0'
    assert report.Number(1E-9, matchto=1E-10, n=2).string() == '1.00e-09'
    assert report.Number(1E-9, matchto=numpy.nan, n=2).string() == '1.0e-09'  # Default to 2
    assert report.Number(.1, matchto=1E-16, n=2).string() == '0.1000000000'
    assert report.Number(.1, matchto=1E-16, n=2, matchtolim=4).string() == '0.1000'

    # Also pull defaults from report when n, etc not specified
    assert report.Number(1.23456E6).string() == '1.2e+06'
    assert report.Number(1.23456E6).string(fmt='sci', n=4) == '1.235e+06'

    # Test f_array method
    assert [r.string() for r in report.Number.number_array([1232, 1233])] == ['1232', '1233']
    assert [r.string() for r in report.Number.number_array([1000, 1001])] == ['1000', '1001']
    assert [r.string(fmt='si') for r in report.Number.number_array([120000, 120010])] == ['120.000k', '120.010k']
    assert [r.string() for r in report.Number.number_array([900.00001, 900.00002])] == ['900.000010', '900.000020']

    # Should work with zero delta
    assert [r.string() for r in report.Number.number_array([100, 100, 100])] == ['100', '100', '100']
    assert [r.string(n=3) for r in report.Number.number_array([1.1, 2.2, 3.3, 4.4])] == ['1.10', '2.20', '3.30', '4.40']

    # Algorithm sorts values, make sure original order is preserved
    assert [r.string(n=3) for r in report.Number.number_array([2.2, 3.3, 4.4, 1.1])] == ['2.20', '3.30', '4.40', '1.10']


def test_runreports():
    ''' Generate reports of each type. Can be manually verified. '''
    u = project.Project.from_configfile('test/project.yaml')  # Project file with all calculation types in it
    u.calculate()

    os.makedirs('testreports', exist_ok=True)

    # Plain text, UTF-8 encoding
    md = u.report_all().get_md(mathfmt='text', figfmt='text')
    with open('testreports/report.txt', 'w', encoding='utf-8') as f:
        f.write(md)

    # Markdown, embedded SVG, UTF-8
    md = u.report_all().get_md(mathfmt='latex', figfmt='svg')
    with open('testreports/report_md.md', 'w', encoding='utf-8') as f:
        f.write(md)

    # Markdown, embedded SVG, ANSI
    md = u.report_all().get_md(mathfmt='latex', figfmt='svg', unicode=False)
    with open('testreports/report_ansi.md', 'w', encoding='utf-8') as f:
        f.write(md)

    # HTML/Mathjax/SVG
    md = u.report_all().get_html(mathfmt='latex', figfmt='svg')
    with open('testreports/html_mj_svg.html', 'w', encoding='utf-8') as f:
        f.write(md)

    # HTML/Mathjax/PNG
    md = u.report_all().get_html(mathfmt='latex', figfmt='png')
    with open('testreports/html_mj_png.html', 'w', encoding='utf-8') as f:
        f.write(md)

    # HTML/MPL/SVG
    md = u.report_all().get_html(mathfmt='svg', figfmt='svg')
    with open('testreports/html_mpl_svg.html', 'w', encoding='utf-8') as f:
        f.write(md)

    if report.pandoc_path:
        md = u.report_all()
        err = md.save_odt('testreports/report.odt')
        assert err is None or err == ''
        err = md.save_docx('testreports/report.docx')
        assert err is None or err == ''
        if report.latex_path:
            err = md.save_pdf('testreports/report.pdf')
            assert err is None or err == ''
