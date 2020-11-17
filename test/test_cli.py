''' Test command-line interface '''

import os
import numpy as np
from scipy import stats

import suncal as uc
from suncal import project
from suncal import reverse
from suncal import risk
from suncal import curvefit
from suncal import output
from suncal import distributions
from suncal import __main__ as cli

def test_file(capsys):
    ''' Test running a yaml file '''
    fname = 'test/ex_expansion.yaml'
    u = project.Project.from_configfile(fname)
    u.calculate()
    report = u.report_short().get_md(mathfmt='text', figfmt='text')
    cli.main_setup([fname])
    report2, err = capsys.readouterr()
    assert report == report2


def test_uc(capsys):
    ''' Test uncertainty calc '''
    u = uc.UncertCalc('f = a * b + c', seed=4848484)
    u.set_input('a', nom=10, std=1)
    u.set_input('b', nom=5, dist='uniform', a=.5)
    u.set_input('c', nom=3, unc=3, k=2)
    u.correlate_vars('a', 'b', .6)
    u.correlate_vars('c', 'b', -.3)
    out = u.calculate()
    report = out.report().get_md(mathfmt='text', figfmt='text')

    cli.main_unc(['f=a*b+c', '--variables', 'a=10', 'b=5', 'c=3', '--uncerts', 'a; std=1', 'b; dist=uniform; a=.5', 'c; unc=3; k=2', '--correlate', 'a; b; .6', 'c; b; -.3', '--seed=4848484'])
    report2, err = capsys.readouterr()
    assert report == report2

    # HTML format
    reporthtml = out.report().get_html(mathfmt='latex', figfmt='svg')
    cli.main_unc(['f=a*b+c', '--variables', 'a=10', 'b=5', 'c=3', '--uncerts', 'a; std=1', 'b; dist=uniform; a=.5', 'c; unc=3; k=2', '--correlate', 'a; b; .6', 'c; b; -.3', '-f', 'html', '--seed=4848484'])
    report2html, err = capsys.readouterr()
    assert reporthtml == report2html

    # MD format
    reportmd = out.report().get_md(mathfmt='latex', figfmt='svg')
    cli.main_unc(['f=a*b+c', '--variables', 'a=10', 'b=5', 'c=3', '--uncerts', 'a; std=1', 'b; dist=uniform; a=.5', 'c; unc=3; k=2', '--correlate', 'a; b; .6', 'c; b; -.3', '-f', 'md', '--seed=4848484'])
    report2md, err = capsys.readouterr()
    assert reportmd == report2md


def test_rev(capsys):
    ''' Test reverse calc '''
    expr = 'rho = w / (k*d**2*h)'
    k = 12.870369  # pi/4*2.54**3, no uncertainty
    h = .5  # inch
    d = .25 # inch
    ud = .001/2
    uh = .001/2

    # Required values for rho
    rho = 2.0  # g/cm3
    urho = .06/2

    np.random.seed(234283742)
    u = reverse.UncertReverse(expr, solvefor='w', targetnom=rho, targetunc=urho)
    u.set_input('h', nom=h, std=uh)
    u.set_input('d', nom=d, std=ud)
    u.set_input('k', nom=k)
    u.add_required_inputs()
    out = u.calculate()
    report = out.report().get_md(mathfmt='text', figfmt='text')

    cli.main_reverse(['rho=w/(k*d**2*h)', '--target={}'.format(rho), '--targetunc={}'.format(urho), '--solvefor=w', '--variables', 'h=.5', 'd=.25', 'k=12.870369', '--uncerts', 'h; std=.0005', 'd; std=.0005', '--seed=234283742'])
    report2, err = capsys.readouterr()
    assert report == report2

    # html format
    reporthtml = out.report().get_html(mathfmt='latex', figfmt='svg')
    cli.main_reverse(['rho=w/(k*d**2*h)', '--target={}'.format(rho), '--targetunc={}'.format(urho), '--solvefor=w', '--variables', 'h=.5', 'd=.25', 'k=12.870369', '--uncerts', 'h; std=.0005', 'd; std=.0005', '-f', 'html', '--seed=234283742'])
    report2html, err = capsys.readouterr()
    assert reporthtml == report2html


def test_risk(capsys):
    ''' Test risk analysis command line '''
    # Normal risk report with test distribution and guardband
    u = risk.Risk()
    u.set_procdist(distributions.get_distribution('normal', loc=0, scale=4))
    u.set_testdist(distributions.get_distribution('normal', loc=0, scale=1))
    u.set_guardband(.2, .2)
    u.calculate()
    report = u.out.report().get_md(mathfmt='text', figfmt='text')
    cli.main_risk(['--procdist', 'loc=0; scale=4', '--testdist', 'loc=0; scale=1', '-LL', '-1', '-UL', '1', '-GBL', '.2', '-GBU', '.2'])
    report2, err = capsys.readouterr()
    assert report == report2

    # Without test distribution
    u = risk.Risk()
    u.set_procdist(distributions.get_distribution('normal', loc=0, scale=4))
    u.set_testdist(None)
    u.calculate()
    report = u.out.report().get_md(mathfmt='text', figfmt='text')
    cli.main_risk(['--procdist', 'loc=0; scale=4', '-LL', '-1', '-UL', '1'])
    report2, err = capsys.readouterr()
    assert report == report2

    # With non-normal distribution
    u = risk.Risk()
    u.set_procdist(distributions.get_distribution('uniform', a=2))
    u.set_testdist(distributions.get_distribution('normal', loc=0, scale=0.5))
    u.calculate()
    report = u.out.report().get_md(mathfmt='text', figfmt='text')
    cli.main_risk([ '--procdist', 'dist=uniform; a=2; median=0', '--testdist', 'loc=0; scale=.5', '-LL', '-1', '-UL', '1'])
    report2, err = capsys.readouterr()
    assert report == report2

    # With plots/verbose
    u = risk.Risk()
    u.set_procdist(distributions.get_distribution('normal', loc=0, scale=4))
    u.set_testdist(distributions.get_distribution('normal', loc=0, scale=1))
    u.calculate()
    report = u.out.report_all().get_md(mathfmt='text', figfmt='text')
    cli.main_risk(['--procdist', 'loc=0; scale=4', '--testdist', 'loc=0; scale=1', '-LL', '-1', '-UL', '1', '-v'])
    report2, err = capsys.readouterr()
    assert report == report2


def test_curve(capsys):
    ''' Test Curve fit command line '''
    x = np.array([30,100,200,300,400,500,600,700,800,900,1000,1100,1200,1300,1400,1500])
    y = np.array([4642,4612,4565,4513,4476,4433,4389,4347,4303,4251,4201,4140,4100,4073,4024,3999])
    arr = curvefit.Array(x, y)
    fit = curvefit.CurveFit(arr)
    fit.calculate(gum=True, lsq=True)
    report = fit.out.report().get_md(mathfmt='text', figfmt='text')

    x = ['{}'.format(k) for k in x]
    y = ['{}'.format(k) for k in y]
    cli.main_curvefit(['-x', *x, '-y', *y, '--methods', 'lsq', 'gum'])
    report2, err = capsys.readouterr()
    assert report == report2

