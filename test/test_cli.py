''' Test command-line interface '''

import numpy as np

from suncal import Model
from suncal import project
from suncal import reverse
from suncal import risk
from suncal import curvefit
from suncal.common import distributions
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
    u = project.ProjectUncert(Model('f = a * b + c'))
    u.seed = 4848484
    u.model.var('a').measure(10).typeb(std=1)
    u.model.var('b').measure(5).typeb(dist='uniform', a=.5)
    u.model.var('c').measure(3).typeb(unc=3, k=2)
    u.model.variables.correlate('a', 'b', .6)
    u.model.variables.correlate('c', 'b', -.3)
    out = u.calculate()
    report = out.report.summary().get_md(mathfmt='text', figfmt='text')

    cli.main_unc(['f=a*b+c', '--variables', 'a=10', 'b=5', 'c=3', '--uncerts',
                  'a; std=1', 'b; dist=uniform; a=.5', 'c; unc=3; k=2',
                  '--correlate', 'a; b; .6', 'c; b; -.3', '--seed=4848484'])
    report2, err = capsys.readouterr()
    assert report == report2

    # HTML format
    reporthtml = out.report.summary().get_html(mathfmt='latex', figfmt='svg')
    cli.main_unc(['f=a*b+c', '--variables', 'a=10', 'b=5', 'c=3', '--uncerts',
                  'a; std=1', 'b; dist=uniform; a=.5', 'c; unc=3; k=2',
                  '--correlate', 'a; b; .6', 'c; b; -.3', '-f', 'html', '--seed=4848484'])
    report2html, err = capsys.readouterr()
    assert reporthtml == report2html

    # MD format
    reportmd = out.report.summary().get_md(mathfmt='latex', figfmt='svg')
    cli.main_unc(['f=a*b+c', '--variables', 'a=10', 'b=5', 'c=3', '--uncerts',
                  'a; std=1', 'b; dist=uniform; a=.5', 'c; unc=3; k=2',
                  '--correlate', 'a; b; .6', 'c; b; -.3', '-f', 'md', '--seed=4848484'])
    report2md, err = capsys.readouterr()
    assert reportmd == report2md


def test_rev(capsys):
    ''' Test reverse calc '''
    expr = 'rho = w / (k*d**2*h)'
    k = 12.870369  # pi/4*2.54**3, no uncertainty
    h = .5  # inch
    d = .25  # inch
    ud = .001/2
    uh = .001/2

    # Required values for rho
    rho = 2.0  # g/cm3
    urho = .06/2

    np.random.seed(234283742)
    model = reverse.ModelReverse(expr, solvefor='w', targetnom=rho, targetunc=urho)
    u = project.ProjectReverse(model)
    u.model.var('h').measure(h).typeb(std=uh)
    u.model.var('d').measure(d).typeb(std=ud)
    u.model.var('k').measure(k)
    out = u.calculate()
    report = out.report.summary().get_md(mathfmt='text', figfmt='text')

    cli.main_reverse(['rho=w/(k*d**2*h)', f'--target={rho}', f'--targetunc={urho}',
                      '--solvefor=w', '--variables', 'h=.5', 'd=.25', 'k=12.870369', '--uncerts',
                      'h; std=.0005', 'd; std=.0005', '--seed=234283742'])
    report2, err = capsys.readouterr()
    assert report == report2

    # html format
    reporthtml = out.report.summary().get_html(mathfmt='latex', figfmt='svg')
    cli.main_reverse(['rho=w/(k*d**2*h)', f'--target={rho}', f'--targetunc={urho}',
                      '--solvefor=w', '--variables', 'h=.5', 'd=.25', 'k=12.870369', '--uncerts',
                      'h; std=.0005', 'd; std=.0005', '-f', 'html', '--seed=234283742'])
    report2html, err = capsys.readouterr()
    assert reporthtml == report2html


def test_risk(capsys):
    ''' Test risk analysis command line '''
    # Normal risk report with test distribution and guardband
    u = project.ProjectRisk()
    u.model.procdist = distributions.get_distribution('normal', loc=0, scale=4)
    u.model.testdist = distributions.get_distribution('normal', loc=0, scale=1)
    u.model.gbofsts = (.2, .2)
    out = u.calculate()
    report = out.report.summary().get_md(mathfmt='text', figfmt='text')
    cli.main_risk(['--procdist', 'loc=0; scale=4', '--testdist', 'loc=0; scale=1',
                   '-LL', '-1', '-UL', '1', '-GBL', '.2', '-GBU', '.2'])
    report2, err = capsys.readouterr()
    assert report == report2

    # Without test distribution
    u = project.ProjectRisk()
    u.model.procdist = distributions.get_distribution('normal', loc=0, scale=4)
    u.model.testdist = None
    out = u.calculate()
    report = out.report.summary().get_md(mathfmt='text', figfmt='text')
    cli.main_risk(['--procdist', 'loc=0; scale=4', '-LL', '-1', '-UL', '1'])
    report2, err = capsys.readouterr()
    assert report == report2

    # With non-normal distribution
    u = project.ProjectRisk()
    u.model.procdist = distributions.get_distribution('uniform', a=2)
    u.model.testdist = distributions.get_distribution('normal', loc=0, scale=0.5)
    out = u.calculate()
    report = out.report.summary().get_md(mathfmt='text', figfmt='text')
    cli.main_risk(['--procdist', 'dist=uniform; a=2; median=0', '--testdist', 'loc=0; scale=.5',
                   '-LL', '-1', '-UL', '1'])
    report2, err = capsys.readouterr()
    assert report == report2

    # With plots/verbose
    u = project.ProjectRisk()
    u.model.procdist = distributions.get_distribution('normal', loc=0, scale=4)
    u.model.testdist = distributions.get_distribution('normal', loc=0, scale=1)
    out = u.calculate()
    report = out.report.all().get_md(mathfmt='text', figfmt='text')
    cli.main_risk(['--procdist', 'loc=0; scale=4', '--testdist', 'loc=0; scale=1', '-LL', '-1', '-UL', '1', '-v'])
    report2, err = capsys.readouterr()
    assert report == report2


def test_curve(capsys):
    ''' Test Curve fit command line '''
    x = np.array([30, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500])
    y = np.array([4642, 4612, 4565, 4513, 4476, 4433, 4389, 4347, 4303, 4251, 4201, 4140, 4100, 4073, 4024, 3999])
    arr = curvefit.Array(x, y)
    fit = curvefit.CurveFit(arr)
    proj = project.ProjectCurveFit(fit)
    out = proj.calculate()
    report = out.report.summary().get_md(mathfmt='text', figfmt='text')

    x = [str(k) for k in x]
    y = [str(k) for k in y]
    cli.main_curvefit(['-x', *x, '-y', *y])
    report2, err = capsys.readouterr()
    assert report == report2
