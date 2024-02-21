''' Test project class - saving/loading config file for multiple calculations '''

from io import StringIO
import numpy as np

import suncal
from suncal.common import distributions
from suncal import project
from suncal import sweep
from suncal import reverse
from suncal import curvefit


def test_saveload_fname(tmpdir):
    # Set up a Project with all types of calculations. Save it to a file (both using file NAME
    # and file OBJECT, read back the results, and compare the output report.
    # Make sure to store seeds for everything that does MC.

    np.random.seed(588132535)
    # Set up several project components of the different types
    u = suncal.Model('f = m/(pi*r**2)')
    u.var('m').measure(2).typeb(std=.2)
    u.var('r').measure(1).typeb(std=.1)
    proj1 = project.ProjectUncert(u)
    proj1.seed = 44444

    u2 = suncal.Model('g = m * 5')
    u2.var('m').measure(5).typeb(std=.5)
    proj2 = project.ProjectUncert(u2)
    proj2.seed = 4444

    projrsk = project.ProjectRisk()
    projrsk.model.procdist = distributions.get_distribution('t', loc=.5, scale=1, df=9)

    swp = sweep.UncertSweep(u)
    swp.add_sweep_nom('m', values=[.5, 1, 1.5, 2])
    projswp = project.ProjectSweep(swp)

    x = np.linspace(-10, 10, num=21)
    y = x*2 + np.random.normal(loc=0, scale=.5, size=len(x))
    arr = curvefit.Array(x, y, uy=0.5)
    fit = curvefit.CurveFit(arr)
    projfit = project.ProjectCurveFit(fit)
    projfit.seed = 909090

    rev = reverse.ModelReverse('f = m/(pi*r**2)', targetnom=20, targetunc=.5, solvefor='m')
    rev.var('r').measure(5).typeb(std=.05)
    projrev = project.ProjectReverse(rev)
    projrev.seed = 5555

    revswp = sweep.UncertSweepReverse('f = m/(pi*r**2)', targetnom=20, targetunc=.5, solvefor='m')
    revswp.model.var('r').measure(5).typeb(std=.05)
    revswp.add_sweep_unc('r', values=[.01, .02, .03, .04], comp='Type B', param='std')
    projrevswp = project.ProjectReverseSweep(revswp)

    projexp = project.ProjectDistExplore()
    projexp.seed = 8888
    projexp.model.dists = {'a': distributions.get_distribution('normal', loc=3, scale=2),
                           'b': distributions.get_distribution('uniform', loc=0, scale=2),
                           'a+b': None}
    projexp.get_config()

    projdset = project.ProjectDataSet()
    projdset.setdata(np.random.normal(loc=10, scale=1, size=(5, 5)))
    projdset.setcolnames([1, 2, 3, 4, 5])

    proj = project.Project()
    proj.add_item(proj1)
    proj.add_item(proj2)
    proj.add_item(projrsk)
    proj.add_item(projswp)
    proj.add_item(projfit)
    proj.add_item(projrev)
    proj.add_item(projrevswp)
    proj.add_item(projexp)
    proj.add_item(projdset)
    reportorig = proj.calculate()

    # Save_config given file NAME
    outfile = tmpdir.join('projconfig.yaml')
    proj.save_config(outfile)

    # Save_config given file OBJECT
    outfileobj = StringIO()
    proj.save_config(outfileobj)
    outfileobj.seek(0)

    proj2 = project.Project.from_configfile(outfile)
    reportnew = proj2.calculate()
    assert str(reportorig) == str(reportnew)

    proj3 = project.Project.from_configfile(outfileobj)
    reportobj = proj3.calculate()
    assert str(reportorig) == str(reportobj)
    outfileobj.close()


def test_projrem():
    # Test add/remove items from project
    u = suncal.Model('f = a*b')
    u2 = suncal.Model('g = c*d')
    p1 = project.ProjectUncert(u, name='function1')
    p2 = project.ProjectUncert(u2, name='function2')

    proj = project.Project()
    proj.add_item(p1)
    proj.add_item(p2)
    assert proj.get_names() == ['function1', 'function2']
    proj.rem_item(0)
    assert proj.get_names() == ['function2']

    proj = project.Project()
    proj.add_item(p1)
    proj.add_item(p2)
    proj.rem_item('function2')
    assert proj.get_names() == ['function1']
