''' Test config load/save
    Usage: run py.test from root folder.
    NOTE: Will reset settings to default.
'''
import pytest

import suncal.gui.configmgr as cfg

s = cfg.Settings()
s.setDefaults()

def test_color():
    s.setColormap('cmapcontour', 'red')
    s.setColormap('cmapscatter', 'yellow')
    assert s.getColormap('cmapcontour') == 'red'
    assert s.getColormap('cmapscatter') == 'yellow'

def test_dist():
    dlist = ['normal', 'uniform', 'alpha']
    s.setDistributions(dlist)
    assert s.getDistributions() == dlist

def test_samples():
    s.setSamples(1000)
    assert s.getSamples() == 1000

    s.setSamples('abc')  # Invalid values go to default
    assert s.getSamples() == 1000000

    s.setSamples(-10)  # Negative values become 1
    assert s.getSamples() == 1

def test_func():
    s.setFunc('sqrt(a)')
    assert s.getFunc() == 'sqrt(a)'

def test_cov():
    vals = [1, 2, 3]
    s.setCoverageMC(vals)

    ivals = s.getCoverageMC()
    assert vals == ivals

def test_default():
    s.setDefaults()
    assert s.getColormap('cmapscatter') == 'viridis'
    assert s.getColormap('cmapcontour') == 'viridis'
    assert s.getFunc() == 'f = x'
    assert s.getSamples() == 1000000
