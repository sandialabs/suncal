''' Test config load/save
    NOTE: Will reset settings to default.
'''
import pytest

from suncal.gui.gui_settings import gui_settings

gui_settings.set_defaults()


def test_color():
    gui_settings.colormap_contour = 'red'
    gui_settings.colormap_scatter = 'yellow'
    assert gui_settings.colormap_contour == 'red'
    assert gui_settings.colormap_scatter == 'yellow'


def test_dist():
    dlist = ['normal', 'uniform', 'alpha']
    gui_settings.distributions = dlist
    assert gui_settings.distributions == dlist


def test_samples():
    gui_settings.samples = 1000
    assert gui_settings.samples == 1000

    gui_settings.samples = 'abc'  # Invalid values go to default
    assert gui_settings.samples == 1000000

    gui_settings.samples = -10  # Negative values become 1
    assert gui_settings.samples == 1


def test_default():
    gui_settings.set_defaults()
    assert gui_settings.colormap_scatter == 'viridis'
    assert gui_settings.colormap_contour == 'viridis'
    assert gui_settings.samples == 1000000
