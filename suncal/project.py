''' Class for managing a project, that is a list of calculator objects.

    Supported calculator objects include:

    - UncertCalc
    - UncertSweep
    - UncertReverse
    - UncertSweepReverse
    - Risk
    - CurveFit
    - DataSet (ANOVA)

    New calculation objects must support the following methods:

    - calculate()  -- Return an output object with report() method
    - get_config() -- Return a configuration dictionary
    - from_config() -- Build a calculator object from the config dictionary
    - from_configfile()  -- Build a calculator object from the yaml file
'''

from io import StringIO
import shutil
import yaml
from contextlib import suppress

from . import uncertainty
from . import dataset
from . import sweeper
from . import report
from . import curvefit
from . import risk
from . import reverse
from . import dist_explore
from . import intervals


class Project(object):
    ''' Uncertainty Project container. Holds a list of calculation objects
        (i.e. uncertainty propagations, sweeps, risks, datasets, etc.)
    '''
    def __init__(self, items=None):
        self.items = []
        if items is not None:
            self.items = items

    def count(self):
        ''' Get number of items in project '''
        return len(self.items)

    def get_mode(self, index):
        ''' Get calculation mode for the index '''
        item = self.items[index]
        # NOTE: order is important here, for example UncertSweepReverse is an instance of UncertSweep too
        if isinstance(item, sweeper.UncertSweepReverse):
            mode = 'reversesweep'
        elif isinstance(item, sweeper.UncertSweep):
            mode = 'sweep'
        elif isinstance(item, reverse.UncertReverse):
            mode = 'reverse'
        elif isinstance(item, uncertainty.UncertCalc):
            mode = 'uncertainty'
        elif isinstance(item, risk.Risk):
            mode = 'risk'
        elif isinstance(item, curvefit.CurveFit):
            mode = 'curvefit'
        elif isinstance(item, dataset.DataSet) or isinstance(item, dataset.DataSetSummary):
            mode = 'data'
        elif isinstance(item, dist_explore.DistExplore):
            mode = 'distributions'
        elif isinstance(item, intervals.attributes.BinomialInterval):
            mode = 'intervalbinom'
        elif isinstance(item, intervals.attributes.BinomialIntervalAssets):
            mode = 'intervalbinomasset'
        elif isinstance(item, intervals.attributes.TestInterval):
            mode = 'intervaltest'
        elif isinstance(item, intervals.attributes.TestIntervalAssets):
            mode = 'intervaltestasset'
        elif isinstance(item, intervals.variables.VariablesInterval):
            mode = 'intervalvariables'
        elif isinstance(item, intervals.variables.VariablesIntervalAssets):
            mode = 'intervalvariablesasset'
        else:
            raise ValueError('Unknown item {}'.format(item))
        return mode

    def add_item(self, item):
        ''' Add calculator item to project.

            Parameters
            ----------
            item: object
                Item to add. Should be a valid uncertainty calculation object,
                such as UncertCalc, UncertSweep, etc.

            Notes
            -----
            Valid project items must have methods for save_config(), from_config(), and calculate().
        '''
        item.project = self  # Add reference to project the item is in (useful for building project tree)
        self.items.append(item)

    def rem_item(self, item):
        ''' Remove item from project

            Parameters
            ----------
            item: int or string
                If int, will remove item at index[int]. If string, will remove first item with
                that name.
        '''
        try:
            self.items.pop(item)
        except TypeError:
            names = self.get_names()
            self.items.pop(names.index(item))

        with suppress(AttributeError):
            del item.project

    def rename_item(self, index, name):
        ''' Rename an item '''
        self.items[index].name = name

    def save_config(self, fname):
        ''' Save project config file '''
        fstr = StringIO()
        for item in self.items:
            item.save_config(fstr)
        fstr.seek(0)
        try:
            shutil.copyfileobj(fstr, fname)  # fname is file object
        except AttributeError:  # fname is string name of file
            fstr.seek(0)
            with open(fname, 'w') as f:
                shutil.copyfileobj(fstr, f)

    @classmethod
    def from_configfile(cls, fname):
        ''' Load project from config file.

        Parameters
        ----------
        fname: string or file object
            File name or file object to read from

        Returns
        -------
        Project instance
            Project loaded from config. Returns None if file cannot be loaded.
        '''
        try:
            try:
                yml = fname.read()  # fname is file object
            except AttributeError:
                with open(fname, 'r') as fobj:  # fname is string
                    yml = fobj.read()
        except UnicodeDecodeError:
            # file is binary, can't be read as yaml
            return None

        try:
            config = yaml.safe_load(yml)
        except yaml.YAMLError:
            return None  # Can't read YAML

        if not isinstance(config, list):
            config = [config]  # Old (<1.1) style config files are dict at top level, wrap in a list

        newproj = cls()
        for configdict in config:
            if not hasattr(configdict, 'get'): # Something not right with file
                return None

            mode = configdict.get('mode', 'uncertainty')
            if mode == 'uncertainty':
                item = uncertainty.UncertCalc.from_config(configdict)
                if item is None:
                    return   # Could have loaded valid YAML that isn't suncal data
            elif mode == 'sweep':
                item = sweeper.UncertSweep.from_config(configdict)
            elif mode == 'reverse':
                item = reverse.UncertReverse.from_config(configdict)
            elif mode == 'reversesweep':
                item = sweeper.UncertSweepReverse.from_config(configdict)
            elif mode == 'risk':
                item = risk.Risk.from_config(configdict)
            elif mode == 'curvefit':
                item = curvefit.CurveFit.from_config(configdict)
            elif mode == 'data':
                item = dataset.DataSet.from_config(configdict)
            elif mode == 'distributions':
                item = dist_explore.DistExplore.from_config(configdict)
            elif mode == 'intervalbinom':
                item = intervals.attributes.BinomialInterval.from_config(configdict)
            elif mode == 'intervalbinomasset':
                item = intervals.attributes.BinomialIntervalAssets.from_config(configdict)
            elif mode == 'intervaltest':
                item = intervals.attributes.TestInterval.from_config(configdict)
            elif mode == 'intervaltestasset':
                item = intervals.attributes.TestIntervalAssets.from_config(configdict)
            elif mode == 'intervalvariables':
                item = intervals.variables.VariablesInterval.from_config(configdict)
            elif mode == 'intervalvariablesasset':
                item = intervals.variables.VariablesIntervalAssets.from_config(configdict)
            else:
                raise ValueError('Unsupported project mode {}'.format(mode))
            newproj.add_item(item)
        return newproj

    def get_names(self):
        ''' Get names of all project components '''
        names = [item.name for item in self.items]
        return names

    def calculate(self):
        ''' Run calculate() method on all items and append all reports '''
        r = report.Report()
        for item in self.items:
            r.hdr(item.name, level=1)
            r.append(item.calculate().report())
            r.div()
        return r

    def report_all(self):
        ''' Report_all for every project component '''
        r = report.Report()
        for i in range(self.count()):
            r.hdr(self.items[i].name, level=1)
            out = self.items[i].out
            if out is not None:
                r.append(out.report_all())
            else:
                r.txt('Calculation not run\n\n')
            r.div()
        return r

    def report_short(self):
        ''' Report for every project component '''
        r = report.Report()
        for i in range(self.count()):
            r.hdr(self.items[i].name, level=1)
            out = self.items[i].out
            if out is not None:
                r.append(out.report())
            else:
                r.txt('Calculation not run\n\n')
            r.div()
        return r

    def report_summary(self):
        ''' Report_summary() for every project component '''
        r = report.Report()
        for i in range(self.count()):
            r.hdr(self.items[i].name, level=1)
            out = self.items[i].out
            if out is not None:
                r.append(out.report_summary())
                r.txt('\n\n')
            else:
                r.txt('Calculation not run\n\n')
            r.div()
        return r
