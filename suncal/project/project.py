''' Class for managing a project, a list of calculator objects. '''

from contextlib import suppress
from io import StringIO
import shutil
import yaml

from .proj_uncert import ProjectUncert
from .proj_risk import ProjectRisk
from .proj_dataset import ProjectDataSet
from .proj_explore import ProjectDistExplore
from .proj_reverse import ProjectReverse
from .proj_sweep import ProjectSweep
from .proj_revsweep import ProjectReverseSweep
from .proj_curvefit import ProjectCurveFit
from .proj_wizard import ProjectUncertWizard
from .proj_interval import (ProjectIntervalTest, ProjectIntervalTestAssets, ProjectIntervalBinom,
                            ProjectIntervalBinomAssets, ProjectIntervalVariables, ProjectIntervalVariablesAssets)
from ..common import report


class Project:
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
        if isinstance(item, ProjectReverseSweep):
            mode = 'reversesweep'
        elif isinstance(item, ProjectSweep):
            mode = 'sweep'
        elif isinstance(item, ProjectReverse):
            mode = 'reverse'
        elif isinstance(item, ProjectUncert):
            mode = 'uncertainty'
        elif isinstance(item, ProjectRisk):
            mode = 'risk'
        elif isinstance(item, ProjectUncertWizard):
            mode = 'wizard'
        elif isinstance(item, ProjectCurveFit):
            mode = 'curvefit'
        elif isinstance(item, ProjectDataSet):
            mode = 'data'
        elif isinstance(item, ProjectDistExplore):
            mode = 'distributions'
        elif isinstance(item, ProjectIntervalBinom):
            mode = 'intervalbinom'
        elif isinstance(item, ProjectIntervalBinomAssets):
            mode = 'intervalbinomasset'
        elif isinstance(item, ProjectIntervalTest):
            mode = 'intervaltest'
        elif isinstance(item, ProjectIntervalTestAssets):
            mode = 'intervaltestasset'
        elif isinstance(item, ProjectIntervalVariables):
            mode = 'intervalvariables'
        elif isinstance(item, ProjectIntervalVariablesAssets):
            mode = 'intervalvariablesasset'
        else:
            raise ValueError(f'Unknown item {item}')
        return mode

    def add_item(self, item):
        ''' Add calculator item to project.

            Args:
                item: ProjectComponent item to add.
        '''
        item.project = self  # Add reference to project the item is in (useful for building project tree)
        self.items.append(item)

    def rem_item(self, item):
        ''' Remove item from project

            Args:
                item (int or string): If int, will remove item at index[int]. If string, will remove
                first item with that name.
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
            with open(fname, 'w', encoding='utf-8') as f:
                shutil.copyfileobj(fstr, f)

    @classmethod
    def from_configfile(cls, fname):
        ''' Load project from config file.

        Args:
            fname: File name or file object to read from

        Returns:
            Project instance loaded from config
        '''
        try:
            try:
                yml = fname.read()  # fname is file object
            except AttributeError:
                with open(fname, 'r', encoding='utf-8') as fobj:  # fname is string
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
            if not hasattr(configdict, 'get'):  # Something not right with file
                return None

            mode = configdict.get('mode', 'uncertainty')
            if mode == 'uncertainty':
                item = ProjectUncert.from_config(configdict)
                if item is None:
                    return None  # Could have loaded valid YAML that isn't suncal data
            elif mode == 'wizard':
                item = ProjectUncertWizard.from_config(configdict)
            elif mode == 'sweep':
                item = ProjectSweep.from_config(configdict)
            elif mode == 'reverse':
                item = ProjectReverse.from_config(configdict)
            elif mode == 'reversesweep':
                item = ProjectReverseSweep.from_config(configdict)
            elif mode == 'risk':
                item = ProjectRisk.from_config(configdict)
            elif mode == 'curvefit':
                item = ProjectCurveFit.from_config(configdict)
            elif mode == 'data':
                item = ProjectDataSet.from_config(configdict)
            elif mode == 'distributions':
                item = ProjectDistExplore.from_config(configdict)
            elif mode == 'intervalbinom':
                item = ProjectIntervalBinom.from_config(configdict)
            elif mode == 'intervalbinomasset':
                item = ProjectIntervalBinomAssets.from_config(configdict)
            elif mode == 'intervaltest':
                item = ProjectIntervalTest.from_config(configdict)
            elif mode == 'intervaltestasset':
                item = ProjectIntervalTestAssets.from_config(configdict)
            elif mode == 'intervalvariables':
                item = ProjectIntervalVariables.from_config(configdict)
            elif mode == 'intervalvariablesasset':
                item = ProjectIntervalVariablesAssets.from_config(configdict)
            else:
                raise ValueError(f'Unsupported project mode {mode}')
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
            r.append(item.calculate().report.summary())
            r.div()
        return r

    def report_all(self):
        ''' Report_all for every project component '''
        r = report.Report()
        for i in range(self.count()):
            r.hdr(self.items[i].name, level=1)
            out = self.items[i].result.report
            if out is not None:
                if hasattr(out, 'all'):
                    r.append(out.all())
                else:
                    r.append(out.summary())
            else:
                r.txt('Calculation not run\n\n')
            r.div()
        return r

    def report_short(self):
        ''' Report for every project component '''
        r = report.Report()
        for i in range(self.count()):
            r.hdr(self.items[i].name, level=1)
            out = self.items[i].result.report
            if out is not None:
                r.append(out.summary())
            else:
                r.txt('Calculation not run\n\n')
            r.div()
        return r

    def report_summary(self):
        ''' Report_summary() for every project component '''
        r = report.Report()
        for i in range(self.count()):
            r.hdr(self.items[i].name, level=1)
            out = self.items[i].result.report
            if out is not None:
                r.append(out.summary())
                r.txt('\n\n')
            else:
                r.txt('Calculation not run\n\n')
            r.div()
        return r
