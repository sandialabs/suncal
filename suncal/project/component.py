''' Project Components, for managing a single calculation within a project. '''

import numpy as np
import yaml

from ..common import unitmgr


# pyYaml can't serialize numpy float64 for some reason. Add custom representer.
def np64_representer(dumper: yaml.Dumper, data: np.float64):
    ''' Represent numpy float64 as yaml '''
    return dumper.represent_float(float(data))  # Just convert to regular float.


def npi64_representer(dumper: yaml.Dumper, data: np.int64):
    ''' Represent numpy int64 as yaml '''
    return dumper.represent_int(int(data))


def ndarray_representer(dumper: yaml.Dumper, array: np.ndarray) -> yaml.Node:
    ''' Represent numpy ndarray as list in yaml '''
    return dumper.represent_list(array.tolist())


def unit_representer(dumper: yaml.Dumper, value):
    return dumper.represent_scalar('!quantity', format(value))


def unit_constructor(loader, value):
    qty = unitmgr.parse_expression(value.value)
    return qty


yaml.add_representer(np.float64, np64_representer)
yaml.add_representer(np.int64, npi64_representer)
yaml.add_representer(np.ndarray, ndarray_representer)
yaml.add_representer(unitmgr.Quantity, unit_representer)
yaml.add_constructor(u'!quantity', unit_constructor, Loader=yaml.Loader)


class ProjectComponent:
    ''' Base class for all project components '''
    def __init__(self, name=None):
        self.name = name
        self.description = ''
        self.project = None  # Project this belongs to
        self._result = None

    @property
    def result(self):
        ''' DataSet calculation result '''
        if self._result is None:
            self.calculate()
        return self._result

    def calculate(self):
        ''' Calculate the result '''
        # Subclass this

    def get_config(self):
        ''' Get configuration dictionary. Subclass this. '''
        return {'name': self.name,
                'description': self.description}

    def load_config(self, config):
        ''' Load configuration into project component. Subclass this. '''

    @property
    def component_type(self) -> str:
        ''' Type name of the project component '''
        return self.__class__.__name__[7:]  # remove the "Project" part

    @classmethod
    def from_config(cls, config):
        ''' Create new project component from the config dictionary '''
        proj = cls()
        proj.load_config(config)
        return proj

    def save_config(self, fname):
        ''' Save configuration to file.

            Args:
                fname: File name or open file object to write configuration to
        '''
        d = self.get_config()
        d = [d]  # Must go in list to support multi-calculation project structure
        out = yaml.dump(d, default_flow_style=False)

        try:
            fname.write(out)
        except AttributeError:
            with open(fname, 'w', encoding='utf-8') as f:
                f.write(out)

    @classmethod
    def from_configfile(cls, fname):
        ''' Read and parse the configuration file. Returns a new UncertCalc
            instance. See Examples folder for sample config file.

            Args:
                fname: File name or open file object to write configuration to

            Returns:
                New ProjectComponent instance
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
            config = yaml.load(yml, Loader=yaml.Loader)
        except yaml.YAMLError:
            return None  # Can't read YAML

        if isinstance(config, list):
            # To support old (1.0) and new (1.1+) style config files
            config = config[0]

        u = cls.from_config(config)
        return u
