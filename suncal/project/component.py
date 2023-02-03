''' Project Components, for managing a single calculation within a project. '''

import numpy as np
import yaml


# pyYaml can't serialize numpy float64 for some reason. Add custom representer.
def np64_representer(dumper, data):
    ''' Represent numpy float64 as yaml '''
    return dumper.represent_float(float(data))  # Just convert to regular float.


def npi64_representer(dumper, data):
    ''' Represent numpy int64 as yaml '''
    return dumper.represent_int(int(data))


yaml.add_representer(np.float64, np64_representer)
yaml.add_representer(np.int64, npi64_representer)


class ProjectComponent:
    ''' Base class for all project components '''
    def get_config(self):
        ''' Get configuration dictionary. Subclass this. '''
        return {}

    def load_config(self, config):
        ''' Load configuration into project component. Subclass this. '''

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
            config = yaml.safe_load(yml)
        except yaml.YAMLError:
            return None  # Can't read YAML

        if isinstance(config, list):
            # To support old (1.0) and new (1.1+) style config files
            config = config[0]

        u = cls.from_config(config)
        return u
