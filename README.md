# Uncertainty Calculator

Sandia UNcertainty CALculator (SUNCAL)

Copyright 2019-2022 National Technology & Engineering Solutions of Sandia, LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this software.
This software is distributed under the GNU General Public License.

---

This tool was developed by the Primary Standards Lab at Sandia National Laboratories to calculate the combined uncertainty of a
multi-variable system. Contact uncertainty@sandia.gov.


## Installation

Installation of the Python package and command line interface requires Python 3.7+ with the following packages:

- numpy
- scipy
- sympy
- matplotlib
- pyqt5
- pyyaml
- markdown
- pint

To install (on Windows, Mac, or Linux), from a command prompt, run:

```
pip install suncal
```

## Example Usage

From a python terminal, script, or notebook:

```
import suncal
u = suncal.UncertaintyCalc('A*B')
u.set_input('A', nom=100, unc=0.1)
u.set_input('B', nom=2, unc=0.01)
u.calculate()
```

See the PDF user's manual and the example notebook files in the docs folder for a complete reference guide.


## Command-line script

A script named suncal will be installed to your system path. From a command line, run:

`suncal file`

where file is the filename of a setup file. See doc/examples folder for
example setup files. Refer to the PDF user's manual for other commands.


## User interface
A graphical user interface is installed with the Python package. Pre-built executables are available from https://sandiapsl.github.io.

To launch the user interface from a command line, run:

`suncalui`

A shortcut to this can be put on your desktop, etc.
