# Suncal Web Interface

The Suncal-Web interface currently provides these tools as a beta-test:

- [Uncertainty Propagation](uncertainty.html): GUM and Monte Carlo Risk Propagation
- [Distribution Explorer](distributions.html): Sample from probability distributions and combine using Monte Carlo (typically for training purposes).
- [Student T Calculator](student.html): Calculate coverage factor, level of confidence, and degrees of freedom
- [Units Converter](units.html): Convert between measurement units

Additional web implementations of features from the full Suncal package are in development.

---

## About

The Suncal-Web interface runs the same back-end code as the full [desktop](../index.html) version, but using
a [pyscript](https://pyscript.net)-based user interface. All data and calculations stay in your browser.
Source code for this website and the web interface can be found on [github](https://github.com/sandialabs/suncal/tree/gh-pages).

Suncal-Web is still under active development and considered in beta testing.


### Limitations

Some features are only available in the desktop version at this time, including:

- Loading and saving calculation setup
- Reverse and swept uncertainty propagation
- Analysis of Variance calculations
- Calibration interval analysis
- Curve fitting uncertainty analysis
- Combining the results of different calculations


### License and Copyright

Copyright 2021-2023 National Technology & Engineering Solutions of Sandia, LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this software.

This software is distributed under the GNU General Public License.
