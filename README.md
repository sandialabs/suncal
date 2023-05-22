# Suncal
## Primary Standards Lab, Sandia National Laboratories

The Sandia Uncertainty Calculator (Suncal) is developed by the Primary Standards Lab at Sandia National Laboratories to calculate the combined uncertainty of a system of multiple input parameters.

In general, a measurement value is calculated from a function of several input measurements: Y = f(x1, x2, x3...).
Each input X value has a measured value and associated uncertainty.
This calculator will determine the total uncertainty in the output quantity Y.
Two approaches are used, the Kline-McClintock uncertainty approximation as described in the Guide to Expression of Uncertainty in Measurements (GUM), and a Monte-Carlo method.

Other features include risk analysis (probability of false accept and reject), analysis of variance, and finding uncertainty in curve fitting.


### Desktop version

Download binaries for Windows and Mac OSX, along with a user manual and some usage examples, from the [Latest Release](https://github.com/sandialabs/suncal/releases/latest) page.

### Online version

Some features of Suncal may be run through the [Web Interface to Suncal](suncal\index.html), an online interface without any download or installation required.


### Screenshots

![Uncertainty Output](/img/output.png)

![Joint PDF output](/img/jointpdf.png)

![Curve Fitting](/img/curvefit.png)

![Risk Analysis](/img/risk.png)


### License and Copyright

Copyright 2019-2023 National Technology & Engineering Solutions of Sandia, LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this software.

This software is distributed under the GNU General Public License.
