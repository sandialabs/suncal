# Suncal Web Interface

The Suncal-Web interface currently provides these tools as a beta-test:

## Suncal 2.0 Apps

- [Measurement Decision Risk Calculator](risk/index.html): Calculate global and specific risk, probability of conformance, and generate risk curves.
- [End-to-end Measurement Quality Assurance](mqa/index.html): 

## Suncal 1.x Apps

- [Uncertainty Propagation](uncertainty.html): GUM and Monte Carlo Risk Propagation
- [Distribution Explorer](distributions.html): Sample from probability distributions and combine using Monte Carlo (typically for training purposes).
- [Student T Calculator](student.html): Calculate coverage factor, level of confidence, and degrees of freedom
- [Units Converter](units.html): Convert between measurement units
- [Decision Risk Calculator (simple)](risktur_py.html): Calculate probability of false accept and reject based on Test Uncertainty Ratios
- [Decision Risk Calculator](risk_py.html): Calculate probability of false accept and reject using arbitrary probability distributions
- [Decision Risk Curves](riskcurves_py.html): Generate plots of decision risk versus guardbanding, in-tolerance probability, or TURs
- [Decision Risk Probability of Conformance](risk_conform_py.html): Plot the probability of conformance versus measured value


Additional web implementations of features from the full Suncal package are in development.

---

## About

The Suncal-Web 1.x interface runs the same back-end code as the full [desktop](../index.html) version, but using
a [pyscript](https://pyscript.net)-based user interface. All data and calculations stay in your browser.
Source code for this website and the web interface can be found on [github](https://github.com/sandialabs/suncal/tree/gh-pages).
The Suncal 2.0 apps have been re-built in Rust for performance and deployability. Source in [github](https://github.com/sandialabs/suncal/tree/webr).

Suncal-Web is still under active development and beta testing.


### License and Copyright

Copyright 2021-2025 National Technology & Engineering Solutions of Sandia, LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this software.

This software is distributed under the GNU General Public License.
