# Change History


## Version 1.5.1

- Select between "mean" and "median" in for distribution entry
- Changed gamma distribution to input alpha and beta parameters, consistent with JCGM.
- Added "observed itp" parameter to PFA_norm and PFR_norm functions
- Fixed scrolling issue on Windows
- Fixed issue when a measurement function is a constant
- Updates for Python 3.8+ and dependency compatibility

## Version 1.5

- Improved uncertainty computation and derivation report in multi-output measurement models
- Improved handling of callable functions with multiple outputs
- Added colorbar to contour plots
- Allow UncertComplex to compute uncertainty on callable functions
- Suppress duplicate plot outputs in Jupyter
- Fixed order of polynomial curve fit coefficients in report
- Fixed date issue introduced by Matplotlib 3.3
- Fixed font scaling issue in Windows with display scaling enabled



## Version 1.4

- Added calibration interval calculator, using NCSLI RP-1 Method A3 and Method S2
- Implemented "Symbolic GUM Only" mode to show GUM equation results without entering values
- Added risk sweeps mode for plotting PFA(R) vs. itp at various TUR, or other combinations
- Report the correlation matrix between multiple uncertainty model equations
- Save user-defined units with project files
- Added distribution fit selection to Data Sets page
- Added random seed option to Distribution Explorer
- Sort the measured variables list alphabetically
- Fixed issue entering t-distribution parameters before degrees of freedom is defined
- Fixed display of degree symbol in output reports
- Handle callable measurement models that raise exception with Unit Quantity inputs
- Minor fixes for Python 3.8 compatibility

## Version 1.3.6

- Fixed potential units conversion error in Welch-Satterthwaite degrees of freedom calculation
- Repeatability and reproducibility, as standard as standard deviation of the mean, can be imported from datasets
- Added guardband sweep, probability of conformance plots to risk calculator


## Version 1.3.5

- Finished flattened uncertainty entry user interface
- Fixed error in computing Poisson distribution mean for Monte Carlo calculation
- Fix a possible unit conversion error


## Version 1.3.4

- Improved output reporting system. Adds right-click option to output reports for changing significant figures and number formats.
- Flattened uncertainty entry user interface and improved keyboard navigation
- GUI enhancements and fixes


## Version 1.3.3

- GUI enhancements and fixes
- Added autocorrelation calculation to Data Sets and ANOVA
- Improved user interface for data sharing between calculation types
- Show confidence band on normal probability plots
- Fix exception due to units incompatibility when calculating uncertainty of k**x
- Fix units conversion issue with Monte Carlo of non-vectorizable Python functions


## Version 1.3.2
- Minor bug fixes


## Version 1.3

- Added processing and conversion of measurement units
- Added Monte-Carlo option for computing risk probabilities
- Other minor user interface enhancements


## Version 1.2

Initial Public Release
