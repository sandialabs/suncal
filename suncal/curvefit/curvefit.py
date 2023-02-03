''' Functions for fitting curves with uncertainties in parameters and prediction '''

from collections import namedtuple
import numpy as np
import scipy.odr
import scipy.optimize


FitCoeff = namedtuple('FitCoeff', ['coeff', 'covariance'])


def fit(func, x, y, ux, uy, p0=None, bounds=(-np.inf, np.inf), odr=None, absolute_sigma=True):
    ''' Generic curve fit. Selects scipy.optimize.curve_fit if ux==0 or scipy.odr otherwise.

        Args:
            func (callable): The function to fit, must take x as first parameter, followed by
                fit coefficients.
            x, y (arrays): X and Y data to fit
            ux, uy (arrays): Standard uncertainty in x and y
            p0 (array-like): Initial guess parameters
            absolute_sigma (boolean): Treat uncertainties in an absolute sense. If false, only
                relative magnitudes matter.

        Returns:
            pcoeff (array): Coefficients of best fit curve
            pcov (array): Covariance of coefficients. Standard error of coefficients is
                np.sqrt(np.diag(pcov)).
    '''
    if odr or not (ux is None or all(ux == 0)):
        return odrfit(func, x, y, ux, uy, p0=p0, absolute_sigma=absolute_sigma)

    if uy is None or all(uy == 0):
        return FitCoeff(*scipy.optimize.curve_fit(func, x, y, p0=p0, bounds=bounds))

    return FitCoeff(*scipy.optimize.curve_fit(
               func, x, y, sigma=uy, absolute_sigma=absolute_sigma, p0=p0, bounds=bounds))


def linefit(x, y, ux, uy, absolute_sigma=True):
    ''' Generic straight line fit. Uses linefit_lsq() if ux==0 or linefitYork() otherwise.

        Args:
            func (callable): The function to fit, must take x as first parameter, followed by
                fit coefficients.
            x, y (arrays): X and Y data to fit
            ux, uy (arrays): Standard uncertainty in x and y
            absolute_sigma (boolean): Treat uncertainties in an absolute sense. If false, only
                relative magnitudes matter.

        Returns:
            pcoeff (array): Coefficients of best fit curve
            pcov (array): Covariance of coefficients. Standard error of coefficients is
                np.sqrt(np.diag(pcov)).
    '''
    if ux is None or all(ux == 0):
        return linefit_lsq(x, y, sig=uy, absolute_sigma=absolute_sigma)
    return linefitYork(x, y, sigx=ux, sigy=uy, absolute_sigma=absolute_sigma)


def linefit_lsq(x, y, sig, absolute_sigma=True):
    ''' Fit a line with uncertainty in y (but not x)

        Args:
            x (array): X values of fit
            y (array): Y values of fit
            sig (array): uncertainty in y values
            absolute_sigma (boolean): Treat uncertainties in an absolute sense. If false, only
                relative magnitudes matter.

        Returns:
            coeff (array): Coefficients of line fit [slope, intercept].
            cov (array): 2x2 Covariance matrix of fit parameters. Standard error is
                np.sqrt(np.diag(cov)).

        Note:
            Returning coeffs and covariance so the return value matches scipy.optimize.curve_fit.
            With sig=0, this algorithm estimates a sigma using the residuals.

        References:
            [1] Numerical Recipes in C, The Art of Scientific Computing. Second Edition.
                W. H. Press, S. A. Teukolsky, W. T. Vetterling, B. P. Flannery.
                Cambridge University Press. 2002.
    '''
    sig = np.atleast_1d(sig)
    if len(sig) == 1:
        sig = np.full(len(x), sig[0])
    if all(sig) > 0:
        wt = 1./sig**2
        ss = sum(wt)
        sx = sum(x*wt)
        sy = sum(y*wt)
        sxoss = sx/ss
        t = (x-sxoss)/sig
        st2 = sum(t*t)
        b = sum(t*y/sig)/st2
    else:
        sx = sum(x)
        sy = sum(y)
        ss = len(x)
        sxoss = sx/ss
        t = (x-sxoss)
        st2 = sum(t*t)
        b = sum(t*y)/st2
    a = (sy-sx*b)/ss
    siga = np.sqrt((1+sx*sx/(ss*st2))/ss)
    sigb = np.sqrt(1/st2)

    resid = sum((y-a-b*x)**2)
    syx = np.sqrt(resid/(len(x)-2))
    cov = -sxoss * sigb**2
    if not all(sig) > 0:
        siga = siga * syx
        sigb = sigb * syx
        cov = cov * syx*syx
    elif not absolute_sigma:
        # See note in scipy.optimize.curve_fit for absolute_sigma parameter.
        chi2 = sum(((y-a-b*x)/sig)**2)/(len(x)-2)
        siga, sigb, cov = np.sqrt(siga**2*chi2), np.sqrt(sigb**2*chi2), cov*chi2
    # rab = -sxoss * sigb / siga  # Correlation can be computed this way
    return FitCoeff(np.array([b, a]), np.array([[sigb**2, cov], [cov, siga**2]]))


def linefitYork(x, y, sigx=None, sigy=None, rxy=None, absolute_sigma=True):
    ''' Find a best-fit line through the x, y points having
        uncertainties in both x and y. Also accounts for
        correlation between the uncertainties. Uses York's algorithm.

        Args:
            x (array): X values to fit
            y (array): Y values to fit
            sigx (array or float): Uncertainty in x values
            sigy (array or float): Uncertainty in y values
            rxy (array or float): Correlation coefficient between sigx and sigy
            absolute_sigma (boolean): Treat uncertainties in an absolute sense. If false, only
                relative magnitudes matter.

        Returns:
            coeff (array): Coefficients of line fit [slope, intercept].
            cov (array): 2x2 Covariance matrix of fit parameters. Standard error is
                np.sqrt(np.diag(cov)).

        Note:
            Returning coeffs and covariance so the return value matches scipy.optimize.curve_fit.
            Implemented based on algorithm in [1] and pseudocode in [2].

        References:
            [1] York, Evensen. Unified equations for the slope, intercept, and standard
                errors of the best straight line. American Journal of Physics. 72, 367 (2004)
            [2] Wehr, Saleska. The long-solved problem of the best-fit straight line:
                application to isotopic mixing lines. Biogeosciences. 14, 17-29 (2017)
    '''
    # Condition inputs so they're all float64 arrays
    x = x.astype(float)
    y = y.astype(float)
    if sigx is None or len(np.nonzero(sigx)[0]) == 0:
        sigx = np.full(len(x), 1E-99, dtype=float)   # Don't use 0, but a really small number
    elif np.isscalar(sigx):
        sigx = np.full_like(x, sigx)

    if sigy is None or len(np.nonzero(sigy)[0]) == 0:
        sigy = np.full(len(y), 1E-99, dtype=float)
    elif np.isscalar(sigy):
        sigy = np.full_like(x, sigy)

    sigy = np.maximum(sigy, 1E-99)
    sigx = np.maximum(sigx, 1E-99)

    if rxy is None:
        rxy = np.zeros_like(y)
    elif np.isscalar(rxy):
        rxy = np.full_like(x, rxy)

    _, b0 = np.polyfit(x, y, deg=1)  # Get initial estimate for slope
    T = 1E-15
    b = b0
    bdiff = np.inf

    wx = 1./sigx**2
    wy = 1./sigy**2
    alpha = np.sqrt(wx*wy)
    while bdiff > T:
        bold = b
        w = alpha**2/(b**2 * wy + wx - 2*b*rxy*alpha)
        sumw = sum(w)
        X = sum(w*x)/sumw
        Y = sum(w*y)/sumw
        U = x - X
        V = y - Y
        beta = w * (U/wy + b*V/wx - (b*U + V)*rxy/alpha)
        Q1 = sum(w*beta*V)
        Q2 = sum(w*beta*U)
        b = Q1/Q2
        bdiff = abs((b-bold)/bold)
    a = Y - b*X

    # Uncertainties
    xi = X + beta
    xbar = sum(w*xi) / sumw
    sigb = np.sqrt(1./sum(w * (xi - xbar)**2))
    siga = np.sqrt(xbar**2 * sigb**2 + 1/sumw)
    # resid = sum((y-b*x-a)**2)

    # Correlation bw a, b
    # rab = -xbar * sigb / siga
    cov = -xbar * sigb**2

    if not absolute_sigma:
        # See note in scipy.optimize.curve_fit for absolute_sigma parameter.
        chi2 = sum(((y-a-b*x)*np.sqrt(w))**2)/(len(x)-2)
        siga, sigb, cov = np.sqrt(siga**2*chi2), np.sqrt(sigb**2*chi2), cov*chi2
    return FitCoeff(np.array([b, a]), np.array([[sigb**2, cov], [cov, siga**2]]))


def odrfit(func, x, y, ux, uy, p0=None, absolute_sigma=True):
    ''' Fit the curve using scipy's orthogonal distance regression (ODR)

        Args:
            func (callable): The function to fit. Must take x as first argument, and other
                parameters as remaining arguments.
            x, y (arrays): X and Y data to fit
            ux, uy (arrays): Standard uncertainty in x and y
            p0 (array): Initial guess of parameters.
            absolute_sigma (boolean): Treat uncertainties in an absolute sense. If false, only
                relative magnitudes matter.

        Returns:
            pcoeff (array): Coefficients of best fit curve
            pcov (array): Covariance of coefficients. Standard error of coefficients is
                np.sqrt(np.diag(pcov)).
    '''
    # Wrap the function because ODR puts params first, x last
    def odrfunc(B, x):
        return func(x, *B)

    if ux is not None and all(ux == 0):
        ux = None
    if uy is not None and all(uy == 0):
        uy = None

    model = scipy.odr.Model(odrfunc)
    mdata = scipy.odr.RealData(x, y, sx=ux, sy=uy)
    modr = scipy.odr.ODR(mdata, model, beta0=p0)
    mout = modr.run()
    if mout.info != 1:
        print('Warning - ODR failed to converge')

    if absolute_sigma:
        # SEE: https://github.com/scipy/scipy/issues/6842.
        # If this issue is fixed, these options may be swapped!
        cov = mout.cov_beta
    else:
        cov = mout.cov_beta*mout.res_var
    ODR = namedtuple('ODR', ['coeff', 'covariance'])
    return ODR(mout.beta, cov)
