
from collections import namedtuple
import numpy as np


def fitpoly(x, y, m=1):
    ''' Fit polynomial, order m, with zero intercept

        Args:
            x: X-values
            y: Y-values
            m: Polynomial order

        Returns:
            b (array): Polynomial coefficients where
              y = b[0] * x + b[1]*x**2 ... + b[i]*x**(i+1)
            cov (array): Covariance matrix of b parameters
    '''
    # Redefine the curve fit here rather than using curvefit.py for speed/efficiency
    x = np.atleast_1d(x)
    y = np.atleast_1d(y)

    T = np.vstack([x**n for n in range(1, m+1)]).T  # Castrup (7)
    TTinv = np.linalg.inv(T.T @ T)
    b = TTinv @ T.T @ y                             # Castrup (6)

    rss = sum((y - y_pred(x, b))**2)                # Castrup (5)
    s2 = rss / (len(x)-m)                           # Castrup (10)
    S = s2 * np.eye(m)                              # Castrup (9)
    cov = TTinv @ S                                 # Castrup (8)
    return b, cov, np.sqrt(s2)


def y_pred(x, b, y0=0):
    ''' Predict y at the x value given b polynomial coefficients from fitpoly() '''
    scalar = not np.asarray(x).shape
    x = np.atleast_1d(x)
    y = np.zeros(len(x))
    m = len(b)
    for i, xval in enumerate(x):
        tprime = np.array([xval**n for n in range(1, m+1)])
        y[i] = tprime @ b                           # Castrup (12)
    if scalar:
        return y0 + y[0]
    return y0 + y


def u_pred(x, b, cov, syx):
    ''' Prediction band at x (based on residual scatter) '''
    scalar = not np.asarray(x).shape
    x = np.atleast_1d(x)
    upred = np.zeros(len(x))
    m = len(b)
    for i, xval in enumerate(x):
        tprime = np.array([xval**n for n in range(1, m+1)])
        # upred[i] = syx * np.sqrt(1 + tprime.T @ cov @ tprime)  # Castrup (19) appears to be wrong???
        upred[i] = np.sqrt(syx**2 + (tprime.T @ cov @ tprime))
    if scalar:
        return upred[0]
    return upred


def u_conf(x, b, cov):
    ''' Confidence band at x (based on residual scatter) '''
    scalar = not np.asarray(x).shape
    x = np.atleast_1d(x)
    uconf = np.zeros(len(x))
    m = len(b)
    for i, xval in enumerate(x):
        tprime = np.array([xval**n for n in range(1, m+1)])
        uconf[i] = np.sqrt(tprime.T @ cov @ tprime)
    if scalar:
        return uconf[0]
    return uconf
