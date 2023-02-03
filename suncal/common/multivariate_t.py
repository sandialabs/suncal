''' Multivariate Student-T Distribution

    Scipy's multivariate_t does not account for covariance/correlation.
'''
import numpy as np


def multivariate_t_rvs(mean, corr, df=np.inf, size=1):
    ''' Generate random variables from multivariate Student t distribution.
        Not implemented in Scipy. Code taken from scikits package.

        Args:
            mean (array): Mean of random variables, shape (M,)
            corr (array): Correlation matrix, shape (M,M)
            df (float): degrees of freedom
            size (int): Number of samples for output array

        Returns:
            rvs (array): Correlated random variables, shape (size, M)
    '''
    mean = np.asarray(mean)
    d = len(mean)
    if df == np.inf:
        x = 1.
    else:
        x = np.random.chisquare(df, size)/df
    z = np.random.multivariate_normal(np.zeros(d), corr, (size,))
    return mean + z / np.sqrt(x)[:, None]
