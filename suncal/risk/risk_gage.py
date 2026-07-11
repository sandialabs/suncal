''' Decision risk using attributes (go/nogo) gages '''
from scipy.integrate import quad


def PFA_gage(dist_proc, dist_gage, tolerance, maximum=True):
    ''' Calculate PFA for an attributes gage acceptance measurement.

        Args:
            dist_proc: Distribution of products being gaged
            dist_gage: Distribution of the gage
            tolerance: The upper or lower tolerance of the product
            maximum: Whether the tolerance is a maximum limit or minimum limit
    '''
    # Note - quad doesn't work well with np.inf upper limit. Use something big.
    upperinf = tolerance + 8*dist_gage.std()
    lowerinf = tolerance - 8*dist_gage.std()

    if maximum:
        def integrand(z):
            return (dist_proc.cdf(z) - dist_proc.cdf(tolerance)) * dist_gage.pdf(z)
        pfa = quad(integrand, tolerance, upperinf)[0]
    else:
        def integrand(z):
            return (dist_proc.cdf(tolerance) - dist_proc.cdf(z)) * dist_gage.pdf(z)
        pfa = quad(integrand, lowerinf, tolerance)[0]
    return pfa


def PFR_gage(dist_proc, dist_gage, tolerance, maximum=True):
    ''' Calculate PFR for an attributes gage acceptance measurement.

        Args:
            dist_proc: Distribution of products being gaged
            dist_gage: Distribution of the gage
            tolerance: The upper or lower tolerance of the product
            maximum: Whether the tolerance is a maximum limit or minimum limit
    '''
    # Note - quad doesn't work well with np.inf upper limit. Use something big.
    upperinf = tolerance + 8*dist_gage.std()
    lowerinf = tolerance - 8*dist_gage.std()

    if maximum:
        def integrand(z):
            return (dist_proc.cdf(tolerance) - dist_proc.cdf(z)) * dist_gage.pdf(z)
        pfr = quad(integrand, lowerinf, tolerance)[0]

    else:
        def integrand(z):
            return (dist_proc.cdf(z) - dist_proc.cdf(tolerance)) * dist_gage.pdf(z)
        pfr = quad(integrand, tolerance, upperinf)[0]
    return pfr
