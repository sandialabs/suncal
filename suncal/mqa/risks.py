''' PFA/PFR risk calculations using numeric integration '''
import numpy as np

from .pdf import Pdf
from ..common.limit import Limit


def pfa(uncert_pdf: Pdf,
        eopr_pdf: Pdf,
        tolerance: Limit,
        accept_limit: Limit = None) -> float:
    ''' Unconditional global probability of false accept '''
    # test_x/test_pdf needs to be centered on 0 (to be shited to measured value)
    N = 500  # Resolution of integration grid

    if accept_limit is None:
        accept_limit = tolerance

    bot, top = eopr_pdf.domain

    accept_lo = accept_limit.flow if np.isfinite(accept_limit.flow) else bot
    accept_hi = accept_limit.fhigh if np.isfinite(accept_limit.fhigh) else top
    test_region = np.linspace(accept_lo, accept_hi, N)
    dy = test_region[1]-test_region[0]

    pfa_upper = pfa_lower = 0

    # Above
    if np.isfinite(tolerance.fhigh):
        prod_region = np.linspace(tolerance.fhigh, top, N)
        pdf_prod = eopr_pdf.pdf(prod_region)
        dx = prod_region[1]-prod_region[0]

        test2d = np.zeros((N, N))
        for i in range(N):
            test2d[i, :] = np.interp(test_region, uncert_pdf._x+prod_region[i], uncert_pdf._y)

        prod2d = np.tile(pdf_prod, (N, 1)).T
        pfa_upper = np.trapz(np.trapz(test2d*prod2d, dx=dx, axis=0), dx=dy)

    # Below
    if np.isfinite(tolerance.flow):
        prod_region = np.linspace(bot, tolerance.flow, N)
        pdf_prod = eopr_pdf.pdf(prod_region)
        dx = prod_region[1]-prod_region[0]

        test2d = np.zeros((N, N))
        for i in range(N):
            test2d[i, :] = np.interp(test_region, uncert_pdf._x+prod_region[i], uncert_pdf._y)

        prod2d = np.tile(pdf_prod, (N, 1)).T
        pfa_lower = np.trapz(np.trapz(test2d*prod2d, dx=dx, axis=0), dx=dy)

    return pfa_upper + pfa_lower


def pfr(uncert_pdf: Pdf,
        eopr_pdf: Pdf,
        tolerance: Limit,
        accept_limit: Limit = None) -> float:
    ''' Unconditional global probability of false reject '''
    N = 500  # Resolution of integration grid
    if accept_limit is None:
        accept_limit = tolerance

    bot, top = eopr_pdf.domain
    tol_lo = tolerance.flow if np.isfinite(tolerance.flow) else bot
    tol_hi = tolerance.fhigh if np.isfinite(tolerance.fhigh) else top
    prod_region = np.linspace(tol_lo, tol_hi, N)
    pdf_prod = eopr_pdf.pdf(prod_region)
    dx = prod_region[1]-prod_region[0]
    pfr_upper = pfr_lower = 0

    # Above
    if np.isfinite(tolerance.fhigh):
        test_region = np.linspace(accept_limit.fhigh, top, N)
        dy = test_region[1]-test_region[0]
        test2d = np.zeros((N, N))
        for i in range(N):
            test2d[i, :] = np.interp(test_region, uncert_pdf._x+prod_region[i], uncert_pdf._y)

        prod2d = np.tile(pdf_prod, (N, 1)).T
        pfr_upper = np.trapz(np.trapz(test2d*prod2d, dx=dx, axis=0), dx=dy)

    # Below
    if np.isfinite(tolerance.flow):
        test_region = np.linspace(bot, accept_limit.flow, N)
        dy = test_region[1]-test_region[0]
        test2d = np.zeros((N, N))
        for i in range(N):
            test2d[i, :] = np.interp(test_region, uncert_pdf._x+prod_region[i], uncert_pdf._y, left=0, right=0)

        prod2d = np.tile(pdf_prod, (N, 1)).T
        pfr_lower = np.trapz(np.trapz(test2d*prod2d, dx=dx, axis=0), dx=dy)

    return pfr_upper + pfr_lower


def cpfa(uncert_pdf: Pdf,
         eopr_pdf: Pdf,
         tolerance: Limit,
         accept_limit: Limit = None) -> float:
    ''' "Traditional" Conditional global probability of false accept '''
    N = 500
    if accept_limit is None:
        accept_limit = tolerance

    bot, top = eopr_pdf.domain
    tol_lo = tolerance.flow if np.isfinite(tolerance.flow) else bot
    tol_hi = tolerance.fhigh if np.isfinite(tolerance.fhigh) else top
    accept_lo = accept_limit.flow if np.isfinite(accept_limit.flow) else bot
    accept_hi = accept_limit.fhigh if np.isfinite(accept_limit.fhigh) else top

    # P(In tolerance and Accepted)
    prod_region = np.linspace(tol_lo, tol_hi, N)
    test_region = np.linspace(accept_lo, accept_hi, N)
    dx = prod_region[1]-prod_region[0]
    dy = test_region[1]-test_region[0]
    pdf_prod = eopr_pdf.pdf(prod_region)

    test2d = np.zeros((N, N))
    for i in range(N):
        test2d[i, :] = np.interp(test_region, uncert_pdf._x+prod_region[i], uncert_pdf._y)

    prod2d = np.tile(pdf_prod, (N, 1)).T
    intol_and_accepted = np.trapz(np.trapz(test2d*prod2d, dx=dx, axis=0), dx=dy)

    # P(Accepted)
    prod_region = np.linspace(bot, top, N)
    dx = prod_region[1]-prod_region[0]
    dy = test_region[1]-test_region[0]
    pdf_prod = eopr_pdf.pdf(prod_region)

    test2d = np.zeros((N, N))
    for i in range(N):
        test2d[i, :] = np.interp(test_region, uncert_pdf._x+prod_region[i], uncert_pdf._y)

    prod2d = np.tile(pdf_prod, (N, 1)).T
    accepted = np.trapz(np.trapz(test2d*prod2d, dx=dx, axis=0), dx=dy)

    return 1 - intol_and_accepted / accepted


def bracket(func, x1: float, x2: float, tolerance: float = .0001):
    ''' Find roots of the function `func` between x1 and x2 '''
    NMAX = 40
    f = func(x1)
    fmid = func(x2)
    if f * fmid >= 0:
        raise ValueError('Root not contained in range')
    dx = x2-x1 if f < 0 else x1-x2
    rtb = x1 if f < 0 else x2

    for _ in range(NMAX):
        dx /= 2
        xmid = rtb + dx
        fmid = func(xmid)
        if fmid <= 0:
            rtb = xmid
        if abs(dx) < tolerance or fmid == 0:
            return rtb
    raise ValueError('Bisection limit exceeded')


def target_pfa(
        uncert_pdf: Pdf,
        eopr_pdf: Pdf,
        tolerance: Limit,
        target: float) -> Limit:
    ''' Calculate guardband limit to achieve desired PFA '''
    # scipy fsolve must be vectorizable, and pfa isn't
    center = float(tolerance.center)
    plusminus = bracket(
        lambda x: pfa(uncert_pdf, eopr_pdf, tolerance, Limit.from_plusminus(center, x))-target,
        float(tolerance.plusminus)/1000,
        float(tolerance.plusminus) * 2,
        tolerance=float(tolerance.plusminus) / 1000
    )
    return Limit.from_plusminus(center, plusminus)


def target_cpfa(
        uncert_pdf: Pdf,
        eopr_pdf: Pdf,
        tolerance: Limit,
        target: float) -> Limit:
    ''' Calculate guardband limit to achieve desired CPFA '''
    # scipy fsolve must be vectorizable, and pfa isn't
    center = float(tolerance.center)
    plusminus = bracket(
        lambda x: cpfa(uncert_pdf, eopr_pdf, tolerance, Limit.from_plusminus(center, x))-target,
        float(tolerance.plusminus)/1000,
        float(tolerance.plusminus) * 2,
        tolerance=float(tolerance.plusminus) / 1000
    )
    return Limit.from_plusminus(center, plusminus)
