''' Student-T table calculator '''
from pyscript import document
import math


def update(event):
    try:
        degf = float(document.querySelector('#degreesfreedom').value)
        conf = float(document.querySelector('#confidence').value)
        k = float(document.querySelector('#coverage').value)
    except ValueError:
        return

    if document.querySelector('#solvefor-cov').checked:
        k = t_inv2t(conf/100, degf)
        document.querySelector('#coverage').value = round(k, 3)
    elif document.querySelector('#solvefor-conf').checked:
        conf = confidence(k, degf)
        document.querySelector('#confidence').value = round(conf*100, 3)
    else:
        degf = degrees_freedom(k, conf/100)
        document.querySelector('#degreesfreedom').value = round(degf, 3)


def norminv(p: float) -> float:
    ''' Inverse CDF of normal distribution

        Args:
            p: Probability

        Implements Acklam's Chebyshev Approximation
        as described in https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=4630740
    '''
    if p < 0 or p > 1:
        raise ValueError('p must be 0 < p < 1.')

    def R1(z):
        r = ((-7.784894002430293E-3 * z**5
              - 3.223964580411365E-1 * z**4
              - 2.400758277161838 * z**3
              - 2.549732539343734 * z**2
              + 4.374664141464968 * z
              + 2.938163982698783) /
             (7.784695709041462E-3 * z**4
              + 3.224671290700398E-1 * z**3
              + 2.445134137142996 * z**2
              + 3.754408661907416 * z**1
              + 1.000000000000000))
        return r

    def R2(z):
        r = ((-3.969683028665376E1 * z**5
              + 2.209460984245205E2 * z**4
              - 2.759285104469687E2 * z**3
              + 1.383577518672690E2 * z**2
              - 3.066479806614716E1 * z
              + 2.506628277459239) /
             (-5.447609879822406E1 * z**5
              + 1.615858368580409E2 * z**4
              - 1.556989798598866E2 * z**3
              + 6.680131188771972E1 * z**2
              - 1.328068155288572E1 * z
              + 1.000000000000000))
        return r

    if p < 0.02425:
        z = R1(math.sqrt(-2*math.log(p)))
    elif p <= 0.97575:
        z = (p-0.5)*R2((p-0.5)**2)
    else:
        z = -R1(math.sqrt(-2*math.log(1-p)))
    return z


def t_inv2t(conf: float, n: float) -> float:
    ''' Inverse 2-tailed student T distribution

        Args:
            conf: Level of Confidence (0-1)
            n: Degrees of Freedom

        Reference:
            Implements "Algorithm 396 Student's T-Quantiles" by G.W. Hill,
            Communications of the ACM, Volume 13, Number 10, October 1970
            (with "Remark on Algorithm 396" July 1979).
            https://dl.acm.org/doi/pdf/10.1145/355598.355600
            https://dl.acm.org/doi/pdf/10.1145/355945.355956
    '''
    if conf < 0 or conf > 1:
        raise ValueError('conf must be 0 < p < 1.')
    if n < 1:
        raise ValueError('n must be >= 1')

    p = 1-conf

    HALFPI = math.pi/2
    if n == 2:
        t = math.sqrt(2/(p*(2-p))-2)
    elif n == 1:
        t = math.cos(p*HALFPI)/math.sin(p*HALFPI)
    else:
        a = 1/(n-.5)
        b = 48/a**2
        c = ((20700*a/b-98)*a-16)*a + 96.36
        d = ((94.5/(b+c)-3)/b+1) * math.sqrt(a*HALFPI) * n
        x = d*p
        y = x**(2/n)
        if y > .05+a:
            x = norminv(p*0.5)
            y = x**2
            if n < 5:
                c = c+0.3*(n-4.5)*(x+0.6)
            c = (((0.05*d*x-5)*x-7)*x-2)*x+b+c
            y = (((((0.4*y+6.3)*y+36)*y+94.5)/c-y-3)/b+1)*x
            y = a*y**2
            if y > 0.1:
                y = math.exp(y) - 1
            else:
                y = ((y+4)*y+12)*y*y/24 + y

        else:
            y = ((1./(((n+6)/(n*y) - 0.089*d - 0.822) * (n+2)*3)+0.5/(n+4))*y-1) * (n+1)/(n+2) + 1/y
        t = math.sqrt(n*y)
    return t


def bisect(f, a: float, b: float) -> float:
    ''' Find roots of function by bisection

        Args:
            f: Function. Must take single parameter.
            a: start of interval containing the root
            b: end of interval containing the root
    '''
    epsilon = 1E-9
    maxsteps = 100

    for i in range(maxsteps):
        c = (a+b)/2
        fa = f(a)
        fb = f(b)
        fc = f(c)
        if fa*fc < 0:
            b = c
            if abs(fa-fc) < epsilon:
                break
        else:
            a = c
            if abs(fb-fc) < epsilon:
                break
    return c


def confidence(k: float, degf: float) -> float:
    ''' Calculate level of confidence from k-factor and degrees of freedom '''
    return bisect(lambda x: t_inv2t(x, degf)-k, 0.00001, .99999)


def degrees_freedom(k: float, conf: float) -> float:
    ''' Calculate degrees of freedom from k-factor and level of confidence '''
    return bisect(lambda x: t_inv2t(conf, x)-k, 1, 1000)


update(None)
