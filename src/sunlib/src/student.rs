use std::{error::Error, fmt};
use std::f64::consts::FRAC_PI_2;


#[derive(Debug, Clone)]
pub struct NormalCdfError;
impl Error for NormalCdfError {}
impl fmt::Display for NormalCdfError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Invalid Range for Confidence")
    }
}

fn r1(z: f64) -> f64 {
    let z5 = z.powi(5);
    let z4 = z.powi(4);
    let z3 = z.powi(3);
    let z2 = z.powi(2);
    let num = -7.784894002430293E-3 * z5
        - 3.223964580411365E-1 * z4
        - 2.400758277161838 * z3
        - 2.549732539343734 * z2
        + 4.374664141464968 * z
        + 2.938163982698783;
    let denom = 7.784695709041462E-3 * z4
        + 3.224671290700398E-1 * z3
        + 2.445134137142996 * z2
        + 3.754408661907416 * z
        + 1.000000000000000;
    num / denom
}

fn r2(z: f64) -> f64 {
    let z5 = z.powi(5);
    let z4 = z.powi(4);
    let z3 = z.powi(3);
    let z2 = z.powi(2);
    let num = -3.969683028665376E1 * z5
              + 2.209460984245205E2 * z4
              - 2.759285104469687E2 * z3
              + 1.383577518672690E2 * z2
              - 3.066479806614716E1 * z
              + 2.506628277459239;
    let denom = -5.447609879822406E1 * z5
                + 1.615858368580409E2 * z4
                - 1.556989798598866E2 * z3
                + 6.680131188771972E1 * z2
                - 1.328068155288572E1 * z
                + 1.000000000000000;
    num / denom
}



pub fn norminv(p: f64) -> Result<f64, NormalCdfError> {
    // Inverse CDF of normal distribution
    // Implements Acklam's Chebyshev Approximation
    // as described in https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=4630740}

    match p {
        x if x < 0.0 => Err(NormalCdfError),
        x if x < 0.0245 => Ok(r1((-2.0*x.ln()).sqrt())),
        x if x <= 0.97575 => Ok((x - 0.5) * r2((x-0.5).powi(2))),
        x if x <= 1.0 => Ok(-r1((-2.0*(1.0-x).ln()).sqrt())),
        _ => Err(NormalCdfError),
    }
}

pub fn t_inv2t(conf: f64, degf: f64) -> Result<f64, NormalCdfError> {
    // Inverse 2-tailed student T
//             Implements "Algorithm 396 Student's T-Quantiles" by G.W. Hill,
//             Communications of the ACM, Volume 13, Number 10, October 1970
//             (with "Remark on Algorithm 396" July 1979).
//             https://dl.acm.org/doi/pdf/10.1145/355598.355600
//             https://dl.acm.org/doi/pdf/10.1145/355945.355956
    if conf < 0.0 || conf > 1.0 {
        return Err(NormalCdfError);
    }

    let n = if degf.is_finite() { degf } else { 1E9 };

    let p = 1.0 - conf;
    match n {
        nn if nn == 2.0 => Ok((2.0 / p*(2.0-p) - 2.0).sqrt()),
        nn if nn == 1.0 => Ok((p*FRAC_PI_2).cos() / (p*FRAC_PI_2).sin()),
        _ => {
                let a = 1.0 / (n - 0.5);
                let b = 48.0 / a.powi(2);
                let mut c = ((20700.0 * a / b - 98.0) * a - 16.0) * a + 96.36;
                let d = ((94.5/(b+c)-3.0)/b+1.0) * (a*FRAC_PI_2).sqrt() * n;
                let mut x = d*p;
                let mut y = x.powf(2.0/n);
                if y > 0.05+a {
                    x = norminv(p*0.5)?;
                    y = x.powi(2);
                    if n < 5.0 {
                        c = c+0.3*(n-4.5)*(x+0.6);
                    };
                    c = (((0.05*d*x-5.0)*x-7.0)*x-2.0)*x+b+c;
                    y = (((((0.4*y+6.3)*y+36.0)*y+94.5)/c-y-3.0)/b+1.0)*x;
                    y = a*y.powi(2);
                    if y > 0.1 {
                        y = y.exp() - 1.0;
                    } else {
                        y = ((y+4.0)*y+12.0)*y*y/24.0 + y;
                    };
                } else {
                    y = ((1.0/(((n+6.0)/(n*y) - 0.089*d - 0.822) * (n+2.0)*3.0)+0.5/(n+4.0))*y-1.0) * (n+1.0)/(n+2.0) + 1.0/y;
                }
                Ok((n * y).sqrt())
            }
    }
}

// def bisect(f, a: float, b: float) -> float:
//     ''' Find roots of function by bisection

//         Args:
//             f: Function. Must take single parameter.
//             a: start of interval containing the root
//             b: end of interval containing the root
//     '''
//     epsilon = 1E-9
//     maxsteps = 100

//     for i in range(maxsteps):
//         c = (a+b)/2
//         fa = f(a)
//         fb = f(b)
//         fc = f(c)
//         if fa*fc < 0:
//             b = c
//             if abs(fa-fc) < epsilon:
//                 break
//         else:
//             a = c
//             if abs(fb-fc) < epsilon:
//                 break
//     return c


// def confidence(k: float, degf: float) -> float:
//     ''' Calculate level of confidence from k-factor and degrees of freedom '''
//     return bisect(lambda x: t_inv2t(x, degf)-k, 0.00001, .99999)


// def degrees_freedom(k: float, conf: float) -> float:
//     ''' Calculate degrees of freedom from k-factor and level of confidence '''
//     return bisect(lambda x: t_inv2t(conf, x)-k, 1, 1000)