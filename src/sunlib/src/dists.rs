// Probability Distributions
use std::fmt;
use std::f64::consts::{PI};
use serde::{Serialize, Deserialize};
use roots::find_root_brent;
use mathru::algebra::linear::{matrix::General};
use mathru::algebra::linear::matrix::{CholeskyDecomposition};
use mathru::statistics::distrib::{Distribution as DistributionRu, Continuous, Normal, Gamma, Uniform};

use crate::cfg::{CorrelationCoeff, TypeBDist, TypeBNormal, TypeBUniform, TypeBTriangular, TypeBGamma, TypeBTolerance, Tolerance};
use crate::result::{QuantityResult, get_qresult};


const PDF_N: usize = 1001;
const SQRT_2PI: f64 = 2.5066282746310002;


#[derive(Debug)]
struct DistributionError {
    msg: String,
}
impl std::error::Error for DistributionError {}
impl fmt::Display for DistributionError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.msg)
    }
}


// TypeB distributions don't have a nominal/mean value
impl TypeBDist {
    pub fn check(&self) -> Result<(), Box<dyn std::error::Error>> {
        match self {
            TypeBDist::Normal(d) => if d.stddev > 0.0 { Ok(()) } else { Err(Box::new(DistributionError{msg:String::from("Standard Deviation must be positive")}))},
            TypeBDist::Uniform(d) => if d.a > 0.0 { Ok(()) } else { Err(Box::new(DistributionError{msg:String::from("Uniform distribution `a` must be positive")}))},
            TypeBDist::Triangular(d) => if d.a > 0.0 { Ok(()) } else { Err(Box::new(DistributionError{msg:String::from("Triangular distribution `a` must be positive")}))},
            TypeBDist::Gamma(d) => if d.a > 0.0 && d.b > 0.0 { Ok(()) } else { Err(Box::new(DistributionError{msg:String::from("Gamma distribution `a` and `b` must be positive")}))},
            _ => Ok(()),
        }
    }
    pub fn scale(&self, scale: f64) -> TypeBDist {
        // Apply scale factor (ie units)
        match self {
            TypeBDist::Normal(d) => TypeBDist::Normal(TypeBNormal::new(d.stddev * scale)),
            TypeBDist::Uniform(d) => TypeBDist::Uniform(TypeBUniform{a: d.a*scale, degf: d.degf, name: d.name.clone()}),
            TypeBDist::Triangular(d) => TypeBDist::Triangular(TypeBTriangular{a: d.a*scale, degf: d.degf, name: d.name.clone()}),
            TypeBDist::Gamma(d) => TypeBDist::Gamma(TypeBGamma{a: d.a*scale, b: d.b, degf: d.degf, name: d.name.clone()}),
            TypeBDist::Tolerance(d) => TypeBDist::Tolerance(TypeBTolerance{tolerance: d.tolerance*scale, confidence:d.confidence, kfactor:d.kfactor, degf:d.degf, name:d.name.clone()}),
            _ => todo!(),
        }
    }
    pub fn variance(&self, qty_results: Option<&Vec<QuantityResult>>) -> f64 {
        match self {
            TypeBDist::Normal(d) => d.stddev.powi(2),
            TypeBDist::Uniform(d) => d.a.powi(2) / 3.0,
            TypeBDist::Triangular(d) => d.a.powi(2) / 6.0,
            TypeBDist::Tolerance(d) => {
                if d.kfactor.is_finite() {
                    (d.tolerance / d.kfactor).powi(2)
                } else {
                    (d.tolerance / normal_inv_cdf((1.0+d.confidence)/2.0, 0.0, 1.0)).powi(2)
                }
            },
            TypeBDist::Gamma(d) => d.a / d.b.powi(2),
            TypeBDist::Symbol(d) => {
                match get_qresult(d, qty_results) {
                    Some(qty) => {
                        match &qty.reliability {
                            Some(r) => r.sigma_aop.powi(2),
                            None => 0.0,
                        }
                    },
                    None => {
                        eprintln!("Warning - Undefined symbol: {}", d);
                        0.0
                    },
                }
            }
        }
    }
    pub fn std_dev(&self, qty_results: Option<&Vec<QuantityResult>>) -> f64 {
        self.variance(qty_results).sqrt()
    }
    pub fn degrees_freedom(&self, qty_results: Option<&Vec<QuantityResult>>) -> f64 {
        match self {
            TypeBDist::Normal(d) => d.degf,
            TypeBDist::Uniform(d) => d.degf,
            TypeBDist::Triangular(d) => d.degf,
            TypeBDist::Gamma(d) => d.degf,
            TypeBDist::Tolerance(d) => d.degf,
            TypeBDist::Symbol(d) => {
                match get_qresult(d, qty_results) {
                    Some(qty) => {
                        qty.uncertainty.degrees_freedom()
                    },
                    None => f64::INFINITY
                }
            }
        }
    }
    pub fn pdf_given_y(&self, y: f64, qty_results: Option<&Vec<QuantityResult>>) -> Distribution {
        // Place uncertainty at the measured value y and convert to Distribution
        match self {
            TypeBDist::Normal(d) => Distribution::Normal{mu: y, sigma: d.stddev},
            TypeBDist::Uniform(d) => Distribution::Uniform{mu: y, a: d.a},
            TypeBDist::Triangular(d) => Distribution::Triangular{mu: y, a: d.a},
            TypeBDist::Gamma(d) => Distribution::Gamma{a: d.a, b: d.b},
            TypeBDist::Tolerance(d) => {
                let sigma = if d.kfactor.is_finite() {
                    d.tolerance / d.kfactor
                } else {
                    d.tolerance / normal_inv_cdf((1.0+d.confidence)/2.0, 0.0, 1.0)
                };
                Distribution::Normal{mu: y, sigma: sigma}
            },
            TypeBDist::Symbol(d) => {
                let sigma = match get_qresult(d, qty_results) {

                    Some(qty) => {
                        match &qty.reliability {
                            Some(r) => r.sigma_aop,
                            None => 0.0,
                        }
                    },
                    None => 0.0,
                };
                Distribution::Normal{mu: y, sigma: sigma}
            },
        }
    }
    pub fn integrate_given_y(&self, a: f64, b: f64, qty_results: Option<&Vec<QuantityResult>>) -> Distribution {
        // Integrate f(x|y) dy from a to b, returning Distribution
        // Same as convolving a step function between a and b
        let rect = Distribution::Step{a: a, b: b};
        rect.convolve(&self, qty_results)
    }
}


pub fn normal_cdf(x: f64, mu: f64, sigma: f64) -> f64 {
    Normal::new(mu, sigma.powi(2)).cdf(x)
}
fn normal_inv_cdf(p: f64, mu: f64, sigma: f64) -> f64 {
    Normal::new(mu, sigma.powi(2)).quantile(p)
}

pub fn linspace(start: f64, stop: f64, n: usize) -> Vec<f64> {
    // Vector of evenly spaced values
    let mut out: Vec<f64> = Vec::with_capacity(n);
    let step = (stop - start) / (n as f64 - 1.0);
    for i in 0..n {
        out.push(start + i as f64 * step);
    }
    out
}

pub fn trapz_integral(v: &[f64], dx: f64) -> f64 {
    // Trapezoidal integration
    let n = v.len();
    if n > 2 {
        let mut i: f64 = v[1..n-1].iter().sum::<f64>() * 2.0;
        i += v[0];
        i += v[n-1];
        i * dx / 2.0
    } else {
        0.0
    }
}

fn interpolate(x: f64, xs: &[f64], ys: &[f64]) -> f64 {
    // Linear interpolation
    let i: usize = xs.iter().position(|v| *v > x).unwrap_or(1);
    if i == 0 {
        ys[0]
    } else if i >= xs.len() {
        ys[xs.len()-1]
    } else {
        ys[i-1] + (x - xs[i-1]) * (ys[i] - ys[i-1]) / (xs[i] - xs[i-1])
    }
}


pub fn std_from_itp(itp: f64, nominal: f64, tolerance: &Tolerance) -> f64 {
    let plusminus = (tolerance.high - tolerance.low) / 2.0;
    let center = (tolerance.high + tolerance.low) / 2.0;
    let bias = (nominal - center) / plusminus;

    if bias.abs() < 1E-12 {
        plusminus / normal_inv_cdf((1.0+itp)/2.0, 0.0, 1.0)
    } else {
        match find_root_brent(1E-9f64, 10.0, |x| {
            let d = Distribution::Normal{mu: bias, sigma: x};
            d.cdf(1.0) - d.cdf(-1.0) - itp
        }, &mut 1E-5f64) {
            Ok(sig) => sig * plusminus,
            Err(_) => f64::NAN,
        }
    }
}
pub fn itp_from_norm(nominal: f64, stddev: f64, tolerance: &Tolerance) -> f64 {
    let pdf = Distribution::Normal{mu: nominal, sigma: stddev};
    pdf.cdf(tolerance.high) - pdf.cdf(tolerance.low)
}


// Distributions are arbitrary PDFs or probability curves
#[derive(Serialize, Deserialize, Clone, Debug)]
pub enum Distribution {
    Normal{mu: f64, sigma: f64},
    NormalItp{mu: f64, itp: f64, low: f64, high: f64},
    Uniform{mu: f64, a: f64},
    Triangular{mu: f64, a: f64},
    Gamma{a: f64, b: f64},
    Discrete{x: Vec<f64>, y: Vec<f64>},
    Step{a: f64, b: f64},
}
impl Distribution {
    pub fn check(&self) -> Result<(), Box<dyn std::error::Error>> {
        match self {
            Distribution::Normal{mu: _, sigma} => if *sigma > 0.0 { Ok(()) } else { Err(Box::new(DistributionError{msg: String::from("Standard Deviation must be positive")}))},
            Distribution::Uniform{mu: _, a} => if *a > 0.0 { Ok(()) } else { Err(Box::new(DistributionError{msg:String::from("Uniform distribution `a` must be positive")})) },
            Distribution::Triangular{mu: _, a} => if *a > 0.0 { Ok(()) } else { Err(Box::new(DistributionError{msg:String::from("Triangular distribution `a` must be positive")}))},
            Distribution::Gamma{a, b} => if *a > 0.0 && *b > 0.0 { Ok(()) } else { Err(Box::new(DistributionError{msg:String::from("Gamma distribution `a` and `b` must be positive")}))},
            Distribution::Discrete{x: _, y: _} => Ok(()),
            Distribution::Step{a: _, b: _} => Ok(()),
            Distribution::NormalItp{mu: _, itp, low: _, high: _} => if *itp > 0.0 && *itp < 1.0 { Ok(()) } else { Err(Box::new(DistributionError{msg: String::from("ITP must be between 0 and 100%")}))},
        }
    }
    pub fn from_itp(center: f64, itp: f64, tolerance: &Tolerance) -> Distribution {
        // Create a Normal distribution from ITP/EOPR and tolerance
        let sigma = std_from_itp(itp, center, tolerance);
        Distribution::Normal{mu: center, sigma: sigma}
    }
    pub fn standard_norm() -> Distribution {
        // Standard Normal distribution
        Distribution::Normal{mu: 0.0, sigma: 1.0}
    }
    pub fn pdf(&self, x: f64) -> f64 {
        // Get probability given x
        match self {
            Distribution::Normal{mu, sigma} => (SQRT_2PI * sigma).recip() * (-(x-mu).powi(2) / (2.0*sigma.powi(2))).exp(),
            Distribution::NormalItp{mu, itp, low, high} => {
                let sigma = std_from_itp(*itp, *mu, &Tolerance{low:*low, high:*high});
                // Getting weird results using mathru::Normal::pdf()...?
                (SQRT_2PI * sigma).recip() * (-(x-mu).powi(2) / (2.0*sigma.powi(2))).exp()
            },
            Distribution::Step{a, b} => {
                if x < *a || x > *b { 0.0 } else { 1.0 }
            },
            Distribution::Uniform{mu, a} => {
                if x < mu-a || x > mu+a { 0.0 } else { (2.0 * a).recip() }
            }
            Distribution::Triangular{mu, a} => {
                if x >= mu-a && x <= *mu {
                    (x-(mu-a))/a.powi(2)
                } else if x > *mu && x <= mu+a {
                    ((mu+a)-x)/a.powi(2)
                } else {
                    0.0
                }
            },
            Distribution::Gamma{a, b} => {
                if *a > 0.0 && *b > 0.0 && x > 0.0 {
                    Gamma::new(*a, *b).pdf(x)
                } else {
                    0.0
                }
            },
            Distribution::Discrete{x: xs, y: ys} => interpolate(x, &xs, &ys),
        }
    }
    pub fn cdf(&self, x: f64) -> f64 {
        // Get cumulative probability up to x
        match self {
            Distribution::Discrete{x: xx, y: _} => {
                self.integrate(xx[0], x)
            },
            Distribution::Normal{mu, sigma} => normal_cdf(x, *mu, *sigma),
            Distribution::NormalItp{mu, itp, low, high} => {
                let sigma = std_from_itp(*itp, *mu, &Tolerance{low:*low, high:*high});
                normal_cdf(x, *mu, sigma)
            },
            Distribution::Uniform{mu, a} => {
                if x < mu-a || x > mu+a { 0.0 } else {
                    (x - mu - a) / (a * 2.0)
                }
            },
            Distribution::Triangular{mu, a} => {
                match x {
                    x if x < mu-a || x > mu + a => 0.0,
                    x if x < *mu => 2.0*(x-mu+a)/(2.0*a*a),
                    _ => 2.0*(mu+a-x)/(2.0*a*a),
                }
            },
            Distribution::Gamma{a, b} => {
                match x {
                    x if x <= 0.0 => 0.0,
                    _ => Gamma::new(*a, *b).cdf(x),
                }
            },
            _ => todo!()
        }
    }
    pub fn inverse_cdf(&self, p: f64) -> f64 {
        // Inverse Cumulative Distribution Function
        match self {
            Distribution::Normal{mu, sigma} => {
                Normal::<f64>::new(*mu, sigma.powi(2)).quantile(p)
            },
            Distribution::NormalItp{mu, itp, low, high} => {
                let sigma = std_from_itp(*itp, *mu, &Tolerance{low:*low, high:*high});
                Normal::<f64>::new(*mu, sigma.powi(2)).quantile(p)
            },
            Distribution::Uniform{mu, a} => Uniform::<f64>::new(mu-a, mu+a).quantile(p),
            Distribution::Gamma{a, b} => Gamma::<f64>::new(*a, *b).quantile(p),
            Distribution::Triangular{mu, a} => {
                if p < 0.5 {
                    mu - a + (p*2.0*a*a).sqrt()
                } else {
                    mu + a - ((1.0-p)*2.0*a*a).sqrt()
                }
            }
            _ => todo!(),
        }
    }
    pub fn domain(&self) -> (f64, f64) {
        // Get domain of values covering .9999... something of values
        match self {
            Distribution::Normal{mu, sigma} => (mu - 6.0*sigma, mu + 6.0*sigma),
            Distribution::Step{a, b} => {let w = b-a; (a-w/2.0, b+w/2.0)},
            Distribution::Uniform{mu, a} => (mu - 2.0*a, mu + 2.0*a),
            Distribution::Triangular{mu, a} => (mu - 2.0*a, mu + 2.0*a),
            Distribution::Gamma{a, b} => (0.0, a/b + 6.0*a.sqrt()/b),
            Distribution::Discrete{x, y: _} => (x[0], x[x.len()-1]),
            Distribution::NormalItp{mu, itp, low, high} => {
                let sigma = std_from_itp(*itp, *mu, &Tolerance{low:*low, high:*high});
                (mu - 6.0*sigma, mu + 6.0*sigma)
            },
        }
    }
    pub fn std_dev(&self) -> f64 {
        match self {
            Distribution::Normal{mu: _, sigma} => *sigma,
            Distribution::Step{a, b} => {(b-a)/2.0/3.0_f64.sqrt()},
            Distribution::Uniform{mu: _, a} => a/3.0_f64.sqrt(),
            Distribution::NormalItp{mu, itp, low, high} => std_from_itp(*itp, *mu, &Tolerance{low:*low, high:*high}),
            Distribution::Triangular{mu: _, a} => a/6.0_f64.sqrt(),
            Distribution::Gamma{a, b} => Gamma::<f64>::new(*a, *b).variance().sqrt(),
            _ => todo!(),
        }
    }
    fn xvalues(&self) -> Vec<f64> {
        // Array of x values for discrete representation
        match self {
            Distribution::Discrete{x, y: _} => x.clone(),
            _ => {
                let (x1, x2) = self.domain();
                linspace(x1, x2, PDF_N)
            },
        }
    }
    fn to_discrete(&self, xvals: &Vec<f64>) -> Distribution {
        // Convert to discrete representation
        let y: Vec<f64> = xvals.iter().map(|x| self.pdf(*x)).collect();
        Distribution::Discrete{x:xvals.clone(), y:y}
    }
    pub fn integrate(&self, a: f64, b: f64) -> f64 {
        // Integrate distribution from a to b
        match self {
            Distribution::Discrete{x, y} => {
                let mut i1 = x.iter().position(|x| *x > a).unwrap_or(1);
                let mut i2 = x.iter().position(|x| *x > b).unwrap_or(y.len()+1);
                i1 = i1.max(1) - 1;
                i2 = i2.max(1) - 1;
                trapz_integral(&y[i1..i2], x[1]-x[0])
            },
            _ => self.cdf(b) - self.cdf(a),
        }
    }
    pub fn mul(&self, other: &Distribution, prob: f64) -> Distribution {
        // Element-wise Multiply the PDFs and multiply by prob
        let (d1, d2) = self.discrete_match(other);
        let mut out: Vec<f64> = Vec::with_capacity(PDF_N);
        let (xx, y1, y2) = match (d1, d2) {
            (Distribution::Discrete{x, y: y1}, Distribution::Discrete{x: _, y: y2}) => (x, y1, y2),
            _ => panic!(),
        };
        for i in 0..PDF_N {
            out.push(y1[i] * y2[i] * prob);
        }
        Distribution::Discrete{x: xx.clone(), y: out}
    }
    pub fn sum(&self, other: &Distribution, weight1: f64, weight2: f64, prob: f64) -> Distribution {
        // Weighted sum of two Distributions
        let (d1, d2) = self.discrete_match(other);
        let mut out: Vec<f64> = Vec::with_capacity(PDF_N);
        let (xx, y1, y2) = match (d1, d2) {
            (Distribution::Discrete{x, y: y1}, Distribution::Discrete{x: _, y: y2}) => (x, y1, y2),
            _ => panic!(),
        };
        for i in 0..PDF_N {
            out.push( (y1[i] * weight1  +  y2[i] * weight2) / prob );
        }
        Distribution::Discrete{x: xx.clone(), y: out}
    }
    pub fn invert(&self) -> Distribution {
        // 1 - Distribution
        let d1 = self.to_discrete(&self.xvalues());
        let (x, y) = match d1 {
            Distribution::Discrete{x, y} => (x, y),
            _ => panic!(),
        };
        let mut out: Vec<f64> = Vec::with_capacity(PDF_N);
        for i in 0..PDF_N {
            out.push(-y[i])
        }
        Distribution::Discrete{x: x.clone(), y: out}
    }
    pub fn discrete_match(&self, other: &Distribution) -> (Distribution, Distribution) {
        // Discretize both Distributions with same x values
        let d1 = self.domain();
        let d2 = other.domain();
        let dmin = d1.0.min(d2.0);
        let dmax = d1.1.max(d2.1);
        let nrange = linspace(dmin, dmax, PDF_N);
        let p1 = self.to_discrete(&nrange);
        let p2 = other.to_discrete(&nrange);
        (p1, p2)
    }
    pub fn convolve(&self, typeb: &TypeBDist, qty_results: Option<&Vec<QuantityResult>>) -> Distribution {
        // Convolve this Distribution with a TypeB distribution
        // Most combos will be done numerically and return a Discrete
            // Normals = RSS
            // Gammas with equal beta = sum the alphas
            // Poisson => Poisson sum lambdas
        match (self, typeb) {
            (Distribution::Normal{mu, sigma}, TypeBDist::Normal(_)) =>
                { Distribution::Normal{mu: *mu, sigma: (sigma.powi(2) + typeb.variance(qty_results)).sqrt()}},
            // Gamma + Gamma =>
            // Poisson + Poisson =>
            _ => {
                let (a, b) = self.domain();
                let (p1, p2) = self.discrete_match(&typeb.pdf_given_y((a+b)/2.0, qty_results));
                let (x, y1, y2) = match (p1, p2) {
                    (Distribution::Discrete{x, y: y1}, Distribution::Discrete{x: _, y: y2}) => (x, y1, y2),
                    _ => panic!(),
                };
                let dx = x[1]-x[0];

                let mut out: Vec<f64> = Vec::with_capacity(PDF_N);
                for m in 0..PDF_N {
                    let mut val: f64 = 0.0;
                    for n in 0..PDF_N {
                        let idx: i32 = PDF_N as i32 / 2 + m as i32 - n as i32;
                        if idx >= 0 && idx < PDF_N as i32 - 1 {
                            val += y1[idx as usize] * y2[n];
                        }
                    }
                    out.push(val * dx);
                }
                Distribution::Discrete{x: x, y: out}
            },
        }
    }
    pub fn cosine(degrade: &Tolerance, fail: &Tolerance) -> Distribution {
        // Cosine utility function
        let span = fail.high - fail.low;
        let low = fail.low - 1.0*span;
        let high = fail.high + 1.0*span;
        let xrange = linspace(low, high, PDF_N);

        let mut y = Vec::<f64>::with_capacity(PDF_N);
        for x in xrange.iter() {
            if *x <= fail.low || *x >= fail.high {
                y.push(0.0);
            } else if *x >= degrade.low && *x <= degrade.high {
                y.push(1.0);
            } else if *x > fail.low && *x < degrade.low {
                y.push(((degrade.low - *x) * PI / (2.0 * (degrade.low - fail.low))).cos().powi(2));
            } else {  // *x > degrade.high && x < fail.high
                y.push(((*x - degrade.high) * PI / (2.0 * (fail.high - degrade.high))).cos().powi(2));
            }
        }
        Distribution::Discrete{x: xrange, y: y}
    }
}

pub fn multivariate_random(nsamples: usize, varnames: Vec<String>, corr: &Vec<CorrelationCoeff>) -> General<f64> {
    let nvars = varnames.len();
    let mut cov = General::<f64>::one(nvars);

    for cc in corr {
        let idx1 = varnames.iter().position(|r| *r == cc.v1).unwrap();
        let idx2 = varnames.iter().position(|r| *r == cc.v2).unwrap();
        let sub = General::<f64>::new(1, 1, vec![cc.coeff]);
        cov = cov.set_slice(&sub, idx1, idx2);
    }

    let standard_norm: Normal<f64> = Normal::new(0.0, 1.0);
    let r = &cov.clone().dec_cholesky().unwrap().l();

    let mut samples = General::<f64>::zero(nsamples, nvars);
    for i in 0..nsamples {
        let z = General::new(nvars, 1, standard_norm.random_sequence(nvars.try_into().unwrap()));
        let row = General::<f64>::from(r.clone()*z.into());
        samples.set_row(&row.get_column(0).transpose(), i);
    }
    samples.apply(&|x| standard_norm.cdf(*x))
}

pub fn random(nsamples: usize, n: usize) -> General<f64> {
    let standard_norm: Normal<f64> = Normal::new(0.0, 1.0);
    let s = standard_norm.random_sequence((n*nsamples).try_into().unwrap());
    let s = s.iter().map(|x| standard_norm.cdf(*x)).collect();
    let samples = General::new(nsamples, n, s);
    samples
}

pub fn norm_pdf(x: f64, mu: f64, sigma: f64) -> f64 {
    (SQRT_2PI * sigma).recip() * (-(x-mu).powi(2) / (2.0*sigma.powi(2))).exp()
}