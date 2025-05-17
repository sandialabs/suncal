use mathru::algebra::linear::{matrix::General, matrix::Diagonal, vector::Vector, matrix::Transpose};
use mathru::algebra::linear::matrix::{Inverse};

use crate::result::CurveResult;
use crate::dists::{linspace};
use crate::cfg::CurveModel;


pub trait FitModel {
    fn name(&self) -> String;
    fn nparams(&self) -> usize;
    fn calc(&self, x: &Vector<f64>, p: &Vector<f64>) -> Vector<f64>;
    fn jacobian(&self, x: &Vector<f64>, p: &Vector<f64>) -> General<f64>;
}


pub struct LineFit {}
impl FitModel for LineFit {
    fn name(&self) -> String {
        "a*x + b".to_string()
    }
    fn nparams(&self) -> usize {2}
    fn calc(&self, x: &Vector<f64>, p: &Vector<f64>) -> Vector<f64> {
        let a = p[0];
        let b = p[1];
        let (_, n) = x.dim();
        let mut y: Vec<f64> = Vec::with_capacity(n);
        for i in 0..n {
            y.push(a*x[i] + b);
        }
        Vector::new_row(y)
    }
    fn jacobian(&self, x: &Vector<f64>, _p: &Vector<f64>) -> General<f64> {
        let (_, n) = x.dim();
        let mut ds: Vec<f64> = Vec::new();
        for i in 0..n {
            ds.push(-x[i]);
        }
        for _i in 0..n {
            ds.push(-1.0);
        }
        General::new(n, 2, ds)
    }
}


pub struct QuadFit {}
impl FitModel for QuadFit {
    fn name(&self) -> String {
        "a*x^2 + b*x + c".to_string()
    }
    fn nparams(&self) -> usize {3}
    fn calc(&self, x: &Vector<f64>, p: &Vector<f64>) -> Vector<f64> {
        let a = p[0];
        let b = p[1];
        let c = p[2];
        let (_, n) = x.dim();
        let mut y: Vec<f64> = Vec::with_capacity(n);
        for i in 0..n {
            y.push(a*x[i].powi(2) + b*x[i] + c);
        }
        Vector::new_row(y)
    }
    fn jacobian(&self, x: &Vector<f64>, _p: &Vector<f64>) -> General<f64> {
        let (_, n) = x.dim();
        let mut ds: Vec<f64> = Vec::new();
        for i in 0..n {
            ds.push(-x[i].powi(2));
        }
        for i in 0..n {
            ds.push(-x[i]);
        }
        for _i in 0..n {
            ds.push(-1.0);
        }
        General::new(n, 3, ds)
    }
}


pub struct CubicFit {}
impl FitModel for CubicFit {
    fn name(&self) -> String {
        "a*x^3 + b*x^2 + c*x + d".to_string()
    }
    fn nparams(&self) -> usize {4}
    fn calc(&self, x: &Vector<f64>, p: &Vector<f64>) -> Vector<f64> {
        let a = p[0];
        let b = p[1];
        let c = p[2];
        let d = p[3];
        let (_, n) = x.dim();
        let mut y: Vec<f64> = Vec::with_capacity(n);
        for i in 0..n {
            y.push(a*x[i].powi(3) + b*x[i].powi(2) + c*x[i] * d);
        }
        Vector::new_row(y)
    }
    fn jacobian(&self, x: &Vector<f64>, _p: &Vector<f64>) -> General<f64> {
        let (_, n) = x.dim();
        let mut ds: Vec<f64> = Vec::new();
        for i in 0..n {
            ds.push(-x[i].powi(3));
        }
        for i in 0..n {
            ds.push(-x[i].powi(2));
        }
        for i in 0..n {
            ds.push(-x[i]);
        }
        for _i in 0..n {
            ds.push(-1.0);
        }
        General::new(n, 4, ds)
    }
}


pub struct ExponentialFit {}
impl FitModel for ExponentialFit {
    fn name(&self) -> String {
        "a*exp(-x/b)".to_string()
    }
    fn nparams(&self) -> usize {2}
    fn calc(&self, x: &Vector<f64>, p: &Vector<f64>) -> Vector<f64> {
        let a = p[0];
        let b = p[1];
        let (_, n) = x.dim();
        let mut y: Vec<f64> = Vec::with_capacity(n);
        for i in 0..n {
            y.push(a*(-x[i]/b).exp());
        }
        Vector::new_row(y)
    }
    fn jacobian(&self, x: &Vector<f64>, p: &Vector<f64>) -> General<f64> {
        let a = p[0];
        let b = p[1];
        let (_, n) = x.dim();
        let mut ds: Vec<f64> = Vec::new();
        for i in 0..n {
            ds.push(-(-x[i]/b).exp());
        }
        for i in 0..n {
            ds.push(-a*x[i]/b/b*(-x[i]/b).exp());
        }
        General::new(n, 2, ds)
    }
}


pub struct DampedSineFit {}
impl FitModel for DampedSineFit {
    fn name(&self) -> String {
        "a*exp(-x/b)*sin(c*x)".to_string()
    }
    fn nparams(&self) -> usize {3}
    fn calc(&self, x: &Vector<f64>, p: &Vector<f64>) -> Vector<f64> {
        let a = p[0];
        let b = p[1];
        let c = p[2];
        let (_, n) = x.dim();
        let mut y: Vec<f64> = Vec::with_capacity(n);
        for i in 0..n {
            y.push(a*(-x[i]/b).exp()*(c*x[i]).sin());
        }
        Vector::new_row(y)
    }
    fn jacobian(&self, x: &Vector<f64>, p: &Vector<f64>) -> General<f64> {
        let a = p[0];
        let b = p[1];
        let c = p[2];
        let (_, n) = x.dim();
        let mut ds: Vec<f64> = Vec::new();
        for i in 0..n {
            ds.push(-(-x[i]/b).exp()*(c*x[i]).sin());
        }
        for i in 0..n {
            ds.push(-a*x[i]/b/b*(-x[i]/b).exp()*(c*x[i]).sin());
        }
        for i in 0..n {
            ds.push(-a*x[i]*(-x[i]/b).exp()*(c*x[i]).cos());
        }
        General::new(n, 3, ds)
    }
}


pub fn curve_eqn(model: &CurveModel) -> String {
    match model {
        CurveModel::Line => LineFit{}.name(),
        CurveModel::Quadratic => QuadFit{}.name(),
        CurveModel::Cubic => CubicFit{}.name(),
        CurveModel::Exponential => ExponentialFit{}.name(),
        CurveModel::DampedSine => DampedSineFit{}.name(),
    }
}


pub fn curve_fit(
    x: &Vector<f64>,
    y: &Vector<f64>,
    uy: Option<Vector<f64>>,
    p0: &Vector<f64>,
    model: &impl FitModel,
) -> CurveResult {
    // Gauss-Newton Method
    let iters: usize = 100;
    let epsilon: f64 = 1E-15;

    let (_, npoints) = x.clone().dim();
    let nparams = model.nparams();

    if npoints < 2 {
        return CurveResult::noresult()
    }
    if npoints != y.clone().dim().1 {
        return CurveResult::noresult()
    }
    if let Some(ref u) = uy {
        if npoints != u.dim().1 {
            return CurveResult::noresult()
        }
    }

    let mut p0 = p0.clone().convert_to_vec();
    p0.truncate(nparams);
    let mut p = Vector::new_row(p0);
    for i in 0..iters {
        let resid = y.clone() - model.calc(&x, &p);
        let w = match &uy {
            Some(sig) => {
                let vy: Vec<f64> = sig.clone().convert_to_vec().iter().map(|s| 1.0/(s*s)).collect();
                General::from(Diagonal::new(&vy))
            },
            None => General::one(npoints),
        };

        let j = model.jacobian(&x, &p);
        let jt = j.clone().transpose();
        let jtwj_inv = (jt.clone()*w.clone()*j.clone()).inv().unwrap();
        let delta1 = jt.clone()*w*resid.clone().transpose();
        let delta = delta1.transpose() * jtwj_inv;
        p -= delta.clone();
        if i > 0 && delta.convert_to_vec().iter().sum::<f64>() < epsilon { break; }
    }

    let resid = y.clone() - model.calc(&x, &p);
    let resid_matrix = General::from(resid.clone());
    let mse = ((resid_matrix.clone()*resid_matrix.clone().transpose()) / ((npoints - nparams) as f64))[[0,0]];

    let cov = match &uy {
        None => {
            // Unknown input uncertainty, use resids
            let j = model.jacobian(&x, &p);
            let jt = j.clone().transpose();
            let c1 = (jt * j).inv().unwrap();
            c1 * mse
        },
        Some(sig) => {
            let j = model.jacobian(&x, &p);
            let jt = j.clone().transpose();
            let vy: Vec<f64> = sig.iter().map(|s| s*s).collect();
            let v = General::from(Diagonal::new(&vy));
            let w = v.inv().unwrap();
            (jt*w*j).inv().unwrap()
        }
    };

    let xmin = x[x.argmin()];
    let xmax = x[x.argmax()];
    let xvals = Vector::new_row(linspace(xmin, xmax, 100));
    let yvals = model.calc(&xvals, &p);
    let conf = uconf(&xvals, &p, &cov, model);

    CurveResult{
        model: model.name(),
        p: p,
        cov: cov,
        mse: mse,
        resid: resid,
        nparams: nparams,
        npoints: npoints,
        xdata: x.clone(),
        ydata: y.clone(),
        xplot: xvals.clone(),
        yplot: yvals.clone(),
        conf_plus: yvals.clone() + conf.clone()*2.0,
        conf_minus: yvals.clone() - conf.clone()*2.0,
    }
}


pub fn uconf(x: &Vector<f64>, p: &Vector<f64>, cov: &General<f64>, model: &impl FitModel) -> Vector<f64> {
    let j = model.jacobian(&x, &p);
    let (_, n) = x.dim();

    let mut uconf: Vec<f64> = Vec::new();
    for i in 0..n {
        let delf = General::from(j.get_row(i)).transpose();
        let uc = (delf.clone().transpose() * cov.clone() * delf)[[0, 0]];
        uconf.push(uc.sqrt());
    }
    Vector::new_row(uconf)
}
// pub fn upred(x: &Vector<f64>, result: &CurveResult, model: &impl FitModel) -> Vector<f64> {
//     let (_, n) = x.dim();
//     let uc = uconf(x, result, model);
//     let mut upred: Vec<f64> = Vec::with_capacity(n);
//     for i in 0..n {
//         upred.push(
//             (uc[i].powi(2) + result.mse).sqrt()
//         );
//     }
//     Vector::new_row(upred)
// }

