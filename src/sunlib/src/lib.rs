// Suncal Library Functions
use std::{error::Error, fmt};
use exmex::prelude::*;
use mathru::vector;
use mathru::algebra::linear::matrix::{General, Diagonal, Transpose};
use mathru::algebra::linear::vector::Vector;

mod unique;
mod student;
mod stats;
pub mod dists;
pub mod cfg;
pub mod result;
pub mod risk;
pub mod curves;
use crate::dists::{Distribution, normal_cdf, std_from_itp, itp_from_norm};
use crate::cfg::{ModelQuantity, ModelFunction, MeasureSystem, TypeBNormal, Tolerance, Utility, Costs, Eopr, Guardband, RenewalPolicy, ReliabilityModel, TypeBTolerance, TypeBDist, Interval, IntervalTarget, Calibration, CurveModel};
use crate::result::{UncertResult, QuantityResult, GumResult, GumComponents, MonteCarloResult, CurveResult, RiskResult, RiskResultGlobal, ReliabilityResult, CostResult, SystemResult, ReliabilityDecay, ReliabilityDecayParameters, get_qresult};
use crate::risk::RiskModel;
use crate::unique::Unique;
use crate::curves::{LineFit, QuadFit, CubicFit, ExponentialFit, DampedSineFit, curve_fit};
use units;


// Quantities, Functions, and eventually Curves implement this Trait
trait Uncertainty {
    fn symbol(&self) -> String;
    fn utility(&self) -> &Option<Utility>;
    fn interval(&self) -> &Option<Interval>;
    fn costs(&self) -> &Option<Costs>;
    fn tolerance(&self) -> Option<Tolerance>;
    fn calibration(&self) -> &Option<Calibration>;
    fn enditem(&self) -> bool;
}

#[derive(Debug, PartialEq, Eq)]
struct UndefinedQuantityError {
    msg: String
}
impl Error for UndefinedQuantityError {}
impl fmt::Display for UndefinedQuantityError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.msg)
    }
}


impl ModelQuantity {
    pub fn new() -> Self {
        // New Quantity for Uncertainty model
        Self{
            name: String::new(),
            symbol: String::from("a"),
            measured: 0.0,
            units: Some(String::new()),
            typeb: vec![],
            repeatability: None,
            reproducibility: None,
            new_meas: 0,
            utility: None,
            interval: None,
            calibration: None,
            cost: None,
        }
    }
    pub fn new_mqa() -> Self {
        // New Quantity for MQA Model
        Self{
            name: String::new(),
            symbol: String::from("a"),
            measured: 0.0,
            units: None,
            typeb: vec![
                TypeBDist::Tolerance(TypeBTolerance{tolerance: 0.25, confidence:0.9545, kfactor: f64::NAN, degf: f64::INFINITY, name: String::new()})
            ],
            repeatability: None,
            reproducibility: None,
            new_meas: 0,
            utility: Some(Utility::default()),
            interval: Some(Interval::default()),
            calibration: None,
            cost: None,
        }
    }
    pub fn expected(&self) -> f64 {
        // Expectation
        let ex = if self.new_meas > 0 {
            self.measured
        } else {
            match &self.repeatability {
                Some(r) => stats::mean(&r),
                None => {
                    match &self.reproducibility {
                        Some(rr) => stats::grand_mean(&rr),
                        None => self.measured,
                    }
                }
            }
        };
        ex
    }
    fn degrees_freedom(&self, variance: f64, qty_results: Option<&Vec<QuantityResult>>) -> f64 {
        // Effective Degrees of Freedom
        // Welch-Satterthwaite
        let mut denom: f64 = 0.0;
        for typeb in self.typeb.iter() {
            let degf = typeb.degrees_freedom(qty_results);
            if degf.is_finite() {
                denom += typeb.variance(qty_results).powi(2) / degf;
            }
        }
        denom += match &self.repeatability {
            Some(v) => {
                let variance = stats::variance(v) / v.len() as f64;
                let num = match self.new_meas {
                    0 => v.len() as f64 - 1.0,
                    _ => self.new_meas as f64,
                };
                variance.powi(2) / num
            },
            None => 0.0,
        };
        denom += match &self.reproducibility {
            Some(v) => stats::reproducibility(v).powi(2) / stats::reproducibility_degf(v),
            None => 0.0,
        };
        if denom == 0.0 {
            f64::INFINITY
        } else {
            variance*variance / denom
        }
    }
    pub fn variance(&self, qty_results: &Vec<QuantityResult>) -> f64 {
        // Standard variance RSSing components
        // combine typebs and R/R
        let mut variance: f64 = 0.0;
        for typeb in self.typeb.iter() {
            variance += typeb.variance(Some(qty_results))
        }

        // Repeatability
        variance += match &self.repeatability {
            Some(v) => stats::variance(v) / v.len() as f64,
            None => 0.0,
            // TODO - autocorrelation if n > some threshold
        };

        // Reproducibility
        variance += match &self.reproducibility {
            Some(v) => stats::reproducibility(v).powi(2),
            None => 0.0,
        };
        variance
    }
    fn pdf(&self) -> Option<Distribution> {
        // Build PDF combining all TypeB into one Distribution
        let nom = units::make_baseqty(self.expected(), self.units.clone(), true).unwrap();
        if self.typeb.len() > 0 {
            let scale = match &self.units {
                Some(u) => {let unit = units::parse_unit(u).unwrap();
                            unit.scale},
                None => 1.0,
            };
            let mut pdf = self.typeb[0].scale(scale).pdf_given_y(nom.magnitude, None);
            for typeb in self.typeb[1..].iter() {
                pdf = pdf.convolve(&typeb.scale(scale), None);
            }
            for ta in self.typea_pdfs() {
                pdf = pdf.convolve(&ta, None);
            }
            Some(pdf)
        } else {
            let tas = self.typea_pdfs();
            if tas.len() > 0 {
                let mut pdf = tas[0].pdf_given_y(nom.magnitude, None);
                if tas.len() > 1 {
                    pdf = pdf.convolve(&tas[1], None);
                }
                Some(pdf)
            } else {
                None
            }
        }
    }
    fn typea_pdfs(&self) -> Vec<TypeBDist> {
        let scale = match &self.units {
            Some(u) => {let unit = units::parse_unit(u).unwrap();
                        unit.scale},
            None => 1.0,
        };
        let mut out: Vec<TypeBDist> = Vec::new();
        match &self.repeatability {
            Some(r) => {
                out.push(
                    TypeBDist::Normal(
                        TypeBNormal::new(stats::std_dev(r) / (r.len() as f64).sqrt())
                    ).scale(scale)
                )
            },
            None => {},
        }
        match &self.reproducibility {
            Some(r) => {
                out.push(
                    TypeBDist::Normal(
                        TypeBNormal::new(stats::reproducibility(r))
                    ).scale(scale)
                )
            },
            None => {},
        }
        out
    }
    fn evaluate(&self, conf: f64, qty_results: &Vec<QuantityResult>) -> Result<UncertResult, Box<dyn Error>> {
        let variance = self.variance(qty_results);
        let degf = self.degrees_freedom(variance, Some(qty_results));
        let gum = GumResult{
            expected: self.expected(),
            variance: variance,
            std_dev: variance.sqrt(),
            units: self.units.clone(),
            degrees_freedom: degf,
            coverage_factor: student::t_inv2t(conf, degf).unwrap_or(f64::INFINITY),
            confidence: conf,
            ..Default::default()
        };
        Ok(UncertResult::Gum(gum))
    }
    fn evaluate_mc(&self, standard_samples: &Vec<f64>, conf: f64) -> Result<UncertResult, Box<dyn Error>> {
        let samples = match self.pdf() {
            Some(pdf) => Vec::<f64>::from(standard_samples.iter().map(|x| {pdf.inverse_cdf(*x)}).collect::<Vec<f64>>()),
            None => {
                vec![self.expected(); standard_samples.len()]
            },
        };
        Ok(UncertResult::Montecarlo(MonteCarloResult::new(samples, self.units.clone(), conf)))
    }
}
impl Uncertainty for ModelQuantity {
    fn symbol(&self) -> String {
        self.symbol.clone()
    }
    fn utility(&self) -> &Option<Utility> {
        &self.utility
    }
    fn interval(&self) -> &Option<Interval> {
        &self.interval
    }
    fn tolerance(&self) -> Option<Tolerance> {
        match &self.utility {
            Some(v) => Some(v.tolerance.clone()),
            None => None,
        }
    }
    fn calibration(&self) -> &Option<Calibration> {
        &self.calibration
    }
    fn costs(&self) -> &Option<Costs> {
        &self.cost
    }
    fn enditem(&self) -> bool {
        match self.costs() {
            Some(c) => {
                c.cost_fa > 0.0 || c.cost_fr > 0.0
            },
            None => false
        }
    }
}


impl ModelFunction {
    fn calc_monte(&self,
                   qty_results: &Vec<QuantityResult>,
                   nsamples: usize,
                    conf: f64,
                ) -> Result<MonteCarloResult, Box<dyn Error>> {
        let expr = units::parse_expr_f64(&self.expr)?;
        let varnames = expr.var_names();
        let mut samples = Vec::<f64>::with_capacity(nsamples);
        let quantities: Vec<&QuantityResult> = varnames.iter().map(|v| get_qresult(v, Some(qty_results)).unwrap()).collect();
        let mcresults: Vec<&MonteCarloResult> = quantities.iter().map(|var| {
            match &var.uncertainty {
                UncertResult::Montecarlo(v) => v,
                UncertResult::Gum(v) => {
                    &v.mcsamples.as_ref().unwrap()
                },
            }
        }).collect();

        for i in 0..nsamples {
            let mut var_samples: Vec<f64> = Vec::new();
            for mcresult in mcresults.iter() {
                var_samples.push(mcresult.samples[i]);
            }
            samples.push(expr.eval(&var_samples)?);
        }
        Ok(MonteCarloResult::new(samples, self.units.clone(), conf))
    }
}
impl Uncertainty for ModelFunction {
    fn symbol(&self) -> String {
        self.symbol.clone()
    }
    fn utility(&self) -> &Option<Utility> {
        &self.utility
    }
    fn interval(&self) -> &Option<Interval> {
        &self.interval
    }
    fn tolerance(&self) -> Option<Tolerance> {
        match &self.utility {
            Some(v) => Some(v.tolerance.clone()),
            None => None,
        }
    }
    fn costs(&self) -> &Option<Costs> {
        &self.cost
    }
    fn enditem(&self) -> bool {
        match self.costs() {
            Some(c) => {
                c.cost_fa > 0.0 || c.cost_fr > 0.0
            },
            None => false
        }
    }
    fn calibration(&self) -> &Option<Calibration> {
        &self.calibration
    }
}


fn prob_conform(uncert: &UncertResult, tolerance: &Tolerance) -> f64 {
    // Probability of Conformance
    match uncert {
        UncertResult::Gum(_) => {
            let mu = uncert.expected();
            let sigma = uncert.std_uncert();
            normal_cdf(tolerance.high, mu, sigma) - normal_cdf(tolerance.low, mu, sigma)
        },
        UncertResult::Montecarlo(v) => {
            let inside = v.samples.iter().filter(|&n| *n >= tolerance.low && *n <= tolerance.high).count();
            inside as f64 / v.samples.len() as f64
        },
    }
}
fn global_pfa(uncert: &UncertResult, true_eopr: f64, item: &impl Uncertainty) -> Option<RiskResultGlobal> {
    // Global/Average Probability of False Accept
    let tolerance = &item.tolerance().unwrap();
    let expected = uncert.expected();
    let product_pdf = dists::Distribution::from_itp(expected, true_eopr, &tolerance);
    match product_pdf.check() {
        Err(_) => None,
        Ok(_) => {
            let fyx = uncert.distribution();  // TypeBDist

            let guardband = match &item.utility() {
                Some(utility) => { 
                    utility.guardband.clone()
                },
                None => Guardband::default(),
            };
            let riskmodel = RiskModel{
                process: product_pdf,
                test: fyx,
                tolerance: tolerance.clone(),
                guardband: guardband.clone(),
            };
            let acceptance = riskmodel.get_guardband();
            let pfa = riskmodel.pfa(&acceptance);
            let pfr = riskmodel.pfr(&acceptance);
            let cpfa = riskmodel.cpfa(&acceptance);
            let tur = riskmodel.tur();
            Some(RiskResultGlobal{
                pfa_true: pfa,
                pfr_true: pfr,
                cpfa_true: cpfa,
                tur: tur,
                acceptance: acceptance,
            })
        },
    }
}

fn risk(uncert: &UncertResult, eopr: &Option<Eopr>, item: &impl Uncertainty) -> Option<RiskResult> {
    // Calculate Global or Specific risks
    match item.utility() {
        Some(utility) => {
            match eopr {
                Some(Eopr::True(v)) => { 
                    // Have an EOPR (assume nominal value), calculate PFA/PFR 
                    match global_pfa(uncert, *v, item) {
                        Some(p) => Some(RiskResult::Global(p)),
                        None => None,
                    }
                },
                _ => {
                    // No EOPR, assume measured value, calculate Prob. Conformance
                    Some(RiskResult::Specific(prob_conform(uncert, &utility.tolerance)))
                },
            }
        },
        None => None,
    }
}


fn true_eopr(uncert: &UncertResult, item: &impl Uncertainty) -> Option<Eopr> {
    // Calculate True EOPR from Observed EOPR
    match item.interval() {
        Some(v) => match v.eopr {
            Eopr::True(t) => Some(Eopr::True(t)),
            Eopr::Observed(obs) => {
                let nom = uncert.expected();
                let trueeopr = match item.utility() {
                    Some(utility) => {
                        let obsstdev = std_from_itp(obs, nom, &utility.tolerance);
                        let truestd = (obsstdev.powi(2) - uncert.std_uncert().powi(2)).sqrt();
                        itp_from_norm(nom, truestd, &utility.tolerance)
                    },
                    None => f64::NAN,
                };
                Some(Eopr::True(trueeopr))
            },
        },
        None => None,
    }
}


fn calc_reliability(uncert: &UncertResult, acceptance: Tolerance, eopr: f64, item: &impl Uncertainty, qty_results: &Vec<QuantityResult>) -> ReliabilityResult {
    // Calculate reliability data
    let qr = Some(qty_results);
    let tolerance = &item.tolerance().unwrap();
    let utility = match &item.utility() {
        Some(v) => v,
        None => panic!(),
    };
    let expected = uncert.expected();

    let fyx = uncert.distribution();  // TypeBDist

    // Pre-measurement stress
    let pdf_bt = match item.calibration() {
        Some(calib) => match &calib.prestress {
            Some(p) => dists::Distribution::from_itp(expected, eopr, &tolerance).convolve(&p, None),
            None => dists::Distribution::from_itp(expected, eopr, &tolerance),
        },
        None => dists::Distribution::from_itp(expected, eopr, &tolerance),
    };

    let pdf_t = pdf_bt.convolve(&fyx, qr);
    let accepted_x = fyx.integrate_given_y(acceptance.low, acceptance.high, qr);
    let p_accepted = pdf_t.integrate(acceptance.low, acceptance.high);
    let x_given_accepted = accepted_x.mul(&pdf_bt, p_accepted.recip());
    // let x_given_rejected = accepted_x.invert().mul(&pdf_bt, (1.0-p_accepted).recip());
    let obs_oot = 1.0 - pdf_bt.convolve(&fyx, qr).integrate(tolerance.low, tolerance.high);

    let mut p_repair = 0.0;
    let mut p_adjust = 0.0;

    let pdf_pt = match item.calibration() {
        None => { x_given_accepted },
        Some(calib) => {
            match &calib.policy {
                RenewalPolicy::Never => { x_given_accepted },
                _ => {  // Always or As-needed

                    let fy_ap = match &calib.mte_adjust {
                        Some(m) => m.pdf_given_y(expected, qr),
                        None => fyx.pdf_given_y(expected, qr),
                    };
                    let fy_rp = match &calib.mte_repair {
                        Some(m) => m.pdf_given_y(expected, qr),
                        None => fyx.pdf_given_y(expected, qr),
                    };

                    let repair_limit = match &calib.repair {
                        None => acceptance.clone(),
                        Some(v) => v.clone(),
                    };

                    // let x_adjust = &fy_rp;
                    p_repair = 1.0 - pdf_t.integrate(repair_limit.low, repair_limit.high);
                    // let notadjust_x = fyx.integrate_given_y(acceptance.low, acceptance.high, qr);
                    // let adjust_x = notadjust_x.invert();
                    p_adjust = 1.0 - pdf_t.integrate(acceptance.low, acceptance.high) - p_repair;
                    p_repair = p_repair * (1.0 - calib.prob_discard);

                    let p_notadjust = 1.0 - p_adjust;
                    // let notadjust_x = adjust_x.invert();
                    // let x_notadjust = notadjust_x.mul(&x_given_rejected, p_notadjust.recip());
                    let p_renewed = p_adjust + p_repair;
                    let x_renewed = fy_ap.sum(&fy_rp, p_adjust, p_repair, p_adjust+p_repair);

                    match &calib.policy {
                        RenewalPolicy::Always => {
                            fy_ap.sum(&fy_rp, p_adjust, p_notadjust * (1.0 - calib.prob_discard), 1.0)
                        },
                        RenewalPolicy::Asneeded => {
                            x_given_accepted.sum(&x_renewed, p_accepted, p_renewed, 1.0)
                        },
                        RenewalPolicy::Never => panic!(),
                    }
                }
            }
        }
    };

    // Post-measurement stress
    let pdf_bop = match item.calibration() {
        Some(calib) => match &calib.prestress {
            Some(p) => pdf_pt.convolve(&p, None),
            None => pdf_pt.clone(),
        },
        None => pdf_pt.clone(),
    };

    let p_bop = pdf_bop.integrate(tolerance.low, tolerance.high);
    let interval = item.interval().as_ref().unwrap();

    let decay = match &item.calibration() {
        Some(c) => {
            match &c.reliability_model {
                ReliabilityModel::None => None,
                _ => {
                    Some(ReliabilityDecay::new(
                        c.reliability_model.clone(),
                        p_bop,
                        eopr,
                        interval.years,
                        tolerance,
                        expected,
                    ))
                },
            }
        },
        None => None,
    };

    let p_aop = match &decay {
        Some(v) => v.p_aop,
        None => p_bop,
    };

    let sigma_aop = std_from_itp(p_aop, expected, &tolerance);
    let degrade = match &utility.degrade {
        Some(v) => v,
        None => tolerance,
    };
    let fail = match &utility.failure {
        Some(v) => v,
        None => tolerance,
    };

    let sig = dists::std_from_itp(eopr, expected, &tolerance);
    let mut utility_curve = dists::Distribution::Step{a: tolerance.low, b: tolerance.high};
    let pdf_eop = dists::Distribution::Normal{mu: expected, sigma: sig};
    let mut p_success = f64::NAN;

    if item.enditem() {
        utility_curve = dists::Distribution::cosine(degrade, fail);
        p_success = match &decay {
            None => {
                utility_curve.mul(&pdf_eop, utility.psr).integrate(-f64::INFINITY, f64::INFINITY)
            },
            Some(d) => {
                d.success(&utility_curve, utility.psr)
            },
        };
    }

    ReliabilityResult{
        _fyx: fyx,
        _pdf_bt: pdf_bt,
        _pdf_t: pdf_t,
        _pdf_pt: pdf_pt,
        pdf_bop: pdf_bop,
        pdf_eop: pdf_eop,
        _obs_oot: obs_oot,
        decay: decay,
        p_bop: p_bop,
        p_aop: p_aop,
        sigma_aop: sigma_aop,
        p_repair: p_repair,
        p_adjust: p_adjust,
        p_success: p_success,
        utility: utility_curve,
    }
}


impl ReliabilityDecay {
    fn new(model: ReliabilityModel, p_bop: f64, p_eop: f64,
           interval: f64, tolerance: &Tolerance, expected: f64) -> ReliabilityDecay {
        match model {
            ReliabilityModel::Exponential => {
                let lambda = -interval.recip() * (p_eop/p_bop).ln();
                let p_aop = p_bop / lambda / interval * (1.0 - (-lambda*interval).exp());
                let sig_aop = dists::std_from_itp(p_aop, expected, &tolerance);
                ReliabilityDecay{
                    expected: expected,
                    tolerance: tolerance.clone(),
                    interval: interval,
                    eopr: p_eop,
                    p_bop: p_bop,
                    p_aop: p_aop,
                    decay: ReliabilityDecayParameters::Exponential{lambda: lambda, sig_aop: sig_aop},
                }
            },
            ReliabilityModel::RandomWalk => {
                let sig_bop = dists::std_from_itp(p_bop, expected, tolerance);
                let sig_eop = dists::std_from_itp(p_eop, expected, tolerance);
                let alpha = (sig_eop.powi(2) - sig_bop.powi(2)) / interval;
                let sig_aop = (sig_bop.powi(2) + alpha*interval/2.0).sqrt();
                let p_aop = itp_from_norm(expected, sig_aop, tolerance);

                ReliabilityDecay{
                    expected: expected,
                    tolerance: tolerance.clone(),
                    interval: interval,
                    eopr: p_eop,
                    p_bop: p_bop,
                    p_aop: p_aop,
                    decay: ReliabilityDecayParameters::RandomWalk{
                        sig_bop: sig_bop,
                        sig_aop: sig_aop,
                        sig_eop: sig_eop,
                        alpha: alpha},
                }
            },
            ReliabilityModel::None => { unreachable!(); },
        }
    }
    fn bop_pdf(&self) -> dists::Distribution {
        // BOP reliability as defined by p_bop, not predicted on decay curve
        dists::Distribution::from_itp(self.expected, self.p_bop, &self.tolerance)
    }
    // fn aop_pdf(&self) -> dists::Distribution {
    //     // Calculate AOP Reliability
    //     match self.decay {
    //         ReliabilityDecayParameters::Exponential{lambda: _, sig_aop} => {
    //             dists::Distribution::Normal{mu:self.expected, sigma: sig_aop}
    //         },
    //         ReliabilityDecayParameters::RandomWalk{sig_bop: _, sig_aop, sig_eop: _, alpha: _} => {
    //             dists::Distribution::Normal{mu: self.expected, sigma: sig_aop}
    //         }
    //     }
    // }
    fn pdf_time(&self, t: f64) -> dists::Distribution {
        // Get PDF at time t in interval
        match self.decay {
            ReliabilityDecayParameters::Exponential{lambda, sig_aop: _} => {
                let reliability = self.p_bop * (-lambda * t).exp();
                let sig = dists::std_from_itp(reliability, self.expected, &self.tolerance);
                dists::Distribution::Normal{mu: self.expected, sigma: sig}
            },
            ReliabilityDecayParameters::RandomWalk{sig_bop, sig_aop: _, sig_eop: _, alpha} => {
                let sig = (sig_bop.powi(2) + alpha * t).sqrt();
                dists::Distribution::Normal{mu: self.expected, sigma: sig}
            },
        }
    }
    pub fn reliability_time(&self, t: f64) -> f64 {
        // Get reliability at time t
        match self.decay {
            ReliabilityDecayParameters::Exponential{lambda, sig_aop: _} => {
                self.p_bop * (-lambda * t).exp()
            },
            ReliabilityDecayParameters::RandomWalk{sig_bop, sig_aop: _, sig_eop: _, alpha} => {
                let sig = (sig_bop.powi(2) + alpha * t).sqrt();
                itp_from_norm(self.expected, sig, &self.tolerance)
            },
        }
    }
    fn success(&self, utility: &dists::Distribution, psr: f64) -> f64 {
        // Probability of success over the interval
        let times = dists::linspace(0.0, self.interval, 50);
        let dt = times[1]-times[0];
        let mut success_t = Vec::<f64>::with_capacity(50);
        for t in times {
            let pdf = if t == 0.0 { self.bop_pdf() } else { self.pdf_time(t) };
            success_t.push(
                utility.mul(&pdf, psr).integrate(-f64::INFINITY, f64::INFINITY).min(1.0)
            );
        }
        dists::trapz_integral(&success_t, dt) / self.interval
    }
    fn find_target(&self, target: IntervalTarget) -> (f64, f64) {
        match self.decay {
            ReliabilityDecayParameters::Exponential{lambda, sig_aop: _} => {
                match target {
                    IntervalTarget::Interval(i) => {
                        let eop = self.p_bop * (-lambda * i).exp();
                        (i, eop)
                    },
                    IntervalTarget::Eopr(eopr) => {
                        let t = (eopr/self.p_bop).ln() / -lambda;
                        (t, eopr)
                    }
                }
            },
            ReliabilityDecayParameters::RandomWalk{sig_bop, sig_aop: _, alpha, sig_eop: _} => {
                match target {
                    IntervalTarget::Interval(i) => {
                        let sigma = (sig_bop.powi(2) - alpha * i).sqrt();
                        let eop = dists::itp_from_norm(self.expected, sigma, &self.tolerance);
                        (i, eop)
                    },
                    IntervalTarget::Eopr(eopr) => {
                        let sigma = dists::std_from_itp(eopr, self.expected, &self.tolerance);
                        let t = (sigma.powi(2) - sig_bop.powi(2)) / alpha;
                        (t, eopr)
                    }
                }
            }
        }
    }
}


fn reliability(uncert: &UncertResult, riskresult: &Option<RiskResult>, eopr: f64, item: &impl Uncertainty, qty_results: &Vec<QuantityResult>) -> Option<ReliabilityResult> {
    // Calculate reliability data
    let acceptance = match riskresult {
        Some(r) => {
            match r {
                RiskResult::Global(v) => v.acceptance.clone(),
                _ => item.tolerance().unwrap().clone(),
            }
        }
        _ => item.tolerance().unwrap().clone(),
    };

    match item.utility() {
        Some(_) => match item.interval() {
            Some(_) => {
                Some(calc_reliability(uncert, acceptance, eopr, item, qty_results))
            },
            None => None,
        },
        None => None,
    }
}

fn calc_costs(risk: &RiskResultGlobal, rel: &ReliabilityResult, item: &impl Uncertainty) -> Option<CostResult> {
    // Calculate cost model
    match item.costs() {
        Some(cost) => {
            let fa = cost.cost_fa * risk.pfa_true;
            let fr = cost.cost_fr * risk.pfr_true;
            let downtime = cost.down_cal 
                + cost.down_adj * rel.p_adjust
                + cost.down_rep * rel.p_repair;

            let interval_years = match &rel.decay { Some(d) => d.interval, None => 1.0, };
            let interval_days = interval_years * 365.25;
            let p_available = (1.0 + downtime / interval_days).recip();
            let num_spares = (downtime / interval_days) * cost.num_uuts * cost.spare_factor;
            let spare_cost = num_spares * cost.new_uut;
            let spares_year = cost.spare_startup * spare_cost;

            let dut_per_year = (cost.num_uuts + num_spares) / interval_years;
            let ccal = cost.cal * dut_per_year;
            let cadj = cost.adjust * dut_per_year * rel.p_adjust;
            let crep = cost.repair * dut_per_year * rel.p_repair;
            let support = ccal + cadj + crep;

            let performance = if cost.cost_fa > 0.0 && rel.p_success.is_finite() {
                cost.cost_fa * cost.num_uuts * cost.p_use * (1.0 - rel.p_success)
            } else {
                0.0
            };
            let total = support + performance;

            Some(CostResult{
                _expected: fa+fr,
                _p_available: p_available,
                _num_spares: num_spares,
                _spares_year: spares_year,
                spare_cost: spare_cost,
                calibration: ccal,
                adjustment: cadj,
                repair: crep,
                support: support,
                performance: performance,
                total: total,
            })
        },
        None => None,
    }
}


#[derive(Debug, Clone)]
pub struct UndefinedCorrelationError;
impl Error for UndefinedCorrelationError {}
impl fmt::Display for UndefinedCorrelationError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Undefined variable in correlations")
    }
}


fn calc_item(
        item: &impl Uncertainty,
        uncert_result: UncertResult,
        qty_results: &Vec<QuantityResult>,
        ) -> Result<QuantityResult, Box<dyn Error>> {

    // Calculate eopr/reliability on the observed/historic reliability decay curve
    let mut eopr_result = true_eopr(&uncert_result, item);
    let mut risk_result = risk(&uncert_result, &eopr_result, item);
    let mut interval = match item.interval() {
        Some(i) => i.years,
        _ => f64::NAN,
    };

    let reliability_result = match (&eopr_result, &risk_result) {
        (Some(Eopr::True(v)), Some(_)) => {
            let mut rel = reliability(&uncert_result, &risk_result, *v, item, qty_results);
            if let Some(ref r) = rel {
                let interval_target = match item.interval() {
                    Some(i) => i.target.clone(),
                    _ => None,
                };
                if let Some(target) = interval_target {
                    if let Some(d) = &r.decay {
                        let (test_interval, test_eop) = d.find_target(target);
                        eopr_result = Some(Eopr::True(test_eop));
                        risk_result = risk(&uncert_result, &eopr_result, item);
                        rel = reliability(&uncert_result, &risk_result, test_eop, item, qty_results);
                        interval = test_interval;
                    }
                }
            }
        rel
        },
        _ => None,
    };

    let cost_result = match (&risk_result, &reliability_result) {
        (Some(RiskResult::Global(riskres)), Some(relres)) => {
            calc_costs(&riskres, &relres, item)
        },
        _ => None,
    };

    Ok(QuantityResult{
        symbol: item.symbol(),
        uncertainty: uncert_result,
        eopr: eopr_result,
        interval: interval,
        risk: risk_result,
        reliability: reliability_result,
        cost: cost_result,
    })
}


impl MeasureSystem {
    pub fn load_toml(config: &str) -> Result<MeasureSystem, Box<dyn std::error::Error>> {
        let model = toml::from_str::<MeasureSystem>(&config)?;
        Ok(model)
    }
    pub fn get_config(&self) -> Result<String, toml::ser::Error> {
        toml::to_string(self)
    }
    pub fn calc_gum(&self) -> Result<Vec<QuantityResult>, Box<dyn Error>> {
        // Calculate direct quantities first
        let mut qty_results: Vec<QuantityResult> = vec![];
        for qty in self.quantity.iter() {
            let uncert_result = qty.evaluate(self.settings.confidence, &qty_results)?;
            qty_results.push(calc_item(qty, uncert_result, &qty_results)?);
        }

        // Then curves
        let curve_results = self.calc_curve()?;
        for c in curve_results.iter() {
            qty_results.extend(c.qtyresult());
        };

        // Now indirect quantities
        let nfunc = self.function.len();
        let funcnames: Vec<String> = self.function.iter().map(|f| f.symbol.clone()).collect();
        if nfunc > 0 {
            let mut varnames: Vec<String> = vec![];
            let mut exprs: Vec<FlatEx<f64>> = Vec::with_capacity(nfunc);
            for func in self.function.iter() {
                let expr_float = units::parse_expr_f64(&func.expr)?;
                varnames.extend(expr_float.var_names().iter().cloned());
                exprs.push(expr_float);
            }
            varnames.unique();

            // Get values/uncertainties for each quantity
            let mut var_expects: Vec<units::BaseQuantity> = Vec::new();
            let mut var_variances: Vec<f64> = Vec::new();
            let mut var_degfs: Vec<f64> = Vec::new();
            for name in varnames.iter() {
                let qresult = get_qresult(name, Some(&qty_results)).ok_or(UndefinedQuantityError{msg:format!("Undefined quantity {}", name)})?;
                match &qresult.uncertainty {
                    UncertResult::Gum(v) => {
                        var_expects.push(units::make_baseqty(v.expected, v.units.clone(), true)?);
                        var_variances.push((units::make_baseqty(v.std_dev, v.units.clone(), false)?).magnitude.powi(2));
                        var_degfs.push(v.degrees_freedom);
                    },
                    UncertResult::Montecarlo(_) => panic!(),
                }
            }
            let var_values: Vec<f64> = var_expects.iter().map(|v| v.magnitude).collect();
            let var_dimensions: Vec<units::Dimension> = var_expects.iter().map(|v| v.dim.clone()).collect();
            let var_uncerts: Vec<f64> = var_variances.iter().map(|v| v.sqrt()).collect();
            let nvars = varnames.len();

            // Covariance Matrix
            let mut covariance = General::<f64>::from(Diagonal::<f64>::new(&var_variances));
            for cc in &self.correlation {
                let idx1 = varnames.iter().position(|r| *r == cc.v1).unwrap_or(0);
                let idx2 = varnames.iter().position(|r| *r == cc.v2).unwrap_or(0);
                let sub = General::<f64>::new(1, 1, vec![cc.coeff * var_variances[idx1].sqrt() * var_variances[idx2].sqrt()]);
                covariance = covariance.set_slice(&sub, idx1, idx2);
                covariance = covariance.set_slice(&sub, idx2, idx1);
            }

            // Sensitivity Matrix
            let mut cx = General::<f64>::zero(self.function.len(), nvars);
            let mut func_expect: Vec<f64> = Vec::with_capacity(nfunc);
            let mut partials: Vec<Vec<String>> = Vec::new();
            for i in 0..nfunc {
                let mut fpartials: Vec<String> = Vec::new();
                let fvarnames = exprs[i].var_names();
                let mut fvar_values: Vec<f64> = Vec::new();
                let mut fvar_indexes: Vec<usize> = Vec::new();
                for (j, name) in varnames.iter().enumerate() {
                    if fvarnames.iter().any(|x| *x==*name) {
                        fvar_values.push(var_values[j]);
                        fvar_indexes.push(j);
                    }
                }
                func_expect.push(exprs[i].clone().eval(&fvar_values)?);
                for j in 0..fvar_values.len() {
                    let diff = exprs[i].clone().partial(j)?;
                    let ci = diff.eval(&fvar_values)?;
                    let sub = General::<f64>::new(1, 1, vec![ci]);
                    cx = cx.set_slice(&sub, i, fvar_indexes[j]);
                    //fpartials.push(diff.unparse().to_string());
                    fpartials.push(
                        format!("d{}/d{} = {}", funcnames[i], fvarnames[j], diff.unparse().to_string())
                    );
                }
                partials.push(fpartials);
            }

            // Uncertainties are sqrt() of diagonal of Uy
            let uy = cx.clone() * covariance.clone() * cx.clone().transpose();
            let func_uncerts: Vec<f64> = (0..nfunc).map(|i| uy[[i, i]].sqrt()).collect();

            // Effective Deg. Freedom
            let mut func_degfs: Vec<f64> = Vec::<f64>::with_capacity(nfunc);
            for i in 0..nfunc {
                let mut denom: f64 = 0.0;
                for j in 0..nvars {
                    if var_degfs[i].is_finite() {
                        denom += (var_uncerts[j]*cx[[i,j]]).powi(4) / var_degfs[i];
                    }
                }
                func_degfs.push(func_uncerts[i].powi(4) / denom);
            }
            let cov_factors: Vec<f64> = func_degfs.iter().map(|v| student::t_inv2t(self.settings.confidence, *v).unwrap_or(f64::INFINITY)).collect();

            // Convert to desired units
            let mut out_val: Vec<f64> = Vec::with_capacity(nfunc);
            let mut out_unc: Vec<f64> = Vec::with_capacity(nfunc);
            for i in 0..nfunc {
                let expr_dim = units::parse_expr_dimension(&self.function[i].expr)?;
                let fvarnames = expr_dim.var_names();
                let mut fvar_dimensions: Vec<units::Dimension> = Vec::new();
                for (j, name) in varnames.iter().enumerate() {
                    if fvarnames.iter().any(|x| *x==*name) {
                        fvar_dimensions.push(var_dimensions[j].clone());
                    }
                }
                let out_dimension = expr_dim.eval(&fvar_dimensions)?;
                let (val, unc) = match &self.function[i].units {
                    Some(u) => {
                        let expected = units::convert_base(func_expect[i], &out_dimension, &u, true)?;
                        let stdev = units::convert_base(func_uncerts[i], &out_dimension, &u, false)?;
                        (expected, stdev)
                    },
                    None => (func_expect[i], func_uncerts[i]),
                };
                out_val.push(val);
                out_unc.push(unc);
            }

            // Build results into qty_results
            for i in 0..nfunc {
                let uncert_result = UncertResult::Gum(
                    GumResult{
                        expected: out_val[i],
                        variance: out_unc[i].powi(2),
                        std_dev: out_unc[i],
                        degrees_freedom: func_degfs[i],
                        units: self.function[i].units.clone(),
                        gum: Some(GumComponents{
                            varnames: varnames.clone(),
                            funcnames: funcnames.clone(),
                            ux: covariance.clone(),
                            cx: cx.clone(),
                            uy: uy.clone(),
                            partial_eqs: partials.clone(),
                        }),
                        curve: None,
                        coverage_factor: cov_factors[i],
                        confidence: self.settings.confidence,
                        mcsamples: None,
                    }
                );
                qty_results.push(calc_item(&self.function[i], uncert_result, &qty_results)?);
            }
        }
        Ok(qty_results)
    }
    pub fn calc_monte(&self) -> Result<Vec<QuantityResult>, Box<dyn Error>> {
        // Monte Carlo
        let mut mc_results: Vec<QuantityResult> = vec![];
        if self.settings.montecarlo > 1 {
            let varnames = self.quantity.iter().map(|q| {q.symbol.clone()}).collect();
            let samples = if self.correlation.len() > 0 {
                dists::multivariate_random(
                    self.settings.montecarlo,
                    varnames,
                    &self.correlation
                )
            } else {
                dists::random(self.settings.montecarlo, varnames.len())
            };

            // Sample quantities
            for (i, qty) in self.quantity.iter().enumerate() {
                let uncert_result = qty.evaluate_mc(&samples.get_column(i).convert_to_vec(), self.settings.confidence)?;
                mc_results.push(
                    calc_item(qty, uncert_result, &mc_results)?
                );
            }
            // Calc curves analytically
            let curve_results = self.calc_curve()?;
            for c in curve_results.iter() {
                for mut r in c.qtyresult() {
                    match r.uncertainty {
                        UncertResult::Gum(ref mut v) => {
                            v.sample(self.settings.montecarlo);
                        },
                        _ => panic!(),
                    }
                    mc_results.push(r);
                }

            };
            // Then calc functions
            for func in self.function.iter() {
                let uncert_result = UncertResult::Montecarlo(func.calc_monte(&mc_results, self.settings.montecarlo, self.settings.confidence)?);
                mc_results.push(
                    calc_item(func, uncert_result, &mc_results)?
                );
            }
        }
        Ok(mc_results)
    }
    pub fn calc_curve(&self) -> Result<Vec<CurveResult>, Box<dyn Error>> {
        let mut curve_results: Vec<CurveResult> = vec![];
        for qty in self.curve.iter() {

            let x = Vector::new_row(qty.x.clone());
            let y = Vector::new_row(qty.y.clone());
            let mut uy: Option<Vector<f64>> = None;
            if let Some(u) = &qty.uy {
                uy = Some(Vector::new_row(u.clone()));
            };
            
            let guess = match &qty.guess {
                Some(g) => Vector::new_row(g.clone()),
                None => vector![1.0, 1.0, 1.0, 1.0],
            };
            let result = match qty.model {
                CurveModel::Line => curve_fit(&x, &y, uy, &vector![1.0, 1.0], &LineFit{}),
                CurveModel::Quadratic => curve_fit(&x, &y, uy, &&vector![1.0, 1.0, 1.0], &QuadFit{}),
                CurveModel::Cubic => curve_fit(&x, &y, uy, &&vector![1.0, 1.0, 1.0, 1.0], &CubicFit{}),
                CurveModel::Exponential => curve_fit(&x, &y, uy, &guess, &ExponentialFit{}),
                CurveModel::DampedSine => curve_fit(&x, &y, uy, &guess, &DampedSineFit{}),
            };
            curve_results.push(result);
        }
        Ok(curve_results)
    }
    pub fn calculate(&self) -> Result<SystemResult, Box<dyn Error>> {
        let qty_results = self.calc_gum()?;
        let mc_results = self.calc_monte()?;
        Ok(SystemResult{
            quantities: qty_results,
            montecarlo: mc_results,
            system: self})
    }
}
