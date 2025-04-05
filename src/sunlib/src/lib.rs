// Suncal Library Functions
use std::{error::Error, fmt};
use std::collections::HashMap;
use std::iter::zip;
use exmex::prelude::*;

mod student;
mod stats;
pub mod dists;
pub mod cfg;
pub mod result;
pub mod risk;
use crate::dists::{normal_cdf, std_from_itp, itp_from_norm};
use crate::cfg::{ModelQuantity, ModelFunction, MeasureSystem, CorrelationCoeff, Tolerance, Utility, Costs, Eopr, Guardband, RenewalPolicy, ReliabilityModel, Interval, IntervalTarget, Calibration};
use crate::result::{UncertResult, QuantityResult, GumResult, MonteCarloResult, RiskResult, RiskResultGlobal, ReliabilityResult, CostResult, SystemResult, ReliabilityDecay, ReliabilityDecayParameters};
use crate::risk::RiskModel;


// Quantities, Functions, and eventually Curves implement this Trait
trait Uncertainty {
    fn utility(&self) -> &Option<Utility>;
    fn interval(&self) -> &Option<Interval>;
    fn costs(&self) -> &Option<Costs>;
    fn tolerance(&self) -> Option<Tolerance>;
    fn calibration(&self) -> &Option<Calibration>;
    fn enditem(&self) -> bool;
    fn evaluate(&self, qty_results: &HashMap<String, QuantityResult>, correlation: &Vec<CorrelationCoeff>) -> Result<UncertResult, Box<dyn Error>>;
    fn evaluate_mc(&self, qty_results: &HashMap<String, QuantityResult>, correlation: &Vec<CorrelationCoeff>, nsamples: usize) -> Result<UncertResult, Box<dyn Error>>;
}


impl ModelQuantity {
    fn expected(&self) -> f64 {
        // Expectation
        self.measured
        // TODO - use R/R mean if defined
    }
    fn degrees_freedom(&self, variance: f64, qty_results: &HashMap<String, QuantityResult>) -> f64 {
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
            // TODO - #new measurements??
            Some(v) => {stats::variance(v).powi(2) / (v.len() as f64 - 1.0)},
            None => 0.0,
        };
        // TODO - Reproducibility
        variance*variance / denom
    }
    fn variance(&self, qty_results: &HashMap<String, QuantityResult>) -> f64 {
        // Standard variance RSSing components
        // combine typebs and R/R
        let mut variance: f64 = 0.0;
        for typeb in self.typeb.iter() {
            variance += typeb.variance(qty_results)
        }

        // Repeatability
        variance += match &self.repeatability {
            Some(v) => stats::variance(v) / v.len() as f64,
            None => 0.0,
            // TODO - autocorrelation if n > some threshold
        };

        // TODO - Reproducibility
        variance
    }
    fn sample(&self, qty_results: &HashMap<String, QuantityResult>) -> f64 {
        // Random sample, combining all uncertainty components
        let mut sample = self.measured;
        for typeb in self.typeb.iter() {
            sample += typeb.sample(qty_results);
        }
        // TODO - R&R
        sample
    }
}
impl Uncertainty for ModelQuantity {
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
    fn evaluate(&self,
        qty_results: &HashMap<String, QuantityResult>,
        _correlation: &Vec<CorrelationCoeff>,
    ) -> Result<UncertResult, Box<dyn Error>> {
        let variance = self.variance(qty_results);
        let gum = GumResult{
            expected: self.expected(),
            variance: variance,
            std_dev: variance.sqrt(),
            degrees_freedom: self.degrees_freedom(variance, qty_results),
            ..Default::default()
        };
        Ok(UncertResult::Gum(gum))
    }
    fn evaluate_mc(&self, qty_results: &HashMap<String, QuantityResult>, _correlation: &Vec<CorrelationCoeff>, nsamples: usize) -> Result<UncertResult, Box<dyn Error>> {
        // TODO - handle correlated
        let mut samples = Vec::<f64>::with_capacity(nsamples);
        for _i in 0..nsamples {
            samples.push(self.sample(qty_results));
        }
        Ok(UncertResult::Montecarlo(MonteCarloResult::new(samples)))
    }
}


impl ModelFunction {
    fn calc_gum(&self,
            qty_results: &HashMap<String, QuantityResult>,
            correlation: &Vec<CorrelationCoeff>) -> Result<GumResult, Box<dyn Error>> {
        let expr = exmex::parse::<f64>(&self.expr)?;
        let varnames = expr.var_names();

        // In order
        let mut expects: Vec<f64> = Vec::new();
        let mut variances: Vec<f64> = Vec::new();
        let mut sensitivities: Vec<f64> = Vec::new();
        let mut degfs: Vec<f64> = Vec::new();

        for name in varnames.iter() {
            // TODO - catch panic here if quantity is not defined
            // Return Err and maybe come back later
            match &qty_results[name].uncertainty {
                UncertResult::Gum(v) => {
                    expects.push(v.expected);
                    variances.push(v.variance);
                    degfs.push(v.degrees_freedom);
                },
                UncertResult::Montecarlo(_) => panic!(),
            }
        }

        let mut variance = 0.0;
        for (idx, _name) in varnames.iter().enumerate() {
            let diff = expr.clone().partial(idx)?;
            let ci = diff.eval(&expects)?;
            sensitivities.push(ci);
            variance += ci.powi(2) * variances[idx];
        }

        // Covaraince Terms
        for corr in correlation {
            let idx1 = varnames.iter().position(|r| *r == corr.v1).ok_or(UndefinedCorrelationError)?;
            let idx2 = varnames.iter().position(|r| *r == corr.v2).ok_or(UndefinedCorrelationError)?;
            let ci = sensitivities[idx1];
            let cj = sensitivities[idx2];
            variance += 2.0 * ci * cj * corr.coeff * variances[idx1].sqrt() * variances[idx2].sqrt();
        }

        // Deg. Freedom (Welch Satterthwaite)
        let mut degf_denom = 0.0;
        for (i, var) in variances.iter().enumerate() {
            degf_denom += var.powi(2) * sensitivities[i].powi(4) / degfs[i];
        }
        let degrees_freedom = variance.powi(2) / degf_denom;
        // TODO - configurable confidence
        let coverage = student::t_inv2t(0.95, degrees_freedom)?;
        let expected = expr.eval(&expects)?;

        Ok(GumResult{
            expected: expected,
            variance: variance,
            std_dev: variance.sqrt(),
            degrees_freedom: degrees_freedom,
            _sensitivities: zip(varnames, sensitivities).map(|(key, value)| {(key.clone(), value)}).collect::<HashMap<String, f64>>(),
            coverage_factor: coverage,
            confidence: 0.95,
        })
    }
    fn calc_monte(&self,
                   qty_results: &HashMap<String, QuantityResult>,
                   nsamples: usize) -> Result<MonteCarloResult, Box<dyn Error>> {
        let expr = exmex::parse::<f64>(&self.expr)?;
        let varnames = expr.var_names();
        let mut samples = Vec::<f64>::with_capacity(nsamples);

        for i in 0..nsamples {
            let mut var_samples: Vec<f64> = Vec::new();
            for name in varnames.iter() {
                match &qty_results[name].uncertainty {
                    UncertResult::Montecarlo(v) => var_samples.push(v.get_sample(i)),
                    UncertResult::Gum(_) => panic!(),
                }
                // TODO - catch panic here if quantity is not defined
            }
            samples.push(expr.eval(&var_samples)?);
        }
        Ok(MonteCarloResult::new(samples))
    }
}
impl Uncertainty for ModelFunction {
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
    fn evaluate(&self, qty_results: &HashMap<String, QuantityResult>, correlation: &Vec<CorrelationCoeff>) -> Result<UncertResult, Box<dyn Error>> {
        let gum = self.calc_gum(qty_results, &correlation)?;
        Ok(UncertResult::Gum(gum))
    }
    fn evaluate_mc(&self, qty_results: &HashMap<String, QuantityResult>, _correlation: &Vec<CorrelationCoeff>, nsamples: usize) -> Result<UncertResult, Box<dyn Error>> {
        let monte = self.calc_monte(qty_results, nsamples)?;
        Ok(UncertResult::Montecarlo(monte))
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
fn global_pfa(uncert: &UncertResult, true_eopr: f64, item: &impl Uncertainty) -> RiskResultGlobal {
    // Global/Average Probability of False Accept
    let tolerance = &item.tolerance().unwrap();
    let expected = uncert.expected();
    let product_pdf = dists::Distribution::from_itp(expected, true_eopr, &tolerance);
    let fyx = uncert.distribution();  // TypeBDist

    let guardband = match &item.utility() {
        Some(utility) => { 
            match &utility.guardband {
                Some(g) => g.clone(),
                None => Guardband::default(),
            }
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
    RiskResultGlobal{
        pfa_true: pfa,
        pfr_true: pfr,
        cpfa_true: cpfa,
        acceptance: acceptance,
    }
}

fn risk(uncert: &UncertResult, eopr: &Option<Eopr>, item: &impl Uncertainty) -> Option<RiskResult> {
    // Calculate Global or Specific risks
    match item.utility() {
        Some(utility) => {
            match eopr {
                Some(Eopr::True(v)) => { 
                    // Have an EOPR (assume nominal value), calculate PFA/PFR 
                    Some(RiskResult::Global(
                        global_pfa(uncert, *v, item)
                    ))
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


fn calc_reliability(uncert: &UncertResult, acceptance: Tolerance, eopr: f64, item: &impl Uncertainty, qty_results: &HashMap<String, QuantityResult>) -> ReliabilityResult {
    // Calculate reliability data
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
            Some(p) => dists::Distribution::from_itp(expected, eopr, &tolerance).convolve(&p, &HashMap::new()),
            None => dists::Distribution::from_itp(expected, eopr, &tolerance),
        },
        None => dists::Distribution::from_itp(expected, eopr, &tolerance),
    };

    let pdf_t = pdf_bt.convolve(&fyx, qty_results);
    let accepted_x = fyx.integrate_given_y(acceptance.low, acceptance.high, qty_results);
    let p_accepted = pdf_t.integrate(acceptance.low, acceptance.high);
    let x_given_accepted = accepted_x.mul(&pdf_bt, p_accepted.recip());
    // let x_given_rejected = accepted_x.invert().mul(&pdf_bt, (1.0-p_accepted).recip());
    let obs_oot = 1.0 - pdf_bt.convolve(&fyx, qty_results).integrate(tolerance.low, tolerance.high);

    let mut p_repair = 0.0;
    let mut p_adjust = 0.0;

    let pdf_pt = match item.calibration() {
        None => { x_given_accepted },
        Some(calib) => {
            match &calib.policy {
                RenewalPolicy::Never => { x_given_accepted },
                _ => {  // Always or As-needed

                    let fy_ap = match &calib.mte_adjust {
                        Some(m) => m.pdf_given_y(expected, qty_results),
                        None => fyx.pdf_given_y(expected, qty_results),
                    };
                    let fy_rp = match &calib.mte_repair {
                        Some(m) => m.pdf_given_y(expected, qty_results),
                        None => fyx.pdf_given_y(expected, qty_results),
                    };

                    let repair_limit = match &calib.repair {
                        None => acceptance.clone(),
                        Some(v) => v.clone(),
                    };

                    // let x_adjust = &fy_rp;
                    p_repair = 1.0 - pdf_t.integrate(repair_limit.low, repair_limit.high);
                    // let notadjust_x = fyx.integrate_given_y(acceptance.low, acceptance.high, qty_results);
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
            Some(p) => pdf_pt.convolve(&p, &HashMap::new()),
            None => pdf_pt.clone(),
        },
        None => pdf_pt.clone(),
    };

    let p_bop = pdf_bop.integrate(tolerance.low, tolerance.high);
    let interval = item.interval().as_ref().unwrap();

    let decay = match &item.calibration() {
        Some(c) => {
            match &c.reliability_model {
                Some(model) => {
                    Some(ReliabilityDecay::new(
                        model.clone(),
                        p_bop,
                        eopr,
                        interval.years,
                        tolerance,
                        expected,
                    ))
                },
                None => None,
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

    let mut p_success = f64::NAN;
    if item.enditem() {
        let utility_curve = dists::Distribution::cosine(degrade, fail);
        p_success = match &decay {
            None => {
                let sig = dists::std_from_itp(eopr, expected, &tolerance);
                let eopr_pdf = dists::Distribution::Normal{mu: expected, sigma: sig};
                utility_curve.mul(&eopr_pdf, utility.psr).integrate(-f64::INFINITY, f64::INFINITY)
            },
            Some(d) => {
                d.success(utility_curve, utility.psr)
            },
        };
    }

    ReliabilityResult{
        _fyx: fyx,
        _pdf_bt: pdf_bt,
        _pdf_t: pdf_t,
        _pdf_pt: pdf_pt,
        _pdf_bop: pdf_bop,
        _obs_oot: obs_oot,
        decay: decay,
        p_bop: p_bop,
        p_aop: p_aop,
        sigma_aop: sigma_aop,
        p_repair: p_repair,
        p_adjust: p_adjust,
        p_success: p_success,
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
                let alpha = (sig_eop.powi(2) + sig_bop.powi(2)) / interval;
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
    fn success(&self, utility: dists::Distribution, psr: f64) -> f64 {
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


fn reliability(uncert: &UncertResult, riskresult: &Option<RiskResult>, eopr: f64, item: &impl Uncertainty, qty_results: &HashMap<String, QuantityResult>) -> Option<ReliabilityResult> {
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

            let interval_years = match &rel.decay { Some(d) => d.interval, None => unreachable!(),};
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
        mcsamples: usize,
        qty_results: &HashMap<String, QuantityResult>,
        correlation: &Vec<CorrelationCoeff>) -> Result<QuantityResult, Box<dyn Error>> {
    // Calculate the quantity - whether Direct or Indirect
    let uncert_result = match mcsamples {
        0 => item.evaluate(&qty_results, &correlation)?,
        _ => item.evaluate_mc(&qty_results, &correlation, mcsamples)?
    };

    // Calculate eopr/reliability on the observed/historic reliability decay curve
    let mut eopr_result = true_eopr(&uncert_result, item);
    let mut risk_result = risk(&uncert_result, &eopr_result, item);
    let mut interval = match item.interval() {
        Some(i) => i.years,
        _ => f64::NAN,
    };

    let reliability_result = match eopr_result {
        Some(Eopr::True(v)) => {
            let mut rel = reliability(&uncert_result, &risk_result, v, item, qty_results);
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
        uncertainty: uncert_result,
        eopr: eopr_result,
        interval: interval,
        risk: risk_result,
        reliability: reliability_result,
        cost: cost_result,
    })
}


pub fn calculate(model: &MeasureSystem) -> Result<SystemResult, Box<dyn Error>> {
    // Calculate all quantities, GUM and Monte Carlo
    let mut qty_results: HashMap<String, QuantityResult> = HashMap::new();
    for qty in model.quantity.iter() {
        qty_results.insert(
            qty.symbol.clone(),
            calc_item(qty, 0, &qty_results, &model.correlation)?
        );
    }
    for func in model.function.iter() {
        qty_results.insert(
            func.symbol.clone(),
            calc_item(func, 0, &qty_results, &model.correlation)?
        );
    }

    // MC
    let mut mc_results: HashMap<String, QuantityResult> = HashMap::new();
    if model.settings.montecarlo > 1 {
        for qty in model.quantity.iter() {
            mc_results.insert(
                qty.symbol.clone(),
                calc_item(qty, model.settings.montecarlo, &mc_results, &model.correlation)?
            );
        }
        for func in model.function.iter() {
            mc_results.insert(
                func.symbol.clone(),
                calc_item(func, model.settings.montecarlo, &mc_results, &model.correlation)?
            );
        }
    }
    Ok(SystemResult{
        quantities: qty_results,
        montecarlo: mc_results,
        system: model})
}
