// Structures for calculation results
use std::collections::HashMap;

use crate::cfg::{MeasureSystem, Eopr, Tolerance, TypeBNormal, TypeBDist};
use crate::stats;
use crate::dists;


#[derive(Debug)]
pub struct ReliabilityDecay {
    pub expected: f64,
    pub tolerance: Tolerance,
    pub interval: f64,       // Observed/historical interval
    pub eopr: f64,           // Historical EOPR
    pub p_bop: f64,
    pub p_aop: f64,
    pub decay: ReliabilityDecayParameters,
}
#[derive(Debug)]
pub enum ReliabilityDecayParameters {
    Exponential{lambda: f64, sig_aop: f64},
    RandomWalk{sig_bop: f64, sig_aop: f64, sig_eop: f64, alpha: f64}
}


#[derive(Debug)]
pub struct GumResult {
    // mean, variance, std, degf
    pub expected: f64,
    pub variance: f64,
    pub std_dev: f64,
    pub degrees_freedom: f64,
    pub _sensitivities: HashMap<String, f64>,
    pub coverage_factor: f64,
    pub confidence: f64
}
impl Default for GumResult {
    fn default() -> Self {
        Self{
            expected: 0.0,
            variance: 0.0,
            std_dev: 0.0,
            degrees_freedom: 0.0,
            _sensitivities: HashMap::new(),  // only filled for Functions
            coverage_factor: 1.0,
            confidence: 0.95,
        }
    }
}

#[derive(Debug)]
pub struct MonteCarloResult {
    pub samples: Vec<f64>,
}
impl MonteCarloResult {
    pub fn new(samples: Vec<f64>) -> Self {
        Self{samples: samples}
    }
    pub fn mean(&self) -> f64 {
        stats::mean(&self.samples)
    }
    //fn variance(&self) -> f64 {
    //    stats::variance(&self.samples)
    //}
    pub fn std_dev(&self) -> f64 {
        stats::std_dev(&self.samples)
    }
    pub fn get_sample(&self, i: usize) -> f64 {
        self.samples[i]
    }
}


#[derive(Default, Debug)]
pub struct RiskResultGlobal {
    pub pfa_true: f64,
    pub pfr_true: f64,
    pub cpfa_true: f64,
    pub acceptance: Tolerance,
}
#[derive(Debug)]
pub enum RiskResult {
    Global(RiskResultGlobal),
    Specific(f64),
}

#[derive(Debug)]
pub enum UncertResult {
    Gum(GumResult),
    Montecarlo(MonteCarloResult)
}
impl UncertResult {
    pub fn expected(&self) -> f64 {
        match self {
            UncertResult::Gum(v) => v.expected,
            UncertResult::Montecarlo(v) => v.mean(),
        }
    }
    pub fn std_uncert(&self) -> f64 {
        match self {
            UncertResult::Gum(v) => v.std_dev,
            UncertResult::Montecarlo(v) => v.std_dev(),
        }
    }
    pub fn distribution(&self) -> TypeBDist {
        match self {
            UncertResult::Gum(v) => TypeBDist::Normal(TypeBNormal::new(v.std_dev)),
            UncertResult::Montecarlo(_v) => todo!(),
        }
    }
    pub fn degrees_freedom(&self) -> f64 {
        match self {
            UncertResult::Gum(v) => v.degrees_freedom,
            UncertResult::Montecarlo(_) => f64::INFINITY,
        }
    }
}

#[derive(Debug)]
pub struct ReliabilityResult {
    pub _fyx: TypeBDist,
    pub _pdf_bt: dists::Distribution,
    pub _pdf_t: dists::Distribution,
    pub _pdf_pt: dists::Distribution,
    pub _pdf_bop: dists::Distribution,
    pub _obs_oot: f64,
    pub decay: Option<ReliabilityDecay>,
    pub p_bop: f64,
    pub p_aop: f64,
    pub sigma_aop: f64,
    pub p_repair: f64,
    pub p_adjust: f64,
    pub p_success: f64
}

#[derive(Debug)]
pub struct CostResult {
    pub _expected: f64,
    pub _p_available: f64,
    pub _num_spares: f64,
    pub _spares_year: f64,
    pub spare_cost: f64,
    pub calibration: f64,
    pub adjustment: f64,
    pub repair: f64,
    pub support: f64,
    pub total: f64,
    pub performance: f64,
}
impl CostResult {
    pub fn _add(&self, other: &CostResult) -> CostResult {
        // Combine costs
        CostResult{
            _expected: 0.0,
            _p_available: 0.0,
            _num_spares: 0.0,
            _spares_year: self._spares_year + other._spares_year,
            spare_cost: self.spare_cost + other.spare_cost,
            calibration: self.calibration + other.calibration,
            adjustment: self.adjustment + other.adjustment,
            repair: self.repair + other.repair,
            support: self.support + other.support,
            total: self.total + other.total,
            performance: self.performance + other.performance,
        }
    }
}


#[derive(Debug)]
pub struct QuantityResult {
    pub uncertainty: UncertResult,
    pub eopr: Option<Eopr>,  // True EOPR at test interval
    pub interval: f64,       // Test interval
    pub risk: Option<RiskResult>,
    pub reliability: Option<ReliabilityResult>,
    pub cost: Option<CostResult>,
}

#[derive(Debug)]
pub struct SystemResult<'a> {
    pub quantities: HashMap<String, QuantityResult>,
    pub montecarlo: HashMap<String, QuantityResult>,
    pub system: &'a MeasureSystem,
}
impl SystemResult<'_> {
    pub fn printit(&self) {
        for (qname, qty) in self.quantities.iter() {
            let gum = match &qty.uncertainty {
                UncertResult::Gum(v) => v,
                UncertResult::Montecarlo(_) => panic!(),
            };
            println!("{} = {} ± {} (ν={}, k={}, {}%)",
                qname, gum.expected, gum.std_dev,
                gum.degrees_freedom, gum.coverage_factor,
                gum.confidence * 100.0,
           );

            match &qty.risk {
                Some(r) => {
                    match r {
                        RiskResult::Specific(v) => println!("  Conformance: {}", v),
                        RiskResult::Global(v) => {
                            println!("  PFA: {}", v.pfa_true);
                            println!("  CPFA: {}", v.cpfa_true);
                            println!("  PFR: {}", v.pfr_true);
                        },
                    }
                }
                None => {}
            }

            match &qty.eopr {
                Some(r) => match r {
                    Eopr::True(t) => { 
                        println!("  EOPR: {}", t);
                        println!("  Interval: {}", qty.interval);
                    },
                    Eopr::Observed(_) => panic!(),
                },
                None => {},
            }
            match &qty.reliability {
                Some(r) => {
                    println!("  AOPR: {}", r.p_aop);
                    println!("  BOPR: {}", r.p_bop);
                },
                None => {},
            }

            match &qty.cost {
                Some(c) => {
                    println!("  COSTS:");
                    println!("    Calibration: {:.0}", c.calibration);
                    println!("    Adjustment: {:.0}", c.adjustment);
                    println!("    Repair: {:.0}", c.repair);
                    println!("    Support Total: {:.0}", c.support);
                    println!("    Annual Total: {:.0}", c.total);
                    println!("    Spares Acquisition: {:.0}", c.spare_cost);
                },
                _ => {}
            }
        }

        if self.system.settings.montecarlo > 1 {
            println!("-- MONTE CARLO --");
            for (qname, qty) in self.montecarlo.iter() {
                let mc = match &qty.uncertainty {
                    UncertResult::Montecarlo(v) => v,
                    UncertResult::Gum(_) => panic!(),
                };
                println!("{} = {} ± {}", qname, mc.mean(), mc.std_dev());

                match &qty.risk {
                    Some(r) => {
                        match r {
                            RiskResult::Specific(v) => println!("  Conformance: {}", v),
                            RiskResult::Global(v) => {
                                println!("  PFA: {}", v.pfa_true);
                                println!("  CPFA: {}", v.cpfa_true);
                                println!("  PFR: {}", v.pfr_true);
                            },
                        }
                    }
                    None => {}
                }

                match &qty.eopr {
                    Some(r) => match r {
                        Eopr::True(t) => println!("  EOPR: {}", t),
                        Eopr::Observed(_) => panic!(),
                    },
                    None => {},
                }
                match &qty.reliability {
                    Some(r) => {
                        println!("  AOPR: {}", r.p_aop);
                        println!("  BOPR: {}", r.p_bop);
                    },
                    None => {},
                }

                match &qty.cost {
                    Some(c) => {
                        println!("  COSTS:");
                        println!("    Calibration: {:.0}", c.calibration);
                        println!("    Adjustment: {:.0}", c.adjustment);
                        println!("    Repair: {:.0}", c.repair);
                        println!("    Support Total: {:.0}", c.support);
                        println!("    Annual Total: {:.0}", c.total);
                        println!("    Spares Acquisition: {:.0}", c.spare_cost);
                        if c.performance > 0.0 {
                            println!("    Annaul Performance: {:.0}", c.performance);
                        }
                    },
                    _ => {}
                }
            }
        }
    }
}
