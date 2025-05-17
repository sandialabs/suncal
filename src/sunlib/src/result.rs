// Structures for calculation results
use mathru::algebra::linear::vector::Vector;
use mathru::algebra::linear::matrix::{General};
use mathru::statistics::distrib::{Distribution, Normal};

use tabled::{builder::Builder, settings::Style};

use crate::cfg::{MeasureSystem, Eopr, Tolerance, TypeBNormal, TypeBDist};
use crate::stats;

use crate::dists;


#[derive(Clone, Debug)]
pub struct ReliabilityDecay {
    pub expected: f64,
    pub tolerance: Tolerance,
    pub interval: f64,       // Observed/historical interval
    pub eopr: f64,           // Historical EOPR
    pub p_bop: f64,
    pub p_aop: f64,
    pub decay: ReliabilityDecayParameters,
}
#[derive(Clone, Debug)]
pub enum ReliabilityDecayParameters {
    Exponential{lambda: f64, sig_aop: f64},
    RandomWalk{sig_bop: f64, sig_aop: f64, sig_eop: f64, alpha: f64}
}


#[derive(Clone, Debug)]
pub struct GumComponents{
    pub varnames: Vec<String>,
    pub funcnames: Vec<String>,
    pub ux: General<f64>,
    pub cx: General<f64>,
    pub uy: General<f64>,
    pub partial_eqs: Vec<Vec<String>>,
}

#[derive(Clone, Debug)]
pub struct GumResult {
    // mean, variance, std, degf
    pub expected: f64,
    pub variance: f64,
    pub std_dev: f64,
    pub units: Option<String>,
    pub degrees_freedom: f64,
    pub gum: Option<GumComponents>,
    pub curve: Option<CurveResult>,
    pub coverage_factor: f64,
    pub confidence: f64,
    pub mcsamples: Option<MonteCarloResult>,
}
impl Default for GumResult {
    fn default() -> Self {
        Self{
            expected: 0.0,
            variance: 0.0,
            std_dev: 0.0,
            degrees_freedom: 0.0,
            units: None,
            gum: None,    // Functions only
            curve: None,  // Curves only
            coverage_factor: 1.0,
            confidence: 0.95,
            mcsamples: None,
        }
    }
}
impl GumResult {
    pub fn sample(&mut self, nsamples: usize) {
        let samples = Normal::new(self.expected, self.variance).random_sequence(nsamples.try_into().unwrap());
        self.mcsamples = Some(
            MonteCarloResult::new(samples, self.units.clone(), self.confidence)
        );
    }
}


#[derive(Clone, Debug)]
pub struct Histogram {
    pub bins: Vec<f64>,  // Left edge
    pub density: Vec<f64>,  // Density of each bin
    pub width: f64,   // Width of all bins
    pub low: f64,     // Lower bound of coverage region
    pub high: f64,    // Upper bound of coverage region
}
impl Histogram {
    fn new(samples: &Vec<f64>, bins: usize, low: f64, high: f64) -> Self {
        let min = samples.into_iter().min_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
        let max = samples.into_iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
        let width = (max - min) / bins as f64;
        let mut counts: Vec<usize> = vec![0; bins];
        let binlefts: Vec<f64> = (0..bins).map(|i| min + i as f64*width + width/2.0).collect();
        for samp in samples.iter() {
            let idx = (samp - min) / width - 0.1;
            counts[idx as usize] += 1;
        }
        let n = samples.len() as f64;
        let density: Vec<f64> = counts.iter().map(|i| *i as f64/n/width).collect();
        Self{
            bins: binlefts,
            density: density,
            width: width,
            low: low,
            high: high,
        }
    }
}


#[derive(Debug, Clone)]
pub struct MonteCarloResult {
    pub samples: Vec<f64>,  // Base units
    pub samples_scaled: Vec<f64>,  // User units
    pub expected: f64,  // User units
    pub variance: f64,
    pub std_dev: f64,
    pub units: Option<String>,
    pub confidence: f64,
    pub low: f64,   // Lower end of coverage region
    pub high: f64,  // Upper end of coverage region
    pub k: f64,     // Effective coverage factor
    pub hist: Histogram,
}
impl MonteCarloResult {
    pub fn new(samples: Vec<f64>, units: Option<String>, conf: f64) -> Self {
        let (scale, offset) = match &units {
            Some(unit) => {
                let u = units::parse_unit(&unit).unwrap();
                (u.scale, u.offset)
            },
            None => (1.0, 0.0),
        };
        let mean = stats::mean(&samples) / scale - offset;
        let variance = stats::variance(&samples) / scale / scale;
        let stddev = variance.sqrt();

        let samples_scaled = samples.iter().map(|s| s/scale - offset).collect();

        let c1 = (1.0-conf)/2.0;
        let c2 = 1.0-c1;
        let quants = stats::quantiles(&samples_scaled, vec![c1, c2]);
        let k = (quants[1] - quants[0]) / (2.0 * stddev);
        let h = Histogram::new(&samples_scaled, 50, quants[0], quants[1]);

        Self{
            samples: samples.clone(),
            samples_scaled: samples_scaled,
            expected: mean,
            variance: variance,
            std_dev: stddev,
            units: units.clone(),
            confidence: conf,
            low: quants[0],
            high: quants[1],
            k: k,
            hist: h,
        }
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
    pub tur: f64,
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
            UncertResult::Montecarlo(v) => v.expected,
        }
    }
    pub fn std_uncert(&self) -> f64 {
        match self {
            UncertResult::Gum(v) => v.std_dev,
            UncertResult::Montecarlo(v) => v.std_dev,
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
    pub pdf_bop: dists::Distribution,
    pub pdf_eop: dists::Distribution,
    pub _obs_oot: f64,
    pub decay: Option<ReliabilityDecay>,
    pub p_bop: f64,
    pub p_aop: f64,
    pub sigma_aop: f64,
    pub p_repair: f64,
    pub p_adjust: f64,
    pub p_success: f64,
    pub utility: dists::Distribution,
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


#[derive(Clone, Debug)]
pub struct CurveResult {
    pub model: String,
    pub p: Vector<f64>,
    pub cov: General<f64>,
    pub mse: f64,
    pub resid: Vector<f64>,
    pub npoints: usize,
    pub nparams: usize,
    pub xdata: Vector<f64>,  // Input x points
    pub ydata: Vector<f64>,  // Input y points
    pub xplot: Vector<f64>,  // X-values for plotting fit line
    pub yplot: Vector<f64>,  // Y-values on fit line
    pub conf_plus: Vector<f64>,  // Upper confidence band
    pub conf_minus: Vector<f64>  // Lower confidence band
}
impl CurveResult {
    pub fn noresult() -> Self {
        Self{
            model: "none".to_string(),
            p: Vector::new_row(vec![]),
            cov: General::new(0, 0, vec![]),
            mse: f64::NAN,
            resid: Vector::new_row(vec![]),
            npoints: 0,
            nparams: 0,
            xdata: Vector::new_row(vec![]),
            ydata: Vector::new_row(vec![]),
            xplot: Vector::new_row(vec![]),
            yplot: Vector::new_row(vec![]),
            conf_plus: Vector::new_row(vec![]),
            conf_minus: Vector::new_row(vec![]),
        }
    }
    pub fn qtyresult(&self) -> Vec<QuantityResult> {
        let mut results: Vec<QuantityResult> = Vec::new();
        for (i, coeff) in self.p.iter().enumerate() {
            results.push(
                QuantityResult{
                    symbol: char::from_u32(('a' as u32) + i as u32).unwrap().to_string(),
                    uncertainty: UncertResult::Gum(
                        GumResult{
                            expected: *coeff,
                            variance: self.cov[[i, i]],
                            std_dev: self.cov[[i, i]].sqrt(),
                            degrees_freedom: (self.npoints - self.nparams) as f64,
                            units: None,
                            gum: None,
                            curve: Some(self.clone()),
                            coverage_factor: 2.0,
                            confidence: 0.95,
                            mcsamples: None,
                        },
                    ),
                    eopr: None,
                    interval: 1.0,
                    risk: None,
                    reliability: None,
                    cost: None,
                }
            );
        }
        results
    }
}



#[derive(Debug)]
pub struct QuantityResult {
    pub symbol: String,
    pub uncertainty: UncertResult,
    pub eopr: Option<Eopr>,  // True EOPR at test interval
    pub interval: f64,       // Test interval
    pub risk: Option<RiskResult>,
    pub reliability: Option<ReliabilityResult>,
    pub cost: Option<CostResult>,
}


pub fn get_qresult<'a>(name: &String, qty_results: Option<&'a Vec<QuantityResult>>) -> Option<&'a QuantityResult> {
    match qty_results {
        Some(v) => {
            let mut q = None;
            for qty in v {
                if &qty.symbol == name {
                    q = Some(qty);
                    break;
                }
            }
            q
        },
        None => None,
    }
}


#[derive(Debug)]
pub struct SystemResult<'a> {
    pub quantities: Vec<QuantityResult>,
    pub montecarlo: Vec<QuantityResult>,
    pub system: &'a MeasureSystem,
}
impl SystemResult<'_> {
    pub fn varnames(&self) -> Vec<String> {
        let mut names: Vec<String> = Vec::new();
        for qty in self.quantities.iter() {
            names.push(qty.symbol.clone());
        }
        names
    }
    pub fn gum_expected(&self) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        // Means, StandardDevs, ks
        let mut means: Vec<f64> = Vec::new();
        let mut stds: Vec<f64> = Vec::new();
        let mut ks: Vec<f64> = Vec::new();
        for qty in self.quantities.iter() {
            let gum = match &qty.uncertainty {
                UncertResult::Gum(v) => v,
                UncertResult::Montecarlo(_) => panic!(),
            };
            means.push(gum.expected);
            stds.push(gum.std_dev);
            ks.push(gum.coverage_factor);
        }
        (means, stds, ks)
    }
    pub fn to_string(&self) -> String {
        // Raw output for CLI, full MQA model
        let mut string_list: Vec<String> = vec![];
        for qty in self.quantities.iter() {
            let gum = match &qty.uncertainty {
                UncertResult::Gum(v) => v,
                UncertResult::Montecarlo(_) => panic!(),
            };
            let units = match &gum.units {
                Some(u) => u,
                None => "",
            };
            string_list.push(format!("{} = ({} ± {}) {}; (ν={}, k={}, {}%)",
                qty.symbol, gum.expected, gum.std_dev,
                units,
                gum.degrees_freedom, gum.coverage_factor,
                gum.confidence * 100.0,
           ));

            match &qty.risk {
                Some(r) => {
                    match r {
                        RiskResult::Specific(v) => println!("  Conformance: {}", v),
                        RiskResult::Global(v) => {
                            string_list.push(format!("  PFA: {}", v.pfa_true));
                            string_list.push(format!("  CPFA: {}", v.cpfa_true));
                            string_list.push(format!("  PFR: {}", v.pfr_true));
                        },
                    }
                }
                None => {}
            }

            match &qty.eopr {
                Some(r) => match r {
                    Eopr::True(t) => { 
                        string_list.push(format!("  EOPR: {}", t));
                        string_list.push(format!("  Interval: {}", qty.interval));
                    },
                    Eopr::Observed(_) => panic!(),
                },
                None => {},
            }
            match &qty.reliability {
                Some(r) => {
                    string_list.push(format!("  AOPR: {}", r.p_aop));
                    string_list.push(format!("  BOPR: {}", r.p_bop));
                    if r.p_success.is_finite() {
                        string_list.push(format!("  Success: {}", r.p_success));
                    };
                },
                None => {},
            }

            match &qty.cost {
                Some(c) => {
                    string_list.push(format!("  COSTS:"));
                    string_list.push(format!("    Calibration: {:.0}", c.calibration));
                    string_list.push(format!("    Adjustment: {:.0}", c.adjustment));
                    string_list.push(format!("    Repair: {:.0}", c.repair));
                    string_list.push(format!("    Support Total: {:.0}", c.support));
                    string_list.push(format!("    Annual Total: {:.0}", c.total));
                    string_list.push(format!("    Spares Acquisition: {:.0}", c.spare_cost));
                },
                _ => {}
            }
        }

        if self.system.settings.montecarlo > 1 {
            string_list.push(format!("-- MONTE CARLO --"));
            for qty in self.montecarlo.iter() {
                let mc = match &qty.uncertainty {
                    UncertResult::Montecarlo(v) => v,
                    UncertResult::Gum(v) => &v.mcsamples.clone().unwrap(),
                };
                let units = match &mc.units {
                    Some(u) => u,
                    None => "",
                };
                string_list.push(format!("{} = ({} ± {}) {}", qty.symbol, mc.expected, mc.std_dev, units));

                match &qty.risk {
                    Some(r) => {
                        match r {
                            RiskResult::Specific(v) => string_list.push(format!("  Conformance: {}", v)),
                            RiskResult::Global(v) => {
                                string_list.push(format!("  PFA: {}", v.pfa_true));
                                string_list.push(format!("  CPFA: {}", v.cpfa_true));
                                string_list.push(format!("  PFR: {}", v.pfr_true));
                            },
                        }
                    }
                    None => {}
                }

                match &qty.eopr {
                    Some(r) => match r {
                        Eopr::True(t) => string_list.push(format!("  EOPR: {}", t)),
                        Eopr::Observed(_) => panic!(),
                    },
                    None => {},
                }
                match &qty.reliability {
                    Some(r) => {
                        string_list.push(format!("  AOPR: {}", r.p_aop));
                        string_list.push(format!("  BOPR: {}", r.p_bop));
                        string_list.push(format!("  Success: {}", r.p_success));
                    },
                    None => {},
                }

                match &qty.cost {
                    Some(c) => {
                        string_list.push(format!("  COSTS:"));
                        string_list.push(format!("    Calibration: {:.0}", c.calibration));
                        string_list.push(format!("    Adjustment: {:.0}", c.adjustment));
                        string_list.push(format!("    Repair: {:.0}", c.repair));
                        string_list.push(format!("    Support Total: {:.0}", c.support));
                        string_list.push(format!("    Annual Total: {:.0}", c.total));
                        string_list.push(format!("    Spares Acquisition: {:.0}", c.spare_cost));
                        if c.performance > 0.0 {
                            string_list.push(format!("    Annaul Performance: {:.0}", c.performance));
                        }
                    },
                    _ => {}
                }
            }
        }
        string_list.join("\n")
    }
    pub fn printit(&self) {
        // Print raw output to command line
        println!("{}", self.to_string());
    }

    pub fn summary(&self, ndig: usize) -> String {
        // GUM/Monte Carlo result summary of all quantities
        let mut report = String::new();
    
        if self.quantities.len() > 0 {
            let mut builder = Builder::default();
            builder.push_record(vec![
                "Quantity", "Value", "Standard Uncertainty", "Expanded Unceratinty",
                "Units", "Deg. Freedom", "Coverage Factor", "Confidence"
            ]);
            for qty in self.quantities.iter() {
                let gum = match &qty.uncertainty {
                    UncertResult::Gum(v) => v,
                    UncertResult::Montecarlo(_) => panic!(),
                };
                let units = match &gum.units {
                    Some(u) => u,
                    None => "",
                };
                builder.push_record(vec![
                    qty.symbol.clone(),
                    format!("{1:.0$}", ndig, gum.expected),
                    format!("{1:.0$}", ndig, gum.std_dev),
                    format!("{1:.0$}", ndig, gum.std_dev * gum.coverage_factor),
                    units.to_string(),
                    format!("{:.1}", gum.degrees_freedom),
                    format!("{:.3}", gum.coverage_factor),
                    format!("{:.2} %", gum.confidence*100.0),
                ]);

            }
            let mut table = builder.build();
            table.with(Style::rounded());
            report.push_str(&table.to_string());
        };

        if self.montecarlo.len() > 0 {
            report.push_str("\n\nMonte Carlo\n\n");
            let mut builder = Builder::default();
            builder.push_record(vec![
                "Quantity", "Value", "Standard Uncertainty", "Interval",
                "Units", "Coverage Factor", "Confidence"
            ]);
            for qty in self.montecarlo.iter() {
                let mc = match &qty.uncertainty {
                    UncertResult::Montecarlo(v) => v,
                    UncertResult::Gum(v) => &v.mcsamples.clone().unwrap(),
                };
                let units = match &mc.units {
                    Some(u) => u,
                    None => "",
                };

                builder.push_record(vec![
                    qty.symbol.clone(),
                    format!("{1:.0$}", ndig, mc.expected),
                    format!("{1:.0$}", ndig, mc.std_dev),
                    // format!("({1:.0$}, {2:.0$})", ndig, quants[0], quants[1]),
                    format!("({1:.0$}, {2:.0$})", ndig, mc.low, mc.high),
                    units.to_string(),
                    format!("{:.3}", mc.k),
                    format!("{:.2} %", mc.confidence*100.0),
                ]);
            }
            let mut table = builder.build();
            table.with(Style::rounded());
            report.push_str(&table.to_string());
        };

        report
    }

    pub fn get_quantity(&self, name: &str, ndig: usize) -> String {
        // Get report for named quantity
        let mut out = String::new();
        if self.quantities.len() == self.montecarlo.len() {
            // Combined GUM + MC report
            for (idx, qty) in self.quantities.iter().enumerate() {
                if qty.symbol == name {
                    out = self.mc_report(name, &qty, &self.montecarlo[idx], ndig)
                }
            }
        } else if self.quantities.len() > 0 {
            // GUM only report
            for qty in &self.quantities {
                if qty.symbol == name {
                    out = self.gum_report(name, &qty, ndig)
                }
            }
        }
        out
    }

    pub fn get_histogram(&self, name: &str) -> Option<Histogram> {
        let mut out: Option<Histogram> = None;
        for qty in self.montecarlo.iter() {
            if qty.symbol == name {
                out = match &qty.uncertainty {
                    UncertResult::Montecarlo(v) => {
                        Some(v.hist.clone())
                    }
                    UncertResult::Gum(v) => Some(v.mcsamples.clone().unwrap().hist.clone()),
                };
            }
        }
        out
    }

    pub fn get_curve(&self, name: &str) -> Option<CurveResult> {
        let mut out: Option<CurveResult> = None;
        for qty in self.quantities.iter() {
            if qty.symbol == name {
                out = match &qty.uncertainty {
                    UncertResult::Gum(v) => {
                        v.curve.clone()
                    },
                    _ => panic!(),
                };
            }
        }
        out
    }

    pub fn gum_report(&self, name: &str, qty: &QuantityResult, ndig: usize) -> String {
        // Get report for a GUM calculation
        let mut out: String = String::new();

        let gum = match &qty.uncertainty {
            UncertResult::Gum(v) => v,
            UncertResult::Montecarlo(_) => panic!(),
        };
        let units = match &gum.units {
            Some(u) => u,
            None => "",
        };

        let mut builder = Builder::default();
        builder.push_record(vec!["Parameter", "Value"]);
        builder.push_record(vec!["Name", name]);
        builder.push_record(vec!["Expected Value".to_string(), format!("{1:.0$}", ndig, gum.expected),]);
        builder.push_record(vec!["Standard Uncertainty".to_string(), format!("{1:.0$}", ndig, gum.std_dev)]);
        builder.push_record(vec!["Expanded Uncertainty".to_string(), format!("{1:.0$}", ndig, gum.std_dev*gum.coverage_factor)]);
        builder.push_record(vec!["Units".to_string(), units.to_string()]);
        builder.push_record(vec!["Degrees of Freedom".to_string(), format!("{:.2}", gum.degrees_freedom)]);
        builder.push_record(vec!["Level of Confidence".to_string(), format!("{:.2} %", gum.confidence*100.0)]);

        let mut table = builder.build();
        table.with(Style::rounded());
        out.push_str(&table.to_string());

        if let Some(g) = &gum.gum {
            out.push_str(&self.gum_sens(name, g, ndig));
        };
        if let Some(c) = &gum.curve {
            out.push_str(&self.curve_report(c, ndig));
        };
        out
    }

    pub fn mc_report(&self, name: &str, gum: &QuantityResult, mc: &QuantityResult, ndig: usize) -> String {
        // Get report for a MC and GUM calculation
        let mut out: String = String::new();

        let gumunc = match &gum.uncertainty {
            UncertResult::Gum(v) => v,
            UncertResult::Montecarlo(_) => panic!(),
        };
        let mcunc = match &mc.uncertainty {
            UncertResult::Montecarlo(v) => v,
            UncertResult::Gum(v) => &v.mcsamples.clone().unwrap(),
        };
        let units = match &gumunc.units {
            Some(u) => u,
            None => "",
        };

        let mut builder = Builder::default();
        builder.push_record(vec!["Parameter", "GUM", "Monte Carlo"]);
        builder.push_record(vec![
            "Expected Value".to_string(),
            format!("{1:.0$}", ndig, gumunc.expected),
            format!("{1:.0$}", ndig, mcunc.expected)]);
        builder.push_record(vec![
            "Standard Uncertainty".to_string(),
            format!("{1:.0$}", ndig, gumunc.std_dev),
            format!("{1:.0$}", ndig, mcunc.std_dev)]);

        builder.push_record(vec![
            "Expanded Uncertainty".to_string(),
            format!("{1:.0$}", ndig, gumunc.std_dev*gumunc.coverage_factor),
            format!("{1:.0$}", ndig, mcunc.std_dev*mcunc.k),
        ]);

        builder.push_record(vec!["Units".to_string(), units.to_string(), units.to_string()]);
        builder.push_record(vec!["Degrees of Freedom".to_string(), format!("{:.2}", gumunc.degrees_freedom), "-".to_string()]);
        builder.push_record(vec!["Coverage Factor".to_string(), format!("{:.2}", gumunc.coverage_factor), format!("{:.2}", mcunc.k)]);
        builder.push_record(vec![
            "Level of Confidence".to_string(),
            format!("{:.2} %", gumunc.confidence*100.0),
            format!("{:.2} %", gumunc.confidence*100.0)]);
        builder.push_record(vec![
            "Coverage Region, Lower Limit".to_string(),
            format!("{1:.0$}", ndig, gumunc.expected - gumunc.std_dev*gumunc.coverage_factor),
            format!("{1:.0$}", ndig, mcunc.low),
        ]);
        builder.push_record(vec![
            "Coverage Region, Upper Limit".to_string(),
            format!("{1:.0$}", ndig, gumunc.expected + gumunc.std_dev*gumunc.coverage_factor),
            format!("{1:.0$}", ndig, mcunc.high),
        ]);

        let mut table = builder.build();
        table.with(Style::rounded());
        out.push_str(&table.to_string());

        if let Some(g) = &gumunc.gum {
            out.push_str("\n\nGUM Calculation\n");
            out.push_str(&self.gum_sens(name, g, ndig));
        };
        if let Some(c) = &gumunc.curve {
            out.push_str("\n\nCurve Calculation\n");
            out.push_str(&self.curve_report(c, ndig));
        };
        out
    }

    fn gum_sens(&self, name: &str, gum: &GumComponents, ndig: usize) -> String {
        let mut out = String::new();
        out.push_str("\nSensitivity Coefficients\n");
        let mut builder = Builder::default();
        builder.push_record(vec![
            "Variable", "Sensitivity (base units)", "Proportion"
        ]);

        let fidx = gum.funcnames.iter().position(|n| n == name).unwrap();
        let sens = gum.cx.get_row(fidx);

        for i in 0..gum.varnames.len() {
            let prop = (sens[i].powi(2) * gum.ux[[i,i]]) / gum.uy[[fidx,fidx]] * 100.0;
            builder.push_record(vec![
                gum.varnames[i].clone(),
                format!("{1:.0$}", ndig, sens[i]),
                format!("{:.2} %", prop),
            ]);
        }
        let mut table = builder.build();
        table.with(Style::rounded());
        out.push_str(&table.to_string());
        out.push_str("\n\n");

        let partials = &gum.partial_eqs[fidx];
        builder = Builder::default();
        builder.push_record(vec![
            "Sensitivity Equations"
        ]);
        for partial in partials {
            builder.push_record(vec![
                partial.replace("{", "").replace("}", "").replace("-0.0", "").replace("0.0-", "-")
            ])
        };
        let mut table = builder.build();
        table.with(Style::rounded());
        out.push_str(&table.to_string());

        // Uy matrix
        if gum.funcnames.len() > 1 {
            out.push_str("\n\nOutput Covariance\n\n");
            builder = Builder::default();
            let mut cols: Vec<String> = vec!["-".to_string()];
            cols.extend(gum.funcnames.clone());
            builder.push_record(cols);
            for i in 0..gum.funcnames.len() {
                let mut row: Vec<String> = vec![gum.funcnames[i].clone()];
                row.extend(gum.uy.get_row(i).iter().map(|x| format!("{1:.0$}", ndig, x)));
                builder.push_record(row)
            }
            let mut table = builder.build();
            table.with(Style::rounded());
            out.push_str(&table.to_string());
        }
        out
    }

    fn curve_report(&self, curve: &CurveResult, ndig: usize) -> String {
        let mut out: String = String::new();

        let mut names: Vec<String> = Vec::new();
        for i in 0..curve.nparams {
            names.push(char::from_u32(('a' as u32) + i as u32).unwrap().to_string());
        };

        out.push_str(&format!("\n\nModel: {}", curve.model));
        out.push_str("\n\nFit Parameters\n\n");
        let mut builder = Builder::default();
        builder.push_record(vec!["Coefficient", "Value", "Std. Uncertainty"]);
        for i in 0..names.len() {
            let row: Vec<String> = vec![
                names[i].clone(),
                format!("{1:.0$}", ndig, curve.p[i]),
                format!("{1:.0$}", ndig, curve.cov[[i,i]].sqrt()),
            ];
            builder.push_record(row);
        }
        let mut table = builder.build();
        table.with(Style::rounded());
        out.push_str(&table.to_string());

        out.push_str("\n\nFit Parameter Covariance\n\n");
        let mut builder = Builder::default();
        let mut cols: Vec<String> = vec!["-".to_string()];
        cols.extend(names.clone());
        builder.push_record(&cols);
        cols.extend(names.clone());
        for i in 0..names.len() {
            let mut row: Vec<String> = vec![names[i].clone()];
            row.extend(curve.cov.get_row(i).iter().map(|x| format!("{1:.0$}", ndig, x)));
            builder.push_record(row)
        }
        let mut table = builder.build();
        table.with(Style::rounded());
        out.push_str(&table.to_string());

        out.push_str("\n\nCurve Fit Properties\n\n");
        builder = Builder::default();
        builder.push_record(vec!["Parameter", "Value"]);
        builder.push_record(vec!["Mean Squared Error", &format!("{1:.0$}", ndig, curve.mse)]);
        builder.push_record(vec!["Data Points", &curve.npoints.to_string()]);
        let mut table = builder.build();
        table.with(Style::rounded());
        out.push_str(&table.to_string());
        out
    }
}
