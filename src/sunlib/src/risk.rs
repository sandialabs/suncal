// Global and Specific Risk Calculation Functions
use std::collections::HashMap;
use serde::{Serialize, Deserialize};
use roots::find_root_brent;

use crate::cfg::{TypeBDist, Tolerance, Guardband, GuardbandMethod};
use crate::dists::Distribution;

#[derive(Serialize, Deserialize, Debug)]
pub struct RiskModel {
    pub process: Distribution,
    pub test: TypeBDist,
    pub tolerance: Tolerance,
    #[serde(default = "Guardband::default")]
    pub guardband: Guardband,
}
impl Default for RiskModel {
    fn default() -> Self {
        Self{
            process: Distribution::Normal{mu: 0.0, sigma: 0.51},
            test: TypeBDist::default(),
            tolerance: Tolerance{low: -1.0, high: 1.0},
            guardband: Guardband::default(),
        }
    }
}
impl RiskModel {
    pub fn load_toml(config: &str) -> Result<RiskModel, Box<dyn std::error::Error>> {
        let model = toml::from_str::<RiskModel>(&config)?;
        model.process.check()?;
        model.test.check()?;
        Ok(model)
    }
    pub fn check(&self) -> Result<(), Box<dyn std::error::Error>> {
        self.process.check()?;
        self.test.check()?;
        Ok(())
    }
    pub fn get_config(&self) -> Result<String, toml::ser::Error> {
        toml::to_string(self)
    }
    pub fn domain(&self) -> (f64, f64) {
        let (mut p1, mut p2) = self.process.domain();
        if self.tolerance.high.is_finite() {
            p2 = p2.max(self.tolerance.high)
        }
        if self.tolerance.low.is_finite() {
            p1 = p1.min(self.tolerance.low)
        }
        (p1, p2)
    }
    pub fn process_low(&self) -> f64 {
        self.process.cdf(self.tolerance.low)
    }
    pub fn process_high(&self) -> f64 {
        1.0 - self.process.cdf(self.tolerance.high)
    }
    pub fn tur(&self) -> f64 {
        let plusminus = (self.tolerance.high - self.tolerance.low) / 2.0;
        let std = self.test.std_dev(&HashMap::new());
        plusminus / std / 2.0
    }
    pub fn get_guardband(&self) -> Tolerance {
        match &self.guardband.method {
            GuardbandMethod::None => Tolerance{low:self.tolerance.low, high:self.tolerance.high},
            GuardbandMethod::Manual => self.guardband.tolerance.clone(),
            GuardbandMethod::Rds => {
                let tur = self.tur();
                let gbf = (1.0 - 1.0 / tur.powi(2)).sqrt();
                let center = (self.tolerance.low + self.tolerance.high) / 2.0;
                let plusminus = self.tolerance.high - center;
                let accept_width = plusminus * gbf;
                Tolerance{low: center-accept_width, high: center+accept_width}
            },
            GuardbandMethod::Dobbert => {
                let tur = self.tur();
                let gbf = 1.0 - (1.04 - (0.38 * tur.ln() - 0.54).exp()) / tur;
                let center = (self.tolerance.low + self.tolerance.high) / 2.0;
                let plusminus = self.tolerance.high - center;
                let accept_width = plusminus * gbf;
                Tolerance{low: center-accept_width, high: center+accept_width}
            },
            GuardbandMethod::Rp10 => {
                let tur = self.tur();
                let gbf = 1.25 - 1.0 / tur;
                let center = (self.tolerance.low + self.tolerance.high) / 2.0;
                let plusminus = self.tolerance.high - center;
                let accept_width = plusminus * gbf;
                Tolerance{low: center-accept_width, high: center+accept_width}
            },
            GuardbandMethod::Pfa => {
                let mut width = (self.tolerance.high - self.tolerance.low) / 2.0;
                if !width.is_finite() {
                    let (p1, p2) = self.process.domain();
                    width = (p2 - p1) / 2.0;
                }
                match find_root_brent(-width*2.0, width, |x| {
                    self.pfa(&Tolerance{low:self.tolerance.low+x,
                                        high: self.tolerance.high-x})
                        - self.guardband.target
                }, &mut 1E-9f64) {
                    Ok(gbw) => Tolerance{low: self.tolerance.low + gbw, high: self.tolerance.high - gbw},
                    Err(_) => Tolerance{low: f64::NAN, high: f64::NAN},
                }
            },
            GuardbandMethod::Cpfa => {
                let mut width = (self.tolerance.high - self.tolerance.low) / 2.0;
                if !width.is_finite() {
                    let (p1, p2) = self.process.domain();
                    width = (p2 - p1) / 2.0;
                }
                let current = self.cpfa(&self.tolerance);
                let (bracket1, bracket2) = if current > self.guardband.target {
                    (0.0, width)
                } else {
                    (-width/100.0, -width*2.0)
                };
                match find_root_brent(bracket1, bracket2, |x| {
                    self.cpfa(&Tolerance{low:self.tolerance.low+x,
                                         high: self.tolerance.high-x})
                        - self.guardband.target
                }, &mut 1E-9f64) {
                    Ok(gbw) => Tolerance{low: self.tolerance.low + gbw, high: self.tolerance.high - gbw},
                    Err(_) => Tolerance{low: f64::NAN, high: f64::NAN},
                }
            },
            GuardbandMethod::Pfr => {
                let mut width = (self.tolerance.high - self.tolerance.low) / 2.0;
                if !width.is_finite() {
                    let (p1, p2) = self.process.domain();
                    width = (p2 - p1) / 2.0;
                }
                match find_root_brent(-width*2.0, width, |x| {
                    self.pfr(&Tolerance{low:self.tolerance.low+x,
                                        high: self.tolerance.high-x})
                        - self.guardband.target
                }, &mut 1E-9f64) {
                    Ok(gbw) => Tolerance{low: self.tolerance.low + gbw, high: self.tolerance.high - gbw},
                    Err(_) => Tolerance{low: f64::NAN, high: f64::NAN},
                }
            },
        }
    }
    pub fn worst_specific(&self, accept: &Tolerance) -> f64 {
        let tdist = self.test.pdf_given_y(0.0, &HashMap::new());
        let risk1: f64 = if self.tolerance.high.is_finite() {
            1.0 - tdist.cdf(self.tolerance.high - accept.high)
        } else { 0.0 };
        let risk2: f64 = if self.tolerance.low.is_finite() {
            tdist.cdf(self.tolerance.low - accept.low)
        } else { 0.0 };
        risk1.max(risk2)
    }
    pub fn pr_conform(&self, x: f64) -> f64 {
        let tdist = self.test.pdf_given_y(0.0, &HashMap::new());
        let risk_lower: f64 = if self.tolerance.low.is_finite() {
            tdist.cdf(self.tolerance.low - x)
        } else { 0.0 };
        let risk_upper: f64 = if self.tolerance.high.is_finite() {
            1.0 - tdist.cdf(self.tolerance.high - x)
        } else { 0.0 };
        1.0 - risk_lower - risk_upper
    }
    pub fn pfa(&self, accept: &Tolerance) -> f64 {
        let n = 5000;
        let tdist = self.test.pdf_given_y(0.0, &HashMap::new());
        let (prange_low, prange_high) = self.process.domain();
        let mut sum_below = 0.0;
        let mut sum_above = 0.0;

        // PFA - Above the limit
        if self.tolerance.high.is_finite() {
            let step = (prange_high - self.tolerance.high) / n as f64;
            for i in 1..n {
                let t = self.tolerance.high + i as f64 * step;
                sum_above += (tdist.cdf(accept.high-t) - tdist.cdf(accept.low-t)) * self.process.pdf(t);
            }
            sum_above *= step;
        }
        // PFA - Below the limit
        if self.tolerance.low.is_finite() {
            let step = (self.tolerance.low - prange_low) / n as f64;
            for i in 1..n {
                let t = prange_low + i as f64 * step;
                sum_below += (tdist.cdf(accept.high-t) - tdist.cdf(accept.low-t)) * self.process.pdf(t);
            }
            sum_below *= step;
        }
        sum_above + sum_below
    }
    pub fn pfr(&self, accept: &Tolerance) -> f64 {
        let n = 5000;
        let tdist = self.test.pdf_given_y(0.0, &HashMap::new());
        let (prange_low, prange_high) = self.process.domain();
        let top = if self.tolerance.high.is_finite() { self.tolerance.high } else { prange_high };
        let bot = if self.tolerance.low.is_finite() { self.tolerance.low } else { prange_low };
        let step = (top - bot) / n as f64;
        let mut pfr = 0.0;
        for i in 1..n {
            let t = bot + i as f64 * step;
            pfr += (1.0 - tdist.cdf(accept.high - t)) * self.process.pdf(t);
            pfr += tdist.cdf(accept.low - t) * self.process.pdf(t);
        }
        pfr * step
    }
    pub fn cpfa(&self, accept: &Tolerance) -> f64 {
        let n = 5000;
        let tdist = self.test.pdf_given_y(0.0, &HashMap::new());
        let (prange_low, prange_high) = self.process.domain();
        let top = if self.tolerance.high.is_finite() { self.tolerance.high } else { prange_high };
        let bot = if self.tolerance.low.is_finite() { self.tolerance.low } else { prange_low };
        let step = (top - bot) / n as f64;

        let mut intol_accept = 0.0;
        let mut accepted = 0.0;
        for i in 1..n {
            let t = bot + i as f64 * step;
            intol_accept += (tdist.cdf(accept.high - t) - tdist.cdf(accept.low - t)) * self.process.pdf(t);
        }
        intol_accept *= step;
        let step = (prange_high - prange_low) / n as f64;
        for i in 1..n {
            let t = prange_low + i as f64 * step;
            accepted += (tdist.cdf(accept.high - t) - tdist.cdf(accept.low - t)) * self.process.pdf(t);
        }
        accepted *= step;
        match accepted {
            0.0 => 0.0,
            _ => 1.0 - intol_accept / accepted,
        }
    }

    pub fn calculate(&self) -> RiskModelResult {
        let plow = self.process_low();
        let phigh = self.process_high();
        let gb = self.get_guardband();
        RiskModelResult{
            domain: self.domain(),
            process_low: plow,
            process_high: phigh,
            process_total: plow + phigh,
            specific: self.worst_specific(&gb),
            tur: self.tur(),
            pfa: self.pfa(&gb),
            pfr: self.pfr(&gb),
            cpfa: self.cpfa(&gb),
            guardband: gb
        }
    }


}

#[derive(Default)]
pub struct RiskModelResult {
    pub domain: (f64, f64),
    pub process_low: f64,
    pub process_high: f64,
    pub process_total: f64,
    pub specific: f64,
    pub tur: f64,
    pub pfa: f64,
    pub pfr: f64,
    pub cpfa: f64,
    pub guardband: Tolerance,
}
impl RiskModelResult {
    pub fn to_string(&self) -> String {
        [
            format!("Process Risk:         {:.3}%", self.process_total*100.0),
            format!("Process Risk (low)    {:.3}%", self.process_low*100.0),
            format!("Process Risk (high)   {:.3}%", self.process_high*100.0),
            format!("Worst Specific Risk:  {:.3}%", self.specific*100.0),
            format!("TUR:                  {:.3}", self.tur),
            format!("Global PFA:           {:.3}%", self.pfa*100.0),
            format!("Global CPFA:          {:.3}%", self.cpfa*100.0),
            format!("Global PFR:           {:.3}%", self.pfr*100.0),
            format!("Guardband Low:        {:.3}", self.guardband.low),
            format!("Guardband High:       {:.3}", self.guardband.high),
        ].join("\n")
    }
}