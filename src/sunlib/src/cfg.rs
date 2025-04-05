// TOML configuration, serializable
use serde::{Serialize, Deserialize};

// serde default functions
fn default_infinity() -> f64 { std::f64::INFINITY }
fn default_1() -> f64 { 1.0 }
fn default_0() -> f64 { 0.0 }
fn default_095() -> f64 { 0.95 }
fn default_02() -> f64 { 0.02 }
fn default_0size() -> usize { 0 as usize }
fn default_eopr() -> Eopr {Eopr::True(0.95)}


// TypeB Distributions are serializable
// And have no center value (they are shifted to measurand expected value)
// But they have degrees of freedom
#[derive(Serialize, Deserialize, Debug)]
pub struct TypeBNormal {
    pub stddev: f64,
    #[serde(default = "default_infinity")]
    pub degf: f64,
    #[serde(default = "String::new")]
    pub name: String,
}
impl TypeBNormal {
    pub fn new(std_dev: f64) -> TypeBNormal {
        TypeBNormal{
            stddev: std_dev,
            degf: f64::INFINITY,
            name: String::from("")
        }
    }
}
#[derive(Serialize, Deserialize, Debug)]
pub struct TypeBUniform {
    pub a: f64,
    #[serde(default = "default_infinity")]
    pub degf: f64,
    #[serde(default = "String::new")]
    pub name: String,
}
#[derive(Serialize, Deserialize, Debug)]
pub struct TypeBTriangular {
    pub a: f64,
    #[serde(default = "default_infinity")]
    pub degf: f64,
    #[serde(default = "String::new")]
    pub name: String,
}
#[derive(Serialize, Deserialize, Debug)]
pub struct TypeBTolerance {
    pub tolerance: f64,
    pub confidence: f64,
    #[serde(default = "default_infinity")]
    pub degf: f64,
    #[serde(default = "String::new")]
    pub name: String,
}
#[derive(Serialize, Deserialize, Debug)]
pub struct TypeBGamma {
    pub a: f64,  // TODO - consider making mu/sigma instead of a, b?
    pub b: f64,
    #[serde(default = "default_infinity")]
    pub degf: f64,
    #[serde(default = "String::new")]
    pub name: String,
}
#[derive(Serialize, Deserialize, Debug)]
pub enum TypeBDist {
    Normal(TypeBNormal),
    Uniform(TypeBUniform),
    Triangular(TypeBTriangular),
    Gamma(TypeBGamma),
    Tolerance(TypeBTolerance),
    Symbol(String),
}
impl Default for TypeBDist {
    fn default() -> Self {
        TypeBDist::Normal(
            TypeBNormal::new(0.125)
        )
    }
}


// Tolerance/Limit
#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct Tolerance{
    pub low: f64,
    pub high: f64
}
impl Default for Tolerance {
    fn default() -> Self {
        Self{low: -1.0, high: 1.0}
    }
}


// End of period reliability, may be True or Observed
#[derive(Serialize, Deserialize, Debug)]
pub enum Eopr {
    True(f64),
    Observed(f64),
}


// Guardband Methods
#[derive(Serialize, Deserialize, PartialEq, Debug, Clone)]
pub enum GuardbandMethod {
    None,
    Manual,
    Rds,
    Rp10,
    Dobbert,
    Pfa,
    Cpfa,
    Pfr,
}
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Guardband {
    pub tolerance: Tolerance,  // For Manual method
    pub method: GuardbandMethod,
    #[serde(default = "default_02")]
    pub target: f64,  // For Pfa, Cpfa, and Pfr methods
}
impl Default for Guardband {
    fn default() -> Self {
        Guardband{
            tolerance: Tolerance{low:-1.0, high:1.0},
            method: GuardbandMethod::None,
            target: 0.02,
        }
    }
}


// End-item utility, tolerance, degrade, and fail limits
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Utility {
    pub tolerance: Tolerance,
    pub degrade: Option<Tolerance>,
    pub failure: Option<Tolerance>,
    pub guardband: Option<Guardband>,
    #[serde(default = "default_1")]
    pub psr: f64,
}


// Calibration renewal policy
#[derive(Clone, Serialize, Deserialize, Debug)]
pub enum RenewalPolicy {
    Never,
    Always,
    Asneeded
}

// Reliability Model - decay over interval
#[derive(Clone, Serialize, Deserialize, Debug)]
pub enum ReliabilityModel {
    Exponential,
    RandomWalk,
}


// Desired interval
#[derive(Serialize, Deserialize, Clone, Debug)]
pub enum IntervalTarget {
    Interval(f64),
    Eopr(f64),
}


// Calibration interval and EOPR
#[derive(Serialize, Deserialize, Debug)]
pub struct Interval {
    #[serde(default = "default_eopr")]
    pub eopr: Eopr,
    #[serde(default = "default_1")]
    pub years: f64,
    pub target: Option<IntervalTarget>,
}


// Calibration/Test information
#[derive(Serialize, Deserialize, Debug)]
pub struct Calibration {
    pub policy: RenewalPolicy,
    pub repair: Option<Tolerance>,
    #[serde(default = "default_0")]
    pub prob_discard: f64,
    pub prestress: Option<TypeBDist>,
    pub poststress: Option<TypeBDist>,
    pub mte_adjust: Option<TypeBDist>,
    pub mte_repair: Option<TypeBDist>,
    pub reliability_model: Option<ReliabilityModel>,
}


// Cost Model
#[derive(Serialize, Deserialize, Debug)]
pub struct Costs {
    #[serde(default = "default_0")]
    pub cal: f64,
    #[serde(default = "default_0")]
    pub adjust: f64,
    #[serde(default = "default_0")]
    pub repair: f64,
    #[serde(default = "default_0")]
    pub new_uut: f64,
    #[serde(default = "default_1")]
    pub num_uuts: f64,
    #[serde(default = "default_1")]
    pub spare_factor: f64,  // 0-1
    #[serde(default = "default_0")]
    pub spare_startup: f64,
    #[serde(default = "default_0")]
    pub down_cal: f64,  // days
    #[serde(default = "default_0")]
    pub down_adj: f64,
    #[serde(default = "default_0")]
    pub down_rep: f64,
    #[serde(default = "default_1")]
    pub p_use: f64,
    #[serde(default = "default_0")]
    pub cost_fa: f64,  //Cost of failure/false-accept (cf in RP-19)
    #[serde(default = "default_0")]
    pub cost_fr: f64
}


// Correlation between two variables
#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct CorrelationCoeff {
    pub v1: String,
    pub v2: String,
    pub coeff: f64
}


// GUM settings
#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct Settings {
    #[serde(default = "default_095")]
    pub confidence: f64,
    #[serde(default = "default_0size")]
    pub montecarlo: usize,
    // seed
    // correlate_typeas
    // mc_confidence  // shortest vs symmetric
}
impl Settings {
    fn new() -> Settings {
        Settings{confidence: 0.95, montecarlo: 0}
    }
}


// Measurement System, consisting of multiple quantities
#[derive(Serialize, Deserialize, Debug)]
pub struct MeasureSystem {
    pub quantity: Vec<ModelQuantity>,  // singular naming makes the TOML nicer
    #[serde(default = "Vec::new")]
    pub function: Vec<ModelFunction>,
    #[serde(default = "Vec::new")]
    pub correlation: Vec<CorrelationCoeff>,
    #[serde(default = "Settings::new")]
    pub settings: Settings,
}


// One direct quantity in a measurement system
#[derive(Serialize, Deserialize, Debug)]
pub struct ModelQuantity {
    #[serde(default = "String::new")]
    pub name: String,
    pub symbol: String,
    #[serde(default = "default_0")]
    pub measured: f64,
    pub units: Option<String>,
    #[serde(default = "Vec::new")]
    pub typeb: Vec<TypeBDist>,
    pub repeatability: Option<Vec<f64>>,
    pub reproducibility: Option<Vec<Vec<f64>>>,

    pub utility: Option<Utility>,
    pub interval: Option<Interval>,
    pub calibration: Option<Calibration>,
    pub cost: Option<Costs>,
}

// One indirect (calculated) quantity in a measurement system
#[derive(Serialize, Deserialize, Debug)]
pub struct ModelFunction {
    pub symbol: String,
    pub expr: String,
    pub units: Option<String>,
    pub utility: Option<Utility>,
    pub interval: Option<Interval>,
    pub calibration: Option<Calibration>,
    pub cost: Option<Costs>,
}
