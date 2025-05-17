use std::error::Error;
use std::fmt;
use std::collections::HashMap;
use std::sync::Mutex;
use std::str::FromStr;
use lazy_static::lazy_static;

use exmex::prelude::*;
use exmex::{BinOp, MakeOperators, MatchLiteral, Operator,
            literal_matcher_from_pattern, ops_factory};
use::exmex::regex::{Regex, Captures};

mod units;


#[derive(Debug, PartialEq, Eq)]
pub struct UndefinedUnitError {
    msg: String
}
impl std::error::Error for UndefinedUnitError {}
impl fmt::Display for UndefinedUnitError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.msg)
    }
}

#[derive(Debug, PartialEq, Eq)]
pub struct DimensionError {
    msg: String
}
impl std::error::Error for DimensionError {}
impl fmt::Display for DimensionError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.msg)
    }
}


// Unit Registry is a static
lazy_static! {
    static ref unit_registry: Mutex<HashMap<String, Unit>> = Mutex::new(HashMap::new());
}


#[derive(Clone, PartialEq, Debug)]
pub struct Dimension {
    length: i8,
    mass: i8,
    time: i8,
    temp: i8,
    current: i8,
    intensity: i8,
    amount: i8,
    valid: bool,
    pow: i8,
}
impl Default for Dimension {
    fn default() -> Self {
        Self{
            length: 0,
            mass: 0,
            time: 0,
            temp: 0,
            current: 0,
            intensity: 0,
            amount: 0,
            valid: true,
            pow: 0
        }
    }
}
impl Dimension {
    fn new(length: i8, mass: i8, time: i8,
           temp: i8, current: i8, intensity: i8, amount: i8) -> Self {
        Self{
            length: length,
            mass: mass,
            time: time,
            temp: temp,
            current: current,
            intensity: intensity,
            amount: amount,
            valid: true,
            pow: 0,
        }
    }
    fn is_compatible(&self, other: &Dimension) -> bool {
        self.length == other.length
            && self.mass == other.mass
            && self.time == other.time
            && self.temp == other.temp
            && self.current == other.current
            && self.intensity == other.intensity
            && self.amount == other.amount
            && self.valid
            && other.valid
    }
    fn is_dimensionless(&self) -> bool {
        self.length == 0
        && self.mass == 0
        && self.time == 0
        && self.temp == 0
        && self.current == 0
        && self.intensity == 0
        && self.amount == 0
    }
    fn invalid() -> Dimension {
        Dimension{
            length: 0,
            mass: 0,
            time: 0,
            temp: 0,
            current: 0,
            intensity: 0,
            amount: 0,
            valid: false,
            pow: 0,
        }
    }
}
impl FromStr for Dimension {
    type Err = DimensionError;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.parse::<i8>() {
            Ok(f) => {
                // Constant
                Ok(Dimension{
                    length: 0, mass: 0, time: 0, temp: 0,
                    current: 0, intensity: 0, amount: 0,
                    valid: true,
                    pow: f
                })
            },
            Err(_) => {
                match parse_dim(s) {
                    Ok(v) => Ok(v),
                    Err(_) => Err(DimensionError{msg:format!("Cannot parse dimension {}", &s)})
                }
            },
        }
    }
}

ops_factory!(
    DimOpsFactory,  // name of struct
    Dimension,            // data type of operands
    Operator::make_bin(
        "^",
        BinOp{
            apply: pow_dim,
            prio: 6,
            is_commutative: false,
        }
    ),
    Operator::make_bin(
        "*",
        BinOp{
            apply: mul_dim,
            prio: 4,
            is_commutative: true,
        }
    ),
    Operator::make_bin(
        "/",
        BinOp{
            apply: div_dim,
            prio: 4,
            is_commutative: false,
        }
    ),
    Operator::make_bin(
        "+",
        BinOp{
            apply: add_dim,
            prio: 2,
            is_commutative: true,
        }
    ),
    Operator::make_bin(
        "-",
        BinOp{
            apply: add_dim,
            prio: 2,
            is_commutative: true,
        }
    ),
    Operator::make_bin(
        "atan2",
        BinOp{
            apply: binary_dimensionless,
            prio: 3,
            is_commutative: false,
        }
    ),
    Operator::make_unary("sqrt", sqrt_dim),
    Operator::make_unary("sin", dimensionless_op),
    Operator::make_unary("cos", dimensionless_op),
    Operator::make_unary("tan", dimensionless_op),
    Operator::make_unary("exp", dimensionless_op),
    Operator::make_unary("asin", dimensionless_op),
    Operator::make_unary("asin", dimensionless_op),
    Operator::make_unary("acos", dimensionless_op),
    Operator::make_unary("atan", dimensionless_op)
    // NO COMMA AT END
);
literal_matcher_from_pattern!(DimMatcher, r"^(\[.*]|[0-9]*[.][0-9]+|[0-9]+)");

fn mul_dim(a: Dimension, b: Dimension) -> Dimension {
    Dimension::new(
        a.length+b.length,
        a.mass+b.mass,
        a.time+b.time,
        a.temp+b.temp,
        a.current+b.current,
        a.intensity+b.intensity,
        a.amount+b.amount,
    )
}
fn div_dim(a: Dimension, b: Dimension) -> Dimension {
    Dimension::new(
        a.length-b.length,
        a.mass-b.mass,
        a.time-b.time,
        a.temp-b.temp,
        a.current-b.current,
        a.intensity-b.intensity,
        a.amount-b.amount,
    )
}
fn add_dim(a: Dimension, b: Dimension) -> Dimension {
    Dimension{
        length: a.length,
        mass: a.mass,
        time: a.time,
        temp: a.temp,
        current: a.current,
        intensity: a.intensity,
        amount: a.amount,
        valid: a.is_compatible(&b),
        pow: 0,
    }
}
fn pow_dim(a: Dimension, b: Dimension) -> Dimension {
    Dimension{
        length: a.length * b.pow,
        mass: a.mass * b.pow,
        time: a.time * b.pow,
        temp: a.temp * b.pow,
        current: a.current * b.pow,
        intensity: a.intensity * b.pow,
        amount: a.amount * b.pow,
        valid: true,
        pow: 0,
    }
}
fn sqrt_dim(a: Dimension) -> Dimension {
    // what if dim is odd?
    Dimension{
        length: a.length / 2,
        mass: a.mass / 2,
        time: a.time / 2,
        temp: a.temp / 2,
        current: a.current / 2,
        intensity: a.intensity / 2,
        amount: a.amount / 2,
        valid: true,
        pow: 0,
    }
}
fn binary_dimensionless(a: Dimension, b: Dimension) -> Dimension{
    // e.g. atan2
    if a.is_compatible(&b) {
        Dimension::default()
    } else {
        Dimension::invalid()
    }
}
fn dimensionless_op(a: Dimension) -> Dimension {
    if a.is_dimensionless() {
        Dimension::default()
    } else {
        Dimension::invalid()
    }
}


#[derive(Clone, PartialEq, Debug)]
pub struct BaseQuantity {
    pub magnitude: f64,
    pub dim: Dimension,
}
impl Default for BaseQuantity {
    fn default() -> Self {
        // Dimensionless 0
        Self{
            magnitude: 0.0,
            dim: Dimension::default(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct Unit {
    pub name: String,
    pub abbr: String,
    pub dim: Dimension,
    pub scale: f64,
    pub offset: f64,
}
impl Unit {
    fn new(name: String, abbr: String, dim: Dimension, scale: f64, offset: f64) -> Self {
        Self{
            name: name,
            abbr: abbr,
            dim: dim,
            scale: scale,
            offset: offset,
        }
    }
}
impl Default for Unit {
    fn default() -> Self {
        Self::new("dimensionless".to_string(),
                  String::new(),
                  Dimension::default(),
                  1.0,
                  0.0
        )
    }
}
impl FromStr for Unit {
    type Err = UndefinedUnitError;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        // Try Numeric First
        match s.parse::<f64>() {
            Ok(v) => {
                Ok(Unit{name: s.to_string(), abbr: s.to_string(),
                       dim: Dimension::default(),
                       scale: v,
                       offset: 0.0})
            },
            Err(_) => {
                let ureg = unit_registry.lock().unwrap();
                let unit = ureg.get(s).ok_or(UndefinedUnitError{msg: format!("Undefined Unit `{}`", &s)})?;
                Ok(unit.clone())
            }
        }
    }
}




fn mul_unit(a: Unit, b: Unit) -> Unit {
    let valid = a.offset == 0.0 && b.offset == 0.0;
    Unit{
        name: format!("{} * {}", a.name, b.name),
        abbr: format!("{}*{}", a.abbr, b.abbr),
        dim: Dimension{
            length: a.dim.length + b.dim.length,
            mass: a.dim.mass + b.dim.mass,
            time: a.dim.time + b.dim.time,
            temp: a.dim.temp + b.dim.temp,
            current: a.dim.temp + b.dim.temp,
            intensity: a.dim.intensity + b.dim.intensity,
            amount: a.dim.amount + b.dim.amount,
            valid: valid,
            pow: 0,
        },
        scale: a.scale * b.scale,
        offset: a.offset + b.offset,
    }
}
fn div_unit(a: Unit, b: Unit) -> Unit {
    let valid = a.offset == 0.0 && b.offset == 0.0;
    Unit{
        name: format!("{} / {}", a.name, b.name),
        abbr: format!("{}/{}", a.abbr, b.abbr),
        dim: Dimension{
            length: a.dim.length - b.dim.length,
            mass: a.dim.mass - b.dim.mass,
            time: a.dim.time - b.dim.time,
            temp: a.dim.temp - b.dim.temp,
            current: a.dim.current - b.dim.current,
            intensity: a.dim.intensity - b.dim.intensity,
            amount: a.dim.amount - b.dim.amount,
            valid: valid,
            pow: 0,
        },
        scale: a.scale / b.scale,
        offset: a.offset - b.offset,
    }
}
fn pow_unit(a: Unit, b: Unit) -> Unit {
    let valid = b.dim.is_dimensionless();
    let pow = b.scale as i8;
    Unit{
        name: a.name.clone(),
        abbr: a.abbr.clone(),
        dim: Dimension{
            length: a.dim.length * pow,
            mass: a.dim.mass * pow,
            time: a.dim.time * pow,
            temp: a.dim.temp * pow,
            current: a.dim.current * pow,
            intensity: a.dim.intensity * pow,
            amount: a.dim.amount * pow,
            valid: valid,
            pow: 0,
        },
        scale: a.scale.powi(pow.into()),
        offset: a.offset,
    }
}
ops_factory!(
    UnitOpsFactory,  // name of struct
    Unit,            // data type of operands
    Operator::make_bin(
        "*",
        BinOp{
            apply: mul_unit,
            prio: 1,
            is_commutative: true,
        }
    ),
    Operator::make_bin(
        "/",
        BinOp{
            apply: div_unit,
            prio: 1,
            is_commutative: false,
        }
    ),
    Operator::make_bin(
        "^",
        BinOp{
            apply: pow_unit,
            prio: 4,
            is_commutative: true,
        }
    )  // NO COMMA AT END
);
literal_matcher_from_pattern!(UnitMatcher, r"^[a-zA-Zα-ωΑ-Ω_0-9]+[a-zA-Zα-ωΑ-Ω_0-9]?");  // Everything not a number


pub fn parse_unit(unitstr: &str) -> Result<Unit, Box<dyn Error>> {
    // Get Units from bracketed unit string from brackets
    let unit_evald = if unitstr.trim_ascii_start().len() == 0 {
        Unit::default()
    } else {
        let units = FlatEx::<Unit, UnitOpsFactory, UnitMatcher>::parse(&unitstr)?;
        let ueval = units.eval(&[])?;
        ueval
    };
    Ok(unit_evald)
}

fn parse_baseqty(expr: &str) -> Result<BaseQuantity,  Box<dyn Error>> {
    // Get BaseQuantity from bracketed units [...]
    let expr = expr.trim_start_matches('[').trim_end_matches(']');
    let parts: Vec<&str> = expr.split_whitespace().collect();
    let magnitude = parts[0].parse::<f64>()?;
    let unitstr = parts[1..].join(" ");
    let qty = make_baseqty(magnitude, Some(unitstr), true)?;
    Ok(qty)
}

pub fn make_baseqty(value: f64, unitstr: Option<String>, offset: bool) -> Result<BaseQuantity, Box<dyn Error>> {
    let units = match unitstr {
        Some(u) => parse_unit(&u)?,
        None => Unit::default(),
    };
    let mag = if offset {
        (value + units.offset) * units.scale
    } else {
        value * units.scale
    };
    let qty = BaseQuantity{
        magnitude: mag,
        dim: units.dim.clone(),
    };
    Ok(qty)
}

fn parse_dim(expr: &str) -> Result<Dimension,  Box<dyn Error>> {
    // Get dimension from bracketed units [...]
    let qty = parse_baseqty(expr)?;
    Ok(qty.dim.clone())
}


fn read_units() {
    let mut ureg = unit_registry.lock().unwrap();
    let mut dim: Dimension = Dimension::default();

    for line in units::UNITS.lines() {
        if line.len() > 0 {
            if line.starts_with("#") {
                // Dimension
                let dimstr: Vec<&str> = line.split(',').collect();
                let dims: Vec<i8> = dimstr[1..].iter().map(|s| {s.trim_ascii_start().parse::<i8>().unwrap()}).collect();
                dim = Dimension::new(
                    dims[0], dims[1], dims[2], dims[3], dims[4], dims[5], dims[6]
                )
            } else {
                // Unit
                let parts: Vec<&str> = line.split(',').collect();
                let name = parts[0].trim_ascii_start().to_string();
                let abbr = parts[1].trim_ascii_start().to_string();
                let scale = parts[2].trim_ascii_start().parse::<f64>().unwrap();
                let offset = if parts.len() > 3 {
                    parts[3].trim_ascii_start().parse::<f64>().unwrap()
                } else { 0.0 };

                let u = Unit::new(
                    name.clone(),
                    abbr.clone(),
                    dim.clone(),
                    scale,
                    offset,
                );
                ureg.insert(name.clone(), u.clone());
                ureg.insert(abbr.clone(), u);

                for prefix in units::PREFIXES.lines() {
                    let parts: Vec<&str> = prefix.split(',').collect();
                    if parts.len() == 3 {
                        let pname = parts[0].trim_ascii_start().to_string() + &name.clone();
                        let pabbr = parts[1].trim_ascii_start().to_string() + &abbr.clone();
                        let pname2 = parts[1].trim_ascii_start().to_string() + &name.clone();

                        let uprefix = Unit::new(
                            pname.clone(),
                            pabbr.clone(),
                            dim.clone(),
                            scale * parts[2].trim_ascii_start().parse::<f64>().unwrap(),
                            offset,
                        );
                        ureg.insert(pname, uprefix.clone());
                        ureg.insert(pname2, uprefix.clone());
                        ureg.insert(pabbr, uprefix);
                    }
                }

            }
        }
    }
}


pub fn convert(value: f64, unit: &str, tounit: &str) -> Result<f64, Box<dyn Error>> {
    let ureg = unit_registry.lock().unwrap();
    let unit_from = ureg.get(unit).ok_or(UndefinedUnitError{msg: format!("Undefined Unit `{}`", unit)})?;
    let unit_to = ureg.get(tounit).ok_or(UndefinedUnitError{msg: format!("Undefined Unit `{}`", tounit)})?;
    if unit_from.dim.is_compatible(&unit_to.dim) {
        let newval = (value + unit_from.offset) * unit_from.scale / unit_to.scale - unit_to.offset;
        // println!("{} -> {} {}", value, newval, tounit);
        Ok(newval)    
    } else {
        Err(Box::new(DimensionError{msg: format!("Cannot convert `{}` to `{}`", unit, tounit)}))
    }
}

pub fn convert2(value: &str, tounit: &str) -> Result<f64, Box<dyn Error>> {
    let q1 = parse_baseqty(value)?;
    let unit = parse_unit(tounit)?;
    if q1.dim.is_compatible(&unit.dim) {
        let out = (q1.magnitude + unit.offset) / unit.scale;
        // println!("{} -> {} {}", value, out, tounit);
        Ok(out)
    } else {
        Err(Box::new(DimensionError{msg: format!("Cannot convert `{}` to `{}`", value, tounit)}))
    }
}

pub fn convert_base(value: f64, dim: &Dimension, unit: &str, offset: bool) -> Result<f64, Box<dyn Error>> {
    // Convert base value to unit
    let units = parse_unit(&unit)?;
    if dim.is_compatible(&units.dim) {
        let out = if offset {
            (value + units.offset) / units.scale
        } else {
            value / units.scale
        };
        Ok(out)
    } else {
        Err(Box::new(DimensionError{msg: format!("Cannot convert `{:?}` to `{}`", dim, unit)}))
    }
}

pub fn parse_expr_dimension(expr: &str) -> Result<FlatEx<Dimension, DimOpsFactory, DimMatcher>, Box<dyn Error>> {
    // Parse math expression, possibly with units, to a dimension
    let dimex = FlatEx::<Dimension, DimOpsFactory, DimMatcher>::parse(expr)?;
    Ok(dimex)
}

pub fn parse_expr_f64(expr: &str) -> Result<FlatEx<f64>, Box<dyn Error>> {
    // Parse string expression to a FlatEx for f64 in base units
    let re = Regex::new(r"\[.*\]").unwrap();
    let expr_f = re.replace_all(expr, |caps: &Captures| {
        let qstr = caps[0].to_string();
        let qty = parse_baseqty(&qstr).unwrap();
        format!("{}", qty.magnitude)
    });
    let func = exmex::parse::<f64>(&expr_f)?;
    Ok(func)
}


pub fn init() {
    // Initialize the units dictionary
    read_units();
}
