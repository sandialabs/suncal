
pub const SOLAR: &'static str = "# RP-19 Solar Experiment Example
#=================================
# CAL SYSTEM: Deuterium Lamp/Comparator
[[quantity]]
symbol = \"comparator\"
measured = 100.0
units = \"mW\"
typeb = [
{Tolerance = {name = \"NIST\", tolerance=0.0625, confidence=0.9986}},
]

[quantity.utility]
tolerance = {low = 99.75, high = 100.25}

[quantity.interval]
eopr.Observed = 0.95
years = 0.5

[quantity.calibration]
policy = \"Asneeded\"
reliability_model = \"Exponential\"

[quantity.cost]
cal = 800
adjust = 0
repair = 9000
new_uut = 90000
num_uuts = 2
spare_factor = 1
spare_startup = 0
down_cal = 4
down_adj = 0
down_rep = 30
p_use = 1


# =============================
# TEST SYSTEM: Deuterium Lamp
[[quantity]]
symbol = \"lamp\"

measured = 100.0
units = \"mW\"
typeb = [
{Symbol = \"comparator\"},
]

[quantity.utility]
tolerance = {low = 99.0, high = 101.0}

[quantity.interval]
eopr.Observed = 0.9973
years = 0.333333

[quantity.calibration]
policy = \"Asneeded\"
reliability_model = \"Exponential\"

[quantity.cost]
cal = 400
adjust = 0
repair = 7500
new_uut = 75000
num_uuts = 3
spare_factor = 1
spare_startup = 0
down_cal = 2
down_adj = 0
down_rep = 30
p_use = 1


# =============================
# END ITEM: Solar Experiment
[[quantity]]
symbol = \"solar\"

measured = 100.0
units = \"mW\"
typeb = [
{Symbol = \"lamp\"},
]

[quantity.utility]
tolerance = {low = 90.0, high = 110.0}
degrade = {low = 90.0, high = 110.0}
failure = {low = 70.0, high = 130.0}

[quantity.interval]
eopr.Observed = 0.99993
years = 1.0
#target = {Eopr=0.9973}

[quantity.calibration]
policy = \"Asneeded\"
reliability_model = \"Exponential\"
repair = {low = 70.0, high = 130.0}

[quantity.cost]
cal = 156800
adjust = 156800
repair = 50000
new_uut = 250000
num_uuts = 1
spare_factor = 1
spare_startup = 0
down_cal = 30
down_adj = 3
down_rep = 30
p_use = 1
cost_fa = 35000000
cost_fr = 0


#===============================
[settings]
montecarlo = 0
";


pub const CANNON: &'static str = "# RP-19 sCannon Ball Example
[[quantity]]
symbol = \"cannonball\"
measured = 19.7
units = \"cm\"
typeb = [
{Tolerance = {name = \"Calipers\", tolerance=0.004, confidence=0.95}},
{Normal = {name=\"Repeatability\", stddev=0.0648}},
{Normal = {name=\"Resolution\", stddev=0.00029}},
]

[quantity.utility]
tolerance = {low = 19.51, high = 19.81}
degrade = {low = 19.575, high = 19.808}
failure = {low = 19.48, high = 19.827}

[quantity.interval]
eopr.True = 0.95
";

pub const ALTIMETER: &'static str = "
[[quantity]]
symbol = \"adt321\"
measured = 29.25

[[quantity.typeb]]

[quantity.typeb.Tolerance]
tolerance = 0.002
confidence = 0.999
degf = inf

[[quantity.typeb]]

[quantity.typeb.Normal]
stddev = 0.000577
degf = inf

[quantity.utility]
psr = 1.0

[quantity.utility.tolerance]
low = 29.248
high = 29.252

[quantity.utility.guardband]
method = \"None\"
target = 0.02
tur = 4.0

[quantity.utility.guardband.tolerance]
low = 29.248
high = 29.252

[quantity.interval]
years = 0.25

[quantity.interval.eopr]
Observed = 0.9364

[quantity.calibration]
policy = \"Asneeded\"
prob_discard = 0.0
reliability_model = \"RandomWalk\"

[quantity.cost]
cal = 250.0
adjust = 0.0
repair = 1000.0
new_uut = 175000.0
num_uuts = 75.0
spare_factor = 0.1
spare_startup = 0.0
down_cal = 30.0
down_adj = 0.0
down_rep = 30.0
p_use = 1.0
cost_fa = 0.0
cost_fr = 0.0

[[quantity]]
symbol = \"ttu205\"
measured = 29.25

[[quantity.typeb]]
Symbol = \"adt321\"

[[quantity.typeb]]

[quantity.typeb.Normal]
stddev = 0.00153
degf = inf

[[quantity.typeb]]

[quantity.typeb.Normal]
stddev = 0.00085
degf = inf

[quantity.utility]
psr = 1.0

[quantity.utility.tolerance]
low = 29.24
high = 29.26

[quantity.utility.guardband]
method = \"None\"
target = 0.02
tur = 4.0

[quantity.utility.guardband.tolerance]
low = 29.24
high = 29.26

[quantity.interval]
years = 0.833333

[quantity.interval.eopr]
Observed = 0.9775

[quantity.calibration]
policy = \"Asneeded\"
prob_discard = 0.0
reliability_model = \"RandomWalk\"

[quantity.cost]
cal = 25.0
adjust = 0.0
repair = 800.0
new_uut = 75000.0
num_uuts = 75.0
spare_factor = 0.2
spare_startup = 0.0
down_cal = 0.2
down_adj = 0.0
down_rep = 14.0
p_use = 1.0
cost_fa = 0.0
cost_fr = 0.0

[[quantity]]
symbol = \"f18alt\"
measured = 29.25

[[quantity.typeb]]
Symbol = \"ttu205\"

[[quantity.typeb]]

[quantity.typeb.Normal]
stddev = 0.00765
degf = inf

[[quantity.typeb]]

[quantity.typeb.Normal]
stddev = 0.0023
degf = inf

[quantity.utility]
psr = 0.9999

[quantity.utility.tolerance]
low = 29.2
high = 29.3

[quantity.utility.degrade]
low = 29.205
high = 29.295

[quantity.utility.failure]
low = 29.195
high = 29.305

[quantity.utility.guardband]
method = \"None\"
target = 0.02
tur = 4.0

[quantity.utility.guardband.tolerance]
low = 29.2
high = 29.3

[quantity.interval]
years = 0.25

[quantity.interval.eopr]
Observed = 0.95

[quantity.calibration]
policy = \"Asneeded\"
prob_discard = 0.0
reliability_model = \"RandomWalk\"

[quantity.cost]
cal = 25.0
adjust = 0.0
repair = 300.0
new_uut = 3500.0
num_uuts = 400.0
spare_factor = 1.0
spare_startup = 0.0
down_cal = 0.5
down_adj = 0.0
down_rep = 7.0
p_use = 0.02
cost_fa = 32000000.0
cost_fr = 0.0

[settings]
confidence = 0.9545
montecarlo = 0
";
