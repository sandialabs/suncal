// name, abbr, scale
pub static PREFIXES: &'static str = "
yotta,Y,1E24
zetta,Z,1E21
exa,E,1E18
peta,P,1E15
tera,T,1E12
giga,G,1E9
mega,M,1E6
kilo,k,1E3
hecto,h,1E2
deka,da,1E1
deci,d,1E-1
centi,c,1E-2
milli,m,1E-3
micro,u,1E-6
micro,µ,1E-6
micro,μ,1E-6
nano,n,1E-9
pico,p,1E-12
femto,f,1E-15
atto,a,1E-18
zepto,z,1E-21
yocto,y,1E-24
";


// #DIMENSION length, mass, time, temp, current, intensity, amount
// name, abbr, scale, [offset]
// Note gram is the base unit here, so quantities with a
// mass dimensionality need to compensate (Newton, Joule, etc.)
pub static UNITS: &'static str = "
#LENGTH, 1, 0, 0, 0, 0, 0, 0
meter, m, 1
angstrom, Å, 1E-10
micron, μm, 1E-6
lightyear, ly, 9.476073E15
parsec, pc, 3.085678E16
astronomical_unit, ua, 1.495979E11
fathom, _, 1.828804
fermi, _, 1E-15
inch, in, .0254
foot, ft, .3048
feet, ft, .3048
yard, yd, 9.144E-1
mile, mi, 1.609344E3
nautical_mile, _, 1.852E3

#ANGLE, 0, 0, 0, 0, 0, 0, 0
radian, rad, 1
degree, °, 1.745329E-2
mil, _, 9.817477E-4
revolution, r, 6.283185
arcsecond, arcsec, 4.848136E-6
arcminute, arcmin, 2.908882E-4

#ACCELERATION, 1, 0, -2, 0, 0, 0, 0
gravity, g0, 9.80665

#AREA, 2, 0, 0, 0, 0, 0, 0
acre, _, 4.046873E3
are, a, 1E2
barn, barn, 1E-28
hectare, ha, 1E4

#CURRENT, 0, 0, 0, 0, 1, 0, 0
ampere, A, 1
amp, _, 1

#RESISTANCE, 2, 1, -3, 0, -2, 0, 0
ohm, Ω, 1E3

#CONDUCTANCE, -2, -1, 3, 0, 2, 0, 0
siemens, S, 1E-3

#CAPACITANCE, -2, -1, 4, 0, 2, 0, 0
farad, F, 1E-3

#CHARGE, 0, 0, 1, 0, 1, 0, 0, 0
coulomb, C, 1E3
electron_charge, e, 1.602176634E-19

#POTENTIAL, 2, 1, -3, 0, -1, 0, 0
volt, V, 1E3

#POWER, 2, 1, -3, 0, 0, 0, 0
watt, W, 1E3
horsepower, hp, 7.456999E2

#ENERGY, 2, 1, -2, 0, 0, 0, 0
joule, J, 1E3
british_thermal_unit, btu, 1.055056E6
calorie, cal, 4.184E3
international_calorie, cal_it, 4.1868E3
electronvolt, eV, 1.602176E-16
erg, _, 1E-4
therm, thm, 1.055056E11

#MAGNETICFLUX, 2, 1, -2, 0, -1, 0, 0
weber, Wb, 1E3
maxwell, Mx, 1E-5
unit_pole, _, 1.256637E-4

#MAGNETICFLUXDENSITY, 0, 1, -2, 0, -1, 0, 0
tesla, T, 1E3
gauss, Gs, 1E-1

#FORCE, 1, 1, -2, 0, 0, 0, 0
newton, N, 1E3
dyne, dyn, 1E-2
kilogram_force, kgf, 9.80665E3
kilopond, kp, 9.80665E3
kip, _, 4.448222E6
ounce_force, ozf, 2.780139E2
poundal, _, 1.382550E2
pound_force, lbf, 4.448222E3
ton_force, ton_force, 8.896443E6

#FREQUENCY, 0, 0, -1, 0, 0, 0, 0
hertz, Hz, 1

#ILLUMINANCE, -2, 0, 0, 0, 0, 1, 0
lux, lx, 1
footcandle, _, 1.550003E3
footlambert, _, 3.426259
lambert, _, 3.183099E3
phot, ph, 1E4

#MASS, 0, 1, 0, 0, 0, 0, 0
gram, g, 1
carat, _, 2E-1
grain, gr, 6.479891E-2
ounce, oz, 2.834952E1
pennyweight, dwt, 1.555174
pound, lb, 4.535924E2
slug, _, 1.459390E4
ton, _, 2.916667E1
ton_long, _, 1.016047E6
metric_ton, ton_metric, 1E6
ton_short, _, 9.071847E5

#PRESSURE, -1, 1, -2, 0, 0, 0, 0
pascal, Pa, 1E3
atmosphere, atm, 1.01325E8
bar, _, 1E8
centimeter_mercury, cmHg, 1.33322E6
centimeter_water, cmH2O, 9.80638E4
foot_mercury, ftHg, 4.063666E7
foot_water, ftH2O, 2.98898E6
inch_mercury, inHg, 3.38638E6
inch_water, inH2O, 2.49082E5
pound_force_per_square_inch, psi, 6.894757E6
torr, _, 1.333224E5

#RADIOLOGY, 0, 0, -1, 0, 0, 0, 0
becquerel, Bq, 1
curie, Ci, 3.7E10

#RADIATIONDOSE, 2, 0, -2, 0, 0, 0, 0
sievert, Sv, 1
rem, _, 1E-2

#RADEXPOSURE, 0, -1, 1, 0, 1, 0, 0
roentgen, R, 2.58E-7

#TEMPERATURE, 0, 0, 0, 1, 0, 0, 0
kelvin, K, 1, 0
Celsius, °C, 1, 273.15
degC, _, 1, 273.15
Fahrenheit, °F, 0.555555555555555, 459.67
degF, _, 0.555555555555555, 459.67
Rankine, °R, 0.555555555555555, 0
degR, _, 0.555555555555555, 0

#TEMPERATUREINTERVAL, 0, 0, 0, 1, 0, 0, 0
delta_degC, Δ°C, 1
delta_degF, Δ°F, 0.555555555555555
delta_degR, Δ°R, 0.555555555555555

#TIME, 0, 0, 1, 0, 0, 0, 0
second, s, 1
sec, _, 1
minute, min, 60
hour, hr, 3600
day, d, 8.64E4
year, yr, 3.1536E7
shake, shake, 1E-8

#VELOCITY, 1, 0, -1, 0, 0, 0, 0
foot_per_minute, fpm, 5.08E-3
foot_per_hour, fph, 8.466667E-5
foot_per_second, fps, 3.048E-1
inch_per_second, inch_per_second, 2.54E-2
knot, kt, 5.144444E-1
mile_per_hour, mph, 4.4704E-1

#VOLUME, 3, 0, 0, 0, 0, 0, 0
barrel, _, 1.589873E-1
bushel, bu, 3.523907E-2
cord, _, 3.624556
cup, _, 2.365882E-4
fluid_ounce, floz, 2.957353E-5
gallon, gal, 3.785412E-3
liter, L, 1E-3
litre, _, 1E-3
peck, pk, 8.809768E-3
pint, pt, 5.506105E-4
quart, qt, 1.101221E-3
stere, st, 1
tablespoon, _, 1.478676E-5
teaspoon, _, 4.928922E-6
hogshead, _, 0.23848094

#ANGULARVELOCITY, 0, 0, -1, 0, 0, 0, 0
revolution_per_minute, rpm, 1.047198E-1
";

