import pytest

import numpy as np
import scipy.stats as stats
import scipy
import sympy
from dateutil.parser import parse

import psluncert
from psluncert import uarray
from psluncert import curvefit
from psluncert import anova


# Made up data, hard-coded here for consistent testing
x = np.array([ 10.        ,  11.66666667,  13.33333333,  15.        ,
    16.66666667,  18.33333333,  20.        ,  21.66666667,
    23.33333333,  25.        ,  26.66666667,  28.33333333,
    30.        ,  31.66666667,  33.33333333,  35.        ,
    36.66666667,  38.33333333,  40.        ,  41.66666667,
    43.33333333,  45.        ,  46.66666667,  48.33333333,  50.        ])
y = np.array([  79.07164419,   71.08765462,  100.78498117,  110.21800243,
     96.17154968,  112.54134419,  118.70909255,  132.55642535,
    143.40133783,  151.2440317 ,  166.38788909,  158.58619681,
    179.77777455,  172.53574337,  192.77214916,  198.89573162,
    202.6281377 ,  201.13723674,  212.80124094,  255.44771032,
    244.13505509,  238.42752461,  255.72531442,  267.466498  ,
    276.16736904])

def test_linefit1():
    ''' Test linear fitting with no uncertainties in x or y. Tests slope/intercept, uslope/uintercept,
        Syx, u_conf and u_pred for linear case against published data (Natrella Stats Handbook)
    '''
    # Line fit example from Natrella - Experimental Statistics Handbook 91,
    # section 5-4.1 (Young's Modulus vs Temperature Data)
    x = np.array([30,100,200,300,400,500,600,700,800,900,1000,1100,1200,1300,1400,1500])
    y = np.array([4642,4612,4565,4513,4476,4433,4389,4347,4303,4251,4201,4140,4100,4073,4024,3999])
    np.random.seed(100)
    arr = curvefit.Array(x, y)  # No uncertainty in x, y in this example
    fit = curvefit.CurveFit(arr)
    fit.calculate(gum=True, mc=True, lsq=True)

    # Check fit results, and uncertainties of slope, intercept, Syx
    # Values from Natrella Worksheet 5-4.1
    assert np.isclose(fit.out.lsq.mean[0], -0.44985482)
    assert np.isclose(fit.out.lsq.mean[1], 4654.9846)
    assert np.isclose(fit.out.lsq.uncert[0], np.sqrt(0.000025649045))
    assert np.isclose(fit.out.lsq.uncert[1], np.sqrt(19.879452))
    assert np.isclose(fit.out.lsq.properties['Syx'], 9.277617)

    # Check conf and pred bands. Relax tolerance due to round-off error
    assert np.isclose(fit.out.lsq.u_conf(1200), np.sqrt(10.53), rtol=.01)   # From page 5-15, var y'c
    assert np.isclose(fit.out.lsq.u_pred(750), np.sqrt(91.45), rtol=.01)   # From page 5-15, var Y'c

    # After version 1.1, GUM and MC should estimate uy same as LSQ and get the same answer.
    assert np.allclose(fit.out.lsq.uncert, fit.out.gum.uncert, rtol=.01)
    assert np.allclose(fit.out.lsq.uncert, fit.out.mc.uncert, rtol=.01)

def test_linefit2():
    ''' Straight line fit, uncertainty in y but not x. Verify GUM/MC/lsq methods match.
        Checking a, b, ua, ub, Syx, u_conf, u_pred, cov, cor
    '''
    np.random.seed(100)
    # Use fake data from above
    # No x uncertainty (use linefit)
    arr = curvefit.Array(x, y, uy=10)
    f = curvefit.CurveFit(arr, func='line')
    f.calculate(gum=True, mc=True, lsq=True)

    assert np.allclose(f.out.lsq.mean, f.out.gum.mean)
    assert np.allclose(f.out.lsq.mean, f.out.mc.mean, rtol=1E-2)  # MC - relax tolerance
    assert np.allclose(f.out.lsq.uncert, f.out.gum.uncert)
    assert np.allclose(f.out.lsq.uncert, f.out.mc.uncert, rtol=1E-2)
    assert np.isclose(f.out.lsq.properties['Syx'], f.out.gum.properties['Syx'])
    assert np.isclose(f.out.lsq.properties['Syx'], f.out.mc.properties['Syx'], rtol=.01)

    assert np.isclose(f.out.lsq.u_conf(20), f.out.gum.u_conf(20))
    assert np.isclose(f.out.lsq.u_conf(20), f.out.mc.u_conf(20), rtol=.01)
    assert np.isclose(f.out.lsq.u_pred(20), f.out.gum.u_pred(20))
    assert np.isclose(f.out.lsq.u_pred(20), f.out.mc.u_pred(20), rtol=.01)

    assert np.allclose(f.out.lsq.properties['cov'], f.out.gum.properties['cov'])
    assert np.allclose(f.out.lsq.properties['cov'], f.out.mc.properties['cov'], rtol=.05)
    assert np.allclose(f.out.lsq.properties['cor'], f.out.gum.properties['cor'])
    assert np.allclose(f.out.lsq.properties['cor'], f.out.mc.properties['cor'], rtol=.05)

    # Repeat with uncertainty in x direction (use linefitYork)
    np.random.seed(100)
    arr = curvefit.Array(x, y, uy=10, ux=0.5)
    f = curvefit.CurveFit(arr, func='line')
    f.calculate(samples=5000, gum=True, mc=True, lsq=True)

    assert np.allclose(f.out.lsq.mean, f.out.gum.mean, rtol=1E-2)
    assert np.allclose(f.out.lsq.mean, f.out.mc.mean, rtol=1E-1)
    assert np.allclose(f.out.lsq.uncert, f.out.gum.uncert, rtol=1E-1)
    assert np.allclose(f.out.lsq.uncert, f.out.mc.uncert, rtol=1E-1)
    assert np.isclose(f.out.lsq.properties['Syx'], f.out.gum.properties['Syx'], rtol=.01)
    assert np.isclose(f.out.lsq.properties['Syx'], f.out.mc.properties['Syx'], rtol=.01)


def test_linefitgum():
    ''' Test least squares based on GUM example H.3. Calibration of a thermometer example.
        Tests curve fitting and uncertainty.
        Note GUM uses the GUM formula on y = mx + b to compute uncertainty, where PSLUC
        uses prediction band, but they give the same results.
    '''
    # GUM Table H.6 columns 2 and 3
    T = np.array([21.521,22.012,22.512,23.003,23.507,23.999,24.513,25.002,25.503,26.010,26.511])
    C = - np.array([.171,.169,.166,.159,.164,.165,.156,.157,.159,.161,.160])
    T0 = 20
    arr = curvefit.Array(T-T0, C, ux=0, uy=0)
    fit = curvefit.CurveFit(arr)
    out = fit.calc_LSQ()
    # Results in H.3.3 - within reported digits
    assert np.isclose(out.mean[0], .00218, atol=.00001)
    assert np.isclose(out.mean[1], -.1712, atol=.0001)
    assert np.isclose(out.uncert[0], .00067, atol=.00001)
    assert np.isclose(out.uncert[1], .0029, atol=.0001)
    assert np.isclose(out.properties['cor'][0,1], -.930, atol=.001)
    assert np.isclose(out.properties['Syx'], .0035, atol=.0001)

    # H.3.4 - predicted value and confidence
    assert np.isclose(out.y(30-T0), -.1494, atol=.0001)
    assert np.isclose(out.u_conf(30-T0), .0041, atol=.0001)

    # Check report for correctly formatted values
    rpt = out.report_confpred_xval(10, k=1).string
    assert '-0.149' in rpt
    assert '0.0041' in rpt

def test_curvefit():
    ''' Curve fit with no uncertainty in x or y. Uses scipy.optimize.curve_fit. '''
    # Data from S. Glantz, B. Slinker, Applied Regression & analysis of Variance, 2nd edition. McGraw Hill, 2001.
    # Data points copied from Table C-25. Fit formula is equation 11-3, with results shown in Figure 11-7.
    y = np.array([
    17.3, 15.9, 13.9, 11.6, 5.0, 8.2, 6.0, .5, .1, 3.3, 3.6, 9, 12.5, 9, 13.5, 10.7, 11.8, 17.4,
    12.8, 14.7, 13., 6.7, 2.5, 9, 4.9, 1.1, .1, 1.1, 4.7, 5.6, 10.7, 13.4, 17.3, 11.8, 20.2,
    19.3, 18.9, 15.7, 4.9, 9, 6.7, .9, 1.3, 0, 0, 2, 6, 3.8, 10.3, 16.2])
    x = np.arange(1, len(y)+1)

    def cosmodel(x, T, Imax):
        ''' Cosine function model '''
        return Imax * (np.cos(np.pi*2*x/T) + 1)/2

    np.random.seed(100)
    arr = curvefit.Array(x, y)  # No uy or ux in this data
    f = curvefit.CurveFit(arr, cosmodel, p0=(17., 18.))
    f.calculate(gum=True, mc=True, lsq=True)

    # Gauss-Newton method. See figure 11-7. Tolerances to match sigfigs in book.
    assert np.isclose(f.out.lsq.mean[0], 17.6, atol=.1)
    assert np.isclose(f.out.lsq.mean[1], 17.4, atol=.1)
    assert np.isclose(f.out.gum.mean[0], 17.6, atol=.1)
    assert np.isclose(f.out.gum.mean[1], 17.4, atol=.1)
    assert np.isclose(f.out.mc.mean[0], 17.6, atol=.1)
    assert np.isclose(f.out.mc.mean[1], 17.4, atol=.1)
    assert np.isclose(sum(f.out.lsq.properties['resids']**2), 356, atol=1)

    assert np.isclose(f.out.lsq.uncert[0], .101838)  # Sigmas shown in fig 11-7 as "asymptotic std error"
    assert np.isclose(f.out.lsq.uncert[1], .65815)
    assert np.allclose(f.out.lsq.properties['cor'], np.array([[1, .0772],[.0772, 1]]), atol=.0001)
    # GUM and MC will not have these sigmas as there's 0 ux, uy, so the GUM has 0 gradient and MC repeats the same point

    #-----
    # Now, add an uncertainty in Y and compare the three methods (not part of book, but testing curve_fit with sigma)
    np.random.seed(100)
    arr = curvefit.Array(x, y, uy=.5)
    f = curvefit.CurveFit(arr, cosmodel, p0=(17., 18.))
    f.calculate(gum=True, mc=True, lsq=True)
    assert np.allclose(f.out.lsq.mean, f.out.gum.mean, atol=.01)  # Reported sigfigs
    assert np.allclose(f.out.lsq.mean, f.out.mc.mean, atol=.01)
    assert np.allclose(f.out.lsq.uncert, f.out.gum.uncert, atol=.01)
    assert np.allclose(f.out.lsq.uncert, f.out.mc.uncert, atol=.01)

    # And with uncertainty in x. GUM will quit working here, but it exercises ODR.
    np.random.seed(100)
    arr = curvefit.Array(x, y, uy=.5, ux=.1)
    f = curvefit.CurveFit(arr, cosmodel, p0=(17., 18.))
    f.calculate(gum=True, mc=True, lsq=True)
    assert np.allclose(f.out.lsq.mean, f.out.mc.mean, atol=.5)
    assert np.allclose(f.out.lsq.uncert, f.out.mc.uncert, atol=.005)


def test_curvefitcustom():
    ''' Test curvefit with custom model '''
    # Leak standard decay example data. Decays to 0, so use custom exponential with c=0
    x = np.array([1.0,380.0,794.0,1247.0,1673.0,2031.0,2747.0])
    y = np.array([6.635e-08,6.531e-08,6.354e-08,6.255e-08,6.212e-08,5.97e-08,5.907e-08])
    arr = curvefit.Array(x, y)
    fit = curvefit.CurveFit(arr, 'a*exp(-b*x)', p0=(6.5E-8, 4.5E-5))
    out = fit.calc_LSQ()
    assert np.isclose(out.mean[0], 6.621E-8, atol=.001E-8)
    assert np.isclose(out.mean[1], 4.404E-5, atol=.001E-5)


def test_curvefitdate():
    ''' Test curve fit where x values are dates '''
    x = ['2-Jun-2013', '4-Sep-2014', '3-Nov-2015', '21-Mar-2016', '22-Aug-2017', '3-Feb-2018']
    y = np.array([.5, .8, .9, 1.2, 1.25, 1.4])
    xdate = np.array([parse(f).toordinal() for f in x])  # Convert to ordinal
    arr = curvefit.Array(xdate, y)
    fit = curvefit.CurveFit(arr, xdates=True)
    out = fit.calculate()
    r = out.lsq.report_confpred_xval('4-Oct-2018')
    assert '4-Oct-2018' in r.string
    assert '736917' not in r.string  # This is 4-Oct-2018 in ordinal date format as used internally.


def test_ttest(capsys):
    ''' Test t-statistic testing '''
    # Data from S. Glantz, B. Slinker, Applied Regression & analysis of Variance, 2nd edition. McGraw Hill, 2001.
    # Table 2-1 height and weight
    h = np.array([31,32,33,34,35,35,40,41,42,46])
    w = np.array([7.8,8.3,7.6,9.1,9.6,9.8,11.8,12.1,14.7,13.0])
    arr = curvefit.Array(h, w)
    fit = curvefit.CurveFit(arr)
    lsqout = fit.calc_LSQ()

    # Verify slope/intercept and uncertainties (some tolerance due to rounding)
    assert np.allclose(lsqout.mean, np.array([.44, -6.0]), rtol=.01)    # pg 19
    assert np.allclose(lsqout.uncert, np.array([.064, 2.4]), rtol=.01)  # Eq 2.9, pg 24

    # Verify t-test values (pg 26)
    assert lsqout.test_t(verbose=True, conf=.999)   # test should pass
    out, err = capsys.readouterr()  # Get verbose printout
    assert '6.90' in out   # t for data
    assert '5.04' in out   # t for .999 confidence

    assert lsqout.test_t_range(conf=.95, verbose=True)  # Next test should pass
    out, err = capsys.readouterr()  # Get verbose printout
    assert '.29' in out  # Lower limit (pg 27)
    assert '.59' in out  # Upper limit (pg 27)


def test_arraythresh():
    ''' Test threshold crossing '''
    x = np.linspace(0,20)
    y = 5*np.exp(-x/5)
    arr = uarray.Array(x, y, uy=.25)
    f = uarray.ArrayThresh(arr, thresh=2, edge='first')
    f.calculate(gum=True, mc=True, lsq=True)

    expected = -5*np.log(2/5)  # Solve analytically
    assert np.isclose(f.out.lsq.mean, expected, atol=.01)  # lsq uses discrete interpolation
    assert np.isclose(f.out.gum.mean, expected, atol=.01)
    assert np.isclose(f.out.mc.mean, expected, atol=.2)

    assert np.isclose(f.out.lsq.uncert, f.out.gum.uncert, atol=.3)  # Different methods will vary a bit
    assert np.isclose(f.out.lsq.uncert, f.out.mc.uncert, atol=.3)


def test_linefitcalc():
    ''' Test CurveFitParam usage as function in calculator '''
    arr = curvefit.Array(x, y, uy=10)
    fit = curvefit.CurveFit(arr)
    slope = curvefit.CurveFitParam(fit, pidx=0, name='slope')

    u = psluncert.UncertaintyCalc(samples=500)
    u.set_function(slope)
    u.set_function('slope + 10', name='s10')
    u.calculate(gum=True, mc=True, lsq=True)

    assert np.isclose(u.out.get_output(fidx=1, method='gum').mean, u.out.get_output(fidx=0, method='gum').mean + 10)
    # Only uncertainty is from slope, so u(slope) == u(slope+10)
    assert np.isclose(u.out.get_output(fidx=0, method='gum').uncert, u.out.get_output(fidx=1, method='gum').uncert)


def test_arraygroup():
    ''' Test ArrayGrouped class - Make sure it takes x and 2D y data and generates
        the correct Array class x, y, and uy.
    '''
    # ArrayGrouped from array data, all same length
    np.random.seed(100)
    x = np.arange(10)
    ynom = 3*x + 2
    samples = 8
    y = np.random.normal(loc=ynom, scale=.4, size=(samples, len(ynom))).T
    arr = anova.ArrayGrouped(x, y).to_array()

    assert np.allclose(arr.x, x)
    assert np.allclose(arr.y, y.mean(axis=1))
    assert np.allclose(arr.uy, y.std(axis=1, ddof=1))


def test_anova():
    ''' Test analysis of variance in ArrayGrouped class '''
    # Data from Dilip Shaw's example Excel spreadsheet at NCSLI ABQ Section Meeting 6/4/2018
    x1 = [0.999156418,0.999474238,0.999339856,1.000160754,1.000744245,1.000737791,1.000495481,0.9995736,1.000723728,1.00060071]
    x2 = [1.000349612,0.999883249,0.999050897,1.00017474,0.999602596,1.000824172,0.999356241,0.999806955,1.000643518,1.000444615]
    x3 = [0.999757491,0.999817772,1.000194482,0.999679647,1.000937242,1.000470251,0.999027869,0.999911651,0.999590441,1.000451023]
    x4 = [0.999719098,0.99964781,1.000822782,0.999409639,0.999109266,1.000785761,0.999307039,1.000469276,1.000654864,1.000987112]

    g = anova.ArrayGrouped()
    g.add_group(x1)
    g.add_group(x2)
    g.add_group(x3)
    g.add_group(x4)
    result = g.anova()
    assert np.isclose(result.F, .0850416)
    assert np.isclose(result.Fcrit, 2.866266)
    assert np.isclose(result.P, .96777478)
    assert np.isclose(g.reproducibility(), 57.6E-6, atol=.01E-5)
    assert np.isclose(g.std_pooled(), 624.4E-6, atol=.1E-6)
    assert np.isclose(g.std_all(), 602.1E-6, atol=.1E-6)
    assert np.allclose(g.group_df(), [9,9,9,9])
    assert np.allclose(g.group_mean(), [1.000100682,1.000013659,0.999983787,1.000091265])
    assert np.allclose(g.group_std(), [0.000646229,0.00057407,0.000542976,0.000719539])
    assert g.df_reproducibility() == 3
    assert g.df_pooled() == 36
    assert g.df_all() == 39

    # Also try groups of uneven lengths, use excel anova function to verify
    g = anova.ArrayGrouped()
    g.add_group(x1)
    g.add_group(x2[:8])
    g.add_group(x3[:9])
    g.add_group(x4[:6])
    assert np.allclose(g.group_df(), [9,7,8,5])
    assert g.df_reproducibility() == 3
    assert g.df_pooled() == 29
    result = g.anova()
    assert np.isclose(result.F, .23239)
    assert np.isclose(result.Fcrit, 2.93403)
    assert np.isclose(result.P, .873061)


def test_anovameans():
    ''' Test ANOVA when only means and standard deviations of each group are given. Use GUM H.5 data. '''
    x = np.arange(1, 11)
    y = np.array([10.000172, 10.000116, 10.000013, 10.000144, 10.000106, 10.000031, 10.000060, 10.000125, 10.000163, 10.000041])
    ystd = np.array([60, 77, 111, 101, 67, 93, 80, 73, 88, 86]) * 1E-6
    nmeas = np.full(len(y), 5)
    a = anova.ArrayGroupedSummary(x, y, ystd, nmeas)
    a.calculate()
    assert np.isclose(a.grand_mean(), 10.000097, atol=.0000005)  # From GUM Table H.9
    assert np.isclose(a.reproducibility(), 57E-6, atol=.5E-6)
    assert np.isclose(a.std_pooled(), 85E-6, atol=.5E-6)
    result = a.anova()
    assert np.isclose(result.F, 2.25, atol=.02)  # Equation H.27
    assert a.ngroups() == 10

    # Data from Glantz Table 7-4, results in table 7-5
    x = ['Healthy', 'Nonmelancholic', 'Melancholoc']  # Check with x as strings too
    y = np.array([9.2, 10.7, 13.5])
    ystd = np.array([2.9, 2.8, 4.7])
    nmeas = np.array([16, 22, 18])
    a = anova.ArrayGroupedSummary(x, y, ystd, nmeas)
    result = a.anova()
    assert np.isclose(result.SSbet, 164.67, atol=.01)
    assert np.isclose(result.SSwit, 666.02, atol=1)  # Rounding error in table, large atol
    assert np.isclose(result.MSbet, 82.3, atol=.1)
    assert np.isclose(result.MSwit, 12.5, atol=.1)
    assert np.isclose(result.F, 6.612, atol=.1)
    assert np.isclose(result.P, .003, atol=.001)


def test_mcmc1():
    ''' Test Markov-Chain Monte Carlo, fit a + b*x**2.
        Data manually extracted from figure 1, expected results at bottom of page 236 for b parameter
        (1.21 +/- 0.18).

        References:
        [1] C. Elster, B. Toman. "Bayesian uncertainty analysis for a regression model versus
            application of GUM Supplement 1 to the least-squares estimate", Metrologia 48 (2011) 233-240.
    '''
    np.random.seed(100)
    # Figure 1 data
    fig1 = np.array([[0.0, 0.9823529],[0.1, 0.6254902],[0.2, 0.95490193],[0.3, 0.7352941],[0.4, 1.327451],
    [0.5, 1.0764706],[0.6, 1.382353],[0.7, 1.382353],[0.8, 1.7039216],[0.9, 1.6921569],[1.0, 2.1039217],])

    def sqfunc(x, a, b):
        return a + b*x**2

    arr = curvefit.Array(fig1[:,0], fig1[:,1], uy=0.2)
    fit = curvefit.CurveFit(arr, sqfunc, p0=(1,1))
    out = fit.calculate(mcmc=True, gum=True, mc=True, lsq=True)
    assert np.isclose(out.mcmc.mean[1], 1.21, rtol=.01, atol=.01)
    assert np.isclose(out.mcmc.uncert[1], 0.18, rtol=.01, atol=.01)

    # In this problem, the other methods are comparable.
    assert np.isclose(out.mc.mean[1], 1.21, rtol=.01, atol=.01)
    assert np.isclose(out.mc.mean[1], 1.21, rtol=.01, atol=.01)
    assert np.isclose(out.lsq.mean[1], 1.21, rtol=.01, atol=.01)
    assert np.isclose(out.lsq.mean[1], 1.21, rtol=.01, atol=.01)


def test_mcmc2():
    ''' Test Markov-Chain Monte Carlo, fit a + b*x**2. Data from Figure 2, using residuals as uy.
        Expected value 0.94 +/- 0.19 just below fig3 on page 237.

        References:
        [1] C. Elster, B. Toman. "Bayesian uncertainty analysis for a regression model versus
            application of GUM Supplement 1 to the least-squares estimate", Metrologia 48 (2011) 233-240.
    '''
    np.random.seed(33233)
    # Figure 2 data
    fig2 = np.array([[0.0, 0.9823529],[0.1, 0.6254902],[0.1, 1.1583828],[0.2, 0.95490193],[0.3, 0.7352941],
                     [0.3, 1.4310395],[0.4, 1.3274510],[0.5, 1.0764706],[0.6, 1.31788900],[0.6, 1.3823530],
                     [0.7, 1.3799986],[0.8, 1.7039216],[0.9, 1.6921569],[0.9, 1.3818178],[1.0, 2.10392170]])
    def sqfunc(x, a, b):
        return a + b*x**2

    arr = curvefit.Array(fig2[:,0], fig2[:,1])
    fit = curvefit.CurveFit(arr, sqfunc, p0=(1,1))
    out = fit.calc_MCMC()
    unc = out.uncert[1] * stats.t.ppf(1-(1-.68)/2, df=out.degf)  # Account for t-distribution
    assert np.isclose(out.mean[1], 0.94, rtol=.01, atol=.01)
    assert np.isclose(unc, 0.19, rtol=.01, atol=.02)

    # Compare against monte-carlo using pooled variance
    pool = np.sqrt((np.var(fig2[1:3,1]) + np.var(fig2[4:6,1]) + np.var(fig2[8:10,1]) + np.var(fig2[12:14,1])))
    arr = curvefit.Array(fig2[:,0], fig2[:,1], uy=pool)
    fit = curvefit.CurveFit(arr, sqfunc, p0=(1,1))
    out = fit.calc_MC(samples=5000)
    assert np.isclose(out.mean[1], 0.94, rtol=.01, atol=.01)
    assert np.isclose(out.uncert[1], 0.37, rtol=.01, atol=.01)


def test_mcmc3():
    ''' Test Markov-Chain Monte Carlo by comparing results of calculator with results
        computed using published code from [1] (in supplemental info). To generate the
        data set (mcmc/K350_trace1000.txt'), the published code was modified to also
        fit background as parameter instead of fitting and subtracting it out. Data
        samples produced by [1] were saved to csv file.

        [1] T. Iamsasri, et. al. "A Bayesian approach to modeling diffraction profiles
            and application to ferroelectric materials."
            J. Appl. Crystallography (2017). 50, 211-220.
    '''
    np.random.seed(1234)

    # Fit to a double gaussian with background offset
    def gaussian(x, N1, x0, f):
        return N1 * np.sqrt(4*np.log(2)/np.pi)/f * np.exp(-4*np.log(2) * ((x-x0)/f)**2)

    def gaussiandouble(x, N1, x01, f1, N2, x02, f2, bg):
        return gaussian(x, N1, x01, f1) + gaussian(x, N2, x02, f2) + bg

    # Load the measurement data
    data = np.genfromtxt('test/mcmc/xrd.csv', skip_header=1)
    x = data[:,0]
    y = data[:,1]

    # Extract first peak from data set
    rng1 = (x>1.6) & (x<1.75)
    x1 = x[rng1]
    y1 = y[rng1]

    # Set up reasonable initial guess and bounds on parameters
    args =   (15,   1.64, .02,  26,   1.67, .02,    205)
    bounds = [(1,   1.62, .005, .01,  1.65, .005,   0),
              (100, 1.65, .5,   1000, 1.69, .5,     1000)]

    # Run the MCMC calculator
    arr = curvefit.Array(x1, y1)
    fit = curvefit.CurveFit(arr, gaussiandouble, p0=args, bounds=bounds)
    out = fit.calc_MCMC(samples=10000, burnin=.2)
    samples = out.samples

    # Load data processed by published code
    xrddat = np.genfromtxt('test/mcmc/K350_trace1000.txt')[20:,:]  # skipping first 20 since not broken in there

    # Verify uncertainty within 1%
    xrdpct = xrddat[:,:-2].std(ddof=1, axis=0) / xrddat[:,:-2].mean(axis=0)
    pslpct = out.uncert/out.mean
    assert np.allclose(pslpct, xrdpct, rtol=.01, atol=.01)


def test_uconf():
    ''' Test confidence/prediction band calculations by comparing the linear upred, uconf formulas with
        the nonlinear expression (ref: Christopher Cox and Guangqin Ma. Asymptotic Confidence Bands for Generalized
        Nonlinear Regression Models. Biometrics Vol. 51, No. 1 (March 1995) pp 142-150.).
    '''
    # Textbook equations for linear regression bands. Nonlinear implementation should give
    # same results when model is linear.
    def uconf(x, xdata, Syx, sigb):
        N = len(xdata)
        xbar = xdata.mean()
        return Syx * np.sqrt(1/N + (x-xbar)**2 * (sigb/Syx)**2)

    def upred(x, xdata, Syx, sigb):
        N = len(xdata)
        xbar = xdata.mean()
        return Syx * np.sqrt(1 + 1/N + (x-xbar)**2 * (sigb/Syx)**2)

    np.random.seed(9875)

    # ux = 0, uy = 0
    x = np.linspace(100, 200, num=15)
    y = .01*(x-100)**2 + np.random.normal(loc=0, scale=4, size=len(x))
    arr = curvefit.Array(x, y)  # NOT providing uy, should use residuals
    fit = curvefit.CurveFit(arr)
    out = fit.calc_LSQ()
    xx = np.linspace(100,200)
    conf_grad = out.u_conf(xx)
    conf_lin = uconf(xx, out.properties['data'][0], out.properties['Syx'], out.properties['sigmas'][0])
    assert np.allclose(conf_grad, conf_lin, rtol=.001)
    pred_grad = out.u_pred(xx)
    pred_lin = upred(xx, out.properties['data'][0], out.properties['Syx'], out.properties['sigmas'][0])
    assert np.allclose(pred_grad, pred_lin, rtol=.001)

    outgum = fit.calc_GUM()
    outmc = fit.calc_MC()
    assert np.allclose(outgum.u_conf(xx), uconf(xx, outgum.properties['data'][0], outgum.properties['Syx'], outgum.properties['sigmas'][0]), rtol=.01)
    assert np.allclose(outmc.u_conf(xx), uconf(xx, outmc.properties['data'][0], outmc.properties['Syx'], outmc.properties['sigmas'][0]), rtol=.05)

    # ux = 0, uy > 0
    x = np.linspace(0, 5, num=8)
    y = -2*x + np.random.normal(loc=0, scale=1.5, size=len(x))
    arr = curvefit.Array(x, y, uy=1.5)
    fit = curvefit.CurveFit(arr)
    out = fit.calc_LSQ()
    xx = np.linspace(0,5)
    conf_grad = out.u_conf(xx)
    conf_lin = uconf(xx, out.properties['data'][0], 1.5, out.properties['sigmas'][0])
    assert np.allclose(conf_grad, conf_lin, rtol=.001)
    pred_grad = out.u_pred(xx, mode='sigy')
    pred_lin = upred(xx, out.properties['data'][0], 1.5, out.properties['sigmas'][0])
    assert np.allclose(pred_grad, pred_lin, rtol=.001)

    # ux > 0, uy > 0. Not expected to match exactly. Use high tolerance for assertion.
    x = np.linspace(0, 5, num=8)
    y = -2*x + np.random.normal(loc=0, scale=1.5, size=len(x))
    arr = curvefit.Array(x, y, ux=0.5, uy=1.5)
    fit = curvefit.CurveFit(arr)
    out = fit.calc_LSQ()
    xx = np.linspace(0,5)
    conf_grad = out.u_conf(xx)
    conf_lin = uconf(xx, out.properties['data'][0], 1.5, out.properties['sigmas'][0])
    assert np.allclose(conf_grad, conf_lin, atol=.2)
    pred_grad = out.u_pred(xx, mode='sigy')
    pred_lin = upred(xx, out.properties['data'][0], 1.5, out.properties['sigmas'][0])
    assert np.allclose(pred_grad, pred_lin, atol=.2)


def test_symbolic():
    ''' Test symbolic expressions for y, uconf, upred '''
    x = np.linspace(100, 200, num=20)
    y = x/2 + np.random.normal(loc=0, scale=5, size=len(x))
    arr = curvefit.Array(x, y)  # NOTE: uconf, upred and expr_uconf, expr_upred will only match exactly when uy=0 in array.
    fit = curvefit.CurveFit(arr, func='line')
    fit.calculate(gum=True, mc=True, lsq=True)
    out = fit.out.lsq

    # Compare symbolic line equation with calculated via .y() method
    x = 150
    sym = out.expr(subs=True, n=10).subs({'x':x})
    assert np.isclose(out.y(x), float(list(sympy.solveset(sym))[0]))

    # Compare symbolic u_pred with calculated (via gradient) u_pred
    sym = out.expr_uconf(subs=True, n=10).subs({'x':x})
    assert np.isclose(float(list(sympy.solveset(sym))[0]), out.u_conf(x))

    sym = out.expr_upred(subs=True, n=10).subs({'x':x})
    assert np.isclose(float(list(sympy.solveset(sym))[0]), out.u_pred(x))

    # Now try higher order
    fit = curvefit.CurveFit(arr, func='cubic')
    fit.calculate(gum=True, mc=True, lsq=True)
    out = fit.out.lsq

    x = 110
    sym = out.expr(subs=True, n=10).subs({'x':x})
    assert np.isclose(float(list(sympy.solveset(sym))[0]), out.y(x))

    with pytest.raises(NotImplementedError):
        # Can't do symbolic conf/pred on anything except line fits.
        # If it is implemented eventually, write a test for it here.
        out.expr_uconf()


def test_namedfits():
    ''' Test pre-defined fit functions, poly, exp, decay, log, etc. '''
    # These can have big tolerance as fit params won't come out equal to nominal params.
    # Mostly checking that the functions/parameters are defined as expected.

    # Log fit
    np.random.seed(994942)
    x = np.linspace(100, 400, num=11)
    y = 20 * np.log(x-80) - 50
    y += np.random.normal(loc=0, scale=1, size=len(x))
    arr = curvefit.Array(x, y)
    fit = curvefit.CurveFit(arr, func='log', p0=(-50, 20, 80))
    fit.calculate()
    assert np.allclose(fit.out.lsq.mean, [-50, 20, 80], rtol=.1)

    # Exponential
    np.random.seed(994942)
    x = np.linspace(100, 200, num=21)
    y = 50 + .05 * np.exp(x/20)
    y += np.random.normal(loc=0, scale=10, size=len(x))
    arr = curvefit.Array(x, y)
    fit = curvefit.CurveFit(arr, func='exp', p0=(.05, 20, 50))
    fit.calculate()
    fit.out.lsq.plot_summary()
    assert np.allclose(fit.out.lsq.mean, [.05, 20, 50], rtol=.2)

    # Exponential Decay
    np.random.seed(994943)
    x = np.linspace(0, 50, num=21)
    y = 8 * np.exp(-x/10)
    y += np.random.normal(loc=0, scale=.50, size=len(x))
    arr = curvefit.Array(x, y)
    fit = curvefit.CurveFit(arr, func='decay')
    fit.calculate()
    fit.out.lsq.plot_summary()
    assert np.allclose(fit.out.lsq.mean, [8, 10], rtol=.2)

    # Logistic
    np.random.seed(994942)
    x = np.linspace(0, 40, num=21)
    y = 2 - 4 / (1 + np.exp((x-20)/3))
    y += np.random.normal(loc=0, scale=.1, size=len(x))
    arr = curvefit.Array(x, y)
    fit = curvefit.CurveFit(arr, func='logistic', p0=(-4, 3, 20, 2))
    fit.calculate()
    fit.out.lsq.plot_summary()
    assert np.allclose(fit.out.lsq.mean, (-4, 3, 20, 2), rtol=.1)


def test_absolutesigma():
    ''' Test absolute_sigma parameter. scipy.optimize.curve_fit implements this in the
        algorithm. ODR, curvefit.linefit, and curvefit.linefitYork base algorithms all
        return absolute_sigma=True, but were wrapped to allow absolute_sigma=False.

        From the scipy curve_fit documentation:
            cov(absolute_sigma=False) = cov(absolute_sigma=True) * chi2/(M-N)
        where chi2 = sum((resids/sigma)**2).
    '''
    # TEST absolute sigma, nonlinear fit
    x = np.array([0,380,794,1247,1673,2031,2747])
    y = np.array([6.635E-08,6.531E-08,6.354E-08,6.255E-08,6.212E-08,5.970E-08,5.907E-08])
    w = np.array([1.82E+18,1.84E+18,2.17E+18,7.72E+18,6.13E+18,6.18E+18,6.51E+18])
    sig = 1/np.sqrt(w)*2
    p0 = [6.5E-8, 4E-5]

    def expfunc(x, b, a):
        return a * np.exp(-b*x)

    # Compares scipy.optimize.curve_fit with and without absolute_sigma to wrapped ODR function
    # with and without absolute_sigma.
    coeff1, cov1 = scipy.optimize.curve_fit(expfunc, x, y, sigma=sig, p0=p0, absolute_sigma=True)
    coeff2, cov2 = curvefit.odrfit(expfunc, x, y, ux=None, uy=sig, p0=p0, absolute_sigma=True)
    assert np.allclose(coeff1, coeff2, rtol=.0001)
    assert np.allclose(cov1, cov2)

    coeff1, cov1 = scipy.optimize.curve_fit(expfunc, x, y, sigma=sig, p0=p0, absolute_sigma=False)
    coeff2, cov2 = curvefit.odrfit(expfunc, x, y, ux=None, uy=sig, p0=p0, absolute_sigma=False)
    assert np.allclose(coeff1, coeff2, rtol=0.0001)
    assert np.allclose(cov1, cov2)
    assert np.allclose(np.sqrt(np.diag(cov1)), [4.69E-6, 5.46E-10])   # Compare to Tablecurve's output values for this problem which assume absolute_sigma=False.

    # TEST absolute sigma, line fit
    def linefunc(x, b, a):
        return a + b * x

    # Compare functions with and without absolute_sigma parameter, for straight line fit
    #  scipy.optimize.curve_fit
    #  ODR, wrapped by curvefit module
    #  linefit implemented in curvefit module
    #  linefityork implemented in curvefit module

    coeff1, cov1 = scipy.optimize.curve_fit(linefunc, x, y, sigma=sig, p0=p0, absolute_sigma=True)
    coeff2, cov2 = curvefit.odrfit(linefunc, x, y, ux=None, uy=sig, p0=p0, absolute_sigma=True)
    coeff3, cov3 = curvefit.linefit(x, y, sig=sig, absolute_sigma=True)
    coeff4, cov4 = curvefit.linefitYork(x, y, sigy=sig, absolute_sigma=True)
    assert np.allclose(coeff1, coeff2)
    assert np.allclose(coeff1, coeff3)
    assert np.allclose(coeff1, coeff4)
    assert np.allclose(cov1, cov2)
    assert np.allclose(cov1, cov3)
    assert np.allclose(cov1, cov4)

    # And with absolute_sigma=False
    coeff1, cov1 = scipy.optimize.curve_fit(linefunc, x, y, sigma=sig, p0=p0, absolute_sigma=False)
    coeff2, cov2 = curvefit.odrfit(linefunc, x, y, ux=None, uy=sig, p0=p0, absolute_sigma=False)
    coeff3, cov3 = curvefit.linefit(x, y, sig=sig, absolute_sigma=False)
    coeff4, cov4 = curvefit.linefitYork(x, y, sigy=sig, absolute_sigma=False)
    assert np.allclose(coeff1, coeff2)
    assert np.allclose(coeff1, coeff3)
    assert np.allclose(coeff1, coeff4)
    assert np.allclose(cov1, cov2)
    assert np.allclose(cov1, cov3)
    assert np.allclose(cov1, cov4)


def test_absolutesigma2():
    ''' Test absolute_sigma parameter through the curvefit module. Results are
        compared against values from tablecurve calculation.
    '''
    x = np.array([0,380,794,1247,1673,2031,2747])
    y = np.array([6.635E-08,6.531E-08,6.354E-08,6.255E-08,6.212E-08,5.970E-08,5.907E-08])
    w = np.array([1.82E+18,1.84E+18,2.17E+18,7.72E+18,6.13E+18,6.18E+18,6.51E+18])
    arr = curvefit.Array(x, y, uy=1/np.sqrt(w))
    fit = curvefit.CurveFit(arr, 'a*exp(-b*x)', p0=(6.5E-8, 4E-5), absolute_sigma=False)
    fit.calculate()

    anom, bnom = 6.6116363E-8, 4.3159717E-5
    uanom, ubnom = 5.46E-10, 4.69E-6
    stderrnom = 5.986E-10
    r2nom = .943671
    Fnom = 84
    pred2747nom = 5.69E-8, 6.06E-8

    assert np.allclose(fit.out.lsq.mean, (anom, bnom))
    assert np.isclose(fit.out.lsq.uncert[0], uanom, atol=.01E-10)
    assert np.isclose(fit.out.lsq.uncert[1], ubnom, atol=.01E-6)
    assert np.isclose(fit.out.lsq.properties['Syx'], stderrnom, atol=0.005E-10)
    assert np.isclose(fit.out.lsq.properties['r']**2, r2nom, atol=.001)
    assert np.isclose(fit.out.lsq.y(2747) + fit.out.lsq.u_pred(2747, conf=.95), pred2747nom[0])
    assert np.isclose(fit.out.lsq.y(2747) - fit.out.lsq.u_pred(2747, conf=.95), pred2747nom[1])
    assert np.isclose(fit.out.lsq.properties['F'], Fnom, atol=.5)
