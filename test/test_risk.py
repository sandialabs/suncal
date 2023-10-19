import numpy as np
from scipy import stats

from suncal import risk


def test_risknorm():
    sigma0 = 1/stats.norm.ppf((1+.95)/2)
    assert np.isclose(risk.PFA_norm(.95, TUR=4),
                      risk.PFA(stats.norm(loc=0, scale=sigma0), stats.norm(loc=0, scale=.125), LL=-1, UL=1),
                      rtol=.01)
    assert np.isclose(risk.PFR_norm(.95, TUR=4),
                      risk.PFR(stats.norm(loc=0, scale=sigma0), stats.norm(loc=0, scale=.125), LL=-1, UL=1),
                      rtol=.01)

    sigma0 = 1/stats.norm.ppf((1+.8)/2)
    assert np.isclose(risk.PFA_norm(.8, TUR=2),
                      risk.PFA(stats.norm(loc=0, scale=sigma0), stats.norm(loc=0, scale=.25), LL=-1, UL=1),
                      rtol=.01)
    assert np.isclose(risk.PFR_norm(.8, TUR=2),
                      risk.PFR(stats.norm(loc=0, scale=sigma0), stats.norm(loc=0, scale=.25), LL=-1, UL=1),
                      rtol=.01)


def test_risk_values():
    # Compare PFA and CPFA with tables in draft RP-19
    tur = 4
    ufar = np.zeros(6)
    cfar = np.zeros(6)
    for i, itp in enumerate([.6, .7, .8, .9, .95, .99]):
        sigp = risk.risk.get_sigmaproc_from_itp(itp)
        dproc = stats.norm(loc=0, scale=sigp)
        dtest = stats.norm(loc=0, scale=sigp/tur)
        LL, UL = -1, 1
        GBL = GBU = 0
        ufar[i], cfar[i] = (risk.PFA(dproc, dtest, LL, UL, GBL, GBU)*100,
                            risk.PFA_conditional(dproc, dtest, LL, UL, GBL, GBU)*100)

    assert np.allclose(ufar, [4.83860326, 3.92234109, 2.85583037, 1.59995312, 0.87100925, 0.19991684], atol=1E-6)
    assert np.allclose(cfar, [8.26008199, 5.72322042, 3.63227145, 1.79880430, 0.92389664, 0.20243866], atol=1E-6)


def test_simpson_quad():
    # Test PFA/PFR using simpson vs scipy.quad
    def testit(proc, test, LL=-1, UL=1):
        pfa1 = risk.risk_simpson.PFA(proc, test, LL, UL)
        pfa2 = risk.risk_quad.PFA(proc, test, LL, UL)
        assert np.isclose(pfa1, pfa2, rtol=.005, atol=.003)

    testit(stats.norm(0, 1), stats.norm(0, 0.25))
    testit(stats.norm(0, scale=1), stats.norm(0, scale=0.25))  # unnamed vs kwarg shouldn't matter
    testit(stats.norm(loc=0, scale=1), stats.norm(loc=0, scale=0.25))
    testit(stats.uniform(0, 2), stats.norm(3, 0.25))


def test_simpson_quad_pfr():
    # Test PFA/PFR using simpson vs scipy.quad
    def testit(proc, test, LL=-1, UL=1):
        pfa1 = risk.risk_simpson.PFR(proc, test, LL, UL)
        pfa2 = risk.risk_quad.PFR(proc, test, LL, UL)
        assert np.isclose(pfa1, pfa2, rtol=.005, atol=.003)

    testit(stats.norm(0, 1), stats.norm(0, 0.25))
    testit(stats.norm(0, scale=1), stats.norm(0, scale=0.25))  # unnamed vs kwarg shouldn't matter
    testit(stats.norm(loc=0, scale=1), stats.norm(loc=0, scale=0.25))
    testit(stats.uniform(0, 2), stats.norm(3, 0.25))


def test_riskmontecarlo():
    # Compare Monte Carlo results
    np.random.seed(883322)

    # Deaver 7 values
    FA, FR, *_ = risk.PFAR_MC(stats.norm(loc=0, scale=4), stats.norm(loc=0, scale=1), LL=-8, UL=8)
    assert np.isclose(FA, .008, atol=.0005)
    assert np.isclose(FR, .015, atol=.0005)

    # Uniform distributions
    d1 = stats.uniform(loc=-1, scale=2)
    d2 = stats.uniform(loc=-.2, scale=.4)
    FA1, FR1, *_ = risk.PFAR_MC(d1, d2, LL=9.9, UL=11, N=1000000)
    assert np.isclose(FA1, risk.PFA(d1, d2, LL=9.9, UL=11), atol=.01)
    assert np.isclose(FR1, risk.PFR(d1, d2, LL=9.9, UL=11), atol=.01)


def test_findguardband():
    d1 = stats.norm(loc=0, scale=4)
    d2 = stats.norm(loc=0, scale=2)
    LL = -8
    UL = 8
    target = .01

    # PFA calculated with guard band should match target
    gb = risk.guardband.target(d1, d2, LL=LL, UL=UL, target_PFA=target)
    assert np.isclose(target, risk.PFA(d1, d2, LL=LL, UL=UL, GBL=gb, GBU=gb))

    # This one won't converge, should return None
    gb2 = risk.guardband.target(d1, d2, LL=LL, UL=UL, target_PFA=.1)
    assert not np.isfinite(gb2)


def test_guardbandnorm():
    # TUR-based guardbands
    TUR = 2.5
    itp = .8

    # Target 1% PFA
    gb = risk.guardband_tur.pfa_target(TUR, pfa=.01, itp=itp)
    assert np.isclose(risk.PFA_norm(itp, TUR, gb), 0.01, rtol=.001)

    # dobbert method
    M = 1.04 - np.exp(.38*np.log(TUR)-0.54)
    assert np.isclose(risk.guardband_tur.dobbert(TUR), 1-M/TUR, rtol=.0001)

    # Same as 4:1
    pfa41 = risk.PFA_norm(itp=itp, TUR=4)  # get 4:1 pfa
    gb = risk.guardband_tur.four_to_1(TUR, itp=itp)
    assert np.isclose(risk.PFA_norm(itp=itp, TUR=TUR, GB=gb), pfa41, rtol=.0001)

    # Other methods
    assert np.isclose(risk.guardband_tur.rss(TUR), np.sqrt(1-1/TUR**2), rtol=.0001)
    assert np.isclose(risk.guardband_tur.test95(TUR), 1-1/TUR)
    assert np.isclose(risk.guardband_tur.rp10(TUR), 1.25-1/TUR)


def test_cpk():
    # Test process capability index
    LL = 9
    UL = 11

    d = stats.norm(loc=10, scale=.4)
    cpk, r, rLL, rUL = risk.specific_risk(d, LL, UL)
    assert np.isclose(cpk, 0.833333, rtol=.001)   # Compared against Dilip's risk spreadsheet
    assert np.isclose(r, .01242, rtol=.01)
    assert np.isclose(rLL, .00621, rtol=.01)
    assert np.isclose(rUL, .00621, rtol=.01)

    d = stats.norm(loc=10.5, scale=.4)
    cpk, r, rLL, rUL = risk.specific_risk(d, LL, UL)
    assert np.isclose(cpk, 0.416667, rtol=.001)
    assert np.isclose(r, .10574, rtol=.01)
    assert np.isclose(rLL, .00009, atol=.0001)
    assert np.isclose(rUL, .10565, atol=.0001)

    d = stats.norm(loc=12.0, scale=.4)
    cpk, r, rLL, rUL = risk.specific_risk(d, LL, UL)
    assert np.isclose(cpk, -.833333, rtol=.001)
    assert np.isclose(r, .99379, rtol=.01)

    # TEST non-normal calculation gives same results as normal
    LL = 50
    UL = 55
    d = stats.norm(loc=51, scale=.5)
    cpk, r, rLL, rUL = risk.specific_risk(d, LL, UL)

    d.name = 'nonnormal'  # Fake a non-normal distribution! Will use ppf functions for cpk
    cpk2, r2, rLL2, rUL2 = risk.specific_risk(d, LL, UL)
    assert np.isclose(cpk, cpk2)
    assert np.isclose(r, r2)
    assert np.isclose(rLL, rLL2)
    assert np.isclose(rUL, rUL2)


def test_gb():
    # Test PFA/PFR with guard band, based on more numbers in Deaver (page 10)
    # First, using PFA_tur() ...
    assert np.isclose(risk.deaver.PFA_deaver(SL=2, TUR=2, GB=0.91), 0.008, atol=.0005)
    assert np.isclose(risk.deaver.PFR_deaver(SL=2, TUR=2, GB=0.91), 0.066, atol=.001)
    # .. and PFA()
    assert np.isclose(risk.PFA(stats.norm(loc=0, scale=1),
                               stats.norm(loc=0, scale=.5), LL=-2, UL=2, GBL=(2-2*.91), GBU=(2-2*.91)),
                      0.008,
                      atol=.0005)
    assert np.isclose(risk.PFR(stats.norm(loc=0, scale=1),
                               stats.norm(loc=0, scale=.5), LL=-2, UL=2, GBL=(2-2*.91), GBU=(2-2*.91)),
                      0.066,
                      atol=.001)


def test_riskdeaver():
    # Verify results against values in Deaver page 7, paragraph 2
    assert np.isclose(risk.deaver.PFA_deaver(SL=2, TUR=4), 0.008, atol=.0001)
    assert np.isclose(risk.deaver.PFA_deaver(SL=2.5, TUR=4), 0.0025, atol=.0001)
    assert np.isclose(risk.deaver.PFA_deaver(SL=2.5, TUR=1), .005, atol=.001)


def test_riskdist():
    # Verify risk.PFA and risk.PFR functions using distributions with normal is same as Deaver method for normal
    assert np.isclose(risk.deaver.PFA_deaver(SL=2, TUR=4),
                      risk.PFA(stats.norm(loc=0, scale=1), stats.norm(loc=0, scale=.25), LL=-2, UL=2),
                      rtol=.01)
    assert np.isclose(risk.deaver.PFR_deaver(SL=2, TUR=4),
                      risk.PFR(stats.norm(loc=0, scale=1), stats.norm(loc=0, scale=.25), LL=-2, UL=2),
                      rtol=.01)

    assert np.isclose(risk.deaver.PFA_deaver(SL=3, TUR=3),
                      risk.PFA(stats.norm(loc=0, scale=1), stats.norm(loc=0, scale=.333), LL=-3, UL=3),
                      rtol=.01)
    assert np.isclose(risk.deaver.PFR_deaver(SL=3, TUR=3),
                      risk.PFR(stats.norm(loc=0, scale=1), stats.norm(loc=0, scale=.333), LL=-3, UL=3),
                      rtol=.01)
