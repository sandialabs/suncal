''' Measurement Decision Risk Calculator - pyscript interface '''
from pyscript import document, display
import math

from suncal.risk.risk_model import RiskModel
from suncal.common import distributions
from suncal.common import plotting
from suncal.common.report import Report

plotting.plotstyle['figure.figsize'] = (6, 4)
plotting.plotstyle['font.size'] = 12
Report.apply_css = False  # Use this page's CSS, not Suncal's built-in CSS


def get_model():
    ''' Get RiskModel from entries on page '''
    proc_median = float(document.getElementById("Median").value)
    meas_median = float(document.getElementById("bias").value)
    measurement = float(document.getElementById("measurement").value)
    proc_dist_name = document.getElementById("procdist").textContent
    meas_dist_name = document.getElementById("measdist").textContent
    if document.getElementById("LowLimitEnable").checked:
        lower_lim = float(document.getElementById("LowL").value)
    else:
        lower_lim = -math.inf

    if document.getElementById("UpLimitEnable").checked:
        upper_lim = float(document.getElementById("UpL").value)
    else:
        upper_lim = math.inf

    if proc_dist_name == 'normal':
        procargs = {
          'std': float(document.getElementById("std").value),
          'median': proc_median
        }
    elif proc_dist_name == 'gamma':
        procargs = {
            'alpha': float(document.getElementById("std").value),
            'beta': float(document.getElementById("beta").value)
        }
    else:
        procargs = {
           'a': float(document.getElementById("std").value),
           'median': measurement
        }
    procdist = distributions.get_distribution(proc_dist_name, **procargs)
    if proc_dist_name == 'gamma':
        procdist.set_shift(proc_median)

    if meas_dist_name == 'normal':
        procargs = {
          'std': float(document.getElementById("std2").value),
          'median': measurement
        }
    elif meas_dist_name == 'gamma':
        procargs = {
            'alpha': float(document.getElementById("std2").value),
            'beta': float(document.getElementById("beta2").value),
            'shift': measurement
        }
    else:
        procargs = {
           'a': float(document.getElementById("std2").value),
           'median': measurement
        }
    measdist = distributions.get_distribution(meas_dist_name, **procargs)

    if meas_dist_name == 'gamma':
        measdist.set_shift(proc_median)

    gbl = gbu = 0
    if document.getElementById("guardband").checked:
        gbl = float(document.getElementById("lowerGuard").value)
        gbu = float(document.getElementById("upperGuard").value)

    model = RiskModel(procdist, measdist, (lower_lim, upper_lim), (gbl, gbu))
    model.testbias = meas_median
    return model


def calculate(event=None):
    ''' Update full risk calculation '''
    model = get_model()
    if not math.isfinite(model.speclimits[0]) and not math.isfinite(model.speclimits[1]):
        display('Risk calculation must have at least one specification limit enabled',
                target='output', append=False)
        display('', target='plot', append=False)
        return

    conditional = document.getElementById("conditional").checked

    result = model.calculate()
    display(result.report.summary(conditional=conditional), target='output', append=False)
    display(result.report.plot.joint(), target='plot', append=False)


def calc_guardband(event=None):
    ''' Calculate guardband '''
    model = get_model()
    gbmode = event.target.id
    if gbmode in ['rss', 'dobbert', 'rp10', '4:1', 'test']:
        model.guardband_tur(gbmode)
    elif gbmode in ['pfa']:
        target = float(document.getElementById("TargetPFA").value) / 100
        conditional = document.getElementById("conditional").checked
        model.guardband_pfa(pfa=target, conditional=conditional)
    elif gbmode in ['pfr']:
        target = float(document.getElementById("TargetPFR").value) / 100
        model.guardband_pfr(pfr=target)
    elif gbmode in ['specific']:
        target = float(document.getElementById("worstspecific").value) / 100
        model.guardband_specific(target=target)

    document.getElementById("guardband").checked = True
    document.getElementById("lowerGuard").value = model.gbofsts[0]
    document.getElementById("upperGuard").value = model.gbofsts[1]
    enable_guardband()
    calculate()


def update_limit_enable(event=None):
    ''' Lower/Upper limit checkbox changed '''
    low_on = document.getElementById("LowLimitEnable").checked
    hi_on = document.getElementById("UpLimitEnable").checked
    document.getElementById("LowL").disabled = not low_on
    document.getElementById("UpL").disabled = not hi_on
    calculate()


def update_process_rows(event):
    ''' Process distribution changed. Update rows and replot. '''
    target = event.target.id
    paramlabel = {
       'normalProc': 'std',
       'triangularProc': 'a',
       'uniformProc': 'a',
       'gammaProc': 'alpha'
    }.get(target)
    document.getElementById('stdLabel').textContent = paramlabel

    distlabel = {
       'normalProc': 'normal',
       'triangularProc': 'triangular',
       'uniformProc': 'uniform',
       'gammaProc': 'gamma'
    }.get(target)
    document.getElementById('procdist').textContent = distlabel

    # Show/hide extra row for gamma distribution
    if target == 'gammaProc':
        document.getElementById('procBetaRow').classList.remove('input-hidden')
    else:
        document.getElementById('procBetaRow').classList.add('input-hidden')
    calculate()


def update_measure_rows(event):
    ''' Measurement distribution changed. Update rows and replot. '''
    target = event.target.id
    paramlabel = {
       'normalMeas': 'std',
       'triangularMeas': 'a',
       'uniformMeas': 'a',
       'gammaMeas': 'alpha'
    }.get(target)
    document.getElementById('std2Label').textContent = paramlabel

    distlabel = {
       'normalMeas': 'normal',
       'triangularMeas': 'triangular',
       'uniformMeas': 'uniform',
       'gammaMeas': 'gamma'
    }.get(target)
    document.getElementById('measdist').textContent = distlabel

    # Show/hide extra row for gamma distribution
    if target == 'gammaMeas':
        document.getElementById('measBetaRow').classList.remove('input-hidden')
    else:
        document.getElementById('measBetaRow').classList.add('input-hidden')
    calculate()


def enable_guardband(event=None):
    ''' Guardband checkbox clicked '''
    if document.getElementById("guardband").checked:
        document.getElementById("lowerGuard").disabled = False
        document.getElementById("upperGuard").disabled = False
    else:
        document.getElementById("lowerGuard").disabled = True
        document.getElementById("upperGuard").disabled = True
    calculate()


# Initialize
calculate()
