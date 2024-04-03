''' Measurement Decision Risk Calculator - pyscript interface '''
from pyscript import document, display

from suncal.risk.risk_model import RiskModel
from suncal.common import distributions
from suncal.common.report import Report

Report.apply_css = False  # Use this page's CSS, not Suncal's built-in CSS


def get_model():
    ''' Get RiskModel from entries on page '''
    proc_median = float(document.getElementById("Median").value)
    meas_median = float(document.getElementById("bias").value)
    measurement = float(document.getElementById("measurement").value)
    lower_lim = float(document.getElementById("LowL").value)
    upper_lim = float(document.getElementById("UpL").value)
    proc_dist_name = document.getElementById("procdist").textContent
    meas_dist_name = document.getElementById("measdist").textContent

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
    result = model.calculate()
    display(result.report.summary(), target='output', append=False)
    display(result.report.plot.joint(), target='plot', append=False)


def calc_guardband(event=None):
    ''' Calculate guardband '''
    pfa = float(document.getElementById("TargetPFA").value) / 100
    model = get_model()
    model.guardband_pfa(pfa=pfa)
    document.getElementById("lowerGuard").value = model.gbofsts[0]
    document.getElementById("upperGuard").value = model.gbofsts[1]
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


def enable_guardband(event):
    ''' Guardband checkbox clicked '''
    if document.getElementById("guardband").checked:
        document.getElementById("lowerGuard").disabled = False
        document.getElementById("upperGuard").disabled = False
        document.getElementById("TargetPFA").disabled = False
        document.getElementById("calculateGBF").disabled = False
    else:
        document.getElementById("lowerGuard").disabled = True
        document.getElementById("upperGuard").disabled = True
        document.getElementById("TargetPFA").disabled = True
        document.getElementById("calculateGBF").disabled = True
    calculate()


# Initialize
calculate()
