from pyscript import document, display
from collections import namedtuple
import numpy as np
import matplotlib.pyplot as plt

from suncal.common.report import Report
from suncal.risk.report.risk import risk_sweeper

Report.apply_css = False  # Use the HTML's css instead of suncal's


def sweep_vars():
    ''' Get sweep variable assignments from page '''
    xstart = float(document.getElementById("sweepXStart").value)
    xstop = float(document.getElementById("sweepXStop").value)
    xnum = int(document.getElementById("numPoints").value)
    xvalues = np.linspace(xstart, xstop, num=xnum)
    zvalues = document.getElementById("stepZ").value
    zvalues = [float(z) for z in zvalues.split(',')]

    itp_vs_sl = document.getElementById("sweepMode").textContent
    itpmode = document.getElementById("variableSelect1").textContent
    turmode = document.getElementById("variableSelect2").textContent
    gbfmode = document.getElementById("variableSelect3").textContent
    pbiasmode = document.getElementById("variableSelect4").textContent 
    mbiasmode = document.getElementById("variableSelect5").textContent

    # default constants
    itpval = float(document.getElementById("var1Constant").value) / 100
    turval = float(document.getElementById("var2Constant").value)
    gbfval = float(document.getElementById("var3Constant").value)
    pbiasval = float(document.getElementById("var4Constant").value) / 100
    mbiasval = float(document.getElementById("var5Constant").value) / 100
    sig0 = float(itpval) if 'SL' in itp_vs_sl else None

    if gbfmode != 'Constant':
        gbfval = gbfmode.lower()
        gbfval = 'test' if 'test' in gbfval else gbfval

    if 'Step' in itpmode and 'In-' in itp_vs_sl:
        zvar = 'itp'
    elif 'Step' in itpmode:
        zvar = 'sig0'
    elif 'Step' in turmode:
        zvar = 'tur'
    elif 'Step' in gbfmode:
        zvar = 'gbf'
    elif 'Step' in pbiasmode:
        zvar = 'pbias'
    elif 'Step' in mbiasmode:
        zvar = 'tbias'
    else:
        zvar = 'none'
        zvalues = [None]  # Need one item to loop

    if 'Sweep' in itpmode and 'In-' in itp_vs_sl:
        xvar = 'itp'
    elif 'Sweep' in itpmode:
        xvar = 'sig0'
    elif 'Sweep' in turmode:
        xvar = 'tur'
    elif 'Sweep' in gbfmode:
        xvar = 'gbf'
    elif 'Sweep' in pbiasmode:
        xvar = 'pbias'
    elif 'Sweep' in mbiasmode:
        xvar = 'tbias'
    else:
        xvar = 'none'

    # Convert percent to decimal 0-1
    if xvar in ['itp', 'tbias', 'pbias']:
        xvalues = xvalues / 100
    if zvar in ['itp', 'tbias', 'pbias']:
        zvalues = zvalues / 100

    threed = document.getElementById("plot3D").checked
    y = document.getElementById("plotSelect").textContent
    logy = document.getElementById("LogScale").checked
    SweepSetup = namedtuple(
        'SweepSetup', ['x', 'z', 'xvals', 'zvals', 'itp', 'tur',
                       'gbf', 'sig0', 'pbias', 'tbias', 'threed',
                       'y', 'logy'])
    return SweepSetup(xvar, zvar, xvalues, zvalues, itpval, turval,
                      gbfval, sig0, pbiasval, mbiasval, threed, y, logy)


def calculate_sweep(event=None):
    ''' Trigger replotting '''
    setup = sweep_vars()
    fig = plt.figure()
    swp_report = risk_sweeper(
        fig,
        xvar=setup.x,
        zvar=setup.z,
        xvals=setup.xvals,
        zvals=setup.zvals,
        yvar=setup.y,
        threed=setup.threed,
        logy=setup.logy,
        gbmode=setup.gbf,
        sig0=setup.sig0,
        pbias=setup.pbias,
        tbias=setup.tbias)

    rpt = Report()
    rpt.plot(fig)
    rpt.append(swp_report)
    display(rpt, target="output", append=False)


# Startup
calculate_sweep()
