''' Probability of Conformance Plot - Without using Suncal for loading speed
    (only normal distributions)
'''
from pyscript import document, display
import statistics
import math

import ziaplot as zp


def limits() -> tuple[float, float]:
    ''' Get specification limits from page '''
    TL = float(document.getElementById('LowL').value)
    TU = float(document.getElementById('UpL').value)
    if not document.getElementById('UpLimitEnable').checked:
        TU = math.inf
    if not document.getElementById('LowLimitEnable').checked:
        TL = -math.inf
    return TL, TU


def prob_conform() -> tuple[list[float], list[float]]:
    ''' Calculate probability of conformance curve '''
    TL, TU = limits()
    stdev = float(document.getElementById('stdev').value)
    bias = float(document.getElementById('bias').value)
    lim1 = TL-stdev*4 if math.isfinite(TL) else TU-stdev*6
    lim2 = TU+stdev*4 if math.isfinite(TU) else TL+stdev*6
    xvals = zp.linspace(lim1, lim2, 75)

    prob = []
    for x in xvals:
        dist = statistics.NormalDist(mu=x-bias, sigma=stdev)
        plower = 1 - dist.cdf(TL) if math.isfinite(TL) else 0
        pupper = dist.cdf(TU) if math.isfinite(TU) else 0
        prob.append((plower + pupper)*100)

    if math.isfinite(TL) and math.isfinite(TU):
        prob = [p-100 for p in prob]

    return xvals, prob


def calculate(event=None) -> None:
    ''' Run calculation and display result '''
    xvals, prob = prob_conform()
    TL, TU = limits()
    guardbandon = document.getElementById('guardband').checked
    gblow = float(document.getElementById('lowerGuard').value)
    gbhigh = float(document.getElementById('upperGuard').value)
    document.getElementById('lowerGuard').disabled = not guardbandon
    document.getElementById('upperGuard').disabled = not guardbandon

    with zp.Graph().axesnames('Measurement Result', 'Probability of Conformance %') as plt:
        zp.PolyLine(xvals, prob)
        if math.isfinite(TL):
            zp.VLine(TL).color('black').stroke('--')
        if math.isfinite(TU):
            zp.VLine(TU).color('black').stroke('--')

        if guardbandon:
            zp.VLine(gblow).color('C2').stroke(':')
            zp.VLine(gbhigh).color('C2').stroke(':')

    display(plt, target='plot', append=False)

    table = '''<table><thead><tr>
    <td>Measurement Result</td>
    <td>Probability of Conformance %</td>
</tr></thead>
<tbody>
'''
    for x, y in zip(xvals, prob):
        table += f'<tr><td>{x:.3g}</td><td>{y:.2f}</td></tr>\n'

    table += '</tbody></table>\n'
    document.getElementById('output').innerHTML = table


def max_pconform_guardband() -> tuple[float, float]:
    ''' Calculate guardband to achieve a worst-case probability of conformance '''
    xvals, prob = prob_conform()
    thresh = float(document.getElementById('worstspecific').value)
    idx1 = [i for i, p in enumerate(prob) if p > thresh][0]
    idx2 = idx1 + 1
    gblow = xvals[idx1] + (thresh - prob[idx1]) * (xvals[idx2] - xvals[idx1]) / (prob[idx2]-prob[idx1])
    idx1 = [i for i, p in enumerate(prob[::-1]) if p > thresh][0]
    idx2 = idx1 + 1
    gbhigh = xvals[-idx1] + (thresh - prob[-idx1]) * (xvals[-idx1] - xvals[-idx2]) / (prob[-idx1]-prob[-idx2])
    return gblow, gbhigh


def calc_guardband(event=None) -> None:
    ''' Calculate guardband '''
    TL = float(document.getElementById('LowL').value)
    TU = float(document.getElementById('UpL').value)
    stdev = float(document.getElementById('stdev').value)
    if not document.getElementById('UpLimitEnable').checked:
        TU = math.inf
    if not document.getElementById('LowLimitEnable').checked:
        TL = -math.inf

    gbmode = event.target.id
    if gbmode in ['rss', 'dobbert', 'rp10', 'test']:
        tur = (TU - TL)/stdev/4
        nominal = (TU+TL)/2
        if not math.isfinite(tur):
            return
        if gbmode == 'rss':
            gbf = math.sqrt(1-1/tur**2)
        elif gbmode == 'rp10':
            gbf = 1.25 - 1/tur
        elif gbmode == 'test':
            gbf = 1 - 1/tur
        elif gbmode == 'dobbert':
            m = 1.04 - math.exp(0.38 * math.log(tur) - 0.54)
            gbf = 1 - m/tur
        gblow = nominal - (nominal-TL)*gbf
        gbhigh = nominal + (TU-nominal)*gbf

    else:
        gblow, gbhigh = max_pconform_guardband()

    document.getElementById("guardband").checked = True
    document.getElementById("lowerGuard").value = gblow
    document.getElementById("upperGuard").value = gbhigh
    calculate()


calculate()
