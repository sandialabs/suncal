''' Simple Risk Calculator pyscript interface

    For efficiency, does not load Suncal. Uses standalone, no-dependency
    risk calculation using only python's builtin statistics library.
    A tad slower to calculate, but so much faster to initialize.
'''
from pyscript import document, display

import statistics

standard_normal = statistics.NormalDist()


def risk_simple(event=None):
    ''' Update simple risk calculation from controls on page '''
    tur = float(document.getElementById('TUR').value)
    itp = float(document.getElementById('ITP').value) / 100
    gbf = float(document.getElementById('GBF').value)
    model = RiskResult(itp, tur, gbf)
    display(model, target='output', append=False)


class RiskResult:
    ''' Simple Risk Calculator '''
    def __init__(self, itp: float = .95, tur: float = 4, gbf: float = 1):
        self.tur = tur
        self.itp = itp
        self.gbf = gbf
        self.T = 1  # Normalize to a tolerance of +/- 1

        self.sigma_p = 1 / standard_normal.inv_cdf((1+self.itp)/2)
        self.sigma_m = 1/self.tur/2
        self.accept_limit = self.T*self.gbf

    def pfa(self, N=10000):
        ''' Calculate global PFA '''
        def pfa_integrand(z):
            phi_z = standard_normal.pdf(z)
            f_z = standard_normal.cdf((self.accept_limit-self.sigma_p*z)/self.sigma_m) - standard_normal.cdf((-self.accept_limit-self.sigma_p*z)/self.sigma_m)
            return phi_z * f_z

        inf = max(2, self.sigma_p*6)  # "infinity" for the integration
        dz = (inf-1)/N
        zvalues = [self.T/self.sigma_p + i*dz for i in range(N)]
        pfa = sum(map(pfa_integrand, zvalues[1:-1])) + (pfa_integrand(zvalues[0]) + pfa_integrand(zvalues[-1]))/2
        pfa = 2 * pfa * dz
        return pfa

    def pfr(self, N=10000):
        ''' Calculate global PFR '''
        def pfr_integrand(z):
            phi_z = standard_normal.pdf(z)
            f_z = standard_normal.cdf((self.accept_limit-self.sigma_p*z)/self.sigma_m) - standard_normal.cdf((-self.accept_limit-self.sigma_p*z)/self.sigma_m)
            return phi_z * (1-f_z)

        dz = 2*self.T/self.sigma_p/N
        zvalues = [-self.T/self.sigma_p + i*dz for i in range(N)]
        pfr = sum(map(pfr_integrand, zvalues[1:-1])) + (pfr_integrand(zvalues[0]) + pfr_integrand(zvalues[-1]))/2
        pfr *= dz
        return pfr

    def worst_specific(self):
        ''' Worst-case specific risk '''
        measuredist = statistics.NormalDist(self.accept_limit, self.sigma_m)
        risk_lower = measuredist.cdf(-self.T)
        risk_upper = 1-measuredist.cdf(self.T)
        return risk_lower + risk_upper

    def process_risk(self):
        ''' Process risk '''
        risk_total = 1-self.itp
        return risk_total

    def _repr_html_(self):
        ''' HTML representation for display function '''
        pfa = self.pfa()
        pfr = self.pfr()
        wspecific = self.worst_specific()
        proc = self.process_risk()
        return f'''<table>
        <thead><tr>
            <td>Process Risk</td>
            <td>Specific Measurement Risk</td>
            <td>Global Risk</td>
        </tr></thead>
        <tbody>
        <tr><td>Process Risk: {proc*100:.2f}%</td><td>TUR: {self.tur:.1f}</td><td>Total PFA: {pfa*100:.3f}%</td></tr>
        <tr><td>Upper Limit Risk: {proc*100/2:.2f}%</td><td>Worst-case Specific Risk: {wspecific*100:.1f}%</td><td>Total PFR: {pfr*100:.2f}%</td></tr>
        <tr><td>Lower Limit Risk: {proc*100/2:.2f}%</td><td></td><td></td></tr>
        </tbody>
       </table>'''


# Initialize
risk_simple()
