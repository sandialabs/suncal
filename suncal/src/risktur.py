''' Simple Risk Calculator pyscript interface '''
from pyscript import document, display

from suncal.risk.risk_model import RiskModelSimple
from suncal.common.report import Report

Report.apply_css = False  # Use this page's CSS, not Suncal's built-in CSS


def risk_simple(event=None):
    ''' Update simple risk calculation '''
    tur = float(document.getElementById('TUR').value)
    itp = float(document.getElementById('ITP').value) / 100
    gbf = float(document.getElementById('GBF').value)
    model = RiskModelSimple(tur=tur, itp=itp, gbf=gbf)
    result = model.calculate()
    display(result.report.summary(n=3), target='output', append=False)


# Initialize
risk_simple()
