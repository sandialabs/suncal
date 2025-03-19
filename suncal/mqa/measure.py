''' MQA Measurement Process '''
from typing import Literal
from dataclasses import dataclass, field
from scipy import stats
import numpy as np

from ..common.limit import Limit
from ..uncertainty.variables import Typeb, RandomVariable
from ..uncertainty import Model
from .equipment import MqaEquipmentRange
from .pdf import Pdf
from .result import MqaUncertaintyResult


RenewalPolicy = Literal['never', 'asneeded', 'always']
ReliabilityModel = Literal['none', 'randomwalk', 'exponential', 'custom']


class ReliabilityExponential:
    ''' Exponential Reliability Model '''
    def __init__(self, rbop: float, reop: float, interval: float, tolerance: Limit, testpoint: float):
        self.reop = reop
        self.rbop = rbop
        self.interval = interval
        self.tolerance = tolerance
        self.testpoint = testpoint

        self.lamb = -1 / self.interval * np.log(self.reop/self.rbop)
        self.r_aop = self.rbop / self.lamb / self.interval * (1 - np.exp(-self.lamb * self.interval))

        if not self.tolerance.symmetric:
            self.sigma_aop = Pdf.from_itp(testpoint, self.r_aop, tolerance).std
        else:
            tol = float(self.tolerance.plusminus)
            self.sigma_aop = tol / stats.norm.ppf((1+self.r_aop)/2)

    def pdf(self, t) -> Pdf:
        ''' Get Pdf at time t in days '''
        reliability_t = self.rbop * np.exp(-self.lamb * t)
        if not self.tolerance.symmetric:
            sigma_t = Pdf.from_itp(self.testpoint, reliability_t, self.tolerance).std
        else:
            sigma_t = float(self.tolerance.plusminus) / stats.norm.ppf((1+reliability_t)/2)
        return Pdf.from_stdev(self.testpoint, sigma_t)

    def aop_pdf(self) -> Pdf:
        ''' Pdf of average-over-period '''
        return Pdf.from_stdev(self.testpoint, self.sigma_aop)

    def find_interval(self, reliability: float) -> float:
        ''' Find the interval (years) where the reliability falls to the set level '''
        return np.log(reliability / self.rbop) / -self.lamb


class ReliabilityRandomWalk:
    ''' Random Walk reliability model '''
    def __init__(self, rbop: float, reop: float, interval: float, tolerance: Limit, testpoint: float):
        self.reop = reop
        self.rbop = rbop
        self.interval = interval
        self.tolerance = tolerance  # Assumes symmetric tolerance!
        self.testpoint = testpoint

        if not self.tolerance.symmetric:
            self.sigma_bop = Pdf.from_itp(testpoint, self.rbop, tolerance).std
            self.sigma_eop = Pdf.from_itp(testpoint, self.reop, tolerance).std
        else:
            tol = float(self.tolerance.plusminus)
            self.sigma_bop = tol / stats.norm.ppf((1 + self.rbop)/2)
            self.sigma_eop = tol / stats.norm.ppf((1 + self.reop)/2)

        self.alpha = (self.sigma_eop**2 - self.sigma_bop**2) / self.interval
        self.sigma_aop = np.sqrt(self.sigma_bop**2 + self.alpha*self.interval/2)  # RP-19 (3-26)
        self.r_aop = self.aop_pdf().itp(self.tolerance)

    def pdf(self, t) -> Pdf:
        ''' Get Pdf at time t in years '''
        sigma_t = np.sqrt(self.sigma_bop**2 + self.alpha * t)
        return Pdf.from_stdev(self.testpoint, sigma_t)

    def aop_pdf(self) -> Pdf:
        ''' Pdf of average-over-period '''
        return Pdf.from_stdev(self.testpoint, self.sigma_aop)

    def find_interval(self, reliability: float) -> float:
        ''' Find the interval (years) where the reliability falls to the set level '''
        if self.tolerance.symmetric:
            sigma_t = float(self.tolerance.plusminus) / stats.norm.ppf((1+reliability)/2)
        else:
            sigma_t = Pdf.from_itp(self.testpoint, reliability, self.tolerance).std
        return (sigma_t**2 - self.sigma_bop**2) / self.alpha


@dataclass
class MqaCalibration:
    ''' Info about the calibration process '''
    repair_limit: Limit = None
    mte_adjust: Pdf = None
    mte_repair: Pdf = None
    stress_pre: Pdf = None
    stress_post: Pdf = None
    policy: RenewalPolicy = 'never'
    p_discard: float = 0


@dataclass
class MqaInterval:
    ''' Info about the interval/reliability model '''
    years: float = 1.0
    reliability_model: ReliabilityModel = 'none'

    # Computed by the model
    model: ReliabilityExponential | ReliabilityRandomWalk = None
    bop: Pdf = None

    @property
    def days(self):
        return self.years * 365.25

    def set_model(
            self,
            rbop: float,
            reop: float,
            bop: Pdf,
            tolerance: Limit,
            testpoint: float) -> None:
        ''' Fit the reliability model and store it '''
        self.bop = bop
        if self.reliability_model == 'none':
            self.model = None
        elif self.reliability_model == 'exponential':
            self.model = ReliabilityExponential(
                rbop, reop, self.years, tolerance, testpoint)
        elif self.reliability_model == 'randomwalk':
            self.model = ReliabilityRandomWalk(
                rbop, reop, self.years, tolerance, testpoint)
        else:
            # Custom/other reliability decay models
            raise NotImplementedError

    def pdf_time(self, t: float) -> Pdf:
        ''' Get projected Pdf at time=t (years) '''
        if self.model is None:
            return self.bop
        return self.model.pdf(t)

    def aop(self) -> Pdf:
        ''' Get average-over-period reliability projected for this model '''
        return self.pdf_time(self.years/2)


@dataclass
class MTE:
    ''' M&TE for one direct measurement.
        Quantity takes precendence over equipment, over accuracy
    '''
    quantity: 'MqaQuantity' = None
    equipment: MqaEquipmentRange = None
    accuracy_plusminus: float = 0.25
    accuracy_eopr: float = .95  # None means uniform distribution
    accuracy_pdf: Pdf = None
    name: str = ''

    def equip_name(self):
        ''' Get name of equipment, or user-entered name of MTE '''
        if self.quantity is not None:
            return self.quantity.measurand.name
        if self.equipment and (name := str(self.equipment)):
            return name
        return self.name

    def uncertainty(self, testpoint: float) -> MqaUncertaintyResult:
        ''' Uncertainty of the M&TE '''
        parent = None
        degf = float('inf')
        if self.quantity is not None:
            result = self.quantity.calculate()
            pdf = result.reliability.aop.pdf.given_y(0)
            parent = result
            accuracy = float(self.quantity.measurand.tolerance.plusminus)

        elif self.equipment is not None and self.equipment.equipment is not None:
            tol = self.equipment.tolerance(testpoint)
            accuracy = float(tol.plusminus)
            pdf = self.equipment.pdf(testpoint)

        elif self.accuracy_pdf is not None:
            pdf = self.accuracy_pdf
            accuracy = pdf.std*2

        else:
            accuracy = self.accuracy_plusminus
            if self.accuracy_eopr is None:
                # Use Uniform distribution
                pdf = Pdf.from_stdev(0, self.accuracy_plusminus / np.sqrt(3))
            else:
                pdf = Pdf.from_itp(0, self.accuracy_eopr, Limit.from_plusminus(0, self.accuracy_plusminus))

        return MqaUncertaintyResult(
            stdev=pdf.std,
            accuracy=accuracy,
            pdf=pdf,  # PDF centered at 0
            degrees_freedom=degf,
            parent=parent
        )


@dataclass
class MqaMeasurement:
    ''' Info about the calibration/test measurement used to measure the measurand '''
    mte: MTE = field(default_factory=MTE)
    equation: str = ''  # For indirect
    mteindirect: dict[str, MTE] = field(default_factory=dict)
    testpoints: dict[str, float] = field(default_factory=dict)
    typebs: list[Typeb] = field(default_factory=list)
    typea: np.ndarray = None  # R&R data
    typea_numnew: int = None  # Num New (len(typea) if None)
    autocorrelation: bool = True  # Adjust for autocorr
    calibration: MqaCalibration = field(default_factory=MqaCalibration)
    interval: MqaInterval = field(default_factory=MqaInterval)

    def mode(self) -> str:
        if self.mte.quantity is not None:
            return 'quantity'
        if self.mteindirect:
            return 'indirect'
        if self.mte.equipment:
            return 'equipment'
        return 'tolerance'

    def gummodel(self) -> Model:
        ''' GUM uncertainty model for indirect measurements '''
        assert self.equation

        gummodel = Model(self.equation)
        for name, qty in self.mteindirect.items():
            v = gummodel.var(name)
            if v is not None:
                testpoint = self.testpoints.get(name, 0)
                result = qty.uncertainty(testpoint)
                v.measure(testpoint)
                v.typeb(std=result.accuracy/2)
        return gummodel

    def mte_uncertainty(self, testpoint: float) -> tuple[float, MqaUncertaintyResult]:
        ''' Get testpoint and uncertainty of M&TE '''
        if self.equation:
            # Indirect. Calculate children quantities first
            gumparents = []
            for mte in self.mteindirect.values():
                if mte.quantity is not None:
                    gumparents.append(mte.quantity.calculate())

            # Then GUM
            gumresult = self.gummodel().calculate_gum()
            uncert = gumresult.uncertainty['f1']
            nominal = gumresult.expected['f1']
            degf = gumresult.degf['f1']
            pdf = Pdf.from_stdev(nominal, uncert)

            return nominal, MqaUncertaintyResult(
                stdev=uncert,
                accuracy=uncert,
                pdf=pdf,
                degrees_freedom=degf,
                parent=None,
                gum=gumresult,
                gumparents=gumparents
            )
        else:
            return testpoint, self.mte.uncertainty(testpoint)

    def uncertainty(self, testpoint: float) -> MqaUncertaintyResult:
        ''' Uncertainty of the measurement (standard + TypeA + TypeBs) '''
        rv = RandomVariable()
        if self.typea is not None:
            rv.measure(self.typea, num_new_meas=self.typea_numnew)
        else:
            rv.measure(testpoint)
        rv._typeb[:] = self.typebs[:]

        mte_nominal, mte_uncert = self.mte_uncertainty(testpoint)
        rv._typeb.append(Typeb('normal', nominal=mte_nominal, std=mte_uncert.pdf.std))
        uncert = rv.uncertainty
        accuracy = mte_uncert.accuracy

        pdf = Pdf.from_stdev(0, uncert)  # PDF centered about 0, to be shifted to measured value
        # COULD DO: Add TypeA and Type B info to UncertaintyResult for reporting
        return MqaUncertaintyResult(
            stdev=uncert,
            accuracy=accuracy,
            pdf=pdf,
            degrees_freedom=rv.degrees_freedom,
            parent=mte_uncert.parent,
            gum=mte_uncert.gum
        )

    @property
    def mean(self) -> float:
        ''' The mean/average value of the Type A measurement data '''
        if self.typea is not None:
            return np.nanmean(self.typea)
        return None
