''' MQA Cost Model '''
from dataclasses import dataclass, field


@dataclass
class MqaItemCost:
    ''' Single-item false decision costs

        Args:
            cfa: Cost of a false accept (called cf in RP-19)
            cfr: Cost of a false reject
    '''
    cfa: float = 0
    cfr: float = 0


@dataclass
class MqaDowntime:
    ''' Downtime in days

        Args:
            cal: Calibration downtime (including shipping/storage)
            adj: Adjustment downtime
            rep: Repair downtime (including shipping/storage)
    '''
    cal: float = 0
    adj: float = 0
    rep: float = 0


@dataclass
class MqaAnnualCost:
    ''' Annual cost parameters

        Args:
            cal: Cost of one calibration
            adjust: Cost of adjustment
            repair: Cost to repair
            uut: Cost of a new UUT
            nuut: Number of UUTs in inventory
            suut: Spare readiness factor (0-1)
            spare_startup: Cost to start up a spare (cd in RP-19)
            downtime: Downtimes for calibration, adjustment, and repair
            pe: Probability of end-item use
    '''
    cal: float = 0
    adjust: float = 0
    repair: float = 0
    uut: float = 0
    nuut: int = 1
    suut: float = 1
    spare_startup: float = 0
    downtime: MqaDowntime = field(default_factory=MqaDowntime)
    pe: float = 1


@dataclass
class MqaCosts:
    ''' MQA cost model '''
    item: MqaItemCost = field(default_factory=MqaItemCost)
    annual: MqaAnnualCost = field(default_factory=MqaAnnualCost)

    @property
    def enabled(self) -> bool:
        ''' The costs are being used '''
        return (
            self.annual.cal != 0 or
            self.annual.adjust != 0 or
            self.annual.repair != 0 or
            self.item.cfa != 0 or
            self.item.cfr != 0)
