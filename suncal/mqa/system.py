''' System of multiple MQA Quantities '''
from .equipment import EquipmentList
from .mqa import MqaQuantity
from .result import MqaSystemResult
from .guardband import MqaGuardbandRuleset


class MqaSystem:
    ''' Measurement system '''
    def __init__(self):
        self.quantities: list[MqaQuantity] = []
        self.equipment: EquipmentList = EquipmentList()
        self.gbrules: MqaGuardbandRuleset = MqaGuardbandRuleset()

    def add(self):
        ''' Add a quantity to the system '''
        qty = MqaQuantity()
        qty.equipmentlist = self.equipment
        qty.measurand.name = 'Quantity'
        self.quantities.append(qty)
        return qty

    def mqa_mode(self) -> int:
        ''' Determine what is enabled (relaibility decay and/or costs) '''
        mode = MqaQuantity.Mode.BASIC
        for qty in self.quantities:
            mode = max(mode, qty.mqa_mode)
        return mode

    def calculate(self) -> MqaSystemResult:
        ''' Calculate all quantities '''
        results = []
        for qty in self.quantities:
            results.append(qty.calculate())
        return MqaSystemResult(results)
