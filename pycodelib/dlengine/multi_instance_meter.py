from typing import Dict, Any, Tuple, Hashable, List

from torchnet.meter.meter import Meter


class MultiInstanceMeter(Meter):

    @property
    def instance_map(self):
        return self._instance_map

    def __init__(self):
        super().__init__()
        self._instance_map: Dict[Hashable, Any] = dict()

    def add(self, value: Tuple[Hashable, Any]):
        key = value[0]
        default_store: List = self.instance_map.get(key, [])
        self.instance_map.update({key: default_store})
        default_store.append(value[1])

    def value(self):
        return self.instance_map.keys(), self.instance_map.values()

    def reset(self):
        self._instance_map.clear()
