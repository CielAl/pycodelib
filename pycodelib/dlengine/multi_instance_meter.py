from typing import Dict, Any, Tuple, Hashable, List, Sequence, Union
import torch
import numpy as np
import numbers
from torchnet.meter.meter import Meter


class MultiInstanceMeter(Meter):

    @property
    def instance_map(self):
        return self._instance_map

    def __init__(self):
        super().__init__()
        self._instance_map: Dict[Hashable, Any] = dict()

    def add(self, elements: Tuple[Union[torch.Tensor, Sequence[Hashable]], Union[torch.Tensor, Sequence]]):
        keys, values = elements
        if torch.is_tensor(keys):
            keys = keys.cpu().squeeze().numpy()
        if isinstance(keys, Hashable):
            keys = np.asarray([keys])
        if torch.is_tensor(values):
            values = np.atleast_1d(values.cpu().squeeze().numpy())
        elif isinstance(values, numbers.Number):
            values = np.asarray([values])
        elif isinstance(values, Sequence):
            values = np.asarray(values)

        assert len(keys) == values.shape[0]
        for k, v in zip(keys, values):
            default_store: List = self.instance_map.get(k, [])
            self.instance_map.update({k: default_store})
            default_store.append(v)

    def add_kv(self, keys: Union[torch.Tensor, Sequence[Hashable]], values: Union[torch.Tensor, Sequence]):
        self.add((keys, values))

    def value(self):
        return self.instance_map.keys(), self.instance_map.values()

    def reset(self):
        self._instance_map.clear()
