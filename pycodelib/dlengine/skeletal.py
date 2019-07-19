from typing import Callable, Tuple, Dict, Any, List
import torch
from torchnet.engine import Engine
from abc import ABC, abstractmethod
from torch.optim.optimizer import Optimizer
import torch.nn as nn
from pycodelib.dataset import Dataset
from torchnet.meter.meter import Meter
import logging
logging.basicConfig(level=logging.debug)


class IteratorBuilder(ABC, Dataset):
    BATCH_SIZE: int = 32
    @classmethod
    @abstractmethod
    def get_iterator(cls, mode, shuffle, num_workers=6, drop_last=False,  pin_memory=True, batch_size=BATCH_SIZE):
        ...


class AbstractEngine(Callable):

    @property
    def engine(self):
        return self._engine

    @property
    def hooks(self):
        return self._hooks

    @property
    def model(self):
        return self._model

    @property
    def loss(self):
        return self._loss

    @abstractmethod
    def model_eval(self, data_batch: List) -> Tuple[torch.Tensor, torch.Tensor]:
        ...

    def __call__(self, data_batch: List) -> Tuple[torch.Tensor, torch.Tensor]:
        loss, pred = self.model_eval(data_batch)
        return loss, pred

    def __init__(self, device: torch.device, model: nn.Module, loss: nn.Module, iterator_getter: IteratorBuilder):
        self._maxepoch: int = -1
        self._engine: Engine = Engine()
        self._hooks: Dict[str, Callable] = dict({
                    "on_start": self.on_start,
                    "on_start_epoch": self.on_start_epoch,  # train exclusive
                    "on_sample": self.on_sample,            # get data point from the data-loader
                    "on_forward": self.on_forward,          # the only phase that both train/test applies
                    "on_update": self.on_update,            # train exclusive, after the "step" of backward updating
                    "on_end_epoch": self.on_end_epoch,      # train exclusive - usually test is invoked here
                    "on_end": self.on_end,
        })
        self.device: torch.device = device
        self._model: nn.Module = model.to(device)
        self._loss: nn.Module = loss
        self.iterator_getter: IteratorBuilder = iterator_getter
        self.meter_dict: Dict = dict()

    def process(self, maxepoch: int, optimizer: Optimizer):
        self._maxepoch = maxepoch
        self.engine.train(self, self.iterator_getter.get_iterator(mode=True, shuffle=True),
                          maxepoch=self._maxepoch, optimizer=optimizer)

    @abstractmethod
    def on_start(self, state: Dict[str, Any]):
        ...

    @abstractmethod
    def on_start_epoch(self, state: Dict[str, Any]):
        ...

    @abstractmethod
    def on_sample(self, state: Dict[str, Any]):
        ...

    @abstractmethod
    def on_forward(self, state: Dict[str, Any]):
        ...

    @abstractmethod
    def on_update(self, state: Dict[str, Any]):
        ...

    @abstractmethod
    def on_end_epoch(self, state: Dict[str, Any]):
        ...

    @abstractmethod
    def on_end(self, state: Dict[str, Any]):
        ...

    def add_meter(self, name: str, meter: Meter):
        self.meter_dict[name] = meter
