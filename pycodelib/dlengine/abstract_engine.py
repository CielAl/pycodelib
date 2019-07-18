from typing import Callable, Tuple, Dict, Iterable, Any
import torch
from torchnet.engine import Engine
from abc import abstractmethod
from torch.optim.optimizer import Optimizer


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
    def model_eval(self, data_batch: Tuple) -> Tuple[torch.Tensor, torch.Tensor]:
        ...

    def __call__(self, data_batch) -> Tuple[torch.Tensor, torch.Tensor]:
        loss, pred = self.model_eval(data_batch)
        return loss, pred

    def __init__(self, model: Callable, loss: Callable, iterator: Iterable):
        self._engine = Engine()
        self._hooks: Dict[str, Callable] = dict({
                    "on_start": self.on_start,
                    "on_start_epoch": self.on_start_epoch,
                    "on_sample": self.on_sample,
                    "on_forward": self.on_forward,
                    "on_update": self.on_update,
                    "on_end_epoch": self.on_end_epoch,
                    "on_end": self.on_end,
        })
        self._model = model
        self._loss = loss
        self.iterator = iterator

    def process(self, maxepoch: int, optimizer: Optimizer):
        self.engine.train(self, self.iterator, maxepoch=maxepoch, optimizer=optimizer)

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


class Test(AbstractEngine):

    def __call__(self, s: Tuple):
        ...
