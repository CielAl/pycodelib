"""
    Abstract classes.
"""
from typing import Callable, Tuple, Dict, Any, List
import torch
from torch.utils.data import DataLoader
from torchnet.engine import Engine
from abc import ABC, abstractmethod
from torch.optim.optimizer import Optimizer
import torch.nn as nn
# from torch.utils.data import Dataset as TorchDataSet
from torchnet.meter.meter import Meter
import logging
logging.basicConfig(level=logging.DEBUG)


class AbstractIteratorBuilder(ABC):
    """
        The Encapsulation to assign the DataLoader given the phases/modes (i.e. train/val).
    """
    BATCH_SIZE: int = 32
    @classmethod
    @abstractmethod
    def get_iterator(cls, mode, shuffle, num_workers=4, drop_last=False,  pin_memory=True, batch_size=4) -> DataLoader:
        """
        Args:
            mode: True if train, otherwise False.
            shuffle: True if data are shuffled each epoch.
            num_workers: Number of CPU.
            drop_last:  Drop the last batch if smaller than batch_size.
            pin_memory: Use pin memory (Details in DataLoader of pytorch)
            batch_size: Size of batch

        Returns:
            The DataLoader given the mode.
        """
        ...

    @property
    @abstractmethod
    def datasets_collection(self):
        """
        Returns:
            The field which contains all dataset, corresponding to all modes.
        """
        ...


class AbstractEngine(Callable):
    """
        The abstraction of training using engine. The class is Callable by itself, where its __call__ is defined to
        evaluate the model as serve as the "network/model" parameter of tnt/engine
    """

    @property
    def engine(self) -> Engine:
        """
        Returns:
            The engine property.
        """
        return self._engine

    @property
    def hooks(self) -> Dict[str, Callable]:
        """
        Returns:
            The hooks property
        """
        return self._hooks

    @property
    def model(self) -> nn.Module:
        """

        Returns:
            The model property. Typically the neural network.
        """
        return self._model

    @property
    def loss(self) -> nn.Module:
        """

        Returns:
            The loss property. The definition of loss append to the network.
        """
        return self._loss

    @abstractmethod
    def model_eval(self, data_batch: List) -> Tuple[torch.Tensor, torch.Tensor]:
        """
            The helper function that evaluate the model that should be invoked in __call__.
            It is separated from __call__ to more or less loose the semantic coupling that "__call__", which is
            a more general procedure, should perform the model evaluation.
        Args:
            data_batch: List batch variables. See detail implementations.
        Returns:
            loss: Evaluated loss value(s).
            prediction: Model output.
        """
        ...

    def __call__(self, data_batch: List) -> Tuple[torch.Tensor, torch.Tensor]:
        """
            Work as the 1st callable parameter of self.engine.train
        Args:
            data_batch: Input batch. See model_eval.

        Returns:
            loss
            prediction
        """
        loss, pred = self.model_eval(data_batch)
        return loss, pred

    def __init__(self,
                 device: torch.device,
                 model: nn.Module,
                 loss: nn.Module,
                 iterator_getter: AbstractIteratorBuilder):
        """

        Args:
            device: Device that hosts all variables/models, i.e. the GPU
            model:  The callable model that is to be trained/evaluated.
            loss:   The loss that is appended to the output layer.
            iterator_getter:    The DataLoader assignment by mode.
        """
        # Initializing the maxepoch field.
        self._maxepoch: int = -1
        # The torchnet.engine object.
        self._engine: Engine = Engine()
        # All possible hooks that is called by engine.train and engine.test
        self._hooks: Dict[str, Callable] = dict({
                    "on_start": self.on_start,              # start of the procedure
                    "on_start_epoch": self.on_start_epoch,  # train exclusive
                    "on_sample": self.on_sample,            # get data point from the data-loader
                    "on_forward": self.on_forward,          # the only phase that both train/test applies
                    "on_update": self.on_update,            # train exclusive, after the "step" of backward updating
                    "on_end_epoch": self.on_end_epoch,      # train exclusive - usually test is invoked here
                    "on_end": self.on_end,                  # end of the procedure
        })
        # The device which host the model and all variables.
        self.device: torch.device = device
        # The neural network model.
        self._model: nn.Module = model.to(device)
        # Loss
        self._loss: nn.Module = loss
        # Iterator getter
        self.iterator_getter: AbstractIteratorBuilder = iterator_getter
        # Empty Meter storage.
        self.meter_dict: Dict = dict()

    def process(self, maxepoch: int, optimizer: Optimizer):
        """
            The exposed interface to user, which is an encapsulation of engine.train, where the 1st callable
             parameter of engine.train is self (i.e. defined by __call__)
        Args:
            maxepoch:
            optimizer:

        Returns:

        """
        self._maxepoch = maxepoch
        self.engine.train(self, self.iterator_getter.get_iterator(mode=True, shuffle=True),
                          maxepoch=self._maxepoch, optimizer=optimizer)

    @abstractmethod
    def on_start(self, state: Dict[str, Any]):
        """
            Hook that is executed on initiation.
        Args:
            state: A dict storage to pass the variables with initial value:
                state = {
                    'network': network,
                    'iterator': iterator,
                    'maxepoch': maxepoch,
                    'optimizer': optimizer,
                    'epoch': 0,
                    't': 0,
                    'train': True,
                }
        Returns:

        """
        ...

    @abstractmethod
    def on_start_epoch(self, state: Dict[str, Any]):
        """
            Hook that is invoked on each single epoch. Typically reset the meters or setting the modes
            of dataset (train or test).
            Only available in the training mode.
        Args:
            state: See on_start

        Returns:

        """
        ...

    @abstractmethod
    def on_sample(self, state: Dict[str, Any]):
        """
            Hook that is invoked immediately after a sample (possibly a batch) is extracted from the DataLoader.
        Args:
            state: See on_start

        Returns:

        """
        ...

    @abstractmethod
    def on_forward(self, state: Dict[str, Any]):
        """
            Hook that is invoked after the forwarding (execution of model) and loss.backward, and before on_update.
        Args:
            state: See on_start

        Returns:

        """
        ...

    @abstractmethod
    def on_update(self, state: Dict[str, Any]):
        """
            Hook that is invoked after the on_forward, i.e. after the back-propagation and optimization.
            Only available in training mode.
        Args:
            state: See on_start

        Returns:

        """
        ...

    @abstractmethod
    def on_end_epoch(self, state: Dict[str, Any]):
        """
            Hook that is invoked on the termination of each epoch.
            Only available in training mode.
        Args:
            state:

        Returns:

        """
        ...

    @abstractmethod
    def on_end(self, state: Dict[str, Any]):
        """
            Hook that is invoked on finalizing of the engine procedure.
        Args:
            state: See on_start

        Returns:

        """
        ...

    def add_meter(self, name: str, meter: Meter):
        """
            Add a meter to the engine.
        Args:
            name: Name of the meter, served as the key in self.meter_dict.
            meter: The meter object
        Returns:
            None.
        """
        self.meter_dict[name] = meter
