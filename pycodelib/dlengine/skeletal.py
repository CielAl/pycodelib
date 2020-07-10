"""
    Abstract classes.
"""
import os
from typing import Callable, Tuple, Dict, Any, List
import torch
from torch.utils.data import DataLoader
from torchnet.engine import Engine
from abc import ABC, abstractmethod
from torch.optim.optimizer import Optimizer
import torch.nn as nn
import numpy as np
# from torch.utils.data import Dataset as TorchDataSet
from pycodelib.dataset import AbstractDataset
from pycodelib.debug import Debugger
debugger = Debugger(__name__)
debugger.level = None


class AbstractIteratorBuilder(ABC):
    """
        The Encapsulation to assign the DataLoader given the phases/modes (i.e. train/val).
    """
    BATCH_SIZE: int = 32

    @classmethod
    @abstractmethod
    def get_iterator(cls, mode, shuffle, num_workers=4, drop_last=False,  pin_memory=True, batch_size=4,
                     truncate_size: float = np.inf,
                     flatten_output: bool = False
                     ) -> DataLoader:
        """
        Args:
            mode: True if train, otherwise False.
            shuffle: True if data are shuffled each epoch.
            num_workers: Number of CPU.
            drop_last:  Drop the last batch if smaller than batch_size.
            pin_memory: Use pin memory (Details in DataLoader of pytorch)
            batch_size: Size of batch
            truncate_size: See AbstractDataset
            flatten_output: See AbstractDataset
        Returns:
            The DataLoader given the mode.
        """
        ...

    @property
    @abstractmethod
    def data_sets_collection(self):
        """
        Returns:
            The field which contains all dataset, corresponding to all modes.
        """
        ...


class EngineHooks(ABC):

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

    def hook(self, name, state: Dict[str, Any]):
        return self.engine.hook(name, state)

    def __init__(self):
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
        self.engine.hooks = self.hooks

    @abstractmethod
    def on_start(self, state: Dict[str, Any], **kwargs):
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
    def on_start_epoch(self, state: Dict[str, Any], **kwargs):
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
    def on_sample(self, state: Dict[str, Any], **kwargs):
        """
            Hook that is invoked immediately after a sample (possibly a batch) is extracted from the DataLoader.
        Args:
            state: See on_start

        Returns:

        """
        ...

    @abstractmethod
    def on_forward(self, state: Dict[str, Any], **kwargs):
        """
            Hook that is invoked after the forwarding (execution of model) and loss.backward, and before on_update.
        Args:
            state: See on_start

        Returns:

        """
        ...

    @abstractmethod
    def on_update(self, state: Dict[str, Any], **kwargs):
        """
            Hook that is invoked after the on_forward, i.e. after the back-propagation and optimization.
            Only available in training mode.
        Args:
            state: See on_start

        Returns:

        """
        ...

    @abstractmethod
    def on_end_epoch(self, state: Dict[str, Any], *args, **kwargs):
        """
            Hook that is invoked on the termination of each epoch.
            Only available in training mode.
        Args:
            state:

        Returns:

        """
        ...

    @abstractmethod
    def on_end(self, state: Dict[str, Any], **kwargs):
        """
            Hook that is invoked on finalizing of the engine procedure.
        Args:
            state: See on_start

        Returns:

        """
        ...


class AbstractEngine(Callable, EngineHooks):
    """
        The abstraction of training using engine. The class is Callable by itself, where its __call__ is defined to
        evaluate the model as serve as the "network/model" parameter of tnt/engine
    """

    def dump_file_name(self, title: str, ext: str):
        title = str(title)
        filename = f'{self.engine_name}_{title}.{ext}'
        full_path = os.path.join(self.model_export_path, filename)
        return full_path

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

    @property
    def optimizer(self) -> Optimizer:
        return self._optimizer

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
        debugger.log(type(data_batch), len(data_batch))
        loss, pred = self.model_eval(data_batch)
        debugger.log(loss, pred)
        return loss, pred

    def __init__(self,
                 device: torch.device,
                 model: nn.Module,
                 loss: nn.Module,
                 iterator_getter: AbstractIteratorBuilder,
                 model_export_path: str,
                 engine_name: str = '',
                 optimizer: Optimizer = None,
                 img_key: str = None,
                 label_key: str = None,
                 filename_key: str = None,
                 index_key: str = None):
        """

        Args:
            device: Device that hosts all variables/models, i.e. the GPU
            model:  The callable model that is to be trained/evaluated.
            loss:   The loss that is appended to the output layer.
            iterator_getter:    The DataLoader assignment by mode.
        """
        super().__init__()

        # cache the kwargs arguments of dataloader (for testing)
        self._test_data_loader_kwargs_cache = dict()
        # Initializing the maxepoch field.
        self._maxepoch: int = -1
        self._optimizer: Optimizer = optimizer
        # The torchnet.engine object.
        # The device which host the model and all variables.
        self.device: torch.device = device
        # The neural network model.
        self._model: nn.Module = model.to(device)
        # Loss
        self._loss: nn.Module = loss
        # Iterator getter
        self.iterator_getter: AbstractIteratorBuilder = iterator_getter
        self.model_export_path = model_export_path
        if not os.path.exists(self.model_export_path):
            os.mkdir(self.model_export_path)
        self.engine_name = engine_name

        self.img_key: str = img_key
        self.label_key: str = label_key
        self.filename_key: str = filename_key
        self.index_key: str = index_key
        # Empty Meter storage.

    def process(self, maxepoch: int = -1, optimizer: Optimizer = None, batch_size: int = 4, num_workers: int = 0,
                mode: bool = True,
                flatten_output: bool = False,
                truncate_size: Dict[bool, float] = None):
        """
            The exposed interface to user, which is an encapsulation of engine.train, where the 1st callable
             parameter of engine.train is self (i.e. defined by __call__)
        Args:
            maxepoch:
            optimizer:
            num_workers:
            batch_size:
            mode (bool):
            flatten_output:
            truncate_size:
        Returns:

        """
        debugger.log("process start")
        self._maxepoch = maxepoch
        self._optimizer = optimizer
        if truncate_size is None:
            truncate_size = {k: np.inf for k in [True, False]}

        dl = self.iterator_getter.get_iterator(
            mode=mode,
            shuffle=True,
            batch_size=batch_size,
            num_workers=num_workers,
            truncate_size=truncate_size[mode],
            flatten_output=flatten_output

        )
        self._test_data_loader_kwargs_cache = {
            'batch_size': batch_size,
            'num_workers': num_workers,
            'truncate_size': truncate_size[False],
            'flatten_output': flatten_output
        }
        if mode:
            assert self._maxepoch > 0
            assert optimizer is not None
            self.engine.train(self,
                              dl,
                              maxepoch=self._maxepoch,
                              optimizer=optimizer)
        else:

            self.engine.test(self, dl)

    def label_collator(self, labels):
        raise NotImplemented


class IteratorBuilderDictSet(AbstractIteratorBuilder):
    @staticmethod
    def __validate_dataset_dict(data_sets):
        for x in data_sets.values():
            if not isinstance(x, AbstractDataset):
                raise TypeError(f"Expect AbstractDataset. Got {type(data_sets)}")

    @property
    def data_sets_collection(self):
        """

        Returns:
            Collection of Datasets per mode.
        """
        return self._data_sets_collection

    def __init__(self, data_sets: Dict[bool, AbstractDataset]):
        super().__init__()
        IteratorBuilderDictSet.__validate_dataset_dict(data_sets)
        self._data_sets_collection: Dict[bool, AbstractDataset] = data_sets

    @classmethod
    @abstractmethod
    def get_iterator(cls, mode, shuffle, num_workers=4, drop_last=False,  pin_memory=True, batch_size=4,
                     truncate_size: float = np.inf,
                     flatten_output: bool = False) -> DataLoader:
        """
        Args:
            mode: True if train, otherwise False.
            shuffle: True if data are shuffled each epoch.
            num_workers: Number of CPU.
            drop_last:  Drop the last batch if smaller than batch_size.
            pin_memory: Use pin memory (Details in DataLoader of pytorch)
            batch_size: Size of batch
            truncate_size:
            flatten_output:
        Returns:
            The DataLoader given the mode.
        """
        ...
