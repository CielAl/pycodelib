"""
    The basic implementation of engines.
"""
from .skeletal import AbstractEngine, AbstractIteratorBuilder
from typing import Tuple, Dict, Any, List
import torch
import torch.nn as nn
from tqdm import tqdm
from pycodelib.debug import Debugger
from .model_stats import AbstractModelStats


class BaseEngine(AbstractEngine):
    """
        Engine for the skin cancer project. Patient-level.
    """
    def __init__(self, device: torch.device,
                 model: nn.Module,
                 loss: nn.Module,
                 iterator_getter: AbstractIteratorBuilder,
                 model_stats: AbstractModelStats,
                 ):

        super().__init__(device, model, loss, iterator_getter)

        self.model_stats: AbstractModelStats = model_stats
        self.membership = None  # todo - EM
        # debug
        self.debugger = Debugger(__name__)
        # explicit printing
        self.debugger.level = -1
        
    def model_eval(self, data_batch: List) -> Tuple[torch.Tensor, torch.Tensor]:
        """

        Args:
            data_batch: List[batches_var1, batches_var2, ...]. Each list element is a batch of data variable
             (i.e. [batch_img, batch_label])
        Returns:
            loss: the loss
            prediction: the prediction of the output layer
        """
        img, label, *rest = data_batch
        img = img.to(self.device)
        label = label.type('torch.LongTensor').to(self.device)
        prediction = self.model(img)
        loss = self.loss(prediction, label)
        return loss, prediction

    def on_start(self, state: Dict[str, Any], **kwargs):
        """
            Initiation of procedure
        Args:
            state:

        Returns:

        """

        # if test mode: need to load the progress bar here
        if not state['train']:
            state['iterator'] = tqdm(state['iterator'])
        self.model_stats.hook(self.on_start.__name__, state)

    def on_end(self, state: Dict[str, Any], **kwargs):
        """
        End of procedure
        Args:
            state: The dict storage to pass variables.

        Returns:
            None
        """
        self.model_stats.hook(self.on_end.__name__, state)
        torch.cuda.empty_cache()

    def on_sample(self, state: Dict[str, Any], **kwargs):
        """

        Args:
            state:

        Returns:

        """
        state['sample'].append(state['train'])
        self.model_stats.hook(self.on_sample.__name__, state)
        self.model.train(state['train'])

    def on_update(self, state: Dict[str, Any], **kwargs):
        """
            Pass
        Args:
            state:

        Returns:

        """
        self.model_stats.hook(self.on_update.__name__, state)

    def on_start_epoch(self, state: Dict[str, Any], **kwargs):
        """
            Reset all meters and mark mode as training.
            Also initialize the tqdm progress bar
        Args:
            state:

        Returns:

        """
        state['train'] = True
        state['iterator'] = tqdm(state['iterator'])
        self.model_stats.hook(self.on_start_epoch.__name__, state)

    def on_end_epoch(self, state: Dict[str, Any], **kwargs):
        """
            Initialize the test procedure to evaluate the model performance.
            Mark mode as False.
        Args:
            state:

        Returns:

        """
        self.model_stats.hook(self.on_end_epoch.__name__, state)
        state['train'] = False
        # todo testing the value() breakpoints
        torch.cuda.empty_cache()
        self.engine.test(self, self.iterator_getter.get_iterator(mode=False, shuffle=False))

    def on_forward(self, state: Dict[str, Any], **kwargs):
        """
            On forward. Perform measurements for both train and test.
        Args:
            state:

        Returns:

        """
        self.model_stats.hook(self.on_forward.__name__, state)
