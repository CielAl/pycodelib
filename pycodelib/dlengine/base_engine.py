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
import pickle


class BaseEngine(AbstractEngine):
    """
        Engine for the skin cancer project. Patient-level.
    """
    def __init__(self,
                 device: torch.device,
                 model: nn.Module,
                 loss: nn.Module,
                 iterator_getter: AbstractIteratorBuilder,
                 model_stats: AbstractModelStats,
                 model_export_path: str = '.',
                 engine_name: str = ''
                 ):

        super().__init__(device, model, loss, iterator_getter, model_export_path, engine_name=engine_name)

        self.model_stats: AbstractModelStats = model_stats
        # debug
        self.debugger = Debugger(__name__)
        # explicit printing
        self.debugger.level = -1
        self.dummy = torch.tensor([0.]).requires_grad_(True)

    def label_collator(self, labels):
        return self.model_stats.label_collator(labels)

    def model_eval(self, data_batch: List) -> Tuple[torch.Tensor, torch.Tensor]:
        """

        Args:
            data_batch: List[batches_var1, batches_var2, ...]. Each list element is a batch of data variable
             (i.e. [batch_img, batch_label])
        Returns:
            loss: the loss
            prediction: the prediction of the output layer
        """
        # (img), label, mask, row, col, img_original, filenames, index, train_flag
        if len(data_batch) == 0 or len(data_batch[1]) == 0:
            dummy = self.dummy
            return dummy, dummy
        img, label_in, *rest = data_batch
        img = img.to(self.device)
        label = self.label_collator(label_in).type('torch.LongTensor').to(self.device)
        prediction = self.model(img)
        loss = self.loss(prediction, label)
        index = rest[-2].cpu().detach().numpy()
        softmax = nn.Softmax(dim=1)
        score = softmax(prediction).cpu().detach().numpy()
        self.model_stats.update_membership(score, index)
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
        if len(state['sample']) == 0:
            # print("bad sample")
            state['valid_sample'] = False
            return
        state['valid_sample'] = True
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
        if not state["valid_sample"]:
            return
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

    def on_end_epoch_checkpoint_helper(self, state):
        checkpoint = {
            'epoch': state['epoch'],
            'maxepoch': state['maxepoch'],
            'model_state_dict': self.model.state_dict(),
            'opt_state_dict': self._optimizer.state_dict()
        }
        chk_full = self.dump_file_name(f"model_{state['epoch']}", 'pth')
        torch.save(checkpoint, chk_full)

        model_full = self.dump_file_name(f"model_obj_{state['epoch']}", 'pth')
        torch.save(self.model, model_full)

        stats_file = self.dump_file_name(f"latent_{state['epoch']}", "pickle")
        with open(stats_file, 'wb') as handle:
            pickle.dump(self.model_stats.membership, handle, protocol=pickle.HIGHEST_PROTOCOL)

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
        torch.cuda.empty_cache()
        self.on_end_epoch_checkpoint_helper(state)
        self.engine.test(self, self.iterator_getter.get_iterator(mode=False, shuffle=False))
        # must revert it back
        state['train'] = True

    def on_forward(self, state: Dict[str, Any], **kwargs):
        """
            On forward. Perform measurements for both train and test.
        Args:
            state:

        Returns:

        """
        if not state["valid_sample"]:
            return
        self.model_stats.hook(self.on_forward.__name__, state)
