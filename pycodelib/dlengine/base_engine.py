"""
    The basic implementation of engines.
"""
from .skeletal import AbstractEngine, AbstractIteratorBuilder
from .multi_instance_meter import MultiInstanceMeter
from pycodelib.patients import SheetCollection, CascadedPred
from typing import Tuple, Dict, Any, Sequence, List, Callable, Union
import torch
import torch.nn as nn
from torchnet.meter import AverageValueMeter, ClassErrorMeter, ConfusionMeter, AUCMeter
from tqdm import tqdm
from pycodelib.debug import Debugger
import numpy as np

softmax: Callable = nn.Softmax(dim=1)


class SkinEngine(AbstractEngine):
    """
        Engine for the skin cancer project. Patient-level.
    """
    def __init__(self, device: torch.device,
                 model: nn.Module,
                 loss: nn.Module,
                 iterator_getter: AbstractIteratorBuilder,
                 val_phases: Sequence[str],
                 patient_col: SheetCollection,
                 sub_class_list: Sequence,
                 class_partition: Sequence[Union[int, Sequence[int]]]):

        super().__init__(device, model, loss, iterator_getter)
        self.val_phases: Sequence[str] = val_phases

        self.patient_info = patient_col
        self.sub_class_list = sub_class_list
        self.class_partition: np.ndarray = np.asarray(class_partition)
        self.patient_pred = CascadedPred(self.patient_info,
                                         class_list=sub_class_list,
                                         partition=self.class_partition)
        self.num_classes: int = self.class_partition.size

        self.membership = None  # todo - EM

        # meters
        self.add_meter('patch_accuracy_meter', ClassErrorMeter(accuracy=True))
        self.add_meter('conf_meter', ConfusionMeter(self.num_classes, normalized=False))
        self.add_meter('loss_meter', AverageValueMeter())
        self.add_meter('multi_instance_meter', MultiInstanceMeter(self.patient_info, self.patient_pred))
        self.add_meter('auc_meter', AUCMeter())

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

    def on_start(self, state: Dict[str, Any]):
        """
            Initiation of procedure
        Args:
            state:

        Returns:

        """

        # if test mode: need to load the progress bar here
        if not state['train']:
            state['iterator'] = tqdm(state['iterator'])

    def on_end(self, state: Dict[str, Any]):
        """
        End of procedure
        Args:
            state: The dict storage to pass variables.

        Returns:
            None
        """
        torch.cuda.empty_cache()

    def on_sample(self, state: Dict[str, Any]):
        """

        Args:
            state:

        Returns:

        """
        state['sample'].append(state['train'])
        self.model.train(state['train'])

    def on_update(self, state: Dict[str, Any]):
        """
            Pass
        Args:
            state:

        Returns:

        """
        ...

    def on_start_epoch(self, state: Dict[str, Any]):
        """
            Reset all meters and mark mode as training.
            Also initialize the tqdm progress bar
        Args:
            state:

        Returns:

        """
        self._reset_meters()
        state['train'] = True
        state['iterator'] = tqdm(state['iterator'])

    def on_end_epoch(self, state: Dict[str, Any]):
        """
            Initialize the test procedure to evaluate the model performance.
            Mark mode as False.
        Args:
            state:

        Returns:

        """
        state['train'] = False
        # todo testing the value() breakpoints
        self.meter_dict['multi_instance_meter'].value()
        self.engine.test(self, self.iterator_getter.get_iterator(mode=False, shuffle=False))

    def _reset_meters(self):
        for k, v in self.meter_dict.items():
            v.reset()

    def _evaluation(self, state):
        """
            (1) Print all results: loss/auc/conf
            (2) Perform Patient-level evaluation
        Args:
            state: See on_start

        Returns:

        """
        # todo test
        ...
        print('[Epoch %d] Testing Loss: %.4f (Accuracy: %.2f%%)' % (
            state['epoch'],
            self.meter_dict['loss_meter'].value()[0],
            self.meter_dict['patch_accuracy_meter'].value()[0])
        )
        print(self.meter_dict['conf_meter'].value())
        # todo self._fetch_prediction()

    def on_forward(self, state: Dict[str, Any]):
        """
            On forward. Perform measurements for both train and test.
        Args:
            state:

        Returns:

        """

        # retrieve prediction and loss.

        # pred_data as the output score. Shape: N samples * M classes.
        pred_data: torch.Tensor = state['output'].data
        loss_scalar: torch.Tensor = state['loss'].data.detach().cpu().numpy()

        # retrieve input data
        img_new, label, img, filename, *rest = state['sample']
        label: torch.LongTensor = label.type("torch.LongTensor")

        part_min = min([min(x) for x in self.class_partition])
        part_max = max([max(x) for x in self.class_partition])
        assert part_min <= label.max() <= part_max \
            and \
            part_min <= label.min() <= part_max,   \
            f"batch_label exceed range {(label.min(), label.max())}. Expected in {self.class_partition}"

        # Add Loss Measurements per batch
        # breakpoint()
        self.meter_dict['loss_meter'].add(loss_scalar)
        # Add accuracy (patch) per batch
        # todo check length of label
        self.meter_dict['patch_accuracy_meter'].add(pred_data, label)

        # If 1-d vector (in case of batch-of-one), adding leading singleton dim.
        # Otherwise the confusion_meter may mistreat the 1-d vector as a vector of
        # score of positive classes.
        if pred_data.ndimension() == 1:
            pred_data = pred_data[None]
        self.meter_dict['conf_meter'].add(pred_data, label)

        # Posterior score
        pred_softmax = softmax(pred_data)
        patch_name_score: Tuple = (filename, pred_softmax)
        """ 
            Note: the score per sample (single data point) is not reduced. Values of all categories are retained.
        """
        self.meter_dict['multi_instance_meter'].add(patch_name_score)

        if self.num_classes == 2:
            self.meter_dict['auc_meter'].add(pred_softmax[:, 1], label)
