from abc import ABC
import torch
import torch.nn as nn
import numpy as np
from typing import Set, Dict, Any, Sequence, Union, Callable, Tuple
from pycodelib.patients import CascadedPred, SheetCollection
from torchnet.meter.meter import Meter
from torchnet.meter import AverageValueMeter, ClassErrorMeter, ConfusionMeter, AUCMeter
from ..multi_instance_meter import MultiInstanceMeter
from ..skeletal import EngineHooks


softmax: Callable = nn.Softmax(dim=1)


class BaseModelStats(EngineHooks, ABC):
    ...


class AbstractModelStats(BaseModelStats, ABC):
    # HOOKS_NAME: Set[str] = {"on_start"}
    PHASE_NAMES: Set[str] = {'train', 'val'}

    def __init__(self, val_phases: Dict[bool, str] = None):
        super().__init__()
        self.meter_dict: Dict[str, Any] = dict()

        if val_phases is None:
            val_phases = dict({True: "train", False: "val"})
        self.val_phases: Dict[bool, str] = val_phases
        assert set(self.val_phases.values()).issubset(AbstractModelStats.PHASE_NAMES),\
            f"Invalid Phase Names{self.val_phases}" \
            f"Expected from {AbstractModelStats.PHASE_NAMES}"

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

    def reset_meters(self):
        for k, v in self.meter_dict.items():
            v.reset()

    def get_meter(self, name):
        return self.meter_dict[name]

    def phase_name(self, state):
        phase = self.val_phases.get(state['train'], None)
        return phase


class DefaultStats(AbstractModelStats):

    def __init__(self,
                 num_classes: int,
                 patient_info: SheetCollection,
                 patient_pred: CascadedPred,
                 sub_class_list: Sequence,
                 class_partition: Sequence[Union[int, Sequence[int]]],
                 positive_class: Sequence[int] = None,
                 val_phases: Dict[bool, str] = None
                 ):
        super().__init__(val_phases)
        self.num_classes = num_classes
        self.patient_info = patient_info
        self.patient_pred = patient_pred

        self._positive_class = positive_class

        self.sub_class_list = sub_class_list
        self.class_partition: np.ndarray = np.asarray(class_partition)

        # class_list=sub_class_list,
        # partition=self.class_partition)
        self.num_classes: int = self.class_partition.size

        self.add_meter('patch_accuracy_meter', ClassErrorMeter(accuracy=True))
        self.add_meter('conf_meter', ConfusionMeter(self.num_classes, normalized=False))
        self.add_meter('loss_meter', AverageValueMeter())
        self.add_meter('multi_instance_meter',
                       MultiInstanceMeter(self.patient_info,
                                          self.patient_pred,
                                          positive_class=MultiInstanceMeter.default_positive(self.class_partition,
                                                                                             self._positive_class)
                                          )
                       )
        self.add_meter('auc_meter', AUCMeter())

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

    def on_sample(self, state: Dict[str, Any], **kwargs):
        """
            Hook that is invoked immediately after a sample (possibly a batch) is extracted from the DataLoader.
        Args:
            state: See on_start

        Returns:

        """
        ...

    def on_update(self, state: Dict[str, Any], **kwargs):
        """
            Hook that is invoked after the on_forward, i.e. after the back-propagation and optimization.
            Only available in training mode.
        Args:
            state: See on_start

        Returns:

        """
        ...

    def on_end_epoch(self, state: Dict[str, Any], **kwargs):
        """
            Hook that is invoked on the termination of each epoch.
            Only available in training mode.
        Args:
            state:

        Returns:

        """
        self._evaluation(state)

    def on_end(self, state: Dict[str, Any], **kwargs):
        """
            Hook that is invoked on finalizing of the engine procedure.
        Args:
            state: See on_start

        Returns:

        """
        if not state['train']:
            self._evaluation(state)

    def on_forward(self, state: Dict[str, Any], **kwargs):
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

        # todo multiclass generalization: probably need a new meter
        # if self.num_classes == 2:
        self.meter_dict['auc_meter'].add(pred_softmax[:, self._positive_class[0]], label)

    def on_start(self, state: Dict[str, Any], *args, **kwargs):
        ...

    def _evaluation(self, state):
        """
            (1) Print all results: loss/auc/conf
            (2) Perform Patient-level evaluation
        Args:
            state: See on_start

        Returns:

        """
        # todo test
        phase = self.phase_name(state)
        if phase is None:
            #  skip
            return
        scores_all_cate, labels_all_cate, row_names, col_names = self.meter_dict['multi_instance_meter'].value()

        loss = self.meter_dict['loss_meter'].value()[0]
        patch_acc = self.meter_dict['patch_accuracy_meter'].value()[0]
        patch_auc = self.meter_dict['auc_meter'].value()[0]
        patch_conf = self.meter_dict['conf_meter'].value()
        # todo move to engine
        basic_verbose = f"{phase} - [Epoch {state['epoch']}/{state['maxepoch']}]. " \
            f"Loss:= {loss:.5f} " \
            f"Patch accuracy:= {patch_acc:.2f}  " \
            f"Patch AUC:= {patch_auc:.2f}"
        print(basic_verbose)
        print(patch_conf)
        breakpoint()
