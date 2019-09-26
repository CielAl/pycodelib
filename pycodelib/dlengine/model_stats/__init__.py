from abc import ABC, abstractmethod
from typing import Set, Dict, Any, Sequence, Union, Callable, Tuple
import numpy as np
import torch
import torch.nn as nn
from torchnet.meter.meter import Meter
from torchnet.meter import AverageValueMeter, ClassErrorMeter, ConfusionMeter, AUCMeter
from torchnet.logger import VisdomPlotLogger  # ,VisdomLogger
from pycodelib.patients import CascadedPred, SheetCollection
from pycodelib.metrics import label_binarize_group

from ..multi_instance_meter import MultiInstanceMeter
from ..skeletal import EngineHooks
from pycodelib.dlengine.skeletal import AbstractEngine
import pickle

# noinspection SpellCheckingInspection
softmax: Callable = nn.Softmax(dim=1)


class BaseModelStats(EngineHooks, ABC):
    ...


class AbstractModelStats(BaseModelStats, ABC):
    # HOOKS_NAME: Set[str] = {"on_start"}
    PHASE_NAMES: Set[str] = {'train', 'val'}

    def __init__(self,
                 sub_class_list: Sequence,
                 class_partition: Sequence[Union[int, Sequence[int]]],
                 is_visualize: bool,
                 val_phases: Dict[bool, str] = None):
        super().__init__()
        self.meter_dict: Dict[str, Any] = dict()
        self._membership = None  # todo - EM
        self.sub_class_list = sub_class_list
        self.class_partition: np.ndarray = np.asarray(class_partition)

        self._is_visualize = is_visualize
        self._viz_logger_dict: Dict[Tuple[str, bool], Any] = dict()
        if val_phases is None:
            val_phases = dict({True: "train", False: "val"})
        self.val_phases: Dict[bool, str] = val_phases
        assert set(self.val_phases.values()).issubset(AbstractModelStats.PHASE_NAMES),\
            f"Invalid Phase Names{self.val_phases}" \
            f"Expected from {AbstractModelStats.PHASE_NAMES}"

        # alternatively, to plot test data of a epoch, one may perform engine.test first,
        # and use the state['epoch'] in the training phase + test result that are not yet reset in the meter
        # however, it makes the code even less readable, as there is even stronger semantic coupling
        # to the control flow of torchnet.engine
        # todo So simply count the epoch value here.
        self.__epoch_count: int = 0

    @abstractmethod
    def _evaluation(self, state):
        ...

    @property
    def epoch_count(self) -> int:
        return self.__epoch_count

    def epoch_incr(self):
        self.__epoch_count += 1

    def epoch_clear(self):
        self.__epoch_count = 0

    def on_end_epoch(self, state: Dict[str, Any], *args, **kwargs):
        self._evaluation(state)
        self.epoch_incr()

    def on_end(self, state: Dict[str, Any], **kwargs):
        if not state['train']:
            self._evaluation(state)
        self.epoch_clear()

    @property
    def is_visualize(self) -> bool:
        return self._is_visualize

    @property
    def membership(self):
        return self._membership

    def update_membership(self, scores, index):
        raise NotImplemented

    @abstractmethod
    def label_collator(self, labels, *args, **kwargs):
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
        if not hasattr(self, 'meter_dict') or self.meter_dict is None:
            self.meter_dict = dict()
        self.meter_dict[name] = meter

    def reset_meters(self):
        for k, v in self.meter_dict.items():
            v.reset()

    def get_meter(self, name):
        return self.meter_dict[name]

    def phase_name(self, state):
        phase = self.val_phases.get(state['train'], None)
        return phase

    def add_logger(self, viz_logger_construct,
                   plot_type: str,
                   name: str,
                   phases: Sequence[bool] = (True, False),
                   control_by_flag: bool = False,
                   **kwargs):
        do_create: bool = True if not control_by_flag else self.is_visualize
        if do_create:
            for phase in phases:
                title = "Train" if phase else "Test"
                title = f"{title} {name}"
                opts = {
                    'title': title
                }
                opts.update(kwargs)
                viz_logger = viz_logger_construct(plot_type, opts=opts)
                key = (name, phase)
                self._viz_logger_dict[key] = viz_logger

    def get_logger(self, name: str, phase: bool, raise_key: bool = True):
        key = (name, phase)
        if raise_key:
            return self._viz_logger_dict[key]
        return self._viz_logger_dict.get(key)

    def visualize_log(self, name, phase, *values):
        if self.is_visualize:
            viz_logger = self.get_logger(name, phase)
            viz_logger.log(*values)


class DefaultStats(AbstractModelStats):
    PATCH_ACC: str = 'patch_accuracy_meter'
    MULTI_INSTANCE: str = 'multi_instance_meter'
    PATCH_CONF: str = 'patch_conf_meter'
    LOSS_METER: str = 'loss_meter'
    PATCH_AUC: str = 'patch_auc_meter'

    VIZ_LOSS: str = 'Loss'
    VIZ_PATCH_TPR: str = 'Patch_TPR'
    VIZ_PATCH_TNR: str = 'Patch_TNR'
    VIZ_PATCH_AUC: str = 'Patch_AUC'

    VIZ_MULTI_TPR: str = 'Patient_TPR'
    VIZ_MULTI_TNR: str = 'Patient_TPR'
    VIZ_MULTI_AUC: str = 'Patient_AUC'

    def __init__(self,
                 patient_info: SheetCollection,
                 patient_pred: CascadedPred,
                 sub_class_list: Sequence,
                 class_partition: Sequence[Union[int, Sequence[int]]],
                 positive_class: Sequence[int],
                 val_phases: Dict[bool, str] = None,
                 is_visualize: bool = True
                 ):
        super().__init__(sub_class_list=sub_class_list,
                         class_partition=class_partition,
                         is_visualize=is_visualize,
                         val_phases=val_phases)
        self.patient_info = patient_info
        self.patient_pred = patient_pred

        self._positive_class = positive_class
        self._best_loss = np.inf
        self._best_stats = dict()
        # class_list=sub_class_list,
        # partition=self.class_partition)
        self.num_classes: int = self.class_partition.size
        self._membership: Dict[str, Union[torch.Tensor, np.ndarray]] = dict()
        self.add_meter(DefaultStats.PATCH_ACC, ClassErrorMeter(accuracy=True))
        self.add_meter(DefaultStats.PATCH_CONF, ConfusionMeter(self.num_classes, normalized=False))
        self.add_meter(DefaultStats.LOSS_METER, AverageValueMeter())
        self.add_meter(DefaultStats.MULTI_INSTANCE,
                       MultiInstanceMeter(self.patient_info,
                                          self.patient_pred,
                                          positive_class=MultiInstanceMeter.default_positive(self.class_partition,
                                                                                             self._positive_class)
                                          )
                       )
        self.add_meter(DefaultStats.PATCH_AUC, AUCMeter())

        self.add_logger(VisdomPlotLogger, 'line', DefaultStats.VIZ_LOSS)
        self.add_logger(VisdomPlotLogger, 'line', DefaultStats.VIZ_PATCH_TPR)
        self.add_logger(VisdomPlotLogger, 'line', DefaultStats.VIZ_PATCH_TNR)
        self.add_logger(VisdomPlotLogger, 'line', DefaultStats.VIZ_PATCH_AUC)

        self.add_logger(VisdomPlotLogger, 'line', DefaultStats.VIZ_MULTI_TPR)
        self.add_logger(VisdomPlotLogger, 'line', DefaultStats.VIZ_MULTI_TNR)
        self.add_logger(VisdomPlotLogger, 'line', DefaultStats.VIZ_MULTI_AUC)

    def update_membership(self, scores, index):
        score_dict = {ind: score_row.copy() for ind, score_row in zip(index, scores)}
        self.membership.update(score_dict)
        score_dict.clear()
        del score_dict

    @property
    def positive_class(self):
        return self._positive_class

    def on_start_epoch(self, state: Dict[str, Any], **kwargs):
        """
            Hook that is invoked on each single epoch. Typically reset the meters or setting the modes
            of dataset (train or test).
            Only available in the training mode.
        Args:
            state: See on_start

        Returns:

        """
        # reset all meters
        self.reset_meters()

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

    def label_collator(self, label_in, **kwargs):
        part_min = min([min(x) for x in self.class_partition])
        part_max = max([max(x) for x in self.class_partition])
        assert part_min <= label_in.max() <= part_max \
            and \
            part_min <= label_in.min() <= part_max,   \
            f"batch_label exceed range {(label_in.min(), label_in.max())}. Expected in {self.class_partition}"
        label = label_in
        label = label_binarize_group(label, anchor_group=self.positive_class, anchor_positive=True)
        return label

    def on_forward(self, state: Dict[str, Any], **kwargs):
        """
            On forward. Perform measurements for both train and test.
        Args:
            state:

        Returns:

        """
        # retrieve prediction and loss.

        # pred_data as the output score. Shape: N samples * M classes.
        pred_data: torch.Tensor = state['output'].data.detach().cpu()
        loss_scalar: torch.Tensor = state['loss'].data.detach().cpu()

        # retrieve input data
        # strong coupling - todo fix later
        img_new, label_in, mask, row, col, img, filename, *rest = state['sample']
        label: torch.LongTensor = self.label_collator(label_in).type("torch.LongTensor")

        # Add Loss Measurements per batch
        self.get_meter(DefaultStats.LOSS_METER).add(loss_scalar)
        # Add accuracy (patch) per batch
        # todo check length of label
        self.get_meter(DefaultStats.PATCH_ACC).add(pred_data, label)

        # If 1-d vector (in case of batch-of-one), adding leading singleton dim.
        # Otherwise the confusion_meter may mistreat the 1-d vector as a vector of
        # score of positive classes.
        if pred_data.ndimension == 1:
            pred_data = pred_data[None]

        self.get_meter(DefaultStats.PATCH_CONF).add(pred_data, label)

        # Posterior score
        pred_softmax = np.atleast_2d(softmax(pred_data).data.detach().cpu().numpy().squeeze())
        patch_name_score: Tuple = (filename, pred_softmax)
        """ 
            Note: the score per sample (single data point) is not reduced. Values of all categories are retained.
        """
        self.get_meter(DefaultStats.MULTI_INSTANCE).add(patch_name_score)

        # todo multiclass generalization: use label_binarize in sklearn, or simply
        # if self.num_classes == 2:
        score_numpy = np.atleast_1d(pred_softmax[:, 1])
        label_numpy = np.atleast_1d(label.cpu().squeeze().numpy())
        self.get_meter(DefaultStats.PATCH_AUC).add(score_numpy, label_numpy)

    def on_start(self, state: Dict[str, Any], *args, **kwargs):
        self.reset_meters()

    def _evaluation(self, state):
        """
            (1) Print all results: loss/auc/conf
            (2) Perform Patient-level evaluation
        Args:
            state: See on_start

        Returns:

        """
        # if train - assert the epoch count.
        assert not state['train'] or state['epoch'] == self.epoch_count
        # todo test
        phase = self.phase_name(state)
        if phase is None:
            #  skip
            return
        conf_mat, roc_auc_dict, raw_data = \
            self.get_meter(DefaultStats.MULTI_INSTANCE).value()

        loss = self.get_meter(DefaultStats.LOSS_METER).value()[0]
        patch_acc = self.get_meter(DefaultStats.PATCH_ACC).value()[0]
        patch_auc = self.get_meter(DefaultStats.PATCH_AUC).value()[0]
        patch_conf = self.get_meter(DefaultStats.PATCH_CONF).value()
        # todo move to engine
        if not state['train'] and loss < self._best_loss:
            self._best_loss = loss
            loss_mark = '*'
            self._best_stats.update({
                'loss': loss,
                'patch_acc': patch_acc,
                'patch_auc': patch_auc,
                'patch_conf': patch_conf,
                'roc_auc_dict': roc_auc_dict,
                'patient_conf': conf_mat,
                'raw_data': raw_data
            })
            engine: AbstractEngine = state['network']
            pickle_full = engine.dump_file_name('best_stat', 'pickle')
            with open(pickle_full, 'wb') as handle:
                pickle.dump(self._best_stats, handle)

            chk_full = engine.dump_file_name('best_model', 'pth')
            model = engine.model
            optimizer = engine.optimizer
            checkpoint = {
                'model_dict': model.state_dict(),
                'optim_dict': optimizer.state_dict()
            }
            torch.save(checkpoint, chk_full)
        else:
            loss_mark = ''

        epoch_profile = f"[Epoch {self.epoch_count}]."
        basic_verbose = f"{phase} -{epoch_profile}" \
            f"Loss:= {loss:.5f}{loss_mark} " \
            f"Patch accuracy:= {patch_acc:.2f}  " \
            f"Patch AUC:= {patch_auc:.2f}"
        patient_lvl_data = roc_auc_dict[1]
        patient_verbose = f"phase - {phase}. Patient AUC: {patient_lvl_data.get('auc', 'Not available')} "
        print(basic_verbose)
        print(patient_verbose)
        print(patch_conf)
        print(conf_mat)

        patch_conf_norm: np.ndarray = patch_conf.astype('float') / patch_conf.sum(axis=1)[:, np.newaxis]
        multi_conf_norm: np.ndarray = conf_mat.astype('float') / conf_mat.sum(axis=1)[:, np.newaxis]

        if self.num_classes <= 2:
            tn_patch, fp_patch, fn_patch, tp_patch = patch_conf_norm.ravel()
            tn_multi, fp_multi, fn_multi, tp_multi = multi_conf_norm.ravel()
            self.visualize_log(DefaultStats.VIZ_PATCH_TPR, state['train'], self.epoch_count, tp_patch)
            self.visualize_log(DefaultStats.VIZ_PATCH_TNR, state['train'], self.epoch_count, tn_patch)
            self.visualize_log(DefaultStats.VIZ_MULTI_TPR, state['train'], self.epoch_count, tp_multi)
            self.visualize_log(DefaultStats.VIZ_MULTI_TNR, state['train'], self.epoch_count, tn_multi)

        self.visualize_log(DefaultStats.VIZ_LOSS, state['train'], self.epoch_count, loss)

        self.visualize_log(DefaultStats.VIZ_PATCH_AUC, state['train'], self.epoch_count, patch_auc)
        self.visualize_log(DefaultStats.VIZ_MULTI_AUC, state['train'],
                           self.epoch_count,
                           patient_lvl_data.get('auc', 0))
        # moved to on_start
