import pickle
from abc import ABC, abstractmethod
from typing import Set, Dict, Any, Sequence, Union, Callable, Tuple

import numpy as np
import torch
import torch.nn as nn
from torchnet.logger.visdomlogger import BaseVisdomLogger
from torchnet.logger import VisdomPlotLogger  # ,VisdomLogger
from torchnet.meter import AverageValueMeter, ClassErrorMeter, ConfusionMeter, AUCMeter
from torchnet.meter.meter import Meter

from pycodelib.dlengine.skeletal import AbstractEngine
from pycodelib.metrics import label_binarize_group
from pycodelib.patients import CascadedPred, SheetCollection
from ..multi_instance_meter import MultiInstanceMeter
from ..skeletal import EngineHooks
import logging
debug_logger = logging.getLogger(__name__)
debug_logger.setLevel(logging.CRITICAL)

# noinspection SpellCheckingInspection
softmax: Callable = nn.Softmax(dim=1)


class BaseModelStats(EngineHooks, ABC):
    ...


class AbstractModelStats(BaseModelStats, ABC):
    # HOOKS_NAME: Set[str] = {"on_start"}
    PHASE_NAMES: Set[str] = {'train', 'val'}
    DEFAULT_PORT: int = 8097
    DEFAULT_ENV: str = 'main'

    def __init__(self,
                 sub_class_list: Sequence,
                 class_partition: Sequence[Union[int, Sequence[int]]],
                 val_phases: Dict[bool, str] = None,
                 is_visualize: bool = True,
                 port: int = DEFAULT_PORT,
                 env: str = DEFAULT_ENV
                 ):
        super().__init__()
        self.meter_dict: Dict[str, Any] = dict()
        self._membership = None  # todo - EM
        self.sub_class_list = sub_class_list
        self.class_partition: np.ndarray = np.asarray(class_partition)
        self._port = port
        self._is_visualize = is_visualize
        self._viz_logger_dict: Dict[Tuple[str, bool], Any] = dict()
        if val_phases is None:
            val_phases = dict({True: "train", False: "val"})
        self.val_phases: Dict[bool, str] = val_phases
        assert set(self.val_phases.values()).issubset(AbstractModelStats.PHASE_NAMES), \
            f"Invalid Phase Names{self.val_phases}" \
            f"Expected from {AbstractModelStats.PHASE_NAMES}"
        self.__env = env

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
    def env(self) -> str:
        return self.__env

    @property
    def epoch_count(self) -> int:
        return self.__epoch_count

    def epoch_incr(self):
        self.__epoch_count += 1

    def epoch_clear(self):
        self.__epoch_count = 0

    def on_end_epoch(self, state: Dict[str, Any], *args, **kwargs):
        # state['epoch'] is updated before on_end_epoch and after on_update,
        # as it starts from 0. Hence, evaluation after on_end_epoch has epoch values starting from 1.
        self.epoch_incr()
        self._evaluation(state)

    def on_end(self, state: Dict[str, Any], **kwargs):
        if not state['train']:
            self._evaluation(state)
        else:
            # only training alter the epoch
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
        return self.meter_dict.get(name, None)

    def phase_name(self, state):
        phase = self.val_phases.get(state['train'], None)
        return phase

    def add_logger(self, viz_logger_construct: type,
                   plot_type: str,
                   name: str,
                   phases: Sequence[bool] = (True, False),
                   control_by_flag: bool = False,
                   **kwargs):
        assert issubclass(viz_logger_construct, BaseVisdomLogger)
        do_create: bool = True if not control_by_flag else self.is_visualize
        if do_create:
            for phase in phases:
                title = "Train" if phase else "Test"
                title = f"{title} {name}"
                opts = {
                    'title': title
                }
                opts.update(kwargs)
                viz_logger = viz_logger_construct(plot_type,
                                                  port=self._port,
                                                  env=self.env,
                                                  opts=opts)
                key = (name, phase)
                self._viz_logger_dict[key] = viz_logger

    def get_logger(self, name: str, phase: bool, raise_key: bool = False):
        key = (name, phase)
        if raise_key:
            return self._viz_logger_dict[key]
        return self._viz_logger_dict.get(key)

    def visualize_log(self, name, phase, *values):
        if self.is_visualize:
            viz_logger = self.get_logger(name, phase)
            if viz_logger is None:
                debug_logger.warning(f"Logger Not Found: {name}, {phase}")
                return
            viz_logger.log(*values)

    def meter_add_value(self, name, *values):
        meter = self.get_meter(name)
        if meter is None:
            debug_logger.warning(f"Meter Not Found: {name}")
            return
        meter.add(*values)

    def meter_get_value(self, name):
        meter = self.get_meter(name)
        if meter is None:
            result = None
        else:
            result = meter.value()
        return result


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
                 sub_class_list: Sequence,
                 class_partition: Sequence[Union[int, Sequence[int]]],
                 positive_class: Sequence[int],
                 val_phases: Dict[bool, str] = None,
                 is_visualize: bool = True,
                 patient_info: SheetCollection = None,
                 patient_pred: CascadedPred = None,
                 port: int = AbstractModelStats.DEFAULT_PORT,
                 env: str = AbstractModelStats.DEFAULT_ENV
                 ):
        super().__init__(sub_class_list=sub_class_list,
                         class_partition=class_partition,
                         val_phases=val_phases,
                         is_visualize=is_visualize,
                         port=port,
                         env=env
                         )
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
        if self.patient_pred is not None and self.patient_info is not None:

            self.add_meter(DefaultStats.MULTI_INSTANCE,
                           MultiInstanceMeter(self.patient_info,
                                              self.patient_pred,
                                              positive_class=MultiInstanceMeter.default_positive(self.class_partition,
                                                                                                 self._positive_class)
                                              )

                           )
            self.add_logger(VisdomPlotLogger, 'line', DefaultStats.VIZ_MULTI_TPR)
            self.add_logger(VisdomPlotLogger, 'line', DefaultStats.VIZ_MULTI_TNR)
            self.add_logger(VisdomPlotLogger, 'line', DefaultStats.VIZ_MULTI_AUC)

        self.add_meter(DefaultStats.PATCH_AUC, AUCMeter())

        self.add_logger(VisdomPlotLogger, 'line', DefaultStats.VIZ_LOSS)
        self.add_logger(VisdomPlotLogger, 'line', DefaultStats.VIZ_PATCH_TPR)
        self.add_logger(VisdomPlotLogger, 'line', DefaultStats.VIZ_PATCH_TNR)
        self.add_logger(VisdomPlotLogger, 'line', DefaultStats.VIZ_PATCH_AUC)

    def update_membership(self, scores, index):
        # debug the memory leak
        # self.membership.update(score_dict)
        # index is also a tensor!
        for ind, score_row in zip(index, scores):
            # breakpoint()
            if isinstance(ind, np.ndarray) and ind.size == 1:
                ind = ind.ravel()[0]
            self.membership[ind] = score_row  # .max()

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
            part_min <= label_in.min() <= part_max, \
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
        self.meter_add_value(DefaultStats.LOSS_METER, loss_scalar)
        # Add accuracy (patch) per batch
        # todo check length of label
        self.meter_add_value(DefaultStats.PATCH_ACC, pred_data, label)

        # If 1-d vector (in case of batch-of-one), adding leading singleton dim.
        # Otherwise the confusion_meter may mistreat the 1-d vector as a vector of
        # score of positive classes.
        if pred_data.ndimension == 1:
            pred_data = pred_data[None]

        self.meter_add_value(DefaultStats.PATCH_CONF, pred_data, label)

        # Posterior score
        pred_softmax = np.atleast_2d(softmax(pred_data).data.detach().cpu().numpy().squeeze())
        patch_name_score: Tuple = (filename, pred_softmax)
        """ 
            Note: the score per sample (single data point) is not reduced. Values of all categories are retained.
        """
        self.meter_add_value(DefaultStats.MULTI_INSTANCE, patch_name_score)
        # multi_instance_meter = self.get_meter(DefaultStats.MULTI_INSTANCE)
        # if multi_instance_meter is not None:
        #    multi_instance_meter.add(patch_name_score)

        # todo multiclass generalization: use label_binarize in sklearn, or simply
        # if self.num_classes == 2:
        score_numpy = np.atleast_1d(pred_softmax[:, 1])
        label_numpy = np.atleast_1d(label.cpu().squeeze().numpy())
        self.meter_add_value(DefaultStats.PATCH_AUC, score_numpy, label_numpy)

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

        loss = self.get_meter(DefaultStats.LOSS_METER).value()[0]
        patch_acc = self.get_meter(DefaultStats.PATCH_ACC).value()[0]
        patch_auc = self.get_meter(DefaultStats.PATCH_AUC).value()[0]
        patch_conf = self.get_meter(DefaultStats.PATCH_CONF).value()
        patch_conf_norm: np.ndarray = patch_conf.astype('float') / patch_conf.sum(axis=1)[:, np.newaxis]
        # todo move to engine

        # conf_mat, roc_auc_dict, raw_data = \
        patient_meter_result = self.meter_get_value(DefaultStats.MULTI_INSTANCE)

        if not state['train'] and loss < self._best_loss:
            self._best_loss = loss
            loss_mark = '*'
            self._best_stats.update({
                'loss': loss,
                'patch_acc': patch_acc,
                'patch_auc': patch_auc,
                'patch_conf': patch_conf,
                'patient_meter_result': patient_meter_result,
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
        print(basic_verbose)
        print(patch_conf)

        tn_patch, fp_patch, fn_patch, tp_patch = patch_conf_norm.ravel()
        if self.get_meter(DefaultStats.MULTI_INSTANCE) is not None:
            conf_mat, roc_auc_dict, raw_data = patient_meter_result
            patient_lvl_data = roc_auc_dict[1]
            patient_verbose = f"phase - {phase}. Patient AUC: {patient_lvl_data.get('auc', 'Not available')} "
            print(patient_verbose)
            print(patch_conf)
            print(conf_mat)
            self.visualize_log(DefaultStats.VIZ_MULTI_AUC, state['train'],
                               self.epoch_count,
                               patient_lvl_data.get('auc', 0))
            multi_conf_norm: np.ndarray = conf_mat.astype('float') / conf_mat.sum(axis=1)[:, np.newaxis]
            tn_multi, fp_multi, fn_multi, tp_multi = multi_conf_norm.ravel()
        else:
            tn_multi, fp_multi, fn_multi, tp_multi = (None, None, None, None)

        if self.num_classes <= 2:
            self.visualize_log(DefaultStats.VIZ_PATCH_TPR, state['train'], self.epoch_count, tp_patch)
            self.visualize_log(DefaultStats.VIZ_PATCH_TNR, state['train'], self.epoch_count, tn_patch)
            self.visualize_log(DefaultStats.VIZ_MULTI_TPR, state['train'], self.epoch_count, tp_multi)
            self.visualize_log(DefaultStats.VIZ_MULTI_TNR, state['train'], self.epoch_count, tn_multi)

        self.visualize_log(DefaultStats.VIZ_LOSS, state['train'], self.epoch_count, loss)

        self.visualize_log(DefaultStats.VIZ_PATCH_AUC, state['train'], self.epoch_count, patch_auc)

        # moved to on_start
