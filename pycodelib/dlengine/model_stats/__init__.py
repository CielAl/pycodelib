import pickle
from abc import ABC, abstractmethod
from typing import Set, Dict, Any, Sequence, Union, Callable, Tuple
from copy import deepcopy
import numpy as np
import torch
import torch.nn as nn
# noinspection PyPackageRequirements
from torchnet.logger.visdomlogger import BaseVisdomLogger
# noinspection PyPackageRequirements
from torchnet.logger import VisdomPlotLogger  # ,VisdomLogger
# noinspection PyPackageRequirements
from torchnet.meter import AverageValueMeter, ClassErrorMeter, ConfusionMeter, AUCMeter
# noinspection PyPackageRequirements
from torchnet.meter.meter import Meter

from pycodelib.common import require_not_none
from pycodelib.dlengine.skeletal import AbstractEngine
from pycodelib.metrics import label_binarize_group, label_encode_by_partition
from pycodelib.patients import CascadedPred, SheetCollection
from ..multi_instance_meter import MultiInstanceMeter
from ..multi_class_auc import MultiAUC
from ..skeletal import EngineHooks
import logging
debug_logger = logging.getLogger(__name__)
debug_logger.setLevel(logging.CRITICAL)

__all__ = ['AbstractModelStats', 'DefaultStats']


class _DictContainer(ABC):

    def _do_if_not_absent(self, key, raise_flag, action, *values, **kwargs):
        # use raise_flag=False --> returns None if key is absent.
        # combine with the validation of whether value is absent
        target = self.get(key, raise_flag=False)

        require_not_none(obj=target, name=key, raise_error=raise_flag)
        if target is None:
            return

        assert isinstance(action, str) or isinstance(action, Callable)
        if isinstance(action, str):
            action = getattr(target, action)
        return action(*values, **kwargs)

    @property
    def container(self) -> Dict:
        return self._container

    def __init__(self):
        self._container: Dict = dict()

    def add(self, key, value, value_type: type = None, raise_flag: bool = False):
        if not hasattr(self, '_container') or self._container is None:
            self._container = dict()
        type_val_result = value_type is None or isinstance(value, value_type)
        if raise_flag:
            assert type_val_result
        self._container[key] = value

    def get(self, key, raise_flag: bool = False, default=None):
        if raise_flag:
            return self._container[key]
        return self._container.get(key, default)

    def act(self, key, action: Union[str, Callable], raise_flag, *values, **kwargs):
        return self._do_if_not_absent(key, raise_flag, action, *values, **kwargs)


class MeterContainer(object):

    @property
    def meter_dict(self):
        return self._meter_dict

    def __init__(self):
        self._meter_dict = _DictContainer()

    def get_meter(self, name):
        return self._meter_dict.get(key=name)

    def reset_meters(self):
        for k, v in self.meter_dict.container.items():
            v.reset()

    def add_meter(self, name, meter, meter_type=Meter):
        self.meter_dict.add(name, meter, value_type=meter_type, raise_flag=False)

    def meter_add_value(self, name, *values):
        self.meter_dict.act(name, 'add', False, *values)

    def meter_get_value(self, name, raise_flag: bool = False):
        return self.meter_dict.act(name, 'value', raise_flag=raise_flag)


class VizLogContainer(object):

    def __init__(self,
                 is_visualize: bool,
                 port: int,
                 env: str):

        self._viz_logger_dict = _DictContainer()
        self.is_visualize = is_visualize
        self._port = port
        self.env = env

    @staticmethod
    def mode_to_phase_name(mode: bool):
        return "Train" if mode else "Test"

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
                if phase == DefaultStats.PHASE_COMBINE:
                    title = ''
                else:
                    title = VizLogContainer.mode_to_phase_name(phase)
                title = f"{title} {name}"
                opts_input = kwargs.get('opts', {})
                opts = {
                    'title': title
                }
                opts = {**opts, **opts_input}
                opts.update(kwargs)
                viz_logger = viz_logger_construct(plot_type,
                                                  port=self._port,
                                                  env=self.env,
                                                  opts=opts)
                key = (name, phase)
                self._viz_logger_dict.add(key, viz_logger)

    def get_logger(self, name: str, phase: bool, raise_flag: bool = False):
        key = (name, phase)
        return self._viz_logger_dict.get(key, raise_flag=raise_flag)

    def visualize_log(self, meter_name, phase, *values, **kwargs):
        """
        Note: do not pass opts into kwargs - already defined in the VisdomPlogLogger.
        Args:
            meter_name ():
            phase ():
            *values ():
            **kwargs ():

        Returns:

        """
        # values contains None:
        if not all(values):
            return
        if self.is_visualize:
            key = (meter_name, phase)
            self._viz_logger_dict.act(key, 'log', False, *values, **kwargs)


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
                 env: str = DEFAULT_ENV,
                 previous_preds=None,
                 ):
        super().__init__()
        self._membership = None  # todo - EM
        self._previous_preds = previous_preds
        self.sub_class_list = sub_class_list
        # object array (data type=list)
        self.class_partition: Sequence[Sequence[int]] = deepcopy(class_partition)
        assert len(self.sub_class_list) == len(self.class_partition), f"# of Class disagree with class partition" \
            f"{self.sub_class_list} vs. {self.class_partition}"
        self._is_visualize = is_visualize
        # self._viz_logger_dict: Dict[Tuple[str, bool], Any] = dict()
        self.meter_container = MeterContainer()
        self.vizlog_container = VizLogContainer(is_visualize=is_visualize, port=port, env=env)
        if val_phases is None:
            val_phases = dict({True: "train", False: "val"})
        self.val_phases: Dict[bool, str] = val_phases
        assert set(self.val_phases.values()).issubset(AbstractModelStats.PHASE_NAMES), \
            f"Invalid Phase Names{self.val_phases}" \
            f"Expected from {AbstractModelStats.PHASE_NAMES}"

        # alternatively, to plot test data of a epoch, one may perform engine.test first,
        # and use the state['epoch'] in the training phase + test result that are not yet reset in the meter
        # however, it makes the code even less readable, as there is even stronger semantic coupling
        # to the control flow of torchnet.engine
        self.__epoch_count: int = 0

        self._num_classes: int = len(self.class_partition)

    @property
    def num_classes(self):
        return self._num_classes

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
    def membership(self):
        return self._membership

    def update_membership(self, scores, index):
        raise NotImplemented

    @abstractmethod
    def label_collator(self, labels, *args, **kwargs):
        ...

    def validation_phase_name(self, state):
        phase = self.val_phases.get(state['train'], None)
        return phase


class DefaultStats(AbstractModelStats):
    PHASE_COMBINE = None

    PATCH_ACC: str = 'patch_accuracy_meter'
    MULTI_INSTANCE: str = 'multi_instance_meter'
    PATCH_CONF: str = 'patch_conf_meter'
    LOSS_METER: str = 'loss_meter'
    PATCH_AUC: str = 'patch_auc_meter'

    VIZ_LOSS: str = 'Loss'
    VIZ_PATCH_TRUE_PRED: str = 'Patch_TRUE_PRED'
    VIZ_PATCH_AUC: str = 'Patch_AUC'

    VIZ_MULTI_TRUE_PRED: str = 'Patient_TRUE_PRED'
    VIZ_MULTI_AUC: str = 'Patient_AUC'

    @staticmethod
    def class_per_phase(x, y):
        return f"{x}_{y}"

    def __init__(self,
                 sub_class_list: Sequence,
                 class_partition: Sequence[Union[int, Sequence[int]]],
                 positive_class: Sequence[int],
                 val_phases: Dict[bool, str] = None,
                 is_visualize: bool = True,
                 patient_info: SheetCollection = None,
                 patient_pred: CascadedPred = None,
                 port: int = AbstractModelStats.DEFAULT_PORT,
                 env: str = AbstractModelStats.DEFAULT_ENV,
                 previous_preds=None
                 ):
        super().__init__(sub_class_list=sub_class_list,
                         class_partition=class_partition,
                         val_phases=val_phases,
                         is_visualize=is_visualize,
                         port=port,
                         env=env,
                         previous_preds=previous_preds,
                         )
        self.patient_info = patient_info
        self.patient_pred = patient_pred

        self._positive_class = positive_class
        self._best_loss = np.inf
        self._best_stats = dict()
        # class_list=sub_class_list,
        # partition=self.class_partition)

        self._membership: Dict[str, Union[torch.Tensor, np.ndarray]] = dict()
        self.meter_container.add_meter(DefaultStats.PATCH_ACC, ClassErrorMeter(accuracy=True))
        self.meter_container.add_meter(DefaultStats.PATCH_CONF, ConfusionMeter(self.num_classes, normalized=False))
        self.meter_container.add_meter(DefaultStats.LOSS_METER, AverageValueMeter())
        self.meter_container.add_meter(DefaultStats.PATCH_AUC, MultiAUC(num_class=self.num_classes))

        phase_name_list = [VizLogContainer.mode_to_phase_name(phase) for phase in [True, False]]
        plot_opts_by_phase = {'legend': phase_name_list}

        class_name_cross_phase = [DefaultStats.class_per_phase(x, y)
                                  for x in self.sub_class_list for y in phase_name_list]
        plot_opts_by_class_name = {'legend': class_name_cross_phase}
        if self.patient_pred is not None and self.patient_info is not None:
            self.meter_container.add_meter(DefaultStats.MULTI_INSTANCE,
                                           MultiInstanceMeter(self.patient_info,
                                                              self.patient_pred,
                                                              positive_class=MultiInstanceMeter.default_positive(
                                                                  self.class_partition,
                                                                  self._positive_class
                                                              )))
            self.vizlog_container.add_logger(VisdomPlotLogger, 'line', DefaultStats.VIZ_MULTI_TRUE_PRED,
                                             [DefaultStats.PHASE_COMBINE],
                                             opts=plot_opts_by_class_name)
            self.vizlog_container.add_logger(VisdomPlotLogger, 'line', DefaultStats.VIZ_MULTI_AUC,
                                             [DefaultStats.PHASE_COMBINE], opts=plot_opts_by_phase)

        self.vizlog_container.add_logger(VisdomPlotLogger, 'line', DefaultStats.VIZ_LOSS,
                                         [DefaultStats.PHASE_COMBINE], opts=plot_opts_by_phase)
        self.vizlog_container.add_logger(VisdomPlotLogger, 'line', DefaultStats.VIZ_PATCH_TRUE_PRED,
                                         [DefaultStats.PHASE_COMBINE],
                                         opts=plot_opts_by_class_name)
        self.vizlog_container.add_logger(VisdomPlotLogger, 'line', DefaultStats.VIZ_PATCH_AUC,
                                         [DefaultStats.PHASE_COMBINE], opts=plot_opts_by_class_name)

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
        self.meter_container.reset_meters()

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

    def label_collator_bin(self, label_in):
        """
        This only works when one of the two partition contains only one label, e.g.
        [[5,2,3], [1]]
        Args:
            label_in ():

        Returns:

        """
        part_min = min([min(x) for x in self.class_partition])
        part_max = max([max(x) for x in self.class_partition])
        assert part_min <= label_in.max() <= part_max \
            and \
            part_min <= label_in.min() <= part_max, \
            f"batch_label exceed range {(label_in.min(), label_in.max())}. Expected in {self.class_partition}"
        label = label_in
        label = label_binarize_group(label, anchor_group=self.positive_class, anchor_positive=True)
        return label

    def label_collator(self, label_in, **kwargs):
        """

        Args:
            label_in ():
            **kwargs ():

        Returns:
            labels (torch.Tensor):
        """
        # more generalized
        labels = label_encode_by_partition(label_in, self.class_partition)
        return torch.from_numpy(labels)

    def on_forward(self, state: Dict[str, Any], **kwargs):
        """
            On forward. Perform measurements for both train and test.
        Args:
            state:

        Returns:

        """
        softmax: Callable = nn.Softmax(dim=1)
        # retrieve prediction and loss.

        # pred_data as the output score. Shape: N samples * M classes.
        pred_data: torch.Tensor = state['output'].data.detach().cpu()
        loss_scalar: torch.Tensor = state['loss'].data.detach().cpu()

        # retrieve input data
        # strong coupling - todo fix later:
        # img_new, label_in, mask, row, col, img, filename, *rest = state['sample']
        engine = DefaultStats._engine_from_state(state)

        label_in = state['sample'][engine.label_key]
        filename = state['sample'][engine.filename_key]

        label: torch.LongTensor = self.label_collator(label_in).type("torch.LongTensor")

        # Add Loss Measurements per batch

        self.meter_container.meter_add_value(DefaultStats.LOSS_METER, loss_scalar)
        # Add accuracy (patch) per batch
        # todo check length of label
        self.meter_container.meter_add_value(DefaultStats.PATCH_ACC, pred_data, label)

        # If 1-d vector (in case of batch-of-one), adding leading singleton dim.
        # Otherwise the confusion_meter may mistreat the 1-d vector as a vector of
        # score of positive classes.
        if pred_data.ndimension == 1:
            pred_data = pred_data[None]
        self.meter_container.meter_add_value(DefaultStats.PATCH_CONF, pred_data, label)

        # Posterior score
        pred_softmax = np.atleast_2d(softmax(pred_data).data.detach().cpu().numpy().squeeze())
        patch_name_score: Tuple = (filename, pred_softmax)
        """ 
            Note: the score per sample (single data point) is not reduced. Values of all categories are retained.
            Ignored if there is no multi-instance meter.
        """
        self.meter_container.meter_add_value(DefaultStats.MULTI_INSTANCE, patch_name_score)
        # multi_instance_meter = self.get_meter(DefaultStats.MULTI_INSTANCE)
        # if multi_instance_meter is not None:
        #    multi_instance_meter.add(patch_name_score)

        # todo multiclass generalization: use label_binarize in sklearn
        # todo however, I would assume the one vs. rest AUC in multi-class case can be highly misleading
        # todo since the one vs. rest AUC can be high by sacrificing performance on other classes.
        # todo in binary case, this poses as an abnormal threshold, while in multi-class case, it cannot be
        # todo avoided by using any single thresholds.
        if self.num_classes == 2:
            score_numpy = np.atleast_1d(pred_softmax[:, 1])
            label_numpy = np.atleast_1d(label.cpu().squeeze().numpy())
            self.meter_container.meter_add_value(DefaultStats.PATCH_AUC, score_numpy, label_numpy)
        else:
            self.meter_container.meter_add_value(DefaultStats.PATCH_AUC, pred_softmax, label)

    def on_start(self, state: Dict[str, Any], *args, **kwargs):
        self.meter_container.reset_meters()

    @staticmethod
    def _engine_from_state(state) -> AbstractEngine:
        return state['network']

    def __best_state_helper(self, state, patch_result: Dict, patient_result: Dict) -> bool:
        assert patch_result is not None or patient_result is not None, f"All results are none"
        loss = patch_result['loss']
        is_best_val = not state['train'] and loss < self._best_loss

        if not is_best_val:
            return False

        self._best_loss = loss
        self._best_stats.update({
            'patch': patch_result,
            'patient': patient_result,
            'score': self.membership
        })
        engine: AbstractEngine = DefaultStats._engine_from_state(state)  # state['network']
        pickle_full = engine.dump_file_name('best_stat', 'pickle')
        with open(pickle_full, 'wb') as handle:
            pickle.dump(self._best_stats, handle)
        chk_full = engine.dump_file_name('best_model', 'pth')
        model = engine.model
        optimizer = engine.optimizer
        save_flag = model is not None and optimizer is not None
        if save_flag:
            checkpoint = {
                'model_dict': model.state_dict(),
                'optim_dict': optimizer.state_dict()
            }
            torch.save(checkpoint, chk_full)

        return True

    def __patch_level_helper(self):
        loss = self.meter_container.meter_get_value(DefaultStats.LOSS_METER, raise_flag=True)[0]
        patch_acc = self.meter_container.meter_get_value(DefaultStats.PATCH_ACC, raise_flag=True)[0]
        patch_auc = self.meter_container.meter_get_value(DefaultStats.PATCH_AUC, raise_flag=True)[0]
        patch_conf = self.meter_container.meter_get_value(DefaultStats.PATCH_CONF, raise_flag=True)
        patch_conf_norm: np.ndarray = patch_conf.astype('float') / patch_conf.sum(axis=1)[:, np.newaxis]
        result_dict: Dict[str, Any] = {
            'loss': loss,
            'patch_acc': patch_acc,
            'patch_auc': patch_auc,
            'patch_conf': patch_conf,
            'patch_conf_norm': patch_conf_norm,
        }
        return result_dict

    @staticmethod
    def _invalid_value_to_nan(val):
        if val is None:
            return np.nan
        return val

    def __basic_verbose_helper_patch(self, phase,
                                     patch_result: Dict,
                                     best_flag: bool,
                                     verbose_flag: bool = True):
        marker = '**' if best_flag else ''
        epoch_profile = f"[Epoch {self.epoch_count}]."

        loss = DefaultStats._invalid_value_to_nan(patch_result['loss'])
        patch_acc = DefaultStats._invalid_value_to_nan(patch_result['patch_acc'])
        patch_auc = DefaultStats._invalid_value_to_nan(patch_result['patch_auc'])
        patch_conf = DefaultStats._invalid_value_to_nan(patch_result['patch_conf'])
        patch_conf_norm = DefaultStats._invalid_value_to_nan(patch_result['patch_conf_norm'])
        patch_auc_print = np.array_str(np.asarray(patch_auc), precision=2, suppress_small=True)
        basic_verbose = f"{phase} -{epoch_profile}" \
            f"Loss:= {loss:.5f}{marker} " \
            f"Patch accuracy:= {patch_acc:.2f}  " \
            f"Patch AUC:= {patch_auc_print}"

        if verbose_flag:
            print(basic_verbose)
            print(patch_conf)
            print(patch_conf_norm)
        return basic_verbose

    def __basic_verbose_helper_patient(self, phase,
                                       patient_result: Dict, verbose_flag: bool = True):
        has_patient = self.meter_container.get_meter(DefaultStats.MULTI_INSTANCE) is not None \
                      and patient_result is not None
        if not has_patient:
            return
        conf_mat = patient_result['conf_mat']
        conf_mat_norm = patient_result['conf_mat_norm']
        roc_auc_dict = patient_result['roc_auc_dict']
        patient_lvl_auc = roc_auc_dict[1]
        patient_verbose = f"phase - {phase}. Patient AUC: {patient_lvl_auc.get('auc', 'Not available')} "

        if verbose_flag:
            print(patient_verbose)
            print(conf_mat)
            print(conf_mat_norm)

    def __viz_accuracy(self, logger_name, current_phase_name, conf_norm: np.ndarray):
        # tn_patch, fp_patch, fn_patch, tp_patch = patch_conf_norm.ravel()
        # current_phase_name is to tag what exactly phase is of the value that is passed in.
        # Since all phases are plot in the same logger, the "phase" in visualize_log is still "PHASE_COMBINE"
        if conf_norm is None:
            return
        if np.isnan(conf_norm).any():
            debug_logger.warning(f"has nan")
        class_accuracy_all = conf_norm.diagonal()
        for idx, class_acc in enumerate(class_accuracy_all):
            # todo class name
            class_name = self.sub_class_list[idx]
            # opts = {'legend': plot_name}
            plot_name = DefaultStats.class_per_phase(class_name, current_phase_name)
            self.vizlog_container.\
                visualize_log(logger_name,
                              DefaultStats.PHASE_COMBINE,
                              self.epoch_count, class_acc,
                              name=plot_name,
                              )

    def __base_log_plot(self):
        ...

    def _evaluation(self, state):
        """
            (1) Print all results: loss/auc/conf
            (2) Perform Patient-level evaluation
        Args:
            state: See on_start

        Returns:

        """
        # if train - assert the epoch count.
        # either test, or epoch num agree
        assert not state['train'] or state['epoch'] == self.epoch_count

        phase = self.validation_phase_name(state)
        if phase is None:
            #  skip
            return

        patch_result: Dict = self.__patch_level_helper()
        patient_result: Dict = self.meter_container.meter_get_value(DefaultStats.MULTI_INSTANCE)

        is_best_loss = self.__best_state_helper(state, patch_result=patch_result,
                                                patient_result=patient_result)
        # loss/acc/auc/conf
        self.__basic_verbose_helper_patch(phase, patch_result, is_best_loss)
        self.__basic_verbose_helper_patient(phase, patient_result=patient_result)

        phase_name = VizLogContainer.mode_to_phase_name(state['train'])
        self.__viz_accuracy(DefaultStats.VIZ_PATCH_TRUE_PRED, phase_name,
                            patch_result.get('patch_conf_norm'))
        loss = patch_result['loss']
        # merge train/val into the same plot
        self.vizlog_container.visualize_log(DefaultStats.VIZ_LOSS, DefaultStats.PHASE_COMBINE, self.epoch_count, loss,
                                            name=phase_name)

        patch_auc = patch_result['patch_auc']
        # merge
        # self.vizlog_container.visualize_log(DefaultStats.VIZ_PATCH_AUC,
        #                                    DefaultStats.PHASE_COMBINE, self.epoch_count, patch_auc,
        #                                    name=phase_name
        #                                    )
        self.__patient_viz_helper(patient_result, phase_name)
        # moved to on_start

    def __patient_viz_helper(self, patient_result, phase_name):
        if patient_result is None:
            return
        self.__viz_accuracy(DefaultStats.VIZ_MULTI_TRUE_PRED, phase_name,
                            patient_result.get('conf_mat_norm'))

        # patient_auc = -2 if patient_result is None else patient_result.get('roc_auc_dict', -1)
        roc_auc_dict = patient_result.get('roc_auc_dict')
        if roc_auc_dict is not None:
            # for now only print the first class
            result_list = list(patient_result['roc_auc_dict'].values())
            if len(result_list) == 0:
                patient_auc = -1
            else:
                patient_auc = result_list[0].get('auc', -2)
        else:
            patient_auc = -3

        self.vizlog_container.visualize_log(DefaultStats.VIZ_MULTI_AUC, DefaultStats.PHASE_COMBINE,
                                            self.epoch_count,
                                            patient_auc,
                                            name=phase_name)
