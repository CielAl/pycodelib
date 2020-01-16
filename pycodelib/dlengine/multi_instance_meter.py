from typing import Dict, Any, Tuple, Hashable, List, Sequence, Union
import torch
import numpy as np
import numbers
from torchnet.meter.meter import Meter
from pycodelib.patients import SheetCollection, CascadedPred
from pycodelib.common import default_not_none
from sklearn.metrics import confusion_matrix
from pycodelib.metrics import multi_class_roc_auc_vs_all
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class MultiInstanceMeter(Meter):
    """
        Bag of scores by groups.
    """
    @property
    def instance_map(self) -> Dict[str, Any]:
        """
        Returns:
            instance_map
        """
        return self._instance_map

    @property
    def patient_info(self) -> SheetCollection:
        """
        Returns:
            patient_col
        """
        return self._patient_info

    @property
    def patient_pred(self) -> CascadedPred:
        return self._patient_pred

    def default_label_collate(self, label_occurrence_in: np.ndarray):
        """
            Ensure the mutual exclusion
        Returns:

        """
        # label_occurrence = np.atleast_2d(label_occurrence_in)
        # default [No Path, BCC, Situ, Invasive]. BCC+Situ = BCC. Invasive+Situ = Invasive
        raise NotImplementedError(self)

    def __init__(self,
                 patient_info: SheetCollection,
                 patient_pred: CascadedPred,
                 positive_class: Sequence[int],  # todo
                 # label_collate=None,
                 binarize: bool = True):
        """
            Constructor.
        Args:
            patient_info: The SheetCollection of Patient Table.
        """
        super().__init__()
        self._instance_map: Dict[str, Any] = dict()
        self._patient_info: SheetCollection = patient_info
        self._patient_pred: CascadedPred = patient_pred
        self._positive_class = positive_class
        # self.label_collate = default_not_none(label_collate, self.default_label_collate)
        self.binarize = binarize
        self._num_classes = len(self._patient_pred.class_list)

    @property
    def num_classes(self):
        return self._num_classes

    @classmethod
    def build(cls,
              patient_info: SheetCollection,
              class_list: Sequence,
              partition: Sequence[Sequence[int]],
              positive_class: Sequence[int] = None):
        patient_pred = CascadedPred(patient_info, class_list=class_list, partition=partition)
        positive_class = cls.default_positive(partition, positive_class=positive_class)
        assert positive_class.size <= len(partition), f"More positive classes than total # of classes" \
            f"{positive_class} vs. Partition:{partition}"
        return cls(patient_info=patient_info, patient_pred=patient_pred, positive_class=positive_class)

    @staticmethod
    def default_positive(partition, positive_class=None):
        if len(partition) == 2:
            positive_class: np.ndarray = np.asarray(default_not_none(positive_class, [1]))
        else:
            positive_class: np.ndarray = np.asarray(default_not_none(positive_class, np.arange(len(partition))))
        return positive_class

    @staticmethod
    def vectorized_obj(obj_in: Any, scalar_type: type, at_least_2d=False, err_msg: str = "") -> np.ndarray:
        """
            Vectorize the input
        Args:
            obj_in: Original input
            scalar_type:    Type if scalar
            at_least_2d:    Whether or not guarantee the dimension
            err_msg:    Message to print if cannot be vectorized

        Returns:
            obj: vectorized form.
        """
        is_matched_scalar: bool = isinstance(obj_in, scalar_type)
        already_vector: bool = isinstance(obj_in, Sequence) or isinstance(obj_in, np.ndarray)

        if torch.is_tensor(obj_in):
            obj: np.ndarray = np.atleast_1d(obj_in.cpu().squeeze().numpy())
        # pack to np array if a scalar of matched type or is already in vector form.
        elif is_matched_scalar or already_vector:
            obj: np.ndarray = np.atleast_1d(np.asarray(obj_in).squeeze())
        else:
            raise TypeError(err_msg)
        # obj = obj.squeeze() squeeze first than np.atleast_1d, as array of one will be degraded to scalar
        if at_least_2d:
            # obj: np.ndarray = np.atleast_1d(obj)
            obj = np.atleast_2d(obj)
        return obj

    def add(self, elements: Tuple[Union[torch.Tensor, Sequence[Hashable]], Union[torch.Tensor, Sequence]]):
        """
            Vectorized implementation to add name, score pair. Input will always be converted into vector forms
            implicitly for this purpose.
        Args:
            elements:

        Returns:

        """
        # unpack
        keys_in, values_in = elements

        # Vectorization:
        # breakpoint()
        # keys must be 1d
        keys = type(self).vectorized_obj(keys_in, Hashable, at_least_2d=False, err_msg=f"{type(keys_in)}")
        values = type(self).vectorized_obj(values_in, numbers.Number, at_least_2d=True, err_msg=f"{type(values_in)}")
        values = values.astype(np.float64)
        # Now keys and values are numpy arrays.
        # Validate if the length agree.

        assert keys.shape[0] == values.shape[0], f"Length not agree. key={keys}|{len(keys)}." \
            f" values={values}|{values.shape[0]}"
        # insert keys and values into a dict: self.instance_map.
        for k, v in zip(keys, values):
            default_store: List = self.instance_map.get(k, [])
            self.instance_map.update({k: default_store})
            default_store.append(v)

    def add_kv(self, keys: Union[torch.Tensor, Sequence[Hashable]], values: Union[torch.Tensor, Sequence]):
        """
            Only a wrapper that pack the input into Tuple.
        Args:
            keys:
            values:

        Returns:

        """
        self.add((keys, values))

    @staticmethod
    def true_label(label):
        ...

    @staticmethod
    def curate_patient_label_helper(true_labels_occurrence, row, high_priority_col: int = 1):
        # use indexing like [][] will create a copy instead of a view
        true_labels_occurrence[row, 0:high_priority_col] = 0
        true_labels_occurrence[row, high_priority_col+1:] = 0
        return true_labels_occurrence

    @staticmethod
    def curate_patient_label(true_labels_occurrence, high_priority_col):
        # if there are multiple labels per patient
        row_multi_label = np.where((true_labels_occurrence > 0).all(axis=1))
        true_labels_occurrence = MultiInstanceMeter.curate_patient_label_helper(true_labels_occurrence,
                                                                                row_multi_label,
                                                                                high_priority_col=high_priority_col)
        return true_labels_occurrence

    def value_helper(self):
        """
            todo Move the fetch_prediction here.
        Returns:
            todo: score and pred
            true_labels are already curated and mapped (from group partition)
        """
        # for each v: List of array<prob_c1, prob_c2, ..., prob_cn>
        # values: List of v
        filenames = list(self.instance_map.keys())
        scores_all_category_roi_collection = list(self.instance_map.values())
        # reduce from patch to ROI by averaging. Note: for patient level there shall be
        # one more level of averaging
        #  np.histogram(np.asarray(scores_all_category_roi_collection[3]), range=(0,1), density=True)
        scores_all_category: List = [np.asarray(scores_per_roi).mean(axis=0)
                                     for scores_per_roi in scores_all_category_roi_collection]
        # class name

        self.patient_pred.load_score(scores_all_category, filenames, flush=True, expand=True)
        score_table = self.patient_pred.get_df(CascadedPred.NAME_SCORE)
        # noinspection PyUnresolvedReferences
        # breakpoint()
        scores_all_class: np.ndarray = score_table.values.copy()
        # add None (extra dim) in dims for broadcasting (divide a column),
        # otherwise the [:, -1] returns a row vector and performs division on rows.
        scores_all_class[:, :-1] /= scores_all_class[:, -1, None]
        scores_all_class = scores_all_class[:, :-1]

        true_label_table = self.patient_pred.get_ground_truth(score_table.index)
        true_labels_occurrence = np.atleast_2d(true_label_table.values)
        true_labels_occurrence = MultiInstanceMeter.curate_patient_label(true_labels_occurrence, high_priority_col=1)
        assert true_labels_occurrence.ndim == 2, f"Label occurrence is not 2d: {true_labels_occurrence.ndim}"
        # bug
        # breakpoint()
        true_labels = np.argwhere(true_labels_occurrence > 0)[:, -1]
        # breakpoint()
        rows_index = np.asarray(score_table.index)
        columns_keys = np.asarray(score_table.keys())[:-1]
        patch_counts = score_table.values[:, -1].copy()
        return scores_all_class, true_labels, patch_counts, rows_index, columns_keys

    def value(self):
        scores_all_class, true_labels, patch_counts, rows_index, columns_keys = self.value_helper()
        raw_data = {
            "rows_index": rows_index,
            "columns_keys": columns_keys,
            "patch_counts": patch_counts,
            "scores_all_class": scores_all_class,
            "true_labels": true_labels,
        }
        conf_pred_label = scores_all_class.argmax(axis=1)

        conf_mat = confusion_matrix(true_labels, conf_pred_label, range(self.num_classes))
        conf_mat_norm = conf_mat.astype('float') / conf_mat.sum(axis=1)[:, np.newaxis]
        conf_mat_norm = np.nan_to_num(conf_mat_norm)
        assert len(np.unique(true_labels)) > 1
        if not self.binarize:
            roc_auc_dict = multi_class_roc_auc_vs_all(true_labels, scores_all_class, self._positive_class)
        else:
            roc_auc_dict = multi_class_roc_auc_vs_all(true_labels, scores_all_class, [1])
        # breakpoint()
        return {
            'conf_mat': conf_mat,
            'conf_mat_norm': conf_mat_norm,
            'roc_auc_dict': roc_auc_dict,
            'raw_data': raw_data,
        }

    def reset(self):
        """
            Clear the map.
            todo - clear the patient collection
        Returns:
            None
        """
        self._instance_map.clear()
