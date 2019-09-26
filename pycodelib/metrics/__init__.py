from functools import reduce
from operator import eq, ne, or_, and_
from typing import Dict, Union, Sequence

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics.ranking import label_binarize


# todo wrap the class-mapping from multi_class_roc_auc


def label_binarize_group(labels: Union[np.ndarray, torch.Tensor], anchor_group, anchor_positive: bool):
    if not isinstance(anchor_group, np.ndarray) or not isinstance(anchor_group, torch.Tensor) \
            or not isinstance(anchor_group, Sequence):
        anchor_group = np.atleast_1d(anchor_group)
    if isinstance(labels, torch.Tensor):
        anchor_group = torch.from_numpy(anchor_group).type("torch.LongTensor")
    labels = to_long(labels)
    anchor_group = to_long(anchor_group)
    if not anchor_positive:
        connector = and_
        comparator = ne
    else:
        connector = or_
        comparator = eq
    mask_array = tuple(comparator(labels, x) for x in anchor_group)

    labels = reduce(connector, mask_array)
    return labels
    # if anchor_positive, then True=Positive, otherwise, True=Neg


def validate_array_class(array: Union[torch.Tensor, np.ndarray]):
    assert isinstance(array, np.ndarray) or isinstance(array, torch.Tensor), f"Type disagree: expect" \
        f"np.ndarray or torch.Tensor, got: {type(array)}"


def to_long(array: Union[torch.Tensor, np.ndarray]):
    validate_array_class(array)
    if not isinstance(array, torch.Tensor):
        type_field = 'astype'
        dtype = np.int64
    else:
        type_field = 'type'
        dtype = torch.LongTensor
    return getattr(array, type_field)(dtype)


def array_copy_heper(array: Union[torch.Tensor, np.ndarray]):
    if not isinstance(array, torch.Tensor):
        copy_func = array.copy
    else:
        copy_func = array.clone
    return copy_func()


# buggy. There appears to be an existing implementation in sklearn/metrics/ranking/label_binarize
def label_binarize_vs_rest(labels: Union[np.ndarray, torch.Tensor],
                           anchor_class: int, anchor_positive: bool = True):
    assert isinstance(labels, np.ndarray) or isinstance(labels, torch.Tensor), f"Type disagree: expect" \
        f"np.ndarray or torch.Tensor, got: {type(labels)}"

    labels = array_copy_heper(labels)
    offset_subtract = 1 if anchor_positive else -1
    comparator = eq if anchor_positive else ne
    labels[labels != anchor_class] = anchor_class - offset_subtract

    # assumption that labels here contains two unique values are false
    # labels = (labels == labels.max())
    labels = comparator(labels, anchor_class)
    labels = to_long(labels)
    return labels


def roc_auc_result_dict(auc, fpr, tpr, threshold):
    return {
        "auc": auc,
        "fpr": fpr,
        "tpr": tpr,
        "thresholds": threshold
    }


def roc_auc_helper(y_true, y_pred, original_class_id, roc_auc_dict=None):
    if roc_auc_dict is None:
        roc_auc_dict = dict()
    auc = roc_auc_score(y_true, y_pred)
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    pk_data = roc_auc_result_dict(auc, fpr, tpr, thresholds)
    roc_auc_dict[original_class_id] = pk_data
    return roc_auc_dict


def multi_class_roc_auc_vs_all(y_true: Union[np.ndarray, Sequence[int]],
                               y_pred_all_classes: Union[np.ndarray, Sequence[int]],
                               positive_classes: Union[np.ndarray, Sequence[int]]
                               ) -> Dict[int, Dict[str, Union[float, np.ndarray]]]:
    roc_auc_dict: Dict[int, Dict] = dict()
    y_true = np.atleast_1d(y_true)
    y_pred_all_classes = np.atleast_2d(y_pred_all_classes)

    assert y_pred_all_classes.shape[0] == y_true.size, f"# of data point disagree." \
        f" {y_pred_all_classes.shape[0]} vs. {y_true.size}"
    if y_pred_all_classes.shape[0] == 1:
        raise ValueError(f"y_pred_all_classes has only one row. Expected to be >=2")

    label_binary_array = label_binarize(positive_classes, y_true)
    score_merged = y_pred_all_classes.shape[1] <= 2

    for original_class, y_true in zip(positive_classes, label_binary_array):
        if not score_merged:
            y_score = np.atleast_1d(y_pred_all_classes[:, original_class]).squeeze()
        else:
            y_score = np.atleast_1d(y_pred_all_classes[:, -1]).squeeze()
        roc_auc_dict = roc_auc_helper(y_true, y_score, original_class, roc_auc_dict=roc_auc_dict)
    return roc_auc_dict
