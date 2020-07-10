from functools import reduce
from operator import eq, ne, or_, and_
from typing import Dict, Union, Sequence, Iterable

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import label_binarize


# todo wrap the class-mapping from multi_class_roc_auc
def is_array_class(array):
    return isinstance(array, np.ndarray) or isinstance(array, torch.Tensor)


def validate_array_class(array: Union[torch.Tensor, np.ndarray]):
    assert is_array_class(array), f"Type disagree: expect" \
        f"np.ndarray or torch.Tensor, got: {type(array)}"


def array_to_numpy(array: Union[torch.Tensor, np.ndarray]):
    validate_array_class(array), f"not ndarray or tensor. {type(array)}"
    if isinstance(array, np.ndarray):
        return array
    return array.cpu().detach().numpy()


def to_long(array: Union[torch.Tensor, np.ndarray]):
    validate_array_class(array)
    if not isinstance(array, torch.Tensor):
        type_field = 'astype'
        dtype = np.int64
    else:
        type_field = 'type'
        dtype = torch.LongTensor
    return getattr(array, type_field)(dtype)


def is_sequence_alike(obj):
    return isinstance(obj, Sequence) or isinstance(obj, Iterable) or is_array_class(obj)


def validate_parititon_structure(label_group):
    assert is_sequence_alike(label_group), f"1st level structure must be sequence-alike. Got{type(label_group)}"
    for x in label_group:
        assert is_sequence_alike(x), f"2nd level must be sequence-alike. Got{type(x)}"


def label_group_elements(label_group: Sequence[Sequence[int]]):
    set_list = [set(x) for x in label_group]
    set_combined = reduce(or_, set_list)
    return set_list, set_combined


def is_disjoint_partition(label_group: Sequence[Sequence[int]]):
    validate_parititon_structure(label_group)

    set_list, set_combined = label_group_elements(label_group)
    sum_len = np.asarray([len(set_obj) for set_obj in set_list]).sum()

    length_combined = len(set_combined)
    # if length of combined set mismatch sum_len, then there are repetitive elements
    return length_combined == sum_len


def validate_disjoint_partition(label_group: Sequence[Sequence[int]]):
    assert is_disjoint_partition(label_group), f"Not Disjoint: {label_group}"


def is_label_defined(labels: Union[np.ndarray, torch.Tensor], label_group: Sequence[Sequence[int]]):
    set_list, set_combined = label_group_elements(label_group)
    labels = array_to_numpy(labels)
    check_result = np.asarray([label_value in set_combined for label_value in labels])
    bool_all_defined = check_result.all()
    set_undefined_labels = set(labels[~check_result])
    return bool_all_defined, set_undefined_labels


def validate_label_definition(labels: Union[np.ndarray, torch.Tensor], label_group: Sequence[Sequence[int]]):
    is_all_defined, set_undefined_labels = is_label_defined(labels, label_group)
    assert is_all_defined, f"Unexpected Label Encounted: {set_undefined_labels}"


def label_encode_by_partition(labels: Union[np.ndarray, torch.Tensor], label_group: Sequence[Sequence[int]]):
    # precondition_1: disjoint
    validate_disjoint_partition(label_group)
    # precondition_2: all labels must be defined in label_group
    validate_label_definition(labels, label_group)
    # x in labels as 0/1 assignment. idx as the target encoded label value defined by the order of label_group
    # if pack/partition in label_group are disjoint, and there are no unexpected labels
    # that are not defined in the label_group then the corresponding
    # idx * (x in pack) across each pack (column) in label_group should have exactly one non-zero value
    # use sum to get the final encoded label.
    labels = array_to_numpy(labels)
    final_label = np.asarray([[idx * (x in pack) for x in labels] for idx, pack in enumerate(label_group)])\
        .sum(axis=0)
    return final_label


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
    # mask array is a tuple of arrays, where each array performs the comparison
    # between label and anchor element
    # for positive anchor --> comparator is eq, as any label == anchor yields 1, and 0 otherwise.
    # for negative anchor --> use ne: label != anchor yields 1, and 0 otherwise.
    # connector: for positive anchor, label is mapped to 1 if it matches any of the anchor element
    # for negative anchor: not [or] ==> and.
    # not ( label == anchor1 or label== anchor2... ) ==> label != anchor1 and label!= anchor2 ...
    mask_array = tuple(comparator(labels, x) for x in anchor_group)

    labels = reduce(connector, mask_array)
    return labels
    # if anchor_positive, then True=Positive, otherwise, True=Neg


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
