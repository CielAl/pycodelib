from sklearn.metrics import roc_auc_score, roc_curve
from typing import Dict, Union
import numpy as np
# todo wrap the class-mapping from multi_class_roc_auc


def multi_2_binary(y_true, y_pred, class_id):
    y_true_vs_rest = y_true.copy()
    y_true_vs_rest[y_true_vs_rest != class_id] = class_id - 1
    y_true_vs_rest += 1
    y_score_vs_rest = y_pred.copy()[:, class_id]
    return y_score_vs_rest, y_score_vs_rest


def multi_class_roc_auc(y_true, y_pred, positive_class) -> Dict[int, Dict[str, Union[float, np.ndarray]]]:
    roc_auc_dict: Dict[int, Dict] = dict()
    for class_id in positive_class:
        y_true_vs_rest, y_score_vs_rest = multi_2_binary(y_true, y_pred, class_id)
        auc = roc_auc_score(y_true_vs_rest, y_score_vs_rest)
        fpr, tpr, thresholds = roc_curve(y_true_vs_rest, y_score_vs_rest, pos_label=class_id)
        pk_data = {
            "auc": auc,
            "fpr": fpr,
            "tpr": tpr,
            "thresholds": thresholds
        }
        roc_auc_dict[class_id] = pk_data
        return roc_auc_dict
