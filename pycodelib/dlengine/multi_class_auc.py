from typing import Dict, Any, Tuple, Hashable, List, Sequence, Union
import torch
import numpy as np
import numbers
from torchnet.meter.meter import Meter
from torchnet.meter import AUCMeter
from pycodelib.patients import SheetCollection, CascadedPred
from pycodelib.common import default_not_none
from sklearn.metrics import confusion_matrix
from pycodelib.metrics import multi_class_roc_auc_vs_all
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class MultiAUC(Meter):

    def __init__(self, num_class):
        super().__init__()
        self.num_class = num_class
        self.meters: List[AUCMeter] = [AUCMeter() for k in range(self.num_class)]

    @staticmethod
    def to_numpy(x):
        assert isinstance(x, (torch.Tensor, np.ndarray))
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
        x = np.atleast_1d(x.squeeze())
        return x

    # noinspection PyMethodOverriding
    def add(self, output, target):
        output = np.atleast_2d(MultiAUC.to_numpy(output))
        target = MultiAUC.to_numpy(target)
        for target_ind in range(self.num_class):
            target_binary_label = (target == target_ind).astype(np.int64)
            self.meters[target_ind].add(output[:, target_ind],
                                        target_binary_label)

    def value(self):
        area_list = []
        tpr_list = []
        fpr_list = []
        for target_ind in range(self.num_class):
            area, tpr, fpr = self.meters[target_ind].value()
            area_list.append(area)
            tpr_list.append(tpr)
            fpr_list.append(fpr)
        return area_list, tpr_list, fpr_list

    def reset(self):
        for target_ind in range(self.num_class):
            self.meters[target_ind].reset()
