from collections import OrderedDict
import torch
from torch import nn
from torch.optim import Optimizer
from typing import Dict, Union
import numpy as np
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def load_dict(obj, path: str):
    state_dict = torch.load(path)
    obj.load_state_dict(state_dict)
    return obj


class ModelExporter:

    def __init__(self):
        super().__init__()
        self._loss_vals: Dict[str, float] = OrderedDict()

    @staticmethod
    def tensor_to_numpy(x):
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
        return x

    def log_loss(self, loss_name: str, loss_value: float):
        loss_value = ModelExporter.tensor_to_numpy(loss_value)
        logger.debug(f"Before Updating{self._loss_vals.get(loss_name, 'unspecified')}")
        self._loss_vals[loss_name] = loss_value
        logger.debug(f"Updated{self._loss_vals[loss_name]}")

    def is_best_loss(self, loss_name: str, loss_value: float, update: bool):
        loss_value = ModelExporter.tensor_to_numpy(loss_value)
        current_loss = self._loss_vals.get(loss_name, np.inf)
        is_best = loss_value <= current_loss
        logger.debug(f"current var {current_loss}. vs. {loss_value}. Is best? {is_best}")
        if update and is_best:
            self.log_loss(loss_name, loss_value)
        return is_best

    @staticmethod
    def save_obj(obj: Union[Optimizer, nn.Module], save_path: str):
        if isinstance(obj, nn.DataParallel):
            obj = obj.module
        torch.save(obj.state_dict(), save_path)

    def save_on_best(self,
                     obj: Union[Optimizer, nn.Module],
                     save_path: str,
                     loss_name: str,
                     loss_value: float,
                     update: bool):
        is_best = self.is_best_loss(loss_name, loss_value=loss_value, update=update)
        if is_best:
            logger.debug('best loss: save')
            # if isinstance(obj, nn.DataParallel):
            #    obj = obj.module
            # torch.save(obj.state_dict(), save_path)
            ModelExporter.save_obj(obj, save_path)
        return is_best

    def reset_loss(self, loss_name: str):
        self._loss_vals[loss_name] = np.inf

    def reset_all(self):
        for key in self._loss_vals.keys():
            self.reset_loss(key)
