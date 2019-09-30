from abc import ABC, abstractmethod
import torch.nn as nn
from typing import Sequence, Union
from torch import device


class BaseModelContainer(nn.Module):
    @property
    def model(self) -> nn.Module:
        return self.__model

    def __init__(self, model):
        super().__init__()
        self.__model = model

    def forward(self, *inputs):
        return self.model(*inputs)


class MultiGPUContainer(BaseModelContainer):
    @property
    def gpu_ids(self) -> Union[Sequence[int], device]:
        return self.__gpu_ids

    @gpu_ids.setter
    def gpu_ids(self, gpu_ids: Union[Sequence[int], device] = None):
        self.__gpu_ids = gpu_ids

    def __init__(self, model: nn.Module, gpu_ids: Union[Sequence[int], device] = None):
        super().__init__(model=model)
        self.__gpu_ids = gpu_ids

    def forward(self, *inputs):
        if self.gpu_ids is not None:
            return nn.parallel.data_parallel(self.modules, inputs, self.gpu_ids)
        return self.models(*inputs)


class MergeContainer(ABC, BaseModelContainer):

    @property
    def features(self):
        return self.features

    def __init__(self, model: nn.Module):
        super().__init__(model=model)

    @abstractmethod
    def get_merge_target(self, *kwargs):
        ...

    def forward(self, *inputs, **kwargs):
        raise NotImplementedError


class _MergeBlock(nn.Module):

    def __init__(self, *model_list: MergeContainer):
        super().__init__()
        self.modules_list = nn.ModuleList(model_list)

    def merge(self, *inputs):
        # use Identity Map if not match
        assert len(inputs) == len(self.modules_list)
        result = [model(x) for model, x in zip(self.modules_list, inputs)]
        return result

    def forward(self, *inputs):
        # simply concat results in a list
        return self.merge(*inputs)


class AbstractMergeNet(ABC, nn.Module):

    def __init__(self, merge_block: _MergeBlock):
        super().__init__()
        self.merge_block = merge_block

    def forward(self, *inputs):
        raise NotImplementedError
