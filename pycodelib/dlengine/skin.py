from .skeletal import AbstractEngine, IteratorBuilder
from typing import Tuple, Dict, Any, Sequence, List
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pycodelib.dataset import Dataset
from tqdm import tqdm
import logging
logging.basicConfig(level=logging.debug)


class H5DataGetter(IteratorBuilder):
    _datasets = None
    BATCH_SIZE: int = 32

    def __init__(self, filename, img_transform, mode):
        super().__init__(filename, img_transform=img_transform)
        self._train_mode = mode

    @classmethod
    def build(cls, filename_dict: Dict[bool, str], img_transform_dict: Dict[bool, object]):
        assert filename_dict.keys() == img_transform_dict.keys(), 'Keys disagree.'
        cls._datasets = {x: H5DataGetter(filename_dict[x], img_transform_dict[x], x) for x in filename_dict.keys()}

    def __len__(self):
        return super().__len__()

    @property
    def mode(self):
        return self._train_mode

    def __getitem__(self, item):
        result = super().__getitem__(item)
        return result  # + tuple([self.mode]) # performed by the on_sample

    @classmethod
    def get_iterator(cls, mode, shuffle, num_workers=6, drop_last=False,  pin_memory=True, batch_size=BATCH_SIZE):
        assert mode == cls._datasets[mode].mode, f"Mode mismatch. input:{mode}"
        return DataLoader(cls._datasets[mode], num_workers=num_workers, pin_memory=pin_memory,
                          batch_size=batch_size, shuffle=mode, drop_last=True)


class SkinEngine(AbstractEngine):

    def __init__(self, device: torch.device, model: nn.Module, loss: nn.Module, iterator_getter: IteratorBuilder,
                 val_phases: Sequence[str, ...]):
        super().__init__(device, model, loss, iterator_getter)
        self.val_phases = val_phases

    def model_eval(self, data_batch: List) -> Tuple[torch.Tensor, torch.Tensor]:
        img, label, *rest = data_batch
        img = img.to(self.device)
        label = label.type('torch.LongTensor').to(self.device)
        prediction = self.model(img)
        loss = self.loss(prediction, label)
        return loss, prediction

    def on_start(self, state: Dict[str, Any]):
        logging.debug(f"Start...")

    def on_end(self, state: Dict[str, Any]):
        torch.cuda.empty_cache()
        logging.debug(f"End...")

    def on_sample(self, state: Dict[str, Any]):
        state['sample'].append(state['train'])
        self.model.train(state['train'])

    def on_update(self, state: Dict[str, Any]):
        ...

    def on_forward(self, state: Dict[str, Any]):
        ...

    def on_start_epoch(self, state: Dict[str, Any]):
        state['train'] = True
        state['iterator'] = tqdm(state['iterator'])

    def on_end_epoch(self, state: Dict[str, Any]):
        state['train'] = False
        self.engine.test(self, self.iterator_getter.get_iterator(mode=False, shuffle=False))

    def evaluation(self, state):
        # todo test
        ...
