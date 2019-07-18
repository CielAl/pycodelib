from .skeletal import AbstractEngine, IteratorBuilder
from typing import Tuple, Dict, Any, Sequence, List
import torch
import torch.nn as nn
from tqdm import tqdm
import logging
logging.basicConfig(level=logging.debug)


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
        prediction = state['output'].data
        loss = state['loss']
        label = state['sample'][1].type("torch.LongTensor")


