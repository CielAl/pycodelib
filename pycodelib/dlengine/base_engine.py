from .skeletal import AbstractEngine, IteratorBuilder
from .multi_instance_meter import MultiInstanceMeter
from pycodelib.patients import SheetCollection
from typing import Tuple, Dict, Any, Sequence, List, Callable
import torch
import torch.nn as nn
from torchnet.meter import AverageValueMeter, ClassErrorMeter, ConfusionMeter, AUCMeter
from tqdm import tqdm
import logging
logging.basicConfig(level=logging.debug)


class SkinEngine(AbstractEngine):

    def __init__(self, device: torch.device, model: nn.Module, loss: nn.Module, iterator_getter: IteratorBuilder,
                 val_phases: Sequence[str, ...],
                 class_names: int,
                 patient_col: SheetCollection):
        super().__init__(device, model, loss, iterator_getter)
        self.val_phases = val_phases
        self.class_names = class_names
        self.num_classes = len(class_names)
        self.membership = None  # todo
        self.patient_col = patient_col

        self.add_meter('accuracy_meter', ClassErrorMeter(accuracy=True))
        self.add_meter('conf_meter', ConfusionMeter(num_classes, normalized=False))
        self.add_meter('loss_meter', AverageValueMeter())
        self.add_meter('multi_instance_meter', MultiInstanceMeter())
        self.add_meter('auc_meter', AUCMeter())

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

    def on_start_epoch(self, state: Dict[str, Any]):
        self._reset_meters()
        state['train'] = True
        state['iterator'] = tqdm(state['iterator'])

    def on_end_epoch(self, state: Dict[str, Any]):
        state['train'] = False
        self.engine.test(self, self.iterator_getter.get_iterator(mode=False, shuffle=False))

    def _reset_meters(self):
        for k, v in self.meter_dict.items():
            v.reset()

    def _evaluation(self, state):
        # todo test
        ...
        print('[Epoch %d] Testing Loss: %.4f (Accuracy: %.2f%%)' % (
            state['epoch'], self.meter_dict['loss_meter'].value()[0], self.meter_dict['accuracy_meter'].value()[0]))
        print(self.meter_dict['conf_meter'].value())
        self._fetch_prediction()

    def _fetch_prediction(self):
        patch_score = self.meter_dict['multi_instance_meter'].value()
        patch_label_collection = {
                        key: [score_array.argmax(dim=1) for score_array in score_array_list]
                        for (key, score_array_list) in patch_score.items()
        }
        patch_label_prediction = {
            key: max(col)
            for (key, col) in patch_label_collection.items()
        }
        pred_labels = list(patch_label_prediction.values())
        pred_labels_str = [self.class_names[x] for x in pred_labels]
        filenames = list(patch_label_prediction.keys())
        self.patient_col.load_prediction(pred_labels_str, filenames)

    # measurements
    def on_forward(self, state: Dict[str, Any]):
        softmax: Callable = nn.Softmax(dim=1)
        pred_data = state['output'].data
        loss_scalar = state['loss'].item()

        img_new, label, img, filename, *rest = state['sample']
        label = label.type("torch.LongTensor")
        self.meter_dict['loss_meter'].add(loss_scalar)
        self.meter_dict['accuracy_meter'].add(pred_data, label)

        pred_expand: torch.Tensor = pred_data
        if pred_data.ndimension(pred_data) == 1:
            pred_expand = pred_data[None]
        self.meter_dict['conf_meter'].add(pred_expand, label)
        pred_softmax = softmax(pred_expand)
        patch_name_score: Tuple = (filename, pred_softmax)
        self.meter_dict['multi_instance_meter'].add(patch_name_score)

        if self.num_classes == 2:
            self.meter_dict['auc_meter'].add(pred_softmax[:, 1], label)
