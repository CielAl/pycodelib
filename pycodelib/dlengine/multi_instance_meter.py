from typing import Dict, Any, Tuple, Hashable, List, Sequence, Union
import torch
import numpy as np
import numbers
from torchnet.meter.meter import Meter
from pycodelib.patients import SheetCollection
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class MultiInstanceMeter(Meter):
    """
        Bag of scores by groups.
    """
    @property
    def instance_map(self) -> Dict[Hashable, Any]:
        """
        Returns:
            instance_map
        """
        return self._instance_map

    @property
    def patient_col(self) -> SheetCollection:
        """
        Returns:
            patient_col
        """
        return self._patient_col

    def __init__(self, patient_col: SheetCollection):
        """
            Constructor.
        Args:
            patient_col: The SheetCollection of Patient Table.
        """
        super().__init__()
        self._instance_map: Dict[Hashable, Any] = dict()
        self._patient_col: SheetCollection = patient_col

    @staticmethod
    def vectorized_obj(obj_in: Any, scalar_type: type, at_least_1d=False, err_msg: str = ""):
        """
            Vectorize the input
        Args:
            obj_in: Original input
            scalar_type:    Type if scalar
            at_least_1d:    Whether or not guarantee the dimension
            err_msg:    Message to print if cannot be vectorized

        Returns:
            obj: vectorized form.
        """
        is_matched_scalar: bool = isinstance(obj_in, scalar_type)
        already_vector: bool = isinstance(obj_in, Sequence) or isinstance(obj_in, np.ndarray)

        if torch.is_tensor(obj_in):
            obj: np.ndarray = obj_in.cpu().squeeze().numpy()
        # pack to np array if a scalar of matched type or is already in vector form.
        elif is_matched_scalar or already_vector:
            obj: np.ndarray = np.asarray([obj_in])
        else:
            raise TypeError(err_msg)
        if at_least_1d:
            obj: np.ndarray = np.atleast_1d(obj)
        else:
            obj: np.ndarray = obj.squeeze()
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
        keys = type(self).vectorized_obj(keys_in, Hashable, at_least_1d=False, err_msg=f"{type(keys_in)}")
        values = type(self).vectorized_obj(values_in, numbers.Number, at_least_1d=True, err_msg=f"{type(values_in)}")

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

    def value(self):
        """
            todo Move the fetch_prediction here.
        Returns:
            todo
        """
        # for each v: List of array<prob_c1, prob_c2, ..., prob_cn>
        for key, scores in self.instance_map.items():
            # self.patient_col.load_prediction()
            breakpoint()
        return self.instance_map.keys(), self.instance_map.values()

    def reset(self):
        """
            Clear the map.
            todo - clear the patient collection
        Returns:
            None
        """
        self._instance_map.clear()
        self.patient_col.flush_df()

    def _fetch_prediction(self):
        """
            -todo Consider move to the multi_instance_meter
        Returns:

        """
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