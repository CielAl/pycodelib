from typing import Dict

from torch.utils.data import DataLoader
# from torch.utils.data import Dataset
from pycodelib.dataset import H5SetTransform as H5Set
from pycodelib.dlengine.skeletal import IteratorBuilder


class H5DataGetter(IteratorBuilder):
    _datasets_container: Dict[bool, IteratorBuilder] = None
    BATCH_SIZE: int = 32

    def __init__(self, filename: str, img_transform, mode, constructor: type = H5SetTransform):
        dataset: constructor = constructor(filename, img_transform=img_transform)
        super().__init__(dataset)
        self._train_mode = mode


    def build(self, filename_dict: Dict[bool, str], img_transform_dict: Dict[bool, object]):
        assert filename_dict.keys() == img_transform_dict.keys(), 'Keys disagree.'
        cls._datasets_container: Dict[bool, cls] = {x: H5DataGetter(filename_dict[x], img_transform_dict[x], x)
                                                    for x in filename_dict.keys()}
        return cls

    @property
    def mode(self):
        return self._train_mode

    @property
    def dataset(self):
        return super()._dataset

    @classmethod
    def get_iterator(cls, mode: bool, shuffle: bool,
                     num_workers: int = 6,
                     drop_last: bool = False,
                     pin_memory: bool = True,
                     batch_size: int = BATCH_SIZE):
        assert mode == cls._datasets_container[mode].mode, f"Mode mismatch. input:{mode}"
        return DataLoader(cls._datasets_container[mode].dataset, num_workers=num_workers, pin_memory=pin_memory,
                          batch_size=batch_size, shuffle=mode, drop_last=True)
