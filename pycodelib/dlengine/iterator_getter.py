from typing import Dict, Callable
from torch.utils.data import DataLoader
# from torch.utils.data import Dataset
from pycodelib.dataset import H5SetBasic
from pycodelib.dlengine.skeletal import AbstractIteratorBuilder


class H5DataGetter(AbstractIteratorBuilder):
    BATCH_SIZE: int = 32

    def __init__(self, datasets: Dict[bool, H5SetBasic], img_transform_dict: Dict[bool, Callable]):
        self._datasets_collection: Dict[bool, H5SetBasic] = datasets
        self._transform: Dict[bool, Callable] = img_transform_dict
        self._mode: bool = True

    @classmethod
    def build(cls, filename_dict: Dict[bool, str], img_transform_dict: Dict[bool, Callable]):
        assert filename_dict.keys() == img_transform_dict.keys(), 'Keys disagree.'
        datasets: Dict[bool, H5SetBasic] = {
                    x: H5SetBasic(filename_dict[x])
                    for x in filename_dict.keys()
        }
        iter_getter = cls(datasets=datasets, img_transform_dict=img_transform_dict)
        return iter_getter

    @property
    def datasets_collection(self):
        return self._datasets_collection

    def get_iterator(self, mode: bool, shuffle: bool = None,
                     num_workers: int = 6,
                     drop_last: bool = False,
                     pin_memory: bool = True,
                     batch_size: int = BATCH_SIZE):
        if shuffle is None:
            shuffle = mode
        return DataLoader(self.datasets_collection[mode], num_workers=num_workers, pin_memory=pin_memory,
                          batch_size=batch_size, shuffle=shuffle, drop_last=True)
