from typing import Dict, Callable
from torch.utils.data import DataLoader
import torch.utils.data.dataloader
from pycodelib.dataset import H5SetBasic
from pycodelib.dlengine.skeletal import AbstractIteratorBuilder
from torchvision.transforms import Compose
from typing import List, Tuple
import logging
logging.basicConfig(level=logging.DEBUG)
# batch=[(data[0][idx], data[1][idx], data[2][idx]) for idx in range(data[0].shape[0])]

default_collate: Callable = torch.utils.data.dataloader.default_collate


class BasicTransform(Callable):

    def __init__(self, collate_fn: Callable, transforms: Compose):
        self.collate_fn: Callable = collate_fn
        self.transforms = transforms

    def __call__(self, batch: List[Tuple]):
        raise NotImplementedError


class TransformAutoCollate(BasicTransform):

    def __init__(self, collate_fn: Callable = None, transforms: Callable = None):
        if collate_fn is None:
            collate_fn = default_collate
        super().__init__(collate_fn, transforms)

    def __call__(self, batch: List[Tuple]):
        if self.transforms is not None:
            for idx, data_point in enumerate(batch):
                # logging.debug(f'collate')
                img, label, filenames, index = data_point
                batch[idx] = self.transforms(img), label, img, filenames, index
        batch = self.collate_fn(batch)
        return batch


class H5DataGetter(AbstractIteratorBuilder):
    BATCH_SIZE: int = 32

    @staticmethod
    def trans2collate(transform: Callable):
        if not isinstance(transform, Callable) and transform is not None:
            raise TypeError(f'Transform is not Callable{type(transform)}')
        if not isinstance(transform, BasicTransform):
            return TransformAutoCollate(transforms=transform)
        return transform

    def __init__(self, datasets: Dict[bool, H5SetBasic], img_transform_dict: Dict[bool, Compose]):
        self._datasets_collection: Dict[bool, H5SetBasic] = datasets
        self._transform: Dict[bool, BasicTransform] = {k: type(self).trans2collate(v)
                                                       for k, v in img_transform_dict.items()}
        self._mode: bool = True

    @classmethod
    def build(cls, filename_dict: Dict[bool, str],
              img_transform_dict: Dict[bool, Compose]):
        assert filename_dict.keys() == img_transform_dict.keys(), 'Keys disagree.'
        datasets: Dict[bool, H5SetBasic] = {
                    x: H5SetBasic(filename_dict[x])
                    for x in filename_dict.keys()
        }
        iter_getter = cls(datasets=datasets, img_transform_dict=img_transform_dict)  # {True: None, False: None}
        return iter_getter

    @property
    def datasets_collection(self):
        return self._datasets_collection

    @property
    def transform(self):
        return self._transform

    def get_iterator(self, mode: bool, shuffle: bool = None,
                     num_workers: int = 6,
                     drop_last: bool = False,
                     pin_memory: bool = True,
                     batch_size: int = BATCH_SIZE):
        if shuffle is None:
            shuffle = mode
        return DataLoader(self.datasets_collection[mode], num_workers=num_workers, pin_memory=pin_memory,
                          batch_size=batch_size, shuffle=shuffle, drop_last=True,
                          collate_fn=self.transform[mode])
