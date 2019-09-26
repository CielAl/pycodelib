from typing import Dict, Callable
from torch.utils.data import DataLoader
import torch.utils.data.dataloader
from pycodelib.dataset import H5SetBasic
from pycodelib.dlengine.skeletal import AbstractIteratorBuilder
from torchvision.transforms import Compose
from typing import List, Tuple, Sequence
import logging
logging.basicConfig(level=logging.DEBUG)
# batch=[(data[0][idx], data[1][idx], data[2][idx]) for idx in range(data[0].shape[0])]

default_collate: Callable = torch.utils.data.dataloader.default_collate


class BasicTransform(Callable):
    """
        Fuse the Transformation into Collate_fn.
    """
    def __init__(self, collate_fn: Callable, transforms: Compose):
        self.collate_fn: Callable = collate_fn
        self.transforms = transforms

    def __call__(self, batch: List[Tuple]):
        raise NotImplementedError


class TransformAutoCollate(BasicTransform):
    """
        Using auto_collation after data transformation
    """
    def __init__(self, collate_fn: Callable = None, transforms: Callable = None, label_skipping: Sequence[int] = None):
        if collate_fn is None:
            collate_fn = default_collate
        self.label_skipping = label_skipping
        super().__init__(collate_fn, transforms)

    def __call__(self, batch: List[Tuple]):
        """
            First perform data transformation prior to the collation. The pre-collated data is in form:
            [ (var1, var2, var3)_batch1,  (var1, var2, var3, ...)_batch2,  ...].
            Transformation is done before the collation since the img transformation of torchvision
            is mostly applied to PIL and do not support batch.
            Collation: Each element of list is a batch of a single variable
               e.g. [ batch_of_images,  batch_of_labels].
            Numpy array will be converted to tensors, however, the dim-order will remain.
        Args:
            batch: A list of Tuples. Each Tuple contains variables of a single data point (e.g. img, label, filename)
        Returns:
            Collated Batch: Batch that is Collated and Transformed.
        """
        #  Transformation - semantic coupling
        if self.transforms is not None:
            for idx, data_point in enumerate(batch):
                # extract data
                img, label, mask, row, col, filenames, index = data_point
                # modify the batch: transform img and concat the original one after label.
                batch[idx] = self.transforms(img), label, mask, row, col, img, filenames, index
        if self.label_skipping is not None:
            label_idx = 1
            batch = [data_tuple for data_tuple in batch if data_tuple[label_idx] not in self.label_skipping]
        if len(batch) != 0:
            batch = self.collate_fn(batch)
        return batch


class H5DataGetter(AbstractIteratorBuilder):
    """
        The DataLoader mapper (from mode). H5Dataset Based.
    """
    BATCH_SIZE: int = 32

    @staticmethod
    def trans2collate(transform: Callable, skip_class) -> Callable:
        """
            Fuse the Transformation into the TransformAutoCollate
        Args:
            transform: Callable to perform the transformation

        Returns:
            new_transform: TransformAutoCollate Object. None is also allowed.
        """

        # If not Callable. Note: None is allowed.
        if not isinstance(transform, Callable) and transform is not None:
            raise TypeError(f'Transform is not Callable{type(transform)}')
        # If Callable but not extended from BasicTransform, then perform fusion.
        if not isinstance(transform, BasicTransform):
            return TransformAutoCollate(transforms=transform, label_skipping=skip_class)
        # Otherwise: if extended from BasicTransform or is None, identity mapping
        return transform

    def __init__(self, datasets: Dict[bool, H5SetBasic], img_transform_dict: Dict[bool, Compose], skip_class):
        """

        Args:
            datasets: Dict of [mode, h5dataset]
            img_transform_dict: Dict of [mode, Compose]
        """
        self._datasets_collection: Dict[bool, H5SetBasic] = datasets
        # Convert the Callable to BasicTransform Object
        self._transform: Dict[bool, BasicTransform] = {k: H5DataGetter.trans2collate(v, skip_class)
                                                       for k, v in img_transform_dict.items()}
        # -todo later
        self._mode: bool = True

    @classmethod
    def build(cls, filename_dict: Dict[bool, str],
              img_transform_dict: Dict[bool, Compose],
              skip_class):
        """
            Factory builder.
        Args:
            filename_dict: Dict in [mode, filename]
            img_transform_dict: Dict of [mode, transformation]. None is allowed.
            skip_class:
        Returns:
            iter_getter (H5DataGetter):
        """
        # Make sure the input dicts has aligned keys.
        assert filename_dict.keys() == img_transform_dict.keys(), 'Keys disagree.'
        # Generate dataset map.
        datasets_dict: Dict[bool, H5SetBasic] = {
                    x: H5SetBasic(filename_dict[x])
                    for x in filename_dict.keys()
        }
        # instantiation
        return cls(datasets=datasets_dict, img_transform_dict=img_transform_dict, skip_class=skip_class)
        # {True: None, False: None}

    @property
    def datasets_collection(self):
        """

        Returns:
            Collection of Datasets per mode.
        """
        return self._datasets_collection

    @property
    def transform(self) -> Dict[bool, BasicTransform]:
        """
            transform property.
        Returns:
            _transform: The transformation dict.
        """
        return self._transform

    def get_iterator(self,
                     mode: bool,
                     shuffle: bool = None,
                     num_workers: int = 6,
                     drop_last: bool = False,
                     pin_memory: bool = True,
                     batch_size: int = BATCH_SIZE):
        """
            The visible interface to provide the DataLoader given training/testing mode.
        Args:
            mode:   True if training. False if testing.
            shuffle: Whether shuffle the data. If unspecified, then shuffle only in training phase.
            num_workers: See DataLoader.
            drop_last:  See DataLoader.
            pin_memory: See DataLoader.
            batch_size: See DataLoader.

        Returns:

        """
        if shuffle is None:
            shuffle = mode
        return DataLoader(self.datasets_collection[mode], num_workers=num_workers, pin_memory=pin_memory,
                          batch_size=batch_size, shuffle=shuffle, drop_last=True,
                          collate_fn=self.transform[mode])
