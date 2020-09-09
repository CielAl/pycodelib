from typing import Dict, Callable, Sequence, Union
from torch.utils.data import DataLoader
from pycodelib.dataset import H5SetBasic, AbstractDataset
from pycodelib.dlengine.skeletal import IteratorBuilderDictSet
from torchvision.transforms import Compose
from copy import deepcopy
import numpy as np
import logging

from pycodelib.dlengine.transform_collate import BasicTransform, TransformAutoCollate

logger = logging.getLogger(__name__)
logger.setLevel(level=logging.DEBUG)
# batch=[(data[0][idx], data[1][idx], data[2][idx]) for idx in range(data[0].shape[0])]


class BaseGetter(IteratorBuilderDictSet):
    """
        The DataLoader mapper (from mode). H5Dataset Based.
    """
    BATCH_SIZE: int = 32

    @staticmethod
    def trans2collate(transform: Callable, skip_class, group_level,
                      img_key: str = 'img',
                      label_key: str = 'label',
                      original_key: str = 'img_original',
                      type_order: np.ndarray = None) -> TransformAutoCollate:
        """
            Fuse the Transformation into the TransformAutoCollate
        Args:
            transform: Callable to perform the transformation
            skip_class (Sequence[int]): List of class id to skip.
            group_level (int): Group level of the dataset. 0: No grouping. 1: Grouping by image.
            type_order ():
            original_key ():
            label_key ():
            img_key ():
        Returns:
            new_transform: TransformAutoCollate Object. None is also allowed.
        """

        # If not Callable. Note: None is allowed.
        if not isinstance(transform, Callable) and transform is not None:
            raise TypeError(f'Transform is not Callable{type(transform)}')
        # If Callable but not extended from BasicTransform, then perform fusion.
        if not isinstance(transform, TransformAutoCollate):
            transform = TransformAutoCollate(transforms=transform,
                                             label_skipping=skip_class,
                                             group_level=group_level,
                                             img_key=img_key,
                                             label_key=label_key,
                                             original_key=original_key,
                                             type_order=type_order,
                                             )
        # Otherwise: if extended from BasicTransform or is None, identity mapping
        return transform

    @staticmethod
    def collate_dict(img_transform_dict: Dict[bool, Compose], skip_class,
                     group_level: int = 0,
                     img_key: str = 'img',
                     label_key: str = 'label',
                     original_key: str = 'img_original',
                     type_order: np.ndarray = None
                     ):
        """

        Args:
            img_transform_dict: Dict of [mode, Compose]
            skip_class (Sequence[int]): List of class id to skip
            group_level (int): group level of dataset. 0: Un-grouped. 1: Grouped by image level. Default: 0.
            img_key ():
            label_key ():
            original_key ():
            type_order ():
        """
        # Convert the Callable to BasicTransform Object
        transform: Dict[bool, BasicTransform] = {k: H5DataGetter.trans2collate(v, skip_class=skip_class,
                                                                               group_level=group_level,
                                                                               img_key=img_key,
                                                                               label_key=label_key,
                                                                               original_key=original_key,
                                                                               type_order=type_order,
                                                                               )
                                                 for k, v in img_transform_dict.items()}
        return transform

    def __init__(self, data_sets: Dict[bool, AbstractDataset], trans_collate: Dict[bool, TransformAutoCollate],
                 dl_constructor: Callable):
        super().__init__(data_sets)
        self._transform: Dict[bool, TransformAutoCollate] = deepcopy(trans_collate)

        # -todo later
        self._mode: bool = True
        set_group_level = set(v.group_level for v in self._transform.values())
        assert len(set_group_level) == 1, f"Inconsistent group level. Got {set_group_level}"
        self.group_level = list(set_group_level)[0]
        self.dl_constructor = dl_constructor

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
                     batch_size: int = BATCH_SIZE,
                     truncate_size: float = np.inf,
                     flatten_output: bool = False,):
        """
            The visible interface to provide the DataLoader given training/testing mode.
        Args:
            mode:   True if training. False if testing.
            shuffle: Whether shuffle the data. If unspecified, then shuffle only in training phase.
            num_workers: See DataLoader.
            drop_last:  See DataLoader.
            pin_memory: See DataLoader.
            batch_size: See DataLoader.
            truncate_size: See AbstractDataset
            flatten_output: See AbstractDataset.

        Returns:

        """
        assert self.group_level == 0 or batch_size == 1, f'group_level 1 only supports batch_of_1'
        if shuffle is None:
            shuffle = mode
        self.data_sets_collection[mode].flatten_output = flatten_output
        self.data_sets_collection[mode].truncate_size = truncate_size
        return self.dl_constructor(self.data_sets_collection[mode],
                                   num_workers=num_workers, pin_memory=pin_memory,
                                   batch_size=batch_size, shuffle=shuffle, drop_last=drop_last,
                                   collate_fn=self.transform[mode])

    @classmethod
    def build(cls, data_sets_dict: Dict,
              img_transform_dict: Dict[bool, Union[Compose, None]],
              skip_class,
              group_level: int = 0,
              img_key: str = 'img',
              label_key: str = 'label',
              original_key: str = 'img_original',
              type_order: np.ndarray = None,
              dl_constructor: Callable = DataLoader,
              ):
        """
            Factory builder.
        Args:
            data_sets_dict: Dict in [mode, dataset]
            img_transform_dict: Dict of [mode, transformation]. None is allowed.
            skip_class:
            group_level:
            img_key ():
            label_key ():
            original_key ():
            type_order ():
            dl_constructor (): Constructor of DataLoader
        Returns:
            iter_getter (H5DataGetter):
        """
        # Make sure the input dicts has aligned keys.
        # instantiation
        img_transform_dict = cls.collate_dict(img_transform_dict=img_transform_dict,
                                              skip_class=skip_class,
                                              group_level=group_level,
                                              img_key=img_key,
                                              label_key=label_key,
                                              original_key=original_key,
                                              type_order=type_order,
                                              )

        return cls(data_sets=data_sets_dict, trans_collate=img_transform_dict,
                   dl_constructor=dl_constructor)


class H5DataGetter(BaseGetter):

    @classmethod
    def build(cls, filename_dict: Dict[bool, str],
              img_transform_dict: Dict[bool, Union[Compose, None]],
              skip_class,
              group_level: int = 0,
              img_key: str = 'img',
              label_key: str = 'label',
              original_key: str = 'img_original',
              type_order: np.ndarray = None,
              flatten_output: bool = False
              ):
        """
            Factory builder.
        Args:
            filename_dict: Dict in [mode, filename]
            img_transform_dict: Dict of [mode, transformation]. None is allowed.
            skip_class:
            group_level:
            img_key ():
            label_key ():
            original_key ():
            type_order ():
            flatten_output (bool): See AbstractDataset
        Returns:
            iter_getter (H5DataGetter):
        """
        # Make sure the input dicts has aligned keys.
        assert filename_dict.keys() == img_transform_dict.keys(), 'Keys disagree.'
        # Generate dataset map.
        data_sets_dict: Dict[bool, H5SetBasic] = {
                    x: H5SetBasic(filename_dict[x], group_level=group_level, flatten_output=flatten_output)
                    for x in filename_dict.keys()
        }
        # instantiation
        img_transform_dict = cls.collate_dict(img_transform_dict=img_transform_dict,
                                              skip_class=skip_class,
                                              group_level=group_level,
                                              img_key=img_key,
                                              label_key=label_key,
                                              original_key=original_key,
                                              type_order=type_order,
                                              )

        return BaseGetter.build(data_sets_dict, img_transform_dict,
                                skip_class=skip_class,
                                group_level=group_level,
                                img_key=img_key,
                                label_key=label_key,
                                original_key=original_key,
                                type_order=type_order)
        # return cls(data_sets=data_sets_dict, trans_collate=img_transform_dict)
        # {True: None, False: None}
