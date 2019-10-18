from typing import Callable, List, Tuple, Sequence, Set, Union

import numpy as np
import torch.utils
from torchvision.transforms import Compose

from pycodelib.common import require_not_none
from pycodelib.dataset import DatasetItem
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# noinspection PyUnresolvedReferences
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
    def __init__(self, collate_fn: Callable = None, transforms: Callable = None,
                 label_skipping: Sequence[int] = None,
                 group_level: int = 0,
                 img_key: str = 'img',
                 label_key: str = 'label',
                 original_key: str = 'img_original',
                 type_order: np.ndarray = None):
        if collate_fn is None:
            collate_fn = default_collate
        require_not_none(img_key, 'img_key')
        require_not_none(label_key, 'label_key')
        require_not_none(original_key, 'original_key')
        super().__init__(collate_fn, transforms)

        self.label_skipping = label_skipping
        self._label_set: Set = set(self.label_skipping) if self.label_skipping is not None else None
        self._group_level: int = group_level
        self._img_key = img_key
        self._original_key = original_key
        self._label_key = label_key
        if type_order is not None:
            self._type_order = np.atleast_1d(type_order).ravel()
        self._type_order = type_order

    @property
    def group_level(self) -> int:
        return self._group_level

    @property
    def original_key(self):
        return self._original_key

    @property
    def img_key(self):
        return self._img_key

    @property
    def label_key(self):
        return self._label_key

    @property
    def type_order(self):
        return self._type_order

    def transform_helper(self, x: np.ndarray):
        if x.ndim <= 3:
            return self.transforms(x)
        temp_list = []
        for element in x:
            temp_list.append(self.transforms(element))
        # breakpoint()
        dtype = type(temp_list[0])
        if issubclass(dtype, torch.Tensor):
            result = torch.stack(temp_list)
        else:
            result = np.asarray(temp_list)
        return result

    def __call__(self, batch: List[Union[Tuple, DatasetItem]]):
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
            # data_point as a single element in a batch
            """
            # no big difference. For loop is more clear and readable.
            [
                data_point.set(self.original_key, data_point[self.img_key])
                .set(self.img_key, self.transform_helper(data_point[self.img_key]))
                .re_order(self.type_order)
                for data_point in batch
            ]
            """
            for idx, data_point in enumerate(batch):
                # extract data
                # img, label, mask, *rest = data_point
                img = data_point[self.img_key]
                data_point[self.img_key] = self.transform_helper(img)
                data_point[self.original_key] = img
                data_point.re_order(self.type_order, inplace_dict=True)
                # batch[idx] = self.transform_helper(img), label, mask, img, *rest
        if self.label_skipping is not None and len(self.label_skipping) > 0:
            # setA - setB is the difference between A and B, i.e. elements in A but not in B.
            # set(np.atleast_1d(data_item[self.label_key])) is the set of single element, which
            # contains a single label in a batch. The result is {x| x in current data, x not in skipped labels}
            # if the length of the resulting set is NOT 0, then x should not be skipped.
            batch = [data_item for data_item in batch
                     if len(
                            set(np.atleast_1d(data_item[self.label_key]))
                            - self._label_set)
                     != 0]
        if len(batch) != 0 and self.group_level == 0:
            batch = self.collate_fn(batch)
        elif self.group_level > 0:
            # unpacking the grouped data
            batch = batch[0]
        else:
            logger.warning(f"Empty Batch. len:{len(batch)}. grp:{self.group_level}")
        return batch
