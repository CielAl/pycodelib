"""
Some of my very old implementation of dict-based dataset with output fields formatted --> seems like it's 
no longer necessary with the presence of better wrappers of trainers such as Lightning.
"""
import os
import glob
import tables
from abc import abstractmethod
import numpy as np
import torch
import torchvision
import imageio
from torch.utils.data import Dataset as TorchDataset
# from torchvision.datasets.folder import make_dataset
from typing import Sequence, Callable, List, Dict, Union, Iterable, Tuple
from pycodelib import common
from openslide import OpenSlide
from collections import OrderedDict
import platform
import sys
from collections.abc import Mapping
import re
from PIL import Image
from lazy_property import LazyProperty
from torchvision.datasets.folder import IMG_EXTENSIONS
import logging
logger = logging.getLogger(__name__)
logger.setLevel(level=logging.CRITICAL)

Image.MAX_IMAGE_PIXELS = 1000000000
# from abc import ABC, abstractmethod

make_dataset_default = torchvision.datasets.folder.make_dataset
has_file_allowed_extension = torchvision.datasets.folder.has_file_allowed_extension


def default_loader(path: str):

    from torchvision import get_image_backend
    from PIL import ImageFile
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def make_dataset_folder(directory, class_to_idx, extensions=None, is_valid_file=None):
    """
    todo
    Args:
        directory:
        class_to_idx:
        extensions:
        is_valid_file:

    Returns:

    """
    instances = []
    directory = os.path.expanduser(directory)
    both_none = extensions is None and is_valid_file is None
    both_something = extensions is not None and is_valid_file is not None
    if both_none or both_something:
        raise ValueError("Both extensions and is_valid_file cannot be None or not None at the same time")
    if extensions is not None:
        def is_valid_file(x):
            return has_file_allowed_extension(x, extensions)
    for target_class in sorted(class_to_idx.keys()):
        class_index = class_to_idx[target_class]
        target_dir = os.path.join(directory, target_class)
        if not os.path.isdir(target_dir):
            continue
        for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
            for fname in sorted(fnames):
                path = os.path.join(root, fname)
                if is_valid_file(path):
                    item = path, class_index
                    instances.append(item)
    return instances


def validate_compatibility():
    impl = platform.python_implementation()
    major = sys.version_info.major
    minor = sys.version_info.minor
    cpython_36 = impl == 'CPython' and ((major == 3 and minor >= 6) or major > 3)
    python37 = ((major == 3 and minor >= 7) or major > 3)
    assert cpython_36 or python37


validate_compatibility()


class DatasetItem(Mapping):
    """
    Must Inherit Mapping inorder to be accepted by Pytorch Collator Functions
    Order of dict is defined only in OrderedDict, or dict in python3.7 and the
    Cpython implementation of python3.6
    """

    @property
    def data_dict(self) -> OrderedDict:
        return self._data_dict

    @data_dict.setter
    def data_dict(self, new_value: OrderedDict):
        if not isinstance(new_value, OrderedDict):
            raise TypeError(f"Expect OrderedDict, got {type(new_value)}")
        self._data_dict = new_value

    def __init__(self, data_dict: Union[OrderedDict, Iterable] = None):
        if data_dict is None:
            data_dict = OrderedDict()
        if not isinstance(data_dict, OrderedDict):
            data_dict = OrderedDict(data_dict)
        super().__init__()
        self._data_dict = data_dict

    @classmethod
    def build(cls, obj):
        if isinstance(obj, DatasetItem):
            return obj
        elif not isinstance(obj, OrderedDict):
            obj = OrderedDict(obj)
        return cls(obj)

    def get(self, k):
        return self.data_dict.get(k)

    def __getitem__(self, item):
        return self.data_dict[item]

    def __setitem__(self, key, value):
        self.data_dict[key] = value

    def set(self, key, value):
        self.__setitem__(key, value)
        return self

    def __len__(self):
        return len(self.data_dict)

    def keys(self):
        return self.data_dict.keys()

    def values(self):
        return self.data_dict.values()

    def items(self):
        return self.data_dict.items()

    def __contains__(self, item):
        return self.data_dict.__contains__(item)

    def __eq__(self, other):
        return self.data_dict.__eq__(other)

    def __ne__(self, other):
        return self.data_dict.__ne__(other)

    '''
    # ### It will break the assumptions in pytorch that the __iter__ of mapping uses its keys.
    def __iter__(self):
        """
        Unpacking Pattern using values instead of keys.
        For the compatibility of old codes that treat dataset items as tuples.
        Returns:
        """
        return iter(self.data_dict.items())
    '''

    def __iter__(self):
        return self.data_dict.__iter__()

    def __repr__(self):
        return f"{type(self).__name__} enclosing: {self.data_dict.__repr__()}"

    def _validate_type_order(self, type_order: np.ndarray):
        type_order = np.atleast_1d(type_order).ravel()
        assert len(self.keys()) == len(type_order), \
            f"length of type order mismatch. Expect {list(self.keys())}. Got {type_order}"

    def _re_order_helper(self, type_order):
        self._validate_type_order(type_order)
        for key in type_order:
            item = self.data_dict.pop(key)
            self.data_dict[key] = item

    def re_order(self, type_order, inplace_dict: bool = True):
        """

        Args:
            type_order (): If None, do no-op.
            inplace_dict (bool): if True, re-sort the OrderedDict. If False, create a new OrderedDict.
                Inplace op may be 40% faster.

        Returns:

        """
        if type_order is None:
            return self
        self._validate_type_order(type_order)
        if not inplace_dict:
            self._data_dict = OrderedDict([(key, self.data_dict[key]) for key in type_order])
        else:
            self._re_order_helper(type_order)
        return self


class DataItemUnpackByVal(object):
    def __init__(self, item: Union[DatasetItem, Dict]):
        if isinstance(item, dict):
            item = DatasetItem.build(item)
        assert isinstance(item, DatasetItem), f"Expect {DatasetItem.__name__}. Got {type(item)}"
        self._item = item

    def __getitem__(self, item):
        return self._item[item]

    def __setitem__(self, key, value):
        self._item[key] = value

    def __len__(self):
        return len(self._item)

    def keys(self):
        return self._item.keys()

    def values(self):
        return self._item.values()

    def items(self):
        return self._item.items()

    def __contains__(self, item):
        return self._item.__contains__(item)

    def __iter__(self):
        return iter(self._item.items())


class AbstractDataset(TorchDataset):

    def collate_not_required_fields(self, o_dict: OrderedDict):
        o_dict_new_ref_mutate = o_dict.copy()
        for k, v in o_dict.items():
            if k not in self.preserved_attributes:
                o_dict_new_ref_mutate.pop(k)
        # in-place, but offers the return
        return o_dict_new_ref_mutate

    @staticmethod
    def slice2array(index: slice, length: int):
        start, stop, step = index.start, index.stop, index.step
        step = common.default_not_none(step, 1)
        start = common.default_not_none(start, 0)
        stop = common.default_not_none(stop, length)
        assert step != 0, 'step is 0'
        return np.arange(start, stop, step)

    @property
    def flatten_output(self) -> bool:
        return self.__flatten_output

    @staticmethod
    def __validate_attribute(attributes: np.ndarray):
        attributes = np.atleast_1d(attributes)
        attr_set = set(attributes)
        assert len(attr_set) == len(attributes), f"attribute not unique"

    @flatten_output.setter
    def flatten_output(self, new_value):
        # assert before making changes
        AbstractDataset.__validate_attribute(new_value)
        self.__flatten_output = np.atleast_1d(new_value)

    @property
    def preserved_attributes(self) -> np.ndarray:
        return self.__preserved_attributes

    @preserved_attributes.setter
    def preserved_attributes(self, new_value):
        self.__preserved_attributes = new_value

    def __init__(self, flatten_output: bool, preserved_attributes,  truncate_size: float = np.inf):
        self.__truncate_size = truncate_size
        self.__flatten_output: bool = flatten_output
        # use ndarray instead of set because set is unordered
        # assert before making changes
        AbstractDataset.__validate_attribute(preserved_attributes)
        self.__preserved_attributes: np.ndarray = np.atleast_1d(preserved_attributes)

    @property
    def truncate_size(self) -> float:
        return self.__truncate_size

    @truncate_size.setter
    def truncate_size(self, new_value):
        self.__truncate_size = new_value

    @abstractmethod
    def length_helper(self):
        return NotImplemented

    def __len__(self) -> int:
        limit = min(self.length_helper(), self.truncate_size)
        if limit is np.inf:
            limit = np.iinfo(np.int64).max
        return limit

    @abstractmethod
    def get_item_helper(self, index) -> OrderedDict:
        """
        Behavior of how to read items by given indices.
        Args:
            index ():

        Returns:

        """
        ...

    def __getitem__(self, index):

        items = self.get_item_helper(index)
        if not isinstance(items, Iterable):
            items = (items, )
        if not isinstance(items, OrderedDict) and not isinstance(items, DatasetItem):
            items = OrderedDict([(key, value) for (key, value) in zip(self.preserved_attributes, items)])
            items = DatasetItem(items)

        # Double check the order. It is prune to mistakes transferring an ordered sequence to a dict.
        # As a result I believe it is better to enforce all subclass using the dict form as the output.
        # Alternatively, there should be an interface to convert the non-dict based dataset into a dict-based one.
        # assert isinstance(items, OrderedDict), f"Expect Dict form from get_item_helper, instead get " \
        #    f"{type(items)}."

        # equal unique elements
        validate_attribute = len(set(self.preserved_attributes) - set(items.keys())) == 0

        # equal length
        eq_length = len(self.preserved_attributes) <= len(items)
        # preserved attributes should at least be a subset of all types.
        assert validate_attribute and eq_length, f"Attributes Mismatch. {self.preserved_attributes}" \
            f" vs. {list(items.keys())}"

        output = items

        if self.flatten_output:
            output = tuple(output.values())
            # if there is only one single value, unpack the tuple.
            if len(output) == 1:
                output = output[0]

        return output


class AdaptorSet(AbstractDataset):
    """
        Convert normal dataset to Dict-based
    """

    def __init__(self, dataset: TorchDataset, flatten_output: bool = False, preserved_attributes=None):
        if preserved_attributes is None:
            preserved_attributes = []
        super().__init__(flatten_output=flatten_output, preserved_attributes=preserved_attributes)
        self._dataset = dataset

    def length_helper(self):
        return len(self._dataset)

    def get_item_helper(self, index) -> DatasetItem:
        output = self._dataset[index]
        num_item = len(output)
        if num_item == len(self.preserved_attributes):
            types_iterable = self.preserved_attributes
        else:
            types_iterable = range(num_item)
        output_dict = OrderedDict([(x, item) for (x, item) in zip(types_iterable, output)])
        output = DatasetItem.build(output_dict)
        return output


class FileFolder(AbstractDataset):
    """
    A plain folder/file.ext structure. Information encoded in the filename.
    Not vectorized. The batchification is handled by DataLoaders.
    """
    KEY_FILENAMES: str = 'filename'
    KEY_INDEX: str = 'index'
    KEY_IMG: str = 'img'
    KEY_LABEL: str = 'label'

    def __init__(self,
                 root: str,
                 ext: str,
                 class_list: Sequence[str],
                 flatten_output: bool,
                 truncate_size: float = np.inf):

        attributes = np.asarray([FileFolder.KEY_IMG,
                                 FileFolder.KEY_LABEL,
                                 FileFolder.KEY_FILENAMES,
                                 FileFolder.KEY_INDEX])
        super().__init__(flatten_output, attributes, truncate_size=truncate_size)
        self.__root = root
        self.__file_list: Sequence[str] = glob.glob(os.path.join(root, f"*.{ext}"))
        self.__class_list = class_list

    @property
    def class_list(self):
        return self.__class_list

    @property
    def root(self):
        return self.__root

    @property
    def file_list(self):
        return self.__file_list

    def length_helper(self):
        return len(self.__file_list)

    def label(self, f_name):
        basename = os.path.basename(f_name)
        if self.class_list is None:
            class_id_list = [0]
        else:
            class_id_list = \
                [idx for idx in range(len(self.class_list)) if
                 re.search(self.class_list[idx], basename, re.IGNORECASE)
                 ]

        if len(class_id_list) != 1:
            logger.warning(f"Class_ID matching warning. Len:{class_id_list}. Name:{basename}")
        class_id = class_id_list[0]
        return class_id

    def get_item_helper(self, index) -> DatasetItem:
        f_name: str = self.file_list[index]
        pil_img = Image.open(f_name)
        label = self.label(f_name)
        data = (pil_img, label, f_name, index)
        keys = (FileFolder.KEY_IMG, FileFolder.KEY_LABEL, FileFolder.KEY_FILENAMES, FileFolder.KEY_INDEX)
        item = OrderedDict([(k, v) for k, v in zip(keys, data)])
        item = DatasetItem.build(item)
        return item


class H5SetBasic(AbstractDataset):
    KEY_FILENAMES: str = 'filename'
    KEY_INDEX: str = 'index'

    @property
    def filename(self):
        return self._filename

    @property
    def types(self):
        return self._types

    @classmethod
    def build_from_db(cls, database, phase, group_level: int = 0):
        table_file_name = database.generate_table_name(phase)[0]
        types = database.types
        return cls(table_file_name, types, group_level=group_level)

    def __init__(self, filename: str, types: Sequence[str] = None, group_level: int = 0, flatten_output: bool = True):
        self._filename = filename
        self._group_level = group_level
        with tables.open_file(self.filename, 'r') as db:
            if hasattr(db.root, 'types'):
                self._types = [x.decode('utf-8') for x in db.root.types[:]]
            else:
                self._types = types
        assert self._types is not None, 'Type is not defined in pytable and signature'
        self._types = np.atleast_1d(self._types)

        attributes = np.append(self._types, [H5SetBasic.KEY_FILENAMES, H5SetBasic.KEY_INDEX])
        super().__init__(flatten_output=flatten_output, preserved_attributes=attributes)
        with tables.open_file(self.filename, 'r') as db:
            if hasattr(db.root, 'class_sizes'):
                self.class_sizes = db.root.class_sizes[:]
            else:
                self.class_sizes = None

    def length_helper(self):
        with tables.open_file(self.filename, 'r') as db:
            return getattr(db.root, self.types[0]).shape[0]

    @staticmethod
    def dim_recovered_data(img, return_tensor: bool = True):
        if not (len(img.shape) < 2 and img.shape[0] == 1):
            return img
        if return_tensor:
            img = torch.from_numpy(img)
        else:  # add leading singleton dim
            img = img[None, ]
        return img

    @staticmethod
    def _flatten_group(grouped_data: Union[List, Dict]):
        length = len(grouped_data)
        if length == 0:
            result = grouped_data
        elif length == 1:
            result = grouped_data[0]
        else:
            result = [x[0] for x in grouped_data]
        return result

    @staticmethod
    def __parse_filename(filenames_in):
        # breakpoint()
        filenames = np.asarray(filenames_in)
        # scalar form (single string)--> ndim = 0
        is_scalar_form = filenames.ndim == 0
        if not is_scalar_form:
            filenames = [x.decode('utf-8') for x in filenames]
        elif filenames.size >= 1:
            # in this case, p.asarray(filenames_in) is a 0-d array.
            element = filenames.ravel()[0]
            if hasattr(element, 'decode'):
                filenames = element.decode('utf-8')
            else:
                logger.warning(f"No decode attribute:{type(element)}_{element}")
                filenames = element
        else:
            raise ValueError(f"Unsupported type of {filenames}")
        return filenames

    def get_item_helper(self, index) -> DatasetItem:
        with tables.open_file(self.filename, 'r') as db:
            # Use List form first. Convert to dict after the "flatten" procedure of grouped dataset
            # to simplify the logic in _flatten_group
            data_list: List = [getattr(db.root, x) for x in self.types]
            filename_list = getattr(db.root, 'filename')
            filename = filename_list[index, ]  # filenames is unaffected
            data_out = [data_array[index, ] for data_array in data_list]
            if self._group_level == 1:
                filename = H5SetBasic._flatten_group(filename)
                data_out = H5SetBasic._flatten_group(data_out)

        filename = self.__parse_filename(filename)

        if isinstance(index, slice):
            index_out = type(self).slice2array(index, len(self))
        else:
            index_out = np.asarray([index]).ravel()
        # image
        data_out[0] = H5SetBasic.dim_recovered_data(data_out[0])
        data_out_dict = OrderedDict([(x, data_array) for (x, data_array) in zip(self.types, data_out)])
        data_out_dict = DatasetItem.build(data_out_dict)
        data_out_dict[H5SetBasic.KEY_FILENAMES] = filename
        data_out_dict[H5SetBasic.KEY_INDEX] = index_out

        return data_out_dict


class H5SetTransform(AbstractDataset):
    KEY_IMG_ORIGIN: str = 'img_origin'

    @property
    def img_transform(self):
        return self._img_transform

    def __init__(self, filename: str, types: Sequence[str] = None,
                 flatten_output: bool = False,
                 img_transform: Callable = None,
                 img_key: str = 'img',):
        # nothing special here, just internalizing the constructor parameters
        self._base_dataset = H5SetBasic(filename, types,)
        self._types = self._base_dataset.types
        self.preserved_attributes = np.append(self._types,
                                              [H5SetTransform.KEY_IMG_ORIGIN,
                                               H5SetBasic.KEY_FILENAMES,
                                               H5SetBasic.KEY_INDEX])
        super().__init__(flatten_output=flatten_output, preserved_attributes=self.preserved_attributes)
        self._img_transform = img_transform
        assert img_key in self._types
        self._img_key = img_key

    def get_item_helper(self, index) -> DatasetItem:
        # opening should be done in __init__ but seems to be
        # an issue with multi-threading so doing here. need to do it every time, otherwise hdf5 crashes
        # img, label, filename, *rest = super().__getitem__(index)
        data_dict: OrderedDict = self._base_dataset.get_item_helper(index).data_dict
        img = data_dict[self._img_key]
        img_new = img
        # if row vector (dimension reduced or not)
        # otherwise do the transformation in prior of the collate function.
        if self.img_transform is not None:
            img_new = self.img_transform(img)

        data_dict[self._img_key] = img_new
        data_dict[H5SetTransform.KEY_IMG_ORIGIN] = img
        # move filenames and index to the end (after img), retain the order
        data_dict.move_to_end(H5SetBasic.KEY_FILENAMES)
        data_dict.move_to_end(H5SetBasic.KEY_INDEX)
        output = DatasetItem.build(data_dict)
        return output

    def length_helper(self):
        return self._base_dataset.length_helper()


class MultiSet(TorchDataset):
    def __init__(self, *dataset):
        self._associated_db_list = dataset
        type(self).validate_len(*dataset)

    '''
        Raise Error if length not identical
        h5datasets -->H5dataset
    '''

    @staticmethod
    def validate_len(*h5datasets):
        assert len(h5datasets) > 0
        length = [len(dataset) for dataset in h5datasets]
        assert (length[1:] == length[:-1])

    def __len__(self):
        return len(self._associated_db_list[0])

    def __getitem__(self, index):
        return tuple(self._associated_db_list[i][index] for i in range(self.num_database()))

    def num_database(self):
        return len(self._associated_db_list)

    @property
    def class_sizes(self):
        return self._associated_db_list[0].class_sizes


class MemSet(TorchDataset):

    def __init__(self, data: np.ndarray, img_transform=None):
        self.data = data
        self.img_transform = img_transform

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        img = self.data[index, ]
        if self.img_transform is not None:
            img = self.img_transform(img)
        return img


class SlideSet(AbstractDataset):
    """
    Floor if not divisible
    """
    DEFAULT_IMG_KEY: str = 'img'
    DEFAULT_COORD_KEY: str = 'coord'
    DEFAULT_INDEX_KEY: str = 'index'

    def __init__(self,
                 file_name: str,
                 patch_size: int,
                 level: int = 0,
                 stride: int = None,
                 by_row: bool = True,
                 img_transform: Callable = None,
                 flatten_output: bool = False,
                 truncate_size: float = np.inf):
        """

        Args:
            file_name: Filename (with path) of the WSI
            patch_size: Size of the tile. For convenience only support square tiles.
            level: Which Level index in the pyramid, corresponding openslide_handle.level_dimensions.
            stride: Stride moving along the axis. If None (default), then extract non-overlapping patches.
            by_row: Moving along row or column
            img_transform: Image Transforms. Callable. Post-processing after extracting a single tile. (e.g. rotation).
                Default is None --> meaning no transforms.
            flatten_output: config for "AbstractDataset". If True, flatten the output from named Dict to plain Tuple.
            truncate_size: whether truncate the dataset. Set to np.inf --> use full dataset.
        """
        img_key: str = SlideSet.DEFAULT_IMG_KEY
        coord_key: str = SlideSet.DEFAULT_COORD_KEY
        index_key: str = SlideSet.DEFAULT_INDEX_KEY

        preserved_attributes = np.asarray([img_key,
                                           coord_key,
                                           index_key])
        super().__init__(flatten_output=flatten_output,
                         preserved_attributes=preserved_attributes,
                         truncate_size=truncate_size)
        self.file_name = file_name
        self.patch_size = patch_size
        self.level = level
        self.by_row = by_row
        self.img_transform = img_transform
        # default stride == patch_size (no overlapping)
        self.stride = stride if stride is not None else self.patch_size

    @staticmethod
    def shape_helper(img_size: int, patch_size: int, stride: int):
        """
        Shape of patch grid after performing the sliding-window operation. e.g. for a 500x500 Image, 250x250 tile size,
        and 250 stride (non_overlapping), the grid of output patch should be 2x2
        Args:
            img_size:
            patch_size:
            stride:

        Returns:

        """
        return (img_size - patch_size) // stride + 1

    def len_width(self):
        """
        Width of the patch grid. Help calculate the length of the entire patch sequence.
        Returns:

        """
        width, _ = self.osh.level_dimensions[self.level]
        # width // self.patch_size --> if stride == patch_size
        return SlideSet.shape_helper(width, self.patch_size, self.stride)

    def len_height(self):
        """
        Height of the patch grid. Help calculate the length of the entire patch sequence.
        Returns:

        """
        _, height = self.osh.level_dimensions[self.level]
        # height // self.patch_size
        return SlideSet.shape_helper(height, self.patch_size, self.stride)

    @property
    def patch_map_shape(self):
        """
        H, W of patch map/Grid (non-overlapping)
        Returns:

        """
        return self.len_height(), self.len_width()

    def length_helper(self):
        """
        Length of Patch Sequence --> Area of patch grid.
        Returns:

        """
        return self.len_width() * self.len_height()

    @staticmethod
    def segment(osh,
                patch_size: int, stride: int,
                level: int = 0,
                by_row: bool = True):
        """
        How many patches in a row or column
        Use flooring instead of ceil -- cut off the remaining if smaller than patch_size
        consistent to fold and stride trick - so that it align with the mask image
        Args:
            osh:
            patch_size:
            stride:
            level:
            by_row:

        Returns:

        """
        width, height = osh.level_dimensions[level]
        if by_row:
            length = width
        else:
            length = height
        # step_num = length // patch_size  # np.ceil(length / patch_size).astype(np.int)
        step_num = SlideSet.shape_helper(length, patch_size, stride=stride)
        return step_num

    @staticmethod
    def idx2loc(chunk_step_num: int,
                # osh: OpenSlide,
                index: int,
                stride: int,
                by_row=True) -> Tuple[int, int]:
        assert chunk_step_num > 0, f" too small compared to patch_size"

        # horiz = index % chunk_step_num * patch_size
        # vert = index // chunk_step_num * patch_size

        chunk_remainder: int = (index % chunk_step_num) * stride
        num_chunk: int = index // chunk_step_num * stride
        output = (chunk_remainder, num_chunk) if by_row else (num_chunk, chunk_remainder)
        return output

    @property
    def osh(self):
        return OpenSlide(self.file_name)

    def thumbnail(self, level):
        osh = self.osh
        thumb = osh.get_thumbnail(osh.level_dimensions[level])
        return np.asarray(thumb)

    def pil_img_from_index(self, index):
        osh = self.osh
        chunk_step_num = SlideSet.segment(osh, self.patch_size,
                                          stride=self.stride,
                                          level=self.level,
                                          by_row=self.by_row)
        c, r = SlideSet.idx2loc(chunk_step_num,
                                index=index,
                                stride=self.stride,
                                by_row=self.by_row)

        pil_region = osh.read_region(location=(c, r), level=self.level, size=(self.patch_size, self.patch_size))
        # read_region returns RGBA
        pil_region = pil_region.convert('RGB')
        return pil_region, (c, r)

    def get_item_helper(self, index):
        """
        Implementation of extractor in the interface __getitem__
        Args:
            index:

        Returns:

        """
        o_dict = OrderedDict()

        img, coordinate_rc = self.pil_img_from_index(index)

        if self.img_transform is not None:
            img = self.img_transform(img)

        o_dict[SlideSet.DEFAULT_IMG_KEY] = img
        o_dict[SlideSet.DEFAULT_COORD_KEY] = coordinate_rc
        o_dict[SlideSet.DEFAULT_INDEX_KEY] = index
        item = DatasetItem(o_dict)
        return item


class ClassSpecifiedFolder(AbstractDataset):
    """
    By Default, transforms to Pillow
    """
    KEY_IMG: str = 'img'
    KEY_LABEL: str = 'label'
    KEY_FILENAME: str = 'filename'
    KEY_INDEX: str = 'index'
    KEY_IMG_ORIGIN: str = 'img_origin'

    @staticmethod
    def _find_classes(directory):
        """
        Finds the class folders in a dataset.

        Args:
            directory (string): Root directory path.

        Returns:
            tuple: (classes, class_to_idx) where classes are relative to (dir), and class_to_idx is a dictionary.

        Ensures:
            No class is a subdirectory of another.
        """
        classes = [d.name for d in os.scandir(directory) if d.is_dir()]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx

    def __init__(self,
                 directory,
                 class_to_idx,
                 is_valid_file=None,
                 transforms: Callable = None,
                 flatten_output: bool = False,
                 truncate_size: float = np.inf,
                 roi_name_parser: Callable = None,
                 loader: Callable = default_loader,
                 file_extension=IMG_EXTENSIONS,
                 ):

        preserved_attributes = np.asarray([ClassSpecifiedFolder.KEY_IMG,
                                           ClassSpecifiedFolder.KEY_LABEL,
                                           ClassSpecifiedFolder.KEY_IMG_ORIGIN,
                                           ClassSpecifiedFolder.KEY_FILENAME,
                                           ClassSpecifiedFolder.KEY_INDEX])
        super().__init__(flatten_output=flatten_output, preserved_attributes=preserved_attributes,
                         truncate_size=truncate_size)

        extensions = file_extension if is_valid_file is None else None
        if class_to_idx is None:
            classes, class_to_idx = self._find_classes(directory)
        self.__class_to_idx = class_to_idx
        self.__samples = make_dataset_default(directory, class_to_idx, extensions, is_valid_file)
        self.roi_name_parser = roi_name_parser\
            if roi_name_parser is not None else\
            ClassSpecifiedFolder._default_roi_name_parser
        self.__transforms = transforms
        self._loader = loader

    @property
    def transforms(self):
        return self.__transforms

    @transforms.setter
    def transforms(self, x):
        assert x is None or isinstance(x, Callable)
        self.__transforms = x

    @property
    def samples(self):
        return self.__samples

    def length_helper(self):
        return len(self.samples)

    # noinspection PyUnusedLocal
    # @staticmethod
    # def _default_roi_name_parser(img_filename, *args, **kwargs):
    #     delimiter = '_'
    #     basename = os.path.basename(img_filename)
    #     file_text, ext = os.path.splitext(basename)
    #     roi_components = file_text.split('_')[:-2]
    #     roi_file_text = delimiter.join(roi_components)
    #     roi_filename = f"{roi_file_text}{ext}"
    #     return roi_filename

    @staticmethod
    def _default_roi_name_parser(img_filename, *args, **kwargs):
        return img_filename

    def get_item_helper(self, index) -> Union[OrderedDict, DatasetItem]:
        sample = self.samples[index]
        img_filename, class_id = sample
        roi_filename = self.roi_name_parser(img_filename)
        o_dict = OrderedDict()

        img_pil = self._loader(img_filename)
        # origin is not affected by transformation, so it must be tensor or ndarray
        img_pil_origin = np.asarray(img_pil)

        if self.transforms is not None:
            img_pil = self.transforms(img_pil)

        o_dict[ClassSpecifiedFolder.KEY_IMG] = img_pil
        o_dict[ClassSpecifiedFolder.KEY_LABEL] = class_id
        o_dict[ClassSpecifiedFolder.KEY_IMG_ORIGIN] = img_pil_origin
        o_dict[ClassSpecifiedFolder.KEY_FILENAME] = roi_filename
        o_dict[ClassSpecifiedFolder.KEY_INDEX] = index
        item = DatasetItem(o_dict)
        return item

    @LazyProperty
    def class_sizes(self):
        class_list = np.asarray([x[1] for x in self.samples])
        class_id_unique_sorted, count = np.unique(class_list, return_counts=True)
        assert len(set(self.__class_to_idx.values()) - set(class_id_unique_sorted)) == 0
        return count

    @staticmethod
    def imageio_loader(fname):
        try:
            return imageio.imread(fname)
        except:
            print(fname)
            breakpoint()

    @property
    def class_to_idx(self):
        return self.__class_to_idx


class SimpleFileListSet(TorchDataset):
    KEY_IMG: str = 'img'
    KEY_IMG_ORIGIN: str = 'origin'
    KEY_FILENAME: str = 'filename'
    KEY_INDEX: str = 'index'

    def __init__(self, file_list: Sequence[str], loader: Callable = default_loader,
                 img_transforms: Callable = None):
        """
        Retain the original order.
        No Labels. Only to pass a list of files into the memory
        Args:
            file_list:
        """
        self.file_list = file_list
        self.loader = loader
        self.img_transforms = img_transforms

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        o_dict = OrderedDict()
        img_filename = self.file_list[index]
        img_pil = self.loader(img_filename)
        # origin is not affected by transformation, so it must be tensor or ndarray

        if self.img_transforms is not None:
            img_pil = self.img_transforms(img_pil)

        o_dict[SimpleFileListSet.KEY_IMG] = img_pil
        o_dict[SimpleFileListSet.KEY_FILENAME] = img_filename
        o_dict[SimpleFileListSet.KEY_INDEX] = index
        item = DatasetItem(o_dict)
        return item
