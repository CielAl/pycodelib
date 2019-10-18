import tables
from abc import abstractmethod
import numpy as np
import torch
from torch.utils.data import Dataset as TorchDataset
from typing import Sequence, Callable, List, Dict, Union, Iterable
from pycodelib import common
from openslide import OpenSlide
from collections import OrderedDict
import platform
import sys
from collections.abc import Mapping
import logging
logger = logging.getLogger(__name__)
logger.setLevel(level=logging.DEBUG)
# from abc import ABC, abstractmethod


def validate_compatability():
    impl = platform.python_implementation()
    major = sys.version_info.major
    minor = sys.version_info.minor
    cpython_36 = impl == 'CPython' and ((major == 3 and minor >= 6) or major > 3)
    python37 = ((major == 3 and minor >= 7) or major > 3)
    assert cpython_36 or python37


validate_compatability()


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
        eq_length = len(self.preserved_attributes) == len(items)
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


class H5SetBasic(AbstractDataset):
    KEY_FILENAMES: str = 'filename'
    KEY_INDEX: str = 'index'
    @staticmethod
    def slice2array(index: slice, length: int):
        start, stop, step = index.start, index.stop, index.step
        step = common.default_not_none(step, 1)
        start = common.default_not_none(start, 0)
        stop = common.default_not_none(stop, length)
        assert step != 0, 'step is 0'
        return np.arange(start, stop, step)

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


class H5SetTransform(H5SetBasic):
    KEY_IMG_ORIGIN: str = 'img_origin'
    @property
    def img_transform(self):
        return self._img_transform

    def __init__(self, filename: str, types: Sequence[str] = None,
                 img_transform: Callable = None,
                 img_key: str = 'img'):
        # nothing special here, just internalizing the constructor parameters
        super().__init__(filename, types)
        self.preserved_attributes = np.append(self._types,
                                              [H5SetTransform.KEY_IMG_ORIGIN,
                                               H5SetBasic.KEY_FILENAMES,
                                               H5SetBasic.KEY_INDEX])
        self._img_transform = img_transform
        assert img_key in self._types
        self._img_key = img_key

    def get_item_helper(self, index) -> DatasetItem:
        # opening should be done in __init__ but seems to be
        # an issue with multi-threading so doing here. need to do it every time, otherwise hdf5 crashes
        # img, label, filename, *rest = super().__getitem__(index)
        data_dict: OrderedDict = super().__getitem__(index)
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


class SlideSet(TorchDataset):
    def __init__(self, file_name, patch_size, level=0, by_row=True):
        self.file_name = file_name
        self.patch_size = patch_size
        self.level = level
        self.by_row = by_row

    @staticmethod
    def segment(osh, patch_size, level=0, by_row=True):
        width, height = osh.level_dimensions[level]
        if by_row:
            length = width
        else:
            length = height
        step_num = np.ceil(length / patch_size).astype(np.int)
        return step_num

    @staticmethod
    def idx2loc(osh, index, patch_size, level=0, by_row=True):
        row_step_num = SlideSet.segment(osh, patch_size, level=level, by_row=by_row)
        vert = index // row_step_num * patch_size
        horiz = index % row_step_num * patch_size
        return horiz, vert

    def __getitem__(self, index):
        osh = OpenSlide(self.file_name)
        c, r = SlideSet.idx2loc(osh, index, self.patch_size, level=self.level, by_row=self.by_row)
        pil_region = osh.read_region(location=(c, r), level=self.level, size=(self.patch_size, self.patch_size))
        return np.array(pil_region)

    def __len__(self):
        osh = OpenSlide(self.file_name)
        step_num = SlideSet.segment(osh,
                                    self.patch_size,
                                    self.level,
                                    by_row=self.by_row)
        step_num_dual = SlideSet.segment(osh,
                                         self.patch_size,
                                         self.level,
                                         by_row=not self.by_row)
        return step_num * step_num_dual
