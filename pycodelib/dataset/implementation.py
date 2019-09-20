import tables
import numpy as np
import torch
from torch.utils.data import Dataset as TorchDataset
from typing import Sequence, Callable
from pycodelib import common
from openslide import OpenSlide
import logging
logging.basicConfig(level=logging.DEBUG)
# from abc import ABC, abstractmethod
# -todo pytable-based base class


class H5SetBasic(TorchDataset):

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

    def __init__(self, filename: str, types: Sequence[str] = None):
        self._filename = filename
        with tables.open_file(self.filename, 'r') as db:
            if hasattr(db.root, 'types'):
                self._types = [x.decode('utf-8') for x in db.root.types[:]]
            else:
                self._types = types
        assert self._types is not None, 'Type is not defined in pytable and signature'
        with tables.open_file(self.filename, 'r') as db:
            if hasattr(db.root, 'class_sizes'):
                self.class_sizes = db.root.class_sizes[:]
            else:
                self.class_sizes = None

    def __len__(self):
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

    def __getitem__(self, index):
        with tables.open_file(self.filename, 'r') as db:
            img_list = getattr(db.root, self.types[0])
            label_list = getattr(db.root, self.types[1])
            filename_list = getattr(db.root, 'filename')
            img = img_list[index, ]
            label = label_list[index, ]
            filenames = filename_list[index, ]
        if isinstance(index, slice):
            filenames = [x.decode('utf-8') for x in filenames]
        else:
            filenames = filenames.decode('utf-8')
        if isinstance(index, slice):
            index_out = type(self).slice2array(index, len(self))
        else:
            index_out = np.asarray(index)
        img = type(self).dim_recovered_data(img)
        return img, label, filenames, index_out


class H5SetTransform(H5SetBasic):

    @property
    def img_transform(self):
        return self._img_transform

    def __init__(self, filename: str, types: Sequence[str] = None, img_transform: Callable = None):
        # nothing special here, just internalizing the constructor parameters
        super().__init__(filename, types)
        self._img_transform = img_transform

    def __getitem__(self, index):
        # opening should be done in __init__ but seems to be
        # an issue with multi-threading so doing here. need to do it every time, otherwise hdf5 crashes
        img, label, filename, *rest = super().__getitem__(index)
        img_new = img
        index_out = rest[-1]
        # if row vector (dimension reduced or not)
        # otherwise do the transformation in prior of the collate function.
        if self.img_transform is not None:
            img_new = self.img_transform(img)
        return img_new, label, img, filename, index_out


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
