import tables
import torch
from torch.utils.data import Dataset as TorchDataset
from typing import Sequence, Callable
from abc import ABC, abstractmethod
# -todo pytable-based base class


class H5SetBasic(ABC, TorchDataset):
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

            with tables.open_file(self.filename, 'r') as db:
                self.img_list = getattr(db.root, self.types[0])
                self.label_list = getattr(db.root, self.types[1])
                self.filename_list = getattr(db.root, 'filename')

    def __len__(self):
        with tables.open_file(self.filename, 'r') as db:
            return getattr(db.root, self.types[0]).shape[0]

    @abstractmethod
    def __getitem__(self, index):
        img = self.img_list[index, ]
        label = self.label_list[index, ]
        filename = self.filename_list.filename[index]
        return img, label, filename, index


class H5SetTransform(H5SetBasic):

    @property
    def img_transform(self):
        return self._img_transform

    def __init__(self, filename: str, img_transform: Callable = None, types:Sequence[str] = None):
        # nothing special here, just internalizing the constructor parameters
        super().__init__(filename, types)
        self._img_transform = img_transform

    def __getitem__(self, index):
        # opening should be done in __init__ but seems to be
        # an issue with multi-threading so doing here. need to do it every time, otherwise hdf5 crashes
        img, label, filename, index = super().__getitem__(index)
        img_new = img

        if len(img.shape) == 2 and img.shape[0] == 1:
            # img_new = np.expand_dims(img_list, axis=0)
            img_new = torch.from_numpy(img_new)
        if self.img_transform is not None:
            img_new = self.img_transform(img)
        return img_new, label, img, filename.decode('utf-8'), index




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

    def __init__(self, ndarray, img_transform=None):
        self.data = ndarray
        self.img_transform = img_transform

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        img = self.data[index, ]
        if self.img_transform is not None:
            img = self.img_transform(img)
        return img


class DatasetArrayTest:
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index, ]


class DatasetPytableTest(TorchDataset):
    def __init__(self, fname, img_transform):
        self.fname = fname
        with tables.open_file(self.fname, 'r') as db:
            self.nitems = db.root.img.shape[0]
        self.img_transform = img_transform

    def __getitem__(self, index):
        with tables.open_file(self.fname, 'r') as db:
            img = db.root.img[index, ]
            label = db.root.label[index]
            img_new = self.img_transform(img)
        return img_new, label, img

    def __len__(self):
        return self.nitems
