import tables
import torch
from torch.utils.data import Dataset as TorchDataset
# -todo pytable-based base class


class Dataset(TorchDataset):
    @property
    def filename(self):
        return self._filename

    @property
    def types(self):
        return self._types

    @property
    def img_transform(self):
        return self._img_transform

    def __init__(self, filename, img_transform=None):
        # nothing special here, just internalizing the constructor parameters
        self._filename: str = filename
        self._img_transform = img_transform
        with tables.open_file(self.filename, 'r') as db:
            self._types = [x.decode('utf-8') for x in db.root.types[:]]

        with tables.open_file(self.filename, 'r') as db:
            if hasattr(db.root, 'class_sizes'):
                self.class_sizes = db.root.class_sizes[:]
            else:
                self.class_sizes = None

        self.img = None
        self.label = None

    def __getitem__(self, index):
        # opening should be done in __init__ but seems to be
        # an issue with multi-threading so doing here. need to do it every time, otherwise hdf5 crashes

        with tables.open_file(self.filename, 'r') as db:
            self.img = getattr(db.root, self.types[0])
            self.label = getattr(db.root, self.types[1])

            # get the requested image and mask from the pytable
            img = self.img[index, ]
            label = self.label[index]
            filename = db.root.filename[index]

        img_new = img

        if len(img.shape) == 2 and img.shape[0] == 1:
            # img_new = np.expand_dims(img, axis=0)
            img_new = torch.from_numpy(img_new)
        if self.img_transform is not None:
            img_new = self.img_transform(img)
        return img_new, label, img, filename.decode('utf-8'), index

    def __len__(self):
        with tables.open_file(self.filename, 'r') as db:
            return getattr(db.root, self.types[0]).shape[0]


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
