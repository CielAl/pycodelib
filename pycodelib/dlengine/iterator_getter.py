from typing import Dict

from torch.utils.data import DataLoader

from pycodelib.dlengine.skeletal import IteratorBuilder


class H5DataGetter(IteratorBuilder):
    _datasets = None
    BATCH_SIZE: int = 32

    def __init__(self, filename, img_transform, mode):
        super().__init__(filename, img_transform=img_transform)
        self._train_mode = mode

    @classmethod
    def build(cls, filename_dict: Dict[bool, str], img_transform_dict: Dict[bool, object]):
        assert filename_dict.keys() == img_transform_dict.keys(), 'Keys disagree.'
        cls._datasets = {x: H5DataGetter(filename_dict[x], img_transform_dict[x], x) for x in filename_dict.keys()}

    def __len__(self):
        return super().__len__()

    @property
    def mode(self):
        return self._train_mode

    def __getitem__(self, item):
        result = super().__getitem__(item)
        return result  # + tuple([self.mode]) # performed by the on_sample

    @classmethod
    def get_iterator(cls, mode, shuffle, num_workers=6, drop_last=False,  pin_memory=True, batch_size=BATCH_SIZE):
        assert mode == cls._datasets[mode].mode, f"Mode mismatch. input:{mode}"
        return DataLoader(cls._datasets[mode], num_workers=num_workers, pin_memory=pin_memory,
                          batch_size=batch_size, shuffle=mode, drop_last=True)