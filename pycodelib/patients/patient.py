import os
import pandas as pd
import numpy as np
import re
from typing import Sequence, List, Dict, Any, Tuple, Set
from abc import ABC, abstractmethod

import logging

logging.basicConfig(level=logging.DEBUG)


class PatientCollection(ABC):

    def __init__(self):
        self.patient_ground_truth = None
        self.patient_prediction = None
        self._class_list = None

    @property
    @abstractmethod
    def class_list(self):
        ...

    @abstractmethod
    def load_prediction(self, **kwargs):
        ...

    @abstractmethod
    def load_ground_truth(self, **kwargs):
        ...

    @abstractmethod
    def build_patient_record(self, **kwargs):
        ...

    @abstractmethod
    def evaluate(self, **kwargs):
        ...


class SheetCollection(PatientCollection):
    _DEFAULT_CLASS = ['No Path', 'BCC', 'Situ', 'Invasive']
    _SLIDE_SEPARATOR: str = '_'

    @property
    def patient2slides(self):
        return self._patient2slides

    def __init__(self, file_list, sheet_name: str, class_list: Sequence[str] = None):
        super().__init__()
        if class_list is None:
            class_list = type(self)._DEFAULT_CLASS
        self._patient2slides: Dict[str, Set[str, ...]] = dict()
        self._class_list: Sequence[str] = class_list
        self.patient_sheet = None
        self._file_list = file_list
        self.load_ground_truth(sheet_name, class_list)

    def add_slides_to_patient(self, patient_id: str, slide_id: str):
        self.patient2slides[patient_id] = self.patient2slides.get(patient_id, set())
        self.patient2slides[patient_id].add(slide_id)

    def parse_class_name_short(self, roi_class: str) -> str:
        parsed = [class_name for class_name in self.class_list
                  if re.search(class_name, roi_class, re.IGNORECASE) is not None]
        logging.debug(f"{roi_class}.{self.class_list}|||{parsed}")
        assert len(parsed) == 1, f"ambiguity in class-parsing:{roi_class}"
        return parsed[0]

    def slide_name(self, file: str) -> Tuple[str, str]:
        basename = os.path.basename(file)
        name_components: List[str] = basename.split(type(self)._SLIDE_SEPARATOR)
        # nonzero returns a tuple of array
        index_match_array = np.asarray(
            [
                np.asarray(
                    [
                        re.search(class_name, component, re.IGNORECASE) is not None
                        for class_name in self.class_list
                    ]).any()
                for component in name_components
            ]
        ).nonzero()[0]
        assert index_match_array.size == 1 and index_match_array[0] < len(name_components) - 1, f"no match. Got:" \
            f"{index_match_array},{name_components},{basename}"
        slide_id = name_components[0].split()[0]
        index_match = index_match_array[0]
        return slide_id, name_components[index_match]

    def slide2patient(self, slide_id: str) -> str:
        patient_id_list = self.patient_sheet["PID"] \
            .where(self.patient_sheet['SLIDE_SUFFIX'] == slide_id) \
            .dropna() \
            .to_numpy()

        assert patient_id_list.shape[0] == 1, f"One Slide must be mapped to a unique patient." \
            f"{slide_id}{patient_id_list}"
        patient_id = patient_id_list[0]
        return patient_id

    def entry(self, class_value: str) -> Dict[str, int]:
        assert class_value in self.class_list, f"Undefined Class{class_value} in {self.class_list}"
        entry: Dict[str, int] = {c: 1 if c == class_value else 0 for c in self.class_list}
        return entry

    @staticmethod
    def insert_entry(dataframe: pd.DataFrame, patient_id, entry: Dict[str, Any]):
        entry_array = np.asarray(list(entry.values()))
        logging.debug(entry_array)
        try:
            existed = dataframe.loc[patient_id]
        except KeyError:
            existed = np.zeros_like(entry_array)
        entry_array += existed
        dataframe.loc[patient_id] = entry_array

    def _write_gt(self):
        for file in self.file_list:
            slide_id, class_name_full = self.slide_name(file)
            patient_id = self.slide2patient(slide_id)
            self.add_slides_to_patient(patient_id, os.path.basename(file))
            class_name_short = self.parse_class_name_short(class_name_full)
            assert class_name_short in self.class_list, f'Class not in the list:{class_name_short}'
            entry = self.entry(class_name_short)
            logging.debug(f"{entry}{patient_id}. Slide:{slide_id}")
            logging.debug(f"{self.patient_ground_truth}")
            type(self).insert_entry(self.patient_ground_truth, patient_id, entry)

    def build_patient_record(self, columns: Sequence[str] = None):
        self.patient_ground_truth = pd.DataFrame(columns=columns)
        self.patient_prediction = pd.DataFrame(columns=columns)

    def load_patient_sheet(self, sheet_name: str):
        patient_sheet = pd.read_excel(sheet_name)
        patient_sheet['SLIDE_SUFFIX'] = patient_sheet['REQNOYR'].map(str) + patient_sheet['REQNO']
        logging.debug(f'Before Drop {np.isnan(patient_sheet["PID"]).sum()}')
        patient_sheet.dropna(axis='index', subset=['PID'], how='any', inplace=True)
        logging.debug(f"Before Drop-Inplace{np.isnan(patient_sheet['PID']).sum()}")
        patient_sheet['PID'] = patient_sheet['PID'].astype(str)
        self.patient_sheet = patient_sheet

    # override
    def load_ground_truth(self, sheet_name: str, class_list: Sequence[str]):
        assert hasattr(self, 'patient2slides')
        self.load_patient_sheet(sheet_name)
        self.build_patient_record(columns=class_list)
        self._write_gt()

    # override
    @property
    def class_list(self):
        return self._class_list

    @property
    def file_list(self):
        return self._file_list

    def load_prediction(self, pred_class_names: Sequence[str], filenames: Sequence[str], flush: bool = True):
        assert self.patient_prediction.size is not None, f"Prediction not initialized"
        if flush:
            self.patient_prediction.drop(self.patient_prediction.index, inplace=True)
        file_basename_list: List[str] = [os.path.basename(f) for f in filenames]
        slide_id_list: List[str] = [self.slide_name(f)[0] for f in file_basename_list]
        entry_list: List[Dict[str, int]] = [self.entry(pred) for pred in pred_class_names]
        for (slide_id, entry) in zip(slide_id_list, entry_list):
            patient_id = self.slide2patient(slide_id)
            self.insert_entry(self.patient_prediction, patient_id, entry)

    # only perform evaluation. manual loading of prediction required
    def prediction(self, target_column: str) -> Tuple[pd.Series, ...]:
        assert self.patient_prediction.size > 0, f"Prediction not loaded"
        pred = self.patient_prediction.loc[self.patient_prediction.index, target_column]
        breakpoint()
        ground_truth = self.patient_ground_truth.loc[self.patient_prediction.index, target_column]
        assert ground_truth.index is pred.index, f'Index not Matched'
        return pred, ground_truth

    @abstractmethod
    def evaluate(self, **kwargs):
        ...


class SkinCollection(SheetCollection):

    def evaluate(self, **kwargs):
        ...
