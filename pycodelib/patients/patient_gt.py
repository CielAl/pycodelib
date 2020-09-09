import os
import pandas as pd
import numpy as np
import re
from typing import Sequence, List, Dict, Tuple, Set
from tqdm import tqdm
from pycodelib.patients.skeletal import PatientSlideCollection
from .skeletal import PandasRecord
import logging

logging.basicConfig(level=logging.WARNING)


class SheetCollection(PatientSlideCollection):
    _DEFAULT_CLASS = ['No Path', 'BCC', 'Situ', 'Invasive']
    SLIDE_SEPARATOR: str = '_'

    def __init__(self, file_list, sheet_name: str, class_list: Sequence[str] = None,
                 multi_parse_return_id: int = None):
        """

        Args:
            file_list:
            sheet_name:
            class_list:
            multi_parse_return_id: If multiple class str in name - which one to pick?
        """
        super().__init__()
        if class_list is None:
            class_list = type(self)._DEFAULT_CLASS
        self._class_list: np.ndarray = np.asarray(class_list)
        self._patient2slides_dict: Dict[str, Set[str, ...]] = dict()
        self.patient_sheet = None
        self._file_list = file_list
        self._multi_parse_return_id = multi_parse_return_id

        # this must be called after all fields being defined
        self.load_ground_truth(sheet_name, class_list)

    def patient2slides(self, patient_id):
        return self._patient2slides_dict.get(patient_id, None)

    def add_slides_to_patient(self, patient_id: str, slide_id: str):
        self._patient2slides_dict[patient_id] = self._patient2slides_dict.get(patient_id, set())
        self._patient2slides_dict[patient_id].add(slide_id)

    def parse_class_name_short(self, roi_class: str) -> str:
        parsed = [class_name for class_name in self.class_list
                  if re.search(class_name, roi_class, re.IGNORECASE) is not None]
        logging.debug(f"{roi_class}.{self.class_list}|||{parsed}")

        assert len(parsed) == 1 or self._multi_parse_return_id is not None, f"ambiguity in class-parsing:{roi_class}"
        return_ind = 0 if len(parsed) == 1 else self._multi_parse_return_id

        return parsed[return_ind]

    def slide_name(self, file: str) -> Tuple[str, str]:
        return type(self).slide_name_static(self, file)

    @staticmethod
    def slide_name_static(sheet_collection, file: str,
                          match_class: bool = True) -> Tuple[str, str]:
        basename = os.path.basename(file)
        name_components: List[str] = basename.split(type(sheet_collection).SLIDE_SEPARATOR)
        # nonzero returns a tuple of array
        index_match_array = np.asarray(
            [
                np.asarray(
                    [
                        re.search(class_name, component, re.IGNORECASE) is not None
                        for class_name in sheet_collection.class_list
                    ]).any()
                for component in name_components
            ]
        ).nonzero()[0]
        if match_class:
            assert index_match_array.size == 1 and index_match_array[0] < len(name_components) - 1, f"no match. Got:" \
                f"{index_match_array},{name_components},{basename}"
            index_match = index_match_array[0]
            matched_class_comp = name_components[index_match]
        else:
            matched_class_comp = None
        slide_id = name_components[0].split()[0]

        return slide_id, matched_class_comp

    def slide2patient(self, slide_id: str) -> str:
        return type(self).slide2patient_static(self, slide_id)

    @staticmethod
    def slide2patient_static(sheet_collection, slide_id):
        patient_id_list = sheet_collection.patient_sheet["PID"] \
            .where(sheet_collection.patient_sheet['SLIDE_SUFFIX'] == slide_id) \
            .dropna() \
            .to_numpy()

        assert patient_id_list.shape[0] == 1, f"One Slide must be mapped to a unique patient." \
            f"{slide_id}{patient_id_list}"
        patient_id = patient_id_list[0]
        return patient_id

    def _write_gt(self):
        for file in tqdm(self.file_list):
            slide_id, class_name_full = self.slide_name(file)
            patient_id = self.slide2patient(slide_id)
            self.add_slides_to_patient(patient_id, os.path.basename(file))
            class_name_short = self.parse_class_name_short(class_name_full)
            assert class_name_short in self.class_list, f'Class not in the list:{class_name_short}'
            entry = self.entry(class_name_short)
            # logging.debug(f"{entry}{patient_id}. Slide:{slide_id}")
            # logging.debug(f"{self.patient_ground_truth}")
            entry_array = np.asarray(list(entry.values()))
            PandasRecord.insert_data(self.patient_ground_truth, patient_id, entry_array)

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
        self.build_df('patient_ground_truth', columns=class_list)
        self._write_gt()

    # override
    @property
    def class_list(self):
        return self._class_list

    @property
    def file_list(self):
        return self._file_list

    def load_data(self, table_name: str, data_list: Sequence, filenames: Sequence[str], flush: bool):
        PatientSlideCollection.load_data_by_patient(self, self.get_df(table_name), data_list, filenames, flush)

    @staticmethod
    def key_to_row(sheet_col, filenames):
        file_basename_list: List[str] = [os.path.basename(f) for f in filenames]
        slide_id_list: List[str] = [SheetCollection.slide_name_static(sheet_col, f, match_class=False)[0]
                                    for f in file_basename_list]
        patient_id_list: List = [SheetCollection.slide2patient_static(sheet_col, slide_id)
                                 for slide_id in slide_id_list]
        return patient_id_list

    @staticmethod
    def load_data_by_patient(patient_src_record, target_data_frame: pd.DataFrame, data_list: Sequence,
                             filenames: Sequence[str], flush: bool):
        if flush:
            type(patient_src_record).flush_df(target_data_frame)
        patient_id_list: List = SheetCollection.key_to_row(patient_src_record, filenames)
        # print(patient_id_list)
        for (patient_id, data) in zip(patient_id_list, data_list):
            PandasRecord.insert_data(target_data_frame, patient_id, data)
"""
    # only perform evaluation. manual loading of prediction required
    def prediction(self, target_column: str) -> Tuple[pd.Series, ...]:
        assert self.patient_prediction.size > 0, f"Prediction not loaded"
        pred = self.patient_prediction.loc[self.patient_prediction.index, target_column]
        breakpoint()
        ground_truth = self.patient_ground_truth.loc[self.patient_prediction.index, target_column]
        assert ground_truth.index is pred.index, f'Index not Matched'
        return pred, ground_truth

    def evaluate(self, **kwargs):
        ...



        file_basename_list: List[str] = [os.path.basename(f) for f in filenames]
        slide_id_list: List[str] = [SheetCollection.slide_name_static(sheet_col, f)[0]
                                    for f in file_basename_list]
        patient_id_list: List = [SheetCollection.slide2patient_static(sheet_col, slide_id)
                                 for slide_id in slide_id_list]
"""
