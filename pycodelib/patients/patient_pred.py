from . import PandasRecord
from .patient_gt import PatientSlideCollection
from typing import Sequence, List, Dict
import numpy as np
import pandas as pd


class PatientPred(PandasRecord):
    NAME_SCORE: str = "score"
    NAME_PRED: str = "prediction"

    def __init__(self,
                 patient_ground_truth: PatientSlideCollection,
                 class_list: Sequence[str],
                 ):
        super().__init__()
        self._patient_ground_truth = patient_ground_truth
        self._class_list: np.ndarray = np.asarray(class_list)
        self.build_df(PatientPred.NAME_SCORE, self._class_list)
        self.build_df(PatientPred.NAME_PRED, self._class_list)

    @property
    def patient_info(self):
        return self._patient_ground_truth

    def load_data(self, table_name, data_list: Sequence, filenames: Sequence[str], flush: bool):
        PatientSlideCollection.load_data_by_patient(patient_src_record=self.patient_info,
                                                    target_data_frame=self.get_df(table_name),
                                                    data_list=data_list,
                                                    filenames=filenames,
                                                    flush=flush
                                                    )

    def load_score(self, scores_all_category: Sequence[Sequence[float]], filenames: Sequence[str], flush: bool):
        self.load_data(PatientPred.NAME_SCORE, scores_all_category, filenames, flush)

    def entry(self, class_value: str) -> Dict[str, int]:
        assert class_value in self.class_list, f"Undefined Class{class_value} in {self.class_list}"
        entry: Dict[str, int] = {c: 1 if c == class_value else 0 for c in self.class_list}
        return entry

    def load_prediction(self, pred_class_names: Sequence[str], filenames: Sequence[str], flush: bool):
        # todo entry from partition
        entry_list: List[Dict[str, int]] = [self.entry(pred) for pred in pred_class_names]
        entry_array = [np.asarray(list(entry.values())) for entry in entry_list]
        self.load_data(PatientPred.NAME_PRED, entry_array, filenames, flush)

    def get_ground_truth(self, patient_ids):
        raise NotImplementedError


class CascadedPred(PatientPred):

    @property
    def partition(self) -> Sequence[Sequence[int]]:
        return self._partition

    def __init__(self,
                 patient_ground_truth: PatientSlideCollection,
                 class_list: Sequence,
                 partition: Sequence[Sequence[int]],
                 ):
        super().__init__(patient_ground_truth, class_list)
        self. _partition = partition
        assert len(self._partition) == len(self.class_list), f"Group # mismatches the class size " \
            f"{len(self._partition) != len(self.class_list)}"

    def class_mapping(self):
        # todo
        ...

    def get_ground_truth(self, patient_ids: List):
        if np.isscalar(patient_ids):
            # otherwise the sub_table collapses to series and column_sum is reduced to scalar
            patient_ids = [patient_ids]
        sub_table: pd.DataFrame = self.patient_info.patient_ground_truth.loc[patient_ids]
        # gt class --> current class grouping. At least binary (occurrance). Better to sum the count.
        final_counts: pd.DataFrame = pd.DataFrame()
        for idx, group_idx in enumerate(self._partition):
            group_column: np.ndarray = self.patient_info.class_list[group_idx]
            column_sum: pd.Series = sub_table[group_column].sum(axis=1).rename(self.class_list[idx])
            final_counts = pd.concat([final_counts, column_sum], axis=1, sort=False)
        return final_counts
