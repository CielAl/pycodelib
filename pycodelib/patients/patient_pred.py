from . import PandasRecord
from .patient_gt import PatientSlideCollection
from typing import Sequence, List, Dict
import numpy as np


class PatientPred(PandasRecord):

    def __init__(self, patient_ground_truth: PatientSlideCollection, class_list):
        super().__init__()
        self._patient_ground_truth = patient_ground_truth
        self._class_list = class_list
        self.build_df('score', self._class_list)
        self.build_df('prediction', self._class_list)

    @property
    def patient_ground_truth(self):
        return self._patient_ground_truth

    def load_data(self, table_name, data_list: Sequence, filenames: Sequence[str], flush: bool):
        PatientSlideCollection.load_data_by_patient(patient_src_record=self.patient_ground_truth,
                                                    target_data_frame=self.get_df(table_name),
                                                    data_list=data_list,
                                                    filenames=filenames,
                                                    flush=flush
                                                    )

    def load_score(self, scores_all_category: Sequence[Sequence[float]], filenames: Sequence[str], flush: bool):
        self.load_data('score', scores_all_category, filenames, flush)

    def entry(self, prediction):
        raise NotImplementedError

    def load_prediction(self, pred_class_names: Sequence[str], filenames: Sequence[str], flush: bool):
        # todo entry from partition
        entry_list: List[Dict[str, int]] = [self.entry(pred) for pred in pred_class_names]
        entry_array = [np.asarray(list(entry.values())) for entry in entry_list]
        self.load_data('prediction', entry_array, filenames, flush)


class CascadedPred(PatientPred):
    ...