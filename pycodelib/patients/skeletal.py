from abc import ABC, abstractmethod
from typing import Sequence, Set, Any
import pandas as pd
import numpy as np


class PandasRecord(ABC):

    @property
    def class_list(self) -> np.ndarray:
        return self._class_list

    def __init__(self):
        self.__table_names: Set[str] = set()
        self._class_list = None  # placeholders

    def build_df(self, table_name: str, columns: Sequence[str] = None):
        if not hasattr(self, '__table_names'):
            self.__table_names = set()
        if table_name not in self.__table_names:
            self.__table_names.add(table_name)
        setattr(self, table_name, pd.DataFrame(columns=columns))

    def get_df(self, table_name) -> pd.DataFrame:
        if table_name not in self.__table_names:
            raise ValueError(f'Unexpected Table{table_name}')
        table = getattr(self, table_name)
        if not isinstance(table, pd.DataFrame):
            raise TypeError(f'Not DataFrame{type(table)}')
        return table

    def flush_df_by_name(self, df_name: str = None):
        if df_name is not None:
            self.get_df(df_name).drop(self.get_df(df_name).index, inplace=True)
        else:
            for name in self.__table_names:
                self.flush_df_by_name(name)

    @staticmethod
    def insert_data(dataframe: pd.DataFrame, row_id, data: Any):
        try:
            existed = dataframe.loc[row_id]
        except KeyError:
            existed = np.zeros_like(data)
        data_new = data + existed
        dataframe.loc[row_id] = data_new

    @staticmethod
    def flush_df(table: pd.DataFrame):
        table.drop(table.index, inplace=True)

    @abstractmethod
    def load_data(self, table_name: str, data_list: Sequence, filenames: Sequence[str], flush: bool):
        ...
