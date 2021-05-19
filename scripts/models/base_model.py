from abc import ABCMeta, abstractmethod
from typing import List, Dict
import numpy as np
import pandas as pd

from scripts.common.element import Element
from scripts.dataset.preprocessing.config import PreprocessingConfig


def calc_not_nans_index(df1: pd.DataFrame):
    return df1[~pd.isna(df1).any(axis=1)].index

class BaseModel(Element):
    __metaclass__ = ABCMeta

    def __init__(self,
                 config: PreprocessingConfig,
                 features: List[str]):
        self.target_name = config.target_column()
        self.original_features = features
        self.is_predicting_duplicate = config.is_predicting_duplicate()
        if self.is_predicting_duplicate:
            self.features = features + ['2_' + feature for feature in features]
        else:
            self.features = features

    def to_data_target(self, df: pd.DataFrame):
        return df[self.features], df[self.target_name]

    def fit(self, full_data: Dict[str, pd.DataFrame]) -> None:
        data_result = {}
        for subset_name, df in full_data.items():
            data, _ = self.to_data_target(df)
            not_nans_index = calc_not_nans_index(data)
            if len(not_nans_index) != len(data):
                print(subset_name, 'has', round(float(len(not_nans_index)) / len(data) * 100, 2),
                      '% of not nans data for', self.features)
                data_result[subset_name] = df.iloc[not_nans_index]
            else:
                data_result[subset_name] = df
        self._fit(data_result)

    def predict_proba(self, df: pd.DataFrame) -> np.array:
        data, _ = self.to_data_target(df)
        not_nans_index = calc_not_nans_index(data)
        if len(not_nans_index) != len(data):
            prob = self._predict_proba(df.iloc[not_nans_index])
            # Predicts for issues with nans are 0
            all_prob = np.zeros((len(df), prob.shape[1]))
            all_prob[not_nans_index] = prob
            return all_prob
        else:
            return self._predict_proba(df)

    def unite_pairs(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.is_predicting_duplicate:
            return df
        second_names = ['2_' + feature for feature in self.original_features]
        return pd.concat((df[self.original_features],
                          df[second_names].rename(mapper=lambda x: x[2:], axis=1)),
                         axis=0)

    def concat_pairs(self, data: np.array) -> np.array:
        if not self.is_predicting_duplicate:
            return data
        middle = len(data) // 2
        return np.concatenate((data[:middle], data[middle:]), axis=1)

    @abstractmethod
    def _fit(self, full_data: Dict[str, pd.DataFrame]) -> None:
        """fits classifier with extracted data"""

    @abstractmethod
    def _predict_proba(self, df: pd.DataFrame) -> np.array:
        """predicts probabilities for each class"""

    def process(self, data):
        pass
