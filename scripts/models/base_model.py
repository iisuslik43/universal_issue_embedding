from abc import ABCMeta, abstractmethod
from typing import List, Dict
import numpy as np
import pandas as pd

from scripts.dataset.preprocessing.config import PreprocessingConfig


class BaseModel:
    __metaclass__ = ABCMeta

    def __init__(self,
                 config: PreprocessingConfig,
                 features: List[str]):
        self.target_name = config.target_column()
        self.features = features

    def to_data_target(self, df: pd.DataFrame):
        return df[self.features], df[self.target_name]

    @abstractmethod
    def fit(self, data: Dict[str, pd.DataFrame]) -> None:
        """fits classifier with extracted data"""

    @abstractmethod
    def transform(self, data: pd.DataFrame) -> np.array:
        """transforms extracted data"""
