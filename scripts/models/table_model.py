import numpy as np
import pandas as pd
from typing import List, Dict

from scripts.dataset.preprocessing.config import PreprocessingConfig
from scripts.models.base_model import BaseModel


class TableModel(BaseModel):
    def __init__(self,
                 clf,
                 config: PreprocessingConfig,
                 features: List[str],
                 cat_features: List[str],
                 use_one_hot=True,
                 fit_params=None
                 ):
        super().__init__(config, features)
        self.cat_features = cat_features
        if fit_params is None:
            fit_params = {}
        self.clf = clf
        self.use_one_hot = use_one_hot
        self.fit_params = fit_params

    def fit(self, data: Dict[str, pd.DataFrame]) -> None:
        data, target = self.to_data_target(data['train'])
        if self.use_one_hot:
            data = pd.get_dummies(data)
        self.clf.fit(data, target, **self.fit_params)

    def transform(self, df: pd.DataFrame) -> np.array:
        data, _ = self.to_data_target(df)
        if self.use_one_hot:
            data[self.cat_features] = pd.get_dummies(data[self.cat_features])
        return self.clf.transform(data)
