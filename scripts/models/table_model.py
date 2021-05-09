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
                 use_one_hot=True
                 ):
        super().__init__(config, features)
        self.cat_features = cat_features
        self.clf = clf
        self.use_one_hot = use_one_hot
        self.train_columns = None

    def _add_one_hot(self, data: pd.DataFrame):
        return pd.concat([data.drop(self.cat_features, axis=1),
                          pd.get_dummies(data[self.cat_features])], axis=1)

    def _fit(self, full_data: Dict[str, pd.DataFrame]) -> None:
        data, target = self.to_data_target(full_data['train'])
        if self.use_one_hot:
            data = self._add_one_hot(data)
        self.train_columns = data.columns
        self.clf.fit(data, target)

    def _predict_proba(self, df: pd.DataFrame) -> np.array:
        data, _ = self.to_data_target(df)
        if self.use_one_hot:
            data = self._add_one_hot(data)
        if set(data.columns) != set(self.train_columns):
            for column_name in self.train_columns:
                if column_name not in data:
                    print(f'Column {column_name} was only on train')
                    data[column_name] = data[data.columns[0]].apply(lambda x: 0)
        if set(data.columns) != set(self.train_columns):
            for column_name in data.columns:
                if column_name not in set(self.train_columns):
                    print(f'Column {column_name} was only on test')
                    data.drop(column_name, axis=1, inplace=True)

        return self.clf.predict_proba(data)
