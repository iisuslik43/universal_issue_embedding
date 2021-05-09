from typing import Dict, List, Optional
import numpy as np
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix

from scripts.dataset.preprocessing.config import PreprocessingConfig
from scripts.models.base_model import BaseModel


class UnifyingModel(BaseModel):

    def __init__(self, config: PreprocessingConfig, models: List[BaseModel]):
        super().__init__(config, [])
        self.models = models

    def fit(self, data: Dict[str, pd.DataFrame]) -> None:
        for model in self.iterate(self.models, 'fitting unifying model'):
            model.fit(data)

    def predict_proba(self, df: pd.DataFrame) -> np.array:
        predicts = np.array([model.predict_proba(df) for model in self.models])
        return predicts.transpose((1, 2, 0)).sum(axis=-1)

    def plot_confusion_matrix(self, df: pd.DataFrame):
        predictions = self.predict(df)
        target = df[self.target_name]
        conf_matrix = confusion_matrix(target, predictions)
        classes = target.unique()
        df_cm = pd.DataFrame(conf_matrix,
                             index=classes,
                             columns=classes)
        plt.figure(figsize=(10, 7))
        return sn.heatmap(df_cm, annot=True)

    def score(self, df: pd.DataFrame, score_func=accuracy_score, score_params=None) -> float:
        if score_params is None:
            score_params = {}
        predictions = self.predict(df)
        return score_func(df[self.target_name], predictions, **score_params)

    def _classes(self):
        return self.models[0].clf.classes_

    def predict(self, df: pd.DataFrame) -> np.array:
        classes = self._classes()
        probs = self.predict_proba(df)
        return np.array([classes[np.argmax(pred)] for pred in probs])

    def _fit(self, full_data: Dict[str, pd.DataFrame]) -> None:
        pass

    def _predict_proba(self, df: pd.DataFrame) -> np.array:
        pass