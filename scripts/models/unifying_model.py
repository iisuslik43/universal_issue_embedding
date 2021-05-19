from typing import Dict, List, Optional, Tuple
import numpy as np
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.utils import resample

from scripts.dataset.preprocessing.config import PreprocessingConfig
from scripts.models.base_model import BaseModel


def confidence_interval(data: List[float], confidence=0.95) -> Tuple[float, float]:
    data.sort()
    c = (1 - confidence) / 2
    count = int(c * len(data))
    left = data[count]
    right = data[len(data) - 1 - count]
    mid = (right + left) / 2
    return mid, right - mid


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

    def score_with_confidence(self, df: pd.DataFrame, score_func=accuracy_score, score_params=None):
        if score_params is None:
            score_params = {}
        predictions = self.predict(df)
        target = df[self.target_name]
        scores = []
        for _ in self.iterate(range(100), 'Calculating confidence'):
            predictions_i, target_i = zip(*resample(list(zip(predictions, target))))
            score = score_func(target_i, predictions_i, **score_params)
            scores.append(score)
        return confidence_interval(scores)

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