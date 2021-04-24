from typing import Dict
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

from scripts.common.element import Element


class UnifyingModel(Element):

    def process(self, data):
        pass

    def __init__(self, model_params: Dict = None, use_one_vs_all=False, use_code=False):
        self.model_params = model_params
        self.use_one_vs_all = use_one_vs_all
        self.use_code = use_code
        self.models = None
        if self.model_params is None:
            self.model_params = {'max_iter': 5000, 'C': 0.001}

    def standard_model(self):
        clf = LogisticRegression(**self.model_params)
        if self.use_one_vs_all:
            clf = OneVsRestClassifier(clf)
        return make_pipeline(StandardScaler(with_mean=False),
                             clf)

    def iterate_models(self, full_data):
        if self.models is None:
            self.models = {}
            for sub_data_name in full_data.keys():
                if sub_data_name != 'res':
                    if self.use_code or (not self.use_code and 'code' not in sub_data_name):
                        self.models[sub_data_name] = self.standard_model()
        res = []
        for sub_data_name, model in self.models.items():
            res.append((model, full_data[sub_data_name]))
        return res

    def fit(self, full_data: Dict[str, np.array]):
        for model, data in self.iterate(self.iterate_models(full_data), True, 'fitting unifying model'):
            if np.isnan(data).any():
                filtered = data[~np.isnan(data[:, 0])]
                if len(filtered) != 0:
                    model.fit(filtered, full_data['res'][~np.isnan(data[:, 0])])
            else:
                model.fit(data, full_data['res'])

    def score(self, full_data: Dict[str, np.array]):
        predictions = self.predict(full_data)
        return accuracy_score(full_data['res'], predictions)

    def predict(self, full_data: Dict[str, np.array]):
        classes = list(self.models.values())[0].classes_
        probs = self.predict_proba(full_data)
        return np.array([classes[np.argmax(pred)] for pred in probs])

    def predict_proba(self, full_data: Dict[str, np.array]):
        predictions = []
        models = self.iterate_models(full_data)
        for i in range(len(full_data['res'])):
            count = len(models)
            predict = []
            for model, data in models:
                if np.isnan(data[i][0]):
                    count -= 1
                else:
                    predict.append(model.predict_proba([data[i]]))
            predict = np.concatenate(predict)
            predict = predict.sum(axis=0) / count
            predictions.append(predict)

        return np.array(predictions)
