import unittest
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from scripts.dataset.preprocessing.config import PreprocessingConfig, OTHER_STR, NAN_STR
from scripts.models.base_model import calc_not_nans_index, BaseModel
from scripts.models.table_model import TableModel
from scripts.models.unifying_model import UnifyingModel
from tests.models.basic_model_test import BasicModelTest


class TestTableModel(BasicModelTest):

    def test_one_model_on_train(self):
        model = UnifyingModel(config=self.config, models=[
            TableModel(clf=LogisticRegression(),
                       config=self.config,
                       features=['Last_Custom', 'without_nans'],
                       cat_features=['Last_Custom'])
        ])
        model.fit(self.full_data)
        model.predict_proba(self.full_data['train'])

    def test_one_model_on_test(self):
        model = UnifyingModel(config=self.config, models=[
            TableModel(clf=LogisticRegression(),
                       config=self.config,
                       features=['Last_Custom', 'without_nans'],
                       cat_features=['Last_Custom'])
        ])
        model.fit(self.full_data)
        model.predict_proba(self.full_data['test'])


if __name__ == '__main__':
    unittest.main()
