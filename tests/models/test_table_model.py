import unittest
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from scripts.dataset.preprocessing.config import PreprocessingConfig, OTHER_STR, NAN_STR
from scripts.models.base_model import calc_not_nans_index, BaseModel
from scripts.models.table_model import TableModel
from tests.models.basic_model_test import BasicModelTest


class TestTableModel(BasicModelTest):

    def test_on_train(self):
        model = TableModel(clf=LogisticRegression(),
                           config=self.config,
                           features=['Last_Custom', 'without_nans'],
                           cat_features=['Last_Custom'])
        model.fit(self.full_data)
        model.predict_proba(self.full_data['train'])

    def test_on_test(self):
        model = TableModel(clf=LogisticRegression(),
                           config=self.config,
                           features=['Last_Custom', 'without_nans'],
                           cat_features=['Last_Custom'])
        model.fit(self.full_data)
        model.predict_proba(self.full_data['test'])


if __name__ == '__main__':
    unittest.main()
