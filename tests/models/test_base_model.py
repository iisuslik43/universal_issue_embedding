import unittest
from typing import Dict

import numpy as np
import pandas as pd

from scripts.dataset.preprocessing.config import PreprocessingConfig, OTHER_STR, NAN_STR
from scripts.models.base_model import calc_not_nans_index, BaseModel
from tests.models.basic_model_test import BasicModelTest


class MockModel(BaseModel):
    def _fit(self, full_data: Dict[str, pd.DataFrame]) -> None:
        pass

    def _predict_proba(self, df: pd.DataFrame) -> np.array:
        return np.ones((len(df), 43))


class TestBaseModel(BasicModelTest):
    def test_not_nan_index(self):
        df = self.full_data['train']
        self.assertTrue(np.allclose(calc_not_nans_index(df), np.array([2])))
        self.assertTrue(np.allclose(calc_not_nans_index(df[['with_nans1', 'with_nans2']]), np.array([2])))
        self.assertTrue(np.allclose(calc_not_nans_index(df[['with_nans1']]), np.array([0, 2])))

    def test_without_nans_1_feature(self):
        model = MockModel(self.config, ['without_nans'])
        model.fit(self.full_data)
        proba = model.predict_proba(self.full_data['train'])
        self.assertTrue(np.allclose(proba, np.ones((len(self.full_data['train']), 43))))

    def test_without_nans_2_features(self):
        model = MockModel(self.config, ['Last_Custom', 'without_nans'])
        model.fit(self.full_data)
        proba = model.predict_proba(self.full_data['train'])
        self.assertTrue(np.allclose(proba, np.ones((len(self.full_data['train']), 43))))

    def test_with_nans_1_feature(self):
        model = MockModel(self.config, ['with_nans1'])
        model.fit(self.full_data)
        proba = model.predict_proba(self.full_data['train'])
        expected = np.array([
            [1] * 43,
            [0] * 43,
            [1] * 43,
            [0] * 43
        ])
        self.assertTrue(np.allclose(proba, expected))

    def test_with_nans_2_features(self):
        model = MockModel(self.config, ['with_nans1', 'with_nans2'])
        model.fit(self.full_data)
        proba = model.predict_proba(self.full_data['train'])
        expected = np.array([
            [0] * 43,
            [0] * 43,
            [1] * 43,
            [0] * 43
        ])
        self.assertTrue(np.allclose(proba, expected))


if __name__ == '__main__':
    unittest.main()
