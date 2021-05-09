import unittest
import numpy as np
import pandas as pd

from scripts.dataset.preprocessing.config import PreprocessingConfig, OTHER_STR, NAN_STR


class BasicModelTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = PreprocessingConfig(custom_fields=['Custom'], target_field='Target')
        self.full_data = {
            'train': pd.DataFrame({
                'Last_Custom': ['Value1', NAN_STR, 'Value2', OTHER_STR],
                'Last_Target': [OTHER_STR, 'Y1', 'Y2', 'Y3'],
                'without_nans': [1, 2, 3, 4],
                'with_nans1': ['a', np.nan, 'b', np.nan],
                'with_nans2': [np.nan, np.nan, 'c', 'd'],
                'text': ['hi', 'my', 'name', 'is']
            }),
            'test': pd.DataFrame({
                'Last_Custom': ['Value1', NAN_STR, 'Value1', NAN_STR],
                'Last_Target': [OTHER_STR, 'Y1', 'Y2', 'Y3'],
                'without_nans': [1, 2, 1, 2],
                'with_nans1': ['a', np.nan, 'b', np.nan],
                'with_nans2': [np.nan, np.nan, 'c', 'd'],
                'text': ['name', 'is', 'slim', 'shady']
            })
        }
