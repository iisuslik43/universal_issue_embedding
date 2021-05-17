import pandas as pd
from typing import Dict

from scripts.common.element import Element
from scripts.dataset.preprocessing.config import PreprocessingConfig, NAN_STR, OTHER_STR


class DataSplitter(Element):
    def __init__(self, config: PreprocessingConfig):
        self.config = config
        self.target_field = config.target_field

    def _drop_last_fields(self, df):
        last_fields_without_target = [f for f in self.config.last_custom_fields()
                                      if f != self.config.target_column()]
        df.drop(last_fields_without_target, axis=1, inplace=True)

    def process(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:

        if not self.config.is_predicting_duplicate():
            # Drop target value in first state
            df = df.drop([self.target_field], axis=1)

            # Drop target NaNs
            df = df[df[self.config.target_column()] != NAN_STR]

            if not self.config.target_use_other:
                df = df[df[self.config.target_column()] != OTHER_STR]

        # Drop all fields from last state except target in last state
        self._drop_last_fields(df)

        # Reset index, because some values may be dropped
        df.reset_index(drop=True, inplace=True)

        train_length = int(len(df) * self.config.train_proportion)

        df_train = df[:train_length]
        if not self.config.target_use_unresolved_on_train:
            df_train = df_train[df_train['is_resolved']]
            df_train.reset_index(drop=True, inplace=True)

        df_test = df[train_length:]
        df_test = df_test[df_test['is_resolved']]
        df_test.reset_index(drop=True, inplace=True)

        data = {
            'train': df_train,
            'test': df_test
        }
        return data
