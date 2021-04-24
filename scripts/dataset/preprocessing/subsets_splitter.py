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
                                      if f != self.config.last_name(self.target_field)]
        df.drop(last_fields_without_target, axis=1, inplace=True)

    def process(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:

        # Drop target value in first state
        df.drop([self.target_field], axis=1)

        # Drop all fields from last state except target in last state
        self._drop_last_fields(df)

        if not self.config.target_use_other:
            df = df[df[self.config.target_column()] != OTHER_STR]

        # Reset index, because some values may be dropped
        df.reset_index(drop=True, inplace=True)

        train_length = int(len(df) * self.config.train_proportion)

        df_train = df[:train_length]
        if not self.config.target_use_unresolved_on_train:
            df_train = df_train[df_train['is_resolved']]

        df_test = df[train_length:]
        df_test = df_test[df_test['is_resolved']]

        data = {
            'train': df_train,
            'test': df_test
        }
        return data
