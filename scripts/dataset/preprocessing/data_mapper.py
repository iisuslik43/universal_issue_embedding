import numpy as np
import pandas as pd
from scripts.common.element import Element
from scripts.dataset.preprocessing.config import PreprocessingConfig, NAN_STR, OTHER_STR


class DataMapper(Element):
    def __init__(self, config: PreprocessingConfig):
        self.config = config

    def fill_nans(self, df):
        df[self.config.all_fields()] = df[self.config.all_fields()].fillna(NAN_STR)

    def map_custom_fields(self, df: pd.DataFrame) -> None:
        for field_name, mapping in self.config.custom_fields_mappings.items():
            df[field_name] = df[field_name].apply(mapping)
            last_field_name = self.config.last_name(field_name)
            df[last_field_name] = df[last_field_name].apply(mapping)

    def set_other_custom_fields(self, df: pd.DataFrame) -> None:
        for field_name in self.config.custom_fields:
            last_field_name = self.config.last_name(field_name)
            counts: dict = df[last_field_name].value_counts().to_dict()
            counts.update(df[field_name].value_counts().to_dict())
            if NAN_STR in counts:
                del counts[NAN_STR]
            max_count = max(counts.values())
            threshold = max_count * self.config.percent_for_other

            def other(x):
                return x if x not in counts or counts[x] > threshold else OTHER_STR

            df[field_name] = df[field_name].apply(other)
            df[last_field_name] = df[last_field_name].apply(other)

    def filter_issues_from_range(self, df: pd.DataFrame) -> pd.DataFrame:
        df['created_datetime'] = pd.to_datetime(df['created'].astype(int), unit='ms')
        start, end = self.config.issues_range
        df = df[start < df['created_datetime']]
        df = df[df['created_datetime'] < end]
        return df

    def _extract_features(self, df: pd.DataFrame) -> None:
        if 'screenshots' in df.columns:
            df['screenshots'] = df['screenshots'].apply(lambda s: np.nan if s == '' else s)

    def process(self, data: pd.DataFrame) -> pd.DataFrame:
        df: pd.DataFrame = data.copy()
        df = self.filter_issues_from_range(df)
        self._extract_features(df)
        self.fill_nans(df)
        self.map_custom_fields(df)
        self.set_other_custom_fields(df)

        return df
