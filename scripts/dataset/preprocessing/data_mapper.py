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
            counts: dict = df[field_name].value_counts().to_dict()
            if NAN_STR in counts:
                del counts[NAN_STR]
            max_count = max(counts.values())
            threshold = max_count * self.config.percent_for_other

            def other(x):
                return x if x not in counts or counts[x] > threshold else OTHER_STR

            df[field_name] = df[field_name].apply(other)
            last_field_name = self.config.last_name(field_name)
            df[last_field_name] = df[last_field_name].apply(other)

    def process(self, data: pd.DataFrame) -> pd.DataFrame:
        df: pd.DataFrame = data.copy()

        self.fill_nans(df)
        self.map_custom_fields(df)
        self.set_other_custom_fields(df)

        return df
