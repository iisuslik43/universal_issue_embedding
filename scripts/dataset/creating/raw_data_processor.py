import pandas as pd
from scripts.common.element import Element
from scripts.dataset.creating.config import DatasetCreatingConfig


class RawDataProcessor(Element):
    def __init__(self, config: DatasetCreatingConfig):
        self.config = config

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        # Create a copy with sorted issues
        df = df.sort_values(by=['entityId', 'updated'])

        # Now all transformations can be done inplace
        df.reset_index(drop=True, inplace=True)
        self._refactor_common_fields(df)
        self._set_first_for_custom_fields(df)
        self._add_last_value_for_custom_fields(df)

        # Add time of last update
        df['last_updated'] = df['updated'].groupby(df['entityId']).transform('last')

        df['is_resolved'] = ~pd.isna(df['resolved'].groupby(df['entityId']).transform('last'))

        # Return only first issue state
        df = df.groupby(df['entityId']).first()
        return df

    @staticmethod
    def _refactor_common_fields(df: pd.DataFrame) -> None:
        df['created'] = pd.to_datetime(df['created'].astype(int), unit='ms')
        df['updated'] = pd.to_datetime(df['updated'].astype(int), unit='ms')
        df['resolved'] = pd.to_datetime(df['resolved'].fillna('').map(lambda x: int(x) if x else None), unit='ms')

        df['description'] = df['description'].fillna('')
        df['summary'] = df['summary'].fillna('')
        df['attachments'] = df['attachments'].apply(lambda x: x if type(x) is list else [])
        df['links'] = df['links'].apply(lambda x: x if type(x) is list else [])

        df['votes'] = df['votes'].astype(int)
        df['commentsCount'] = df['commentsCount'].astype(int)

    def _set_first_for_custom_fields(self, df: pd.DataFrame) -> None:
        def get_value(x):
            if type(x) is list:
                first = x[0]
                if type(first) is dict and 'value' in first:
                    return first['value']
                else:
                    return first
            else:
                return x

        for field_name in self.config.custom_fields:
            df[field_name] = df[field_name].apply(get_value)

    def _add_last_value_for_custom_fields(self, df: pd.DataFrame) -> None:
        for field_name in self.config.custom_fields:
            last_field_name = self.config.last_name(field_name)
            df[last_field_name] = df[field_name].groupby(df['entityId']).transform('last')
