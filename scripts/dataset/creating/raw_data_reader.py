from pathlib import Path
import pandas as pd
from typing import List

from jsonlines import jsonlines

from scripts.common.element import Element
from scripts.dataset.creating.config import DatasetCreatingConfig, COMMON_FIELDS


def _list_without(a: List, b: List):
    return list(filter(lambda f: f not in b, a))


class RawDataReader(Element):
    def __init__(self, config: DatasetCreatingConfig):
        self.config = config

    def _read_json_file(self, file: Path) -> List:
        with open(file, mode='r') as f:
            total_count = sum([1 for _ in f])
        data_json = []
        with jsonlines.open(file) as reader:
            for obj in self.iterate(reader, f'Reading file {file.name}', total_count=total_count):
                if 'field' in obj:
                    data_json.append(obj['field'])
                    data_json[-1]['entityId'] = obj['entityId']
        return data_json

    def process(self, data_filenames: List[Path]) -> pd.DataFrame:
        dfs_list = []
        for file in data_filenames:
            data_json = self._read_json_file(file)
            df = pd.DataFrame(data_json)
            self._print_missing_columns(df, file)
            self._filter_df(df)
            dfs_list.append(df)
        df_final = pd.concat(dfs_list)
        self._filter_df(df_final)
        return df_final

    def _print_missing_columns(self, df: pd.DataFrame, file: Path) -> None:
        for field in COMMON_FIELDS:
            if field not in df:
                print(f'Common field "{field}" not in data {file}')
        for field in self.config.custom_fields:
            if field not in df:
                print(f'Custom field "{field}" not in data {file}')

    def _filter_df(self, df: pd.DataFrame) -> None:
        not_none_columns = _list_without(COMMON_FIELDS,
                                         ['resolved', 'summary', 'description', 'attachments', 'links'])
        df.dropna(how='any',
                  subset=not_none_columns,
                  inplace=True)
        df.drop_duplicates(['entityId', 'updated'], inplace=True)
        for column in df.columns:
            if column not in COMMON_FIELDS + self.config.custom_fields:
                df.drop(column, axis=1, inplace=True)
