import random

import pandas as pd
from typing import Dict, List, Optional
from scripts.common.element import Element
from scripts.dataset.preprocessing.config import PreprocessingConfig, NAN_STR, OTHER_STR, DUPLICATE_TARGET


class DuplicatesDataCreator(Element):
    def __init__(self, config: PreprocessingConfig):
        self.config = config

    def process(self, data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        return {subset: self._process(df) for subset, df in data.items()}

    def _add_duplicates_links(self, links: List) -> Optional[str]:
        if type(links) is not list:
            return None
        for link in links:
            if link['type'] == 'Duplicate' and link['role'] == 'duplicates':
                return link['value']
        return None

    def _get_duplicated_by(self, df: pd.DataFrame) -> Dict:
        project_id_to_index = {id1: id2 for id1, id2 in zip(df['in_project_id'].values,
                                                            df.index.values)}
        duplicated_by = {}

        def add_duplicate(x):
            duplicate_index, original_id = x
            if original_id is not None and original_id in project_id_to_index:
                original_index = project_id_to_index[original_id]
                duplicated_by[original_index] = duplicated_by.get(original_index, []) + [duplicate_index]
        df['index'] = df.index
        df[['index', 'duplicates']].apply(add_duplicate, axis=1)
        return duplicated_by

    def _positive_samples(self, df: pd.DataFrame, duplicates_groups: List):
        df_first, df_second = pd.DataFrame(), pd.DataFrame()
        for group in self.iterate(duplicates_groups, 'Positive samples'):
            for i in range(len(group)):
                for j in range(len(group)):
                    if i == j:
                        continue
                    df_first = df_first.append(df.iloc[group[i]])
                    df_second = df_second.append(df.iloc[group[j]])

        df_first[self.config.target_column()] = 'Duplicate'
        df_second[self.config.target_column()] = 'Duplicate'
        return self._unite(df_first, df_second)

    def _negative_samples(self, df: pd.DataFrame, count: int, duplicates_groups: List):
        df_first, df_second = pd.DataFrame(), pd.DataFrame()
        group_index = {}
        for i, group in enumerate(duplicates_groups):
            for duplicate in group:
                group_index[duplicate] = i
        for _ in self.iterate(range(count), 'Negative samples', count):
            for __ in range(10):
                i = df['index'].sample(n=1).values[0]
                j = df['index'].sample(n=1).values[0]
                if i != j and (i not in group_index or j not in group_index or group_index[i] != group_index[j]):
                    df_first = df_first.append(df.iloc[i])
                    df_second = df_second.append(df.iloc[j])
                    break
        df_first[self.config.target_column()] = 'Not_duplicate'
        df_second[self.config.target_column()] = 'Not_duplicate'
        return self._unite(df_first, df_second)

    def _unite(self, df_first, df_second):
        df_first.reset_index(drop=True, inplace=True)
        df_second.reset_index(drop=True, inplace=True)
        df_second.rename(mapper=lambda x: '2_' + x, inplace=True, axis=1)
        return pd.concat((df_first, df_second), axis=1)

    def _process(self, df: pd.DataFrame) -> pd.DataFrame:
        df['duplicates'] = df['links'].apply(self._add_duplicates_links)
        df['in_project_id'] = df['projectShortName'] + '-' + df['numberInProject'].astype(str)

        duplicated_by = self._get_duplicated_by(df)

        duplicates_groups = [[original] + duplicates for original, duplicates in duplicated_by.items()]
        df_positive = self._positive_samples(df, duplicates_groups)
        df_negative = self._negative_samples(df, len(df_positive), duplicates_groups)

        return pd.concat((df_positive, df_negative), axis=0)
