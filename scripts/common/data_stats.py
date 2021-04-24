from typing import Any

import pandas as pd
from scripts.common.element import Element


class DataFrameStats(Element):
    def process(self, data: Any):
        if type(data) is pd.DataFrame:
            print(f'Pure length is {len(data)}, issue count is {len(DataFrameStats._get_last(data))}')
        elif type(data) is dict:
            for k, v in data.items():
                if type(v) is dict:
                    length = len(list(v.items())[0][1])
                elif type(v) is tuple:
                    length = len(v[0])
                else:
                    length = len(v)
                print(f'Length of {k} is {length}')
        return data

    @staticmethod
    def _get_last(df: pd.DataFrame) -> pd.DataFrame:
        return df.groupby(df['entityId']).last()
