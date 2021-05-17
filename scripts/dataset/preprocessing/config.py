from dataclasses import dataclass, field

from typing import Dict, Any, Tuple

import pandas as pd

from scripts.dataset.creating.config import DatasetCreatingConfig

NAN_STR = 'NAN_VALUE'
OTHER_STR = 'OTHER_VALUE'
DUPLICATE_TARGET = 'DUPLICATE_TARGET'


def mapping_from_dict(d: Dict[str, str]):
    def mapping(x):
        return x if x not in d else d[x]

    return mapping


@dataclass
class PreprocessingConfig(DatasetCreatingConfig):
    target_field: str
    percent_for_other: float = 0.1
    target_use_other: bool = True
    target_use_unresolved_on_train: bool = True
    train_proportion: float = 0.7
    custom_fields_mappings: Dict[str, Any] = field(default_factory=dict)
    issues_range: Tuple[pd.Timestamp, pd.Timestamp] = (
        pd.Timestamp(2016, 3, 20, 9),
        pd.Timestamp(2020, 3, 20, 9)
    )

    def last_custom_fields(self):
        return [self.last_name(field) for field in self.custom_fields]

    def all_fields(self):
        return self.custom_fields + self.last_custom_fields()

    def target_column(self) -> str:
        return self.last_name(self.target_field)

    def is_predicting_duplicate(self) -> bool:
        return self.target_field == DUPLICATE_TARGET
