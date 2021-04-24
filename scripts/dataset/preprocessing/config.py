from dataclasses import dataclass, field

from typing import Dict, Any

from scripts.dataset.creating.config import DatasetCreatingConfig

NAN_STR = 'NAN_VALUE'
OTHER_STR = 'OTHER_VALUE'


def mapping_from_dict(d: Dict[str, str]):
    def mapping(x):
        return x if x not in d else d[x]

    return mapping


@dataclass
class PreprocessingConfig(DatasetCreatingConfig):
    target_field: str
    custom_fields_mappings: Dict[str, Any] = field(default_factory=dict)
    percent_for_other = 0.05
    target_use_other: bool = True
    target_use_unresolved_on_train: bool = True
    train_proportion = 0.8

    def last_custom_fields(self):
        return [self.last_name(field) for field in self.custom_fields]

    def all_fields(self):
        return self.custom_fields + self.last_custom_fields()

    def target_column(self):
        return self.last_name(self.target_field)
