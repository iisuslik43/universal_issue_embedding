from dataclasses import dataclass
from typing import List

COMMON_FIELDS = ['entityId',
                 'summary', 'description',
                 'projectShortName', 'numberInProject',
                 'reporterName',
                 'commentsCount', 'votes',
                 'links', 'attachments',
                 'updated', 'created', 'resolved']


@dataclass
class DatasetCreatingConfig:
    custom_fields: List[str]

    @staticmethod
    def last_name(field_name: str):
        return f'Last_{field_name}'
