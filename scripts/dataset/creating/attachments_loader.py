from pathlib import Path
import os
import pandas as pd
from typing import Set, Dict, List
import requests
import urllib.request

from scripts.common.element import Element
from scripts.dataset.creating.config import DatasetCreatingConfig


class AttachmentsLoader(Element):
    def __init__(self,
                 youtrack_token: str,
                 config: DatasetCreatingConfig,
                 dir_to_save: Path,
                 extensions_to_load: Set[str]):
        self.youtrack_token = youtrack_token
        self.config = config
        self.dir_to_save = dir_to_save
        self.extensions_to_load = extensions_to_load
        self.url = 'https://youtrack.jetbrains.com'

    def _download_file(self, url: str, path_to_save: Path):
        if not path_to_save.exists():
            urllib.request.urlretrieve(self.url + url, path_to_save)

    def _get_attachments(self, issue_id: str) -> List:
        auth_header = {'Authorization': 'Bearer ' + self.youtrack_token}
        response = requests.get(f'{self.url}/api/issues/{issue_id}/attachments?fields='
                                f'url,extension,id', headers=auth_header)
        if 'error' not in response.json():
            return response.json()
        else:
            print('Error in downloading')
            return []

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        files_for_issue = {}
        for issue_id, issue_attachments in self.iterate(zip(df.entityId.values.tolist(),
                                                            df.attachments.values.tolist()),
                                                        'Downloading attachments',
                                                        total_count=len(df)):
            files_for_issue[issue_id] = []
            something_not_downloaded = False
            for attachment in issue_attachments:
                extension = attachment['value'].split('.')[-1]
                if extension in self.extensions_to_load:
                    file_name = attachment['id'] + '.' + extension
                    if not (self.dir_to_save / file_name).exists():
                        something_not_downloaded = True

            if something_not_downloaded:
                attachments = self._get_attachments(issue_id)
                for attachment in attachments:
                    if attachment['extension'] in self.extensions_to_load:
                        file_name = attachment['id'] + '.' + attachment['extension']
                        self._download_file(attachment['url'],
                                            self.dir_to_save / file_name)
                        files_for_issue[issue_id] = files_for_issue[issue_id] + [file_name]

        df['files'] = df.entityId.apply(
            lambda issue_id: files_for_issue[issue_id] if issue_id in files_for_issue else []
        )
        return df
