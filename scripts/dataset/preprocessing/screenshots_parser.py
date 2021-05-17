from pathlib import Path

import pandas as pd
import pytesseract as pytesseract
from PIL import Image
from typing import Dict, List, Optional
from scripts.common.element import Element
from scripts.dataset.preprocessing.config import PreprocessingConfig, NAN_STR, OTHER_STR, DUPLICATE_TARGET
from multiprocessing import Pool


def _recognize_screenshots_texts(screenshots: List[Path]) -> str:
    try:
        return '\n'.join([pytesseract.image_to_string(Image.open(file)) for file in screenshots
                          if file.exists()])
    except Exception:
        return ''


class ScreenshotsParser(Element):
    def __init__(self, config: PreprocessingConfig, dir_to_save: Path):
        self.config = config
        self.dir_to_save = dir_to_save

    def process(self, data: pd.DataFrame) -> pd.DataFrame:
        files = [[self.dir_to_save / file for file in files] for files in data['files'].values]
        with Pool(10) as p:
            result_list = p.map(_recognize_screenshots_texts, files)
        data['screenshots'] = pd.Series(result_list)
        return data
