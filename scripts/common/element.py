from abc import ABCMeta, abstractmethod
from typing import Optional

from tqdm.auto import tqdm


class Element:
    __metaclass__ = ABCMeta

    @abstractmethod
    def process(self, data):
        """processes data and returns the result"""

    def __call__(self, data):
        return self.process(data)

    @staticmethod
    def iterate(it, description: str, total_count: Optional[int] = None):
        return tqdm(it, description, total=total_count)

