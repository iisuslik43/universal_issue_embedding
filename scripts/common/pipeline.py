from typing import List
from scripts.common.element import Element


class Pipeline(Element):
    def __init__(self, *elements: Element, stats_shower=None):
        elements = list(elements)
        self.elements: List = elements
        self.stats_shower = stats_shower

    def process(self, data):
        for element in self.elements:
            data = element(data)
            if self.stats_shower is not None:
                self.stats_shower(data)
        return data
