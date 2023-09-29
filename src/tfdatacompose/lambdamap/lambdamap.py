from typing import Callable

from tensorflow.python.data import Dataset

from tfdatacompose.datasetoperation import DatasetOperation


class LambdaMap(DatasetOperation):
    def __init__(self, map: Callable):
        self.map = map

    def apply(self, dataset: Dataset) -> Dataset:
        return dataset.map(self.map, name=(self.__class__.__name__,))
