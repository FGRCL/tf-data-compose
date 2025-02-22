from typing import Callable

from tensorflow.python.data import Dataset

from tfdatacompose.datasetoperation import DatasetOperation


class LambdaMap(DatasetOperation):
    def __init__(self, map_fn: Callable):
        self.map_fn = map_fn

    def apply(self, dataset: Dataset) -> Dataset:
        return dataset.map(self.map_fn, name=(self.__class__.__name__,))
