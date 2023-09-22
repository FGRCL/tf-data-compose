from tensorflow.python.data import Dataset

from src.tfdatacompose.datasetoperation import DatasetOperation


class Take(DatasetOperation):
    def __init__(self, count: int):
        self.count = count

    def apply(self, dataset: Dataset) -> Dataset:
        return dataset.take(self.count, self.__class__.__name__)
