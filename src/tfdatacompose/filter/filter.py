from abc import abstractmethod

from tensorflow.python.data import Dataset

from src.tfdatacompose.base import DatasetOperation


class Filter(DatasetOperation):
    def apply(self, dataset) -> Dataset:
        return dataset.filter(self.filter, name=self.__class__.__name__)

    @abstractmethod
    def filter(self, *args) -> bool:
        ...
