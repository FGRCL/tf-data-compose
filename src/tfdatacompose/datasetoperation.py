from abc import ABC, abstractmethod

from tensorflow.python.data import Dataset


class DatasetOperation(ABC):
    @abstractmethod
    def apply(self, dataset: Dataset) -> Dataset:
        ...
