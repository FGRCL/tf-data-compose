from abc import abstractmethod

from tensorflow.python.data import AUTOTUNE, Dataset

from src.tfdatacompose.base import DatasetOperation


class Batch(DatasetOperation):
    def apply(self, dataset: Dataset) -> Dataset:
        return dataset.batch(self.get_batch_size(), num_parallel_calls=AUTOTUNE, deterministic=False, name=self.__class__.__name__)

    @abstractmethod
    def get_batch_size(self):
        ...
