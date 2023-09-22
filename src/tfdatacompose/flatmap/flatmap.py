from abc import abstractmethod
from typing import Tuple

from tensorflow import Tensor
from tensorflow.python.data import Dataset

from src.tfdatacompose.datasetoperation import DatasetOperation


class FlatMap(DatasetOperation):
    def apply(self, dataset: Dataset) -> Dataset:
        return dataset.flat_map(self.flatmap, name=self.__class__.__name__)

    @abstractmethod
    def flatmap(self, *args: Tensor) -> Tuple[Tensor, ...]:
        ...
