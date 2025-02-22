from abc import abstractmethod
from typing import Tuple

from tensorflow import Tensor
from tensorflow.python.data import Dataset

from src.tfdatacompose.datasetoperation import DatasetOperation


class Map(DatasetOperation):
    def apply(self, dataset: Dataset) -> Dataset:
        return dataset.map(
            self.map,
            name=self.__class__.__name__,
        )

    @abstractmethod
    def map(self, *args: Tensor) -> Tuple[Tensor, ...]: ...
