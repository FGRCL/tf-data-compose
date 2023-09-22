from abc import abstractmethod
from typing import Any, Tuple

from tensorflow import Tensor
from tensorflow.python.data import AUTOTUNE, Dataset

from src.tfdatacompose.base import DatasetOperation


class Map(DatasetOperation):
    def apply(self, dataset: Dataset) -> Dataset:
        return dataset.map(
            self.map,
            num_parallel_calls=AUTOTUNE,
            deterministic=False,
            name=self.__class__.__name__,
        )

    @abstractmethod
    def map(self, *args: Tensor) -> Tuple[Tensor, ...]:
        ...
