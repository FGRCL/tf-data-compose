from abc import abstractmethod
from typing import Any

from tensorflow import Tensor, numpy_function
from tensorflow.python.data import Dataset

from src.tfdatacompose.datasetoperation import DatasetOperation


class NumpyFilter(DatasetOperation):
    def __init__(self, stateful: bool = False) -> None:
        super().__init__()
        self.stateful: bool = stateful

    def apply(self, dataset: Dataset) -> Dataset:
        return dataset.filter(self.adapted_function, name=self.__class__.__name__)

    def adapted_function(self, *args: Tensor):
        ls_args = list(args)
        return numpy_function(
            self.filter,
            ls_args,
            bool,
            self.stateful,
            self.__class__.__name__,
        )

    @abstractmethod
    def filter(self, *args) -> bool:
        ...
