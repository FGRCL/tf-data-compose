from abc import abstractmethod
from typing import Tuple, Union

from tensorflow import DType, Tensor, py_function
from tensorflow.python.data import Dataset

from src.tfdatacompose.base import DatasetOperation


class PythonFunctionFlatMap(DatasetOperation):
    def __init__(self, out_type: Union[DType, Tuple[DType, ...]]):
        self.out_type = out_type

    def apply(self, dataset: Dataset) -> Dataset:
        pass

    def adapted_flatmap(self, *args: Tensor):
        ls_args = list(args)
        return py_function(
            self.flatmap,
            ls_args,
            self.out_type,
            self.__class__.__name__,
        )

    @abstractmethod
    def flatmap(self, *args: Tensor) -> Tuple[Tensor, ...]:
        ...
