from abc import abstractmethod
from typing import Any, Tuple, Union

from tensorflow import DType, Tensor, py_function
from tensorflow.python.data import Dataset

from src.tfdatacompose.datasetoperation import DatasetOperation


class PythonFunctionMap(DatasetOperation):
    def __init__(self, out_type: Union[DType, Tuple[DType, ...]]):
        self.out_type = out_type

    def apply(self, dataset: Dataset) -> Dataset:
        return dataset.map(
            self.adapted_function,
            name=self.__class__.__name__,
        )

    def adapted_function(self, *args: Tensor):
        ls_args = list(args)
        return py_function(
            self.map,
            ls_args,
            self.out_type,
            self.__class__.__name__,
        )

    @abstractmethod
    def map(self, *args) -> Any: ...
