from typing import Callable, Tuple, Union

from tensorflow import DType, Tensor, py_function
from tensorflow.python.data import Dataset

from tfdatacompose.datasetoperation import DatasetOperation


class PythonFunctionLambdaMap(DatasetOperation):
    def __init__(self, out_type: Union[DType, Tuple[DType, ...]], map: Callable):
        self.out_type = out_type
        self.map = map

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
