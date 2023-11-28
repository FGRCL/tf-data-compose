from typing import Callable, Tuple, Union

from tensorflow import DType, Tensor, numpy_function
from tensorflow.python.data import Dataset

from tfdatacompose.datasetoperation import DatasetOperation


class NumpyLambdaMap(DatasetOperation):
    def __init__(
        self,
        out_type: Union[DType, Tuple[DType, ...]],
        map: Callable,
        stateful: bool = False,
    ):
        self.out_type = out_type
        self.map = map
        self.stateful = stateful

    def apply(self, dataset: Dataset) -> Dataset:
        return dataset.map(
            self.adapted_function,
            name=self.__class__.__name__,
        )

    def adapted_function(self, *args: Tensor):
        ls_args = list(args)
        return numpy_function(
            self.map,
            ls_args,
            self.out_type,
            self.stateful,
            self.__class__.__name__,
        )
