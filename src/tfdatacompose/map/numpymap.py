from abc import abstractmethod
from typing import Any, Tuple, Union

from tensorflow import DType, Tensor, numpy_function
from tensorflow.python.data import AUTOTUNE, Dataset

from src.tfdatacompose.datasetoperation import DatasetOperation


class NumpyMap(DatasetOperation):
    def __init__(
        self, out_type: Union[DType, Tuple[DType, ...]], stateful: bool = False
    ):
        self.out_type = out_type
        self.stateful = stateful

    def apply(self, dataset: Dataset) -> Dataset:
        return dataset.map(
            self.adapted_function,
            num_parallel_calls=AUTOTUNE,
            deterministic=False,
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

    @abstractmethod
    def map(self, *args: Tensor) -> Tuple[Tensor]:
        ...
