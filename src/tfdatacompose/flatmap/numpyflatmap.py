from abc import abstractmethod
from typing import Tuple, Union

from tensorflow import DType, Tensor, numpy_function
from tensorflow.python.data import Dataset

from src.tfdatacompose.datasetoperation import DatasetOperation


class NumpyFlatMap(DatasetOperation):
    def __init__(
            self, out_type: Union[DType, Tuple[DType, ...]], stateful: bool = False
    ):
        self.out_type = out_type
        self.stateful = stateful

    def apply(self, dataset: Dataset) -> Dataset:
        return dataset.flat_map(self.adapted_flatmap, name=self.__class__.__name__)

    def adapted_flatmap(self, *args: Tensor):
        ls_args = list(args)
        return numpy_function(
            self.flatmap,
            ls_args,
            self.out_type,
            self.stateful,
            self.__class__.__name__,
        )

    @abstractmethod
    def flatmap(self, *args: Tensor) -> Tuple[Tensor, ...]:
        ...
