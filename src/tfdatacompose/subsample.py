from tfdatacompose.map.numpymap import NumpyMap
from tensorflow import DType
from typing import Union, Tuple
from numpy import ndarray
from numpy.random import default_rng


class Subsample(NumpyMap):
    def __init__(
        self, out_type: Union[DType, Tuple[DType, ...]], seed: int, sample_rate: float
    ):
        super().__init__(out_type)
        self.random_generator = default_rng(seed)
        self.sample_rate = sample_rate

    def transform(self, *arrays: ndarray) -> Tuple[ndarray, ...]:
        subsampledArrays = []
        for array in arrays:
            n_elements = array.shape[0]
            sample_size = int(n_elements * self.sample_rate)
            sample_idx = self.random_generator.choice(
                range(n_elements), sample_size, False
            )
            subsampledArrays += array[sample_idx]

        return tuple(subsampledArrays)
