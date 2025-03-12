from tfdatacompose.map.numpymap import NumpyMap
from tensorflow import DType
from typing import Union, Tuple
from numpy import ndarray
from numpy.random import default_rng


class Subsample(NumpyMap):
    """
    Tensor subsamling transformation.

    Takes a random subsample from the input tensors by the given `sample_rate`.
    All input tensors are randomdly sampled with the same sample rate.

    :param sample_rate: The sampling rate of the input tensor's elements. between `0.0` and `1.0`.
    :param seed: random seed for picking the random samples.
    """

    def __init__(
        self, out_type: Union[DType, Tuple[DType, ...]], sample_rate: float, seed: int
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
