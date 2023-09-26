from typing import Tuple

from numpy import arange
from tensorflow import Tensor, int64
from tensorflow.python.data import Dataset

from tfdatacompose.map.numpymap import NumpyMap


class Double(NumpyMap):
    def map(self, number: int) -> Tuple[Tensor, ...]:
        return number * 2


class TestNumpyMap:
    def test_map(self):
        number_dataset = Dataset.from_tensor_slices(range(100))
        map_double = Double(int64)

        result = map_double(number_dataset)

        expected = list(arange(100) * 2)
        result = list(result.as_numpy_iterator())
        assert result == expected
