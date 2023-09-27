from typing import Tuple

from numpy import arange
from tensorflow import Tensor, int32
from tensorflow.python.data import Dataset

from tfdatacompose.map.pythonfunctionmap import PythonFunctionMap


class Double(PythonFunctionMap):
    def map(self, number: int) -> Tuple[Tensor, ...]:
        return number * 2


class TestPythonFunctionMap:
    def test_map(self):
        number_dataset = Dataset.from_tensor_slices(range(100))
        map_double = Double(int32)

        result = map_double(number_dataset)

        expected = list(arange(100) * 2)
        result = list(result.as_numpy_iterator())
        assert result == expected
