from tfdatacompose.map.map import Map
from typing import Tuple
from tensorflow import Tensor, shape, gather, reshape
import tensorflow


class SlidingWindow(Map):
    def __init__(self, width: int, shift: int):
        self.width = width
        self.shift = shift

    def transform(self, *tensors: Tensor) -> Tuple[Tensor, ...]:
        return tuple([self._sliding_window(tensor) for tensor in tensors])

    def _sliding_window(self, signal):
        hops = (shape(signal)[0] - self.width + self.shift) // self.shift
        window_idx = tensorflow.range(0, self.width) + self.shift * reshape(
            tensorflow.range(0, hops), (-1, 1)
        )
        return gather(signal, window_idx)
