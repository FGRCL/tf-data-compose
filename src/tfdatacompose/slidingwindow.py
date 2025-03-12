from tfdatacompose.map.map import Map
from typing import Tuple
from tensorflow import Tensor, shape, gather, reshape
import tensorflow


class SlidingWindow(Map):
    """
    .. _Map:
    Sliding Window transformation.

    Creates a tensor of sliding windows from the input tensors.
    Each tensors will be split into windows of size `width` moving by `shift` places for each window.
    For example the tensor `[1 2 3 4 5]` with `width=3` and `shift=1` would yield `[[1 2 3] [2 3 4] [3 4 5]]`.
    The sliding windows is applied too all the inputs tensors in the same fashion.

    :param width: The length of the sliding windows.
    :param shift: The numbers steps to take between each sliding windows.
    """

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
