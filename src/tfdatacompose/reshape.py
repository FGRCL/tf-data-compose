from tfdatacompose.map.map import Map
from typing import List, Optional
from tensorflow import reshape, Tensor, Tuple


class Reshape(Map):
    """
    Reshape tensors transformation.

    Wraps the `Tensorflow Reshape`_ operation on the dataset.
    Reshapes the tensors with the new given shapes.

    :param count: list of the new shapes to reshape the input tensors.

    .. _Tensorflow Map: https://www.tensorflow.org/api_docs/python/tf/data/Dataset#reshape
    """

    def __init__(self, *shapes: List[Optional[int]]):
        self.shapes = shapes

    def transform(self, *args: Tensor) -> Tuple[Tensor, ...]:
        return tuple(
            (reshape(tensor, shape) for tensor, shape in zip(args, self.shapes))
        )
