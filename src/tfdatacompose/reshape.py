from tfdatacompose.map.map import Map
from typing import List, Optional
from tensorflow import reshape, Tensor, Tuple


class Reshape(Map):
    def __init__(self, *shapes: List[Optional[int]]):
        self.shapes = shapes

    def transform(self, *args: Tensor) -> Tuple[Tensor, ...]:
        return tuple(
            (reshape(tensor, shape) for tensor, shape in zip(args, self.shapes))
        )
