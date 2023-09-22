import tensorflow
from tensorflow import Tensor, shape

from src.tfdatacompose.map.map import Map


class PrintShape(Map):
    number = 1

    def __init__(self, name: str | None = None):
        if name is None:
            self.name = PrintShape.number
            PrintShape.number += 1
        else:
            self.name = name

    def map(self, *args: Tensor) -> tuple[Tensor, ...]:
        tensorflow.print(self.name, *(shape(a) for a in args))
        return args
