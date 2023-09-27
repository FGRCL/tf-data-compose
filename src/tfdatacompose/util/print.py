from typing import Tuple

import tensorflow
from tensorflow import Tensor

from src.tfdatacompose.map.map import Map


class Print(Map):
    number = 1

    def __init__(self, name: str | None = None):
        if name is None:
            self.name = Print.number
            Print.number += 1
        else:
            self.name = name

    def map(self, *args) -> Tuple[Tensor, ...]:
        tensorflow.print(f"Print:{self.name}", *args)
        return args
