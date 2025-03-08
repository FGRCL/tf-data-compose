from tfdatacompose.map.map import Map
from tensorflow import reduce_mean, reduce_std, Tensor


class StandardScaling(Map):
    """
    TODO is this accurate scaling. should it use the mean and std of the entire dataset
    """

    def __init__(self, axis):
        self.axis = axis

    def transform(self, input_window: Tensor, pressures: Tensor) -> (Tensor, Tensor):
        mu = reduce_mean(input_window, self.axis, True)
        sigma = reduce_std(input_window, self.axis, True)
        scaled = (input_window - mu) / sigma
        return scaled, pressures
