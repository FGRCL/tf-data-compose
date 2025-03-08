from tfdatacompose.datasetoperation import DatasetOperation
from tensorflow import Dataset


class Shuffle(DatasetOperation):
    def __init__(self, buffer_size=None, seed=None, reshuffle_each_iteration=None):
        self.buffer_size = buffer_size
        self.seed = seed
        self.reshuffle_each_iteration = reshuffle_each_iteration

    def apply(self, dataset: Dataset) -> Dataset:
        if self.buffer_size is not None:
            self.buffer_size = int(dataset.reduce(0, lambda x, _: x + 1))
        return dataset.shuffle(
            self.buffer_size,
            seed=self.seed,
            reshuffle_iteration_iteration=self.reshuffle_each_iteration,
            name=self.__class__.__name__,
        )
