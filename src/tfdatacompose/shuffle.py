from tfdatacompose.datasetoperation import DatasetOperation
from tensorflow import Dataset


class Shuffle(DatasetOperation):
    """
    Dataset shuffling operation.

    Wraps the `Tensorflow shuffle`_ operation on the dataset.
    Shuffles the elements of the dataset bt taking `buffer_size` at a time.
    If `buffer_size` is unspecified, it will default to the number of elements in the dataset.
    This might be very slow on large datasets.

    :param buffer_size: the size of the buffer used to shuffle elements. Defaults to the number of elements in the dataset
    :param seed:
    :param reshuffle_each_iteration:

    .. _Tensorflow Skip: https://www.tensorflow.org/api_docs/python/tf/data/Dataset#shuffle
    """

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
