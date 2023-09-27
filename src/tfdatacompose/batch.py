from tensorflow.python.data import AUTOTUNE, Dataset

from src.tfdatacompose.datasetoperation import DatasetOperation


class Batch(DatasetOperation):
    def __init__(self, batch_size: int):
        self.batch_size = batch_size

    def apply(self, dataset: Dataset) -> Dataset:
        return dataset.batch(
            self.batch_size,
            num_parallel_calls=AUTOTUNE,
            deterministic=False,
            name=self.__class__.__name__,
        )
