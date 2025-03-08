from tfdatacompose.datasetoperation import DatasetOperation
from tensorflow import Dataset


class Sample(DatasetOperation):
    def __init__(self, sample_rate: float):
        self.sample_rate = sample_rate

    def apply(self, dataset: Dataset) -> Dataset:
        count = int(dataset.reduce(0, lambda x, _: x + 1))
        sample_size = round(count * self.sample_rate)
        return dataset.take(sample_size)
