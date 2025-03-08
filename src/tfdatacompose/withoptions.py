from tfdatacompose.datasetoperation import DatasetOperation
from tensorflow import Options, Dataset


class WithOptions(DatasetOperation):
    def __init__(self, options: Options):
        self.options = options

    def apply(self, dataset: Dataset) -> Dataset:
        return dataset.with_options(self.options)
