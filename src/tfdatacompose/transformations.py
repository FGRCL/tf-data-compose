from tfdatacompose.datasetoperation import DatasetOperation


class Subsample(NumpyTransformOperation):
    def __init__(
        self, out_type: Union[DType, Tuple[DType, ...]], seed: int, sample_rate: float
    ):
        super().__init__(out_type)
        self.random_generator = default_rng(seed)
        self.sample_rate = sample_rate

    def transform(self, inputs_windows: ndarray, output_windows: ndarray) -> Any:
        n_elements = inputs_windows.shape[0]
        sample_size = int(n_elements * self.sample_rate)
        sample_idx = self.random_generator.choice(range(n_elements), sample_size, False)

        return inputs_windows[sample_idx], output_windows[sample_idx]
