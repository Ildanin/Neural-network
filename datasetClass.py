from numpy import ndarray

class DataSample:
    def __init__(self, input_value: ndarray, output_value: ndarray) -> None:
        self.input_value = input_value
        self.output_value = output_value

class Dataset(list):
    def __init__(self, input_values: list[ndarray] = [], output_values: list[ndarray] = []) -> None:
        super().__init__(DataSample(input_value, output_value) for input_value, output_value in zip(input_values, output_values))