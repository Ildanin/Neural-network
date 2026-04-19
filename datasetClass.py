from numpy import ndarray
from typing import Iterator, overload

class DataSample:
    def __init__(self, input_value: ndarray, output_value: ndarray) -> None:
        self.input_value = input_value
        self.output_value = output_value

class Dataset():
    def __init__(self, input_values: list[ndarray] = [], output_values: list[ndarray] = []) -> None:
        self.samples = []
        for input_value, output_value in zip(input_values, output_values):
            self.samples.append(DataSample(input_value, output_value))
    
    def __iter__(self) -> Iterator[DataSample]:
        return iter(self.samples)
    
    def __len__(self) -> int:
        return len(self.samples)
    
    @overload
    def __getitem__(self, key: int) -> DataSample: ...
    @overload
    def __getitem__(self, key: slice) -> list[DataSample]: ...
    def __getitem__(self, key: int | slice) -> DataSample | list[DataSample]:
        return self.samples[key]
    
    def append(self, input_value: ndarray, output_value: ndarray) -> None:
        self.samples.append(DataSample(input_value, output_value))