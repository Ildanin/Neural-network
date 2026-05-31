from collections.abc import Iterator
from numpy import ndarray
from typing import overload

class DataSample:
    def __init__(self, input_value: ndarray, output_value: ndarray) -> None:
        self.input_value = input_value
        self.output_value = output_value

class Dataset(list):
    def __init__(self, input_values: list[ndarray] = [], output_values: list[ndarray] = []) -> None:
        super().__init__(DataSample(input_value, output_value) for input_value, output_value in zip(input_values, output_values))
    
    def __iter__(self) -> Iterator[DataSample]:
        return super().__iter__()
    
    @overload
    def __getitem__(self, key: int) -> DataSample: ...
    @overload
    def __getitem__(self, key: slice) -> list[DataSample]: ...
    def __getitem__(self, key: int | slice) -> DataSample | list[DataSample]:
        return super().__getitem__(key)