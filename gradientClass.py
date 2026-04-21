from numpy import ndarray
from typing import Iterator, Self 

class Gradient:
    def __init__(self, layers: list[tuple[ndarray, ndarray]] = []) -> None:
        self.layers = layers.copy()
    
    def __iter__(self) -> Iterator[tuple[ndarray, ndarray]]:
        return iter(self.layers)
    
    def __getitem__(self, index: int) -> tuple[ndarray, ndarray]:
        return self.layers[index]
    
    def __setitem__(self, index: int, value: tuple[ndarray, ndarray]) -> None:
        self.layers[index] = value
    
    def __len__(self) -> int:
        return len(self.layers)

    def __add__(self, gradient: Self) -> Self:
        for i, layer in enumerate(gradient):
            self[i] = (self[i][0] + layer[0], self[i][1] + layer[1])
        return self
    
    def __mul__(self, coef: float) -> Self:
        for i, layer in enumerate(self):
            self[i] = (coef * layer[0], coef * layer[1])
        return self
    __rmul__ = __mul__

    def insert(self, index: int, weights: ndarray, biases: ndarray) -> None:
        self.layers.insert(index, (weights, biases))
    
    def copy(self):
        return Gradient(self.layers)