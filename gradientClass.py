from typing import Iterator, Self
from .layerClass import Layer

class Gradient:
    def __init__(self, layers: list[Layer] = []) -> None:
        self.layers = layers.copy()
    
    def __iter__(self) -> Iterator[Layer]:
        return iter(self.layers)
    
    def __getitem__(self, index: int) -> Layer:
        return self.layers[index]
    
    def __setitem__(self, index: int, value: Layer) -> None:
        self.layers[index] = value
    
    def __len__(self) -> int:
        return len(self.layers)
    
    def __add__(self, gradient: Self) -> Self:
        for i, layer in enumerate(gradient):
            self[i] += layer
        return self
    
    def __mul__(self, coef: float) -> Self:
        for i, layer in enumerate(self):
            self[i] = coef * layer
        return self
    __rmul__ = __mul__
    
    def insert(self, index: int, layer: Layer) -> None:
        self.layers.insert(index, layer)
    
    def copy(self):
        return Gradient(self.layers)