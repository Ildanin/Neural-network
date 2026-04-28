from typing import Iterator, Iterable, Self
from .layerClass import Layer

class Gradient(list):
    def __init__(self, layers: Iterable[Layer] = []) -> None:
        super().__init__(layers)
    
    def __iter__(self) -> Iterator[Layer]:
        return super().__iter__()

    def __add__(self, gradient: Self) -> Self:
        for i, layer in enumerate(gradient):
            self[i] += layer
        return self
    
    def __iadd__(self, gradient: Self) -> Self:
        return self.__add__(gradient)
    
    def __mul__(self, coef: float) -> Self:
        for i, layer in enumerate(self):
            self[i] = coef * layer
        return self
    
    def __rmul__(self, coef: float) -> list:
        for i, layer in enumerate(self):
            self[i] = coef * layer
        return self
    
    def copy(self):
        return Gradient(self)