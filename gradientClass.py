from typing import Iterator, Iterable, Self
from numpy import ndarray

_layer_gradient = list[ndarray]
class Gradient(list):
    def __init__(self, layer_gradients: Iterable[_layer_gradient] = []) -> None:
        super().__init__(layer_gradients)
    
    def __iter__(self) -> Iterator[_layer_gradient]:
        return super().__iter__()

    def __add__(self, gradient: Self) -> Self:
        for i, layer_gradient in enumerate(gradient):
            self[i][0] += layer_gradient[0]
            self[i][1] += layer_gradient[1]
        return self
    
    def __iadd__(self, gradient: Self) -> Self:
        return self.__add__(gradient)
    
    def __mul__(self, coef: float) -> Self:
        for i, layer_gradient in enumerate(self):
            self[i][0] = coef * layer_gradient[0]
            self[i][1] = coef * layer_gradient[1]
        return self
    
    def copy(self):
        return Gradient(self)