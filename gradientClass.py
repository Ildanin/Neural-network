from typing import Iterator, Iterable, Self
from numpy import ndarray

_layer_gradient = list[ndarray]
class Gradient(list):
    def __init__(self, layer_gradients: Iterable[_layer_gradient] = []) -> None:
        super().__init__(layer_gradients)
    
    def __iter__(self) -> Iterator[_layer_gradient]:
        return super().__iter__()

    def __add__(self, gradient: Self) -> Self:
        for self_layer_gradient, layer_gradient in zip(self, gradient):
            self_layer_gradient[0] += layer_gradient[0]
            self_layer_gradient[1] += layer_gradient[1]
        return self
    
    def __iadd__(self, gradient: Self) -> Self:
        return self.__add__(gradient)
    
    def __mul__(self, coef: float) -> Self:
        for layer_gradient in self:
            layer_gradient[0] *= coef
            layer_gradient[1] *= coef
        return self
    
    def __truediv__(self, coef: float) -> Self:
        for layer_gradient in self:
            layer_gradient[0] /= coef
            layer_gradient[1] /= coef
        return self
    
    def apply(self, learning_rate: list[float]) -> Self:
        for layer_gradient, coef in zip(self, learning_rate):
            layer_gradient[0] *= -coef
            layer_gradient[1] *= -coef
        return self
    
    def copy(self):
        layer_gradients = []
        for layer_gradient in self:
            layer_gradients.append([layer_gradient[0].copy(), layer_gradient[1].copy()])
        return Gradient(layer_gradients)