from numpy import ndarray
import numpy as np
from typing import Iterator, Self
from .utiles import random_array
from .activators import Activators

class Layer:
    def __init__(self, weight: ndarray, bias: ndarray, activator: str = "linear") -> None:
        self.weight = weight.copy()
        self.bias = bias.copy()
        self.activator = activator
        self.function, self.derivative = Activators[activator]
    
    def __iter__(self) -> Iterator:
        return iter((self.weight, self.bias))

    def __add__(self, layer: Self) -> Self:
        self.weight += layer.weight
        self.bias += layer.bias
        return self
    
    def __mul__(self, coef: float) -> Self:
        self.weight = self.weight * coef
        self.bias = self.bias * coef
        return self
    __rmul__ = __mul__
    
    def process(self, data: ndarray) -> ndarray:
        return self.function(np.dot(self.weight, data) + self.bias)
    
    def copy(self):
        return Layer(self.weight, self.bias, self.activator)
    
    def glue(self) -> ndarray:
        return np.append(self.weight, np.reshape(self.bias, (-1, 1)), 1)
    
    def flush(self, weight_range: tuple[float, float], bias_range: tuple[float, float]) -> None:
        self.weight = random_array(*weight_range, self.weight.size)
        self.bias = random_array(*bias_range, self.bias.size)


def random_layer(size: tuple[int, int], weight_range: tuple[float, float], bias_range: tuple[float, float], activator: str) -> Layer:
    weight = random_array(*weight_range, size)
    bias = random_array(*bias_range, size[0])
    return Layer(weight, bias, activator)