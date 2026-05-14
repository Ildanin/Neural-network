from numpy import ndarray
import numpy as np
from typing import Iterator, Self, TextIO
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
        self.input = data.copy()
        return self.function(np.dot(self.weight, data) + self.bias)
    
    def backprop(self, chain: ndarray):
        weight_gradient = self.input * np.atleast_2d(chain).T
        return Layer(weight_gradient, chain)

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

_layer_like = tuple[str, int, str] #type size activator
def initialize_layer(current_layer: _layer_like, next_layer: _layer_like, weight_range: tuple[float, float], bias_range: tuple[float, float]) -> Layer:
    return random_layer((current_layer[1], next_layer[1]), weight_range, bias_range, current_layer[2])

def initialize_layers(info: list[_layer_like], weight_range: tuple[float, float], bias_range: tuple[float, float]) -> list[Layer]:
    layers: list[Layer] = []
    for i in range(1, len(info)-1):
        layers.append(random_layer((info[i][1], info[i-1][1]), weight_range, bias_range, info[i][2]))
    layers.append(random_layer((info[-1][1], info[-2][1]), weight_range, bias_range, info[-1][2]))
    return layers

def load_info(file: TextIO) -> list[_layer_like]:
    info = []
    for line in file:
        if line == '\n':
            break
        layer_type, size, activator = line.split()
        info.append((layer_type, int(size), activator))
    return info