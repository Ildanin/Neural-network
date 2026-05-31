from numpy import ndarray
import numpy as np
from typing import Iterator, Self, TextIO
from .utiles import random_array
from .activators import Activators

class FC:
    def __init__(self, size: int, activator: str = "linear") -> None:
        self.size = size
        self.activator = activator
        self.function, self.derivative = Activators[activator]
    
    def set(self, input_size: int, weight_range: tuple[float, float], bias_range: tuple[float, float]) -> None:
        self.weight_range = weight_range
        self.bias_range = bias_range
        self.weight = random_array(*weight_range, (self.size, input_size))
        self.bias = random_array(*bias_range, self.size)
    
    def __str__(self) -> str:
        return f"FC {self.size} {self.activator} {self.weight_range[0]} {self.weight_range[1]} {self.bias_range[0]} {self.bias_range[1]}"

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
        dummy = FC(self.size)
        dummy.weight = weight_gradient
        dummy.bias = chain
        return dummy

    def copy(self):
        dummy = FC(self.size, self.activator)
        dummy.weight = self.weight.copy()
        dummy.bias = self.bias.copy()
        return dummy
    
    def get_input_size(self) -> int:
        return self.weight.shape[1]
    
    def flush(self) -> None:
        self.weight = random_array(*self.weight_range, self.weight.size)
        self.bias = random_array(*self.bias_range, self.bias.size)

class CN(FC):
    pass

class PL(FC):
    pass

Layer = FC | CN | PL

def load_layers(file: TextIO) -> list[Layer]:
    layers: list[Layer] = []
    for line in file:
        if line == '\n':
            break
        layer_type, size, activator, weight_range1, weight_range2, bias_range1, bias_range2 = line.split()
        if layer_type == 'FC':
            layers.append(FC(int(size), activator))
            layers[-1].weight_range = float(weight_range1), float(weight_range2)
            layers[-1].bias_range = float(bias_range1), float(bias_range2)
        else: raise ValueError("Work in progress")
    return layers