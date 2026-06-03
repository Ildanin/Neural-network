from numpy import ndarray
import numpy as np
from scipy.signal import correlate2d
from typing import Iterator, Self, TextIO
from itertools import product
from math import prod
from .utiles import random_array
from .activators import Activators

class FC:
    def __init__(self, size: int, activator: str = "linear") -> None:
        self.size = size
        self.output_shape = (self.size,)
        self.activator = activator
        self.function, self.derivative = Activators[activator]
    
    def set(self, input_shape: tuple[int, ...], weights_range: tuple[float, float], biases_range: tuple[float, float]) -> None:
        self.input_shape = input_shape
        self.weights = random_array(*weights_range, (self.size, prod(input_shape)))
        self.biases = random_array(*biases_range, self.size)
    
    def __str__(self) -> str:
        return f"FC {self.size} {self.activator}"
    
    def __iter__(self) -> Iterator:
        return iter((self.weights, self.biases))
    
    def __add__(self, layer: Self) -> Self:
        self.weights += layer.weights
        self.biases += layer.biases
        return self
    
    def __mul__(self, coef: float) -> Self:
        self.weights = self.weights * coef
        self.biases = self.biases * coef
        return self
    __rmul__ = __mul__
    
    def process(self, data: ndarray) -> ndarray:
        if len(data.shape) != 1:
            data = data.flatten()
        self.input = data.copy()            
        return self.function(np.dot(self.weights, data) + self.biases)
    
    def backprop(self, chain: ndarray):
        weights_gradient = self.input * np.atleast_2d(chain).T
        dummy = FC(self.size)
        dummy.weights = weights_gradient
        dummy.biases = chain
        return dummy
    
    def copy(self):
        dummy = FC(self.size, self.activator)
        dummy.weights = self.weights.copy()
        dummy.biases = self.biases.copy()
        return dummy
    
    def save(self, file: TextIO) -> None:
        file.writelines(['\n' + str(weight) for weight in self.weights.flatten()])
        file.writelines(['\n' + str(bias) for bias in self.biases.flatten()])
    
    def load(self, file: TextIO) -> None:
        self.weights = np.reshape([float(file.readline()[:-1]) for _ in range(self.weights.size)], self.weights.shape)
        self.biases = np.array([float(file.readline()[:-1]) for _ in range(self.biases.size)])

class CN():
    def __init__(self, size: int, kernel_size: int, activator: str = "linear") -> None:
        self.size = size
        self.kernel_size = kernel_size
        self.activator = activator
        self.function, self.derivative = Activators[activator]
    
    def set(self, input_shape: tuple[int, ...], weights_range: tuple[float, float], biases_range: tuple[float, float]) -> None:
        input_depth, input_height, input_width = input_shape
        self.depth = input_depth
        self.input_shape = input_shape
        self.output_shape = (self.size, input_height - self.kernel_size + 1, input_width - self.kernel_size + 1)
        self.kernels = random_array(*weights_range, (self.size, input_depth, self.kernel_size, self.kernel_size))
        self.biases = random_array(*biases_range, self.output_shape)
    
    def __str__(self) -> str:
        return f"CN {self.size} {self.kernel_size} {self.activator}"
    
    def process(self, data: ndarray) -> ndarray:
        self.input = data.copy()
        self.output = self.biases.copy()
        for i, j in product(range(self.size), range(self.depth)):
            self.output[i] += correlate2d(data[j], self.kernels[i, j], mode="valid")
        return self.function(self.output)
    
    def backprop(self, chain: ndarray):
        pass
    
    def copy(self):
        dummy = CN(self.size, self.kernel_size, self.activator)
        dummy.kernels = self.kernels.copy()
        dummy.biases = self.biases.copy()
        return dummy
    
    def save(self, file: TextIO) -> None:
        file.writelines(['\n' + str(weight) for weight in self.kernels.flatten()])
        file.writelines(['\n' + str(bias) for bias in self.biases.flatten()])
    
    def load(self, file: TextIO) -> None:
        self.kernels = np.reshape([float(file.readline()[:-1]) for _ in range(self.kernels.size)], self.kernels.shape)
        self.biases = np.reshape([float(file.readline()[:-1]) for _ in range(self.biases.size)], self.biases.shape)


class PL(FC):
    pass

Layer = FC | CN | PL

def load_layers(file: TextIO) -> list[Layer]:
    layers: list[Layer] = []
    for line in file:
        if line == '\n':
            break
        layer_type, info = line.split(maxsplit=1)
        if layer_type == 'FC':
            size, activator = info.split()
            layers.append(FC(int(size), activator))
        elif layer_type == 'CN':
            size, kernel_size, activator = info.split()
            layers.append(CN(int(size), int(kernel_size), activator))
        else: raise ValueError("Invalid layer type")
    return layers