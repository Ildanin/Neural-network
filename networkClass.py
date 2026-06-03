import numpy as np
from random import sample
from typing import Iterable
from math import prod
from .datasetClass import DataSample, Dataset
from .progressBar import ProgressBar
from .gradientClass import Gradient
from .layerClass import Layer, load_layers

class Network:
    def __init__(self, input_shape: tuple[int, ...], layers: list[Layer], weights_range: tuple[float, float] = (-1, 1), biases_range: tuple[float, float] = (-1, 1)) -> None:
        if len(layers) == 0:
            return None
        self.input_shape = input_shape
        self.layers = layers
        self.output_size = prod(self.layers[-1].output_shape)
        layers[0].set(self.input_shape, weights_range, biases_range)
        for i, layer in enumerate(layers[1:]):
            layer.set(layers[i].output_shape, weights_range, biases_range)
    
    def copy(self) -> object:
        net = Network(self.input_shape, [])
        net.layers = [layer.copy() for layer in self.layers]
        return net
    
    def save(self, filename: str) -> None:
        file = open(filename, 'w')
        file.write(' '.join(str(d) for d in self.input_shape) + '\n')
        for layer in self.layers:
            file.write(str(layer) + '\n')
        for layer in self.layers:
            layer.save(file)
        file.close()
    
    def process(self, data: np.ndarray) -> np.ndarray:
        result = data
        for layer in self.layers:
            result = layer.process(result)
        self.last_result = result
        return result
    
    def backprop(self, sample: DataSample) -> Gradient:
        "Returns gradient for weightss and biaseses"
        self.process(sample.input_value)
        gradient = Gradient()
        "chain is a vector that represents the influence on the loss function for each neuron's output in a layer"        
        chain = 2 * (self.last_result - sample.output_value)
        for layer in reversed(self.layers[1:]):
            layer_gradient, chain = layer.backprop(chain)
            chain = layer.update_chain(chain)
            gradient.insert(0, layer_gradient)
        
        layer_gradient, chain = self.layers[0].backprop(chain)
        gradient.insert(0, layer_gradient)
        return gradient
    
    def backprop_dataset(self, dataset: Iterable[DataSample]) -> Gradient:
        gradient = Gradient()
        for data in dataset:
            gradient = self.backprop(data) + gradient
        return gradient
    
    def backprop_dataset_loss(self, dataset: Iterable[DataSample]) -> tuple[Gradient, float]:
        gradient = Gradient()
        loss = 0
        for data in dataset:
            gradient = self.backprop(data) + gradient
            loss += self._unaverage_loss(data.output_value)
        return gradient, loss
    
    def modify(self, gradient: Gradient, learning_rate: float) -> None:
        gradient *= -learning_rate
        for layer, layer_gradient in zip(self.layers, gradient):
            layer += layer_gradient
    
    def _unaverage_loss(self, answer: np.ndarray) -> float:
        return float(sum((self.last_result - answer)**2))
    
    def loss(self, answer: np.ndarray) -> float:
        return self._unaverage_loss(answer) / self.output_size
    
    def validate(self, dataset: Dataset) -> float:
        loss = 0
        for data in dataset:
            self.process(data.input_value)
            loss += self._unaverage_loss(data.output_value)
        return loss / self.output_size
    
    def train_vanilla(self, dataset: Dataset, 
                      learning_rate: float, 
                      cycles: int = 1, 
                      display_progress: bool = False) -> None:
        if display_progress == False:
            for _ in range(cycles):
                gradient = self.backprop_dataset(dataset)
                self.modify(gradient, learning_rate / len(dataset))
        else:
            progress_bar = ProgressBar("Vanilla", cycles)
            for i in range(cycles):
                gradient, loss = self.backprop_dataset_loss(dataset)
                self.modify(gradient, learning_rate / len(dataset))
                progress_bar(loss / (len(dataset) * self.output_size))
    
    def train_stochastic(self, dataset: Dataset, 
                        learning_rate: float, 
                        cycles: int = 1, 
                        batchsize: int = 1, display_progress: bool = False) -> None:
        if display_progress == False:
            for _ in range(cycles):
                batch: list[DataSample] = sample(dataset, batchsize)
                gradient = self.backprop_dataset(batch)
                self.modify(gradient, learning_rate / batchsize)
        else:
            progress_bar = ProgressBar("Stochastic", cycles)
            for i in range(cycles):
                batch: list[DataSample] = sample(dataset, batchsize)
                gradient, loss = self.backprop_dataset_loss(batch)
                self.modify(gradient, learning_rate / batchsize)
                progress_bar(loss / (batchsize * self.output_size))
    
    def train_momentum(self, dataset: Dataset,
                       learning_rate: float, momentum_conservation: float, 
                       cycles: int = 1, 
                       display_progress: bool = False) -> None:
        if display_progress == False:
            momentum = Gradient()
            for _ in range(cycles):
                gradient = self.backprop_dataset(dataset)
                gradient += momentum * momentum_conservation
                momentum = gradient.copy()
                self.modify(gradient, learning_rate / len(dataset))
        else:
            progress_bar = ProgressBar("Momentum", cycles)
            momentum = Gradient()
            for i in range(cycles):
                gradient, loss = self.backprop_dataset_loss(dataset)
                gradient += momentum * momentum_conservation
                momentum = gradient.copy()
                self.modify(gradient, learning_rate / len(dataset))
                progress_bar(loss / (len(dataset) * self.output_size))
    
    def train_stochastic_momentum(self, dataset: Dataset,
                                  learning_rate: float, momentum_conservation: float, 
                                  cycles: int = 1, batchsize: int = 1, 
                                  display_progress: bool = False) -> None:
        if display_progress == False:
            momentum = Gradient()
            for _ in range(cycles):
                batch: list[DataSample] = sample(dataset, batchsize)
                gradient = self.backprop_dataset(batch)
                gradient += momentum * momentum_conservation
                momentum = gradient.copy()
                self.modify(gradient, learning_rate / batchsize)
        else:
            progress_bar = ProgressBar("Stochastic + Momentum", cycles)
            momentum = Gradient()
            for i in range(cycles):
                batch: list[DataSample] = sample(dataset, batchsize)
                gradient, loss = self.backprop_dataset_loss(batch)
                gradient += momentum * momentum_conservation
                momentum = gradient.copy()
                self.modify(gradient, learning_rate / batchsize)
                progress_bar(loss / (batchsize * self.output_size))


def load(filename: str) -> Network:
    file = open(filename, 'r')
    input_shape = tuple(int(d) for d in file.readline().split())
    layers = load_layers(file)
    net = Network(input_shape, layers)
    for layer in net.layers:
        layer.load(file)
    file.close()
    return net