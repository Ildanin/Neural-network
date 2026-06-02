import numpy as np
from random import sample
from typing import Iterable
from .datasetClass import DataSample, Dataset
from .progressBar import ProgressBar
from .gradientClass import Gradient
from .layerClass import FC, Layer, load_layers

class Network:
    def __init__(self, input_shape: int | tuple[int, ...], layers: list[Layer], weights_range: tuple[float, float] = (-1, 1), biases_range: tuple[float, float] = (-1, 1)) -> None:
        if len(layers) < 2:
            return None
        self.input_shape = input_shape
        self.layers = layers
        self.output_shape = layers[-1].size
        input_sizes = [input_shape] + [layer.output_shape for layer in self.layers]
        for layer, input_size in zip(self.layers, input_sizes):
            layer.set(input_size, weights_range, biases_range)
    
    def flush(self) -> None:
        for layer in self.layers:
            layer.flush()
    
    def copy(self) -> object:
        net = Network(self.input_shape, [])
        net.output_shape = self.output_shape
        net.layers = [layer.copy() for layer in self.layers]
        return net
    
    def save(self, filename: str) -> None:
        file = open(filename, 'w')
        file.write(str(self.input_shape))
        for layer in self.layers:
            file.write(str(layer) + '\n')
        for layer in self.layers:
            layer.save(file)
        file.close()
    
    def load(self, filename: str) -> None:
        file = open(filename, 'r')
        self.layers = load_layers(file)
        self.output_shape = self.layers[-1].size
        values = [float(x[:-1]) for x in file.readlines()]
        start = 0
        for i, layer in enumerate(self.layers[1:]):
            layer.weights = np.reshape(values[start : (start + layer.size * self.layers[i].size)], (layer.size, self.layers[i].size))
            start += layer.size * self.layers[i].size
            layer.biases = np.array(values[start : (start + layer.size)])
            start += layer.size
        self.layers.pop(0)
        print(file.readline())
        file.close()
    
    def process(self, data: np.ndarray) -> np.ndarray:
        result = data
        for layer in self.layers:
            result = layer.process(result)
        self.last_result = result
        return result
    
    def backprop(self, sample: DataSample) -> Gradient:
        "Returns the gradient for weightss and biaseses"
        self.process(sample.input_value)
        gradient = Gradient()
        "chain is a vector that represents the influence on the loss function for each neuron's output in a layer"
        chain = 2 * (self.last_result - sample.output_value) * self.layers[-1].derivative(self.last_result)
        gradient.insert(0, self.layers[-1].backprop(chain))
        
        for i in range(len(self.layers)-1, 0, -1):
            chain = self.layers[i-1].derivative(self.layers[i].input) * np.dot(self.layers[i].weights.T, chain)
            gradient.insert(0, self.layers[i-1].backprop(chain))
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
        for i, layer in enumerate(gradient):
            self.layers[i] += -learning_rate * layer
    
    def _unaverage_loss(self, answer: np.ndarray) -> float:
        return float(sum((self.last_result - answer)**2))
    
    def loss(self, answer: np.ndarray) -> float:
        return self._unaverage_loss(answer) / self.output_shape

    def validate(self, dataset: Dataset) -> float:
        loss = 0
        for data in dataset:
            self.process(data.input_value)
            loss += self._unaverage_loss(data.output_value)
        return loss / self.output_shape
    
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
                progress_bar(loss / (len(dataset) * self.output_shape))
    
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
                progress_bar(loss / (batchsize * self.output_shape))
    
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
                progress_bar(loss / (len(dataset) * self.output_shape))
    
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
                progress_bar(loss / (batchsize * self.output_shape))


def load(filename: str) -> Network:
    net = Network([])
    net.load(filename)
    return net