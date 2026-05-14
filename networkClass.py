import numpy as np
from random import sample
from typing import Iterable, Iterator
from .datasetClass import DataSample, Dataset
from .progressBar import ProgressBar
from .gradientClass import Gradient
from .layerClass import Layer, random_layer

class Network:
    def __init__(self, info: list[int], activator: str, normalizer: str | None = None, 
                 weight_range: tuple[float, float] = (-1, 1), bias_range: tuple[float, float] = (-1, 1)) -> None:
        if len(info) < 2:
            return None
        self.info = info
        self.activator = activator
        if normalizer == None:
            self.normalizer = activator
        else:
            self.normalizer = normalizer
        self.weight_range = weight_range
        self.bias_range = bias_range
        self.layers: list[Layer] = []
        for i in range(1, len(self.info)-1):
            self.layers.append(random_layer((self.info[i], self.info[i-1]), self.weight_range, self.bias_range, self.activator))
        self.layers.append(random_layer((self.info[-1], self.info[-2]), self.weight_range, self.bias_range, self.normalizer))
    

    def flush(self) -> None:
        for layer in self.layers:
            layer.flush(self.weight_range, self.bias_range)
    
    def copy(self) -> object:
        net = Network(self.info, self.activator, self.normalizer)
        net.layers = [layer.copy() for layer in self.layers]
        return net
    
    def save(self, filename: str) -> None:
        file = open(filename, 'w')
        file.write(' '.join([str(layer_info) for layer_info in self.info]))
        file.write('\n' + self.activator)
        file.write('\n' + self.normalizer)
        for layer in self.layers:
            file.writelines(['\n' + str(coef) for coef in layer.weight.flatten()])
            file.writelines(['\n' + str(coef) for coef in layer.bias.flatten()])
        file.close()
    
    def load(self, filename: str) -> None:
        file = open(filename, 'r')
        self.info = [int(n) for n in file.readline().split()]
        self.activator = file.readline()[:-1]
        self.normalizer = file.readline()[:-1]
        self.layers = []
        values = [float(x[:-1]) for x in file.readlines()]
        start = 0
        for i in range(1, len(self.info)):
            weight = np.reshape(values[start : (start + self.info[i] * self.info[i-1])], (self.info[i], self.info[i-1]))
            start += self.info[i] * self.info[i-1]
            bias = np.array(values[start : (start + self.info[i])])
            start += self.info[i]
            if i != len(self.info)-1:
                self.layers.append(Layer(weight, bias, self.activator))
            else:
                self.layers.append(Layer(weight, bias, self.normalizer))
        file.close()
    
    def process(self, data: np.ndarray) -> np.ndarray:
        result = data
        for layer in self.layers:
            result = layer.process(result)
        self.last_result = result
        return result
    
    def backprop(self, sample: DataSample) -> Gradient:
        "Returns the gradient for weights and biases"
        self.process(sample.input_value)
        gradient = Gradient()
        "chain is a vector that represents the influence on the loss function for each neuron's output in a layer"
        chain = 2 * (self.last_result - sample.output_value) * self.layers[-1].derivative(self.last_result)
        gradient.insert(0, self.layers[-1].backprop(chain))
        
        for i in range(len(self.info)-2, 0, -1):
            chain = self.layers[i-1].derivative(self.layers[i].input) * np.dot(self.layers[i].weight.T, chain)
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
        return self._unaverage_loss(answer) / self.info[-1][1]

    def validate(self, dataset: Dataset) -> float:
        loss = 0
        for data in dataset:
            self.process(data.input_value)
            loss += self._unaverage_loss(data.output_value)
        return loss / self.info[-1][1]
    
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
                progress_bar(loss / (len(dataset) * self.info[-1]))
    
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
                progress_bar(loss / (batchsize * self.info[-1]))
    
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
                progress_bar(loss / (len(dataset) * self.info[-1]))
    
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
                progress_bar(loss / (batchsize * self.info[-1]))


def load(filename: str) -> Network:
    net = Network([0], 'linear')
    net.load(filename)
    return net