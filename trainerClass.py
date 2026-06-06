from .networkClass import Network
from .datasetClass import Dataset, DataSample
from .progressBar import ProgressBar
from .gradientClass import Gradient
from random import sample

class Trainer:
    def __init__(self, 
                 net: Network, 
                 dataset: Dataset, 
                 #learning_rate: float | list[float] = 1, 
                 learning_rate: float = 1, 
                 momentum_conservation: float = 1, 
                 display_progress = False,
                 validation_dataset: Dataset = Dataset()) -> None:
        self.net = net
        self.dataset = dataset
        self.learning_rate = learning_rate
        #if type(learning_rate) == list:
        #    self.learning_rate = learning_rate
        #else:
        #    self.learning_rate = [learning_rate for _ in range(len(self.net.layers))]
        self.momentum_conservation = momentum_conservation
        self.display_progress = display_progress
        self.validation_dataset = validation_dataset
    
    def vanilla(self, cycles: int) -> None:
        if self.validation_dataset:
            progress_bar = ProgressBar("Vanilla", cycles, True)
            for _ in range(cycles):
                gradient, loss = self.net.backprop_dataset_loss(self.dataset)
                self.net.modify(gradient, self.learning_rate / len(self.dataset))
                progress_bar(loss / (len(self.dataset) * self.net.output_size), self.net.validate(self.validation_dataset))
        elif self.display_progress:
            progress_bar = ProgressBar("Vanilla", cycles)
            for _ in range(cycles):
                gradient, loss = self.net.backprop_dataset_loss(self.dataset)
                self.net.modify(gradient, self.learning_rate / len(self.dataset))
                progress_bar(loss / (len(self.dataset) * self.net.output_size))
        else:
            for _ in range(cycles):
                gradient = self.net.backprop_dataset(self.dataset)
                self.net.modify(gradient, self.learning_rate / len(self.dataset))
    
    def stochastic(self, cycles: int = 1, batchsize: int = 1) -> None:
        if self.validation_dataset:
            progress_bar = ProgressBar("Stochastic", cycles, True)
            for _ in range(cycles):
                batch: list[DataSample] = sample(self.dataset, batchsize)
                gradient, loss = self.net.backprop_dataset_loss(batch)
                self.net.modify(gradient, self.learning_rate / batchsize)
                progress_bar(loss / (batchsize * self.net.output_size), self.net.validate(self.validation_dataset))
        if self.display_progress:
            progress_bar = ProgressBar("Stochastic", cycles)
            for _ in range(cycles):
                batch: list[DataSample] = sample(self.dataset, batchsize)
                gradient, loss = self.net.backprop_dataset_loss(batch)
                self.net.modify(gradient, self.learning_rate / batchsize)
                progress_bar(loss / (batchsize * self.net.output_size))
        else:
            for _ in range(cycles):
                batch: list[DataSample] = sample(self.dataset, batchsize)
                gradient = self.net.backprop_dataset(batch)
                self.net.modify(gradient, self.learning_rate / batchsize)
    
    def momentum(self, cycles: int = 1) -> None:
        if self.validation_dataset:
            progress_bar = ProgressBar("Momentum", cycles, True)
            momentum = Gradient()
            for _ in range(cycles):
                gradient, loss = self.net.backprop_dataset_loss(self.dataset)
                gradient += momentum * self.momentum_conservation
                momentum = gradient.copy()
                self.net.modify(gradient, self.learning_rate / len(self.dataset))
                progress_bar(loss / (len(self.dataset) * self.net.output_size), self.net.validate(self.validation_dataset))
        elif self.display_progress:
            progress_bar = ProgressBar("Momentum", cycles)
            momentum = Gradient()
            for _ in range(cycles):
                gradient, loss = self.net.backprop_dataset_loss(self.dataset)
                gradient += momentum * self.momentum_conservation
                momentum = gradient.copy()
                self.net.modify(gradient, self.learning_rate / len(self.dataset))
                progress_bar(loss / (len(self.dataset) * self.net.output_size))
        else:
            momentum = Gradient()
            for _ in range(cycles):
                gradient = self.net.backprop_dataset(self.dataset)
                gradient += momentum * self.momentum_conservation
                momentum = gradient.copy()
                self.net.modify(gradient, self.learning_rate / len(self.dataset))
    
    def stochastic_momentum(self, cycles: int = 1, batchsize: int = 1) -> None:
        if self.validation_dataset:
            progress_bar = ProgressBar("Stochastic + Momentum", cycles, True)
            momentum = Gradient()
            for _ in range(cycles):
                batch: list[DataSample] = sample(self.dataset, batchsize)
                gradient, loss = self.net.backprop_dataset_loss(batch)
                gradient += momentum * self.momentum_conservation
                momentum = gradient.copy()
                self.net.modify(gradient, self.learning_rate / batchsize)
                progress_bar(loss / (batchsize * self.net.output_size), self.net.validate(self.validation_dataset))
        elif self.display_progress:
            progress_bar = ProgressBar("Stochastic + Momentum", cycles)
            momentum = Gradient()
            for _ in range(cycles):
                batch: list[DataSample] = sample(self.dataset, batchsize)
                gradient, loss = self.net.backprop_dataset_loss(batch)
                gradient += momentum * self.momentum_conservation
                momentum = gradient.copy()
                self.net.modify(gradient, self.learning_rate / batchsize)
                progress_bar(loss / (batchsize * self.net.output_size))
        else:
            momentum = Gradient()
            for _ in range(cycles):
                batch: list[DataSample] = sample(self.dataset, batchsize)
                gradient = self.net.backprop_dataset(batch)
                gradient += momentum * self.momentum_conservation
                momentum = gradient.copy()
                self.net.modify(gradient, self.learning_rate / batchsize)