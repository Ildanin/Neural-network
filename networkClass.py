import numpy as np
from time import perf_counter
from .utiles import random_array, gradient_product, gradient_sum
from .activators import Activators

class Network:
    def __init__(self, info: list[int], activator: str, normalizer: str | None = None, 
                 factor_range: tuple[float, float] = (-1, 1), bias_range: tuple[float, float] = (-1, 1)) -> None:
        self.info = info
        self.activator = activator
        self.func, self.func_derivative = Activators[activator]
        if normalizer == None:
            self.norm, self.norm_derivative = Activators[activator]
            self.normalizer = normalizer = activator
        else:
            self.norm, self.norm_derivative = Activators[normalizer]
            self.normalizer = normalizer
        self.factor_range = factor_range
        self.bias_range = bias_range
        self.weights: list[np.ndarray] = []
        self.biases: list[np.ndarray] = []
        #self.layer_dictionary = [self.layer_definer(i) for i in range(sum(info[1:]))]
        for i in range(1, len(info)):
            self.weights.append(random_array(factor_range[0], factor_range[1], size=(info[i], info[i-1])))
            self.biases.append(random_array(bias_range[0], bias_range[1], size=(info[i])))
    
    def __str__(self, layer_ID: int | None = None, separator: str = '==========================') -> str:
        "Layer ID 0 means the first layer after the input layer"
        if layer_ID == None:
            layers = [np.append(weight, np.reshape(bias, (-1, 1)), 1) for weight, bias in zip(self.weights, self.biases)]
            string = ''
            for i in range(len(self.info)-1):
                string += f'{separator} \n'
                string += f'Layer {i} \n'
                string += f'{layers[i]} \n'
            string += separator
            return(string)
        elif 0 <= layer_ID <= len(self.info) - 2:
            layer = np.append(self.weights[layer_ID], np.reshape(self.biases[layer_ID], (-1, 1)), 1)
            string =  f'separator \n'
            string += f'Layer {layer_ID} \n'
            string += f'{layer} \n'
            string += separator
            return(string)
        else:
            raise ValueError(f"Network has no layer with such ID({layer_ID})")
    
    def flush(self) -> None:
        self.weights: list[np.ndarray] = []
        self.biases: list[np.ndarray] = []
        for i in range(1, len(self.info)):
            self.weights.append(random_array(self.factor_range[0], self.factor_range[1], size=(self.info[i], self.info[i-1])))
            self.biases.append(random_array(self.bias_range[0], self.bias_range[1], size=(self.info[i])))

    def copy(self) -> object:
        net = Network(self.info, self.activator, self.normalizer)
        net.weights = [layer_weights.copy() for layer_weights in self.weights]
        net.biases = [layer_biases.copy() for layer_biases in self.biases]
        return(net)

    def save(self, filename: str) -> None:
        file = open(filename, 'w')
        file.write(' '.join([str(layer_info) for layer_info in self.info]))
        file.write('\n' + self.activator)
        file.write('\n' + self.normalizer)
        for weights, biases in zip(self.weights, self.biases):
            file.writelines(['\n' + str(weight) for weight in weights.flatten()])
            file.writelines(['\n' + str(bias) for bias in biases.flatten()])
        file.close()
    
    def load(self, filename: str) -> None:
        file = open(filename, 'r')
        self.info = [int(n) for n in file.readline().split()]
        self.activator = file.readline()[:-1]
        self.normalizer = file.readline()[:-1]
        self.func, self.func_derivative = Activators[self.activator]
        self.norm, self.norm_derivative = Activators[self.normalizer]
        self.weights: list[np.ndarray] = []
        self.biases: list[np.ndarray] = []
        values = [float(x[:-1]) for x in file.readlines()]
        start = 0
        for i in range(1, len(self.info)):
            self.weights.append(np.reshape(values[start : (start + self.info[i] * self.info[i-1])], (self.info[i], self.info[i-1])))
            start += self.info[i] * self.info[i-1]
            self.biases.append(np.array(values[start : (start + self.info[i])]))
            start += self.info[i]
        file.close()
    
    def layer_definer(self, neuron_number: int) -> tuple[int, int]: 
        'returns layer number(first after input layer is 0) and neuron number(in that layer)'
        counter = 1
        while neuron_number + 1 - self.info[counter] > 0:
            neuron_number -= self.info[counter]
            counter += 1
        return(counter-1, neuron_number)
    
    def process(self, data: np.ndarray | list) -> np.ndarray:
        self.layer_results = [np.array(data)]
        for weight, bias in zip(self.weights[:-1], self.biases[:-1]):
            self.layer_results.append(self.func(np.dot(weight, self.layer_results[-1]) + bias))
        self.layer_results.append(self.norm(np.dot(self.weights[-1], self.layer_results[-1]) + self.biases[-1]))
        return(self.layer_results[-1])
    
    def backpropagate(self, data: np.ndarray | list, answer: np.ndarray | list) -> list[tuple[np.ndarray, np.ndarray]]:
        "Returns the list of gradient for weights and biases"
        self.process(data)
        gradient: list[tuple[np.ndarray, np.ndarray]] = []
        "chain is a vector that represents the influence on the cost function for each neuron's output in a layer"
        chain = 2 * (self.layer_results[-1] - np.array(answer)) * self.norm_derivative(self.layer_results[-1])
        weight_gradient = self.layer_results[-2] * np.atleast_2d(chain).T
        bias_gradient = chain
        gradient.insert(0, (weight_gradient, bias_gradient))
        
        for i in range(len(self.info)-2, 0, -1):
            chain = np.array(self.func_derivative(self.layer_results[i]) * np.dot(self.weights[i].T, chain))
            weight_gradient = self.layer_results[i-1] * np.atleast_2d(chain).T
            bias_gradient = chain
            gradient.insert(0, (weight_gradient, bias_gradient))
        return(gradient)
    
    def modify(self, gradient: list[tuple[np.ndarray, np.ndarray]], learning_rate: float) -> None:
        for i, layer in enumerate(gradient):
            self.weights[i] += -learning_rate * layer[0]
            self.biases[i]  += -learning_rate * layer[1]
    
    def cost(self, answer: np.ndarray | list) -> float:
        return(float(sum((self.layer_results[-1] - np.array(answer))**2)))
    
    def train_vanilla(self, dataset: list[np.ndarray] | list[list], 
                      answerset: list[np.ndarray] | list[list], 
                      learning_rate: float, 
                      cycles: int = 1, 
                      display_progress: bool = False) -> None:
        if display_progress == False:
            data_size = min(len(dataset), len(answerset))
            for _ in range(cycles):
                gradient = self.backpropagate(dataset[0], answerset[0])
                for data, answer in zip(dataset[1:], answerset[1:]):
                    gradient = gradient_sum(gradient, self.backpropagate(data, answer))
                self.modify(gradient, learning_rate/data_size)
        else:
            start_time = perf_counter()
            data_size = min(len(dataset), len(answerset))
            for i in range(1, cycles+1):
                gradient = self.backpropagate(dataset[0], answerset[0])
                cost = self.cost(answerset[0])
                for data, answer in zip(dataset[1:], answerset[1:]):
                    gradient = gradient_sum(gradient, self.backpropagate(data, answer))
                    cost += self.cost(answer)
                self.modify(gradient, learning_rate/data_size)
                runtime = perf_counter() - start_time
                print(f'Cycles finished: {i}/{cycles} | Cycle cost: {round(cost / data_size, 3)} | Runtime: {round(runtime, 1)}/{round(runtime * cycles/i, 1)}')

    def train_stochastic(self, dataset: list[np.ndarray] | list[list], 
                        answerset: list[np.ndarray] | list[list], 
                        learning_rate: float, 
                        cycles: int = 1, 
                        batchsize: int = 1, display_progress: bool = False) -> None:
        if display_progress == False:
            data_size = min(len(dataset), len(answerset))
            for _ in range(cycles):
                rand = np.random.randint(0, data_size)
                gradient = self.backpropagate(dataset[rand], answerset[rand])
                for _ in range(batchsize-1):
                    rand = np.random.randint(0, data_size)
                    gradient = gradient_sum(gradient, self.backpropagate(dataset[rand], answerset[rand]))
                self.modify(gradient, learning_rate / batchsize)
        else:
            start_time = perf_counter()
            data_size = min(len(dataset), len(answerset))
            for i in range(1, cycles+1):
                rand = np.random.randint(0, data_size)
                gradient = self.backpropagate(dataset[rand], answerset[rand])
                cost = self.cost(answerset[rand])
                for _ in range(batchsize-1):
                    rand = np.random.randint(0, data_size)
                    gradient = gradient_sum(gradient, self.backpropagate(dataset[rand], answerset[rand]))
                    cost += self.cost(answerset[rand])
                self.modify(gradient, learning_rate / batchsize)
                runtime = perf_counter() - start_time
                print(f'Cycles finished: {i}/{cycles} | Cycle cost: {round(cost / batchsize, 3)} | Runtime: {round(runtime, 1)}/{round(runtime * cycles/i, 1)}')

    def train_momentum(self, dataset: list[np.ndarray] | list[list], 
                       answerset: list[np.ndarray] | list[list], 
                       learning_rate: float, momentum_conservation: float, 
                       cycles: int = 1, 
                       display_progress: bool = False) -> None:
        if display_progress == False:
            data_size = min(len(dataset), len(answerset))
            momentum = [(np.array([0]), np.array([0]))  for _ in range(len(self.info)-1)]
            for _ in range(cycles):
                gradient = self.backpropagate(dataset[0], answerset[0])
                for data, answer in zip(dataset[1:], answerset[1:]):
                    gradient = gradient_sum(gradient, self.backpropagate(data, answer))
                gradient = gradient_sum(gradient, gradient_product(momentum, momentum_conservation))
                momentum = gradient
                self.modify(gradient, learning_rate/data_size)
        else:
            start_time = perf_counter()
            data_size = min(len(dataset), len(answerset))
            momentum = [(np.array([0]), np.array([0]))  for _ in range(len(self.info)-1)]
            for i in range(1, cycles+1):
                gradient = self.backpropagate(dataset[0], answerset[0])
                cost = self.cost(answerset[0])
                for data, answer in zip(dataset[1:], answerset[1:]):
                    gradient = gradient_sum(gradient, self.backpropagate(data, answer))
                    cost += self.cost(answer)
                gradient = gradient_sum(gradient, gradient_product(momentum, momentum_conservation))
                momentum = gradient
                self.modify(gradient, learning_rate/data_size)
                runtime = perf_counter() - start_time
                print(f'Cycles finished: {i}/{cycles} | Cycle cost: {round(cost / data_size, 3)} | Runtime: {round(runtime, 1)}/{round(runtime * cycles/i, 1)}')

    def train_stochastic_momentum(self, dataset: list[np.ndarray] | list[list], 
                                  answerset: list[np.ndarray] | list[list], 
                                  learning_rate: float, momentum_conservation: float, 
                                  cycles: int = 1, batchsize: int = 1, 
                                  display_progress: bool = False) -> None:
        if display_progress == False:
            momentum = [(np.array([0]), np.array([0]))  for _ in range(len(self.info)-1)]
            data_size = min(len(dataset), len(answerset))
            for _ in range(cycles):
                rand = np.random.randint(0, data_size)
                gradient = self.backpropagate(dataset[rand], answerset[rand])
                for _ in range(batchsize-1):
                    rand = np.random.randint(0, data_size)
                    gradient = gradient_sum(gradient, self.backpropagate(dataset[rand], answerset[rand]))
                gradient = gradient_sum(gradient, gradient_product(momentum, momentum_conservation))
                momentum = gradient
                self.modify(gradient, learning_rate/data_size)
        else:
            start_time = perf_counter()
            data_size = min(len(dataset), len(answerset))
            momentum = [(np.array([0]), np.array([0]))  for _ in range(len(self.info)-1)]
            for i in range(1, cycles+1):
                rand = np.random.randint(0, data_size)
                gradient = self.backpropagate(dataset[rand], answerset[rand])
                cost = self.cost(answerset[rand])
                for _ in range(batchsize-1):
                    rand = np.random.randint(0, data_size)
                    gradient = gradient_sum(gradient, self.backpropagate(dataset[rand], answerset[rand]))
                    cost += self.cost(answerset[rand])
                gradient = gradient_sum(gradient, gradient_product(momentum, momentum_conservation))
                momentum = gradient
                self.modify(gradient, learning_rate/batchsize)
                runtime = perf_counter() - start_time
                print(f'Cycles finished: {i}/{cycles} | Cycle cost: {round(cost / batchsize, 3)} | Runtime: {round(runtime, 1)}/{round(runtime * cycles/i, 1)}')


def load(filename: str) -> Network:
    net = Network([0], 'linear')
    net.load(filename)
    return(net)