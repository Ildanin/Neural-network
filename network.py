import numpy as np
from math import e
from collections.abc import Callable
from time import perf_counter
rng = np.random.default_rng()

def array_decorator(function: Callable) -> Callable:
    def wrapper(array: np.ndarray) -> np.ndarray:
        return(np.array(list(map(function, array))))
    return(wrapper)

def random_array(low: float, high: float, size: int | tuple[int] | tuple[int, int]) -> np.ndarray:
    return((high - low) * rng.random(size) + low)

def add_noise(data: np.ndarray, noise_range: tuple[float, float]) -> np.ndarray:
    noise = random_array(noise_range[0], noise_range[1], data.size)
    return(data + noise)

def apply_threshold(result: np.ndarray, threshold: float) -> np.ndarray:
    return(np.maximum(threshold, result))

def gradient_product(gradient: list[tuple[np.ndarray, np.ndarray]], coef: float) -> list[tuple[np.ndarray, np.ndarray]]:
    return([(coef * weight_grad, coef * bias_grad) for weight_grad, bias_grad in gradient])

def gradient_sum(gradient1: list[tuple[np.ndarray, np.ndarray]], gradient2: list[tuple[np.ndarray, np.ndarray]]) -> list[tuple[np.ndarray, np.ndarray]]:
    return([(layer_grad1[0] + layer_grad2[0], layer_grad1[1] + layer_grad2[1]) for layer_grad1, layer_grad2 in zip(gradient1, gradient2)])

#Activators
def linear(array: np.ndarray) -> np.ndarray:
    return(array)

def linear_derivative(array: np.ndarray) -> np.ndarray:
    return(np.ones_like(array))

def sigmoid(array: np.ndarray) -> np.ndarray:
    return(1 / (e ** (-array) + 1))

def sigmoid_derivative(array: np.ndarray) -> np.ndarray:
    return((e ** array) / ((e ** array + 1)**2))

def ReLU(array: np.ndarray) -> np.ndarray:
    return(np.maximum(0, array))

def ReLU_derivative(array: np.ndarray) -> np.ndarray:
    return(np.maximum(0, np.sign(array)))

def L_ReLU(array: np.ndarray) -> np.ndarray:
    return(np.maximum(0.1*array, array))

def L_ReLU_derivative(array: np.ndarray) -> np.ndarray:
    return(abs(np.maximum(0.1*np.sign(array), np.sign(array))))

Activators: dict[str, tuple[Callable[[np.ndarray], np.ndarray], Callable[[np.ndarray], np.ndarray]]] = {\
"linear": (linear, linear_derivative), 
"sigmoid": (sigmoid, sigmoid_derivative), 
"ReLU": (ReLU, ReLU_derivative), 
"L_ReLU": (L_ReLU, L_ReLU_derivative)
}

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

def compare_networks(filename1: str, filename2: str, writefile: str | None = None) -> str:
    file1 = [float(x) for x in open(filename1).readlines()[3:]]
    file2 = [float(x) for x in open(filename2).readlines()[3:]]
    if len(file1) != len(file2):
        raise ValueError("Networks are of different sizes")
    delta = [str(x2-x1) + '\n' for x1, x2 in zip(file1, file2)]
    if writefile == None:
        result_file = open(f'{'.'.join(filename1.split('.')[:-1])}-{'.'.join(filename2.split('.')[:-1])}.txt', 'w')
    else:
        result_file = open(writefile, 'w')
    result_file.writelines(delta)
    return(result_file.name)