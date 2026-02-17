import numpy as np
from collections.abc import Callable
from math import e

def array_decorator(function: Callable) -> Callable:
    def wrapper(array: np.ndarray) -> np.ndarray:
        return(np.array(list(map(function, array))))
    return(wrapper)

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

Activators: dict[str, tuple[Callable[[np.ndarray], np.ndarray], Callable[[np.ndarray], np.ndarray]]] = {
"linear": (linear, linear_derivative), 
"sigmoid": (sigmoid, sigmoid_derivative), 
"ReLU": (ReLU, ReLU_derivative), 
"L_ReLU": (L_ReLU, L_ReLU_derivative)
}

def add_activator(name: str, 
                  activator: Callable[[np.ndarray], np.ndarray], 
                  activator_derivative: Callable[[np.ndarray], np.ndarray]) -> None:
    Activators.update({name: (activator, activator_derivative)})