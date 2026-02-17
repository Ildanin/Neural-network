import numpy as np
from .utiles import random_array

def add_noise(data: np.ndarray, noise_range: tuple[float, float]) -> np.ndarray:
    noise = random_array(noise_range[0], noise_range[1], data.size)
    return(data + noise)

def apply_threshold(result: np.ndarray, threshold: float) -> np.ndarray:
    return(np.maximum(threshold, result))