import numpy as np
from sys import stdout
rng = np.random.default_rng()

def random_array(low: float, high: float, size: int | tuple[int] | tuple[int, int]) -> np.ndarray:
    return((high - low) * rng.random(size) + low)

def gradient_product(gradient: list[tuple[np.ndarray, np.ndarray]], coef: float) -> list[tuple[np.ndarray, np.ndarray]]:
    return([(coef * weight_grad, coef * bias_grad) for weight_grad, bias_grad in gradient])

def gradient_sum(gradient1: list[tuple[np.ndarray, np.ndarray]], gradient2: list[tuple[np.ndarray, np.ndarray]]) -> list[tuple[np.ndarray, np.ndarray]]:
    return([(layer_grad1[0] + layer_grad2[0], layer_grad1[1] + layer_grad2[1]) for layer_grad1, layer_grad2 in zip(gradient1, gradient2)])

def print_info(finished_cycles: int, all_cycles: int, cost: float, data_size: int, runtime: float) -> None:
    stdout.write('\r')
    stdout.flush()
    stdout.write(f"Cycles finished: {finished_cycles}/{all_cycles} | Cycle cost: {round(cost / data_size, 3)} | Runtime: {round(runtime, 1)}/{round(runtime * all_cycles / finished_cycles, 1)}")