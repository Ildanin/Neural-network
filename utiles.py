import numpy as np

rng = np.random.default_rng()

def random_array(low: float, high: float, size: int | tuple[int, ...]) -> np.ndarray:
    return((high - low) * rng.random(size) + low)