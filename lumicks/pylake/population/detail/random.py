import numpy as np


def draw_bootstrap_indices(n):
    return np.random.choice(n, size=n, replace=True)
