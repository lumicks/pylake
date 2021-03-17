import numpy as np
import contextlib


@contextlib.contextmanager
def temp_seed(seed):
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.seed(None)


def draw_bootstrap_indices(n):
    return np.random.choice(n, size=n, replace=True)
