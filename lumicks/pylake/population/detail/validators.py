import numpy as np


def row(x):
    """Return x as row vector."""
    return np.reshape(x, (1, -1))


def col(x):
    """Return x as column vector."""
    return np.reshape(x, (-1, 1))
