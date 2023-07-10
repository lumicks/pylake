import numpy as np
import scipy.signal


def generate_bspline_kernels(num_kernels):
    """Returns a list of à trous b3 spline kernels.

    Parameters
    ----------
    num_kernels : int
        Number of kernels we want

    Returns
    -------
    kernels : list of np.ndarray
        1-dimensional kernels of varying sizes
    """
    k = np.arange(1, num_kernels + 1)
    base_kernel = np.array([1 / 16, 1 / 4, 3 / 8, 1 / 4, 1 / 16])
    kernels = []
    n_kernels = len(base_kernel)
    for zeros in 2 ** (k - 1) - 1:
        kernel = np.zeros(n_kernels + zeros * (n_kernels - 1))
        kernel[:: zeros + 1] = base_kernel
        kernels.append(kernel.reshape((-1, 1)))

    return kernels
