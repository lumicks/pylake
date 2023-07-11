import numpy as np
import scipy.signal


def equal_length(kernels):
    """Make the kernels the same length (while keeping them centered)

    Parameters
    ----------
    kernels : list of np.ndarray
        List of kernels to symmetrize.

    Raises
    ------
    ValueError
        if any of the kernel isn't odd sized (violates the ability to center it).
    """
    target_length = len(kernels[-1])

    def pad(kernel):
        if len(kernel) % 2 == 0:
            raise ValueError("This function should only be used with odd sized kernels")

        padding = np.zeros(((target_length - len(kernel)) // 2, 1))
        return np.vstack((padding, kernel, padding))

    return [pad(k) for k in kernels]


def generate_product_filters(kernels):
    """Generate product filters

    Generates a list of kernels that consist of the previous kernels convolved.

    Parameters
    ----------
    kernels : list of np.ndarray
        List of 1-dimensional kernels to calculate the convolutional product of.
    """
    current = np.asarray([[1.0]])
    convolved_kernels = [current]
    for k in kernels:
        current = scipy.signal.convolve(current, k, mode="full")
        convolved_kernels.append(current)

    return convolved_kernels


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
