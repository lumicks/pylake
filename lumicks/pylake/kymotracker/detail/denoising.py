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


def tau(kernel, power):
    """Calculate filter summation from Sec 2. Paragraph 1."""
    return np.sum(kernel**power)


def calculate_vst_coefficients(kernel):
    """Calculate optimal transform coefficients and expected standard deviation.

    We calculate b and c for the following transform:

        Z = b sign(Y + c) * sqrt(Y + c)

    Using the coefficients `c` from Eq. 10 and `b` (which is defined above Eq. 12 in [3]_), we
    estimate a transform such that Z - b sqrt(tau_1 lambda) ~ N(0, 1) where lambda is the Poisson
    parameter [3]_.

    Parameters
    ----------
    kernel : np.ndarray
        Kernel at scale `j`. Note that this is the full kernel, so Kj * Kj-1 .. Kn
        where * is convolution operator.

    References
    ----------
    .. [3] Zhang, B., Fadili, J. M., & Starck, J. L. (2008). Wavelets, ridgelets, and curvelets
           for Poisson noise removal. IEEE Transactions on image processing, 17(7), 1093-1108.
    """
    b = np.sign(tau(kernel, 1)) / np.sqrt(np.abs(tau(kernel, 1)))  # Defined above eqn 12
    c = 7.0 * tau(kernel, 2) / (8.0 * tau(kernel, 1)) - tau(kernel, 3) / (2.0 * tau(kernel, 2))
    return b, c


def calculate_vst_stdev(kernel, prev_kernel):
    """Calculate expected standard deviation of detail layer

    Parameters
    ----------
    kernel, prev_kernel : np.ndarary
        kernel at scale j and j - 1

    Raises
    ------
    ValueError
        if kernel sizes are not equal
    """
    if prev_kernel.shape != kernel.shape:
        raise ValueError("Kernel shapes must be equal to calculate standard deviation")

    term1 = tau(prev_kernel, 2) / (4 * tau(prev_kernel, 1) ** 2)
    term2 = tau(kernel, 2) / (4 * tau(kernel, 1) ** 2)
    term3 = np.sum(prev_kernel * kernel) / (2 * tau(prev_kernel, 1) * tau(kernel, 1))
    expected_sd = np.sqrt(term1 + term2 - term3)

    return expected_sd


def variance_stabilizing_transform(image, coefficients):
    b, c = coefficients
    return b * np.sign(image + c) * np.sqrt(np.abs(image + c))


def inverse_variance_stabilizing_transform(image, coefficients):
    b, c = coefficients
    return np.sign(image) * (image / b) ** 2 - c


class MultiScaleVarianceStabilizingTransform:
    """Calculates the required coefficients for determining the variance stabilizing transform (VST)

    The VST is a generalization of the Anscombe transform which can convert Poisson distributed.
    noise to approximately Gaussian distributed noise. For Poisson distributed data, the variance
    scales with the mean. After the transform the distribution is roughly Gaussian with a variance
    of 1. This transformation allows you to apply statistics that rely on normality post-transform.

    By combining this with a wavelet procedure, the variance and bias of this procedure goes down
    more quickly than when applied without any filters [1, 2].

    Parameters
    ----------
    kernels : list of np.ndarray
        List of kernels to use for the multiscale variance stabilizing transform.
    two_dimensional : bool
        Convert the kernel into a two-dimensional kernel.

    References
    ----------
    .. [1] Zhang, B., Fadili, M. J., Starck, J. L., & Olivo-Marin, J. C. (2007, September).
           Multiscale variance-stabilizing transform for mixed-Poisson-Gaussian processes and its
           applications in bioimaging. In 2007 IEEE International Conference on Image Processing
           (Vol. 6, pp. VI-233). IEEE.
    .. [2] Zhang, B., Fadili, J. M., & Starck, J. L. (2008). Wavelets, ridgelets, and curvelets
           for Poisson noise removal. IEEE Transactions on image processing, 17(7), 1093-1108.
    """

    def __init__(self, kernels, *, two_dimensional=False):
        self._kernels = [k * k.T for k in kernels] if two_dimensional else kernels

        # Full kernels contain h0 * h1 * ... * hn where * is convolution.
        equal_length_kernels = equal_length(generate_product_filters(kernels))
        self._full_kernels = [k * k.T if two_dimensional else k for k in equal_length_kernels]
        self._coefficients = [calculate_vst_coefficients(k) for k in self._full_kernels]

        self._stdev = [
            calculate_vst_stdev(k1, k2)
            for k1, k2 in zip(self._full_kernels[1:], self._full_kernels[:-1])
        ]
