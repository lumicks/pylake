import numpy as np
from scipy.optimize import minimize


def row(x):
    """Reshapes 1D array to [1xN] row vector."""
    return np.atleast_1d(x)[np.newaxis, :]


def col(x):
    """Reshapes 1D array to [Nx1] column vector."""
    return np.atleast_1d(x)[:, np.newaxis]


def gaussian(x, amplitude, center, scale):
    """Evaluates the gaussian function at x for one or more peaks.

    Result is returned as a 2D array with shape [len(x), len(center)].

    Parameters
    __________
    x : array-like
        Data at which the function is to be evaluated.
    amplitude : float or array-like
        Peak amplitude(s)
    center : float or array-like
        Peak center(s)
    scale : float or array-like
        Peak standard deviation(s); controls peak width.
    """
    return amplitude * np.exp(-0.5 * ((col(x) - row(center)) / row(scale)) ** 2)


def peak_expectation(x, amplitude, center, scale, offset):
    """Calculates the expectation value of a gaussian peak evaluated at x with baseline offset.

    Result is returns as a 2D array with shape [len(x), len(center)]
    """
    return gaussian(x, amplitude, center, scale) + offset


def peak_expectation_derivatives(x, amplitude, center, scale, offset):
    """Calculates the derivatives of the peak expectation w.r.t. each parameter."""
    g = gaussian(x, amplitude, center, scale)
    x_center_diff = col(x) - row(center)

    d_damplitude = g / amplitude
    d_dcenter = g * x_center_diff / row(scale) ** 2
    d_dscale = g * x_center_diff ** 2 / row(scale) ** 3
    d_doffset = np.ones(g.shape)
    return d_damplitude, d_dcenter, d_dscale, d_doffset


def _extract_params_full(params):
    params = params.reshape((4, -1))
    amplitude, center, scale, offset = params
    return amplitude, center, scale, offset


def poisson_log_likelihood(params, x, photon_count):
    """Calculates the log likelihood of Poisson distributed values.

    Parameters
    __________
    params : array-like
        Model parameters to be estimated; order is [amplitude, center, scale, offset]
    x : array-like
        Data at which the function is to be evaluated.
    photon_count : array-like
        Signal data to be fitted with shape [len(x), # of frames]
    """
    amplitude, center, scale, offset = _extract_params_full(params)
    expectation = peak_expectation(x, amplitude, center, scale, offset)
    log_likelihood = photon_count * np.log(expectation) - expectation
    return np.sum(-log_likelihood)


def poisson_log_likelihood_jacobian(params, x, photon_count):
    amplitude, center, scale, offset = _extract_params_full(params)
    expectation = peak_expectation(x, amplitude, center, scale, offset)
    derivatives = np.hstack(peak_expectation_derivatives(x, amplitude, center, scale, offset))
    dL_dparam = derivatives * ((photon_count / expectation) - 1)
    return -np.sum(dL_dparam, axis=0)
