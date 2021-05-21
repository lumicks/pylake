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
    """Calculates the derivatives of the peak expectation w.r.t. each parameter.

    Results are returned as 2D arrays with shape [len(x), len(parameter)]"""
    g = gaussian(x, amplitude, center, scale)
    x_center_diff = col(x) - row(center)

    d_damplitude = g / amplitude
    d_dcenter = g * x_center_diff / row(scale) ** 2

    d_dscale_numerator = (lambda x: col(x.sum(axis=1))) if len(scale) == 1 else (lambda x: (x))
    d_dscale = d_dscale_numerator(g * x_center_diff ** 2) / scale ** 3

    d_doffset = col(np.ones(g.shape).sum(axis=1)) if len(offset) == 1 else np.ones(g.shape)

    return d_damplitude, d_dcenter, d_dscale, d_doffset


def _extract_params(params, n_frames, shared_variance, shared_offset):
    """Separates single array of parameters into individual components."""
    n_var = 1 if shared_variance else n_frames
    n_off = 1 if shared_offset else n_frames
    assert len(params) == 2 * n_frames + n_var + n_off

    amplitude = params[:n_frames]
    center = params[n_frames : 2 * n_frames]

    scale_slice = 2 * n_frames if shared_variance else slice(2 * n_frames, 3 * n_frames)
    offset_start = 2 * n_frames + 1 if shared_variance else 3 * n_frames

    scale = np.atleast_1d(params[scale_slice])
    offset = params[offset_start:]

    return amplitude, center, scale, offset


def poisson_log_likelihood(params, x, photon_count, shared_variance=False, shared_offset=False):
    """Calculates the log likelihood of Poisson distributed values.

    Parameters
    __________
    params : array-like
        Model parameters to be estimated; order is [amplitude, center, scale, offset]
    x : array-like
        Data at which the function is to be evaluated.
    photon_count : array-like
        Signal data to be fitted with shape [len(x), # of frames]
    shared_variance : bool
        If the variance is a single value shared across all frames
    shared_offset : bool
        If the offset is a single value shared across all frames
    """
    n_frames = photon_count.shape[1]
    amplitude, center, scale, offset = _extract_params(
        params, n_frames, shared_variance, shared_offset
    )

    expectation = peak_expectation(x, amplitude, center, scale, offset)
    log_likelihood = photon_count * np.log(expectation) - expectation
    return np.sum(-log_likelihood)


def poisson_log_likelihood_jacobian(
    params, x, photon_count, shared_variance=False, shared_offset=False
):
    """Calculates the derivatives of the log likelihood w.r.t. each parameter."""
    n_frames = photon_count.shape[1]
    amplitude, center, scale, offset = _extract_params(
        params, n_frames, shared_variance, shared_offset
    )

    tile = lambda x: np.hstack(
        (
            x,
            x,
            col(x.sum(axis=1)) if len(scale) == 1 else x,
            col(x.sum(axis=1)) if len(offset) == 1 else x,
        )
    )

    expectation = tile(peak_expectation(x, amplitude, center, scale, offset))
    photon_count = tile(photon_count)

    derivatives = np.hstack(peak_expectation_derivatives(x, amplitude, center, scale, offset))
    dL_dparam = derivatives * ((photon_count / expectation) - 1)
    return -np.sum(dL_dparam, axis=0)


def run_gaussian_mle(
    x, photon_count, pixel_size, shared_variance=False, shared_offset=False, init_center=None
):
    """Calculates maximum likelihood estimate of gaussian parameters.

    Parameters
    __________
    x : array-like
        Data at which the function is to be evaluated.
    photon_count : array-like
        Signal data to be fitted with shape [len(x), # of frames]
    pixel_size : float
        Pixel size in um
    shared_variance : bool
        If the variance is a single value shared across all frames
    shared_offset : bool
        If the offset is a single value shared across all frames
    """
    try:
        n_frames = photon_count.shape[1]
    except IndexError:
        photon_count = col(photon_count)
        n_frames = photon_count.shape[1]

    # initial guesses for parameters
    init_amplitude = photon_count.max(axis=0)
    init_center = x[np.argmax(photon_count, axis=0)] if init_center is None else init_center
    init_scale = np.full(1 if shared_variance else n_frames, 3 * pixel_size)
    init_offset = np.full(1 if shared_offset else n_frames, 1)
    initial_guess = np.hstack((init_amplitude, init_center, init_scale, init_offset))

    # bounds
    bounds_amplitude = [(y * 0.1, None) for y in photon_count.max(axis=0)]
    bounds_center = [(x.min(), x.max())] * n_frames
    bounds_scale = [(pixel_size, 6 * pixel_size)] * (1 if shared_variance else n_frames)
    bounds_offset = [(np.finfo(float).eps, None)] * (1 if shared_offset else n_frames)
    bounds = (*bounds_amplitude, *bounds_center, *bounds_scale, *bounds_offset)

    result = minimize(
        poisson_log_likelihood,
        initial_guess,
        args=(x, photon_count, shared_variance, shared_offset),
        method="L-BFGS-B",
        jac=poisson_log_likelihood_jacobian,
        bounds=bounds,
    )
    return result.x
