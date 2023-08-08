from functools import partial

import numpy as np
import scipy


def overlapping_pixels(coordinates, width):
    """Determine which fitting regions are expected to overlap.

    Returns a list of indices grouped by overlapping coordinates within a window 2 * width.

    This function constructs a list of lists. Each inner list contains the indices of the
    coordinates which are separated by < 2 * width. The outer list is sorted by spatial position.

    Parameters
    ----------
    coordinates : array_like
        coordinates
    width : float
        width in either direction
    """
    if not len(coordinates):
        return []

    coordinates = np.asarray(coordinates)
    indices = np.argsort(coordinates)
    sorted_coordinates = coordinates[indices]
    differences = np.diff(sorted_coordinates)
    groups = np.split(indices, np.flatnonzero(differences > 2 * width) + 1)

    return groups


def poisson_log_likelihood(params, expectation_fun, photon_count):
    """Calculates the log likelihood of Poisson distributed values.

    Parameters
    ----------
    params : np.ndarray
        Model parameters to be estimated
    expectation_fun : callable
        expectation function to be evaluated in the likelihood; takes model parameters as arguments.
    photon_count : np.ndarray
        Measured photon counts at each position to be fitted.
    """
    expectation = expectation_fun(params)
    log_likelihood = photon_count * np.log(expectation) - expectation
    return np.sum(-log_likelihood)


def poisson_log_likelihood_jacobian(params, expectation_fun, derivatives_fun, photon_count):
    """Evaluate the derivatives of the likelihood function w.r.t. each parameter."""
    derivatives = derivatives_fun(params)
    count_over_expectation = photon_count / expectation_fun(params)
    return -np.sum(derivatives * (count_over_expectation - 1), axis=1)


def _mle_optimize(initial_guess, expectation_fun, derivatives_fun, photon_count, bounds):
    """Calculate the maximum likelihood estimate of the model parameters given measured photon count.

    This function is meant to be generalizable for use with either 1D or 2D functions.

    Parameters
    ----------
    initial_guess : np.ndarray
        Initial guesses for the model parameters.
    expectation_fun : callable
        expectation function to be evaluated in the likelihood; takes model parameters as arguments.
    derivatives_fun : callable
        expectation derivatives function to be evaluated in the Jacobian; takes model parameters as arguments.
    photon_count : np.ndarray
        Measured photon counts at each position to be fitted.
    bounds : tuple
        Tuple of (`min`, `max`) pairs for each parameter. `None` is used to specify no bound.
    """
    optimization_fun = partial(
        poisson_log_likelihood, expectation_fun=expectation_fun, photon_count=photon_count
    )
    jac_fun = partial(
        poisson_log_likelihood_jacobian,
        expectation_fun=expectation_fun,
        derivatives_fun=derivatives_fun,
        photon_count=photon_count,
    )

    return scipy.optimize.minimize(
        optimization_fun,
        initial_guess,
        jac=jac_fun,
        method="L-BFGS-B",
        bounds=bounds,
    )


def normal_pdf_1d(x, center, sigma):
    """Evaluates the probablitity density function for a 1D normal distribution

    Parameters
    ----------
    x : np.ndarray
        Position data at which the function is to be evaluated.
    center : float
        Distribution center.
    sigma : float
        Distribution sigma; square root of the variance.
    """
    norm_factor = 1 / np.sqrt(2 * np.pi * sigma**2)
    return norm_factor * np.exp(-0.5 * ((x - center) / sigma) ** 2)


def peak_expectation_1d(x, params, fixed_background, pixel_size, num_peaks):
    """Calculates the expectation value of a sum of gaussian peaks evaluated at x with baseline
    offset.

    Parameters
    ----------
    x : np.ndarray
        Position data at which the function is to be evaluated.
    params : np.ndarray
        Model parameters.
        Model parameters should be provided as an array of parameters. The parameters are:
        photon counts for all peaks, center (in microns) for all peaks and sigma (in microns) for
        all peaks followed by an optional offset parameter (in photons per pixel).
    fixed_background : float or None
        Fixed background value in photons per pixel.
    pixel_size : float
        Pixel size in um.
    num_peaks : int
        Number of peaks to fit.
    """
    total_photons, center, sigma = (
        params[np.newaxis, :num_peaks],
        params[np.newaxis, num_peaks : 2 * num_peaks],
        params[np.newaxis, 2 * num_peaks : 3 * num_peaks],
    )
    x = x[:, np.newaxis]

    signal = np.sum(total_photons * pixel_size * normal_pdf_1d(x, center, sigma), axis=1)

    background = fixed_background if fixed_background is not None else params[-1]
    return signal + background


def peak_expectation_1d_derivatives(x, params, fixed_background, pixel_size, num_peaks):
    """Evaluate the derivatives of the expectation w.r.t. each parameter."""
    total_photons, center, sigma = (
        params[np.newaxis, :num_peaks],
        params[np.newaxis, num_peaks : 2 * num_peaks],
        params[np.newaxis, 2 * num_peaks : 3 * num_peaks],
    )
    x = x[:, np.newaxis]

    pdf = normal_pdf_1d(x, center, sigma)
    d_dphotons = pixel_size * pdf
    d_dcenter = total_photons * pixel_size / sigma**2 * pdf * (x - center)
    d_dsigma = total_photons * pixel_size * pdf * ((x - center) ** 2 - sigma**2) / sigma**3
    derivatives = np.hstack((d_dphotons, d_dcenter, d_dsigma))

    if fixed_background is None:
        derivatives = np.hstack((derivatives, np.ones(x.shape)))

    return derivatives.T


def _estimation_parameters_simultaneous(x, photon_count, sorted_initial_position):
    """Compute initial guess, max counts and positional bounds per peak for a simultaneous fit

    Parameters
    ----------
    x : np.ndarray
        Position data at which the function is to be evaluated.
    photon_count : np.ndarray
        Measured photon counts at each position.
    sorted_initial_position : np.ndarray
        Sorted initial guess for the peak position, in um.
    """

    sorted_initial_position = np.clip(sorted_initial_position, np.min(x), np.max(x))
    position_range = np.hstack(
        [np.min(x), (sorted_initial_position[:-1] + sorted_initial_position[1:]) / 2.0, np.max(x)]
    )
    position_bounds = np.vstack((position_range[:-1], position_range[1:])).T
    position_indices = [np.where(x >= pos)[0][0] for pos in position_range]

    max_photons = np.array(
        [
            float(np.max(photon_count[p1 : p2 + 1]))
            for p1, p2 in zip(position_indices[:-1], position_indices[1:])
        ]
    )

    return sorted_initial_position, position_bounds, max_photons


def gaussian_mle_1d(
    x,
    photon_count,
    pixel_size,
    initial_position=None,
    initial_sigma=0.250,
    fixed_background=None,
    enforce_position_bounds=True,
):
    """Calculate the maximum likelihood estimate of the model parameters given measured photon count
    for 1D data.

    Returns a tuple of length {number of fitted peaks} containing
    (position, total photons, width, background) for each fitted peak

    Parameters
    ----------
    x : np.ndarray
        Position data at which the function is to be evaluated.
    photon_count : np.ndarray
        Measured photon counts at each position.
    pixel_size : float
        Pixel size in um.
    initial_position : float or np.ndarray, optional
        Initial guess for the peak position, in um.
    initial_sigma : float
        Initial guess for the `sigma` parameter, in um.
    fixed_background : float
        Fixed background parameter in photons per second.
        When supplied, the background is not estimated but fixed at this value.
    enforce_position_bounds : bool
        Enforce bounds between each peak. This ensures that track positions do not swap places.
    """
    if fixed_background is not None and fixed_background <= 0:
        raise ValueError("Fixed background should be larger than zero.")

    initial_position = np.atleast_1d(
        x[np.argmax(photon_count)] if initial_position is None else initial_position
    )

    num_peaks = initial_position.size
    expectation_fun = partial(
        peak_expectation_1d,
        x,
        fixed_background=fixed_background,
        pixel_size=pixel_size,
        num_peaks=num_peaks,
    )
    derivatives_fun = partial(
        peak_expectation_1d_derivatives,
        x,
        fixed_background=fixed_background,
        pixel_size=pixel_size,
        num_peaks=num_peaks,
    )

    # Sort the positions for consistency during optimization
    idx = np.argsort(initial_position)
    initial_position = initial_position[idx]

    # Keep a set of reverse sort indices so the original order can be preserved on output
    reverse_idx = np.argsort(idx)

    # Divide up the range of positions such that each track gets its own range.
    max_to_counts = 1.0 / pixel_size * np.sqrt(2 * np.pi * initial_sigma**2)
    if enforce_position_bounds:
        initial_position, position_bounds, max_photons = _estimation_parameters_simultaneous(
            x, photon_count, initial_position
        )
        amp_estimate = max_photons * max_to_counts
    else:
        position_bounds = np.tile((np.min(x), np.max(x)), (num_peaks, 1))
        amp_estimate = np.max(photon_count) * max_to_counts / initial_position.size

    # parameters are ordered as follows: amplitudes, centers, widths, background offset
    num_pars = num_peaks * 3 + (0 if fixed_background else 1)
    initial_guess = np.empty(num_pars)
    initial_guess[:num_peaks] = amp_estimate
    initial_guess[num_peaks : 2 * num_peaks] = initial_position
    initial_guess[2 * num_peaks :] = initial_sigma

    bounds = np.vstack(
        (
            np.tile((0.01, np.inf), (num_peaks, 1)),
            position_bounds,
            np.tile((pixel_size, 10 * pixel_size), (num_peaks, 1)),
        )
    )

    if fixed_background is None:
        initial_guess[-1] = max(1, np.min(photon_count))
        bounds = np.vstack((bounds, (np.finfo(float).eps, None)))

    result = _mle_optimize(initial_guess, expectation_fun, derivatives_fun, photon_count, bounds)

    # Pack the results
    background = result.x[-1] if fixed_background is None else fixed_background
    peak_parameters = result.x[: initial_position.size * 3].reshape(3, -1).T
    # re-order position, photons, width
    peak_parameters = peak_parameters[:, [1, 0, 2]]

    # Restore original order; sort by peak position
    sort_idx = np.argsort(peak_parameters[:, 0])
    peak_parameters = peak_parameters[sort_idx[reverse_idx], :]
    return tuple((*param, background) for param in peak_parameters)
