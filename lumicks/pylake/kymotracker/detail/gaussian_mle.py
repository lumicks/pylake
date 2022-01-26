import numpy as np
from functools import partial
from scipy.optimize import minimize


def poisson_log_likelihood(params, expectation_fun, photon_count):
    """Calculates the log likelihood of Poisson distributed values.

    Parameters
    ----------
    params : array-like
        Model parameters to be estimated
    expectation_fun : callable
        expectation function to be evaluated in the likelihood; takes model parameters as arguments.
    photon_count : array-like
        Measured photon counts at each position to be fitted.
    """
    expectation = expectation_fun(params)
    log_likelihood = photon_count * np.log(expectation) - expectation
    return np.sum(-log_likelihood)


def poisson_log_likelihood_jacobian(params, expectation_fun, derivatives_fun, photon_count):
    """Evaluate the derivatives of the likelihood function w.r.t. each parameter."""
    derivatives = derivatives_fun(params)
    count_over_expectation = photon_count / expectation_fun(params)
    return [-np.sum(d * ((count_over_expectation) - 1)) for d in derivatives]


def _mle_optimize(initial_guess, expectation_fun, derivatives_fun, photon_count, bounds):
    """Calculate the maximum likelihood estimate of the model parameters given measured photon count.

    This function is meant to be generalizable for use with either 1D or 2D functions.

    Parameters
    ----------
    initial_guess : array-like
        Initial guesses for the model parameters.
    expectation_fun : callable
        expectation function to be evaluated in the likelihood; takes model parameters as arguments.
    derivatives_fun : callable
        expectation derivatives function to be evaluated in the Jacobian; takes model parameters as arguments.
    photon_count : array-like
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

    return minimize(
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
    x : array-like
        Position data at which the function is to be evaluated.
    center : float
        Distribution center.
    sigma : float
        Distribution sigma; square root of the variance.
    """
    norm_factor = 1 / np.sqrt(2 * np.pi * sigma ** 2)
    return norm_factor * np.exp(-0.5 * ((x - center) / sigma) ** 2)


def peak_expectation_1d(x, params, fixed_background, pixel_size, n_components):
    """Calculates the expectation value of a sum of gaussian peaks evaluated at x with baseline
    offset.

    Parameters
    ----------
    x : array-like
        Position data at which the function is to be evaluated.
    params : array-like
        Model parameters.
        Model parameters should be provided as an array of parameters for each peak in the fit
        followed by an optional offset parameter (in photons per pixel).
        The peak parameters are: photon counts, center (in microns) and sigma (in microns).
    fixed_background : float or None
        Fixed background value in photons per pixel.
    pixel_size : float
        Pixel size in um.
    """
    signal = np.zeros(x.shape)
    for total_photons, center, sigma in params[: n_components * 3].reshape(-1, 3):
        signal += total_photons * pixel_size * normal_pdf_1d(x, center, sigma)
    background = fixed_background if fixed_background is not None else params[-1]
    return signal + background


def peak_expectation_1d_derivatives(x, params, fixed_background, pixel_size, n_components):
    """Evaluate the derivatives of the expectation w.r.t. each parameter."""
    components = []
    for total_photons, center, sigma in params[: n_components * 3].reshape(-1, 3):
        pdf = normal_pdf_1d(x, center, sigma)
        d_dphotons = pixel_size * pdf
        d_dcenter = total_photons * pixel_size / sigma ** 2 * pdf * (x - center)
        d_dsigma = total_photons * pixel_size * pdf * ((x - center) ** 2 - sigma ** 2) / sigma ** 3
        components.extend((d_dphotons, d_dcenter, d_dsigma))

    if fixed_background is None:
        components.append(1)

    return components


def gaussian_mle_1d(
    x, photon_count, pixel_size, initial_position=None, initial_sigma=0.250, fixed_background=None
):
    """Calculate the maximum likelihood estimate of the model parameters given measured photon count
    for 1D data.

    Parameters
    ----------
    x : array-like
        Position data at which the function is to be evaluated.
    photon_count : array-like
        Measured photon counts at each position.
    pixel_size : float
        Pixel size in um.
    initial_position : float
        Initial guess for the peak position, in um.
    initial_sigma : float
        Initial guess for the `sigma` parameter, in um.
    fixed_background : float
        Fixed background parameter in photons per second.
        When supplied, the background is not estimated but fixed at this value.
    """
    if fixed_background is not None and fixed_background <= 0:
        raise ValueError("Fixed background should be larger than zero.")

    initial_position = x[np.argmax(photon_count)] if initial_position is None else initial_position
    initial_position = np.atleast_1d(initial_position)

    expectation_fun = partial(
        peak_expectation_1d,
        x,
        fixed_background=fixed_background,
        pixel_size=pixel_size,
        n_components=initial_position.size,
    )
    derivatives_fun = partial(
        peak_expectation_1d_derivatives,
        x,
        fixed_background=fixed_background,
        pixel_size=pixel_size,
        n_components=initial_position.size,
    )

    amp_estimate = np.max(photon_count) / pixel_size * np.sqrt(2 * np.pi * initial_sigma ** 2)
    initial_guess, bounds = [], []
    for init_pos in initial_position:
        initial_guess.extend([amp_estimate / initial_position.size, init_pos, initial_sigma])
        bounds.extend([(0.01, None), (np.min(x), np.max(x)), (pixel_size, 10 * pixel_size)])
    if fixed_background is None:
        initial_guess.append(1)
        bounds.append((np.finfo(float).eps, None))

    result = _mle_optimize(initial_guess, expectation_fun, derivatives_fun, photon_count, bounds)

    # Pack the results
    background = result.x[-1] if fixed_background is None else fixed_background
    result = tuple(
        (*param, background) for param in result.x[: initial_position.size * 3].reshape(-1, 3)
    )
    return result
