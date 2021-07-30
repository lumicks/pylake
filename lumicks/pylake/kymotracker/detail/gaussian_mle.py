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
    expectation = expectation_fun(*params)
    log_likelihood = photon_count * np.log(expectation) - expectation
    return np.sum(-log_likelihood)


def poisson_log_likelihood_jacobian(params, expectation_fun, derivatives_fun, photon_count):
    """Evaluate the derivatives of the likelihood function w.r.t. each parameter."""
    derivatives = derivatives_fun(*params)
    count_over_expectation = photon_count / expectation_fun(*params)
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


def peak_expectation_1d(x, total_photons, center, sigma, background, pixel_size):
    """Calculates the expectation value of a gaussian peak evaluated at x with baseline offset.

    Parameters
    ----------
    x : array-like
        Position data at which the function is to be evaluated.
    total_photons : float
        Total number of photons emitted by the imaged particle.
    center : float
        Peak center in um.
    sigma : float
        sigma parameter in um.
    background : float
        Background in photons per pixel.
    pixel_size : float
        Pixel size in um.
    """
    return total_photons * pixel_size * normal_pdf_1d(x, center, sigma) + background


def peak_expectation_1d_derivatives(x, total_photons, center, sigma, background, pixel_size):
    """Evaluate the derivatives of the expectation w.r.t. each parameter."""
    pdf = normal_pdf_1d(x, center, sigma)
    d_dphotons = pixel_size * pdf
    d_dcenter = total_photons * pixel_size / sigma ** 2 * pdf * (x - center)
    d_dsigma = total_photons * pixel_size * pdf * ((x - center) ** 2 - sigma ** 2) / sigma ** 3
    d_dbackground = 1
    return d_dphotons, d_dcenter, d_dsigma, d_dbackground


def gaussian_mle_1d(x, photon_count, pixel_size, initial_position=None, initial_sigma=0.250):
    """Calculate the maximum likelihood estimate of the model parameters given measured photon count for 1D data.

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
    """
    expectation_fun = partial(peak_expectation_1d, x, pixel_size=pixel_size)
    derivatives_fun = partial(peak_expectation_1d_derivatives, x, pixel_size=pixel_size)
    initial_guess = (
        np.max(photon_count) / pixel_size * np.sqrt(2 * np.pi * initial_sigma ** 2),
        initial_position if initial_position is not None else x[np.argmax(photon_count)],
        initial_sigma,
        1,
    )
    bounds = (
        (0.01, None),
        (np.min(x), np.max(x)),
        (pixel_size, 10 * pixel_size),
        (np.finfo(float).eps, None),
    )
    result = _mle_optimize(initial_guess, expectation_fun, derivatives_fun, photon_count, bounds)
    return result
