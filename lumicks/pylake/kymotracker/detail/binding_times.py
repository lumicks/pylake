import numpy as np
from scipy.special import logsumexp
from scipy.optimize import minimize
from functools import partial


def exponential_mixture_log_likelihood_components(
    amplitudes, lifetimes, t, min_observation_time, max_observation_time
):
    """Calculate each component of the log likelihood of an exponential mixture distribution.

    The full log likelihood for a single observation is given by:
        log(L) = log( sum_i( component_i ) )

    with the output of this function being log(component_i) defined as:
        log(component_i) = log(a_i) - log(N) + log(tau_i) - t/tau_i

    where a_i and tau_i are the amplitude and lifetime of component i and N is a normalization
    factor that takes into account the minimum and maximum observation times of the experiment.

    Therefore, the full log likelihood is calculated from the output of this function by applying
    logsumexp(output, axis=0) where the summation is taken over the components.

    Parameters
    ----------
    amplitudes : array_like
        fractional amplitude parameters for each component
    lifetimes : array_like
        lifetime parameters for each component in seconds
    t : array_like
        dwelltime observations in seconds
    min_observation_time : float
        minimum observation time in seconds, generally the line/frame time of the experiment
    max_observation_time : float
        maximum observation time in seconds, generally the length of the experiment
    """
    amplitudes = amplitudes[:, np.newaxis]
    lifetimes = lifetimes[:, np.newaxis]
    t = t[np.newaxis, :]

    norm_factor = np.log(amplitudes) + np.log(
        np.exp(-min_observation_time / lifetimes) - np.exp(-max_observation_time / lifetimes)
    )
    norm_factor = -logsumexp(norm_factor, axis=0)

    components = norm_factor + np.log(amplitudes) - np.log(lifetimes) - t / lifetimes
    return components


def exponential_mixture_log_likelihood(params, t, min_observation_time, max_observation_time):
    """Calculate the log likelihood of an exponential mixture distribution.

    The full log likelihood for a single observation is given by:
        log(L) = log( sum_i( exp( log(component_i) ) ) )

    where log(component_i) is output from `exponential_mixture_log_likelihood_components()`

    Parameters
    ----------
    amplitudes : array_like
        fractional amplitude parameters for each component
    lifetimes : array_like
        lifetime parameters for each component in seconds
    t : array_like
        dwelltime observations in seconds
    min_observation_time : float
        minimum observation time in seconds, generally the line/frame time of the experiment
    max_observation_time : float
        maximum observation time in seconds, generally the length of the experiment
    """
    params = np.reshape(params, (2, -1))
    components = exponential_mixture_log_likelihood_components(
        params[0], params[1], t, min_observation_time, max_observation_time
    )
    log_likelihood = logsumexp(components, axis=0)
    return -np.sum(log_likelihood)


def _kinetic_mle_optimize(
    n_components, t, min_observation_time, max_observation_time, initial_guess=None
):
    """Calculate the maximum likelihood estimate of the model parameters given measured dwelltimes.

    Parameters
    ----------
    n_components : int
        number of components in the mixture model
    t : array_like
        dwelltime observations in seconds
    min_observation_time : float
        minimum observation time in seconds, generally the line/frame time of the experiment
    max_observation_time : float
        maximum observation time in seconds, generally the length of the experiment
    initial_guess : array_like
        initial guess for the model parameters ordered as
        [amplitude1, amplitude2, ..., lifetime1, lifetime2, ...]
    """

    cost_fun = partial(
        exponential_mixture_log_likelihood,
        t=t,
        min_observation_time=min_observation_time,
        max_observation_time=max_observation_time,
    )

    if initial_guess is None:
        initial_guess_amplitudes = np.ones(n_components) / n_components
        initial_guess_lifetimes = np.mean(t) * np.arange(1, n_components + 1)
        initial_guess = np.hstack([initial_guess_amplitudes, initial_guess_lifetimes])

    bounds = (
        *[(np.finfo(float).eps, 1) for _ in range(n_components)],
        *[(min_observation_time * 0.1, max_observation_time * 1.1) for _ in range(n_components)],
    )
    constraints = {"type": "eq", "fun": lambda x, n: 1 - sum(x[:n]), "args": [n_components]}
    result = minimize(
        cost_fun, initial_guess, method="SLSQP", bounds=bounds, constraints=constraints
    )
