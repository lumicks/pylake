import numpy as np
from functools import partial

from lumicks.pylake.kymotracker.detail import gaussian_mle
from lumicks.pylake.fitting.detail.derivative_manipulation import numerical_jacobian


def test_likelihood(gaussian_1d):
    coordinates, expectation, photon_count, [parameters, *_] = gaussian_1d
    expectation_fcn = partial(gaussian_mle.peak_expectation_1d, coordinates, pixel_size=parameters.pixel_size)

    p = np.array([parameters.total_photons, parameters.center, parameters.width, parameters.background])
    log_like = gaussian_mle.poisson_log_likelihood(p, expectation_fcn, photon_count[:,0])
    assert np.allclose(log_like, 41.739248864299796)


def test_likelihood_jacobian(gaussian_1d):
    coordinates, expectation, photon_count, [parameters, *_] = gaussian_1d
    expectation_fcn = partial(gaussian_mle.peak_expectation_1d, coordinates, pixel_size=parameters.pixel_size)
    derivatives_fcn = partial(gaussian_mle.peak_expectation_1d_derivatives, coordinates, pixel_size=parameters.pixel_size)

    p = np.array([parameters.total_photons, parameters.center, parameters.width, parameters.background])
    d_log_like = gaussian_mle.poisson_log_likelihood_jacobian(p, expectation_fcn, derivatives_fcn, photon_count[:,0])
    assert np.allclose(d_log_like, [-0.007709675671345822, 0.8178342688946003, -17.876743019790844, 15.385483783567295])

    f = lambda pp: np.array([gaussian_mle.poisson_log_likelihood(pp, expectation_fcn, photon_count[:,0])])
    trial_p = p * [0.8, 1.1, 1.25, 0.85]
    num_result = numerical_jacobian(f, trial_p, dx=1e-6).squeeze()
    test_result = gaussian_mle.poisson_log_likelihood_jacobian(trial_p, expectation_fcn, derivatives_fcn, photon_count[:,0])
    assert np.allclose(num_result, test_result)


def test_gaussian_1d(gaussian_1d):
    coordinates, expectation, photon_count, [parameters, *_] = gaussian_1d
    expectation_fcn = partial(gaussian_mle.peak_expectation_1d, coordinates, pixel_size=parameters.pixel_size)
    p = np.array([parameters.total_photons, parameters.center, parameters.width, parameters.background])
    assert np.allclose(expectation_fcn(*p), expectation)

    result = gaussian_mle.gaussian_mle_1d(coordinates, photon_count[:,0], parameters.pixel_size)
    assert np.allclose(result.x, [54.3777483, 3.50778329,  0.2836636, 0.80622237])
    assert np.allclose(result.x, p, rtol=0.2)
