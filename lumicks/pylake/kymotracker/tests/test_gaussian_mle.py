import pytest
import numpy as np
from functools import partial

from lumicks.pylake.kymotracker.detail import gaussian_mle
from lumicks.pylake.fitting.detail.derivative_manipulation import numerical_jacobian


def test_likelihood(gaussian_1d):
    coordinates, expectation, photon_count, [params, *_] = gaussian_1d
    expectation_fcn = partial(
        gaussian_mle.peak_expectation_1d,
        coordinates,
        pixel_size=params.pixel_size,
        fixed_background=None,
        num_peaks=1,
    )

    p = np.array([params.total_photons, params.center, params.width, params.background])
    log_like = gaussian_mle.poisson_log_likelihood(p, expectation_fcn, photon_count[:, 0])
    assert np.allclose(log_like, 41.739248864299796)


def test_likelihood_jacobian(gaussian_1d):
    coordinates, expectation, photon_count, [params, *_] = gaussian_1d
    args = {"pixel_size": params.pixel_size, "fixed_background": None, "num_peaks": 1}
    expectation_fcn = partial(gaussian_mle.peak_expectation_1d, coordinates, **args)
    derivatives_fcn = partial(gaussian_mle.peak_expectation_1d_derivatives, coordinates, **args)

    p = np.array([params.total_photons, params.center, params.width, params.background])
    d_log_like = gaussian_mle.poisson_log_likelihood_jacobian(
        p, expectation_fcn, derivatives_fcn, photon_count[:, 0]
    )
    assert np.allclose(
        d_log_like,
        [-0.007709675671345822, 0.8178342688946003, -17.876743019790844, 15.385483783567295],
    )

    f = lambda pp: np.array(
        [gaussian_mle.poisson_log_likelihood(pp, expectation_fcn, photon_count[:, 0])]
    )
    trial_p = p * [0.8, 1.1, 1.25, 0.85]
    num_result = numerical_jacobian(f, trial_p, dx=1e-6).squeeze()
    test_result = gaussian_mle.poisson_log_likelihood_jacobian(
        trial_p, expectation_fcn, derivatives_fcn, photon_count[:, 0]
    )
    assert np.allclose(num_result, test_result)


def test_gaussian_1d(gaussian_1d):
    coordinates, expectation, photon_count, [params, *_] = gaussian_1d
    args = {"pixel_size": params.pixel_size, "fixed_background": None, "num_peaks": 1}
    expectation_fcn = partial(gaussian_mle.peak_expectation_1d, coordinates, **args)
    p = np.array([params.total_photons, params.center, params.width, params.background])
    assert np.allclose(expectation_fcn(p), expectation)

    result = gaussian_mle.gaussian_mle_1d(coordinates, photon_count[:, 0], params.pixel_size)
    assert np.allclose(result, [54.3777483, 3.50778329, 0.2836636, 0.80622237])
    assert np.allclose(result, p, rtol=0.2)


def test_gaussian_1d_fixed_offset(gaussian_1d):
    """Providing the true background we get closer to the truth values of 50, 3.5 and 0.25"""
    coordinates, expectation, photon_count, [parameters, *_] = gaussian_1d
    result = gaussian_mle.gaussian_mle_1d(
        coordinates,
        photon_count[:, 0],
        fixed_background=1.0,
        pixel_size=parameters.pixel_size,
    )
    assert np.allclose(result, [51.46192613, 3.50342814, 0.272388, 1.0])


def test_gaussian_1d_invalid_background_value():
    with pytest.raises(ValueError, match="Fixed background should be larger than zero"):
        gaussian_mle.gaussian_mle_1d([], [], fixed_background=-1.0, pixel_size=1.0)

    with pytest.raises(ValueError, match="Fixed background should be larger than zero"):
        gaussian_mle.gaussian_mle_1d([], [], fixed_background=0.0, pixel_size=1.0)


def test_two_component_jacobian(two_gaussians_1d):
    coordinates, expectation, photon_count, [param1, param2] = two_gaussians_1d
    args = {"pixel_size": param1.pixel_size, "fixed_background": None, "num_peaks": 2}
    expectation_fcn = partial(gaussian_mle.peak_expectation_1d, coordinates, **args)
    derivatives_fcn = partial(gaussian_mle.peak_expectation_1d_derivatives, coordinates, **args)

    p = np.array([
        param1.total_photons,
        param1.center,
        param1.width,
        param2.total_photons,
        param2.center,
        param2.width,
        param1.background + param2.background
    ])

    f = lambda pp: np.array(
        [gaussian_mle.poisson_log_likelihood(pp, expectation_fcn, photon_count[:, 0])]
    )
    trial_p = p * [0.8, 1.1, 1.25, 1.1, 0.8, 0.95, 0.85]
    num_result = numerical_jacobian(f, trial_p, dx=1e-6).squeeze()
    test_result = gaussian_mle.poisson_log_likelihood_jacobian(
        trial_p, expectation_fcn, derivatives_fcn, photon_count[:, 0]
    )
    assert np.allclose(num_result, test_result)


def test_two_gaussians_1d(two_gaussians_1d):
    coordinates, expectation, photon_count, [params, params2] = two_gaussians_1d
    result = gaussian_mle.gaussian_mle_1d(
        coordinates,
        photon_count[:, 0],
        params.pixel_size,
        initial_position=[params.center, params2.center]
    )
    assert np.allclose(result, [
        [122.51841130812686, 3.364875418032175, 0.5090362675785531, 2.842552407036063],
        [56.22762877317684, 4.980369459818554, 0.24417742594275202, 2.842552407036063],
    ])
