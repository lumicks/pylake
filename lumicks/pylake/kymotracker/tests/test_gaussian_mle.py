from functools import partial

import numpy as np
import pytest

from lumicks.pylake.kymotracker.detail import gaussian_mle
from lumicks.pylake.kymotracker.detail.gaussian_mle import (
    overlapping_pixels,
    _estimation_parameters_simultaneous,
)
from lumicks.pylake.fitting.detail.derivative_manipulation import numerical_jacobian


@pytest.mark.parametrize(
    "positions,width,output",
    [
        [[], 1, []],
        [[35, 50], 3, [[0], [1]]],
        [[35, 50], 10, [[0, 1]]],
        [[2, 6], 1, [[0], [1]]],
        [[2, 5], 1, [[0], [1]]],
        [[2, 4], 1, [[0, 1]]],
        [[2, 8], 2, [[0], [1]]],
        [[2, 7], 2, [[0], [1]]],
        [[2, 6], 2, [[0, 1]]],
        [[2, 10], 3, [[0], [1]]],
        [[2, 9], 3, [[0], [1]]],
        [[2, 8], 3, [[0, 1]]],
        [[3, 4, 5], 1, [[0, 1, 2]]],
        [[3, 5, 7], 1, [[0, 1, 2]]],
        [[3, 6, 7], 1, [[0], [1, 2]]],
        [[3, 6, 9], 1, [[0], [1], [2]]],
        # test order
        [[4, 6, 3, 5], 1, [[2, 0, 3, 1]]],
        [[6, 3, 1], 1, [[2, 1], [0]]],
        [[4, 7, 2], 1, [[2, 0], [1]]],
        [[6, 4, 3, 5], 1, [[2, 1, 3, 0]]],
        [[18, 8, 10, 22, 24], 2, [[1, 2], [0, 3, 4]]],
    ],
)
def test_overlapping_pixels(positions, width, output):
    result = overlapping_pixels(positions, width)
    assert len(result) == len(output), (
        f"Input: {positions} / {width}, Expected: {output}, "
        f"Got {overlapping_pixels(positions, width)}"
    )

    for p, q in zip(result, output):
        assert np.all(p == q), (
            f"Input: {positions} / {width}, Expected: {output}, "
            f"Got {overlapping_pixels(positions, width)}"
        )


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
    assert np.allclose(result, [3.50778329, 54.3777483, 0.2836636, 0.80622237])
    assert np.allclose(result, p[[1, 0, 2, 3]], rtol=0.2)


def test_gaussian_1d_fixed_offset(gaussian_1d):
    """Providing the true background we get closer to the truth values of 50, 3.5 and 0.25"""
    coordinates, expectation, photon_count, [parameters, *_] = gaussian_1d
    result = gaussian_mle.gaussian_mle_1d(
        coordinates,
        photon_count[:, 0],
        fixed_background=1.0,
        pixel_size=parameters.pixel_size,
    )
    assert np.allclose(result, [3.50342814, 51.46192613, 0.272388, 1.0])


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

    p = np.array(
        [
            param1.total_photons,
            param1.center,
            param1.width,
            param2.total_photons,
            param2.center,
            param2.width,
            param1.background + param2.background,
        ]
    )

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
    results = [
        (
            (5.0496279177192696, 47.702216189137744, 0.4838043219100664, 2.7456107735641186),
            (3.3868117401535005, 129.73484102160324, 0.5745921025399469, 2.7456107735641186),
        ),
        (
            (4.929325534055994, 45.94223242123102, 0.31309274959446765, 2.894382749438883),
            (3.5004575903825286, 105.61894719746905, 0.4240287734219947, 2.894382749438883),
        ),
        (
            (4.638319692346674, 75.19624738527664, 0.559119416956572, 2.8428257609186396),
            (3.362889409366999, 87.7363514000491, 0.5215512990465202, 2.8428257609186396),
        ),
        (
            (4.702762509587468, 58.09433012021379, 0.5237418364070028, 2.958675741403134),
            (3.592267806357643, 83.02862851678475, 0.47312921286887555, 2.958675741403134),
        ),
        (
            (4.897131422972051, 52.35521545574318, 0.3122140080829003, 3.177850712658297),
            (3.4868007809654684, 103.85573245951218, 0.482927090038521, 3.177850712658297),
        ),
    ]

    coordinates, expectation, photon_count, [params, params2] = two_gaussians_1d
    for counts, peaks in zip(photon_count.T, results):
        result = gaussian_mle.gaussian_mle_1d(
            coordinates, counts, params.pixel_size, initial_position=[params.center, params2.center]
        )
        assert np.allclose(result[0], peaks[0])
        assert np.allclose(result[1], peaks[1])


@pytest.mark.parametrize(
    "x, photon_count, sorted_initial_position, ref_pos, ref_bounds, ref_max",
    [
        # fmt:off
        # Happy path 1 element
        (np.arange(5), [0, 1, 0, 2, 0], [2], [2], [[0, 4]], [2]),
        # Happy path 2 elements
        (np.arange(5), [0, 1, 0, 2, 0], [1, 3], [1, 3], [[0, 2], [2, 4]], [1, 2]),
        # Happy path 3 elements
        (np.arange(5), [0, 1, 0, 2, 0], [1, 2, 3], [1, 2, 3], [[0, 1.5], [1.5, 2.5], [2.5, 4]], [1, 2, 2]),
        # Make sure left-side oob initials get clamped and bounds make sense
        (np.arange(5), [0, 1, 0, 2, 0], [-1, 3], [0, 3], [[0, 1.5], [1.5, 4]], [1, 2]),
        # Make sure right-side oob initials get clamped and bounds make sense
        (np.arange(5), [0, 1, 0, 2, 0], [1, 5], [1, 4], [[0, 2.5], [2.5, 4]], [2, 2]),
        # Going further out of bounds doesn't change anything
        (np.arange(5), [0, 1, 0, 2, 0], [1, 6], [1, 4], [[0, 2.5], [2.5, 4]], [2, 2]),
        # Multiple on the left side end up clamped to zero
        (np.arange(5), [0, 1, 0, 2, 0], [-1, -1, -1], [0, 0, 0], [[0, 0], [0, 0], [0, 4]], [0, 0, 2]),
        # Multiple on the right side end up clamped to max
        (np.arange(5), [0, 1, 0, 2, 0], [6, 6, 6], [4, 4, 4], [[0, 4], [4, 4], [4, 4]], [2, 0, 0]),
        # fmt:on
    ],
)
def test_simultaneous_fit_guess(
    x, photon_count, sorted_initial_position, ref_pos, ref_bounds, ref_max
):
    pos, bounds, max_photons = _estimation_parameters_simultaneous(
        x, np.array(photon_count), np.array(sorted_initial_position)
    )

    np.testing.assert_allclose(pos, ref_pos)
    np.testing.assert_allclose(max_photons, ref_max)
    np.testing.assert_allclose(bounds, ref_bounds)
