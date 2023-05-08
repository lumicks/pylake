import pytest
import numpy as np
from functools import partial

from lumicks.pylake.kymotracker.detail import gaussian_mle
from lumicks.pylake.fitting.detail.derivative_manipulation import numerical_jacobian
from lumicks.pylake.kymotracker.detail.gaussian_mle import overlapping_pixels


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
            (5.049623664367611, 47.70151829477461, 0.4837961928139574, 2.7456452253505845),
            (3.386815247330237, 129.7348658314938, 0.5745882898374802, 2.7456452253505845),
        ),
        (
            (4.929305738020595, 45.94006945227121, 0.3130773023551995, 2.894367526470737),
            (3.5004643610433916, 105.6217619306042, 0.42403450347720467, 2.894367526470737),
        ),
        (
            (4.786900845519071, 57.04561184476914, 0.46766482862726744, 2.850299825221769),
            (3.474928744123804, 104.92093886285318, 0.5773045443247272, 2.850299825221769),
        ),
        (
            (4.588347916057934, 70.2003361271053, 0.5905108701664894, 2.9526205824012703),
            (3.5342391799857973, 71.5386123351624, 0.452419765001986, 2.9526205824012703),
        ),
        (
            (4.897119242991173, 52.35976694214814, 0.312228022953662, 3.177890841264013),
            (3.4867976853741123, 103.84912200797376, 0.4829434399566099, 3.177890841264013),
        ),
    ]

    coordinates, expectation, photon_count, [params, params2] = two_gaussians_1d
    for counts, peaks in zip(photon_count.T, results):
        result = gaussian_mle.gaussian_mle_1d(
            coordinates, counts, params.pixel_size, initial_position=[params.center, params2.center]
        )
        assert np.allclose(result[0], peaks[0])
        assert np.allclose(result[1], peaks[1])
