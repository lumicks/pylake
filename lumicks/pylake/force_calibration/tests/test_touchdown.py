import warnings

import numpy as np
import pytest

from lumicks.pylake.force_calibration.touchdown import (
    touchdown,
    mack_model,
    fit_piecewise_linear,
    fit_damped_sine_with_polynomial,
)


@pytest.mark.parametrize(
    "direction, surface, slope1, slope2, offset, p_value",
    [
        [0.1, 7.5, 3.0, 5.0, 7.0, 0.0],
        [0.1, 7.5, -3.0, 5.0, 7.0, 0.0],
        [0.1, 7.5, -3.0, 5.0, -7.0, 0.0],
        [0.1, 7.5, 0.0, 5.0, -7.0, 0.0],
    ],
)
@pytest.mark.filterwarnings("ignore:Denominator in F-Test is zero")
def test_piecewise_linear_fit(direction, surface, slope1, slope2, offset, p_value):
    def y_func(x):
        return (
            slope1 * (x - surface) * (x < surface)
            + slope2 * (x - surface) * (x >= surface)
            + offset
        )

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "Covariance of the parameters could not be estimated")
        warnings.filterwarnings("ignore", "Denominator in F-Test is zero")
        independent = np.arange(5.0, 10.0, 0.1)
        pars, p_value = fit_piecewise_linear(independent, y_func(independent))
        np.testing.assert_allclose(pars, [surface, offset, slope1, slope2], atol=1e-12)
        np.testing.assert_allclose(p_value, 0, atol=1e-12)

        pars, p_value = fit_piecewise_linear(np.flip(independent), np.flip(y_func(independent)))
        np.testing.assert_allclose(pars, [surface, offset, slope1, slope2], atol=1e-12)
        np.testing.assert_allclose(p_value, 0, atol=1e-12)


def test_failed_linear():
    """Touchdown relies on being able to tell a linear slope from a segmented one. Here we test
    whether we can detect whether the segmented model provides a better fit"""
    x, y = np.arange(5), np.array([1.99420679, 6.99469027, 11.99487776, 16.97811258, 22.00990092])
    pars, p_value = fit_piecewise_linear(x, y)
    np.testing.assert_allclose(p_value, 1.0)


@pytest.mark.parametrize(
    "amplitude, frequency, phase_shift, poly_coeffs",
    [
        [0.478463, 1.9889, 1.51936, [-1.26749, 8.06]],
        [1.6, 1.3, 1.51936, [-1.26749, 8.06]],
        [1.6, 1.3, 0.21936, [3.26749, -4.06]],
        [0.6, 1.6, 0.21936, [3.26749, -4.06]],
    ],
)
def test_sine_with_polynomial(amplitude, frequency, phase_shift, poly_coeffs):
    h = np.arange(2.5554, 0.0654, -0.01)
    test_data = amplitude * np.sin(2.0 * np.pi * frequency * h + phase_shift) + np.polyval(
        [-1.26749, 8.06], h
    )

    par, sim = fit_damped_sine_with_polynomial(h, test_data, [0.0, 5.0], background_degree=2)
    np.testing.assert_allclose(frequency, par, atol=1e-5)
    np.testing.assert_allclose(np.sum((sim - test_data) ** 2), 0, atol=1e-8)


def simulate_touchdown(
    start_z, end_z, analysis_stepsize, mack_parameters, sample_rate=78125, analysis_rate=52
):
    """Simulate a touchdown curve

    start_z : float
        Starting nanostage z position.
    end_z : float
        End nanostage z position.
    analysis_stepsize : float
        Stage step-size we want to have after downsampling for analysis.
    mack_parameters : dict
        Model parameters for the mack model.
    sample_rate : int
        Sample rate for `nanostage` and `axial_force` signal.
    analysis_rate : int
        Sample rate used in the actual analysis.
    """
    oversampling = sample_rate // analysis_rate
    stage_positions = np.arange(start_z, end_z, analysis_stepsize / oversampling)
    return stage_positions, mack_model(nanostage_z_position=stage_positions, **mack_parameters)


def test_touchdown(mack_parameters):
    stage_positions, simulation = simulate_touchdown(99.5, 103.5, 0.01, mack_parameters)
    touchdown_result = touchdown(stage_positions, simulation, 78125)
    np.testing.assert_allclose(touchdown_result.surface_position, 101.65989249166964)
    np.testing.assert_allclose(touchdown_result.focal_shift, 0.9212834464971221)


def test_plot():
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "Covariance of the parameters could not be estimated")
        warnings.filterwarnings("ignore", "Denominator in F-Test is zero")
        warnings.filterwarnings("ignore", "Insufficient data available to reliably fit touchdown")
        touchdown_result = touchdown(
            np.array([1, 2, 3, 4]), np.array([1, 2, 3, 4]), sample_rate=1, analysis_rate=1
        )
        touchdown_result.plot()


@pytest.mark.parametrize(
    "amplitude, frequency, phase_shift",
    [
        [0.478463, 5.23242, 1.51936],
        [1.478463, 5.23242, 0.51936],
        [1.478463, 3.83242, 0.51936],
        [1.478463, 3.43242, 0.51936],
        [1.478463, 10.0, 0.51936],
    ],
)
def test_sine_fits(amplitude, frequency, phase_shift):
    x = np.arange(1, 10, 0.01)
    y = np.sin(2 * np.pi * frequency * x + phase_shift)
    par, sim = fit_damped_sine_with_polynomial(x, y, [0.0, 20.0], background_degree=0)
    np.testing.assert_allclose(par, frequency, rtol=1e-6)


@pytest.mark.parametrize(
    "decay, amplitude, frequency, phase_shift",
    [
        [2, 0.478463, 5.23242, 1.51936],
        [2, 1.478463, 5.23242, 0.51936],
        [1, 1.478463, 3.83242, 0.51936],
        [3, 1.478463, 3.43242, 0.51936],
        [0.5, 1.478463, 10.0, 0.51936],
    ],
)
def test_exp_sine_fits(decay, amplitude, frequency, phase_shift):
    x = np.arange(1, 10, 0.01)
    y = np.exp(-decay * x) * np.sin(2 * np.pi * frequency * x + phase_shift)
    par, sim = fit_damped_sine_with_polynomial(x, y, [0.0, 20.0], background_degree=0)
    np.testing.assert_allclose(par, frequency, rtol=1e-6)


@pytest.mark.filterwarnings("ignore:Covariance of the parameters could not be estimated")
@pytest.mark.filterwarnings("ignore:Surface detection failed")
def test_insufficient_data(mack_parameters):
    stage_positions, simulation = simulate_touchdown(102.5, 103.5, 0.01, mack_parameters)

    with pytest.warns(
        RuntimeWarning, match="Insufficient data available to reliably fit touchdown curve"
    ):
        touchdown_result = touchdown(stage_positions, simulation, 78125)
        assert touchdown_result.focal_shift is None


def test_fail_touchdown_too_little_data():
    stage_positions = np.arange(25.0, 40.0, 0.1 / 1502)

    with pytest.warns(
        RuntimeWarning,
        match="Surface detection failed [(]piecewise linear fit not better than linear fit[)]",
    ):
        touchdown_result = touchdown(
            stage_positions, stage_positions**2 + 100 * np.sin(10 * stage_positions), 78125
        )
        assert touchdown_result.surface_position is None

    with pytest.warns(
        RuntimeWarning,
        match="Surface detection failed [(]piecewise linear fit not better than linear fit[)]",
    ):
        touchdown_result = touchdown(stage_positions, 100 * np.sin(10 * stage_positions), 78125)
        assert touchdown_result.surface_position is None
