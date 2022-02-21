import pytest
import warnings
import numpy as np
from matplotlib.testing.decorators import cleanup
from lumicks.pylake.force_calibration.touchdown import (
    fit_piecewise_linear,
    fit_sine_with_polynomial,
    touchdown,
    mack_model,
)


@pytest.mark.parametrize(
    "direction, surface, slope1, slope2, offset",
    [
        [0.1, 7.5, 3.0, 5.0, 7.0],
        [0.1, 7.5, -3.0, 5.0, 7.0],
        [0.1, 7.5, -3.0, 5.0, -7.0],
        [0.1, 7.5, 0.0, 5.0, -7.0],
    ],
)
def test_piecewise_linear_fit(direction, surface, slope1, slope2, offset):
    def y_func(x):
        return (
            slope1 * (x - surface) * (x < surface)
            + slope2 * (x - surface) * (x >= surface)
            + offset
        )

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "Covariance of the parameters could not be estimated")
        independent = np.arange(5.0, 10.0, 0.1)
        pars = fit_piecewise_linear(independent, y_func(independent))
        np.testing.assert_allclose(pars, [surface, offset, slope1, slope2], atol=1e-12)

        pars = fit_piecewise_linear(np.flip(independent), np.flip(y_func(independent)))
        np.testing.assert_allclose(pars, [surface, offset, slope1, slope2], atol=1e-12)


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

    par, sim = fit_sine_with_polynomial(h, test_data, [0.0, 5.0], background_degree=2)
    np.testing.assert_allclose(frequency, par, atol=1e-5)
    np.testing.assert_allclose(np.sum((sim - test_data) ** 2), 0, atol=1e-8)


def test_touchdown():
    stage_positions = np.arange(100.5, 103.5, 0.01)
    simulation = mack_model(
        wavelength_nm=1064.0,
        refractive_index_medium=1.333,
        nanostage_z_position=stage_positions,
        surface_position=101.65754300557573,
        displacement_sensitivity=9.724913160609043,
        intensity_amplitude=-0.10858224326787835,
        intensity_phase_shift=1.6535670092299886,
        intensity_decay_length=0.308551871490813,
        scattering_polynomial_coeffs=[
            -0.043577454353825644,
            0.22333743993836863,
            -0.33331150250090585,
            0.1035148152731559,
        ],
        focal_shift=0.921283446497108,
        nonlinear_shift=0.0,
    )

    touchdown_result = touchdown(stage_positions, simulation)
    np.testing.assert_allclose(touchdown_result.surface_position, 101.663872)
    np.testing.assert_allclose(touchdown_result.focal_shift, 0.9282888602488462)


@cleanup
def test_plot():
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "Covariance of the parameters could not be estimated")
        touchdown_result = touchdown(np.array([1, 2, 3, 4]), np.array([1, 2, 3, 4]))
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
    par, sim = fit_sine_with_polynomial(x, y, [0.0, 20.0], background_degree=0)
    np.testing.assert_allclose(par, frequency, rtol=1e-6)
