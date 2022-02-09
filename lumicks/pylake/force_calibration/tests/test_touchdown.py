import pytest
import numpy as np
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

    independent = np.arange(5.0, 10.0, 0.1)
    pars = fit_piecewise_linear(independent, y_func(independent))
    np.testing.assert_allclose(pars, [surface, offset, slope1, slope2], atol=1e-12)

    pars = fit_piecewise_linear(np.flip(independent), np.flip(y_func(independent)))
    np.testing.assert_allclose(pars, [surface, offset, slope1, slope2], atol=1e-12)


@pytest.mark.parametrize(
    "amplitude, frequency, phase_shift, poly_coeffs, f_guess",
    [
        [0.478463, 1.9889, 1.51936, [-1.26749, 8.06], 2.3],
        [1.6, 1.3, 1.51936, [-1.26749, 8.06], 1.5],
        [1.6, 1.3, 0.21936, [3.26749, -4.06], 1.5],
        [0.6, 1.6, 0.21936, [3.26749, -4.06], 1.5],
    ],
)
def test_sine_with_polynomial(amplitude, frequency, phase_shift, poly_coeffs, f_guess):
    h = np.arange(2.5554, 0.0654, -0.01)
    test_data = amplitude * np.sin(2.0 * np.pi * frequency * h + phase_shift) + np.polyval(
        [-1.26749, 8.06], h
    )

    par, sim = fit_sine_with_polynomial(h, test_data, f_guess, [0.0, 5.0], background_degree=2)
    np.testing.assert_allclose(frequency, par)
    np.testing.assert_allclose(np.sum((sim - test_data) ** 2), 0, atol=1e-12)


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

    surface_position, focal_shift = touchdown(stage_positions, simulation)
    np.testing.assert_allclose(surface_position, 101.663872)
    np.testing.assert_allclose(focal_shift, 0.9282888602488462)
