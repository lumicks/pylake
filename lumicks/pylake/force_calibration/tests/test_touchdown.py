import pytest
import numpy as np
from lumicks.pylake.force_calibration.touchdown import fit_piecewise_linear


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
