import numpy as np
import pytest

from lumicks.pylake.force_calibration.detail.drag_models import (
    coth,
    cosech,
    to_curvilinear_coordinates,
)


@pytest.mark.parametrize(
    "ref_distance, ref_r1, ref_r2",
    [
        [3.0e-6, 0.5e-6, 0.5e-6],
        [6.0e-6, 0.5e-6, 0.8e-6],
        [6.0e-6, 0.9e-6, 0.8e-6],
        [2.0e-6, 1.0e-6, 0.9999e-6],
    ],
)
def test_coordinate_transform(ref_distance, ref_r1, ref_r2):
    # Validate the transformation
    a, alpha, beta = to_curvilinear_coordinates(ref_r1, ref_r2, ref_distance)
    d1 = a * coth(alpha)
    d2 = -a * coth(beta)
    r1 = a * cosech(alpha)
    r2 = -a * cosech(beta)

    np.testing.assert_allclose(d1 + d2, ref_distance)
    np.testing.assert_allclose(r1, ref_r1)
    np.testing.assert_allclose(r2, ref_r2)


def test_coordinate_transform_bad():
    with pytest.raises(
        ValueError, match="Distance between beads 1.9999 has to be bigger than their summed radii"
    ):
        to_curvilinear_coordinates(1.0, 1.0, 1.9999)
