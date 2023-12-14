import pytest

from lumicks.pylake.force_calibration.calibration_models import coupling_correction_2d
from lumicks.pylake.force_calibration.detail.drag_models import *


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


def test_coupling_solution_coefficients():
    """For same bead radii, the terms bn and dn should vanish"""
    n = 2
    a, alpha, beta = to_curvilinear_coordinates(0.5, 0.5, 6.0)
    k = calculate_k(n, a)
    delta = calculate_delta(n, alpha, beta)
    np.testing.assert_allclose(calculate_bn(n, k, alpha, beta, delta), 0)
    np.testing.assert_allclose(calculate_dn(n, k, alpha, beta, delta), 0)

    a, alpha, beta = to_curvilinear_coordinates(0.5, 0.6, 6.0)
    k = calculate_k(n, a)
    delta = calculate_delta(n, alpha, beta)
    assert np.abs(calculate_bn(n, k, alpha, beta, delta)) > 0
    assert np.abs(calculate_dn(n, k, alpha, beta, delta)) > 0


# Validate correctness
@pytest.mark.parametrize(
    "distance, r1, r2",
    [
        [3.0e-6, 0.5e-6, 0.5e-6],
        [6.0e-6, 0.5e-6, 0.8e-6],
        [6.0e-6, 0.9e-6, 0.8e-6],
        [2.0e-6, 1.0e-6, 0.9999e-6],
    ],
)
def test_coupling_solution(distance, r1, r2):
    """The coefficients an, bn, cn and dn are the solution of a system of equations presented in
    equation 26 of Stimson et al. Here we verify whether those hold."""
    a, alpha, beta = to_curvilinear_coordinates(r1, r2, distance)
    n = 2
    k = calculate_k(n, a)
    delta = calculate_delta(n, alpha, beta)
    an = calculate_an(n, k, alpha, beta, delta)
    bn = calculate_bn(n, k, alpha, beta, delta)
    cn = calculate_cn(n, k, alpha, beta, delta)
    dn = calculate_dn(n, k, alpha, beta, delta)

    eq1a = (
        an * np.cosh((n - 0.5) * alpha)
        + bn * np.sinh((n - 0.5) * alpha)
        + cn * np.cosh((n + 1.5) * alpha)
        + dn * np.sinh((n + 1.5) * alpha)
    )
    eq1b = -k * (
        (2 * n + 3) * np.exp(-(n - 0.5) * alpha) - (2 * n - 1) * np.exp(-(n + 1.5) * alpha)
    )
    eq2a = (
        an * np.cosh((n - 0.5) * beta)
        + bn * np.sinh((n - 0.5) * beta)
        + cn * np.cosh((n + 1.5) * beta)
        + dn * np.sinh((n + 1.5) * beta)
    )
    eq2b = -k * ((2 * n + 3) * np.exp((n - 0.5) * beta) - (2 * n - 1) * np.exp((n + 1.5) * beta))
    eq3a = (2 * n - 1) * (an * np.sinh((n - 0.5) * alpha) + bn * np.cosh((n - 0.5) * alpha)) + (
        2 * n + 3
    ) * (cn * np.sinh((n + 1.5) * alpha) + dn * np.cosh((n + 1.5) * alpha))
    eq3b = (2 * n - 1) * (2 * n + 3) * k * (np.exp(-(n - 0.5) * alpha) - np.exp(-(n + 1.5) * alpha))
    eq4a = (2 * n - 1) * (an * np.sinh((n - 0.5) * beta) + bn * np.cosh((n - 0.5) * beta)) + (
        2 * n + 3
    ) * (cn * np.sinh((n + 1.5) * beta) + dn * np.cosh((n + 1.5) * beta))
    eq4b = -(2 * n - 1) * (2 * n + 3) * k * (np.exp((n - 0.5) * beta) - np.exp((n + 1.5) * beta))

    np.testing.assert_allclose(eq1a, eq1b)
    np.testing.assert_allclose(eq2a, eq2b)
    np.testing.assert_allclose(eq3a, eq3b)
    np.testing.assert_allclose(eq4a, eq4b)


@pytest.mark.parametrize(
    "distance, radius1, radius2, ref_factor1, ref_factor2",
    [
        [3.0e-6, 0.5e-6, 0.5e-6, 0.8047215852074429, 0.8047215852074429],
        [6.0e-6, 0.5e-6, 0.8e-6, 0.8224250265105119, 0.8982938795749217],
        [6.0e-6, 0.9e-6, 0.8e-6, 0.8402922788501829, 0.8147951224475662],
        [2.0e-6, 1.0e-6, 0.9999e-6, 0.01958146863802238, 0.019580489165418463],
    ],
)
def test_coupling_factors(distance, radius1, radius2, ref_factor1, ref_factor2):
    f1, f2 = coupling_correction_factor_stimson(radius1, radius2, distance, summands=5)
    np.testing.assert_allclose(f1, ref_factor1)
    np.testing.assert_allclose(f2, ref_factor2)


@pytest.mark.parametrize(
    "distance, radius, allow_rotation, ref_factor",
    [
        [3.0e-6, 0.5e-6, True, 0.8870592515374689],
        [6.0e-6, 0.5e-6, False, 0.9409521069093458],
        [6.0e-6, 0.5e-6, True, 0.9409201499139558],
        [6.0e-6, 0.9e-6, True, 0.8975126023400348],
        [1.0, 1.0e-6, True, 0.9999992500005626],
    ],
)
def test_coupling_factors(distance, radius, allow_rotation, ref_factor):
    factor = coupling_correction_factor_goldmann(radius, distance, allow_rotation=allow_rotation)
    np.testing.assert_allclose(factor, ref_factor)


@pytest.mark.parametrize(
    "dx, dy, radius, allow_rotation, ref_factor, vertical",
    [
        # First we test the same cases as above (consistency check)
        [0.0, 3.0e-6, 0.5e-6, True, 0.8870592515374689, False],
        [0.0, 6.0e-6, 0.5e-6, False, 0.9409521069093458, False],
        [0.0, 1.0, 1.0e-6, True, 0.9999992500005626, False],
        [3.0e-6, 0.0, 0.5e-6, True, 0.8870592515374689, True],
        [6.0e-6, 0.0, 0.5e-6, False, 0.9409521069093458, True],
        [1.0, 0.0, 1.0e-6, True, 0.9999992500005626, True],
        [3.0e-6, 0.0e-6, 0.5e-6, True, 0.8047217606696428, False],
        [0.0, 3.0e-6, 0.5e-6, True, 0.8047217606696428, True],
        # When the beads are diagonal, it doesn't matter whether we oscillate vertically or
        # horizontally.
        [3.0e-6, 3.0e-6, 0.5e-6, True, 0.8847851093315549, True],
        [-3.0e-6, -3.0e-6, 0.5e-6, True, 0.8847851093315549, True],
        [3.0e-6, -3.0e-6, 0.5e-6, True, 0.8847851093315549, True],
        [3.0e-6, 3.0e-6, 0.5e-6, True, 0.8847851093315549, True],
        [3.0e-6, 3.0e-6, 0.5e-6, True, 0.8847851093315549, False],
        [-3.0e-6, -3.0e-6, 0.5e-6, True, 0.8847851093315549, False],
        [3.0e-6, -3.0e-6, 0.5e-6, True, 0.8847851093315549, False],
        [3.0e-6, 3.0e-6, 0.5e-6, True, 0.8847851093315549, False],
    ],
)
def test_2d_coupling_factors(dx, dy, radius, allow_rotation, ref_factor, vertical):
    factor = coupling_correction_2d(
        dx, dy, bead_diameter=2 * radius, allow_rotation=allow_rotation, is_y_oscillation=vertical
    )
    np.testing.assert_allclose(factor, ref_factor)
