import pytest

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
