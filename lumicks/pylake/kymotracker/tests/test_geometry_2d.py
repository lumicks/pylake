import numpy as np
import scipy
import pytest

from lumicks.pylake.kymotracker.detail.linalg_2d import (
    eigenvalues_2d_symmetric,
    eigenvector_2d_symmetric,
)
from lumicks.pylake.kymotracker.detail.geometry_2d import (
    get_candidate_generator,
    calculate_image_geometry,
)


def test_eigen_2d():
    def test_eigs(a, b, d):
        a = np.array(a)
        b = np.array(b)
        d = np.array(d)
        np_eigen_values, np_eigen_vectors = np.linalg.eig([[a, b], [b, d]])
        pl_eigen_values = np.sort(eigenvalues_2d_symmetric(a, b, d))

        # Test whether eigen values are correct
        i = np.argsort(np_eigen_values)
        np.testing.assert_allclose(
            np_eigen_values[i],
            pl_eigen_values,
            rtol=1e-6,
            err_msg=f"Eigen values invalid. Calculated {pl_eigen_values}, "
            f"expected: {np_eigen_vectors} ",
        )

        # Test whether eigen vectors are correct
        vs = [np.array(eigenvector_2d_symmetric(a, b, d, x)) for x in pl_eigen_values]

        np.testing.assert_allclose(
            abs(np.dot(vs[0], np_eigen_vectors[:, i[0]])),
            1.0,
            rtol=1e-6,
            err_msg="First eigen vector invalid",
        )
        np.testing.assert_allclose(
            abs(np.dot(vs[1], np_eigen_vectors[:, i[1]])),
            1.0,
            rtol=1e-6,
            err_msg="Second eigen vector invalid",
        )

    def np_eigenvalues(a, b, d):
        eig1 = np.empty(a.shape)
        eig2 = np.empty(a.shape)
        ex = np.empty(a.shape)
        ey = np.empty(a.shape)
        for x in np.arange(a.shape[0]):
            for y in np.arange(a.shape[1]):
                np_eigen_values, np_eigen_vectors = np.linalg.eig(
                    np.array([[a[x, y], b[x, y]], [b[x, y], d[x, y]]])
                )
                idx = np_eigen_values.argsort()
                np_eigen_values.sort()
                eig1[x, y] = np_eigen_values[0]
                eig2[x, y] = np_eigen_values[1]

                ex[x, y] = np_eigen_vectors[0, idx[0]]
                ey[x, y] = np_eigen_vectors[1, idx[0]]

        return np.stack((eig1, eig2), axis=len(eig1.shape)), ex, ey

    test_eigs(3, 4, 8)
    test_eigs(3, 0, 4)
    test_eigs(3, 4, 0)
    test_eigs(0, 4, 0)
    test_eigs(0, 0, 0)
    test_eigs(1, 1, 0)
    test_eigs(-0.928069046998319, 0.9020129898294712, -0.9280690469983189)
    test_eigs(0.000001, -1, 0.000001)
    test_eigs(0.000001, -0.000001, 0.00001)

    a = np.array([[3, 3, 3], [3, 0, 0], [3, 0, 0]])
    b = np.array([[4, 0, 0], [4, 4, 4], [3, 0, 0]])
    d = np.array([[8, 4, 4], [0, 0, 0], [3, 0, 0]])

    eigenvalues = eigenvalues_2d_symmetric(a, b, d)
    np_eigenvalues, np_eigenvector_x, np_eigenvector_y = np_eigenvalues(a, b, d)

    eigenvalues.sort(axis=-1)
    eigenvector_x, eigenvector_y = eigenvector_2d_symmetric(a, b, d, eigenvalues[:, :, 0])

    # Given that there are some zeroes, we should include an absolute tolerance.
    np.testing.assert_allclose(eigenvalues, np_eigenvalues, rtol=1e-6, atol=1e-14)

    # Eigen vectors have to point in the same direction, but are not necessarily the same sign
    np.testing.assert_allclose(
        np.abs(np_eigenvector_x * eigenvector_x + np_eigenvector_y * eigenvector_y),
        np.ones(a.shape),
    )


@pytest.mark.parametrize(
    "loc,scale,sig_x,sig_y,transpose",
    [
        (25.25, 2, 3, 3, False),
        (25.45, 2, 3, 3, False),
        (25.65, 2, 3, 3, False),
        (25.85, 2, 3, 3, False),
        (25.25, 2, 3, 3, True),
        (25.45, 2, 3, 3, True),
        (25.65, 2, 3, 3, True),
        (25.85, 2, 3, 3, True),
    ],
)
def test_position_determination(loc, scale, sig_x, sig_y, transpose, tol=1e-2):
    data = np.tile(0.0001 + scipy.stats.norm.pdf(np.arange(0, 50, 1), loc=loc, scale=scale), (5, 1))
    if transpose:
        data = data.transpose()
    max_derivative, normals, positions, inside = calculate_image_geometry(data, sig_x, sig_y)

    if transpose:
        assert np.abs(positions[round(loc), 3, 0] - (loc - round(loc))) < tol
        assert inside[round(loc), 3] == 1
    else:
        assert np.abs(positions[3, round(loc), 1] - (loc - round(loc))) < tol
        assert inside[3, round(loc)] == 1


def test_geometry():
    sig_x = 2.0 / np.sqrt(3.0)
    sig_y = 2.0 / np.sqrt(3.0)

    # Test vectors obtained from the receptive fields
    # First coordinate changes, hence we expect the normal to point in the direction of the second
    data = np.zeros((5, 5))
    data[1, 2] = 10
    data[2, 2] = 10
    data[3, 2] = 10
    max_derivative, normals, positions, inside = calculate_image_geometry(data, sig_x, sig_y)
    assert normals[2, 2][0] == 0
    assert np.abs(normals[2, 2][1]) > 0

    # Second coordinate changes, expect vector to point in direction of the first
    data = np.zeros((5, 5))
    data[2, 1] = 10
    data[2, 2] = 10
    data[2, 3] = 10
    max_derivative, normals, positions, inside = calculate_image_geometry(data, sig_x, sig_y)
    assert normals[2, 2][1] == 0
    assert np.abs(normals[2, 2][0]) > 0

    # Diagonal line y=x, expect normal's coordinates to have different signs
    data = np.zeros((5, 5))
    data[1, 1] = 10
    data[2, 2] = 10
    data[3, 3] = 10
    max_derivative, normals, positions, inside = calculate_image_geometry(data, sig_x, sig_y)
    np.testing.assert_allclose(normals[2, 2][1], -normals[2, 2][0])

    # Diagonal line y=x, expect normal's coordinates to have same sign
    data = np.zeros((5, 5))
    data[3, 1] = 10
    data[2, 2] = 10
    data[1, 3] = 10
    max_derivative, normals, positions, inside = calculate_image_geometry(data, sig_x, sig_y)
    np.testing.assert_allclose(normals[2, 2][1], normals[2, 2][0])


def test_candidates():
    candidates = get_candidate_generator()

    normal_angle = (-22.4 - 90) * np.pi / 180
    np.testing.assert_allclose(candidates(normal_angle)[0], np.array([1, -1]))
    np.testing.assert_allclose(candidates(normal_angle)[1], np.array([1, 0]))
    np.testing.assert_allclose(candidates(normal_angle)[2], np.array([1, 1]))

    normal_angle = (22.4 - 90) * np.pi / 180
    np.testing.assert_allclose(candidates(normal_angle)[0], np.array([1, -1]))
    np.testing.assert_allclose(candidates(normal_angle)[1], np.array([1, 0]))
    np.testing.assert_allclose(candidates(normal_angle)[2], np.array([1, 1]))

    normal_angle = (-22.6 - 90) * np.pi / 180
    assert not np.allclose(candidates(normal_angle)[0], np.array([1, -1]))
    assert not np.allclose(candidates(normal_angle)[1], np.array([1, 0]))
    assert not np.allclose(candidates(normal_angle)[2], np.array([1, 1]))

    normal_angle = (22.6 - 90) * np.pi / 180
    assert not np.allclose(candidates(normal_angle)[0], np.array([1, -1]))
    assert not np.allclose(candidates(normal_angle)[1], np.array([1, 0]))
    assert not np.allclose(candidates(normal_angle)[2], np.array([1, 1]))

    for normal_angle in np.arange(-np.pi, np.pi, np.pi / 100):
        options = candidates(normal_angle)
        assert len(options) == 3

        # Check if the options are adjacent to the center cell
        assert np.max(np.max(np.abs(options))) == 1, print(options)

        # Check if the options are perpendicular to the direction we were sent in.
        # Normal will be at cos(angle), sin(angle). Rotate by 90 degrees, results in -sin(angle),
        # cos(angle)
        direction = np.array([-np.sin(normal_angle), np.cos(normal_angle)])
        direction = np.sign(np.round(direction))

        np.testing.assert_allclose(
            np.sort([np.max(np.abs(direction - option)) for option in options]), [0, 1, 1]
        ), f"Failed for normal angle {normal_angle} / direction {direction} => {options}"
