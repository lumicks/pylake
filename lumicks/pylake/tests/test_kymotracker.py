import numpy as np
from ..kymotracker.detail.linalg_2d import eigenvalues_2d_symmetric, eigenvector_2d_symmetric
from ..kymotracker.detail.geometry_2d import calculate_image_geometry


def test_eigen_2d():
    def test_eigs(a, b, d):
        a = np.array(a)
        b = np.array(b)
        d = np.array(d)
        np_eigen_values, np_eigen_vectors = np.linalg.eig([[a, b], [b, d]])
        pl_eigen_values = np.sort(eigenvalues_2d_symmetric(a, b, d))

        # Test whether eigen values are correct
        i = np.argsort(np_eigen_values)
        assert np.allclose(np_eigen_values[i], pl_eigen_values), print(
            f"Eigen values invalid. Calculated {pl_eigen_values}, expected: {np_eigen_vectors} ")

        # Test whether eigen vectors are correct
        vs = [np.array(eigenvector_2d_symmetric(a, b, d, x)) for x in pl_eigen_values]

        assert np.allclose(abs(np.dot(vs[0], np_eigen_vectors[:, i[0]])), 1.0), print("First eigen vector invalid")
        assert np.allclose(abs(np.dot(vs[1], np_eigen_vectors[:, i[1]])), 1.0), print("Second eigen vector invalid")

    test_eigs(3, 4, 8)
    test_eigs(3, 0, 4)
    test_eigs(3, 4, 0)
    test_eigs(0, 4, 0)
    test_eigs(0, 0, 0)
    test_eigs(1, 1, 0)
    test_eigs(-0.928069046998319, 0.9020129898294712, -0.9280690469983189)
    test_eigs(.000001, -1, .000001)
    test_eigs(.000001, -.000001, .00001)


def test_position_determination(loc, scale, sig_x, sig_y, transpose, tol):
    from scipy.stats import norm

    data = np.tile(.0001 + norm.pdf(np.arange(0, 50, 1), loc=loc, scale=scale), (5, 1))
    if transpose:
        data = data.transpose()
    max_derivative, normals, positions, inside = calculate_image_geometry(data, sig_x, sig_y)

    if transpose:
        assert np.abs(positions[round(loc), 3, 0] - (loc - round(loc))) < tol
        assert inside[round(loc), 3] == 1
    else:
        assert np.abs(positions[3, round(loc), 1] - (loc - round(loc))) < tol
        assert inside[3, round(loc)] == 1


tol = 1e-2
test_position_determination(25.25, 2, 3, 3, False, tol)
test_position_determination(25.45, 2, 3, 3, False, tol)
test_position_determination(25.65, 2, 3, 3, False, tol)
test_position_determination(25.85, 2, 3, 3, False, tol)

test_position_determination(25.25, 2, 3, 3, True, tol)
test_position_determination(25.45, 2, 3, 3, True, tol)
test_position_determination(25.65, 2, 3, 3, True, tol)
test_position_determination(25.85, 2, 3, 3, True, tol)


def test_geometry():
    sig_x = 2.0 / np.sqrt(3.0)
    sig_y = 2.0 / np.sqrt(3.0)

    # Test vectors obtained from the receptive fields
    # First coordinate changes, hence we expect the normal to point in the direction of the second
    data = np.zeros((5, 5))
    data[1,2] = 10
    data[2,2] = 10
    data[3,2] = 10
    max_derivative, normals, positions, inside = calculate_image_geometry(data, sig_x, sig_y)
    assert normals[2, 2][0] == 0
    assert np.abs(normals[2, 2][1]) > 0

    # Second coordinate changes, expect vector to point in direction of the first
    data = np.zeros((5, 5))
    data[2,1] = 10
    data[2,2] = 10
    data[2,3] = 10
    max_derivative, normals, positions, inside = calculate_image_geometry(data, sig_x, sig_y)
    assert normals[2, 2][1] == 0
    assert np.abs(normals[2, 2][0]) > 0

    # Diagonal line y=x, expect normal's coordinates to have different signs
    data = np.zeros((5, 5))
    data[1,1] = 10
    data[2,2] = 10
    data[3,3] = 10
    max_derivative, normals, positions, inside = calculate_image_geometry(data, sig_x, sig_y)
    assert np.allclose(normals[2, 2][1], -normals[2, 2][0])

    # Diagonal line y=x, expect normal's coordinates to have same sign
    data = np.zeros((5, 5))
    data[3,1] = 10
    data[2,2] = 10
    data[1,3] = 10
    max_derivative, normals, positions, inside = calculate_image_geometry(data, sig_x, sig_y)
    assert np.allclose(normals[2, 2][1], normals[2, 2][0])
