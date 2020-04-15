import numpy as np
from ..kymotracker.detail.linalg_2d import eigenvalues_2d_symmetric, eigenvector_2d_symmetric


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
