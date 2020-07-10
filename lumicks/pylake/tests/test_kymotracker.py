import pytest
import numpy as np
from ..kymotracker.detail.linalg_2d import eigenvalues_2d_symmetric, eigenvector_2d_symmetric
from ..kymotracker.detail.geometry_2d import calculate_image_geometry, get_candidate_generator
from ..kymotracker.detail.trace_line_2d import _traverse_line_direction, KymoLine
from ..kymotracker.detail.stitch import distance_line_to_point, stitch_kymo_lines
from copy import deepcopy


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


def test_subpixel_methods():
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
    assert np.allclose(normals[2, 2][1], -normals[2, 2][0])

    # Diagonal line y=x, expect normal's coordinates to have same sign
    data = np.zeros((5, 5))
    data[3, 1] = 10
    data[2, 2] = 10
    data[1, 3] = 10
    max_derivative, normals, positions, inside = calculate_image_geometry(data, sig_x, sig_y)
    assert np.allclose(normals[2, 2][1], normals[2, 2][0])


def test_candidates():
    candidates = get_candidate_generator()

    normal_angle = (-22.4 - 90) * np.pi / 180
    assert np.allclose(candidates(normal_angle)[0], np.array([1, -1]))
    assert np.allclose(candidates(normal_angle)[1], np.array([1, 0]))
    assert np.allclose(candidates(normal_angle)[2], np.array([1, 1]))

    normal_angle = (22.4 - 90) * np.pi / 180
    assert np.allclose(candidates(normal_angle)[0], np.array([1, -1]))
    assert np.allclose(candidates(normal_angle)[1], np.array([1, 0]))
    assert np.allclose(candidates(normal_angle)[2], np.array([1, 1]))

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
        # Normal will be at cos(angle), sin(angle). Rotate by 90 degrees, results in -sin(angle), cos(angle)
        direction = np.array([-np.sin(normal_angle), np.cos(normal_angle)])
        direction = np.sign(np.round(direction))

        assert np.allclose(np.sort([np.max(np.abs(direction - option)) for option in options]),
                           [0, 1, 1]), f"Failed for normal angle {normal_angle} / direction {direction} => {options}"


def test_tracing():
    """Draw a pattern like this:
             X
           X
     X X X X X
       X
     X
    with appropriate normals and verify that lines are being traced correctly."""
    n = 7
    hx = int(n / 2)
    a = -np.eye(n)
    a[:hx, :hx] = -2 * np.eye(n - hx - 1)
    a[int(n / 2), :] = -1

    positions = np.zeros((n, n))
    normals = np.zeros((n, n, 2))
    normals[:, :, 0] = - np.eye(n) * 1.0 / np.sqrt(2)
    normals[:, :, 1] = np.eye(n) * 1.0 / np.sqrt(2)
    normals[hx, :, 0] = 1
    normals[hx, hx, 0] = - 1.0 / np.sqrt(2)
    normals[hx, hx, 1] = 1.0 / np.sqrt(2)

    candidates = get_candidate_generator()
    assert np.allclose(_traverse_line_direction([0, 0], deepcopy(a), positions, normals, -0.5, 1, candidates, -1),
                       np.array([[0, 0], [1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [6, 6]]))
    assert np.allclose(
        _traverse_line_direction([n - 1, n - 1], deepcopy(a), positions, normals, -0.5, 1, candidates, 1),
        np.array([[6, 6], [5, 5], [4, 4], [3, 3], [2, 2], [1, 1], [0, 0]]))
    assert np.allclose(_traverse_line_direction([hx, 0], deepcopy(a), positions, normals, -0.5, 1, candidates, 1),
                       np.array([[hx, 0], [hx, 1], [hx, 2], [hx, 3], [4, 4], [5, 5], [6, 6]]))

    # Test whether the threshold is enforced
    assert np.allclose(_traverse_line_direction([0, 0], deepcopy(a), positions, normals, -1.5, 1, candidates, -1),
                       np.array([[0, 0], [1, 1], [2, 2]]))


def test_kymo_line():
    k1 = KymoLine(np.array([1, 2, 3]), np.array([2, 3, 4]))
    assert np.allclose(k1[1], [2, 3])
    assert np.allclose(k1[-1], [3, 4])
    assert np.allclose(k1[0:2], [[1, 2], [2, 3]])
    assert np.allclose(k1[0:2][:, 1], [2, 3])

    k2 = KymoLine(np.array([4, 5, 6]), np.array([5, 6, 7]))
    assert np.allclose((k1 + k2)[:], [[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7]])

    assert np.allclose(k1.extrapolate(True, 3, 2.0), [5, 6])

    # Need at least 2 points for linear extrapolation
    with pytest.raises(AssertionError):
        KymoLine([1], [1]).extrapolate(True, 5, 2.0)

    with pytest.raises(AssertionError):
        KymoLine([1, 2, 3], [1, 2, 3]).extrapolate(True, 1, 2.0)


def test_distance_line_to_point():
    assert distance_line_to_point(np.array([0, 0]), np.array([0, 1]), np.array([0, 2])) == np.inf
    assert distance_line_to_point(np.array([0, 0]), np.array([0, 2]), np.array([0, 2])) == 0.0
    assert distance_line_to_point(np.array([0, 0]), np.array([1, 1]), np.array([0, 1])) == np.sqrt(.5)
    assert distance_line_to_point(np.array([0, 0]), np.array([1, 0]), np.array([0, 1])) == 1.0


def test_stitching():
    segment_1 = KymoLine([0, 1], [0, 1])
    segment_2 = KymoLine([2, 3], [2, 3])
    segment_3 = KymoLine([2, 3], [0, 0])
    segment_1b = KymoLine([0, 1], [0, 0])
    segment_1c = KymoLine([-1, 0, 1], [0, 0, 1])

    radius = 0.05
    segment_1d = KymoLine([0.0, 1.0], [radius+.01, radius+.01])

    # Out of stitch range (maximum extension = 1)
    assert len(stitch_kymo_lines([segment_1, segment_3, segment_2], radius, 1, 2)) == 3

    # Out of stitch radius
    assert len(stitch_kymo_lines([segment_1d, segment_3, segment_2], radius, 2, 2)) == 3

    stitched = stitch_kymo_lines([segment_1, segment_3, segment_2], radius, 2, 2)
    assert len(stitched) == 2
    assert np.allclose(stitched[0].coordinate, [0, 1, 2, 3])
    assert np.allclose(stitched[1].coordinate, [0, 0])

    stitched = stitch_kymo_lines([segment_1b, segment_3, segment_2], radius, 2, 2)
    assert np.allclose(stitched[0].coordinate, [0, 0, 0, 0])
    assert np.allclose(stitched[0].time, [0, 1, 2, 3])
    assert np.allclose(stitched[1].coordinate, [2, 3])

    # Check whether only the last two points are used (meaning we extrapolate [0, 0], [1, 1])
    stitched = stitch_kymo_lines([segment_1c, segment_3, segment_2], radius, 2, 2)
    assert np.allclose(stitched[0].coordinate, [0, 0, 1, 2, 3])
    assert np.allclose(stitched[0].time, [-1, 0, 1, 2, 3])

    # When using all three points, we shouldn't stitch
    assert len(stitch_kymo_lines([segment_1c, segment_3, segment_2], radius, 2, 3)) == 3

    # Check whether the alignment has to work in both directions
    # - and - should connect
    assert len(stitch_kymo_lines([KymoLine([0, 1], [0, 0]), KymoLine([2, 2.01], [0, 0])], radius, 1, 2)) == 1
    # - and | should not connect.
    assert len(stitch_kymo_lines([KymoLine([0, 1], [0, 0]), KymoLine([2, 2.01], [0, 1])], radius, 1, 2)) == 2