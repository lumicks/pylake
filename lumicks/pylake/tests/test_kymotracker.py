import pytest
import numpy as np
from lumicks.pylake.kymotracker.detail.calibrated_images import CalibratedKymographChannel
from lumicks.pylake.kymotracker.detail.linalg_2d import eigenvalues_2d_symmetric, eigenvector_2d_symmetric
from lumicks.pylake.kymotracker.detail.geometry_2d import calculate_image_geometry, get_candidate_generator
from lumicks.pylake.kymotracker.detail.trace_line_2d import _traverse_line_direction, detect_lines
from lumicks.pylake.kymotracker.detail.stitch import distance_line_to_point
from lumicks.pylake.kymotracker.stitching import stitch_kymo_lines
from lumicks.pylake.kymotracker.kymotracker import track_greedy, track_lines, filter_lines
from lumicks.pylake.kymotracker.kymoline import KymoLine, KymoLineGroup
from lumicks.pylake.kymotracker.detail.trace_line_2d import KymoLineData
from lumicks.pylake.tests.data.mock_confocal import generate_kymo
from copy import deepcopy
from scipy.stats import norm


def test_eigen_2d():
    def test_eigs(a, b, d):
        a = np.array(a)
        b = np.array(b)
        d = np.array(d)
        np_eigen_values, np_eigen_vectors = np.linalg.eig([[a, b], [b, d]])
        pl_eigen_values = np.sort(eigenvalues_2d_symmetric(a, b, d))

        # Test whether eigen values are correct
        i = np.argsort(np_eigen_values)
        np.testing.assert_allclose(np_eigen_values[i], pl_eigen_values, rtol=1e-6,
                                   err_msg=f"Eigen values invalid. Calculated {pl_eigen_values}, "
                                           f"expected: {np_eigen_vectors} ")

        # Test whether eigen vectors are correct
        vs = [np.array(eigenvector_2d_symmetric(a, b, d, x)) for x in pl_eigen_values]

        np.testing.assert_allclose(abs(np.dot(vs[0], np_eigen_vectors[:, i[0]])), 1.0, rtol=1e-6,
                                   err_msg="First eigen vector invalid")
        np.testing.assert_allclose(abs(np.dot(vs[1], np_eigen_vectors[:, i[1]])), 1.0, rtol=1e-6,
                                   err_msg="Second eigen vector invalid")

    def np_eigenvalues(a, b, d):
        eig1 = np.empty(a.shape)
        eig2 = np.empty(a.shape)
        ex = np.empty(a.shape)
        ey = np.empty(a.shape)
        for x in np.arange(a.shape[0]):
            for y in np.arange(a.shape[1]):
                np_eigen_values, np_eigen_vectors = np.linalg.eig(np.array([[a[x, y], b[x, y]], [b[x, y], d[x, y]]]))
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
    test_eigs(.000001, -1, .000001)
    test_eigs(.000001, -.000001, .00001)

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
    np.testing.assert_allclose(np.abs(np_eigenvector_x*eigenvector_x + np_eigenvector_y*eigenvector_y), np.ones(a.shape))


@pytest.mark.parametrize("loc,scale,sig_x,sig_y,transpose", [
    (25.25, 2, 3, 3, False),
    (25.45, 2, 3, 3, False),
    (25.65, 2, 3, 3, False),
    (25.85, 2, 3, 3, False),
    (25.25, 2, 3, 3, True),
    (25.45, 2, 3, 3, True),
    (25.65, 2, 3, 3, True),
    (25.85, 2, 3, 3, True),
])
def test_position_determination(loc, scale, sig_x, sig_y, transpose, tol=1e-2):
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
        # Normal will be at cos(angle), sin(angle). Rotate by 90 degrees, results in -sin(angle), cos(angle)
        direction = np.array([-np.sin(normal_angle), np.cos(normal_angle)])
        direction = np.sign(np.round(direction))

        np.testing.assert_allclose(np.sort([np.max(np.abs(direction - option)) for option in options]),
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

    positions = np.zeros((n, n, 2))
    normals = np.zeros((n, n, 2))
    normals[:, :, 0] = - np.eye(n) * 1.0 / np.sqrt(2)
    normals[:, :, 1] = np.eye(n) * 1.0 / np.sqrt(2)
    normals[hx, :, 0] = 1
    normals[hx, hx, 0] = - 1.0 / np.sqrt(2)
    normals[hx, hx, 1] = 1.0 / np.sqrt(2)

    candidates = get_candidate_generator()
    np.testing.assert_allclose(_traverse_line_direction([0, 0], deepcopy(a), positions, normals, -0.5, 1, candidates, 1, True),
                       np.array([[0, 0], [1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [6, 6]]))
    np.testing.assert_allclose(
        _traverse_line_direction([n - 1, n - 1], deepcopy(a), positions, normals, -0.5, 1, candidates, -1, True),
        np.array([[6, 6], [5, 5], [4, 4], [3, 3], [2, 2], [1, 1], [0, 0]]))
    np.testing.assert_allclose(_traverse_line_direction([hx, 0], deepcopy(a), positions, normals, -0.5, 1, candidates, 1, True),
                       np.array([[hx, 0], [hx, 1], [hx, 2], [hx, 3], [4, 4], [5, 5], [6, 6]]))

    # Test whether the threshold is enforced
    np.testing.assert_allclose(_traverse_line_direction([0, 0], deepcopy(a), positions, normals, -1.5, 1, candidates, 1, True),
                       np.array([[0, 0], [1, 1], [2, 2]]))


def test_uni_directional():
    data = np.zeros((100, 100)) + .0001
    for i in np.arange(634):
        for j in np.arange(25, 35, .5):
            data[int(50 + j * np.sin(.01 * i)), int(50 + j * np.cos(.01 * i))] = 1

    def detect(min_length, force_dir):
        lines = detect_lines(data, 6, max_lines=5, start_threshold=.005,
                             continuation_threshold=.095, angle_weight=1, force_dir=force_dir)

        return [line for line in lines if len(line) > min_length]

    assert len(detect(5, True)) == 2
    assert len(detect(5, False)) == 1


def test_distance_line_to_point():
    assert distance_line_to_point(np.array([0, 0]), np.array([0, 1]), np.array([0, 2])) == np.inf
    assert distance_line_to_point(np.array([0, 0]), np.array([0, 2]), np.array([0, 2])) == 0.0
    assert distance_line_to_point(np.array([0, 0]), np.array([1, 1]), np.array([0, 1])) == \
           np.sqrt(0.5)
    assert distance_line_to_point(np.array([0, 0]), np.array([1, 0]), np.array([0, 1])) == 1.0


def test_stitching():
    channel = CalibratedKymographChannel("test_data", np.array([[]]), 1e9, 1)

    segment_1 = KymoLine([0, 1], [0, 1], channel)
    segment_2 = KymoLine([2, 3], [2, 3], channel)
    segment_3 = KymoLine([2, 3], [0, 0], channel)
    segment_1b = KymoLine([0, 1], [0, 0], channel)
    segment_1c = KymoLine([-1, 0, 1], [0, 0, 1], channel)

    radius = 0.05
    segment_1d = KymoLine([0.0, 1.0], [radius+.01, radius+.01], channel)

    # Out of stitch range (maximum extension = 1)
    assert len(stitch_kymo_lines([segment_1, segment_3, segment_2], radius, 1, 2)) == 3

    # Out of stitch radius
    assert len(stitch_kymo_lines([segment_1d, segment_3, segment_2], radius, 2, 2)) == 3

    stitched = stitch_kymo_lines([segment_1, segment_3, segment_2], radius, 2, 2)
    assert len(stitched) == 2
    np.testing.assert_allclose(stitched[0].coordinate_idx, [0, 1, 2, 3])
    np.testing.assert_allclose(stitched[1].coordinate_idx, [0, 0])

    stitched = stitch_kymo_lines([segment_1b, segment_3, segment_2], radius, 2, 2)
    np.testing.assert_allclose(stitched[0].coordinate_idx, [0, 0, 0, 0])
    np.testing.assert_allclose(stitched[0].time_idx, [0, 1, 2, 3])
    np.testing.assert_allclose(stitched[1].coordinate_idx, [2, 3])

    # Check whether only the last two points are used (meaning we extrapolate [0, 0], [1, 1])
    stitched = stitch_kymo_lines([segment_1c, segment_3, segment_2], radius, 2, 2)
    np.testing.assert_allclose(stitched[0].coordinate_idx, [0, 0, 1, 2, 3])
    np.testing.assert_allclose(stitched[0].time_idx, [-1, 0, 1, 2, 3])

    # When using all three points, we shouldn't stitch
    assert len(stitch_kymo_lines([segment_1c, segment_3, segment_2], radius, 2, 3)) == 3

    # Check whether the alignment has to work in both directions
    # - and - should connect
    assert len(stitch_kymo_lines([KymoLine([0, 1], [0, 0], channel),
                                  KymoLine([2, 2.01], [0, 0], channel)], radius, 1, 2)) == 1
    # - and | should not connect.
    assert len(stitch_kymo_lines([KymoLine([0, 1], [0, 0], channel),
                                  KymoLine([2, 2.01], [0, 1], channel)], radius, 1, 2)) == 2


def raw_test_data():
    test_data = np.ones((30, 30))
    test_data[10, 10:20] = 10
    test_data[11, 10:20] = 30
    test_data[12, 10:20] = 10

    test_data[20, 15:25] = 10
    test_data[21, 15:25] = 20
    test_data[22, 15:25] = 10
    return test_data


def kymo_integration_test_data():
    return generate_kymo("test",
                         raw_test_data(),
                         pixel_size_nm=5000,
                         start=int(4e9),
                         dt=int(5e9 / 100),
                         samples_per_pixel=3,
                         line_padding=5)


def test_kymotracker_integration_tests():
    test_data = kymo_integration_test_data()

    line_time = test_data.line_time_seconds
    pixel_size = test_data.pixelsize_um[0]

    lines = track_greedy(test_data, "red", 3 * pixel_size, 4)
    np.testing.assert_allclose(lines[0].coordinate_idx, [11] * np.ones(10))
    np.testing.assert_allclose(lines[1].coordinate_idx, [21] * np.ones(10))
    np.testing.assert_allclose(lines[0].position, [11 * pixel_size] * np.ones(10))
    np.testing.assert_allclose(lines[1].position, [21 * pixel_size] * np.ones(10))
    np.testing.assert_allclose(lines[0].time_idx, np.arange(10, 20))
    np.testing.assert_allclose(lines[1].time_idx, np.arange(15, 25))
    np.testing.assert_allclose(lines[0].seconds, np.arange(10, 20) * line_time)
    np.testing.assert_allclose(lines[1].seconds, np.arange(15, 25) * line_time)
    np.testing.assert_allclose(lines[0].sample_from_image(1), [50] * np.ones(10))
    np.testing.assert_allclose(lines[1].sample_from_image(1), [40] * np.ones(10))

    lines = track_lines(test_data, "red", 3 * pixel_size, 4)
    np.testing.assert_allclose(lines[0].coordinate_idx, [11] * len(lines[0].coordinate_idx))
    np.testing.assert_allclose(lines[1].coordinate_idx, [21] * len(lines[1].coordinate_idx))
    np.testing.assert_allclose(lines[0].time_idx, np.arange(9, 21))
    np.testing.assert_allclose(lines[1].time_idx, np.arange(14, 26))
    np.testing.assert_allclose(np.sum(lines[0].sample_from_image(1)), 50 * 10 + 6)
    np.testing.assert_allclose(np.sum(lines[1].sample_from_image(1)), 40 * 10 + 6)

    line_time = test_data.line_time_seconds
    pixel_size = test_data.pixelsize_um[0]
    rect = [[0.0 * line_time, 15.0 * pixel_size], [30 * line_time, 30.0 * pixel_size]]

    lines = track_greedy(test_data, "red", 3 * pixel_size, 4, rect=rect)
    np.testing.assert_allclose(lines[0].coordinate_idx, [21] * np.ones(10))
    np.testing.assert_allclose(lines[0].time_idx, np.arange(15, 25))

    lines = track_lines(test_data, "red", 3 * pixel_size, 4, rect=rect)
    np.testing.assert_allclose(lines[0].coordinate_idx, [21] * len(lines[0].coordinate_idx))
    np.testing.assert_allclose(lines[0].time_idx, np.arange(14, 26))


def test_regression_sample_from_image_clamp():
    """This tests for a regression that occurred in sample_from_image. When sampling the image, we
    sample pixels in a region around the line. This sampling procedure is constrained to stay within
    the image. Previously, we used the incorrect axis to clamp the coordinate.
    """
    # Sampling the bottom row of a three pixel tall image will return [0, 0] instead of [1, 3];
    # since both coordinates would be clamped to the edge of the image (sampling nothing)."""
    img = CalibratedKymographChannel("test_data", np.array([[1, 1, 1], [3, 3, 3]]).T, 1e9, 1)
    assert np.array_equal(KymoLine([0, 1], [2, 2], img).sample_from_image(0), [1, 3])


def test_kymotracker_integration_tests_subset():
    """If this test fires, it likely means that either the coordinates are not coordinates w.r.t. the original image,
    or that the reference to the image held by KymoLine is a reference to a subset of the image, while the coordinates
    are still in the global coordinate system."""
    test_data = kymo_integration_test_data()

    line_time = test_data.line_time_seconds
    pixel_size = test_data.pixelsize_um[0]
    rect = [[0.0 * line_time, 15.0 * pixel_size], [30 * line_time, 30.0 * pixel_size]]

    lines = track_greedy(test_data, "red", 3 * pixel_size, 4, rect=rect)
    np.testing.assert_allclose(lines[0].sample_from_image(1), [40] * np.ones(10))

    lines = track_lines(test_data, "red", 3 * pixel_size, 4, rect=rect)
    np.testing.assert_allclose(np.sum(lines[0].sample_from_image(1)), 40 * 10 + 6)


def test_kymotracker_test_bias_rect():
    """Computing the kymograph of a subset of the image should not affect the results of the
    tracking. If this test fires, it means that kymotracking on a subset of the image does not
    produce the same result as on the full thing for `track_greedy()`.
    """

    # Generate a checkerboard pattern with a single line that we wish to track. The line is on the
    # 12th pixel.
    img_data = np.array([np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 6, 0, 1])])
    img_data = np.tile(img_data.T, (1, 2))
    kymo = generate_kymo("chan", img_data, dt=int(0.01e9), pixel_size_nm=1e3)

    # We grab a subset of the image right beyond the bright pixel. If there's a bias induced
    # by the rectangle crop, we'll see it!
    tracking_settings = {"line_width": 3,
                         "pixel_threshold": 4,
                         "sigma": 5,
                         "window": 9}
    traces_rect = track_greedy(kymo, "red", **tracking_settings, rect=[[0, 2], [1000, 12]])
    traces_full = track_greedy(kymo, "red", **tracking_settings)

    for t1, t2 in zip(traces_rect, traces_full):
        np.testing.assert_allclose(t1.position, t2.position)


def test_kymotracker_test_bias_rect_lines():
    """Computing the kymograph of a subset of the image should not affect the results of the
    tracking. If this test fires, it means that kymotracking on a subset of the image does not
    produce the same result as on the full thing for `track_lines()`.
    """

    img_data = np.array([np.array([1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 6, 0, 1, 0]),
                         np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 6, 1, 0, 1])])
    img_data = np.tile(img_data.T, (1, 7))

    kymo = generate_kymo("chan", img_data, dt=int(0.01e9), pixel_size_nm=1e3)

    tracking_settings = {"line_width": 1.5, "max_lines": 0}
    traces_rect = track_lines(kymo, "red", **tracking_settings, rect=[[0, 2], [22, 12]])
    traces_full = track_lines(kymo, "red", **tracking_settings)

    for t1, t2 in zip(traces_rect, traces_full):
        np.testing.assert_allclose(t1.position, t2.position)


def test_filter_lines():
    channel = CalibratedKymographChannel("test_data", np.array([[]]), 1e9, 1)

    k1 = KymoLine([1, 2, 3], [1, 2, 3], channel)
    k2 = KymoLine([2, 3], [1, 2], channel)
    k3 = KymoLine([2, 3, 4, 5], [1, 2, 4, 5], channel)
    lines = KymoLineGroup([k1, k2, k3])
    assert len(filter_lines(lines, 5)) == 0
    assert all([line1 == line2 for line1, line2 in zip(filter_lines(lines, 5), [k1, k3])])
    assert all([line1 == line2 for line1, line2 in zip(filter_lines(lines, 2), [k1, k2, k3])])
