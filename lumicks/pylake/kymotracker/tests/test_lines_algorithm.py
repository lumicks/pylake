from copy import deepcopy

import numpy as np
import pytest

from lumicks.pylake.kymo import _kymo_from_array
from lumicks.pylake.kymotracker.kymotracker import track_lines, _interp_to_frame
from lumicks.pylake.tests.data.mock_confocal import generate_kymo
from lumicks.pylake.kymotracker.detail.geometry_2d import get_candidate_generator
from lumicks.pylake.kymotracker.detail.trace_line_2d import detect_lines, _traverse_line_direction


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
    normals[:, :, 0] = -np.eye(n) * 1.0 / np.sqrt(2)
    normals[:, :, 1] = np.eye(n) * 1.0 / np.sqrt(2)
    normals[hx, :, 0] = 1
    normals[hx, hx, 0] = -1.0 / np.sqrt(2)
    normals[hx, hx, 1] = 1.0 / np.sqrt(2)

    candidates = get_candidate_generator()
    np.testing.assert_allclose(
        _traverse_line_direction(
            [0, 0], deepcopy(a), positions, normals, -0.5, 1, candidates, 1, True
        ),
        np.array([[0, 0], [1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [6, 6]]),
    )
    np.testing.assert_allclose(
        _traverse_line_direction(
            [n - 1, n - 1], deepcopy(a), positions, normals, -0.5, 1, candidates, -1, True
        ),
        np.array([[6, 6], [5, 5], [4, 4], [3, 3], [2, 2], [1, 1], [0, 0]]),
    )
    np.testing.assert_allclose(
        _traverse_line_direction(
            [hx, 0], deepcopy(a), positions, normals, -0.5, 1, candidates, 1, True
        ),
        np.array([[hx, 0], [hx, 1], [hx, 2], [hx, 3], [4, 4], [5, 5], [6, 6]]),
    )

    # Test whether the threshold is enforced
    np.testing.assert_allclose(
        _traverse_line_direction(
            [0, 0], deepcopy(a), positions, normals, -1.5, 1, candidates, 1, True
        ),
        np.array([[0, 0], [1, 1], [2, 2]]),
    )


def test_uni_directional():
    data = np.zeros((100, 100)) + 0.0001
    for i in np.arange(634):
        for j in np.arange(25, 35, 0.5):
            data[int(50 + j * np.sin(0.01 * i)), int(50 + j * np.cos(0.01 * i))] = 1

    def detect(min_length, force_dir):
        lines = detect_lines(
            data,
            6,
            max_lines=5,
            start_threshold=0.005,
            continuation_threshold=0.095,
            angle_weight=1,
            force_dir=force_dir,
        )

        return [line for line in lines if len(line) > min_length]

    assert len(detect(5, True)) == 2
    assert len(detect(5, False)) == 1


def test_kymotracker_test_bias_rect_lines():
    """Computing the kymograph of a subset of the image should not affect the results of the
    tracking. If this test fires, it means that kymotracking on a subset of the image does not
    produce the same result as on the full thing for :func:`~lumicks.pylake.track_lines()`.
    """

    img_data = np.array(
        [
            np.array([1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 6, 0, 1, 0]),
            np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 6, 1, 0, 1]),
        ]
    )
    img_data = np.tile(img_data.T, (1, 7))

    kymo = generate_kymo("chan", img_data, dt=int(0.01e9), pixel_size_nm=1e3)

    tracking_settings = {"line_width": 3.0, "max_lines": 0}
    traces_rect = track_lines(kymo, "red", **tracking_settings, rect=[[0, 2], [22, 12]])
    traces_full = track_lines(kymo, "red", **tracking_settings)

    for t1, t2 in zip(traces_rect, traces_full):
        np.testing.assert_allclose(t1.position, t2.position)


def test_kymotracker_subset_test_lines(kymo_integration_test_data):
    """If this test fires, it likely means that either the coordinates are not coordinates w.r.t.
    the original image, or that the reference to the image held by KymoLine is a reference to a
    subset of the image, while the coordinates are still in the global coordinate system."""
    line_time = kymo_integration_test_data.line_time_seconds
    pixel_size = kymo_integration_test_data.pixelsize_um[0]
    rect = [[0.0 * line_time, 15.0 * pixel_size], [30 * line_time, 30.0 * pixel_size]]

    lines = track_lines(kymo_integration_test_data, "red", 3 * pixel_size, 4, rect=rect)
    np.testing.assert_allclose(
        np.sum(lines[0].sample_from_image(1, correct_origin=True)), 40 * 10 + 6
    )


def test_kymotracker_lines_algorithm_integration_tests(kymo_integration_test_data):
    line_time = kymo_integration_test_data.line_time_seconds
    pixel_size = kymo_integration_test_data.pixelsize_um[0]

    lines = track_lines(kymo_integration_test_data, "red", 3 * pixel_size, 4)
    np.testing.assert_allclose(lines[0].coordinate_idx, [11] * len(lines[0].coordinate_idx))
    np.testing.assert_allclose(lines[1].coordinate_idx, [21] * len(lines[1].coordinate_idx))
    np.testing.assert_allclose(lines[0].time_idx, np.arange(9, 21))
    np.testing.assert_allclose(lines[1].time_idx, np.arange(14, 26))
    np.testing.assert_allclose(
        np.sum(lines[0].sample_from_image(1, correct_origin=True)), 50 * 10 + 6
    )
    np.testing.assert_allclose(
        np.sum(lines[1].sample_from_image(1, correct_origin=True)), 40 * 10 + 6
    )

    rect = [[0.0 * line_time, 15.0 * pixel_size], [30 * line_time, 30.0 * pixel_size]]
    lines = track_lines(kymo_integration_test_data, "red", 3 * pixel_size, 4, rect=rect)
    np.testing.assert_allclose(lines[0].coordinate_idx, [21] * len(lines[0].coordinate_idx))
    np.testing.assert_allclose(lines[0].time_idx, np.arange(14, 26))


def test_lines_algorithm_input_validation(kymo_integration_test_data):
    for line_width in (-1, 0):
        with pytest.raises(ValueError, match="should be larger than zero"):
            track_lines(kymo_integration_test_data, "red", line_width=line_width, max_lines=1000)


@pytest.mark.parametrize(
    "time, coord, ref_time, ref_coord",
    [
        ([1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [1, 2, 3], [1.0, 2.0, 3.0]),
        ([1.0, 2.0, 2.48], [1.0, 2.0, 2.98], [1, 2], [1.0, 2.0]),
        ([1.49, 2.0, 3.0], [1.49, 2.0, 3.0], [1, 2, 3], [1.49, 2.0, 3.0]),
        ([1.5, 2.0, 3.0], [1.5, 2.0, 3.0], [2, 3], [2.0, 3.0]),
        # Include both even and odd rounding test cases (top)
        ([0, 1.6, 2.0, 2.5], [0, 1.6, 2.0, 2.5], [0, 1, 2], [0.0, 1.0, 2.0]),
        ([0, 1.6, 2.0, 2.501], [0, 1.6, 2.0, 2.501], [0, 1, 2, 3], [0.0, 1.0, 2.0, 2.501]),
        ([0, 1.6, 3.5], [0, 1.6, 3.5], [0, 1, 2, 3], [0.0, 1.0, 2.0, 3.0]),
        ([0, 1.6, 3.501], [0, 1.6, 3.501], [0, 1, 2, 3, 4], [0.0, 1.0, 2.0, 3.0, 3.501]),
        # Include both even and odd rounding test cases (bottom)
        ([0.49, 2.501], [0.49, 2.501], [0, 1, 2, 3], [0.49, 1.0, 2.0, 2.501]),
        ([0.5, 2.501], [0.5, 2.501], [1, 2, 3], [1.0, 2.0, 2.501]),
        ([1.49, 2.501], [1.49, 2.501], [1, 2, 3], [1.49, 2.0, 2.501]),
        ([1.5, 2.501], [1.5, 2.501], [2, 3], [2.0, 2.501]),
    ],
)
def test_back_interpolation(time, coord, ref_time, ref_coord):
    interp_time, interp_coord = _interp_to_frame(time, coord)
    np.testing.assert_equal(interp_time, ref_time)
    np.testing.assert_allclose(interp_coord, ref_coord)


def test_lines_refine():
    """To test this case we make a specific Kymo that has the following pattern:

    _|_|_|_|_|_|_|_|_|_|_|
    _|_|_|X|_|_|_|_|_|_|_|
    _|_|_|_|X|_|_|_|_|_|_|
    _|_|_|_|_|X|_|_|_|_|_|
    _|_|_|_|_|_|X|_|_|_|_|
    _|_|_|_|_|_|_|X|_|_|_|
    _|_|_|_|_|_|_|_|_|_|_|
    _|_|_|_|_|_|_|X|_|_|_|
    _|_|_|_|_|_|_|_|_|_|_|

    Without refinement, the last element will simply be the lowest most point. With refinement,
    we will get the center of those two pixels.
    """
    image = np.ones((12, 11))
    image[9, 7] = 10
    for k in np.arange(3, 8):
        image[k, k] = 10

    kymo = _kymo_from_array(image, "r", line_time_seconds=0.5)
    lines = track_lines(kymo, "red", 4, 1)
    np.testing.assert_allclose(lines[0].coordinate_idx, [3.0, 4.0, 5.0, 6.0, 8.0])


def test_tracking_max_lines():
    image = np.ones((30, 20))
    for k in range(1, 7):
        image[4 * k, 2:-1] = 10

    kymo = _kymo_from_array(image, "r", line_time_seconds=0.5)
    lines = track_lines(kymo, "red", line_width=3, max_lines=100)
    assert len(lines) > 5  # Note that it finds more than 6, some single spurious noise peaks

    lines = track_lines(kymo, "red", line_width=3, max_lines=4)
    assert len(lines) == 4
