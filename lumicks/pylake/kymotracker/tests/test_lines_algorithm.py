import pytest
import numpy as np
from copy import deepcopy
from lumicks.pylake.kymotracker.detail.trace_line_2d import _traverse_line_direction, detect_lines
from lumicks.pylake.kymotracker.kymotracker import track_lines
from lumicks.pylake.tests.data.mock_confocal import generate_kymo
from lumicks.pylake.kymotracker.detail.geometry_2d import get_candidate_generator


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
    produce the same result as on the full thing for `track_lines()`.
    """

    img_data = np.array(
        [
            np.array([1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 6, 0, 1, 0]),
            np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 6, 1, 0, 1]),
        ]
    )
    img_data = np.tile(img_data.T, (1, 7))

    kymo = generate_kymo("chan", img_data, dt=int(0.01e9), pixel_size_nm=1e3)

    tracking_settings = {"line_width": 1.5, "max_lines": 0}
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
    np.testing.assert_allclose(np.sum(lines[0].sample_from_image(1)), 40 * 10 + 6)


def test_kymotracker_lines_algorithm_integration_tests(kymo_integration_test_data):
    line_time = kymo_integration_test_data.line_time_seconds
    pixel_size = kymo_integration_test_data.pixelsize_um[0]

    lines = track_lines(kymo_integration_test_data, "red", 3 * pixel_size, 4)
    np.testing.assert_allclose(lines[0].coordinate_idx, [11] * len(lines[0].coordinate_idx))
    np.testing.assert_allclose(lines[1].coordinate_idx, [21] * len(lines[1].coordinate_idx))
    np.testing.assert_allclose(lines[0].time_idx, np.arange(9, 21))
    np.testing.assert_allclose(lines[1].time_idx, np.arange(14, 26))
    np.testing.assert_allclose(np.sum(lines[0].sample_from_image(1)), 50 * 10 + 6)
    np.testing.assert_allclose(np.sum(lines[1].sample_from_image(1)), 40 * 10 + 6)

    rect = [[0.0 * line_time, 15.0 * pixel_size], [30 * line_time, 30.0 * pixel_size]]
    lines = track_lines(kymo_integration_test_data, "red", 3 * pixel_size, 4, rect=rect)
    np.testing.assert_allclose(lines[0].coordinate_idx, [21] * len(lines[0].coordinate_idx))
    np.testing.assert_allclose(lines[0].time_idx, np.arange(14, 26))


def test_lines_algorithm_input_validation(kymo_integration_test_data):
    for line_width in (-1, 0):
        with pytest.raises(ValueError, match="should be larger than zero"):
            track_lines(kymo_integration_test_data, "red", line_width=line_width, max_lines=1000)
