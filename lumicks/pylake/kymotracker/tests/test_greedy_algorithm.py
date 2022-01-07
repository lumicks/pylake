import numpy as np
from lumicks.pylake.kymotracker.kymotracker import track_greedy
from lumicks.pylake.tests.data.mock_confocal import generate_kymo


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
    tracking_settings = {"line_width": 3, "pixel_threshold": 4, "sigma": 5, "window": 9}
    traces_rect = track_greedy(kymo, "red", **tracking_settings, rect=[[0, 2], [1000, 12]])
    traces_full = track_greedy(kymo, "red", **tracking_settings)

    for t1, t2 in zip(traces_rect, traces_full):
        np.testing.assert_allclose(t1.position, t2.position)


def test_kymotracker_subset_test_greedy(kymo_integration_test_data):
    """If this test fires, it likely means that either the coordinates are not coordinates w.r.t.
    the original image, or that the reference to the image held by KymoLine is a reference to a
    subset of the image, while the coordinates are still in the global coordinate system."""
    line_time = kymo_integration_test_data.line_time_seconds
    pixel_size = kymo_integration_test_data.pixelsize_um[0]
    rect = [[0.0 * line_time, 15.0 * pixel_size], [30 * line_time, 30.0 * pixel_size]]

    lines = track_greedy(kymo_integration_test_data, "red", 3 * pixel_size, 4, rect=rect)
    np.testing.assert_allclose(lines[0].sample_from_image(1), [40] * np.ones(10))


def test_kymotracker_greedy_algorithm_integration_tests(kymo_integration_test_data):
    test_data = kymo_integration_test_data
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

    rect = [[0.0 * line_time, 15.0 * pixel_size], [30 * line_time, 30.0 * pixel_size]]
    lines = track_greedy(test_data, "red", 3 * pixel_size, 4, rect=rect)
    np.testing.assert_allclose(lines[0].coordinate_idx, [21] * np.ones(10))
    np.testing.assert_allclose(lines[0].time_idx, np.arange(15, 25))
