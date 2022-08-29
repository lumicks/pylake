import pytest
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
    tracking_settings = {"track_width": 3, "pixel_threshold": 4, "sigma": 5, "window": 9}
    traces_rect = track_greedy(kymo, "red", **tracking_settings, rect=[[0, 2], [1000, 12]])
    traces_full = track_greedy(kymo, "red", **tracking_settings)

    for t1, t2 in zip(traces_rect, traces_full):
        np.testing.assert_allclose(t1.position, t2.position)


def test_kymotracker_subset_test_greedy(kymo_integration_test_data):
    """If this test fires, it likely means that either the coordinates are not coordinates w.r.t.
    the original image, or that the reference to the image held by KymoTrack is a reference to a
    subset of the image, while the coordinates are still in the global coordinate system."""
    line_time = kymo_integration_test_data.line_time_seconds
    pixel_size = kymo_integration_test_data.pixelsize_um[0]
    rect = [[0.0 * line_time, 15.0 * pixel_size], [30 * line_time, 30.0 * pixel_size]]

    tracks = track_greedy(kymo_integration_test_data, "red", 3 * pixel_size, 4, rect=rect)
    np.testing.assert_allclose(tracks[0].sample_from_image(1), [40] * np.ones(10))


def test_kymotracker_greedy_algorithm_integration_tests(kymo_integration_test_data):
    test_data = kymo_integration_test_data
    line_time = test_data.line_time_seconds
    pixel_size = test_data.pixelsize_um[0]

    tracks = track_greedy(test_data, "red", 3 * pixel_size, 4)
    np.testing.assert_allclose(tracks[0].coordinate_idx, [11] * np.ones(10))
    np.testing.assert_allclose(tracks[1].coordinate_idx, [21] * np.ones(10))
    np.testing.assert_allclose(tracks[0].position, [11 * pixel_size] * np.ones(10))
    np.testing.assert_allclose(tracks[1].position, [21 * pixel_size] * np.ones(10))
    np.testing.assert_allclose(tracks[0].time_idx, np.arange(10, 20))
    np.testing.assert_allclose(tracks[1].time_idx, np.arange(15, 25))
    np.testing.assert_allclose(tracks[0].seconds, np.arange(10, 20) * line_time)
    np.testing.assert_allclose(tracks[1].seconds, np.arange(15, 25) * line_time)
    np.testing.assert_allclose(tracks[0].sample_from_image(1), [50] * np.ones(10))
    np.testing.assert_allclose(tracks[1].sample_from_image(1), [40] * np.ones(10))

    rect = [[0.0 * line_time, 15.0 * pixel_size], [30 * line_time, 30.0 * pixel_size]]
    tracks = track_greedy(test_data, "red", 3 * pixel_size, 4, rect=rect)
    np.testing.assert_allclose(tracks[0].coordinate_idx, [21] * np.ones(10))
    np.testing.assert_allclose(tracks[0].time_idx, np.arange(15, 25))


def test_greedy_algorithm_input_validation(kymo_integration_test_data):
    test_data = kymo_integration_test_data

    for track_width in (-1, 0):
        with pytest.raises(ValueError, match="should be larger than zero"):
            track_greedy(test_data, "red", track_width=track_width, pixel_threshold=10)

    # Any positive value will do
    track_greedy(test_data, "red", track_width=0.00001, pixel_threshold=10)

    with pytest.raises(ValueError, match="should be positive"):
        track_greedy(test_data, "red", track_width=10, diffusion=-1, pixel_threshold=10)

    for pixel_threshold in (-1, 0):
        with pytest.raises(ValueError, match="should be larger than zero"):
            track_greedy(test_data, "red", track_width=10, pixel_threshold=pixel_threshold)


def test_default_parameters(kymo_integration_test_data):
    ref_tracks = track_greedy(kymo_integration_test_data, "red", 0.35, 1)

    tracks = track_greedy(kymo_integration_test_data, "red", track_width=None, pixel_threshold=1)
    for ref, track in zip(ref_tracks, tracks):
        np.testing.assert_allclose(ref.position, track.position)

    tracks = track_greedy(kymo_integration_test_data, "red", track_width=0.35, pixel_threshold=None)
    for ref, track in zip(ref_tracks, tracks):
        np.testing.assert_allclose(ref.position, track.position)


def test_deprecated_args(kymo_integration_test_data):
    with pytest.warns(
        DeprecationWarning,
        match="The argument `line_width` is deprecated; use `track_width` instead.",
    ):
        track_greedy(kymo_integration_test_data, "red", line_width=5)
