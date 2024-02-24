import re

import numpy as np
import pytest

from lumicks.pylake.kymo import _kymo_from_array
from lumicks.pylake.kymotracker.kymotrack import KymoTrackGroup
from lumicks.pylake.kymotracker.kymotracker import track_greedy
from lumicks.pylake.tests.data.mock_confocal import generate_kymo


def test_kymotracker_test_bias_rect():
    """Computing the kymograph of a subset of the image should not affect the results of the
    tracking. If this test fires, it means that kymotracking on a subset of the image does not
    produce the same result as on the full thing for :func:`~lumicks.pylake.track_greedy()`.
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

    tracks = track_greedy(
        kymo_integration_test_data, "red", track_width=3 * pixel_size, pixel_threshold=4, rect=rect
    )
    np.testing.assert_allclose(
        tracks[0].sample_from_image(1, correct_origin=True), [40] * np.ones(10)
    )


def test_kymotracker_regression_test_subset_comes_up_empty(kymo_integration_test_data):
    """Test whether we gracefully handle the case where the ROI results in no lines."""
    tracks = track_greedy(
        kymo_integration_test_data,
        "red",
        track_width=3 * kymo_integration_test_data.pixelsize_um[0],  # Must be at least 3 pixels
        pixel_threshold=4,
        rect=[[0, 0], [1, 1]],
    )
    assert len(tracks) == 0


def test_kymotracker_greedy_algorithm_integration_tests(kymo_integration_test_data):
    test_data = kymo_integration_test_data
    line_time = test_data.line_time_seconds
    pixel_size = test_data.pixelsize_um[0]

    tracks = track_greedy(test_data, "red", track_width=3 * pixel_size, pixel_threshold=4)
    np.testing.assert_allclose(tracks[0].coordinate_idx, [11] * np.ones(10))
    np.testing.assert_allclose(tracks[1].coordinate_idx, [21] * np.ones(10))
    np.testing.assert_allclose(tracks[0].position, [11 * pixel_size] * np.ones(10))
    np.testing.assert_allclose(tracks[1].position, [21 * pixel_size] * np.ones(10))
    np.testing.assert_allclose(tracks[0].time_idx, np.arange(10, 20))
    np.testing.assert_allclose(tracks[1].time_idx, np.arange(15, 25))
    np.testing.assert_allclose(tracks[0].seconds, np.arange(10, 20) * line_time)
    np.testing.assert_allclose(tracks[1].seconds, np.arange(15, 25) * line_time)
    np.testing.assert_allclose(
        tracks[0].sample_from_image(1, correct_origin=True), [50] * np.ones(10)
    )
    np.testing.assert_allclose(
        tracks[1].sample_from_image(1, correct_origin=True), [40] * np.ones(10)
    )
    np.testing.assert_allclose(tracks[0].photon_counts, np.full((10,), 52))
    np.testing.assert_allclose(tracks[1].photon_counts, np.full((10,), 42))

    rect = [[0.0 * line_time, 15.0 * pixel_size], [30 * line_time, 30.0 * pixel_size]]
    tracks = track_greedy(
        test_data, "red", track_width=3 * pixel_size, pixel_threshold=4, rect=rect
    )
    np.testing.assert_allclose(tracks[0].coordinate_idx, [21] * np.ones(10))
    np.testing.assert_allclose(tracks[0].time_idx, np.arange(15, 25))


@pytest.mark.parametrize(
    "track_width, ref_counts", [(3, 7), (4, 9), (5, 9), (5.01, 11), (6.99, 11), (7, 11), (7.1, 13)]
)
def test_greedy_sampling_track_width(track_width, ref_counts):
    kymo = _kymo_from_array(np.vstack([np.array([1, 1, 1, 2, 3, 2, 1, 1, 1])] * 2).T, "r", 1)
    group = track_greedy(
        kymo, "red", pixel_threshold=2, track_width=track_width * kymo.pixelsize[0]
    )
    np.testing.assert_equal(group[0].photon_counts, np.full((2,), ref_counts))


def test_greedy_algorithm_empty_result(kymo_integration_test_data):
    test_data = kymo_integration_test_data
    # This tracking comes up empty because threshold is much higher than the data
    tracks = track_greedy(test_data, "red", track_width=300, pixel_threshold=1337)
    assert isinstance(tracks, KymoTrackGroup)
    assert len(tracks) == 0


def test_greedy_algorithm_input_validation(kymo_integration_test_data):
    test_data = kymo_integration_test_data

    for track_width in (-1, 0, 2.99 * test_data.pixelsize_um[0]):
        with pytest.raises(
            ValueError, match=re.escape("track_width must at least be 3 pixels (0.150 [um])")
        ):
            track_greedy(test_data, "red", track_width=track_width, pixel_threshold=10)

    # Width must be at least 3 pixels
    track_greedy(test_data, "red", track_width=3 * test_data.pixelsize_um[0], pixel_threshold=10)

    with pytest.raises(ValueError, match="should be positive"):
        track_greedy(test_data, "red", track_width=10, diffusion=-1, pixel_threshold=10)

    for pixel_threshold in (-1, 0):
        with pytest.raises(ValueError, match="should be larger than zero"):
            track_greedy(test_data, "red", track_width=10, pixel_threshold=pixel_threshold)


def test_default_parameters(kymo_pixel_calibrations):
    # calibrated in microns, kilobase pairs, pixels
    for kymo, default_width in zip(kymo_pixel_calibrations, [0.35, 0.35 / 0.34, 4]):
        # test that default values are used when `None` is supplied
        default_threshold = np.percentile(kymo.get_image("red"), 98)
        ref_tracks = track_greedy(
            kymo, "red", track_width=default_width, pixel_threshold=default_threshold
        )

        tracks = track_greedy(kymo, "red", track_width=None, pixel_threshold=default_threshold)
        for ref, track in zip(ref_tracks, tracks):
            np.testing.assert_allclose(ref.position, track.position)

        tracks = track_greedy(kymo, "red", track_width=default_width, pixel_threshold=None)
        for ref, track in zip(ref_tracks, tracks):
            np.testing.assert_allclose(ref.position, track.position)

        tracks = track_greedy(kymo, "red", track_width=None, pixel_threshold=None)
        for ref, track in zip(ref_tracks, tracks):
            np.testing.assert_allclose(ref.position, track.position)

        # We want to see that when setting the tracking parameter to something other than the
        # defaults actually has an effect
        ref_tracks = track_greedy(kymo, "red", track_width=None, pixel_threshold=None)
        tracks = track_greedy(
            kymo, "red", track_width=None, pixel_threshold=default_threshold * 0.7
        )
        with np.testing.assert_raises(AssertionError):
            for ref, track in zip(ref_tracks, tracks):
                np.testing.assert_allclose(ref.position, track.position)

        # To verify this for the width, we have to make sure we go to the next odd window size.
        tracks = track_greedy(
            kymo,
            "red",
            track_width=default_width / kymo.pixelsize[0] + 2,
            pixel_threshold=None,
            bias_correction=False,
        )
        with np.testing.assert_raises(AssertionError):
            for ref, track in zip(ref_tracks, tracks):
                np.testing.assert_allclose(ref.position, track.position)
