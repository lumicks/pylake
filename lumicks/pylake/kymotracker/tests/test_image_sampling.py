import re

import numpy as np
import pytest

from lumicks.pylake.kymo import _kymo_from_array
from lumicks.pylake.channel import Slice, Continuous
from lumicks.pylake.detail.imaging_mixins import _FIRST_TIMESTAMP
from lumicks.pylake.kymotracker.kymotrack import KymoTrack
from lumicks.pylake.kymotracker.kymotracker import track_greedy
from lumicks.pylake.tests.data.mock_confocal import generate_kymo


def test_sampling():
    test_data = np.array(
        [
            [0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 1, 1, 0, 0],
            [0, 0, 1, 1, 0],
            [0, 0, 1, 1, 0],
            [0, 0, 0, 0, 0],
        ]
    )
    test_img = generate_kymo(
        "",
        test_data,
        pixel_size_nm=5000,
        start=np.int64(20e9),
        dt=np.int64(1e9),
        samples_per_pixel=1,
        line_padding=0,
    )

    # Tests the bound handling
    kymotrack = KymoTrack([0, 1, 2, 3, 4], [0, 1, 2, 3, 4], test_img, "red", 0)
    np.testing.assert_allclose(
        kymotrack.sample_from_image(50, correct_origin=True), [0, 2, 3, 2, 0]
    )
    np.testing.assert_allclose(kymotrack.sample_from_image(2, correct_origin=True), [0, 2, 3, 2, 0])
    np.testing.assert_allclose(kymotrack.sample_from_image(1, correct_origin=True), [0, 2, 2, 2, 0])
    np.testing.assert_allclose(kymotrack.sample_from_image(0, correct_origin=True), [0, 1, 1, 1, 0])
    np.testing.assert_allclose(
        KymoTrack([0, 1, 2, 3, 4], [4, 4, 4, 4, 4], test_img, "red", 0).sample_from_image(
            0, correct_origin=True
        ),
        [0, 0, 1, 1, 0],
    )

    kymotrack = KymoTrack([0, 1, 2, 3, 4], [0.1, 1.1, 2.1, 3.1, 4.1], test_img, "red", 0)
    np.testing.assert_allclose(
        kymotrack.sample_from_image(50, correct_origin=True), [0, 2, 3, 2, 0]
    )
    np.testing.assert_allclose(kymotrack.sample_from_image(2, correct_origin=True), [0, 2, 3, 2, 0])
    np.testing.assert_allclose(kymotrack.sample_from_image(1, correct_origin=True), [0, 2, 2, 2, 0])
    np.testing.assert_allclose(kymotrack.sample_from_image(0, correct_origin=True), [0, 1, 1, 1, 0])
    kymotrack = KymoTrack([0, 1, 2, 3, 4], [4.1, 4.1, 4.1, 4.1, 4.1], test_img, "red", 0)
    np.testing.assert_allclose(kymotrack.sample_from_image(0, correct_origin=True), [0, 0, 1, 1, 0])


def test_kymotrack_regression_sample_from_image_clamp():
    """This tests for a regression that occurred in sample_from_image. When sampling the image, we
    sample pixels in a region around the track. This sampling procedure is constrained to stay within
    the image. Previously, we used the incorrect axis to clamp the coordinate.
    """
    # Sampling the bottom row of a three pixel tall image will return [0, 0] instead of [1, 3];
    # since both coordinates would be clamped to the edge of the image (sampling nothing)."""

    img = generate_kymo(
        "",
        np.array([[1, 1, 1], [3, 3, 3]]).T,
        pixel_size_nm=1000,
        start=np.int64(20e9),
        dt=np.int64(1e9),
        samples_per_pixel=1,
        line_padding=0,
    )
    assert np.array_equal(
        KymoTrack([0, 1], [2, 2], img, "red", 0).sample_from_image(0, correct_origin=True), [1, 3]
    )


@pytest.mark.parametrize(
    "img",
    [
        np.asarray([[0, 0, 0], [100, 100, 100], [50, 50, 50]]),
        np.asarray([[0, 0, 0], [100, 100, 100], [1, 1, 1]]),
        np.asarray([[0, 0, 0], [100, 100, 100], [0, 0, 0]]),
        np.asarray([[1, 1, 1], [100, 100, 100], [0, 0, 0]]),  # Failed previously
        np.asarray([[50, 50, 50], [100, 100, 100], [0, 0, 0]]),  # Failed previously
    ],
)
def test_pixel_origin_sample_from_image(img):
    """Pixel coordinates are defined with the origin at the center of the pixel area. Previously,
    we had a bug where sample_from_image assumed the pixel center to be at the leftmost corner
    of the pixel. In that case, what happens in this test is that pulling the track slightly off
    the single pixel line in the negative direction results in the center of the sampling window
    shifting towards the previous pixel."""
    tracks = track_greedy(_kymo_from_array(img, "r", 0.2), "red", pixel_threshold=51)
    np.testing.assert_equal(tracks[0].sample_from_image(0, correct_origin=True), [100, 100, 100])


def test_origin_warning_sample_from_image():
    img = np.asarray([[0, 0, 0], [100, 100, 100], [50, 50, 50]])
    tracks = track_greedy(_kymo_from_array(img, "r", 0.2), "red", pixel_threshold=11)

    with pytest.warns(
        RuntimeWarning,
        match=re.escape(
            "Prior to version 1.1.0 the method `sample_from_image` had a bug that assumed "
            "the origin of a pixel to be at the edge rather than the center of the pixel. "
            "Consequently, the sampled window could frequently be off by one pixel. To get "
            "the correct behavior and silence this warning, specify `correct_origin=True`. "
            "The old (incorrect) behavior is maintained until the next major release to "
            "ensure backward compatibility. To silence this warning use `correct_origin=False`"
        ),
    ):
        tracks[0].sample_from_image(0)


@pytest.mark.parametrize(
    "time_idx, ref_dead_included, ref_dead_excluded, ref_reduce_max",
    [
        ([0, 1, 2], [14.5, 24.5, 34.5], [12.5, 22.5, 32.5], [15.0, 25.0, 35.0]),
        ([0, 2], [14.5, 34.5], [12.5, 32.5], [15.0, 35.0]),
        ([], [], [], []),
    ],
)
def test_sample_from_channel(time_idx, ref_dead_included, ref_dead_excluded, ref_reduce_max):
    img = np.zeros((5, 5))
    kymo = _kymo_from_array(
        img,
        "r",
        line_time_seconds=1.0,
        exposure_time_seconds=0.6,
        start=_FIRST_TIMESTAMP + int(1e9),
    )

    data = Slice(Continuous(np.arange(100), start=_FIRST_TIMESTAMP, dt=int(1e8)))  # 10 Hz
    kymotrack = KymoTrack(time_idx, time_idx, kymo, "red", 0)

    sampled = kymotrack.sample_from_channel(data)
    np.testing.assert_allclose(sampled.data, ref_dead_included)

    sampled = kymotrack.sample_from_channel(data, include_dead_time=False)
    np.testing.assert_allclose(sampled.data, ref_dead_excluded)

    sampled = kymotrack.sample_from_channel(data, include_dead_time=False, reduce=np.max)
    np.testing.assert_allclose(sampled.data, ref_reduce_max)


def test_sample_from_channel_out_of_bounds():
    kymo = _kymo_from_array(np.zeros((5, 5)), "r", line_time_seconds=1.0)
    data = Slice(Continuous(np.arange(100), start=0, dt=int(1e8)))
    kymotrack = KymoTrack([0, 6], [0, 6], kymo, "red", 0)

    with pytest.raises(IndexError):
        kymotrack.sample_from_channel(data, include_dead_time=False)


def test_sample_from_channel_no_overlap():
    img = np.zeros((5, 5))
    kymo = _kymo_from_array(img, "r", start=_FIRST_TIMESTAMP, line_time_seconds=int(1e8))
    data = Slice(Continuous(np.arange(100), start=kymo.stop + 100, dt=int(1e8)))
    kymotrack = KymoTrack([0, 1, 2], [0, 1, 2], kymo, "red", 0)

    with pytest.raises(RuntimeError, match="No overlap"):
        _ = kymotrack.sample_from_channel(data)
