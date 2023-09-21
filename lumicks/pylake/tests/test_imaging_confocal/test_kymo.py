import numpy as np
import pytest

from lumicks.pylake.kymo import _default_line_time_factory
from lumicks.pylake.channel import Slice, Continuous

from ..data.mock_confocal import generate_kymo


def test_kymo_properties(test_kymo):
    kymo, ref = test_kymo

    assert repr(kymo) == f"Kymo(pixels={ref.metadata.pixels_per_line})"
    assert kymo.pixels_per_line == ref.metadata.pixels_per_line
    assert len(kymo.infowave) == len(ref.infowave.data)

    np.testing.assert_equal(kymo.get_image("rgb"), ref.image)
    assert kymo.shape == ref.image.shape
    assert kymo.get_image("rgb").shape == ref.image.shape
    assert kymo.get_image("red").shape == ref.image[:, :, 0].shape
    assert kymo.get_image("blue").shape == ref.image[:, :, 1].shape
    assert kymo.get_image("green").shape == ref.image[:, :, 2].shape

    np.testing.assert_equal(kymo.timestamps, ref.timestamps.data)
    assert kymo.start == ref.start
    assert kymo.stop == ref.stop

    assert kymo.fast_axis == ref.metadata.fast_axis
    np.testing.assert_equal(kymo.pixelsize_um, ref.metadata.pixelsize_um)
    np.testing.assert_allclose(kymo.line_time_seconds, ref.timestamps.line_time_seconds)
    np.testing.assert_allclose(
        kymo.duration, ref.timestamps.line_time_seconds * ref.metadata.lines_per_frame
    )
    np.testing.assert_equal(kymo.center_point_um, ref.metadata.center_point_um)
    np.testing.assert_allclose(
        kymo.size_um, [ref.metadata.pixels_per_line * ref.metadata.pixelsize_um[0]]
    )
    np.testing.assert_allclose(kymo.pixel_time_seconds, ref.timestamps.pixel_time_seconds)
    np.testing.assert_allclose(kymo.motion_blur_constant, 0)  # We neglect motion blur for confocal


def test_empty_kymo_properties(test_kymo):
    kymo, ref = test_kymo
    empty_kymo = kymo["3s":"2s"]

    assert empty_kymo.fast_axis == ref.metadata.fast_axis
    np.testing.assert_equal(empty_kymo.pixelsize_um, ref.metadata.pixelsize_um)
    np.testing.assert_equal(empty_kymo.duration, 0)
    np.testing.assert_equal(empty_kymo.center_point_um, ref.metadata.center_point_um)
    np.testing.assert_allclose(
        empty_kymo.size_um, [ref.metadata.pixels_per_line * ref.metadata.pixelsize_um[0]]
    )

    with pytest.raises(RuntimeError, match="Can't get pixel timestamps if there are no pixels"):
        empty_kymo.line_time_seconds

    with pytest.raises(RuntimeError, match="Can't get pixel timestamps if there are no pixels"):
        empty_kymo.pixel_time_seconds


def test_damaged_kymo(truncated_kymo):
    # Assume the user incorrectly exported only a partial Kymo
    kymo, ref = truncated_kymo

    with pytest.warns(RuntimeWarning):
        assert kymo.get_image("red").shape == (5, 3)
    np.testing.assert_allclose(kymo.get_image("red"), ref.image[:, 1:, 0])


def test_line_timestamp_ranges(test_kymo):
    kymo, ref = test_kymo

    expected_ranges = (ref.timestamps.timestamp_ranges, ref.timestamps.timestamp_ranges_deadtime)

    pixel_infowave = np.hstack((np.ones(ref.infowave.samples_per_pixel - 1, dtype=int), 2))
    expected_iw_chunks = [
        list(pixel_infowave) * ref.metadata.pixels_per_line,
        list(pixel_infowave) * ref.metadata.pixels_per_line + ([0] * ref.infowave.line_padding * 2),
    ]

    for include, ref_ranges, ref_iw_chunks in zip(
        (False, True), expected_ranges, expected_iw_chunks
    ):
        ranges = kymo.line_timestamp_ranges(include_dead_time=include)
        np.testing.assert_equal(ranges, ref_ranges)

        for rng in ranges:
            iw_chunk = kymo.infowave[slice(*rng)].data
            np.testing.assert_equal(iw_chunk, ref_iw_chunks[: iw_chunk.size])


def test_regression_unequal_timestamp_spacing():
    """This particular set of initial timestamp and sampler per pixel led to unequal timestamp
    spacing in an actual dataset."""
    kymo = generate_kymo(
        "Mock",
        np.array([[1, 1, 1, 1, 1], [1, 1, 1, 1, 1]]),
        pixel_size_nm=100,
        start=1536582124217030400,
        dt=int(1e9 / 78125),
        samples_per_pixel=47,
        line_padding=0,
    )
    assert len(np.unique(np.diff(kymo.timestamps))) == 1


def test_partial_pixel_kymo():
    """This function tests whether a partial pixel at the end is fully dropped. This is important,
    since in the timestamp reconstruction, we subtract the minimum value from a row prior to
    averaging (to allow taking averages of larger chunks). Without this functionality, the lowest
    pixel to be reconstructed can be smaller than the first timestamp, which means the subtraction
    of the minimum is rendered less effective (leading to unnecessarily long reconstruction
    times)."""
    kymo = generate_kymo(
        "Mock",
        np.ones((5, 5)),
        pixel_size_nm=100,
        start=1536582124217030400,
        dt=int(1e9 / 78125),
        samples_per_pixel=47,
        line_padding=0,
    )

    kymo.infowave.data[-60:] = 0  # Remove the last pixel entirely, and a partial pixel before that
    np.testing.assert_equal(kymo.timestamps[-1, -1], 0)
    np.testing.assert_equal(kymo.timestamps[-2, -1], 0)
    np.testing.assert_equal(kymo.get_image("red")[-1, -1], 0)
    np.testing.assert_equal(kymo.get_image("red")[-2, -1], 0)


@pytest.mark.parametrize(
    "data, ref_line_time, pixels_per_line, bad",
    [
        ([1, 1, 2, 1, 1, 2, 0, 0, 1, 1, 2, 1, 1, 2], 8 * 2, 2, False),
        ([1, 1, 2, 1, 1, 2, 0, 1], 7 * 2, 2, False),
        ([0, 0, 1, 1, 2, 1, 1, 2, 0, 1], 7 * 2, 2, False),
        ([1, 1, 2, 1, 1, 2, 1, 1, 2, 0, 1], 10 * 2, 3, False),
        ([1, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 2], 9 * 2, 3, False),  # No dead time
        ([2, 2, 2, 0, 1], 4 * 2, 3, False),  # Three pixels and a one sample dead time
        ([2, 2], 2 * 2, 3, True),  # Expected 3 pixels per line, but got partial line
        ([2, 2, 2], 3 * 2, 3, True),  # A single line without dead time (can't be sure)
        ([2, 2, 2, 0], 4 * 2, 3, True),  # 3 pixels, 1 sample dead time, but deadtime undefined
        ([2, 2, 2, 2], 3 * 2, 3, False),  # Three pixels, a line and a little, without dead-time
        ([0, 2, 2], 2 * 2, 3, True),  # Expected 3 pixels per line, but partial line
        ([0, 2, 2, 2], 3 * 2, 3, True),  # A single line, but can't be sure
        ([0, 2, 2, 2, 0], 4 * 2, 3, True),  # 3 pixels, 1 sample dead time, but deadtime undefined
        ([0, 2, 2, 2, 2], 3 * 2, 3, False),  # Three pixels, a line and a little, without dead-time
        ([0, 0, 2, 2, 2, 0, 1], 4 * 2, 3, False),  # Three pixels and a pixel dead time
        ([1, 1], 2 * 2, 2, True),  # No full pixel available at all
        ([1, 1, 2, 1, 1], 5 * 2, 2, True),  # No full line available
        ([0, 0, 1, 1, 2, 1, 1], 5 * 2, 2, True),  # No full line available but padded
        ([0, 0, 1, 1, 2, 1, 1, 2], 6 * 2, 2, True),  # Exactly a full line, w/o dead time
        ([0, 0, 1, 1, 2, 1, 1, 2, 0], 7 * 2, 2, True),  # Full line, dead time not yet defined
        ([0, 0, 1, 1, 2, 1, 1, 2, 0, 0], 8 * 2, 2, True),  # Full line, dead time not yet defined
        ([0, 0, 1, 1, 2, 1, 1, 2, 0, 0, 1], 8 * 2, 2, False),  # Well defined
        ([0, 0, 1, 1, 2, 1, 1, 2, 0, 0, 1, 1], 8 * 2, 2, False),  # Well defined
    ],
)
def test_direct_infowave_linetime(data, ref_line_time, pixels_per_line, bad):
    class KymoWave:
        def __init__(self):
            self.infowave = Slice(Continuous(data, int(100e9), int(2e9)))
            self._has_default_factories = lambda: True
            self.pixels_per_line = pixels_per_line

    if not bad:
        assert _default_line_time_factory(KymoWave()) == ref_line_time
    else:
        with pytest.raises(RuntimeError, match=r"This kymograph consists of only a single line"):
            _default_line_time_factory(KymoWave())
