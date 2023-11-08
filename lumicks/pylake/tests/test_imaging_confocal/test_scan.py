import numpy as np
import pytest


def test_scan_attrs(test_scans, test_scans_multiframe):
    for key, (scan, ref) in (test_scans | test_scans_multiframe).items():
        assert (
            repr(scan)
            == f"Scan(pixels=({ref.metadata.pixels_per_line}, {ref.metadata.lines_per_frame}))"
        )
        np.testing.assert_allclose(scan.timestamps, ref.timestamps.data)
        assert scan.num_frames == ref.metadata.number_of_frames
        np.testing.assert_equal(scan.pixel_time_seconds, ref.timestamps.pixel_time_seconds)
        assert scan.pixels_per_line == ref.metadata.pixels_per_line
        assert scan.lines_per_frame == ref.metadata.lines_per_frame
        assert len(scan.infowave) == len(ref.infowave.data)
        assert scan.get_image("rgb").shape == ref.image.shape
        assert scan.get_image("red").shape == ref.image.shape[:-1]
        assert scan.get_image("blue").shape == ref.image.shape[:-1]
        assert scan.get_image("green").shape == ref.image.shape[:-1]

        assert scan.fast_axis == ref.metadata.fast_axis
        np.testing.assert_allclose(scan.pixelsize_um, ref.metadata.pixelsize_um)
        for key, value in ref.metadata.center_point_um.items():
            np.testing.assert_allclose(scan.center_point_um[key], value)
        np.testing.assert_allclose(
            scan.size_um, np.array(ref.metadata.num_pixels) * ref.metadata.pixelsize_um
        )

        np.testing.assert_equal(
            scan.frame_timestamp_ranges(include_dead_time=True),
            ref.timestamps.timestamp_ranges  # For the single frame case, there is no dead time
            if scan.num_frames == 1
            else ref.timestamps.timestamp_ranges_deadtime,
        )
        np.testing.assert_equal(scan.frame_timestamp_ranges(), ref.timestamps.timestamp_ranges)


def test_missing_channels(test_scan_missing_channels):
    channel_map = {"r": 0, "g": 1, "b": 2}

    for missing_channels, (scan, ref) in test_scan_missing_channels.items():
        rgb = scan.get_image("rgb")
        assert rgb.shape == ref.image.shape
        np.testing.assert_equal(scan.get_image("rgb"), ref.image)

        for channel in missing_channels:
            assert not np.any(rgb[:, :, channel_map[channel[0]]])
            np.testing.assert_equal(scan.get_image(channel), np.zeros(ref.image.shape[:2]))


def test_damaged_scan(test_scan_truncated):
    # Assume the user incorrectly exported only a partial scan
    scan, ref = test_scan_truncated
    with pytest.raises(
        RuntimeError,
        match=(
            "Start of the scan was truncated. Reconstruction cannot proceed. "
            "Did you export the entire scan time in Bluelake?"
        ),
    ):
        scan.get_image("red").shape


def test_sted_bug(test_scan_sted_bug):
    # Test for workaround for a bug in the STED delay mechanism which could result in scan start
    # times ending up within the sample time.
    scan, ref, corrected_start = test_scan_sted_bug

    # should not raise, but change the start appropriately to work around sted bug
    # start is only adjusted only during image reconstruction
    original_start = scan.start
    scan.get_image("red").shape
    assert scan.start != original_start
    np.testing.assert_allclose(scan.start, corrected_start)
