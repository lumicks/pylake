import numpy as np
import pytest

from .mock_confocal import generate_timestamps


@pytest.mark.parametrize(
    (
        "number_of_frames, lines_per_frame, pixels_per_line, start, dt, samples_per_pixel, "
        "line_padding, scan, x_axis_fast"
    ),
    [
        (1, 2, 2, 0, 1, 1, 0, False, False),  # kymo, minimal, y_axis_fast
        (1, 2, 2, 0, 1, 1, 0, False, True),  # kymo, minimal, x_axis_fast
        (1, 2, 2, 0, 1, 1, 0, True, False),  # scan, minimal, y_axis_fast
        (1, 2, 2, 0, 1, 1, 0, True, True),  # scan, minimal, x_axis_fast
        (1, 2, 2, 2**63 - 5, 1, 1, 0, False, False),  # kymo, max start, y_axis_fast
        (1, 2, 2, 2**63 - 5, 1, 1, 0, False, True),  # kymo, max start, x_axis_fast
        (1, 2, 2, 2**63 - 5, 1, 1, 0, True, False),  # scan, max start, y_axis_fast
        (1, 2, 2, 2**63 - 5, 1, 1, 0, True, True),  # scan, max start, x_axis_fast
        (1, 2, 2, 0, 2**63 // 5, 1, 0, False, False),  # kymo, max dt, y_axis_fast
        (1, 2, 2, 0, 2**63 // 5, 1, 0, False, True),  # kymo, max dt, x_axis_fast
        (1, 2, 2, 0, 2**63 // 5, 1, 0, True, False),  # scan, max dt, y_axis_fast
        (1, 2, 2, 0, 2**63 // 5, 1, 0, True, True),  # scan, max dt, x_axis_fast
        (1, 2, 3, int(20e9), int(16e9), 7, 15, False, False),  # kymo, y_axis_fast
        (1, 2, 3, int(20e9), int(16e9), 7, 15, False, True),  # kymo, x_axis_fast
        (1, 2, 3, int(20e9), int(16e9), 7, 15, True, False),  # scan, 1 frame, y_axis_fast
        (1, 2, 3, int(20e9), int(16e9), 7, 15, True, True),  # scan, 1 frame, x_axis_fast
        (1, 2, 3, int(20e9), int(16e9), 6, 14, False, False),  # kymo, even, y_axis_fast
        (1, 2, 3, int(20e9), int(16e9), 6, 14, False, True),  # kymo, even, x_axis_fast
        (1, 2, 3, int(20e9), int(16e9), 6, 15, True, False),  # scan, even, 1 frame, y_axis_fast
        (1, 2, 3, int(20e9), int(16e9), 6, 15, True, True),  # scan, even, 1 frame, x_axis_fast
        (2, 2, 3, int(20e9), int(16e9), 7, 15, True, True),  # scan, 2 frames, x_axis_fast
        (2, 2, 3, int(20e9), int(16e9), 7, 15, False, True),  # scan (auto), 2 frames, x_axis_fast
        (3, 2, 3, int(20e9), int(16e9), 7, 15, True, True),  # scan, 3 frames, x_axis_fast
        (3, 3, 2, int(20e9), int(16e9), 7, 15, False, True),  # scan (auto), 3 frames, x_axis_fast
    ],
)
def test_generate_timestamps(
    number_of_frames,
    lines_per_frame,
    pixels_per_line,
    start,
    dt,
    samples_per_pixel,
    line_padding,
    scan,
    x_axis_fast,
):
    timestamps, ranges, ranges_deadtime = generate_timestamps(
        number_of_frames,
        lines_per_frame,
        pixels_per_line,
        start,
        dt,
        samples_per_pixel,
        line_padding,
        scan=scan,
        x_axis_fast=x_axis_fast,
    )
    # `generate_timestamps` automatically creates scan timestamp ranges if `number_of_frames` > 1
    scan = scan or number_of_frames > 1

    # Test dtype and shape
    assert timestamps.dtype == np.int64
    shape_ref = (
        (lines_per_frame, pixels_per_line) if x_axis_fast else (pixels_per_line, lines_per_frame)
    )
    shape_ref = (number_of_frames, *shape_ref) if number_of_frames > 1 else shape_ref
    assert timestamps.shape == shape_ref
    assert ranges.dtype == np.int64
    assert ranges.shape == (number_of_frames if scan else lines_per_frame, 2)
    assert ranges_deadtime.dtype == np.int64
    assert ranges_deadtime.shape == (number_of_frames if scan else lines_per_frame, 2)

    # Test crucial timestamp values (i.e. the first timestamp and increments in all dimensions)
    pixel_time = dt * samples_per_pixel
    padding_time = dt * line_padding
    line_time = pixels_per_line * pixel_time + 2 * padding_time
    frame_time = lines_per_frame * line_time

    # Test the first timestamp
    frame = timestamps if number_of_frames == 1 else timestamps[0]
    # First pixel starts at an offset of padding time + half of the pixel mean time
    assert frame[0, 0] == start + padding_time + dt * (samples_per_pixel - 1) // 2

    # Time increments should consist of frame time, line time and pixel time, only
    if number_of_frames > 1:
        np.testing.assert_equal(np.diff(timestamps, axis=0), frame_time)
    y_axis = (number_of_frames > 1) + (not x_axis_fast)
    np.testing.assert_equal(np.diff(timestamps, axis=y_axis), line_time)
    x_axis = (number_of_frames > 1) + x_axis_fast
    np.testing.assert_equal(np.diff(timestamps, axis=x_axis), pixel_time)

    # Test crucial ranges timestamp values
    assert ranges[0, 0] == start + padding_time
    assert ranges_deadtime[0, 0] == start + padding_time
    np.testing.assert_equal(np.diff(ranges, axis=0), frame_time if scan else line_time)
    np.testing.assert_equal(
        np.diff(ranges, axis=1), (frame_time if scan else line_time) - 2 * padding_time
    )
    np.testing.assert_equal(np.diff(ranges_deadtime, axis=0), frame_time if scan else line_time)
    np.testing.assert_equal(np.diff(ranges_deadtime, axis=1), frame_time if scan else line_time)


@pytest.mark.parametrize(
    ("scan, x_axis_fast"),
    [
        (False, False),
        (False, True),
        (True, False),
        (True, True),
    ],
)
def test_generate_timestamps_errors(scan, x_axis_fast):
    with pytest.raises(ValueError, match=r"`start` needs to be non negative"):
        generate_timestamps(1, 2, 2, -1, 1, 1, 0, scan, x_axis_fast)
    with pytest.raises(ValueError, match=r"`dt` needs to be positive"):
        generate_timestamps(1, 2, 2, 0, 0, 1, 0, scan, x_axis_fast)

    # Error due to start and/or dt too big
    with pytest.raises(OverflowError, match="timestamps are too big for int64"):
        generate_timestamps(1, 1, 1, 2**63 - 1, 1, 1, 0, scan, x_axis_fast)
    with pytest.raises(OverflowError, match="timestamps are too big for int64"):
        generate_timestamps(1, 2, 2, 2**63 - 4, 2, 1, 0, scan, x_axis_fast)
    with pytest.raises(OverflowError, match="timestamps are too big for int64"):
        generate_timestamps(1, 1, 1, 0, 2**63 // 2 + 1, 1, 0, scan, x_axis_fast)
    with pytest.raises(OverflowError, match="timestamps are too big for int64"):
        generate_timestamps(1, 2, 2, 0, 2**63 // 5 + 1, 1, 0, scan, x_axis_fast)
    with pytest.raises(OverflowError, match="timestamps are too big for int64"):
        # overflow error has priority over float not int errors
        generate_timestamps(1.0, 2.0, 2.0, 2**63 - 3, 1, 1.0, 0.0, scan, x_axis_fast)

    # Error due to shaping parameters being of type float
    with pytest.raises(TypeError, match="`start` needs to be an integer"):
        generate_timestamps(1, 2, 2, 0.0, 1, 1, 0, scan, x_axis_fast)
    with pytest.raises(TypeError, match="`dt` needs to be an integer"):
        generate_timestamps(1, 2, 2, 0, 1.0, 1, 0, scan, x_axis_fast)
    with pytest.raises(TypeError, match="'float' object cannot be interpreted as an integer"):
        generate_timestamps(1.0, 2, 2, 0, 1, 1, 0, scan, x_axis_fast)
    with pytest.raises(TypeError, match="'float' object cannot be interpreted as an integer"):
        generate_timestamps(1, 2.0, 2, 0, 1, 1, 0, scan, x_axis_fast)
    with pytest.raises(TypeError, match="'float' object cannot be interpreted as an integer"):
        generate_timestamps(1, 2, 2.0, 0, 1, 1, 0, scan, x_axis_fast)
    with pytest.raises(TypeError, match="'float' object cannot be interpreted as an integer"):
        generate_timestamps(1, 2, 2, 0, 1, 1.0, 0, scan, x_axis_fast)
    with pytest.raises(TypeError, match="'float' object cannot be interpreted as an integer"):
        generate_timestamps(1, 2, 2, 0, 1, 1, 0.0, scan, x_axis_fast)
