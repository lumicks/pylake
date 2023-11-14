import numpy as np
import pytest

from lumicks.pylake.scan import EmptyScan


def test_slicing(test_scans_multiframe):
    scan, _ = test_scans_multiframe["fast X slow Y multiframe"]
    assert scan.num_frames == 10

    def compare_frames(original_frames, new_scan):
        assert new_scan.num_frames == len(original_frames)
        for new_frame_index, index in enumerate(original_frames):
            frame = scan.get_image("red")[index]
            new_frame = (
                new_scan.get_image("red")[new_frame_index]
                if new_scan.num_frames > 1
                else new_scan.get_image("red")
            )
            np.testing.assert_equal(frame, new_frame)

    compare_frames([0], scan[0])  # first frame
    compare_frames([9], scan[-1])  # last frame
    compare_frames([3], scan[3])  # single frame
    compare_frames([2, 3, 4], scan[2:5])  # slice
    compare_frames([0, 1, 2], scan[:3])  # from beginning
    compare_frames([8, 9], scan[8:])  # until end
    compare_frames([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], scan[:])  # all
    compare_frames([7], scan[-3])  # negative index
    compare_frames([0, 1, 2, 3], scan[:-6])  # until negative index
    compare_frames([5, 6, 7], scan[5:-2])  # mixed sign indices
    compare_frames([6, 7], scan[-4:-2])  # full negative slice
    compare_frames([2], scan[2:3])  # slice to single frame
    compare_frames([3, 4], scan[2:6][1:3])  # iterative slicing

    compare_frames([1, 2, 3, 4, 5, 6, 7, 8, 9], scan[1:100])  # test clamping past the end
    compare_frames([0, 1, 2, 3, 4, 5, 6, 7, 8], scan[-100:9])  # test clamping past the beginning
    compare_frames([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], scan[-100:100])  # test clamping both dirs

    # reverse slice (leads to empty scan)
    assert isinstance(scan[5:3], EmptyScan)
    assert isinstance(scan[-2:-4], EmptyScan)

    # empty slice
    assert isinstance(scan[5:5], EmptyScan)
    assert isinstance(scan[15:16], EmptyScan)

    # Verify no side effects
    scan[0]
    assert scan.num_frames == 10


@pytest.mark.parametrize(
    "x_min, x_max, y_min, y_max",
    [
        (None, None, None, None),
        (None, 3, None, 2),
        (0, None, 1, None),
        (1, 4, 1, 5),
        (1, 3, None, None),
        (None, None, 1, 3),
        (1, 2, 1, 2),  # Single pixel
        (1, 2, None, None),
        (None, None, 1, 2),
    ],
)
def test_scan_cropping(
    x_min, x_max, y_min, y_max, test_scans, test_scans_multiframe, test_scan_missing_channels
):
    valid_scans = [
        "fast Y slow X multiframe",
        "fast Y slow X",
        "fast X slow Z multiframe",
        "fast Y slow Z multiframe",
        ("red",),
        ("red", "blue"),
    ]
    test_set = test_scans | test_scans_multiframe | test_scan_missing_channels

    for key in valid_scans:
        scan, _ = test_set[key]

        # Slice how we would with numpy
        all_slices = tuple([slice(None), slice(y_min, y_max), slice(x_min, x_max)])
        numpy_slice = all_slices[1:] if scan.num_frames == 1 else all_slices

        cropped_scan = scan.crop_by_pixels(x_min, x_max, y_min, y_max)
        np.testing.assert_allclose(cropped_scan.timestamps, scan.timestamps[numpy_slice])
        np.testing.assert_allclose(cropped_scan.shape, scan.get_image("rgb")[numpy_slice].shape)
        np.testing.assert_allclose(scan[all_slices].timestamps, scan.timestamps[numpy_slice])

        for channel in ("rgb", "green"):
            ref_img = scan.get_image(channel)[numpy_slice]
            np.testing.assert_allclose(cropped_scan.get_image(channel), ref_img)
        # Numpy array is given as Y, X, while number of pixels is given sorted by spatial axis
        # i.e. X, Y
        np.testing.assert_allclose(
            np.flip(cropped_scan._num_pixels), cropped_scan.get_image("green").shape[-2:]
        )


@pytest.mark.parametrize(
    "all_slices",
    [
        [slice(None), slice(None), slice(None)],
        [slice(None), slice(None, 3), slice(None, 2)],
        [slice(None), slice(0, None), slice(1, None)],
        [slice(None), slice(1, 4), slice(1, 5)],
        [slice(None), slice(1, 3), slice(None)],
        [slice(None), slice(None), slice(1, 3)],
        [slice(None), slice(1, 2), slice(1, 2)],  # Single pixel
        [slice(None), slice(1, 2), slice(None)],
        [slice(None), slice(None), slice(1, 2)],
        [0],  # Only indexing scans
        [0, slice(1, 2), slice(1, 2)],
        [-1, slice(1, 3), slice(None)],
        [0, slice(None), slice(1, 2)],
        [0, slice(1, 3)],  # This tests a very specific corner case where _num_pixels could fail
    ],
)
def test_scan_get_item_slicing(
    all_slices, test_scans, test_scans_multiframe, test_scan_missing_channels
):
    """Test slicing, slicing is given as image, y, x"""

    valid_scans = [
        "fast Y slow X multiframe",
        "fast Y slow X",
        "fast X slow Z multiframe",
        "fast Y slow Z multiframe",
        ("red",),
        ("red", "blue"),
    ]
    test_set = test_scans | test_scans_multiframe | test_scan_missing_channels

    for key in valid_scans:
        scan, _ = test_set[key]

        # Slice how we would with numpy
        slices = tuple(all_slices if scan.num_frames > 1 else all_slices[1:])
        cropped_scan = scan[all_slices]
        np.testing.assert_allclose(cropped_scan.timestamps, scan.timestamps[slices])
        np.testing.assert_allclose(cropped_scan.shape, scan.get_image("rgb")[slices].shape)

        for channel in ("rgb", "green"):
            ref_img = scan.get_image(channel)[slices]
            np.testing.assert_allclose(cropped_scan.get_image(channel), ref_img)

        # Numpy array is given as Y, X, while number of pixels is given sorted by spatial axis
        # i.e. X, Y
        np.testing.assert_allclose(
            np.flip(cropped_scan._num_pixels), cropped_scan.get_image("green").shape[-2:]
        )


def test_slicing_cropping_separate_actions(test_scans, test_scans_multiframe):
    """Test whether cropping works in a sequential fashion"""

    single_frame, _ = test_scans["fast X slow Y"]
    multi_frame, _ = test_scans_multiframe["fast X slow Y multiframe"]

    def assert_equal(first, second):
        np.testing.assert_allclose(first.get_image("red"), second.get_image("red"))
        np.testing.assert_allclose(first.get_image("rgb"), second.get_image("rgb"))
        np.testing.assert_allclose(first.get_image("rgb").shape, second.get_image("rgb").shape)
        assert first.num_frames == second.num_frames
        assert first._num_pixels == second._num_pixels

    assert_equal(multi_frame[1:3][:, 1:3, 2:3], multi_frame[1:3, 1:3, 2:3])
    assert_equal(multi_frame[1][:, 1:3, 2:3], multi_frame[1, 1:3, 2:3])
    assert_equal(multi_frame[:, 1:3, 2:3][3], multi_frame[3, 1:3, 2:3])
    assert_equal(multi_frame[:, 1:, 2:][3], multi_frame[3, 1:, 2:])
    assert_equal(multi_frame[:, :3, :3][3], multi_frame[3, :3, :3])
    assert_equal(single_frame[0][:, 1:5, 2:6][:, 1:3, 2:8], single_frame[0, 2:4, 4:6])


def test_error_handling_multidim_indexing(test_scans, test_scans_multiframe):
    single_frame, _ = test_scans["fast X slow Y"]
    multi_frame, _ = test_scans_multiframe["fast X slow Y multiframe"]

    with pytest.raises(IndexError, match="Frame index out of range"):
        multi_frame[18]
    with pytest.raises(IndexError, match="Frame index out of range"):
        multi_frame[-19]
    with pytest.raises(IndexError, match="Frame index out of range"):
        single_frame[1]
    with pytest.raises(
        IndexError, match="Scalar indexing is not supported for spatial coordinates"
    ):
        multi_frame[0, 4, 3]


@pytest.mark.parametrize(
    "name",
    [
        "fast Y slow X multiframe",
        "fast Y slow X",
        "fast X slow Z multiframe",
        "fast Y slow Z multiframe",
        ("red",),
        ("red", "blue"),
    ],
)
def test_empty_slices_due_to_out_of_bounds(
    name, test_scans, test_scans_multiframe, test_scan_missing_channels
):
    test_set = test_scans | test_scans_multiframe | test_scan_missing_channels
    scan, _ = test_set[name]

    shape = scan.get_image("green").shape[-2:]

    with pytest.raises(NotImplementedError, match="Slice is empty."):
        scan[0, shape[0] : shape[0] + 10, 0:2]
    with pytest.raises(NotImplementedError, match="Slice is empty."):
        scan[0, shape[0] : shape[0] + 10]
    with pytest.raises(NotImplementedError, match="Slice is empty."):
        scan[0, :, shape[1] : shape[1] + 10]
        print("***", shape[1], shape[1] + 10)
        print(scan[0, :, shape[1] : shape[1] + 10])
    with pytest.raises(NotImplementedError, match="Slice is empty."):
        scan[0, :, -10:0]


def test_slice_by_list_disallowed(test_scans_multiframe):
    scan, _ = test_scans_multiframe["fast Y slow X multiframe"]

    with pytest.raises(IndexError, match="Indexing by list is not allowed"):
        scan[[0, 1], :, :]

    with pytest.raises(IndexError, match="Indexing by list is not allowed"):
        scan[:, [0, 1], :]

    class Dummy:
        pass

    with pytest.raises(IndexError, match="Indexing by Dummy is not allowed"):
        scan[Dummy(), :, :]

    with pytest.raises(IndexError, match="Slicing by Dummy is not supported"):
        scan[Dummy() : Dummy(), :, :]


def test_crop_missing_channel(test_scan_missing_channels):
    """Make sure that missing channels are handled appropriately when cropping"""

    scan, _ = test_scan_missing_channels[("red",)]
    np.testing.assert_equal(scan[:, 0:2, 1:3].get_image("red"), np.zeros((2, 2)))


def test_scan_slicing_by_time(test_scan_slicing):
    multi_frame, ref = test_scan_slicing
    multi_frame = multi_frame[:]

    rng = (np.vstack(ts := ref.timestamps.timestamp_ranges) - ts[0][0]) * 1e-9
    wiggle = 0.01

    # start of frame 2, until frame 2 signal stop
    assert isinstance(multi_frame[f"{rng[2, 0]}s":f"{rng[2, 1]}s"], EmptyScan)

    # start of frame 2, just over frame 2 signal stop
    sliced = multi_frame[f"{rng[2, 0]}s":f"{rng[2, 1] + wiggle}s"]
    np.testing.assert_equal(sliced.get_image("rgb"), ref.image[2])

    # start of frame 2, until frame 3 signal stop
    sliced = multi_frame[f"{rng[2, 0]}s":f"{rng[3, 1]}s"]
    np.testing.assert_equal(sliced.get_image("rgb"), ref.image[2])

    # start of frame 2, just over frame 3 signal stop
    sliced = multi_frame[f"{rng[2, 0]}s":f"{rng[3, 1] + wiggle}s"]
    np.testing.assert_equal(sliced.get_image("rgb"), ref.image[2:4])

    # start of frame 2, until start of frame 4
    sliced = multi_frame[f"{rng[2, 0]}s":f"{rng[4, 0]}s"]
    np.testing.assert_equal(sliced.get_image("rgb"), ref.image[2:4])

    # from start of frame 2
    sliced = multi_frame[f"{rng[2, 0]}s":]
    np.testing.assert_equal(sliced.get_image("rgb"), ref.image[2:])

    # from just after start of frame 2
    sliced = multi_frame[f"{rng[2, 0] + wiggle}s":]
    np.testing.assert_equal(sliced.get_image("rgb"), ref.image[3:])

    # until just after end of frame 1 signal
    sliced = multi_frame[:f"{rng[2, 1] + wiggle}s"]
    np.testing.assert_equal(sliced.get_image("rgb"), ref.image[:3])

    # until end of frame 1 signal
    sliced = multi_frame[:f"{rng[2, 1]}s"]
    np.testing.assert_equal(sliced.get_image("rgb"), ref.image[:2])


def test_scan_slicing_by_negative_time(test_scan_slicing):
    multi_frame, ref = test_scan_slicing
    multi_frame = multi_frame[:]

    rng = (np.vstack(ts := ref.timestamps.timestamp_ranges) - ts[0][0]) * 1e-9
    rng_dead = (np.vstack(ts := ref.timestamps.timestamp_ranges_deadtime) - ts[0][0]) * 1e-9

    scan_time = rng[0, 1] - rng[0, 0]
    dead_time = rng_dead[0, 1] - rng[0, 1]
    frame_time = scan_time + dead_time
    wiggle = 0.01

    # until just after end of frame 8
    sliced = multi_frame[:f"{-(frame_time + dead_time) + wiggle}s"]
    np.testing.assert_equal(sliced.get_image("rgb"), ref.image[:9])

    # until end of frame 8 signal stop
    sliced = multi_frame[:f"{-(frame_time + dead_time)}s"]
    np.testing.assert_equal(sliced.get_image("rgb"), ref.image[:8])

    # from start of frame 7
    sliced = multi_frame[f"{-3 * frame_time}s":]
    np.testing.assert_equal(sliced.get_image("rgb"), ref.image[7:])

    # from middle of frame 7
    sliced = multi_frame[f"{-(2 * frame_time + scan_time) + wiggle}s":]
    np.testing.assert_equal(sliced.get_image("rgb"), ref.image[8:])


def test_scan_slicing_by_time_iterative(test_scan_slicing):
    multi_frame, ref = test_scan_slicing
    multi_frame = multi_frame[:]

    rng = (np.vstack(ts := ref.timestamps.timestamp_ranges) - ts[0][0]) * 1e-9
    rng_dead = (np.vstack(ts := ref.timestamps.timestamp_ranges_deadtime) - ts[0][0]) * 1e-9

    scan_time = rng[0, 1] - rng[0, 0]
    dead_time = rng_dead[0, 1] - rng[0, 1]
    frame_time = scan_time + dead_time
    wiggle = 0.01

    # first slice frame 2 through 5 (inclusive)
    first_slice = multi_frame[f"{rng[2, 0]}s":f"{rng[5, 1] + wiggle}s"]
    ref_sliced = ref.image[2:6]

    # start of frame 1, just over frame 3 signal stop
    sliced = first_slice[f"{rng[1, 0]}s":f"{rng[3, 1] + wiggle}s"]
    np.testing.assert_equal(sliced.get_image("rgb"), ref_sliced[1:4])

    # start of frame 1, until frame 3 signal stop
    sliced = first_slice[f"{rng[1, 0]}s":f"{rng[3, 1]}s"]
    np.testing.assert_equal(sliced.get_image("rgb"), ref_sliced[1:3])

    # from start of frame 1
    sliced = first_slice[f"{rng[1, 0]}s":]
    np.testing.assert_equal(sliced.get_image("rgb"), ref_sliced[1:])

    # until just after frame 2 signal stop
    sliced = first_slice[:f"{-(frame_time + dead_time) + wiggle}s"]
    np.testing.assert_equal(sliced.get_image("rgb"), ref_sliced[:-1])

    # until frame 2 signal stop
    sliced = first_slice[:f"{-(frame_time + dead_time)}s"]
    np.testing.assert_equal(sliced.get_image("rgb"), ref_sliced[:-2])


def test_scan_slicing_by_timestamps(test_scan_slicing):
    multi_frame, ref = test_scan_slicing
    multi_frame = multi_frame[:]

    dt = np.int64(ref.timestamps.dt)
    start = ref.infowave.data.start + (dt * ref.infowave.line_padding)

    scan_time = (
        dt
        * ref.infowave.samples_per_pixel
        * ref.metadata.lines_per_frame
        * ref.metadata.pixels_per_line
        + (ref.metadata.lines_per_frame * dt * ref.infowave.line_padding)
    )
    dead_time = ref.infowave.line_padding * dt * 2
    frame_time = scan_time + dead_time

    # from frame 2 start until frame 4 start
    sliced = multi_frame[start + 2 * frame_time : start + 4 * frame_time]
    np.testing.assert_equal(sliced.get_image("rgb"), ref.image[2:4])

    # from frame 2 start until end of frame 4 signal stop
    sliced = multi_frame[start + 2 * frame_time : start + (4 * frame_time + scan_time)]
    np.testing.assert_equal(sliced.get_image("rgb"), ref.image[2:4])

    # from frame 2 start until just after end of frame 4 signal stop
    sliced = multi_frame[start + 2 * frame_time : start + (4 * frame_time + scan_time + dt)]
    np.testing.assert_equal(sliced.get_image("rgb"), ref.image[2:5])

    # until just after end of frame 4 signal stop
    sliced = multi_frame[: (start + (4 * frame_time + scan_time + dt))]
    np.testing.assert_equal(sliced.get_image("rgb"), ref.image[:5])

    # from frame 5 start
    sliced = multi_frame[(start + 5 * frame_time) :]
    np.testing.assert_equal(sliced.get_image("rgb"), ref.image[5:])

    # iterative slicing; first slice from frame 2 start until just after frame 4 signal stop
    first_slice = multi_frame[start + 2 * frame_time : start + (4 * frame_time + scan_time + dt)]
    ref_sliced = ref.image[2:5]

    # from start, past stop
    sliced = first_slice[start : start + 10 * frame_time]
    np.testing.assert_equal(sliced.get_image("rgb"), ref_sliced)

    # from frame 3 start until just after frame 3 signal stop
    sliced = first_slice[start + 3 * frame_time : start + (3 * frame_time + scan_time + dt)]
    np.testing.assert_equal(sliced.get_image("rgb"), ref_sliced[1])
