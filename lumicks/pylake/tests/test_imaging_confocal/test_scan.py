import matplotlib.pyplot as plt
import numpy as np
import pytest
from lumicks.pylake.adjustments import ColorAdjustment
from lumicks.pylake.channel import empty_slice
from lumicks.pylake.detail.imaging_mixins import _FIRST_TIMESTAMP

from ..data.mock_confocal import generate_scan

start = np.int64(20e9)
dt = np.int64(62.5e9)


# CAVE: If you want to test a cached property, after having modified parameters that change the
# value of the property, ensure to clear the `_cache` attribute before and after testing. To achieve
# both, you can monkeypatch the `_cache` attribute with an empty dict.
@pytest.fixture(scope="module")
def test_scans():
    scans = {}

    oneframe = np.random.poisson(10, (4, 5, 3))
    multiframe = np.random.poisson(10, (2, 4, 3, 3))

    scans["fast Y slow X"] = generate_scan(
        "fast Y slow X",
        oneframe,
        pixel_sizes_nm=[191.0, 197.0],
        axes=[1, 0],
        with_ref=True,
        multi_color=True,
    )

    scans["fast Y slow X multiframe"] = generate_scan(
        "fast Y slow X multiframe",
        multiframe,
        pixel_sizes_nm=[191.0, 197.0],
        axes=[1, 0],
        with_ref=True,
        multi_color=True,
    )

    scans["fast X slow Z multiframe"] = generate_scan(
        "fast X slow Z multiframe",
        multiframe,
        pixel_sizes_nm=[191.0, 197.0],
        axes=[0, 2],
        with_ref=True,
        multi_color=True,
    )

    scans["fast Y slow Z multiframe"] = generate_scan(
        "fast Y slow Z multiframe",
        multiframe,
        pixel_sizes_nm=[191.0, 197.0],
        axes=[1, 2],
        with_ref=True,
        multi_color=True,
    )

    image_wo_red = oneframe.copy()
    image_wo_red[:, :, 0] = 0
    scans["red channel missing"] = generate_scan(
        "red channel missing",
        image_wo_red,
        pixel_sizes_nm=[191.0, 197.0],
        axes=[1, 0],
        with_ref=True,
        multi_color=True,
    )
    setattr(scans["red channel missing"][0].file, "red_photon_count", empty_slice)

    image_wo_rb = oneframe.copy()
    image_wo_rb[:, :, (0, 2)] = 0
    scans["rb channels missing"] = generate_scan(
        "rb channels missing",
        image_wo_rb,
        pixel_sizes_nm=[191.0, 197.0],
        axes=[1, 0],
        with_ref=True,
        multi_color=True,
    )
    setattr(scans["rb channels missing"][0].file, "red_photon_count", empty_slice)
    setattr(scans["rb channels missing"][0].file, "blue_photon_count", empty_slice)

    return scans


@pytest.mark.parametrize(
    "scanname",
    [
        "fast Y slow X",
        "fast Y slow X multiframe",
        "fast X slow Z multiframe",
        "fast Y slow Z multiframe",
    ],
)
def test_scan_attrs(test_scans, scanname):
    scan, ref = test_scans[scanname]

    assert repr(scan) == f"Scan(pixels=({ref.pixels_per_line}, {ref.lines_per_frame}))"
    np.testing.assert_allclose(scan.timestamps, ref.timestamps)
    assert scan.num_frames == ref.number_of_frames
    assert scan.pixels_per_line == ref.pixels_per_line
    assert scan.lines_per_frame == ref.lines_per_frame
    assert len(scan.infowave) == len(ref.infowave)
    assert scan.get_image("rgb").shape == ref.shape
    assert scan.get_image("red").shape == ref.shape[:-1]
    assert scan.get_image("blue").shape == ref.shape[:-1]
    assert scan.get_image("green").shape == ref.shape[:-1]
    assert scan.fast_axis == ref.fast_axis
    np.testing.assert_allclose(scan.pixelsize_um, ref.pixelsize_um)
    np.testing.assert_allclose(scan.center_point_um["x"], ref.center_point_um["x"])
    np.testing.assert_allclose(scan.center_point_um["y"], ref.center_point_um["y"])
    np.testing.assert_allclose(scan.center_point_um["z"], ref.center_point_um["z"])
    np.testing.assert_allclose(scan.size_um, ref.size_um)

    with pytest.deprecated_call():
        assert scan.rgb_image.shape == ref.shape
    with pytest.deprecated_call():
        assert scan.red_image.shape == ref.shape[:-1]
    with pytest.deprecated_call():
        assert scan.blue_image.shape == ref.shape[:-1]
    with pytest.deprecated_call():
        assert scan.green_image.shape == ref.shape[:-1]


@pytest.mark.parametrize(
    "missing_channels",
    [["red"], ["red", "blue"], ["red", "green", "blue"]],
)
def test_scan_attrs_missing_channels(test_scans, missing_channels, monkeypatch):
    scan, ref = test_scans["fast Y slow X"]
    for channel in missing_channels:
        monkeypatch.setattr(scan.file, f"{channel}_photon_count", empty_slice)
        monkeypatch.setattr(scan, "_cache", {})

    rgb = scan.get_image("rgb")
    assert rgb.shape == ref.shape
    for i, channel in enumerate(["red", "green", "blue"]):
        if channel in missing_channels:
            assert not np.any(rgb[:, :, :, i] if ref.multi_frame else rgb[:, :, i])
            np.testing.assert_equal(scan.get_image(channel), np.zeros(ref.shape[:-1]))
        else:
            assert scan.get_image(channel).shape == ref.shape[:-1]
            np.testing.assert_allclose(
                scan.get_image(channel),
                ref.image[:, :, :, i] if ref.multi_frame else ref.image[:, :, i],
            )


def test_slicing(test_scans):
    scan = generate_scan(
        "multiframe_poisson",
        np.random.poisson(10, (10, 3, 4)),
        pixel_sizes_nm=[5, 5],
        start=start,
        dt=dt,
        samples_per_pixel=5,
        line_padding=3,
    )
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

    # reverse slice
    with pytest.raises(NotImplementedError, match="Slice is empty."):
        scan[5:3]
    with pytest.raises(NotImplementedError, match="Slice is empty."):
        scan[-2:-4]

    # empty slice
    with pytest.raises(NotImplementedError, match="Slice is empty."):
        scan[5:5]
    with pytest.raises(NotImplementedError, match="Slice is empty."):
        scan[15:16]

    # Verify no side effects
    scan[0]
    assert scan.num_frames == 10


def test_damaged_scan(test_scans, monkeypatch):
    scan, ref = test_scans["fast Y slow X"]
    middle = scan.red_photon_count.timestamps[scan.red_photon_count.timestamps.size // 2]

    # Assume the user incorrectly exported only a partial scan
    monkeypatch.setattr(scan, "start", ref.start - ref.dt)
    monkeypatch.setattr(scan, "_cache", {})

    with pytest.raises(RuntimeError):
        scan.get_image()

    # Test for workaround for a bug in the STED delay mechanism which could result in scan start
    # times ending up within the sample time.
    monkeypatch.setattr(scan, "start", middle - ref.dt + 1)
    monkeypatch.setattr(scan, "_cache", {})

    # `get_image()` should not raise, but change the start appropriately to work around sted bug
    scan.get_image("red").shape
    np.testing.assert_allclose(scan.start, middle)


@pytest.mark.parametrize(
    "scanname, channel",
    [
        ("fast Y slow X multiframe", "blue"),
        ("fast X slow Z multiframe", "rgb"),
        ("fast Y slow Z multiframe", "rgb"),
    ],
)
def test_plotting(test_scans, scanname, channel):
    scan, ref = test_scans[scanname]
    scan.plot(channel=channel)
    image = plt.gca().get_images()[0]
    scan_image = scan.get_image(channel)[0]
    if channel == "rgb":
        np.testing.assert_allclose(image.get_array(), scan_image / np.max(scan_image))
    else:
        np.testing.assert_allclose(image.get_array(), scan_image)
    np.testing.assert_allclose(image.get_extent(), [0, *ref.size_um, 0])
    plt.close()

    # test invalid indices (num_frames=2)
    with pytest.raises(IndexError):
        scan.plot(channel="rgb", frame=ref.number_of_frames + 1)
    with pytest.raises(IndexError, match="negative indexing is not supported."):
        scan.plot(channel="rgb", frame=-1)


def test_deprecated_plotting(test_scans):
    scan, _ = test_scans["fast Y slow X multiframe"]
    with pytest.deprecated_call():
        scan.plot_red()
    with pytest.deprecated_call():
        scan.plot_green()
    with pytest.deprecated_call():
        scan.plot_blue()
    with pytest.deprecated_call():
        scan.plot_rgb()
    with pytest.warns(
        DeprecationWarning,
        match=r"The call signature of `plot\(\)` has changed: Please, provide `axes` as a "
        "keyword argument.",
    ):
        ih = scan.plot("blue", None)
        np.testing.assert_allclose(ih.get_array(), scan.get_image("blue")[0])
        plt.close()
    # Test rejection of deprecated call with positional `axes` and double keyword assignment
    with pytest.raises(TypeError, match=r"`Scan.plot\(\)` got multiple values for argument `axes`"):
        scan.plot("rgb", None, axes=None)


@pytest.mark.parametrize(
    "scanname, tiffname",
    [
        ("fast Y slow X", "single_frame.tiff"),
        ("fast Y slow X multiframe", "multi_frame.tiff"),
    ],
)
def test_export_tiff(scanname, tiffname, tmp_path, test_scans, grab_tiff_tags):
    from os import stat

    scan, _ = test_scans[scanname]
    filename = tmp_path / tiffname
    scan.export_tiff(filename)
    assert stat(filename).st_size > 0
    # Check if tags were properly stored, i.e. test functionality of `_tiff_image_metadata()`,
    # `_tiff_timestamp_ranges()` and `_tiff_writer_kwargs()`
    tiff_tags = grab_tiff_tags(filename)
    assert len(tiff_tags) == scan.num_frames
    for tags, timestamp_range in zip(tiff_tags, scan._tiff_timestamp_ranges()):
        assert tags["ImageDescription"] == scan._tiff_image_metadata()
        assert tags["DateTime"] == f"{timestamp_range[0]}:{timestamp_range[1]}"
        assert tags["Software"] == scan._tiff_writer_kwargs()["software"]
        np.testing.assert_allclose(
            tags["XResolution"][0] / tags["XResolution"][1],
            scan._tiff_writer_kwargs()["resolution"][0],
            rtol=1e-1,
        )
        np.testing.assert_allclose(
            tags["YResolution"][0] / tags["YResolution"][1],
            scan._tiff_writer_kwargs()["resolution"][1],
            rtol=1e-1,
        )
        assert tags["ResolutionUnit"] == 3  # 3 = Centimeter


def test_deprecated_save_tiff(tmp_path, test_scans):
    from os import stat

    scan, ref = test_scans["fast Y slow X"]
    match = (
        r"This method has been renamed to `export_tiff\(\)` to more accurately reflect that it is "
        r"exporting to a different format."
    )
    with pytest.warns(DeprecationWarning, match=match):
        scan.save_tiff(f"{tmp_path}/single_frame_dep.tiff")
        assert stat(f"{tmp_path}/single_frame_dep.tiff").st_size > 0


def test_movie_export(tmpdir_factory, test_scans):
    from os import stat

    tmpdir = tmpdir_factory.mktemp("pylake")

    scan, _ = test_scans["fast Y slow X multiframe"]
    scan.export_video("red", f"{tmpdir}/red.gif", start_frame=0, stop_frame=2)
    assert stat(f"{tmpdir}/red.gif").st_size > 0
    scan.export_video("rgb", f"{tmpdir}/rgb.gif", start_frame=0, stop_frame=2)
    assert stat(f"{tmpdir}/rgb.gif").st_size > 0

    # test stop frame > num frames
    with pytest.raises(IndexError):
        scan.export_video("rgb", f"{tmpdir}/rgb.gif", start_frame=0, stop_frame=4)

    with pytest.raises(ValueError, match="Channel should be red, green, blue or rgb"):
        scan.export_video("gray", "dummy.gif")  # Gray is not a color!


def test_deprecated_movie_export(tmpdir_factory, test_scans):
    from os import stat

    tmpdir = tmpdir_factory.mktemp("pylake")
    scan, _ = test_scans["fast Y slow X multiframe"]
    for channel in ("red", "green", "blue", "rgb"):
        with pytest.warns(DeprecationWarning):
            getattr(scan, f"export_video_{channel}")(
                f"{tmpdir}/dep_{channel}.gif", start_frame=0, end_frame=2
            )
            assert stat(f"{tmpdir}/dep_{channel}.gif").st_size > 0


@pytest.mark.parametrize(
    "dim_x, dim_y, line_padding, start, dt, samples_per_pixel",
    [
        (5, 6, 3, 14, 4, 4),
        (3, 4, 60, 1592916040906356300, 12800, 30),
        (3, 2, 60, 1592916040906356300, 12800, 3000),
    ],
)
def test_single_frame_times(dim_x, dim_y, line_padding, start, dt, samples_per_pixel):
    img = np.ones((dim_x, dim_y))
    scan = generate_scan(
        "test",
        img,
        pixel_sizes_nm=[1, 1],
        start=start,
        dt=dt,
        samples_per_pixel=samples_per_pixel,
        line_padding=line_padding,
    )
    frame_times = scan.frame_timestamp_ranges()
    assert len(frame_times) == 1
    assert frame_times[0][0] == start + line_padding * dt
    line_time = dt * (img.shape[1] * samples_per_pixel + 2 * line_padding) * img.shape[0]
    assert frame_times[0][1] == start + line_time - line_padding * dt

    # For the single frame case, there is no dead time, so these are identical
    frame_times_inclusive = scan.frame_timestamp_ranges(include_dead_time=True)
    assert len(frame_times_inclusive) == 1
    assert frame_times_inclusive[0][0] == frame_times[0][0]
    assert frame_times_inclusive[0][1] == frame_times[0][1]


@pytest.mark.parametrize(
    "dim_x, dim_y, frames, line_padding, start, dt, samples_per_pixel",
    [
        (5, 6, 3, 3, 14, 4, 4),
        (3, 4, 4, 60, 1592916040906356300, 12800, 30),
        (3, 2, 3, 60, 1592916040906356300, 12800, 3000),
    ],
)
def test_multiple_frame_times(dim_x, dim_y, frames, line_padding, start, dt, samples_per_pixel):
    img = np.ones((frames, dim_x, dim_y))
    scan = generate_scan(
        "test",
        img,
        pixel_sizes_nm=[1, 1],
        start=start,
        dt=dt,
        samples_per_pixel=samples_per_pixel,
        line_padding=line_padding,
    )
    frame_times = scan.frame_timestamp_ranges()

    line_time = dt * (img.shape[2] * samples_per_pixel + 2 * line_padding) * img.shape[1]
    assert scan.num_frames == frames
    assert len(frame_times) == scan.num_frames
    assert frame_times[0][0] == start + line_padding * dt
    assert frame_times[0][1] == start + line_time - line_padding * dt
    assert frame_times[1][0] == start + line_padding * dt + line_time
    assert frame_times[1][1] == start + 2 * line_time - line_padding * dt
    assert frame_times[-1][0] == start + line_padding * dt + (len(frame_times) - 1) * line_time
    assert frame_times[-1][1] == start + len(frame_times) * line_time - line_padding * dt

    def compare_inclusive(frame_times_inclusive):
        # Start times should be the same
        assert len(frame_times_inclusive) == scan.num_frames
        assert frame_times_inclusive[0][0] == frame_times[0][0]
        assert frame_times_inclusive[1][0] == frame_times[1][0]
        assert frame_times_inclusive[-1][0] == frame_times[-1][0]

        assert frame_times_inclusive[0][1] == frame_times[1][0]
        assert frame_times_inclusive[1][1] == frame_times[2][0]
        assert frame_times_inclusive[-1][1] == frame_times[-1][0] + (
            frame_times[1][0] - frame_times[0][0]
        )

    with pytest.deprecated_call():
        compare_inclusive(scan.frame_timestamp_ranges(False))

    with pytest.deprecated_call():
        compare_inclusive(scan.frame_timestamp_ranges(exclude=False))

    compare_inclusive(scan.frame_timestamp_ranges(include_dead_time=True))

    with pytest.raises(
        ValueError, match="Do not specify both exclude and include_dead_time parameters"
    ):
        scan.frame_timestamp_ranges(False, include_dead_time=True)


def test_scan_plot_rgb_absolute_color_adjustment(test_scans):
    """Tests whether we can set an absolute color range for an RGB plot."""
    scan, _ = test_scans["fast Y slow X"]

    fig = plt.figure()
    lb, ub = np.array([1, 2, 3]), np.array([2, 3, 4])
    scan.plot(channel="rgb", adjustment=ColorAdjustment(lb, ub, mode="absolute"))
    image = plt.gca().get_images()[0]
    np.testing.assert_allclose(
        image.get_array(), np.clip((scan.get_image("rgb") - lb) / (ub - lb), 0, 1)
    )
    plt.close(fig)


def test_scan_plot_single_channel_absolute_color_adjustment(test_scans):
    """Tests whether we can set an absolute color range for a single channel plot."""
    scan, _ = test_scans["fast Y slow X"]

    lbs, ubs = np.array([1, 2, 3]), np.array([2, 3, 4])
    for lb, ub, channel in zip(lbs, ubs, ("red", "green", "blue")):
        # Test whether setting RGB values and then sampling one of them works correctly.
        fig = plt.figure()
        scan.plot(channel=channel, adjustment=ColorAdjustment(lbs, ubs, mode="absolute"))
        image = plt.gca().get_images()[0]
        np.testing.assert_allclose(
            image.get_array(), scan.get_image(channel)
        )  # getattr(scan, f"{channel}_image"))
        np.testing.assert_allclose(image.get_clim(), [lb, ub])
        plt.close(fig)

        # Test whether setting a single color works correctly (should use the same for R G and B).
        fig = plt.figure()
        scan.plot(channel=channel, adjustment=ColorAdjustment(lb, ub, mode="absolute"))
        image = plt.gca().get_images()[0]
        np.testing.assert_allclose(
            image.get_array(), scan.get_image(channel)
        )  # getattr(scan, f"{channel}_image"))
        np.testing.assert_allclose(image.get_clim(), [lb, ub])
        plt.close(fig)


def test_plot_rgb_percentile_color_adjustment(test_scans):
    """Tests whether we can set a percentile color range for an RGB plot."""
    scan, _ = test_scans["fast Y slow X multiframe"]

    fig = plt.figure()
    lb, ub = np.array([1, 5, 10]), np.array([80, 90, 80])
    scan.plot(channel="rgb", adjustment=ColorAdjustment(lb, ub, mode="percentile"))
    bounds = np.array(
        [
            np.percentile(img, [mini, maxi])
            for img, mini, maxi in zip(np.moveaxis(scan.get_image("rgb")[0], 2, 0), lb, ub)
        ]
    )
    lb, ub = (b for b in np.moveaxis(bounds, 1, 0))
    image = plt.gca().get_images()[0]
    np.testing.assert_allclose(
        image.get_array(), np.clip((scan.get_image("rgb")[0] - lb) / (ub - lb), 0, 1)
    )
    plt.close(fig)


def test_plot_single_channel_percentile_color_adjustment(test_scans):
    """Tests whether we can set a percentile color range for separate channel plots."""
    scan, _ = test_scans["fast Y slow X multiframe"]

    lbs, ubs = np.array([1, 5, 10]), np.array([80, 90, 80])
    for lb, ub, channel in zip(lbs, ubs, ("red", "green", "blue")):
        # We need to test both specifying all bounds at once and specifying just one bound
        for used_lb, used_ub in ((lb, ub), (lbs, ubs)):
            fig = plt.figure()
            scan.plot(
                channel=channel, adjustment=ColorAdjustment(used_lb, used_ub, mode="percentile")
            )
            image = scan.get_image(channel)
            lb_abs, ub_abs = np.percentile(image[0], [lb, ub])
            plotted_image = plt.gca().get_images()[0]
            np.testing.assert_allclose(plotted_image.get_array(), image[0])
            np.testing.assert_allclose(plotted_image.get_clim(), [lb_abs, ub_abs])
            plt.close(fig)


@pytest.mark.parametrize(
    "scanname",
    [
        "fast Y slow X",
        "fast X slow Z multiframe",
        "fast Y slow X multiframe",
        "fast Y slow Z multiframe",
    ],
)
def test_scan_pixel_time(test_scans, scanname):
    scan, ref = test_scans[scanname]
    np.testing.assert_allclose(scan.pixel_time_seconds, ref.dt * ref.samples_per_pixel * 1e-9)


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
def test_scan_cropping(x_min, x_max, y_min, y_max, test_scans):
    valid_scans = [
        "fast Y slow X",
        "fast Y slow X multiframe",
        "fast X slow Z multiframe",
        "fast Y slow Z multiframe",
        "red channel missing",
        "rb channels missing",
    ]
    for key in valid_scans:
        scan, ref = test_scans[key]

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
def test_scan_get_item_slicing(all_slices, test_scans):
    """Test slicing, slicing is given as image, y, x"""

    valid_scans = [
        "fast Y slow X",
        "fast Y slow X multiframe",
        "fast X slow Z multiframe",
        "fast Y slow Z multiframe",
        "red channel missing",
        "rb channels missing",
    ]

    for key in valid_scans:
        scan, ref = test_scans[key]

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


def test_slicing_cropping_separate_actions(test_scans):
    """Test whether cropping works in a sequential fashion"""
    multi_frame, single_frame = (
        generate_scan(
            "test",
            np.random.rand(num_frames, 7, 5),
            pixel_sizes_nm=[1, 1],
            start=0,
            dt=100,
            samples_per_pixel=1,
            line_padding=1,
        )
        for num_frames in (8, 1)
    )

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


def test_error_handling_multidim_indexing():
    multi_frame, single_frame = (
        generate_scan(
            "test",
            np.random.rand(num_frames, 7, 5),
            pixel_sizes_nm=[1, 1],
            start=0,
            dt=100,
            samples_per_pixel=1,
            line_padding=1,
        )
        for num_frames in (8, 1)
    )

    with pytest.raises(IndexError, match="Frame index out of range"):
        multi_frame[8]
    with pytest.raises(IndexError, match="Frame index out of range"):
        multi_frame[-9]
    with pytest.raises(IndexError, match="Frame index out of range"):
        single_frame[1]
    with pytest.raises(
        IndexError, match="Scalar indexing is not supported for spatial coordinates"
    ):
        multi_frame[0, 4, 3]


@pytest.mark.parametrize(
    "scanname",
    [
        "fast Y slow X",
        "fast Y slow X multiframe",
        "fast X slow Z multiframe",
        "fast Y slow Z multiframe",
        "red channel missing",
        "rb channels missing",
    ],
)
def test_empty_slices_due_to_out_of_bounds(scanname, test_scans):
    scan, _ = test_scans[scanname]
    shape = scan[0].get_image("green").shape
    with pytest.raises(NotImplementedError, match="Slice is empty."):
        scan[0][0, shape[0] : shape[0] + 10, 0:2]
    with pytest.raises(NotImplementedError, match="Slice is empty."):
        scan[0][0, shape[0] : shape[0] + 10]
    with pytest.raises(NotImplementedError, match="Slice is empty."):
        scan[0][0, :, shape[1] : shape[1] + 10]
    with pytest.raises(NotImplementedError, match="Slice is empty."):
        scan[0][0, :, -10:0]


def test_slice_by_list_disallowed(test_scans):
    scan, _ = test_scans["fast Y slow X multiframe"]

    with pytest.raises(IndexError, match="Indexing by list is not allowed"):
        scan[[0, 1], :, :]

    with pytest.raises(IndexError, match="Indexing by list is not allowed"):
        scan[:, [0, 1], :]

    class Dummy:
        pass

    with pytest.raises(IndexError, match="Indexing by Dummy is not allowed"):
        scan[Dummy(), :, :]

    with pytest.raises(IndexError, match="Slicing by Dummy is not supported"):
        scan[Dummy():Dummy(), :, :]


def test_crop_missing_channel(test_scans):
    """Make sure that missing channels are handled appropriately when cropping"""
    scan, _ = test_scans["rb channels missing"]
    np.testing.assert_equal(
        scan[:, 0:2, 1:3].get_image("red"),
        np.zeros((2, 2))
    )


def test_scan_slicing_by_time():
    start = _FIRST_TIMESTAMP + 100
    num_frames = 10
    line_padding = 10
    dt = int(1e7)
    multi_frame = generate_scan(
        "test",
        np.random.randint(0, 10, size=(num_frames, 2, 2)),
        pixel_sizes_nm=[1, 1],
        start=start - line_padding * dt,
        dt=dt,
        samples_per_pixel=15,
        line_padding=line_padding,
    )

    # We deliberately shift the start to the start of the first frame (post padding). This makes
    # the tests much more readable.
    multi_frame = multi_frame[:]
    ts = multi_frame.frame_timestamp_ranges()

    def compare_frames(frames, new_scan):
        assert new_scan.num_frames == len(frames)
        for new_frame_index, index in enumerate(frames):
            frame = multi_frame[index].get_image("rgb")
            new_frame = new_scan[new_frame_index].get_image("rgb")
            np.testing.assert_equal(frame, new_frame)

    compare_frames([2], multi_frame["2s":"2.81s"])
    compare_frames([2], multi_frame["2s":"3.8s"])
    compare_frames([2, 3], multi_frame["2s":"3.81s"])
    compare_frames([1, 2], multi_frame["1s":"3s"])
    compare_frames([2, 3, 4, 5, 6, 7, 8, 9], multi_frame["2s":])  # until end
    compare_frames([3, 4, 5, 6, 7, 8, 9], multi_frame["2.1s":])  # from beginning
    compare_frames([0, 1], multi_frame[:"1.81s"])  # until end
    compare_frames([0], multi_frame[:"1.8s"])  # from beginning

    compare_frames([3, 4, 5], multi_frame["2s":"5.81s"]["1s":"3.81s"])  # iterative
    compare_frames([3, 4], multi_frame["2s":"5.81s"]["1s":"3.80s"])  # iterative
    compare_frames([3, 4, 5], multi_frame["2s":"5.81s"]["1s":])  # iterative

    # Note that the from-end tests are different than the ones on correlatedstacks because the mock
    # scan has the dead time half on the end of this frame, and half before this frame.
    # The stop time of this scan is at 10 seconds, the last frame runs from 9 to 9.8 seconds, this
    # means that to chop off the last two, we need to go -1.2 from end.
    compare_frames([0, 1, 2, 3, 4, 5, 6, 7, 8], multi_frame[:"-1.199s"])  # negative indexing
    compare_frames([0, 1, 2, 3, 4, 5, 6, 7], multi_frame[:"-1.2s"])  # negative indexing with time
    compare_frames([8, 9], multi_frame["-2s":])  # negative indexing with time
    compare_frames([9], multi_frame["-1.79s":])  # negative indexing with time
    compare_frames([2, 3, 4], multi_frame["2s":"5.81s"][:"-1.199s"])  # iterative with from end
    compare_frames([2, 3], multi_frame["2s":"5.81s"][:"-1.2s"])  # iterative with from end

    # Slice by timestamps
    compare_frames([2, 3], multi_frame[start + int(2e9):start + int(4e9)])
    compare_frames([2, 3], multi_frame[start + int(2e9):start + int(4.8e9)])
    compare_frames([2, 3, 4], multi_frame[start + int(2e9):start + int(4.81e9)])
    compare_frames([0, 1, 2, 3, 4], multi_frame[:start + int(4.81e9)])
    compare_frames([5, 6, 7, 8, 9], multi_frame[start + int(5e9):])
    compare_frames(
        [2, 3, 4], multi_frame[start + int(2e9):start + int(4.81e9)][start:start+int(100e9)]
    )
    compare_frames(
        [3], multi_frame[start + int(2e9):start + int(4.81e9)][start + int(3e9):start + int(3.81e9)]
    )
