import re

import numpy as np
import pytest
import matplotlib.pyplot as plt

from lumicks.pylake.adjustments import ColorAdjustment
from lumicks.pylake.detail.imaging_mixins import _FIRST_TIMESTAMP

from ..data.mock_confocal import generate_scan


def test_scan_attrs(test_scans):
    scan = test_scans["fast Y slow X"]
    assert repr(scan) == "Scan(pixels=(4, 5))"

    # fmt: off
    reference_timestamps = np.array(
        [
            [20062500000, 20812500000, 22187500000, 23562500000, 24937500000],
            [20250000000, 21625000000, 22375000000, 23750000000, 25125000000],
            [20437500000, 21812500000, 23187500000, 23937500000, 25312500000],
            [20625000000, 22000000000, 23375000000, 24750000000, 25500000000],
        ]
    ).T

    np.testing.assert_allclose(scan.timestamps, np.transpose(reference_timestamps))
    assert scan.num_frames == 1
    assert scan.pixels_per_line == 4
    assert scan.lines_per_frame == 5
    assert len(scan.infowave) == 90
    assert scan.get_image("rgb").shape == (4, 5, 3)
    assert scan.get_image("red").shape == (4, 5)
    assert scan.get_image("blue").shape == (4, 5)
    assert scan.get_image("green").shape == (4, 5)

    assert scan.fast_axis == "Y"
    np.testing.assert_allclose(scan.pixelsize_um, [197 / 1000, 191 / 1000])
    np.testing.assert_allclose(scan.center_point_um["x"], 58.075877109272604)
    np.testing.assert_allclose(scan.center_point_um["y"], 31.978375270573267)
    np.testing.assert_allclose(scan.center_point_um["z"], 0)
    np.testing.assert_allclose(scan.size_um, [0.197 * 5, 0.191 * 4])

    scan = test_scans["fast Y slow X multiframe"]
    reference_timestamps2 = np.zeros((2, 4, 3))
    reference_timestamps2[0, :, :] = reference_timestamps.T[:, :3]
    reference_timestamps2[1, :, :2] = reference_timestamps.T[:, 3:]

    np.testing.assert_allclose(scan.timestamps, reference_timestamps2)
    assert scan.num_frames == 2
    assert scan.pixels_per_line == 4
    assert scan.lines_per_frame == 3
    assert len(scan.infowave) == 90
    assert scan.get_image("rgb").shape == (2, 4, 3, 3)
    assert scan.get_image("red").shape == (2, 4, 3)
    assert scan.get_image("blue").shape == (2, 4, 3)
    assert scan.get_image("green").shape == (2, 4, 3)
    assert scan.fast_axis == "Y"
    np.testing.assert_allclose(scan.pixelsize_um, [197 / 1000, 191 / 1000])
    np.testing.assert_allclose(scan.center_point_um["x"], 58.075877109272604)
    np.testing.assert_allclose(scan.center_point_um["y"], 31.978375270573267)
    np.testing.assert_allclose(scan.center_point_um["z"], 0)
    np.testing.assert_allclose(scan.size_um, [0.197 * 3, 0.191 * 4])

    scan = test_scans["fast X slow Z multiframe"]
    reference_timestamps2 = np.zeros((2, 4, 3))
    reference_timestamps2[0, :, :] = reference_timestamps.T[:, :3]
    reference_timestamps2[1, :, :2] = reference_timestamps.T[:, 3:]
    reference_timestamps2 = reference_timestamps2.transpose([0, 2, 1])

    np.testing.assert_allclose(scan.timestamps, reference_timestamps2)
    assert scan.num_frames == 2
    assert scan.pixels_per_line == 4
    assert scan.lines_per_frame == 3
    assert len(scan.infowave) == 90
    assert scan.get_image("rgb").shape == (2, 3, 4, 3)
    assert scan.get_image("red").shape == (2, 3, 4)
    assert scan.get_image("blue").shape == (2, 3, 4)
    assert scan.get_image("green").shape == (2, 3, 4)
    assert scan.fast_axis == "X"
    np.testing.assert_allclose(scan.pixelsize_um, [191 / 1000, 197 / 1000])
    np.testing.assert_allclose(scan.center_point_um["x"], 58.075877109272604)
    np.testing.assert_allclose(scan.center_point_um["y"], 31.978375270573267)
    np.testing.assert_allclose(scan.center_point_um["z"], 0)
    np.testing.assert_allclose(scan.size_um, [0.191 * 4, 0.197 * 3])

    scan = test_scans["fast Y slow Z multiframe"]
    reference_timestamps2 = np.zeros((2, 4, 3))
    reference_timestamps2[0, :, :] = reference_timestamps.T[:, :3]
    reference_timestamps2[1, :, :2] = reference_timestamps.T[:, 3:]
    reference_timestamps2 = reference_timestamps2.transpose([0, 2, 1])

    np.testing.assert_allclose(scan.timestamps, reference_timestamps2)
    assert scan.num_frames == 2
    assert scan.pixels_per_line == 4
    assert scan.lines_per_frame == 3
    assert len(scan.infowave) == 90
    assert scan.get_image("rgb").shape == (2, 3, 4, 3)
    assert scan.get_image("red").shape == (2, 3, 4)
    assert scan.get_image("blue").shape == (2, 3, 4)
    assert scan.get_image("green").shape == (2, 3, 4)
    assert scan.fast_axis == "Y"
    np.testing.assert_allclose(scan.pixelsize_um, [191 / 1000, 197 / 1000])
    np.testing.assert_allclose(scan.center_point_um["x"], 58.075877109272604)
    np.testing.assert_allclose(scan.center_point_um["y"], 31.978375270573267)
    np.testing.assert_allclose(scan.center_point_um["z"], 0)
    np.testing.assert_allclose(scan.size_um, [0.191 * 4, 0.197 * 3])

    scan = test_scans["red channel missing"]
    rgb = scan.get_image("rgb")
    assert rgb.shape == (4, 5, 3)
    assert not np.any(rgb[:, :, 0])
    np.testing.assert_equal(scan.get_image("red"), np.zeros((4, 5)))

    assert scan.get_image("blue").shape == (4, 5)
    assert scan.get_image("green").shape == (4, 5)

    scan = test_scans["rb channels missing"]
    rgb = scan.get_image("rgb")
    assert rgb.shape == (4, 5, 3)
    assert not np.any(rgb[:, :, 0])
    assert not np.any(rgb[:, :, 2])
    np.testing.assert_equal(scan.get_image("red"), np.zeros((4, 5)))
    np.testing.assert_equal(scan.get_image("blue"), np.zeros((4, 5)))
    assert scan.get_image("green").shape == (4, 5)

    scan = test_scans["all channels missing"]
    np.testing.assert_equal(scan.get_image("red"), np.zeros((4, 5)))
    np.testing.assert_equal(scan.get_image("green"), np.zeros((4, 5)))
    np.testing.assert_equal(scan.get_image("blue"), np.zeros((4, 5)))


def test_slicing(test_scans):
    scan0 = test_scans["multiframe_poisson"]
    assert scan0.num_frames == 10

    def compare_frames(original_frames, new_scan):
        assert new_scan.num_frames == len(original_frames)
        for new_frame_index, index in enumerate(original_frames):
            frame = scan0.get_image("red")[index]
            new_frame = (
                new_scan.get_image("red")[new_frame_index]
                if new_scan.num_frames > 1
                else new_scan.get_image("red")
            )
            np.testing.assert_equal(frame, new_frame)

    compare_frames([0], scan0[0])  # first frame
    compare_frames([9], scan0[-1])  # last frame
    compare_frames([3], scan0[3])  # single frame
    compare_frames([2, 3, 4], scan0[2:5])  # slice
    compare_frames([0, 1, 2], scan0[:3])  # from beginning
    compare_frames([8, 9], scan0[8:])  # until end
    compare_frames([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], scan0[:])  # all
    compare_frames([7], scan0[-3])  # negative index
    compare_frames([0, 1, 2, 3], scan0[:-6])  # until negative index
    compare_frames([5, 6, 7], scan0[5:-2])  # mixed sign indices
    compare_frames([6, 7], scan0[-4:-2])  # full negative slice
    compare_frames([2], scan0[2:3])  # slice to single frame
    compare_frames([3, 4], scan0[2:6][1:3])  # iterative slicing

    compare_frames([1, 2, 3, 4, 5, 6, 7, 8, 9], scan0[1:100])  # test clamping past the end
    compare_frames([0, 1, 2, 3, 4, 5, 6, 7, 8], scan0[-100:9])  # test clamping past the beginning
    compare_frames([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], scan0[-100:100])  # test clamping both dirs

    # reverse slice (leads to empty scan)
    assert not scan0[5:3]
    assert not scan0[-2:-4]

    # empty slice
    assert not scan0[5:5]
    assert not scan0[15:16]

    # Verify no side effects
    scan0[0]
    assert scan0.num_frames == 10


def test_damaged_scan(test_scans):
    # Assume the user incorrectly exported only a partial scan (62500000 is the time step)
    scan = test_scans["truncated_scan"]
    with pytest.raises(RuntimeError):
        scan.get_image("red").shape

    # Test for workaround for a bug in the STED delay mechanism which could result in scan start times ending up
    # within the sample time.
    scan = test_scans["sted bug"]
    middle = test_scans["fast Y slow X"].red_photon_count.timestamps[5]
    scan.get_image(
        "red"
    ).shape  # should not raise, but change the start appropriately to work around sted bug
    np.testing.assert_allclose(scan.start, middle)


def test_plotting(test_scans):
    scan = test_scans["fast Y slow X multiframe"]
    scan.plot(channel="blue")
    image = plt.gca().get_images()[0]
    np.testing.assert_allclose(image.get_array(), scan.get_image("blue")[0])
    np.testing.assert_allclose(image.get_extent(), [0, 0.197 * 3, 0.191 * 4, 0])
    plt.close()

    scan = test_scans["fast X slow Z multiframe"]
    scan.plot(channel="rgb")
    image = plt.gca().get_images()[0]
    np.testing.assert_allclose(
        image.get_array(), scan.get_image("rgb")[0] / np.max(scan.get_image("rgb")[0])
    )
    np.testing.assert_allclose(image.get_extent(), [0, 0.191 * 4, 0.197 * 3, 0])
    plt.close()

    scan = test_scans["fast Y slow Z multiframe"]
    scan.plot(channel="rgb")
    image = plt.gca().get_images()[0]
    np.testing.assert_allclose(
        image.get_array(), scan.get_image("rgb")[0] / np.max(scan.get_image("rgb")[0])
    )
    np.testing.assert_allclose(image.get_extent(), [0, 0.191 * 4, 0.197 * 3, 0])
    plt.close()

    # test invalid inidices (num_frames=2)
    with pytest.raises(IndexError):
        scan.plot(channel="rgb", frame=4)
    with pytest.raises(IndexError, match="negative indexing is not supported."):
        scan.plot(channel="rgb", frame=-1)


def test_deprecated_plotting(test_scans):
    scan = test_scans["fast Y slow X multiframe"]
    with pytest.raises(
        TypeError, match=re.escape("plot() takes from 1 to 2 positional arguments but 3 were given")
    ):
        ih = scan.plot("blue", None)
        np.testing.assert_allclose(ih.get_array(), scan.get_image("blue")[0])
        plt.close()
    # Test rejection of deprecated call with positional `axes` and double keyword assignment
    with pytest.raises(
        TypeError,
        match=re.escape(
            "plot() takes from 1 to 2 positional arguments but 3 positional "
            "arguments (and 1 keyword-only argument) were given"
        ),
    ):
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

    scan = test_scans[scanname]
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


def test_movie_export(tmpdir_factory, test_scans):
    from os import stat

    tmpdir = tmpdir_factory.mktemp("pylake")

    scan = test_scans["fast Y slow X multiframe"]
    scan.export_video("red", f"{tmpdir}/red.gif", start_frame=0, stop_frame=2)
    assert stat(f"{tmpdir}/red.gif").st_size > 0
    scan.export_video("rgb", f"{tmpdir}/rgb.gif", start_frame=0, stop_frame=2)
    assert stat(f"{tmpdir}/rgb.gif").st_size > 0

    # test stop frame > num frames
    with pytest.raises(IndexError):
        scan.export_video("rgb", f"{tmpdir}/rgb.gif", start_frame=0, stop_frame=4)

    with pytest.raises(ValueError, match="Channel should be red, green, blue or rgb"):
        scan.export_video("gray", "dummy.gif")  # Gray is not a color!


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
        [1, 1],
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
        [1, 1],
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

    compare_inclusive(scan.frame_timestamp_ranges(include_dead_time=True))


def test_scan_plot_rgb_absolute_color_adjustment(test_scans):
    """Tests whether we can set an absolute color range for an RGB plot."""
    scan = test_scans["fast Y slow X"]

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
    scan = test_scans["fast Y slow X"]

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
    scan = test_scans["fast Y slow X multiframe"]

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
    scan = test_scans["fast Y slow X multiframe"]

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
    "scan, pixel_time",
    [
        ("fast Y slow X", 0.1875),
        ("fast X slow Z multiframe", 0.1875),
        ("fast Y slow X multiframe", 0.1875),
        ("fast Y slow Z multiframe", 0.1875),
        ("fast Y slow X", 0.1875),
    ],
)
def test_scan_pixel_time(test_scans, scan, pixel_time):
    np.testing.assert_allclose(test_scans[scan].pixel_time_seconds, pixel_time)


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
        "fast Y slow X multiframe",
        "fast Y slow X",
        "fast X slow Z multiframe",
        "fast Y slow Z multiframe",
        "red channel missing",
        "rb channels missing",
    ]

    for key in valid_scans:
        scan = test_scans[key]

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
        "fast Y slow X multiframe",
        "fast Y slow X",
        "fast X slow Z multiframe",
        "fast Y slow Z multiframe",
        "red channel missing",
        "rb channels missing",
    ]

    for key in valid_scans:
        scan = test_scans[key]

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
            np.random.rand(num_frames, 5, 7),
            [1, 1],
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
            np.random.rand(num_frames, 5, 7),
            [1, 1],
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
    "name",
    [
        "fast Y slow X multiframe",
        "fast Y slow X",
        "fast X slow Z multiframe",
        "fast Y slow Z multiframe",
        "red channel missing",
        "rb channels missing",
    ],
)
def test_empty_slices_due_to_out_of_bounds(name, test_scans):
    shape = test_scans[name][0].get_image("green").shape
    with pytest.raises(NotImplementedError, match="Slice is empty."):
        test_scans[name][0][0, shape[0] : shape[0] + 10, 0:2]
    with pytest.raises(NotImplementedError, match="Slice is empty."):
        test_scans[name][0][0, shape[0] : shape[0] + 10]
    with pytest.raises(NotImplementedError, match="Slice is empty."):
        test_scans[name][0][0, :, shape[1] : shape[1] + 10]
    with pytest.raises(NotImplementedError, match="Slice is empty."):
        test_scans[name][0][0, :, -10:0]


def test_slice_by_list_disallowed(test_scans):
    with pytest.raises(IndexError, match="Indexing by list is not allowed"):
        test_scans["fast Y slow X multiframe"][[0, 1], :, :]

    with pytest.raises(IndexError, match="Indexing by list is not allowed"):
        test_scans["fast Y slow X multiframe"][:, [0, 1], :]

    class Dummy:
        pass

    with pytest.raises(IndexError, match="Indexing by Dummy is not allowed"):
        test_scans["fast Y slow X multiframe"][Dummy(), :, :]

    with pytest.raises(IndexError, match="Slicing by Dummy is not supported"):
        test_scans["fast Y slow X multiframe"][Dummy() : Dummy(), :, :]


def test_crop_missing_channel(test_scans):
    """Make sure that missing channels are handled appropriately when cropping"""
    np.testing.assert_equal(
        test_scans["rb channels missing"][:, 0:2, 1:3].get_image("red"), np.zeros((2, 2))
    )


def test_scan_slicing_by_time():
    start = _FIRST_TIMESTAMP + 100
    num_frames = 10
    line_padding = 10
    dt = int(1e7)
    multi_frame = generate_scan(
        "test",
        np.random.randint(0, 10, size=(num_frames, 2, 2)),
        [1, 1],
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
    compare_frames([2, 3], multi_frame[start + int(2e9) : start + int(4e9)])
    compare_frames([2, 3], multi_frame[start + int(2e9) : start + int(4.8e9)])
    compare_frames([2, 3, 4], multi_frame[start + int(2e9) : start + int(4.81e9)])
    compare_frames([0, 1, 2, 3, 4], multi_frame[: (start + int(4.81e9))])
    compare_frames([5, 6, 7, 8, 9], multi_frame[(start + int(5e9)) :])
    compare_frames(
        [2, 3, 4], multi_frame[start + int(2e9) : start + int(4.81e9)][start : start + int(100e9)]
    )
    compare_frames(
        [3],
        multi_frame[start + int(2e9) : start + int(4.81e9)][start + int(3e9) : start + int(3.81e9)],
    )
