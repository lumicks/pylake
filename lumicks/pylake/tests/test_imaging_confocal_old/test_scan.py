import re

import numpy as np
import pytest
import matplotlib.pyplot as plt

from lumicks.pylake.adjustments import ColorAdjustment
from lumicks.pylake.detail.imaging_mixins import _FIRST_TIMESTAMP

from ..data.mock_confocal import generate_scan


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

    with pytest.raises(
        ValueError,
        match=(
            "channel must be 'red', 'green', 'blue' or a combination of 'r', 'g', "
            "and/or 'b', got 'gray'."
        ),
    ):
        scan.export_video("gray", "dummy.gif")  # Gray is not a color!


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
