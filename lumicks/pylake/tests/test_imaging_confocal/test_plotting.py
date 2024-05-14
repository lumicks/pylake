import re

import numpy as np
import pytest
import matplotlib.pyplot as plt

from lumicks.pylake.channel import Slice, Continuous
from lumicks.pylake.adjustments import ColorAdjustment


def test_kymo_plot_rgb_absolute_color_adjustment(test_kymo):
    """Tests whether we can set an absolute color range for the RGB plot."""
    kymo, ref = test_kymo

    fig = plt.figure()
    lb, ub = np.array([1, 2, 3]), np.array([2, 3, 4])
    kymo.plot(channel="rgb", adjustment=ColorAdjustment(lb, ub, mode="absolute"))
    image = plt.gca().get_images()[0]
    np.testing.assert_allclose(image.get_array(), np.clip((ref.image - lb) / (ub - lb), 0, 1))
    plt.close(fig)


def test_kymo_plot_rgb_percentile_color_adjustment(test_kymo):
    """Tests whether we can set a percentile color range for the RGB plot."""
    kymo, ref = test_kymo

    fig = plt.figure()
    lb, ub = np.array([10, 10, 10]), np.array([80, 80, 80])
    kymo.plot(channel="rgb", adjustment=ColorAdjustment(lb, ub, mode="percentile"))
    image = plt.gca().get_images()[0]
    bounds = np.array(
        [
            np.percentile(img, [mini, maxi])
            for img, mini, maxi in zip(np.moveaxis(ref.image, 2, 0), lb, ub)
        ]
    )
    lb, ub = (b for b in np.moveaxis(bounds, 1, 0))
    np.testing.assert_allclose(image.get_array(), np.clip((ref.image - lb) / (ub - lb), 0, 1))
    plt.close(fig)


def test_kymo_plot_single_channel_absolute_color_adjustment(test_kymo):
    """Tests whether we can set an absolute color range for a single channel plot."""
    kymo, ref = test_kymo

    lbs, ubs = np.array([1, 2, 3]), np.array([2, 3, 4])
    for lb, ub, channel in zip(lbs, ubs, channel_map := ("red", "green", "blue")):
        # Test whether setting RGB values and then sampling one of them works correctly.
        fig = plt.figure()
        kymo.plot(channel=channel, adjustment=ColorAdjustment(lbs, ubs, mode="absolute"))
        image = plt.gca().get_images()[0]
        np.testing.assert_allclose(image.get_array(), ref.image[:, :, channel_map.index(channel)])
        np.testing.assert_allclose(image.get_clim(), [lb, ub])
        plt.close(fig)

        # Test whether setting a single color works correctly (should use the same for R G and B).
        fig = plt.figure()
        kymo.plot(channel=channel, adjustment=ColorAdjustment(lb, ub, mode="absolute"))
        image = plt.gca().get_images()[0]
        np.testing.assert_allclose(image.get_array(), ref.image[:, :, channel_map.index(channel)])
        np.testing.assert_allclose(image.get_clim(), [lb, ub])
        plt.close(fig)


def test_scan_plotting(test_scans_multiframe):
    for scan, ref in test_scans_multiframe.values():
        pixelsize = ref.metadata.pixelsize_um
        n_pixels = ref.metadata.num_pixels
        ref_extent = [0, pixelsize[0] * n_pixels[0], pixelsize[1] * n_pixels[1], 0]

        scan.plot(channel="rgb")
        image = plt.gca().get_images()[0]
        np.testing.assert_allclose(image.get_array(), ref.image[0] / np.max(ref.image[0]))
        np.testing.assert_allclose(image.get_extent(), ref_extent)
        plt.close()

        scan.plot(channel="blue")
        image = plt.gca().get_images()[0]
        np.testing.assert_allclose(image.get_array(), ref.image[0, :, :, 2])
        np.testing.assert_allclose(image.get_extent(), ref_extent)
        plt.close()

    # test invalid inidices
    with pytest.raises(IndexError):
        scan.plot(channel="rgb", frame=ref.metadata.number_of_frames + 2)
    with pytest.raises(IndexError, match="negative indexing is not supported."):
        scan.plot(channel="rgb", frame=-1)


def test_deprecated_plotting(test_scans_multiframe):
    scan, ref = test_scans_multiframe["fast Y slow X multiframe"]
    with pytest.raises(
        TypeError, match=re.escape("plot() takes from 1 to 2 positional arguments but 3 were given")
    ):
        ih = scan.plot("blue", None)
        np.testing.assert_allclose(ih.get_array(), ref.image[0, :, :, 2])
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


def test_scan_plot_rgb_absolute_color_adjustment(test_scans):
    """Tests whether we can set an absolute color range for an RGB plot."""
    scan, ref = test_scans["fast Y slow X"]

    fig = plt.figure()
    lb, ub = np.array([1, 2, 3]), np.array([2, 3, 4])
    scan.plot(channel="rgb", adjustment=ColorAdjustment(lb, ub, mode="absolute"))
    image = plt.gca().get_images()[0]
    np.testing.assert_allclose(image.get_array(), np.clip((ref.image - lb) / (ub - lb), 0, 1))
    plt.close(fig)


def test_scan_plot_single_channel_absolute_color_adjustment(test_scans):
    """Tests whether we can set an absolute color range for a single channel plot."""
    scan, ref = test_scans["fast Y slow X"]

    lbs, ubs = np.array([1, 2, 3]), np.array([2, 3, 4])
    for lb, ub, channel in zip(lbs, ubs, channel_map := ("red", "green", "blue")):
        # Test whether setting RGB values and then sampling one of them works correctly.
        fig = plt.figure()
        scan.plot(channel=channel, adjustment=ColorAdjustment(lbs, ubs, mode="absolute"))
        image = plt.gca().get_images()[0]
        np.testing.assert_allclose(image.get_array(), ref.image[:, :, channel_map.index(channel)])
        np.testing.assert_allclose(image.get_clim(), [lb, ub])
        plt.close(fig)

        # Test whether setting a single color works correctly (should use the same for R G and B).
        fig = plt.figure()
        scan.plot(channel=channel, adjustment=ColorAdjustment(lb, ub, mode="absolute"))
        image = plt.gca().get_images()[0]
        np.testing.assert_allclose(image.get_array(), ref.image[:, :, channel_map.index(channel)])
        np.testing.assert_allclose(image.get_clim(), [lb, ub])
        plt.close(fig)


def test_plot_rgb_percentile_color_adjustment(test_scans_multiframe):
    """Tests whether we can set a percentile color range for an RGB plot."""
    scan, ref = test_scans_multiframe["fast Y slow X multiframe"]

    fig = plt.figure()
    lb, ub = np.array([1, 5, 10]), np.array([80, 90, 80])
    scan.plot(channel="rgb", adjustment=ColorAdjustment(lb, ub, mode="percentile"))
    bounds = np.array(
        [
            np.percentile(img, [mini, maxi])
            for img, mini, maxi in zip(np.moveaxis(ref.image[0], 2, 0), lb, ub)
        ]
    )
    lb, ub = (b for b in np.moveaxis(bounds, 1, 0))
    image = plt.gca().get_images()[0]
    np.testing.assert_allclose(image.get_array(), np.clip((ref.image[0] - lb) / (ub - lb), 0, 1))
    plt.close(fig)


def test_plot_single_channel_percentile_color_adjustment(test_scans_multiframe):
    """Tests whether we can set a percentile color range for separate channel plots."""
    scan, ref = test_scans_multiframe["fast Y slow X multiframe"]

    lbs, ubs = np.array([1, 5, 10]), np.array([80, 90, 80])
    for lb, ub, channel in zip(lbs, ubs, channel_map := ("red", "green", "blue")):
        # We need to test both specifying all bounds at once and specifying just one bound
        for used_lb, used_ub in ((lb, ub), (lbs, ubs)):
            fig = plt.figure()
            scan.plot(
                channel=channel, adjustment=ColorAdjustment(used_lb, used_ub, mode="percentile")
            )
            ref_image = ref.image[0, :, :, channel_map.index(channel)]
            lb_abs, ub_abs = np.percentile(ref_image, [lb, ub])
            image = plt.gca().get_images()[0]
            np.testing.assert_allclose(image.get_array(), ref_image)
            np.testing.assert_allclose(image.get_clim(), [lb_abs, ub_abs])
            plt.close(fig)


@pytest.mark.parametrize(
    "frame, vertical, channel, downsample",
    [
        (0, False, "red", True),
        (0, False, "blue", True),
        (0, True, "red", True),
        (1, True, "red", True),
        (0, False, "red", False),
    ],
)
def test_scan_plot_correlated(test_scans_multiframe, frame, vertical, channel, downsample):
    import matplotlib as mpl

    scan, ref = test_scans_multiframe["fast Y slow X multiframe"]
    corr_data = Slice(
        Continuous(np.arange(1000), scan.start, int(1e9)), labels={"title": "title", "y": "y"}
    )

    image_axis = 0 if vertical else 1
    scan.plot_correlated(
        channel=channel,
        channel_slice=corr_data,
        frame=frame,
        vertical=vertical,
        downsample_to_frames=downsample,
    )
    axes = plt.gcf().get_axes()
    imgs = [obj for obj in axes[image_axis].get_children() if isinstance(obj, mpl.image.AxesImage)]

    assert len(imgs) == 1
    np.testing.assert_allclose(imgs[0].get_array(), scan[frame].get_image(channel))
    assert axes[1 - image_axis].get_xlabel() == "Time [s]"
    assert axes[image_axis].get_title() == f"[frame {frame + 1} / 10]"

    # Fetch raw data
    lines = [o for o in axes[1 - image_axis].get_children() if isinstance(o, mpl.lines.Line2D)]
    ref_time_vector, y = lines[0].get_data()

    ts_ranges = scan.frame_timestamp_ranges()
    if downsample:
        ds = corr_data.downsampled_over(ts_ranges)
        # Last sample is double because of step plot
        np.testing.assert_allclose(np.hstack((ds.data, ds.data[-1])), y)
        # Last frame does _not_ include dead time
        seconds = (np.asarray(ts_ranges) - ts_ranges[0][0]) / 1e9
        time_vector = np.hstack((seconds[:, 0], seconds[-1, 1]))
        np.testing.assert_allclose(time_vector, ref_time_vector)
    else:
        slc = corr_data[ts_ranges[0][0] : ts_ranges[-1][-1]]
        np.testing.assert_allclose(np.hstack((slc.data, slc.data[-1])), y)
        time_vector = np.hstack((slc.seconds, slc.seconds[-1] + slc.seconds[-1] - slc.seconds[-2]))
        np.testing.assert_allclose(time_vector, ref_time_vector)
