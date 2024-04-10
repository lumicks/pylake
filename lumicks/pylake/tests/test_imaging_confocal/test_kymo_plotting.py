import re
from contextlib import nullcontext

import numpy as np
import pytest
import matplotlib.pyplot as plt

import lumicks.pylake as lk


def test_plotting(test_kymo):
    kymo, ref = test_kymo
    line_time = ref.timestamps.line_time_seconds
    n_lines = ref.metadata.lines_per_frame
    n_pixels = ref.metadata.pixels_per_line
    pixel_size = ref.metadata.pixelsize_um[0]

    plt.figure()
    kymo.plot(channel="red")

    np.testing.assert_allclose(
        np.sort(plt.xlim()),
        [-0.5 * line_time, (n_lines - 0.5) * line_time],
        atol=0.05,
    )

    image = plt.gca().get_images()[0]
    np.testing.assert_allclose(image.get_array(), ref.image[:, :, 0])
    np.testing.assert_allclose(
        image.get_extent(),
        [
            -0.5 * line_time,
            (n_lines - 0.5) * line_time,
            (n_pixels * pixel_size - (pixel_size / 2)),
            -(pixel_size / 2),
        ],
    )

    # test original kymo is labeled with microns and
    # that kymo calibrated with base pairs has appropriate label
    assert plt.gca().get_ylabel() == r"position (μm)"
    plt.close()

    kymo_bp = kymo.calibrate_to_kbp(10.000)
    kymo_bp.plot(channel="red")
    assert plt.gca().get_ylabel() == "position (kbp)"
    plt.close()


def test_deprecated_plotting(test_kymo):
    kymo, _ = test_kymo

    with pytest.raises(
        TypeError, match=re.escape("plot() takes from 1 to 2 positional arguments but 3 were given")
    ):
        ih = kymo.plot("red", None)
        np.testing.assert_allclose(ih.get_array(), kymo.get_image("red"))
        plt.close()

    # Test rejection of deprecated call with positional `axes` and double keyword assignment
    with pytest.raises(
        TypeError,
        match=re.escape(
            "plot() takes from 1 to 2 positional arguments but 3 positional"
            " arguments (and 1 keyword-only argument) were given"
        ),
    ):
        kymo.plot("rgb", None, axes=None)


def test_plotting_with_channels(kymo_h5_file):
    f = lk.File.from_h5py(kymo_h5_file)
    kymo = f.kymos["tester"]

    linetime = kymo.line_time_seconds
    ranges = kymo.line_timestamp_ranges()
    starts = np.vstack(ranges)[:, 0]
    ranges_sec = (starts - starts[0]) * 1e-9
    force_over_kymolines = f.force2x.downsampled_over(ranges)

    def plot_with_force():
        kymo.plot_with_force(force_channel="2x", color_channel="red")

    def plot_channels():
        kymo.plot_with_channels(
            force_over_kymolines,
            color_channel="red",
            labels="single",
            colors=[0, 0, 1],
            scale_bar=lk.ScaleBar(5.0, 5.0),
        )

    def plot_with_multiple_channels():
        kymo.plot_with_channels(
            [force_over_kymolines, force_over_kymolines],
            color_channel="red",
            labels=["f2x", "f2x"],
            colors=["red", [1.0, 0.0, 0.1]],
        )

    for plot_func in (plot_with_force, plot_channels, plot_with_multiple_channels):
        plot_func()
        plot_line = plt.gca().lines[0].get_ydata()
        np.testing.assert_allclose(plot_line[:2], 30)
        np.testing.assert_allclose(plot_line[2:], 10)
        np.testing.assert_allclose(np.sort(plt.ylim()), [10, 30])

        np.testing.assert_allclose(plt.gca().lines[0].get_xdata(), ranges_sec)
        np.testing.assert_allclose(plt.xlim(), [-(linetime / 2), ranges_sec[-1] + (linetime / 2)])


def test_plotting_with_channels_no_downsampling(kymo_h5_file):
    f = lk.File.from_h5py(kymo_h5_file)
    kymo = f.kymos["tester"]
    kymo.plot_with_channels(f.force2x, color_channel="red")
    plot_line = plt.gca().lines[0].get_ydata()
    np.testing.assert_allclose(plot_line, f.force2x[kymo.start : kymo.stop + 1].data)

    linetime = kymo.line_time_seconds
    ranges = np.vstack(kymo.line_timestamp_ranges())[:, 0]
    ranges_sec = (ranges - ranges[0]) * 1e-9
    np.testing.assert_allclose(plt.xlim(), [-(linetime / 2), ranges_sec[-1] + (linetime / 2)])


def test_plotting_labels(kymo_h5_file):
    f = lk.File.from_h5py(kymo_h5_file)
    kymo = f.kymos["tester"]

    def check_axes(labels, titles):
        axes = plt.gcf().get_axes()
        for label, title, ax in zip(labels, titles, axes[1:]):
            assert ax.get_title() == title
            assert ax.get_ylabel() == label

    kymo.plot_with_channels(f.force2x, color_channel="red")
    check_axes(["Force (pN)"], ["Force HF/Force 2x"])

    kymo.plot_with_channels(f.force2x, color_channel="red", labels="force")
    check_axes(["force"], ["Force HF/Force 2x"])

    kymo.plot_with_channels(f.force2x, color_channel="red", labels="force", title_vertical=True)
    check_axes(["force"], [""])

    kymo.plot_with_channels(f.force2x, color_channel="red", title_vertical=True)
    check_axes(["Force 2x\nForce (pN)"], [""])

    kymo.plot_with_channels(
        [f.force2x, f["Photon count"]["Red"]], color_channel="red", title_vertical=True
    )
    check_axes(["Force 2x\nForce (pN)", "Red"], [""] * 2)

    kymo.plot_with_channels([f.force2x, f["Photon count"]["Red"]], color_channel="red")
    check_axes(["Force (pN)", "y"], ["Force HF/Force 2x", "Photon count/Red"])


def test_plotting_with_channels_bad_args(kymo_h5_file):
    f = lk.File.from_h5py(kymo_h5_file)
    kymo = f.kymos["tester"]

    for channel_arg in (["potato"], "potato"):
        with pytest.raises(
            ValueError, match="channel is not a Slice or list of Slice objects. Got str instead."
        ):
            kymo.plot_with_channels(channel_arg, color_channel="red")

    with pytest.raises(
        ValueError,
        match="channel must be 'red', 'green', 'blue' or a combination of 'r', 'g', and/or 'b', got 'boo'.",
    ):
        kymo.plot_with_channels(f.force1x, color_channel="boo")

    for channels, labels, colors in (
        (f.force1x, ["too", "many"], None),
        ([f.force1x], ["too", "many"], None),
        ([f.force1x, f.force1x], ["too_few_labels"], None),
        ([f.force1x, f.force1x], ["too", "many", "labels"], None),
        ([f.force1x, f.force1x], None, ["Red", "g", "b"]),
        ([f.force1x, f.force1x], None, ["Red"]),
        ([f.force1x, f.force1x], None, [0, 1, 0]),  # Valid single color that is still a list!
    ):
        with pytest.raises(
            ValueError,
            match="needs to have the same length as the number of channels",
        ):
            kymo.plot_with_channels(channels, color_channel="rgb", labels=labels, colors=colors)


def test_regression_plot_with_force(kymo_h5_file):
    # Plot_with_force used to fail when the last line of a kymograph was incomplete. The reason for
    # this was that the last few timestamps on the last line had zero as their timestamp. This meant
    # it was trying to downsample a range from X to 0, which made the downsampler think that there
    # was no overlap between the kymograph and the force channel (as it checks the last timestamp
    # of the ranges to downsample to against the first one of the channel to downsample).
    f = lk.File.from_h5py(kymo_h5_file)

    # Kymo ends before last pixel is finished. All but the last timestamp are OK.
    kymo = f.kymos["tester"]
    pixel_ends = np.argwhere(kymo.infowave.data == 2).squeeze()

    kymo.stop = kymo.infowave.timestamps[pixel_ends[-1] - 1]
    np.testing.assert_equal(kymo.timestamps[-1, -1], 0)
    assert kymo.timestamps[-2, -1] != 0

    kymo.plot_with_force(force_channel="2x", color_channel="red")
    ds = f.force2x.downsampled_over(kymo.line_timestamp_ranges(include_dead_time=False))
    np.testing.assert_allclose(ds.data[:2], 30)
    np.testing.assert_allclose(ds.data[2:], 10)

    # Kymo ends on a partial last line. Multiple timestamps are zero now.
    kymo = f.kymos["tester"]
    kymo.stop = kymo.infowave.timestamps[pixel_ends[-3] - 1]
    np.testing.assert_equal(kymo.timestamps[-3:, -1], 0)
    assert kymo.timestamps[-4, -1] != 0

    kymo.plot_with_force(force_channel="2x", color_channel="red")
    ds = f.force2x.downsampled_over(kymo.line_timestamp_ranges(include_dead_time=False))
    np.testing.assert_allclose(ds.data[:2], 30)
    np.testing.assert_allclose(ds.data[2:], 10)


def test_plot_with_lf_force(kymo_h5_file):
    f = lk.File.from_h5py(kymo_h5_file)
    kymo = f.kymos["tester"]
    n_lines = kymo.get_image("red").shape[1]

    with pytest.warns(
        RuntimeWarning, match="Using downsampled force since high frequency force is unavailable."
    ):
        kymo.plot_with_force("2y", "red")
        np.testing.assert_allclose(plt.gca().lines[0].get_ydata(), np.arange(n_lines) + 1)

    with pytest.raises(RuntimeError, match="Desired force channel 1x not available in h5 file"):
        kymo.plot_with_force("1x", "red")


def test_plotting_with_histograms(test_kymo):
    def get_hist_data(axis):
        assert axis in ("position", "time")

        patches = plt.gca().patches
        widths = [p.get_width() for p in patches]
        heights = [p.get_height() for p in patches]

        counts, bin_widths = (widths, heights) if axis == "position" else (heights, widths)

        return counts, bin_widths

    kymo, ref = test_kymo

    def process_binning(px_per_bin, axis):
        assert axis in ("position", "time")

        image = ref.image[:, :, 0]
        if axis == "time":
            image = image.T
        n_pixels = image.shape[0]

        n_full_bins = n_pixels // px_per_bin
        leftover_pixels = n_pixels % px_per_bin

        counts = np.hstack(
            (
                image[: (xx := n_full_bins * px_per_bin)].reshape((n_full_bins, -1)).sum(axis=1),
                image[xx:].sum() if leftover_pixels else [],
            )
        )

        size = (
            ref.metadata.pixelsize_um[0] if axis == "position" else ref.timestamps.line_time_seconds
        )
        bin_widths = np.hstack(
            ([size * px_per_bin] * n_full_bins, size * leftover_pixels if leftover_pixels else [])
        )

        warning_msg = (
            (
                f"{n_pixels} pixels is not divisible by {px_per_bin}, final bin only "
                f"contains {leftover_pixels} pixels"
            )
            if leftover_pixels
            else ""
        )

        return counts, bin_widths, warning_msg

    for px_per_bin in range(1, ref.image.shape[0] + 1):
        ref_counts, ref_bin_widths, msg = process_binning(px_per_bin, "position")
        with pytest.warns(
            UserWarning,
            match=msg,
        ) if msg else nullcontext():
            kymo.plot_with_position_histogram(color_channel="red", pixels_per_bin=px_per_bin)
            counts, bin_widths = get_hist_data("position")

            np.testing.assert_allclose(bin_widths, ref_bin_widths)
            np.testing.assert_equal(counts, ref_counts)
            plt.close("all")

    for px_per_bin in range(1, ref.image.shape[1] + 1):
        ref_counts, ref_bin_widths, msg = process_binning(px_per_bin, "time")
        with pytest.warns(UserWarning, match=msg) if msg else nullcontext():
            kymo.plot_with_time_histogram(color_channel="red", pixels_per_bin=px_per_bin)
            counts, bin_widths = get_hist_data("time")

            np.testing.assert_allclose(bin_widths, ref_bin_widths)
            np.testing.assert_equal(counts, ref_counts)
            plt.close("all")

    with pytest.raises(ValueError, match="bin size is larger than the available pixels"):
        kymo.plot_with_position_histogram(color_channel="red", pixels_per_bin=6)

    with pytest.raises(ValueError, match="bin size is larger than the available pixels"):
        kymo.plot_with_time_histogram(color_channel="red", pixels_per_bin=12)
