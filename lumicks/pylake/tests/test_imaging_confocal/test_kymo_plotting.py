import re

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

    # todo: this is confusing even in the context of the old test, check on this
    # # The following assertion fails because of unequal line times in the test data. These
    # # unequal line times are not typical for BL data. Kymo nowadays assumes equal line times
    # # which is why the old version of this test fails.
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
    kymo, ref = test_kymo

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


def test_plotting_with_force(kymo_h5_file):
    f = lk.File.from_h5py(kymo_h5_file)
    kymo = f.kymos["tester"]

    linetime = kymo.line_time_seconds
    ranges = np.vstack(kymo.line_timestamp_ranges())[:, 0]
    ranges_sec = (ranges - ranges[0]) * 1e-9

    kymo.plot_with_force(force_channel="2x", color_channel="red")

    plot_line = plt.gca().lines[0].get_ydata()
    np.testing.assert_allclose(plot_line[:2], 30)
    np.testing.assert_allclose(plot_line[2:], 10)
    np.testing.assert_allclose(np.sort(plt.ylim()), [10, 30])

    np.testing.assert_allclose(plt.gca().lines[0].get_xdata(), ranges_sec)
    np.testing.assert_allclose(plt.xlim(), [-(linetime / 2), ranges_sec[-1] + (linetime / 2)])


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

    print(f.downsampled_force2y.timestamps)

    with pytest.warns(
        RuntimeWarning, match="Using downsampled force since high frequency force is unavailable."
    ):
        kymo.plot_with_force("2y", "red")
        np.testing.assert_allclose(plt.gca().lines[0].get_ydata(), np.arange(n_lines) + 1)

    with pytest.raises(RuntimeError, match="Desired force channel 1x not available in h5 file"):
        kymo.plot_with_force("1x", "red")


def test_plotting_with_histograms(test_kymo):
    import matplotlib as mpl

    mpl.use("macosx")
    import numpy as np
    import matplotlib.pyplot as plt

    def get_rectangle_data():
        widths = [p.get_width() for p in plt.gca().patches]
        heights = [p.get_height() for p in plt.gca().patches]
        return widths, heights

    kymo, ref = test_kymo
    image = ref.image[:, :, 0]
    n_pos, n_time = image.shape

    pixelsize = ref.metadata.pixelsize_um[0]
    linetime = ref.timestamps.line_time_seconds

    # position, no binning
    kymo.plot_with_position_histogram(color_channel="red", pixels_per_bin=1)
    counts = image.sum(axis=1)
    w, h = get_rectangle_data()

    np.testing.assert_allclose(h, pixelsize)
    np.testing.assert_equal(w, counts)
    plt.close("all")

    # time, no binning
    kymo.plot_with_time_histogram(color_channel="red", pixels_per_bin=1)
    w, h = get_rectangle_data()
    counts = image.sum(axis=0)
    np.testing.assert_allclose(w, linetime)
    np.testing.assert_equal(h, counts)
    plt.close("all")

    px_per_bin = 3
    n_full_bins = n_pos // px_per_bin
    with pytest.warns(
        UserWarning,
        match=(
            f"{n_pos} pixels is not divisible by {px_per_bin}, final bin only "
            f"contains {(leftover_pixels := n_pos % px_per_bin)} pixels"
        ),
    ):
        kymo.plot_with_position_histogram(color_channel="red", pixels_per_bin=3)
        w, h = get_rectangle_data()

        counts_full = (
            image[: (xx := n_full_bins * px_per_bin)].reshape((n_full_bins, -1)).sum(axis=1)
        )
        counts_partial = image[xx:].sum()
        counts = np.hstack((counts_full, counts_partial))

        bin_widths = np.hstack(
            ([pixelsize * px_per_bin] * n_full_bins, pixelsize * leftover_pixels)
        )

        np.testing.assert_allclose(h, bin_widths)
        np.testing.assert_equal(w, counts)
        plt.close("all")

    n_full_bins = n_time // px_per_bin
    with pytest.warns(
        UserWarning,
        match=(
            f"{n_time} pixels is not divisible by {px_per_bin}, final bin only "
            f"contains {(leftover_pixels := n_time % px_per_bin)} pixels"
        ),
    ):
        kymo.plot_with_time_histogram(color_channel="red", pixels_per_bin=3)
        w, h = get_rectangle_data()

        counts_full = (
            image[:, : (xx := n_full_bins * px_per_bin)]
            .reshape((-1, n_full_bins), order="F")
            .sum(axis=0)
        )
        counts_partial = image[:, xx:].sum()
        counts = np.hstack((counts_full, counts_partial))

        bin_widths = np.hstack(
            ([linetime * px_per_bin] * n_full_bins, linetime * leftover_pixels)
        )
        np.testing.assert_allclose(w, bin_widths)
        np.testing.assert_equal(h, counts)
        plt.close("all")

    with pytest.raises(ValueError, match="bin size is larger than the available pixels"):
        kymo.plot_with_position_histogram(color_channel="red", pixels_per_bin=6)

    with pytest.raises(ValueError, match="bin size is larger than the available pixels"):
        kymo.plot_with_time_histogram(color_channel="red", pixels_per_bin=12)
