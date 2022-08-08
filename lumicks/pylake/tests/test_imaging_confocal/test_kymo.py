import re
import numpy as np
from lumicks import pylake
import pytest
from lumicks.pylake.channel import Slice, TimeSeries, empty_slice
from lumicks.pylake.kymo import EmptyKymo
from lumicks.pylake.adjustments import ColorAdjustment
import matplotlib.pyplot as plt
from matplotlib.testing.decorators import cleanup
from ..data.mock_confocal import generate_kymo


def with_offset(t, start_time=1592916040906356300):
    return np.array(t, dtype=np.int64) + start_time


def test_kymo_properties(test_kymos):
    kymo = test_kymos["Kymo1"]

    # fmt: off
    reference_timestamps = np.array([[2.006250e+10, 2.109375e+10, 2.206250e+10, 2.309375e+10],
                                    [2.025000e+10, 2.128125e+10, 2.225000e+10, 2.328125e+10],
                                    [2.043750e+10, 2.146875e+10, 2.243750e+10, 2.346875e+10],
                                    [2.062500e+10, 2.165625e+10, 2.262500e+10, 2.365625e+10],
                                    [2.084375e+10, 2.187500e+10, 2.284375e+10, 2.387500e+10]], np.int64)
    # fmt: on

    assert repr(kymo) == "Kymo(pixels=5)"
    assert kymo.pixels_per_line == 5
    assert len(kymo.infowave) == 64
    assert kymo.shape == (5, 4, 3)
    assert kymo.get_image("rgb").shape == (5, 4, 3)
    assert kymo.get_image("red").shape == (5, 4)
    assert kymo.get_image("blue").shape == (5, 4)
    assert kymo.get_image("green").shape == (5, 4)
    np.testing.assert_allclose(kymo.timestamps, reference_timestamps)
    assert kymo.fast_axis == "X"
    np.testing.assert_allclose(kymo.pixelsize_um, 10/1000)
    np.testing.assert_allclose(kymo.line_time_seconds, 1.03125)
    np.testing.assert_allclose(kymo.center_point_um["x"], 58.075877109272604)
    np.testing.assert_allclose(kymo.center_point_um["y"], 31.978375270573267)
    np.testing.assert_allclose(kymo.center_point_um["z"], 0)
    np.testing.assert_allclose(kymo.size_um, [0.050])
    np.testing.assert_allclose(kymo.pixel_time_seconds, 0.1875)

    with pytest.deprecated_call():
        assert kymo.rgb_image.shape == (5, 4, 3)
    with pytest.deprecated_call():
        assert kymo.red_image.shape == (5, 4)
    with pytest.deprecated_call():
        assert kymo.blue_image.shape == (5, 4)
    with pytest.deprecated_call():
        assert kymo.green_image.shape == (5, 4)


def test_kymo_slicing(test_kymos):
    kymo = test_kymos["Kymo1"]
    kymo_reference = np.transpose([[2, 0, 0, 0, 2], [0, 0, 0, 0, 0], [1, 0, 0, 0, 1], [0, 1, 1, 1, 0]])

    assert kymo.get_image("red").shape == (5, 4)
    assert kymo.shape == (5, 4, 3)
    np.testing.assert_allclose(kymo.get_image("red").data, kymo_reference)

    sliced = kymo[:]
    assert sliced.get_image("red").shape == (5, 4)
    np.testing.assert_allclose(sliced.get_image("red").data, kymo_reference)

    sliced = kymo["1s":]
    assert sliced.get_image("red").shape == (5, 3)
    assert sliced.shape == (5, 3, 3)
    np.testing.assert_allclose(sliced.get_image("red").data, kymo_reference[:, 1:])

    sliced = kymo["0s":]
    assert sliced.get_image("red").shape == (5, 4)
    np.testing.assert_allclose(sliced.get_image("red").data, kymo_reference)

    sliced = kymo["0s":"2s"]
    assert sliced.get_image("red").shape == (5, 2)
    assert sliced.shape == (5, 2, 3)
    np.testing.assert_allclose(sliced.get_image("red").data, kymo_reference[:, :2])

    sliced = kymo["0s":"-1s"]
    assert sliced.get_image("red").shape == (5, 3)
    np.testing.assert_allclose(sliced.get_image("red").data, kymo_reference[:, :-1])

    sliced = kymo["0s":"-2s"]
    assert sliced.get_image("red").shape == (5, 2)
    np.testing.assert_allclose(sliced.get_image("red").data, kymo_reference[:, :-2])

    sliced = kymo["0s":"3s"]
    assert sliced.get_image("red").shape == (5, 3)
    np.testing.assert_allclose(sliced.get_image("red").data, kymo_reference[:, :3])

    sliced = kymo["1s":"2s"]
    assert sliced.get_image("red").shape == (5, 1)
    assert sliced.shape == (5, 1, 3)
    np.testing.assert_allclose(sliced.get_image("red").data, kymo_reference[:, 1:2])

    sliced = kymo["0s":"10s"]
    assert sliced.get_image("red").shape == (5, 4)
    assert sliced.shape == (5, 4, 3)
    np.testing.assert_allclose(sliced.get_image("red").data, kymo_reference[:, 0:10])

    with pytest.raises(IndexError):
        kymo["0s"]

    with pytest.raises(IndexError):
        kymo["0s":"10s":"1s"]

    empty_kymograph = kymo["3s":"2s"]
    assert isinstance(empty_kymograph, EmptyKymo)

    empty_kymograph = kymo["5s":]
    assert isinstance(empty_kymograph, EmptyKymo)

    with pytest.raises(RuntimeError):
        empty_kymograph.timestamps

    with pytest.raises(RuntimeError):
        empty_kymograph.export_tiff("test")

    with pytest.raises(RuntimeError):
        empty_kymograph.plot_rgb()

    assert empty_kymograph.get_image("red").shape == (5, 0)
    assert empty_kymograph.infowave.data.size == 0
    assert empty_kymograph.shape == (5, 0, 3)
    assert empty_kymograph.pixels_per_line == 5
    assert empty_kymograph.get_image("red").size == 0
    assert empty_kymograph.get_image("rgb").size == 0


def test_damaged_kymo(test_kymos):
    # Assume the user incorrectly exported only a partial Kymo
    kymo = test_kymos["truncated_kymo"]
    kymo_reference = np.transpose([[2, 0, 0, 0, 2], [0, 0, 0, 0, 0], [1, 0, 0, 0, 1], [0, 1, 1, 1, 0]])

    with pytest.warns(RuntimeWarning):
        assert kymo.get_image("red").shape == (5, 3)
    np.testing.assert_allclose(kymo.get_image("red").data, kymo_reference[:, 1:])


@cleanup
def test_plotting(test_kymos):
    kymo = test_kymos["Kymo1"]

    plt.figure()
    kymo.plot(channel="red")
    # # The following assertion fails because of unequal line times in the test data. These
    # # unequal line times are not typical for BL data. Kymo nowadays assumes equal line times
    # # which is why the old version of this test fails.
    # np.testing.assert_allclose(np.sort(plt.xlim()), [-0.5, 3.5], atol=0.05)

    image = plt.gca().get_images()[0]
    np.testing.assert_allclose(image.get_array(), kymo.get_image("red"))
    np.testing.assert_allclose(image.get_extent(), [-0.515625, 3.609375, 0.045, -0.005])

    # test original kymo is labeled with microns and
    # that kymo calibrated with base pairs has appropriate label
    assert plt.gca().get_ylabel() == r"position ($\mu$m)"
    plt.close()

    kymo_bp = kymo.calibrate_to_kbp(10.000)
    kymo_bp.plot(channel="red")
    assert plt.gca().get_ylabel() == "position (kbp)"
    plt.close()


@cleanup
def test_deprecated_plotting(test_kymos):
    kymo = test_kymos["Kymo1"]
    with pytest.deprecated_call():
        kymo.plot_red()
    with pytest.deprecated_call():
        kymo.plot_green()
    with pytest.deprecated_call():
        kymo.plot_blue()
    with pytest.deprecated_call():
        kymo.plot_rgb()


def test_line_timestamp_ranges(test_kymos):
    kymo = test_kymos["Kymo1"]

    expected_ranges = (
        [
            (20000000000, 21000000000),
            (21062500000, 22000000000),
            (22000000000, 23000000000),
            (23062500000, 24000000000)
        ],
        [
            (20000000000, 21062500000),
            (21062500000, 22125000000),
            (22000000000, 23062500000),
            (23062500000, 24125000000)]
    )
    expected_iw_chunks = (
        [
            [1, 0, 2, 1, 0, 2, 1, 0, 2, 1, 0, 2, 1, 0, 0, 2],
            [1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 0, 2],
            [1, 0, 2, 1, 0, 2, 1, 0, 2, 1, 0, 2, 1, 0, 0, 2],
            [1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 0, 2]
        ],
        [
            [1, 0, 2, 1, 0, 2, 1, 0, 2, 1, 0, 2, 1, 0, 0, 2, 0],
            [1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 0, 2, 1, 0],
            [1, 0, 2, 1, 0, 2, 1, 0, 2, 1, 0, 2, 1, 0, 0, 2, 0],
            [1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 0, 2]
        ]
    )

    for include, ref_ranges, ref_iw_chunks in zip(
        (False, True),
        expected_ranges,
        expected_iw_chunks
    ):
        ranges = kymo.line_timestamp_ranges(include_dead_time=include)
        np.testing.assert_equal(ranges, ref_ranges)

        with pytest.deprecated_call():
            np.testing.assert_equal(kymo.line_timestamp_ranges(not include), ref_ranges)

        with pytest.deprecated_call():
            np.testing.assert_equal(kymo.line_timestamp_ranges(exclude=not include), ref_ranges)

        iw_chunks = [kymo.infowave[slice(*rng)].data for rng in ranges]
        np.testing.assert_equal(iw_chunks, ref_iw_chunks)

    with pytest.raises(
        ValueError, match="Do not specify both exclude and include_dead_time parameters"
    ):
        kymo.line_timestamp_ranges(False, include_dead_time=False)


def test_plotting_with_force_downsampling(kymo_h5_file):
    f = pylake.File.from_h5py(kymo_h5_file)
    kymo = f.kymos["Kymo1"]
    ranges = kymo.line_timestamp_ranges(include_dead_time=False)

    # Check timestamps for downsampled channel
    # Note that if the kymo would have the same samples per pixel, a simple:
    #    np.testing.assert_allclose(np.mean(kymo.timestamps, axis=0)[:-1], ds.timestamps[:-1])
    # would have sufficed. However, in this case we need the following solution:
    ds = f.force2x.downsampled_over(ranges)
    min_ts, max_ts = (
        reduce(kymo._timestamps("timestamps", reduce), axis=0) for reduce in (np.min, np.max)
    )
    target_timestamps = np.array(
        [
            np.mean(kymo.infowave[int(start) : int(stop) + 1].timestamps)
            for start, stop in zip(min_ts, max_ts)
        ]
    )
    np.testing.assert_allclose(ds.timestamps, target_timestamps)
    np.testing.assert_allclose(ds.data, [30, 30, 10, 10])


@cleanup
def test_plotting_with_force(kymo_h5_file):
    f = pylake.File.from_h5py(kymo_h5_file)
    kymo = f.kymos["Kymo1"]

    kymo.plot_with_force(force_channel="2x", color_channel="red")
    np.testing.assert_allclose(plt.gca().lines[0].get_ydata(), [30, 30, 10, 10])

    # The following assertion fails because of unequal line times in the test data. These
    # unequal line times are not typical for BL data. Kymo nowadays assumes equal line times
    # which is why the old version of this test fails.
    # np.testing.assert_allclose(np.sort(plt.xlim()), [-0.5, 3.5], atol=0.05)
    np.testing.assert_allclose(plt.xlim(), [-0.515625, 3.609375])
    np.testing.assert_allclose(np.sort(plt.ylim()), [10, 30])


@cleanup
def test_downsample_channel_downsampled_kymo(kymo_h5_file):
    f = pylake.File.from_h5py(kymo_h5_file)
    kymo = f.kymos["Kymo1"]
    kymo_ds = kymo.downsampled_by(position_factor=2)

    ds = f.force2x.downsampled_over(kymo_ds.line_timestamp_ranges(include_dead_time=False))
    np.testing.assert_allclose(ds.data, [30, 30, 10, 10])

    # Downsampling by a factor of two in position means that the last pixel will be dropped
    # from this kymo when downsampling (as it is 5 pixels wide). This is why the before last
    # sample is taken when determining the maxima.
    mins = kymo._timestamp_factory(kymo, np.min)[0, :]
    maxs = kymo._timestamp_factory(kymo, np.max)[-2, :]
    np.testing.assert_allclose(ds.timestamps, (maxs + mins) / 2)

    # Downsampling by a factor of five in position means no pixel will be dropped.
    kymo_ds = kymo.downsampled_by(position_factor=5)
    ds = f.force2x.downsampled_over(kymo_ds.line_timestamp_ranges(include_dead_time=False))
    mins = kymo._timestamp_factory(kymo, np.min)[0, :]
    maxs = kymo._timestamp_factory(kymo, np.max)[-1, :]
    np.testing.assert_allclose(ds.timestamps, (maxs + mins) / 2)

    # Down-sampling by time should invalidate plot_with_force as it would correspond to
    # non-contiguous sampling
    with pytest.raises(NotImplementedError, match="Per-pixel timestamps are no longer available"):
        kymo.downsampled_by(time_factor=2).plot_with_force("1x", "red")


@cleanup
def test_regression_plot_with_force(kymo_h5_file):
    # Plot_with_force used to fail when the last line of a kymograph was incomplete. The reason for
    # this was that the last few timestamps on the last line had zero as their timestamp. This meant
    # it was trying to downsample a range from X to 0, which made the downsampler think that there
    # was no overlap between the kymograph and the force channel (as it checks the last timestamp
    # of the ranges to downsample to against the first one of the channel to downsample).
    f = pylake.File.from_h5py(kymo_h5_file)

    # Kymo ends before last pixel is finished. All but the last timestamps are OK.
    kymo = f.kymos["Kymo1"]
    kymo.stop = int(kymo.stop - 2 * 1e9 / 16)
    kymo.plot_with_force(force_channel="2x", color_channel="red")
    ds = f.force2x.downsampled_over(kymo.line_timestamp_ranges(include_dead_time=False))
    np.testing.assert_allclose(ds.data, [30, 30, 10, 10])

    # Kymo ends on a partial last line. Multiple timestamps are zero now.
    kymo = f.kymos["Kymo1"]
    kymo.stop = int(kymo.stop - 10 * 1e9 / 16)
    kymo.plot_with_force(force_channel="2x", color_channel="red")
    ds = f.force2x.downsampled_over(kymo.line_timestamp_ranges(include_dead_time=False))
    np.testing.assert_allclose(ds.data, [30, 30, 10, 10])


@cleanup
def test_plotting_with_histograms(test_kymos):
    def get_rectangle_data():
        widths = [p.get_width() for p in plt.gca().patches]
        heights = [p.get_height() for p in plt.gca().patches]
        return widths, heights

    kymo = test_kymos["Kymo1"]

    kymo.plot_with_position_histogram(color_channel="red", pixels_per_bin=1)
    w, h = get_rectangle_data()
    np.testing.assert_allclose(h, 0.01)
    assert np.all(np.equal(w, [3, 1, 1, 1, 3]))
    np.testing.assert_allclose(np.sort(plt.xlim()), [0, 3], atol=0.05)

    kymo.plot_with_time_histogram(color_channel="red", pixels_per_bin=1)
    w, h = get_rectangle_data()
    np.testing.assert_allclose(w, 1.03, atol=0.002)
    assert np.all(np.equal(h, [4, 0, 2, 3]))
    np.testing.assert_allclose(np.sort(plt.ylim()), [0, 4], atol=0.05)

    with pytest.warns(UserWarning):
        kymo.plot_with_position_histogram(color_channel="red", pixels_per_bin=3)
        w, h = get_rectangle_data()
        np.testing.assert_allclose(h, [0.03, 0.02])
        assert np.all(np.equal(w, [5, 4]))
        np.testing.assert_allclose(np.sort(plt.xlim()), [0, 5], atol=0.05)

    with pytest.warns(UserWarning):
        kymo.plot_with_time_histogram(color_channel="red", pixels_per_bin=3)
        w, h = get_rectangle_data()
        np.testing.assert_allclose(w, [3.09, 1.03], atol=0.02)
        assert np.all(np.equal(h, [6, 3]))
        np.testing.assert_allclose(np.sort(plt.ylim()), [0, 6], atol=0.05)

    with pytest.raises(ValueError):
        kymo.plot_with_position_histogram(color_channel="red", pixels_per_bin=6)

    with pytest.raises(ValueError):
        kymo.plot_with_time_histogram(color_channel="red", pixels_per_bin=6)


def test_save_tiff(tmpdir_factory, test_kymos):
    from os import stat

    tmpdir = tmpdir_factory.mktemp("pylake")

    kymo = test_kymos["Kymo1"]
    kymo.export_tiff(f"{tmpdir}/kymo1.tiff")
    assert stat(f"{tmpdir}/kymo1.tiff").st_size > 0

    with pytest.warns(DeprecationWarning, match="This method has been renamed to `export_tiff`"):
        kymo.save_tiff(f"{tmpdir}/kymo2.tiff")
        assert stat(f"{tmpdir}/kymo2.tiff").st_size > 0


def test_downsampled_kymo():
    image = np.array(
        [
            [0, 12, 0, 12, 0, 6, 0],
            [0, 0, 0, 0, 0, 6, 0],
            [12, 0, 0, 0, 12, 6, 0],
            [0, 12, 12, 12, 0, 6, 0],
        ],
        dtype=np.uint8
    )

    kymo = generate_kymo(
        "Mock",
        image,
        pixel_size_nm=1,
        start=with_offset(0),
        dt=7,
        samples_per_pixel=5,
        line_padding=2
    )

    kymo_ds = kymo.downsampled_by(time_factor=2)
    ds = np.array(
        [
            [12, 12, 6],
            [0, 0, 6],
            [12, 0, 18],
            [12, 24, 6],
        ],
        dtype=np.uint8
    )

    assert kymo_ds.name == "Mock"
    np.testing.assert_allclose(kymo_ds.get_image("red"), ds)
    np.testing.assert_allclose(kymo_ds.shape, kymo_ds.get_image("rgb").shape)
    np.testing.assert_allclose(kymo_ds.start, with_offset(0))
    np.testing.assert_allclose(kymo_ds.pixelsize_um, 1 / 1000)
    np.testing.assert_allclose(kymo_ds.pixelsize, 1 / 1000)
    np.testing.assert_allclose(kymo_ds.line_time_seconds, 2 * 7 * (5 * 4 + 2 + 2) / 1e9)

    with pytest.raises(
            NotImplementedError,
            match=re.escape("Per-pixel timestamps are no longer available after downsampling"),
    ):
        kymo_ds.timestamps

    # Verify that we can pass a different reduce function
    np.testing.assert_allclose(kymo.downsampled_by(time_factor=2, reduce=np.mean).get_image("red"), ds / 2)

    with pytest.raises(
            NotImplementedError,
            match="Per-pixel timestamps are no longer available after downsampling",
    ):
        kymo_ds.pixel_time_seconds


def test_downsampled_kymo_position():
    """Test downsampling over the spatial axis"""
    image = np.array(
        [
            [0, 12, 0, 12, 0, 6, 0],
            [0, 0, 0, 0, 0, 6, 0],
            [12, 0, 0, 0, 12, 6, 0],
            [0, 12, 12, 12, 0, 6, 0],
            [0, 12, 12, 12, 0, 6, 0],
        ],
        dtype=np.uint8
    )

    kymo = generate_kymo(
        "Mock",
        image,
        pixel_size_nm=1,
        start=with_offset(100),
        dt=5,
        samples_per_pixel=5,
        line_padding=2
    )

    kymo_ds = kymo.downsampled_by(position_factor=2)
    ds = np.array([[0, 12, 0, 12, 0, 12, 0], [12, 12, 12, 12, 12, 12, 0]], dtype=np.uint8)
    ds_ts = with_offset(
        [
            [132.5,  277.5,  422.5,  567.5,  712.5,  857.5, 1002.5],
            [182.5,  327.5,  472.5,  617.5,  762.5,  907.5, 1052.5],
        ],
    )

    assert kymo_ds.name == "Mock"
    np.testing.assert_allclose(kymo_ds.get_image("red"), ds)
    np.testing.assert_equal(kymo_ds.timestamps, ds_ts)
    np.testing.assert_allclose(kymo_ds.start, with_offset(100))
    np.testing.assert_allclose(kymo_ds.pixelsize_um, 2 / 1000)
    np.testing.assert_allclose(kymo_ds.pixelsize, 2 / 1000)
    np.testing.assert_allclose(kymo_ds.line_time_seconds, kymo.line_time_seconds)

    # We lost one line while downsampling
    np.testing.assert_allclose(kymo_ds.size_um[0], kymo.size_um[0] - kymo.pixelsize_um[0])

    np.testing.assert_allclose(kymo_ds.pixel_time_seconds, kymo.pixel_time_seconds * 2)


def test_downsampled_kymo_both_axes():
    image = np.array(
        [
            [0, 12, 0, 12, 0, 6, 0],
            [0, 0, 0, 0, 0, 6, 0],
            [12, 0, 0, 0, 12, 6, 0],
            [0, 12, 12, 12, 0, 6, 0],
            [0, 12, 12, 12, 0, 6, 0],
        ],
        dtype=np.uint8
    )

    kymo = generate_kymo(
        "Mock", image, pixel_size_nm=1, start=100, dt=5, samples_per_pixel=5, line_padding=2
    )
    ds = np.array([[12, 12, 12], [24, 24, 24]], dtype=np.uint8)

    downsampled_kymos = [
        kymo.downsampled_by(time_factor=2, position_factor=2),
        # Test whether sequential downsampling works out correctly as well
        kymo.downsampled_by(position_factor=2).downsampled_by(time_factor=2),
        kymo.downsampled_by(time_factor=2).downsampled_by(position_factor=2)
    ]

    for kymo_ds in downsampled_kymos:
        assert kymo_ds.name == "Mock"
        np.testing.assert_allclose(kymo_ds.get_image("red"), ds)
        np.testing.assert_allclose(kymo_ds.get_image("green"), np.zeros(ds.shape))  # missing
        np.testing.assert_allclose(kymo_ds.start, 100)
        np.testing.assert_allclose(kymo_ds.pixelsize_um, 2 / 1000)
        np.testing.assert_allclose(kymo_ds.pixelsize, 2 / 1000)
        np.testing.assert_allclose(kymo_ds.line_time_seconds, 2 * 5 * (5 * 5 + 2 + 2) / 1e9)
        with pytest.raises(
                NotImplementedError,
                match=re.escape("Per-pixel timestamps are no longer available after downsampling"),
        ):
            kymo_ds.timestamps


def test_side_no_side_effects_downsampling():
    """Test whether downsampling doesn't have side effects on the original kymo"""
    image = np.array(
        [
            [0, 12, 0, 12, 0, 6, 0],
            [0, 0, 0, 0, 0, 6, 0],
            [12, 0, 0, 0, 12, 6, 0],
            [0, 12, 12, 12, 0, 6, 0],
            [0, 12, 12, 12, 0, 6, 0],
        ],
        dtype=np.uint8
    )

    kymo = generate_kymo(
        "Mock",
        image,
        pixel_size_nm=1,
        start=with_offset(100),
        dt=5,
        samples_per_pixel=5,
        line_padding=2
    )
    timestamps = kymo.timestamps.copy()
    downsampled_kymos = kymo.downsampled_by(time_factor=2, position_factor=2)

    np.testing.assert_allclose(kymo.get_image("red"), image)
    np.testing.assert_allclose(kymo.start, with_offset(100))
    np.testing.assert_allclose(kymo.pixelsize_um, 1 / 1000)
    np.testing.assert_allclose(kymo.line_time_seconds, 5 * (5 * 5 + 2 + 2) / 1e9)
    np.testing.assert_equal(kymo.timestamps, timestamps)


def test_downsampled_slice():
    """There was a regression bug that if a Kymo was downsampled and then sliced, it would undo the
    downsampling. For now, we just flag it as not implemented behaviour."""
    kymo = generate_kymo(
        "Mock",
        image=np.array([[2, 2], [2, 2]], dtype=np.uint8),
        pixel_size_nm=1,
        start=100,
        dt=7,
        samples_per_pixel=5,
        line_padding=2,
    )

    with pytest.raises(NotImplementedError):
        kymo.downsampled_by(time_factor=2)["1s":"2s"]


def test_kymo_crop():
    """Test basic cropping functionality"""
    image = np.array(
        [
            [0, 12, 0, 12, 0, 6, 0],
            [0, 0, 0, 0, 0, 6, 0],
            [12, 0, 0, 0, 12, 6, 0],
            [0, 12, 12, 12, 0, 6, 0],
            [0, 12, 12, 12, 0, 6, 0],
        ],
        dtype=np.uint8
    )

    # Test basic functionality
    pixel_size_nm = 2
    kymo = generate_kymo(
        "Mock",
        image,
        pixel_size_nm=pixel_size_nm,
        start=with_offset(100),
        dt=5,
        samples_per_pixel=5,
        line_padding=2
    )
    cropped = kymo.crop_by_distance(4e-3, 8e-3)
    ref_img = np.array([
        [12.0,  0.0,  0.0,  0.0, 12.0,  6.0,  0.0],
        [0.0, 12.0, 12.0, 12.0,  0.0,  6.0,  0.0]
    ])
    np.testing.assert_allclose(cropped.get_image("red"), ref_img)
    np.testing.assert_allclose(cropped.get_image("rgb")[:, :, 0], ref_img)
    np.testing.assert_allclose(cropped.get_image("rgb")[:, :, 1], np.zeros(ref_img.shape))
    np.testing.assert_allclose(cropped.get_image("green"), np.zeros(ref_img.shape))  # missing
    np.testing.assert_equal(
        cropped.timestamps,
        with_offset([[170, 315, 460, 605, 750, 895, 1040], [195, 340, 485, 630, 775, 920, 1065]]),
    )
    assert cropped.timestamps.dtype == np.int64
    np.testing.assert_allclose(cropped.pixelsize_um, kymo.pixelsize_um)
    np.testing.assert_allclose(cropped.line_time_seconds, kymo.line_time_seconds)
    np.testing.assert_allclose(cropped.pixels_per_line, 2)
    np.testing.assert_allclose(cropped._position_offset, 4e-3)

    with pytest.raises(ValueError, match="Cropping by negative positions not allowed"):
        kymo.crop_by_distance(-4e3, 1e3)

    with pytest.raises(ValueError, match="Cropping by negative positions not allowed"):
        kymo.crop_by_distance(1e3, -4e3)

    with pytest.raises(IndexError, match="Cropped image would be empty"):
        kymo.crop_by_distance(5e-3, 2e-3)

    with pytest.raises(IndexError, match="Cropped image would be empty"):
        kymo.crop_by_distance(2e-3, 2e-3)

    with pytest.raises(IndexError, match="Cropped image would be empty"):
        kymo.crop_by_distance(20e3, 21e3)

    # Test rounding internally
    np.testing.assert_allclose(
        kymo.crop_by_distance(pixel_size_nm * 1.6 * 1e-3, pixel_size_nm * 1.6 * 1e-3).get_image("red"),
        image[1:2, :]
    )
    np.testing.assert_allclose(
        kymo.crop_by_distance(pixel_size_nm * 1.6 * 1e-3, pixel_size_nm * 2.1 * 1e-3).get_image("red"),
        image[1:3, :]
    )
    np.testing.assert_allclose(
        kymo.crop_by_distance(pixel_size_nm * 2.1 * 1e-3, pixel_size_nm * 2.1 * 1e-3).get_image("red"),
        image[2:3, :]
    )

    # Test cropping in base pairs
    kymo_bp = kymo.calibrate_to_kbp(1.000) # pixelsize = 0.2 kbp
    np.testing.assert_allclose(kymo_bp.crop_by_distance(0.2, 0.6).get_image("red"),
                               [[0, 0, 0, 0, 0, 6, 0],
                                [12, 0, 0, 0, 12, 6, 0]])
    np.testing.assert_allclose(kymo_bp.crop_by_distance(0.2, 0.7).get_image("red"),
                               [[0, 0, 0, 0, 0, 6, 0],
                                [12, 0, 0, 0, 12, 6, 0],
                                [0, 12, 12, 12, 0, 6, 0]])
    np.testing.assert_allclose(kymo_bp.crop_by_distance(0.2, 0.8).get_image("red"),
                               [[0, 0, 0, 0, 0, 6, 0],
                                [12, 0, 0, 0, 12, 6, 0],
                                [0, 12, 12, 12, 0, 6, 0]])


def test_kymo_crop_ds():
    """Test cropping interaction with downsampling"""

    image = np.array(
        [
            [0, 12, 0, 12, 0, 6, 0],
            [0, 0, 0, 0, 0, 6, 0],
            [12, 0, 0, 0, 12, 6, 0],
            [0, 12, 12, 12, 0, 6, 0],
            [0, 12, 12, 12, 0, 6, 0],
            [12, 12, 12, 12, 0, 6, 0],
            [24, 12, 12, 12, 0, 6, 0],
        ],
        dtype=np.uint8
    )

    pixel_size_nm = 2
    kymo = generate_kymo(
        "Mock",
        image,
        pixel_size_nm=pixel_size_nm,
        start=with_offset(100),
        dt=5,
        samples_per_pixel=5,
        line_padding=2
    )

    kymo_ds_pos = kymo.downsampled_by(position_factor=2)
    cropped = kymo_ds_pos.crop_by_distance(4e-3, 8e-3)
    np.testing.assert_allclose(cropped.get_image("red"), kymo_ds_pos.get_image("red")[1:2, :])
    np.testing.assert_allclose(cropped.timestamps, kymo_ds_pos.timestamps[1:2, :])
    np.testing.assert_allclose(cropped.pixelsize_um, kymo_ds_pos.pixelsize_um)
    np.testing.assert_allclose(cropped.line_time_seconds, kymo_ds_pos.line_time_seconds)
    np.testing.assert_allclose(cropped.pixels_per_line, 1)
    np.testing.assert_allclose(cropped._position_offset, 4e-3)

    kymo_ds_time = kymo.downsampled_by(time_factor=2)
    cropped = kymo_ds_time.crop_by_distance(4e-3, 8e-3)
    np.testing.assert_allclose(cropped.get_image("red"), kymo_ds_time.get_image("red")[2:4, :])
    np.testing.assert_allclose(cropped.pixelsize_um, kymo_ds_time.pixelsize_um)
    np.testing.assert_allclose(cropped.line_time_seconds, kymo_ds_time.line_time_seconds)
    np.testing.assert_allclose(cropped.pixels_per_line, 2)
    np.testing.assert_allclose(cropped._position_offset, 4e-3)

    def check_order_of_operations(time_factor, pos_factor, crop_x, crop_y):
        crop_ds = kymo.crop_by_distance(crop_x, crop_y).downsampled_by(time_factor, pos_factor)
        ds_crop = kymo.downsampled_by(time_factor, pos_factor).crop_by_distance(crop_x, crop_y)

        np.testing.assert_allclose(crop_ds.get_image("red"), ds_crop.get_image("red"))
        np.testing.assert_allclose(crop_ds.line_time_seconds, ds_crop.line_time_seconds)
        np.testing.assert_allclose(crop_ds.pixelsize_um, ds_crop.pixelsize_um)
        np.testing.assert_allclose(crop_ds.pixels_per_line, ds_crop.pixels_per_line)
        np.testing.assert_allclose(crop_ds._position_offset, ds_crop._position_offset)

        if time_factor == 1:
            np.testing.assert_allclose(crop_ds.get_image("red"), ds_crop.get_image("red"))

    # Note that the order of operations check only makes sense for where the cropping happens on
    # a multiple of the downsampling.
    check_order_of_operations(2, 1, 4e-3, 8e-3)
    check_order_of_operations(3, 1, 4e-3, 8e-3)
    check_order_of_operations(1, 2, 4e-3, 8e-3)
    check_order_of_operations(2, 2, 4e-3, 12e-3)
    check_order_of_operations(1, 3, 6e-3, 14e-3)


def test_kymo_slice_crop():
    """Test cropping after slicing"""
    image = np.array(
        [
            [0, 12, 0, 12, 0, 6, 0],
            [0, 0, 0, 0, 0, 6, 0],
            [12, 0, 0, 0, 12, 6, 0],
            [0, 12, 12, 12, 0, 6, 0],
            [0, 12, 12, 12, 0, 6, 0],
        ],
        dtype=np.uint8
    )
    kymo = generate_kymo(
        "Mock",
        image,
        pixel_size_nm=4,
        start=int(100e9),
        dt=int(5e9),
        samples_per_pixel=5,
        line_padding=2
    )

    sliced_cropped = kymo["245s":"725s"].crop_by_distance(8e-3, 14e-3)
    np.testing.assert_equal(
        sliced_cropped.timestamps, [[460e9, 605e9, 750e9], [485e9, 630e9, 775e9]]
    )
    np.testing.assert_allclose(sliced_cropped.get_image("red"), [[0, 0, 12], [12, 12, 0]])
    np.testing.assert_allclose(sliced_cropped._position_offset, 8e-3)
    np.testing.assert_equal(
        sliced_cropped._timestamps("timestamps", reduce=np.min),
        [[450e9, 595e9, 740e9], [475e9, 620e9, 765e9]],
    )



def test_incremental_offset():
    """Test whether cropping twice propagates the offset correctly"""
    image = np.array(
        [
            [0, 12, 0, 12, 0, 6, 0],
            [0, 0, 0, 0, 0, 6, 0],
            [12, 0, 0, 0, 12, 6, 0],
            [0, 12, 12, 12, 0, 6, 0],
            [0, 12, 12, 12, 0, 6, 0],
        ],
        dtype=np.uint8,
    )

    kymo = generate_kymo(
        "Mock",
        image,
        pixel_size_nm=2,
        start=with_offset(100),
        dt=5,
        samples_per_pixel=5,
        line_padding=2
    )
    cropped = kymo.crop_by_distance(2e-3, 8e-3)
    twice_cropped = cropped.crop_by_distance(2e-3, 8e-3)
    np.testing.assert_allclose(
        twice_cropped.get_image("red"),
        [[12.0, 0.0, 0.0, 0.0, 12.0, 6.0, 0.0], [0.0, 12.0, 12.0, 12.0, 0.0, 6.0, 0.0]],
    )
    np.testing.assert_equal(
        twice_cropped.timestamps,
        with_offset([[170, 315, 460, 605, 750, 895, 1040], [195, 340, 485, 630, 775, 920, 1065]]),
    )
    np.testing.assert_allclose(twice_cropped.pixelsize_um, kymo.pixelsize_um)
    np.testing.assert_allclose(twice_cropped.line_time_seconds, kymo.line_time_seconds)
    np.testing.assert_allclose(twice_cropped.pixels_per_line, 2)
    np.testing.assert_allclose(twice_cropped._position_offset, 4e-3)


def test_slice_timestamps():
    """Test slicing with realistically sized timestamps (this tests against floating point errors
    induced in kymograph reconstruction)"""
    image = np.array(
        [
            [0, 12, 0, 12, 0, 6, 0],
            [0, 0, 0, 0, 0, 6, 0],
            [12, 0, 0, 0, 12, 6, 0],
        ],
        dtype=np.uint8
    )

    kymo = generate_kymo(
        "Mock",
        image,
        pixel_size_nm=4,
        start=1623965975045144000,
        dt=int(1e9),
        samples_per_pixel=5,
        line_padding=2
    )

    # Kymo line is 3 * 5 samples long, while there is 2 pixel padding on each side.
    # Starting pixel time stamps relative to the original are as follows:
    # [[2  21  40  59  78  97 116]
    #  [7  26  45  64  83 102 121]
    # [12  31  50  69  88 107 126]]
    ref_ts = kymo.timestamps
    sliced = kymo["2s":"120s"]
    np.testing.assert_allclose(sliced.timestamps, ref_ts)
    sliced = kymo["3s":"120s"]
    np.testing.assert_allclose(sliced.timestamps, ref_ts[:, 1:])
    sliced = kymo["21s":"120s"]
    np.testing.assert_allclose(sliced.timestamps, ref_ts[:, 1:])
    sliced = kymo["22s":"120s"]
    np.testing.assert_allclose(sliced.timestamps, ref_ts[:, 2:])
    sliced = kymo["22s":"97s"]
    np.testing.assert_allclose(sliced.timestamps, ref_ts[:, 2:5])
    sliced = kymo["22s":"98s"]
    np.testing.assert_allclose(sliced.timestamps, ref_ts[:, 2:6])
    sliced = kymo["0s":"98s"]
    np.testing.assert_allclose(sliced.timestamps, ref_ts[:, :6])


def test_roundoff_errors_kymo():
    """Test slicing with realistically sized timestamps (this tests against floating point errors
    induced in kymograph reconstruction)"""
    image = np.array(
        [
            [0, 12, 0, 12, 0, 6, 0],
            [0, 0, 0, 0, 0, 6, 0],
            [12, 0, 0, 0, 12, 6, 0],
            [12, 0, 0, 0, 12, 6, 0],
        ],
        dtype=np.uint8
    )

    test_parameters = {
        "start": 1623965975045144000,
        "dt": int(1e9),
        "samples_per_pixel": 10,
        "line_padding": 2,
    }

    kymo = generate_kymo(
        "Mock",
        image,
        pixel_size_nm=4,
        **test_parameters,
    )

    pixel_time = test_parameters["dt"] * test_parameters["samples_per_pixel"]
    padding_time = test_parameters["dt"] * test_parameters["line_padding"]

    first_pixel_start = test_parameters["start"] + padding_time
    pixel_area_time = image.shape[0] * pixel_time
    line_time = pixel_area_time + 2 * padding_time
    # Note that the - dt comes from the mean over the pixel being not inclusive of the end.
    first_pixel_center = (2 * first_pixel_start + pixel_time - test_parameters["dt"]) // 2
    timestamp_line = first_pixel_center + np.arange(image.shape[0], dtype=np.int64) * pixel_time

    ref_timestamps = np.tile(timestamp_line, (image.shape[1], 1)).T
    ref_timestamps += np.arange(image.shape[1], dtype=np.int64) * line_time

    np.testing.assert_equal(kymo.timestamps, ref_timestamps)


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
        line_padding=0
    )
    assert len(np.unique(np.diff(kymo.timestamps))) == 1


def test_calibrate_to_kbp():

    image = np.array(
        [
            [0, 12, 0, 12, 0, 6, 0],
            [0, 0, 0, 0, 0, 6, 0],
            [12, 0, 0, 0, 12, 6, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 6, 6, 0, 0, 12],
            [6, 12, 12, 0, 0, 0, 0],
        ],
        dtype=np.uint8
    )

    kymo = generate_kymo(
        "Mock",
        image,
        pixel_size_nm=100,
        start=1623965975045144000,
        dt=int(1e9),
        samples_per_pixel=5,
        line_padding=2
    )

    kymo_bp = kymo.calibrate_to_kbp(12.000)

    # test that default calibration is in microns
    assert kymo._calibration.unit == "um"
    assert kymo._calibration.value == 0.1

    # test that calibration is stored as kilobase-pairs
    assert kymo_bp._calibration.unit == "kbp"
    np.testing.assert_allclose(kymo_bp._calibration.value, 2.0)

    # test conversion from microns to calibration units
    np.testing.assert_allclose(kymo._calibration.value * kymo._num_pixels[0], 0.6)
    np.testing.assert_allclose(kymo.pixelsize, 0.1)
    np.testing.assert_allclose(kymo_bp._calibration.value * kymo._num_pixels[0], 12.0)
    np.testing.assert_allclose(kymo_bp.pixelsize, 2.0)

    # test that all factories were forwarded from original instance
    def check_factory_forwarding(kymo1, kymo2, check_timestamps):
        assert kymo1._image_factory == kymo2._image_factory
        assert kymo1._timestamp_factory == kymo2._timestamp_factory
        assert kymo1._line_time_factory == kymo2._line_time_factory
        assert kymo1._pixelsize_factory == kymo2._pixelsize_factory
        assert kymo1._pixelcount_factory == kymo2._pixelcount_factory
        np.testing.assert_allclose(kymo1.get_image("red"), kymo2.get_image("red"))
        if check_timestamps:
            np.testing.assert_allclose(kymo1.timestamps, kymo2.timestamps)

    # check that calibration is supported for any processing (downsampling/cropping)
    # and that data remains the same after calibration
    ds_kymo_time = kymo.downsampled_by(time_factor=2)
    ds_kymo_pos = kymo.downsampled_by(position_factor=3)
    ds_kymo_both = kymo.downsampled_by(time_factor=2, position_factor=3)
    sliced_kymo = kymo["0s":"110s"]
    cropped_kymo = kymo.crop_by_distance(0, 0.5)

    ds_kymo_time_bp = ds_kymo_time.calibrate_to_kbp(12.000)
    ds_kymo_pos_bp = ds_kymo_pos.calibrate_to_kbp(12.000) # total length does not change
    ds_kymo_both_bp = ds_kymo_both.calibrate_to_kbp(12.000)
    sliced_kymo_bp = sliced_kymo.calibrate_to_kbp(12.000)
    cropped_kymo_bp = cropped_kymo.calibrate_to_kbp(int(12.000 * (5/6)))

    check_factory_forwarding(kymo, kymo_bp, True)
    check_factory_forwarding(ds_kymo_time, ds_kymo_time_bp, False)
    check_factory_forwarding(ds_kymo_pos, ds_kymo_pos_bp, True)
    check_factory_forwarding(ds_kymo_both, ds_kymo_both_bp, False)
    check_factory_forwarding(sliced_kymo, sliced_kymo_bp, True)
    check_factory_forwarding(cropped_kymo, cropped_kymo_bp, True)

    # if properly calibrated, cropping should not change pixel size
    np.testing.assert_allclose(kymo_bp.pixelsize[0], cropped_kymo_bp.pixelsize[0])
    # but will change total length
    np.testing.assert_allclose(kymo_bp._calibration.value * kymo._num_pixels[0] * 5/6,
                               cropped_kymo_bp._calibration.value * cropped_kymo_bp._num_pixels[0])


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
        line_padding=0
    )

    kymo.infowave.data[-60:] = 0  # Remove the last pixel entirely, and a partial pixel before that
    np.testing.assert_equal(kymo.timestamps[-1, -1], 0)
    np.testing.assert_equal(kymo.timestamps[-2, -1], 0)
    np.testing.assert_equal(kymo.get_image("red")[-1, -1], 0)
    np.testing.assert_equal(kymo.get_image("red")[-2, -1], 0)


@cleanup
def test_plot_with_lf_force():
    dt = int(1e9 / 78125)
    start = 1536582124217030400
    kymo = generate_kymo(
        "Mock",
        np.ones((5, 5)),
        pixel_size_nm=100,
        start=start,
        dt=dt,
        samples_per_pixel=2,
        line_padding=0,
    )

    # Mock a force channel onto this file
    data = np.array([-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    timestamps = data * 5 * dt + start
    channel = Slice(TimeSeries(data, timestamps))
    setattr(kymo.file, "force1x", empty_slice)  # Not present in file
    setattr(kymo.file, "downsampled_force1x", empty_slice)  # Not present in file
    setattr(kymo.file, "force2y", empty_slice)  # Not present in file
    setattr(kymo.file, "downsampled_force2y", channel)

    with pytest.warns(
        RuntimeWarning, match="Using downsampled force since high frequency force is unavailable."
    ):
        kymo.plot_with_force("2y", "red")
        np.testing.assert_allclose(plt.gca().lines[0].get_ydata(), [0.5, 2.5, 4.5, 6.5, 8.5])

    with pytest.raises(RuntimeError, match="Desired force channel 1x not available in h5 file"):
        kymo.plot_with_force("1x", "red")


@cleanup
def test_kymo_plot_rgb_absolute_color_adjustment(test_kymos):
    """Tests whether we can set an absolute color range for the RGB plot."""
    kymo = test_kymos["Kymo1"]

    fig = plt.figure()
    lb, ub = np.array([1, 2, 3]), np.array([2, 3, 4])
    kymo.plot(channel="rgb", adjustment=ColorAdjustment(lb, ub, mode="absolute"))
    image = plt.gca().get_images()[0]
    np.testing.assert_allclose(image.get_array(), np.clip((kymo.get_image("rgb") - lb) / (ub - lb), 0, 1))
    plt.close(fig)


@cleanup
def test_kymo_plot_rgb_percentile_color_adjustment(test_kymos):
    """Tests whether we can set a percentile color range for the RGB plot."""
    kymo = test_kymos["Kymo1"]

    fig = plt.figure()
    lb, ub = np.array([10, 10, 10]), np.array([80, 80, 80])
    kymo.plot(channel="rgb", adjustment=ColorAdjustment(lb, ub, mode="percentile"))
    image = plt.gca().get_images()[0]
    bounds = np.array(
        [
            np.percentile(img, [mini, maxi])
            for img, mini, maxi in zip(np.moveaxis(kymo.get_image("rgb"), 2, 0), lb, ub)
        ]
    )
    lb, ub = (b for b in np.moveaxis(bounds, 1, 0))
    np.testing.assert_allclose(image.get_array(), np.clip((kymo.get_image("rgb") - lb) / (ub - lb), 0, 1))
    plt.close(fig)


@cleanup
def test_kymo_plot_single_channel_absolute_color_adjustment(test_kymos):
    """Tests whether we can set an absolute color range for a single channel plot."""
    kymo = test_kymos["Kymo1"]

    lbs, ubs = np.array([1, 2, 3]), np.array([2, 3, 4])
    for lb, ub, channel in zip(lbs, ubs, ("red", "green", "blue")):
        # Test whether setting RGB values and then sampling one of them works correctly.
        fig = plt.figure()
        kymo.plot(channel=channel, adjustment=ColorAdjustment(lbs, ubs, mode="absolute"))
        image = plt.gca().get_images()[0]
        np.testing.assert_allclose(image.get_array(), kymo.get_image(channel)) #getattr(kymo, f"{channel}_image"))
        np.testing.assert_allclose(image.get_clim(), [lb, ub])
        plt.close(fig)

        # Test whether setting a single color works correctly (should use the same for R G and B).
        fig = plt.figure()
        kymo.plot(channel=channel, adjustment=ColorAdjustment(lb, ub, mode="absolute"))
        image = plt.gca().get_images()[0]
        np.testing.assert_allclose(image.get_array(), kymo.get_image(channel)) #getattr(kymo, f"{channel}_image"))
        np.testing.assert_allclose(image.get_clim(), [lb, ub])
        plt.close(fig)


@pytest.mark.parametrize("crop", [False, True])
def test_flip_kymo(test_kymos, crop):
    kymo = test_kymos["Kymo1"]
    if crop:
        kymo = kymo.crop_by_distance(0.01, 0.03)
    kymo_flipped = kymo.flip()
    for channel in ("red", "green", "blue", "rgb"):
        np.testing.assert_allclose(
            kymo_flipped.get_image(channel=channel),
            np.flip(kymo.get_image(channel=channel), axis=0),
        )
        np.testing.assert_allclose(
            kymo_flipped.flip().get_image(channel=channel),
            kymo.get_image(channel=channel),
        )
