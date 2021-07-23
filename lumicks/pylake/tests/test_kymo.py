import re
import numpy as np
from lumicks import pylake
import pytest
from lumicks.pylake.kymotracker.detail.calibrated_images import CalibratedKymographChannel
from lumicks.pylake.kymo import EmptyKymo
import matplotlib.pyplot as plt
from matplotlib.testing.decorators import cleanup
from .data.mock_confocal import generate_kymo


def test_kymo_properties(h5_file):

    f = pylake.File.from_h5py(h5_file)
    if f.format_version == 2:
        kymo = f.kymos["Kymo1"]
        reference_timestamps = np.array([[2.006250e+10, 2.109375e+10, 2.206250e+10, 2.309375e+10],
                                        [2.025000e+10, 2.128125e+10, 2.225000e+10, 2.328125e+10],
                                        [2.043750e+10, 2.146875e+10, 2.243750e+10, 2.346875e+10],
                                        [2.062500e+10, 2.165625e+10, 2.262500e+10, 2.365625e+10],
                                        [2.084375e+10, 2.187500e+10, 2.284375e+10, 2.387500e+10]], np.int64)

        assert repr(kymo) == "Kymo(pixels=5)"
        with pytest.deprecated_call():
            kymo.json
        with pytest.deprecated_call():
            assert kymo.has_fluorescence
        with pytest.deprecated_call():
            assert not kymo.has_force
        assert kymo.pixels_per_line == 5
        assert len(kymo.infowave) == 64
        assert kymo.rgb_image.shape == (5, 4, 3)
        assert kymo.red_image.shape == (5, 4)
        assert kymo.blue_image.shape == (5, 4)
        assert kymo.green_image.shape == (5, 4)
        np.testing.assert_allclose(kymo.timestamps, reference_timestamps)
        assert kymo.fast_axis == "X"
        np.testing.assert_allclose(kymo.pixelsize_um, 10/1000)
        np.testing.assert_allclose(kymo.line_time_seconds, 1.03125)
        np.testing.assert_allclose(kymo.center_point_um["x"], 58.075877109272604)
        np.testing.assert_allclose(kymo.center_point_um["y"], 31.978375270573267)
        np.testing.assert_allclose(kymo.center_point_um["z"], 0)
        np.testing.assert_allclose(kymo.size_um, [0.050])


def test_kymo_slicing(h5_file):
    f = pylake.File.from_h5py(h5_file)
    if f.format_version == 2:
        kymo = f.kymos["Kymo1"]
        kymo_reference = np.transpose([[2, 0, 0, 0, 2], [0, 0, 0, 0, 0], [1, 0, 0, 0, 1], [0, 1, 1, 1, 0]])

        assert kymo.red_image.shape == (5, 4)
        np.testing.assert_allclose(kymo.red_image.data, kymo_reference)

        sliced = kymo[:]
        assert sliced.red_image.shape == (5, 4)
        np.testing.assert_allclose(sliced.red_image.data, kymo_reference)

        sliced = kymo["1s":]
        assert sliced.red_image.shape == (5, 3)
        np.testing.assert_allclose(sliced.red_image.data, kymo_reference[:, 1:])

        sliced = kymo["0s":]
        assert sliced.red_image.shape == (5, 4)
        np.testing.assert_allclose(sliced.red_image.data, kymo_reference)

        sliced = kymo["0s":"2s"]
        assert sliced.red_image.shape == (5, 2)
        np.testing.assert_allclose(sliced.red_image.data, kymo_reference[:, :2])

        sliced = kymo["0s":"-1s"]
        assert sliced.red_image.shape == (5, 3)
        np.testing.assert_allclose(sliced.red_image.data, kymo_reference[:, :-1])

        sliced = kymo["0s":"-2s"]
        assert sliced.red_image.shape == (5, 2)
        np.testing.assert_allclose(sliced.red_image.data, kymo_reference[:, :-2])

        sliced = kymo["0s":"3s"]
        assert sliced.red_image.shape == (5, 3)
        np.testing.assert_allclose(sliced.red_image.data, kymo_reference[:, :3])

        sliced = kymo["1s":"2s"]
        assert sliced.red_image.shape == (5, 1)
        np.testing.assert_allclose(sliced.red_image.data, kymo_reference[:, 1:2])

        sliced = kymo["0s":"10s"]
        assert sliced.red_image.shape == (5, 4)
        np.testing.assert_allclose(sliced.red_image.data, kymo_reference[:, 0:10])

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
            empty_kymograph.save_tiff("test")

        with pytest.raises(RuntimeError):
            empty_kymograph.plot_rgb()

        assert empty_kymograph.red_image.shape == (5, 0)
        with pytest.deprecated_call():
            empty_kymograph.json
        with pytest.deprecated_call():
            assert empty_kymograph.has_fluorescence
        with pytest.deprecated_call():
            assert not empty_kymograph.has_force
        assert empty_kymograph.infowave.data.size == 0
        assert empty_kymograph.pixels_per_line == 5
        assert empty_kymograph.red_image.size == 0
        assert empty_kymograph.rgb_image.size == 0


def test_damaged_kymo(h5_file):
    f = pylake.File.from_h5py(h5_file)

    if f.format_version == 2:
        kymo = f.kymos["Kymo1"]
        kymo_reference = np.transpose([[2, 0, 0, 0, 2], [0, 0, 0, 0, 0], [1, 0, 0, 0, 1], [0, 1, 1, 1, 0]])

        # Assume the user incorrectly exported only a partial Kymo
        kymo.start = kymo.red_photon_count.timestamps[0] - 62500000
        with pytest.warns(RuntimeWarning):
            assert kymo.red_image.shape == (5, 3)
        np.testing.assert_allclose(kymo.red_image.data, kymo_reference[:, 1:])


@cleanup
def test_plotting(h5_file):
    f = pylake.File.from_h5py(h5_file)
    if f.format_version == 2:
        kymo = f.kymos["Kymo1"]

        kymo.plot_red()
        # The following assertion fails because of unequal line times in the test data. These
        # unequal line times are not typical for BL data. Kymo nowadays assumes equal line times
        # which is why the old version of this test fails.
        # np.testing.assert_allclose(np.sort(plt.xlim()), [-0.5, 3.5], atol=0.05)
        np.testing.assert_allclose(plt.xlim(), [-0.515625, 3.609375])
        np.testing.assert_allclose(np.sort(plt.ylim()), [0, 0.05])


@cleanup
def test_plotting_with_force(h5_file):
    f = pylake.File.from_h5py(h5_file)
    if f.format_version == 2:
        kymo = f.kymos["Kymo1"]

        ds = kymo._downsample_channel(2, "x", reduce=np.mean)
        np.testing.assert_allclose(ds.data, [30, 30, 10, 10])
        assert np.all(np.equal(ds.timestamps, kymo.timestamps[2]))

        kymo.plot_with_force(force_channel="2x", color_channel="red")

        # The following assertion fails because of unequal line times in the test data. These
        # unequal line times are not typical for BL data. Kymo nowadays assumes equal line times
        # which is why the old version of this test fails.
        # np.testing.assert_allclose(np.sort(plt.xlim()), [-0.5, 3.5], atol=0.05)
        np.testing.assert_allclose(plt.xlim(), [-0.515625, 3.609375])
        np.testing.assert_allclose(np.sort(plt.ylim()), [10, 30])


@cleanup
def test_regression_plot_with_force(h5_file):
    # Plot_with_force used to fail when the last line of a kymograph was incomplete. The reason for
    # this was that the last few timestamps on the last line had zero as their timestamp. This meant
    # it was trying to downsample a range from X to 0, which made the downsampler think that there
    # was no overlap between the kymograph and the force channel (as it checks the last timestamp
    # of the ranges to downsample to against the first one of the channel to downsample).
    f = pylake.File.from_h5py(h5_file)
    if f.format_version == 2:
        kymo = f.kymos["Kymo1"]
        incomplete_last_line_timestamps = kymo.timestamps
        incomplete_last_line_timestamps[-1, -1] = 0
        kymo._timestamp_factory = lambda self: incomplete_last_line_timestamps
        kymo.plot_with_force(force_channel="2x", color_channel="red")


@cleanup
def test_plotting_with_histograms(h5_file):
    def get_rectangle_data():
        widths = [p.get_width() for p in plt.gca().patches]
        heights = [p.get_height() for p in plt.gca().patches]
        return widths, heights

    f = pylake.File.from_h5py(h5_file)
    if f.format_version == 2:
        kymo = f.kymos["Kymo1"]

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


def test_save_tiff(tmpdir_factory, h5_file):
    from os import stat

    f = pylake.File.from_h5py(h5_file)
    tmpdir = tmpdir_factory.mktemp("pylake")

    if f.format_version == 2:
        kymo = f.kymos["Kymo1"]
        kymo.save_tiff(f"{tmpdir}/kymo1.tiff")
        assert stat(f"{tmpdir}/kymo1.tiff").st_size > 0


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
        "Mock", image, pixel_size_nm=1, start=100, dt=7, samples_per_pixel=5, line_padding=2
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
    np.testing.assert_allclose(kymo_ds.red_image, ds)
    np.testing.assert_allclose(kymo_ds.start, 100)
    np.testing.assert_allclose(kymo_ds.pixelsize_um, 1 / 1000)
    np.testing.assert_allclose(kymo_ds.line_time_seconds, 2 * 7 * (5 * 4 + 2 + 2) / 1e9)

    with pytest.raises(
            AttributeError,
            match=re.escape("Per-pixel timestamps are no longer available after downsampling"),
    ):
        kymo_ds.timestamps

    # Verify that we can pass a different reduce function
    np.testing.assert_allclose(kymo.downsampled_by(time_factor=2, reduce=np.mean).red_image, ds / 2)


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
        "Mock", image, pixel_size_nm=1, start=100, dt=5, samples_per_pixel=5, line_padding=2
    )

    kymo_ds = kymo.downsampled_by(position_factor=2)
    ds = np.array([[0, 12, 0, 12, 0, 12, 0], [12, 12, 12, 12, 12, 12, 0]], dtype=np.uint8)
    ds_ts = np.array([[132.5,  277.5,  422.5,  567.5,  712.5,  857.5, 1002.5],
                      [182.5,  327.5,  472.5,  617.5,  762.5,  907.5, 1052.5]])

    assert kymo_ds.name == "Mock"
    np.testing.assert_allclose(kymo_ds.red_image, ds)
    np.testing.assert_allclose(kymo_ds.timestamps, ds_ts)
    np.testing.assert_allclose(kymo_ds.start, 100)
    np.testing.assert_allclose(kymo_ds.pixelsize_um, 2 / 1000)
    np.testing.assert_allclose(kymo_ds.line_time_seconds, kymo.line_time_seconds)

    # We lost one line while downsampling
    np.testing.assert_allclose(kymo_ds.size_um[0], kymo.size_um[0] - kymo.pixelsize_um[0])

    # Verify that we can pass a different reduce function
    alt_ds = kymo.downsampled_by(position_factor=2, reduce=np.mean, reduce_timestamps=np.sum)
    np.testing.assert_allclose(alt_ds.red_image, ds / 2)
    np.testing.assert_allclose(alt_ds.timestamps, ds_ts * 2)


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
        np.testing.assert_allclose(kymo_ds.red_image, ds)
        np.testing.assert_allclose(kymo_ds.start, 100)
        np.testing.assert_allclose(kymo_ds.pixelsize_um, 2 / 1000)
        np.testing.assert_allclose(kymo_ds.line_time_seconds, 2 * 5 * (5 * 5 + 2 + 2) / 1e9)
        with pytest.raises(
                AttributeError,
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
        "Mock", image, pixel_size_nm=1, start=100, dt=5, samples_per_pixel=5, line_padding=2
    )
    timestamps = kymo.timestamps.copy()
    downsampled_kymos = kymo.downsampled_by(time_factor=2, position_factor=2)

    np.testing.assert_allclose(kymo.red_image, image)
    np.testing.assert_allclose(kymo.start, 100)
    np.testing.assert_allclose(kymo.pixelsize_um, 1 / 1000)
    np.testing.assert_allclose(kymo.line_time_seconds, 5 * (5 * 5 + 2 + 2) / 1e9)
    np.testing.assert_allclose(kymo.timestamps, timestamps)


def test_calibrated_channels():
    image = np.array(
        [
            [0, 12, 0, 12, 0],
            [0, 0, 0, 0, 0],
            [12, 0, 0, 0, 12],
            [0, 12, 12, 12, 0],
        ],
        dtype=np.uint8,
    )

    kymo = generate_kymo(
        "Mock",
        image,
        pixel_size_nm=7e3,
        start=int(4e9),
        dt=int(3e9),
        samples_per_pixel=5,
        line_padding=2
    )

    calibrated_channel = CalibratedKymographChannel.from_kymo(kymo, "red")
    np.testing.assert_allclose(calibrated_channel.data, image)
    np.testing.assert_allclose(calibrated_channel.time_step_ns, kymo.line_time_seconds * int(1e9))
    np.testing.assert_allclose(calibrated_channel._pixel_size, kymo.pixelsize_um[0])


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
