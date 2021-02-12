import numpy as np
from lumicks import pylake
import pytest
from lumicks.pylake.kymo import EmptyKymo
import matplotlib.pyplot as plt
from matplotlib.testing.decorators import cleanup


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
        assert np.allclose(kymo.timestamps, reference_timestamps)
        assert kymo.fast_axis == "X"
        assert np.allclose(kymo.pixelsize_um, 10/1000)
        assert np.allclose(kymo.line_time_seconds, 1.03125)
        assert np.allclose(kymo.center_point_um["x"], 58.075877109272604)
        assert np.allclose(kymo.center_point_um["y"], 31.978375270573267)
        assert np.allclose(kymo.center_point_um["z"], 0)
        assert np.allclose(kymo.scan_width_um, [0.050])


def test_kymo_slicing(h5_file):
    f = pylake.File.from_h5py(h5_file)
    if f.format_version == 2:
        kymo = f.kymos["Kymo1"]
        kymo_reference = np.transpose([[2, 0, 0, 0, 2], [0, 0, 0, 0, 0], [1, 0, 0, 0, 1], [0, 1, 1, 1, 0]])

        assert kymo.red_image.shape == (5, 4)
        assert np.allclose(kymo.red_image.data, kymo_reference)

        sliced = kymo[:]
        assert sliced.red_image.shape == (5, 4)
        assert np.allclose(sliced.red_image.data, kymo_reference)

        sliced = kymo["1s":]
        assert sliced.red_image.shape == (5, 3)
        assert np.allclose(sliced.red_image.data, kymo_reference[:, 1:])

        sliced = kymo["0s":]
        assert sliced.red_image.shape == (5, 4)
        assert np.allclose(sliced.red_image.data, kymo_reference)

        sliced = kymo["0s":"2s"]
        assert sliced.red_image.shape == (5, 2)
        assert np.allclose(sliced.red_image.data, kymo_reference[:, :2])

        sliced = kymo["0s":"-1s"]
        assert sliced.red_image.shape == (5, 3)
        assert np.allclose(sliced.red_image.data, kymo_reference[:, :-1])

        sliced = kymo["0s":"-2s"]
        assert sliced.red_image.shape == (5, 2)
        assert np.allclose(sliced.red_image.data, kymo_reference[:, :-2])

        sliced = kymo["0s":"3s"]
        assert sliced.red_image.shape == (5, 3)
        assert np.allclose(sliced.red_image.data, kymo_reference[:, :3])

        sliced = kymo["1s":"2s"]
        assert sliced.red_image.shape == (5, 1)
        assert np.allclose(sliced.red_image.data, kymo_reference[:, 1:2])

        sliced = kymo["0s":"10s"]
        assert sliced.red_image.shape == (5, 4)
        assert np.allclose(sliced.red_image.data, kymo_reference[:, 0:10])

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
        assert np.allclose(kymo.red_image.data, kymo_reference[:, 1:])


@cleanup
def test_plotting(h5_file):
    f = pylake.File.from_h5py(h5_file)
    if f.format_version == 2:
        kymo = f.kymos["Kymo1"]

        kymo.plot_red()
        assert np.allclose(np.sort(plt.xlim()), [-0.5, 3.5], atol=0.05)
        assert np.allclose(np.sort(plt.ylim()), [0, 0.05])


@cleanup
def test_plotting_with_force(h5_file):
    f = pylake.File.from_h5py(h5_file)
    if f.format_version == 2:
        kymo = f.kymos["Kymo1"]

        ds = kymo._downsample_channel(2, "x", reduce=np.mean)
        assert np.allclose(ds.data, [30, 30, 10, 10])
        assert np.all(np.equal(ds.timestamps, kymo.timestamps[2]))

        kymo.plot_with_force(force_channel="2x", color_channel="red")
        assert np.allclose(np.sort(plt.xlim()), [-0.5, 3.5], atol=0.05)
        assert np.allclose(np.sort(plt.ylim()), [10, 30])


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
        assert np.allclose(h, 0.01)
        assert np.all(np.equal(w, [3, 1, 1, 1, 3]))
        assert np.allclose(np.sort(plt.xlim()), [0, 3], atol=0.05)

        kymo.plot_with_time_histogram(color_channel="red", pixels_per_bin=1)
        w, h = get_rectangle_data()
        assert np.allclose(w, 1.03, atol=0.002)
        assert np.all(np.equal(h, [4, 0, 2, 3]))
        assert np.allclose(np.sort(plt.ylim()), [0, 4], atol=0.05)

        with pytest.warns(UserWarning):
            kymo.plot_with_position_histogram(color_channel="red", pixels_per_bin=3)
            w, h = get_rectangle_data()
            assert np.allclose(h, [0.03, 0.02])
            assert np.all(np.equal(w, [5, 4]))
            assert np.allclose(np.sort(plt.xlim()), [0, 5], atol=0.05)

        with pytest.warns(UserWarning):
            kymo.plot_with_time_histogram(color_channel="red", pixels_per_bin=3)
            w, h = get_rectangle_data()
            assert np.allclose(w, [3.09, 1.03], atol=0.02)
            assert np.all(np.equal(h, [6, 3]))
            assert np.allclose(np.sort(plt.ylim()), [0, 6], atol=0.05)

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
