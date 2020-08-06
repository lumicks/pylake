import numpy as np
from lumicks import pylake
import pytest
from lumicks.pylake.kymo import EmptyKymo


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
        assert kymo.has_fluorescence
        assert not kymo.has_force
        assert kymo.pixels_per_line == 5
        assert len(kymo.infowave) == 64
        assert kymo.rgb_image.shape == (5, 4, 3)
        assert kymo.red_image.shape == (5, 4)
        assert kymo.blue_image.shape == (5, 4)
        assert kymo.green_image.shape == (5, 4)
        assert np.allclose(kymo.timestamps, reference_timestamps)
        assert kymo.fast_axis == "X"


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
        assert empty_kymograph.has_fluorescence
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

        kymo.start = kymo.red_photon_count.timestamps[0] - 1  # Assume the user incorrectly exported only a partial Kymo
        with pytest.warns(RuntimeWarning):
            assert kymo.red_image.shape == (5, 3)
        assert np.allclose(kymo.red_image.data, kymo_reference[:, 1:])
