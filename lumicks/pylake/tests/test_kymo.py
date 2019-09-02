import numpy as np
from lumicks import pylake
import pytest


def test_kymo_slicing(h5_file):
    f = pylake.File.from_h5py(h5_file)
    if f.format_version == 2:
        kymo = f.kymos["Kymo1"]
        assert kymo.red_image.shape == (5, 4)
        assert kymo[:].red_image.shape == (5, 4)
        assert kymo["1s":].red_image.shape == (5, 3)
        assert kymo["0s":].red_image.shape == (5, 4)
        assert kymo["0s":"2s"].red_image.shape == (5, 2)
        assert kymo["0s":"-1s"].red_image.shape == (5, 3)
        assert kymo["0s":"-2s"].red_image.shape == (5, 2)
        assert kymo["0s":"3s"].red_image.shape == (5, 3)
        assert kymo["1s":"2s"].red_image.shape == (5, 1)
        assert kymo["0s":"10s"].red_image.shape == (5, 4)

        with pytest.raises(IndexError):
            kymo["5s":]

        with pytest.raises(IndexError):
            kymo["3s":"2s"]
