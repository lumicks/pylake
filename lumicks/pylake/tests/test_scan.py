import numpy as np
from lumicks import pylake
import pytest


def test_scans(h5_file):
    f = pylake.File.from_h5py(h5_file)
    if f.format_version == 2:
        scan = f.scans["Scan1"]

        assert repr(scan) == "Scan(pixels=(5, 4))"

        reference_timestamps = [[2.006250e+10, 2.109375e+10, 2.206250e+10, 2.309375e+10],
                                [2.025000e+10, 2.128125e+10, 2.225000e+10, 2.328125e+10],
                                [2.043750e+10, 2.146875e+10, 2.243750e+10, 2.346875e+10],
                                [2.062500e+10, 2.165625e+10, 2.262500e+10, 2.365625e+10],
                                [2.084375e+10, 2.187500e+10, 2.284375e+10, 2.387500e+10]]

        assert np.allclose(scan.timestamps, np.transpose(reference_timestamps))
        assert scan.num_frames == 1
        assert scan.has_fluorescence
        assert not scan.has_force
        assert scan.pixels_per_line == 5
        assert scan.lines_per_frame == 4
        assert len(scan.infowave) == 64
        assert scan.rgb_image.shape == (4, 5, 3)
        assert scan.red_image.shape == (4, 5)
        assert scan.blue_image.shape == (4, 5)
        assert scan.green_image.shape == (4, 5)

        with pytest.raises(NotImplementedError):
            scan["1s":"2s"]
