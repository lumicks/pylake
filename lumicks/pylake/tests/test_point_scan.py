import pytest
import matplotlib.pyplot as plt
import numpy as np
from lumicks import pylake
from matplotlib.testing.decorators import cleanup


def test_point_scans(h5_file):
    f = pylake.File.from_h5py(h5_file)
    if f.format_version == 2:
        ps = f.point_scans["PointScan1"]
        ps_red = ps.red_photon_count

        assert ps_red.data.shape == (64, )
        reference_timestamps = np.arange(2.0e+10, 2.0e+10+(6.25e+7*len(ps_red)) , 6.25e+7)
        reference_data = np.array([2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 1, 8, 0, 0, 0, 0,
                                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 0, 1, 0, 0, 0, 0, 0,
                                   0, 0, 0, 0, 0, 0, 1, 0, 8, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0,
                                   0, 1, 0, 0, 0, 8, 0])
        np.testing.assert_allclose(ps_red.timestamps, reference_timestamps)
        np.testing.assert_allclose(ps_red.data, reference_data)

        with pytest.deprecated_call():
            ps.json
        with pytest.deprecated_call():
            assert ps.has_fluorescence
        with pytest.deprecated_call():
            assert not ps.has_force


@cleanup
def test_plotting(h5_file):
    import matplotlib.pyplot as plt
    f = pylake.File.from_h5py(h5_file)
    if f.format_version == 2:
        ps = f.point_scans["PointScan1"]
        for plot_func in (ps.plot_red, ps.plot_green, ps.plot_blue):
            plot_func()
            np.testing.assert_allclose(np.sort(plt.xlim()), [0, 0.0625 * 64])
            np.testing.assert_allclose(np.sort(plt.ylim()), [0, 8])

        ps.plot_rgb(lw=5)
        np.testing.assert_allclose(np.sort(plt.xlim()), [0, 0.0625 * 64])
        np.testing.assert_allclose(np.sort(plt.ylim()), [0, 8])
