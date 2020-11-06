import pytest
import matplotlib.pyplot as plt
import numpy as np
from lumicks import pylake
from matplotlib.testing.decorators import cleanup


def test_scans(h5_file):
    f = pylake.File.from_h5py(h5_file)
    if f.format_version == 2:
        scan = f.scans["fast Y slow X"]

        assert repr(scan) == "Scan(pixels=(4, 5))"

        reference_timestamps = np.array([[2.006250e+10, 2.025000e+10, 2.043750e+10, 2.062500e+10],
                                        [2.084375e+10, 2.109375e+10, 2.128125e+10, 2.146875e+10],
                                        [2.165625e+10, 2.187500e+10, 2.206250e+10, 2.225000e+10],
                                        [2.243750e+10, 2.262500e+10, 2.284375e+10, 2.309375e+10],
                                        [2.328125e+10, 2.346875e+10, 2.365625e+10, 2.387500e+10]])

        assert np.allclose(scan.timestamps, np.transpose(reference_timestamps))
        assert scan.num_frames == 1
        assert scan.has_fluorescence
        assert not scan.has_force
        assert scan.pixels_per_line == 4
        assert scan.lines_per_frame == 5
        assert len(scan.infowave) == 64
        assert scan.rgb_image.shape == (4, 5, 3)
        assert scan.red_image.shape == (4, 5)
        assert scan.blue_image.shape == (4, 5)
        assert scan.green_image.shape == (4, 5)
        assert scan.fast_axis == "Y"
        assert np.allclose(scan.pixelsize_um, [197/1000, 191/1000])

        with pytest.raises(NotImplementedError):
            scan["1s":"2s"]

        scan = f.scans["fast Y slow X multiframe"]
        reference_timestamps2 = np.zeros((2, 4, 3))
        reference_timestamps2[0, :, :] = reference_timestamps.T[:, :3]
        reference_timestamps2[1, :, :2] = reference_timestamps.T[:, 3:]

        assert np.allclose(scan.timestamps, reference_timestamps2)
        assert scan.num_frames == 2
        assert scan.has_fluorescence
        assert not scan.has_force
        assert scan.pixels_per_line == 4
        assert scan.lines_per_frame == 3
        assert len(scan.infowave) == 64
        assert scan.rgb_image.shape == (2, 4, 3, 3)
        assert scan.red_image.shape == (2, 4, 3)
        assert scan.blue_image.shape == (2, 4, 3)
        assert scan.green_image.shape == (2, 4, 3)
        assert scan.fast_axis == "Y"
        assert np.allclose(scan.pixelsize_um, [197 / 1000, 191 / 1000])

        scan = f.scans["fast X slow Z multiframe"]
        reference_timestamps2 = np.zeros((2, 4, 3))
        reference_timestamps2[0, :, :] = reference_timestamps.T[:, :3]
        reference_timestamps2[1, :, :2] = reference_timestamps.T[:, 3:]
        reference_timestamps2 = reference_timestamps2.transpose([0, 2, 1])

        assert np.allclose(scan.timestamps, reference_timestamps2)
        assert scan.num_frames == 2
        assert scan.has_fluorescence
        assert not scan.has_force
        assert scan.pixels_per_line == 4
        assert scan.lines_per_frame == 3
        assert len(scan.infowave) == 64
        assert scan.rgb_image.shape == (2, 3, 4, 3)
        assert scan.red_image.shape == (2, 3, 4)
        assert scan.blue_image.shape == (2, 3, 4)
        assert scan.green_image.shape == (2, 3, 4)
        assert scan.fast_axis == "X"
        assert np.allclose(scan.pixelsize_um, [191 / 1000, 197 / 1000])

        scan = f.scans["fast Y slow Z multiframe"]
        reference_timestamps2 = np.zeros((2, 4, 3))
        reference_timestamps2[0, :, :] = reference_timestamps.T[:, :3]
        reference_timestamps2[1, :, :2] = reference_timestamps.T[:, 3:]
        reference_timestamps2 = reference_timestamps2.transpose([0, 2, 1])

        assert np.allclose(scan.timestamps, reference_timestamps2)
        assert scan.num_frames == 2
        assert scan.has_fluorescence
        assert not scan.has_force
        assert scan.pixels_per_line == 4
        assert scan.lines_per_frame == 3
        assert len(scan.infowave) == 64
        assert scan.rgb_image.shape == (2, 3, 4, 3)
        assert scan.red_image.shape == (2, 3, 4)
        assert scan.blue_image.shape == (2, 3, 4)
        assert scan.green_image.shape == (2, 3, 4)
        assert scan.fast_axis == "Y"
        assert np.allclose(scan.pixelsize_um, [191 / 1000, 197 / 1000])

        with pytest.raises(NotImplementedError):
            scan.line_time_seconds


def test_damaged_scan(h5_file):
    f = pylake.File.from_h5py(h5_file)

    if f.format_version == 2:
        scan = f.scans["fast Y slow X"]

        # Assume the user incorrectly exported only a partial scan (62500000 is the time step)
        scan.start = scan.red_photon_count.timestamps[0] - 62500000
        with pytest.raises(RuntimeError):
            scan.red_image.shape

        # Test for workaround for a bug in the STED delay mechanism which could result in scan start times ending up
        # within the sample time.
        scan = f.scans["fast Y slow X"]
        scan.start = scan.red_photon_count.timestamps[0] - 62400000
        scan.red_image.shape


@cleanup
def test_plotting(h5_file):
    f = pylake.File.from_h5py(h5_file)
    if f.format_version == 2:
        scan = f.scans["fast Y slow X multiframe"]
        scan.plot_blue()
        assert np.allclose(np.sort(plt.xlim()), [0, .197 * 3])
        assert np.allclose(np.sort(plt.ylim()), [0, .191 * 4])

        scan = f.scans["fast X slow Z multiframe"]
        scan.plot_rgb()
        assert np.allclose(np.sort(plt.xlim()), [0, .191 * 4])
        assert np.allclose(np.sort(plt.ylim()), [0, .197 * 3])

        scan = f.scans["fast Y slow Z multiframe"]
        scan.plot_rgb()
        assert np.allclose(np.sort(plt.xlim()), [0, .191 * 4])
        assert np.allclose(np.sort(plt.ylim()), [0, .197 * 3])


def test_movie_export(tmpdir_factory, h5_file):
    from os import stat

    f = pylake.File.from_h5py(h5_file)
    tmpdir = tmpdir_factory.mktemp("pylake")

    if f.format_version == 2:
        scan = f.scans["fast Y slow X multiframe"]
        scan.export_video_red(f"{tmpdir}/red.gif", 0, 4)
        assert stat(f"{tmpdir}/red.gif").st_size > 0
        scan.export_video_rgb(f"{tmpdir}/rgb.gif", 0, 4)
        assert stat(f"{tmpdir}/rgb.gif").st_size > 0
