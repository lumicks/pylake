import pytest
import matplotlib.pyplot as plt
import numpy as np
from lumicks import pylake
from matplotlib.testing.decorators import cleanup


def test_point_scans(test_point_scans, reference_timestamps, reference_counts):
    ps = test_point_scans["PointScan1"]
    ps_red = ps.red_photon_count

    assert ps_red.data.shape == (64,)

    np.testing.assert_allclose(ps_red.timestamps, reference_timestamps)
    np.testing.assert_allclose(ps_red.data, reference_counts)


@cleanup
def test_plotting(test_point_scans):
    ps = test_point_scans["PointScan1"]

    for channel in ("red", "green", "blue"):
        ps.plot(channel=channel)

        xline, yline = plt.gca().get_lines()[0].get_xydata().T
        count = getattr(ps, f"{channel}_photon_count")
        np.testing.assert_allclose(xline, (count.timestamps - count.timestamps[0]) * 1e-9)
        np.testing.assert_allclose(yline, count.data)
        plt.close()

    ps.plot(channel="rgb", lw=5)
    lines = plt.gca().get_lines()
    for channel, line in zip(("red", "green", "blue"), lines):
        xline, yline = line.get_xydata().T
        count = getattr(ps, f"{channel}_photon_count")
        np.testing.assert_allclose(xline, (count.timestamps - count.timestamps[0]) * 1e-9)
        np.testing.assert_allclose(yline, count.data)
    plt.close()


@cleanup
def test_deprecated_plotting(test_point_scans):
    ps = test_point_scans["PointScan1"]
    with pytest.deprecated_call():
        ps.plot_red()
    with pytest.deprecated_call():
        ps.plot_green()
    with pytest.deprecated_call():
        ps.plot_blue()
    with pytest.deprecated_call():
        ps.plot_rgb()
