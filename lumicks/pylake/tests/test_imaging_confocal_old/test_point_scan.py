import pytest
import matplotlib.pyplot as plt
import numpy as np
import re


def test_point_scans(test_point_scans, reference_timestamps, reference_counts):
    ps = test_point_scans["PointScan1"]
    ps_red = ps.red_photon_count

    assert ps_red.data.shape == (64,)

    np.testing.assert_allclose(ps_red.timestamps, reference_timestamps)
    np.testing.assert_allclose(ps_red.data, reference_counts)


def test_plotting(test_point_scans):
    ps = test_point_scans["PointScan1"]

    for channel in ("red", "green", "blue"):
        xline, yline = ps.plot(channel=channel)[0].get_xydata().T

        count = getattr(ps, f"{channel}_photon_count")
        np.testing.assert_allclose(xline, (count.timestamps - count.timestamps[0]) * 1e-9)
        np.testing.assert_allclose(yline, count.data)
        plt.close()

    lines = ps.plot(channel="rgb", lw=5)
    for channel, line in zip(("red", "green", "blue"), lines):
        xline, yline = line.get_xydata().T
        count = getattr(ps, f"{channel}_photon_count")
        np.testing.assert_allclose(xline, (count.timestamps - count.timestamps[0]) * 1e-9)
        np.testing.assert_allclose(yline, count.data)
    plt.close()


def test_deprecated_plotting(test_point_scans):
    ps = test_point_scans["PointScan1"]
    with pytest.raises(
        TypeError,
        match=re.escape(
            "plot() takes from 1 to 2 positional arguments but 3 were given"
        )
    ):
        xline, yline = ps.plot("red", None)[0].get_xydata().T

        count = getattr(ps, "red_photon_count")
        np.testing.assert_allclose(xline, (count.timestamps - count.timestamps[0]) * 1e-9)
        np.testing.assert_allclose(yline, count.data)
        plt.close()
    # Test rejection of deprecated call with positional `axes` and double keyword assignment
    with pytest.raises(
        TypeError,
        match=re.escape(
            "plot() takes from 1 to 2 positional arguments but 3 positional "
            "arguments (and 1 keyword-only argument) were given"
        )
    ):
        ps.plot("rgb", None, axes=None)
