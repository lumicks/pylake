import pytest
import matplotlib.pyplot as plt
import numpy as np


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
    with pytest.deprecated_call():
        ps.plot_red()
    with pytest.deprecated_call():
        ps.plot_green()
    with pytest.deprecated_call():
        ps.plot_blue()
    with pytest.deprecated_call():
        ps.plot_rgb()
    with pytest.warns(
        DeprecationWarning,
        match=r"The call signature of `plot\(\)` has changed: Please, provide `axes` as a "
        "keyword argument."
    ):
        xline, yline = ps.plot("red", None)[0].get_xydata().T

        count = getattr(ps, "red_photon_count")
        np.testing.assert_allclose(xline, (count.timestamps - count.timestamps[0]) * 1e-9)
        np.testing.assert_allclose(yline, count.data)
        plt.close()
    # Test rejection of deprecated call with positional `axes` and double keyword assignment
    with pytest.raises(
        TypeError,
        match=r"`PointScan.plot\(\)` got multiple values for argument `axes`"
    ):
        ps.plot("rgb", None, axes=None)
