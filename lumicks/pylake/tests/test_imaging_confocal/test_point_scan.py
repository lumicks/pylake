import matplotlib.pyplot as plt
import numpy as np
import pytest

from lumicks.pylake.point_scan import PointScan
from ..data.mock_confocal import MockConfocalFile, generate_image_data

start = np.int64(20e9)
dt = np.int64(62.5e6)
reference_image = np.random.poisson(10, (64, 1))
reference_infowave, counts = generate_image_data(reference_image, 5, 3)


@pytest.fixture(scope="module")
def reference_counts():
    return counts[0]


@pytest.fixture(scope="module")
def reference_timestamps():
    stop = start + len(reference_infowave) * dt
    return np.arange(start, stop, 6.25e7).astype(np.int64)


@pytest.fixture(scope="module")
def test_point_scans(reference_counts):
    point_scans = {}

    mock_file, metadata, stop = MockConfocalFile.from_streams(
        start,
        dt,
        [],
        [],
        [],
        infowave=reference_infowave,
        red_photon_counts=reference_counts,
        green_photon_counts=reference_counts,
        blue_photon_counts=reference_counts,
    )
    point_scans["PointScan1"] = PointScan("PointScan1", mock_file, start, stop, metadata)

    return point_scans


def test_point_scan_attrs(test_point_scans, reference_timestamps, reference_counts):
    ps = test_point_scans["PointScan1"]
    ps_red = ps.red_photon_count

    assert ps_red.data.shape == (64 * 5 + 2 * 3,)

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
        "keyword argument.",
    ):
        xline, yline = ps.plot("red", None)[0].get_xydata().T

        count = getattr(ps, "red_photon_count")
        np.testing.assert_allclose(xline, (count.timestamps - count.timestamps[0]) * 1e-9)
        np.testing.assert_allclose(yline, count.data)
        plt.close()
    # Test rejection of deprecated call with positional `axes` and double keyword assignment
    with pytest.raises(
        TypeError, match=r"`PointScan.plot\(\)` got multiple values for argument `axes`"
    ):
        ps.plot("rgb", None, axes=None)
