import re

import numpy as np
import pytest
import matplotlib.pyplot as plt


def test_point_scans_basic(test_point_scan):
    ps, ref = test_point_scan

    for color, ref_data in ref["data"].items():
        count = getattr(ps, f"{color}_photon_count")

        assert count.data.shape == ref_data.shape

        np.testing.assert_equal(count.timestamps, ref["timestamps"])
        np.testing.assert_equal(count.data, ref_data)


def test_point_scan_slicing(test_point_scan):
    ps, ref = test_point_scan
    dt = ref["dt"]

    for crop in (
        (ps.start, ps.stop),  # No crop
        (ps.start + 10 * dt, ps.stop),
        (ps.start, ps.stop - 10 * dt),
        (None, ps.stop - 10 * dt),
        (ps.start + 10 * dt, None),
        ("0s", "10s"),
        ("0s", "3.5s"),
        ("3.5s", "10s"),
        ("3.5s", "6s"),
    ):
        p_sliced_red = ps[crop[0] : crop[1]].red_photon_count
        ps_red = ps.red_photon_count[crop[0] : crop[1]]
        np.testing.assert_equal(p_sliced_red.timestamps, ps_red.timestamps)
        np.testing.assert_equal(p_sliced_red.data, ps_red.data)

    with pytest.raises(IndexError, match="Scalar indexing is not supported, only slicing"):
        ps[5]

    with pytest.raises(IndexError, match="Slice steps are not supported"):
        ps["0s":"10s":"1s"]


def test_plotting(test_point_scan):
    ps, _ = test_point_scan

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


def test_deprecated_plotting(test_point_scan):
    ps, _ = test_point_scan
    with pytest.raises(
        TypeError, match=re.escape("plot() takes from 1 to 2 positional arguments but 3 were given")
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
        ),
    ):
        ps.plot("rgb", None, axes=None)
