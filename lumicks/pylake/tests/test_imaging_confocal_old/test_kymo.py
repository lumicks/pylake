import re

import numpy as np
import pytest
import matplotlib.pyplot as plt

from lumicks import pylake
from lumicks.pylake.kymo import EmptyKymo, _default_line_time_factory
from lumicks.pylake.channel import Slice, Continuous, TimeSeries, empty_slice
from lumicks.pylake.adjustments import ColorAdjustment

from ..data.mock_confocal import generate_kymo


def with_offset(t, start_time=1592916040906356300):
    return np.array(t, dtype=np.int64) + start_time


def test_export_tiff(tmp_path, test_kymos, grab_tiff_tags):
    from os import stat

    kymo = test_kymos["Kymo1"]
    kymo.export_tiff(tmp_path / "kymo1.tiff")
    assert stat(tmp_path / "kymo1.tiff").st_size > 0

    # Check if tags were properly stored, i.e. test functionality of `_tiff_image_metadata()`,
    # `_tiff_timestamp_ranges()` and `_tiff_writer_kwargs()`
    tiff_tags = grab_tiff_tags(tmp_path / "kymo1.tiff")
    assert len(tiff_tags) == 1
    for tags, timestamp_range in zip(tiff_tags, kymo._tiff_timestamp_ranges()):
        assert tags["ImageDescription"] == kymo._tiff_image_metadata()
        assert tags["DateTime"] == f"{timestamp_range[0]}:{timestamp_range[1]}"
        assert tags["Software"] == kymo._tiff_writer_kwargs()["software"]
        np.testing.assert_allclose(
            tags["XResolution"][0] / tags["XResolution"][1],
            kymo._tiff_writer_kwargs()["resolution"][0],
            rtol=1e-1,
        )
        np.testing.assert_allclose(
            tags["YResolution"][0] / tags["YResolution"][1],
            kymo._tiff_writer_kwargs()["resolution"][1],
            rtol=1e-1,
        )
        assert tags["ResolutionUnit"] == 3  # 3 = Centimeter


def test_kymo_plot_rgb_absolute_color_adjustment(test_kymos):
    """Tests whether we can set an absolute color range for the RGB plot."""
    kymo = test_kymos["Kymo1"]

    fig = plt.figure()
    lb, ub = np.array([1, 2, 3]), np.array([2, 3, 4])
    kymo.plot(channel="rgb", adjustment=ColorAdjustment(lb, ub, mode="absolute"))
    image = plt.gca().get_images()[0]
    np.testing.assert_allclose(
        image.get_array(), np.clip((kymo.get_image("rgb") - lb) / (ub - lb), 0, 1)
    )
    plt.close(fig)


def test_kymo_plot_rgb_percentile_color_adjustment(test_kymos):
    """Tests whether we can set a percentile color range for the RGB plot."""
    kymo = test_kymos["Kymo1"]

    fig = plt.figure()
    lb, ub = np.array([10, 10, 10]), np.array([80, 80, 80])
    kymo.plot(channel="rgb", adjustment=ColorAdjustment(lb, ub, mode="percentile"))
    image = plt.gca().get_images()[0]
    bounds = np.array(
        [
            np.percentile(img, [mini, maxi])
            for img, mini, maxi in zip(np.moveaxis(kymo.get_image("rgb"), 2, 0), lb, ub)
        ]
    )
    lb, ub = (b for b in np.moveaxis(bounds, 1, 0))
    np.testing.assert_allclose(
        image.get_array(), np.clip((kymo.get_image("rgb") - lb) / (ub - lb), 0, 1)
    )
    plt.close(fig)


def test_kymo_plot_single_channel_absolute_color_adjustment(test_kymos):
    """Tests whether we can set an absolute color range for a single channel plot."""
    kymo = test_kymos["Kymo1"]

    lbs, ubs = np.array([1, 2, 3]), np.array([2, 3, 4])
    for lb, ub, channel in zip(lbs, ubs, ("red", "green", "blue")):
        # Test whether setting RGB values and then sampling one of them works correctly.
        fig = plt.figure()
        kymo.plot(channel=channel, adjustment=ColorAdjustment(lbs, ubs, mode="absolute"))
        image = plt.gca().get_images()[0]
        np.testing.assert_allclose(image.get_array(), kymo.get_image(channel))
        np.testing.assert_allclose(image.get_clim(), [lb, ub])
        plt.close(fig)

        # Test whether setting a single color works correctly (should use the same for R G and B).
        fig = plt.figure()
        kymo.plot(channel=channel, adjustment=ColorAdjustment(lb, ub, mode="absolute"))
        image = plt.gca().get_images()[0]
        np.testing.assert_allclose(image.get_array(), kymo.get_image(channel))
        np.testing.assert_allclose(image.get_clim(), [lb, ub])
        plt.close(fig)
