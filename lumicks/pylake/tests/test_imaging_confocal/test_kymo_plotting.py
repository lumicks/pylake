import re

import numpy as np
import pytest
import matplotlib.pyplot as plt


def test_plotting(test_kymo):
    kymo, ref = test_kymo
    line_time = ref.timestamps.line_time_seconds
    n_lines = ref.metadata.lines_per_frame
    n_pixels = ref.metadata.pixels_per_line
    pixel_size = ref.metadata.pixelsize_um[0]

    plt.figure()
    kymo.plot(channel="red")

    # todo: this is confusing even in the context of the old test, check on this
    # # The following assertion fails because of unequal line times in the test data. These
    # # unequal line times are not typical for BL data. Kymo nowadays assumes equal line times
    # # which is why the old version of this test fails.
    np.testing.assert_allclose(
        np.sort(plt.xlim()),
        [-0.5 * line_time, (n_lines - 0.5) * line_time],
        atol=0.05,
    )

    image = plt.gca().get_images()[0]
    np.testing.assert_allclose(image.get_array(), ref.image[:, :, 0])
    np.testing.assert_allclose(
        image.get_extent(),
        [
            -0.5 * line_time,
            (n_lines - 0.5) * line_time,
            (n_pixels * pixel_size - (pixel_size / 2)),
            -(pixel_size / 2),
        ],
    )

    # test original kymo is labeled with microns and
    # that kymo calibrated with base pairs has appropriate label
    assert plt.gca().get_ylabel() == r"position (μm)"
    plt.close()

    kymo_bp = kymo.calibrate_to_kbp(10.000)
    kymo_bp.plot(channel="red")
    assert plt.gca().get_ylabel() == "position (kbp)"
    plt.close()


def test_deprecated_plotting(test_kymo):
    kymo, ref = test_kymo

    with pytest.raises(
        TypeError, match=re.escape("plot() takes from 1 to 2 positional arguments but 3 were given")
    ):
        ih = kymo.plot("red", None)
        np.testing.assert_allclose(ih.get_array(), kymo.get_image("red"))
        plt.close()

    # Test rejection of deprecated call with positional `axes` and double keyword assignment
    with pytest.raises(
        TypeError,
        match=re.escape(
            "plot() takes from 1 to 2 positional arguments but 3 positional"
            " arguments (and 1 keyword-only argument) were given"
        ),
    ):
        kymo.plot("rgb", None, axes=None)
