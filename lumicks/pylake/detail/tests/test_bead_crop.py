import pathlib
from contextlib import nullcontext

import numpy as np
import pytest
import matplotlib.pyplot as plt

from lumicks.pylake.detail.bead_cropping import find_beads_template, find_beads_brightness


@pytest.mark.parametrize(
    "filename, ref_edges, ref_edges_no_negative",
    [
        ("kymo_sum0", [148, 354], [148, 354]),
        ("kymo_sum1", [144, 323], [144, 323]),
        ("kymo_sum2", [161, 349], [161, 349]),
        ("kymo_sum3", [181, 491], []),  # Negative beads
        ("kymo_sum4", [209, 489], []),  # Negative beads
        ("kymo_sum5", [56, 266], []),  # Negative beads
        ("kymo_sum6", [185, 519], []),  # Crowded kymo (dark beads)
        ("kymo_sum7", [188, 523], []),  # Crowded kymo (dark beads)
    ],
)
def test_bead_cropping_allow_negative_beads(filename, ref_edges, ref_edges_no_negative):
    data = np.load(pathlib.Path(__file__).parent / "data" / f"{filename}.npz")

    edges = find_beads_brightness(
        np.atleast_2d(data["green"]).T, bead_diameter_pixels=4.84 / data["pixelsize"], plot=False
    )
    # Asserting equal to pin behavior, but I wouldn't expect more accuracy than half a micron
    # because the fluorescence typically tails out wide.
    np.testing.assert_equal(edges, ref_edges)

    with nullcontext() if ref_edges_no_negative else pytest.raises(
        RuntimeError, match="Did not find two beads"
    ):
        edges = find_beads_brightness(
            np.atleast_2d(data["green"]).T,
            bead_diameter_pixels=4.84 / data["pixelsize"],
            allow_negative_beads=False,
        )
        np.testing.assert_equal(edges, ref_edges_no_negative)


def test_bead_cropping_failure():
    mock = np.zeros((100, 1))
    mock[50] = 1  # Standard deviation of the data should not be zero or the KDE estimate fails.
    with pytest.raises(RuntimeError, match="Did not find two beads"):
        find_beads_brightness(mock, bead_diameter_pixels=1, plot=True)


def test_plotting():
    data = np.load(pathlib.Path(__file__).parent / "data" / f"kymo_sum0.npz")
    for allow_negative in (False, True):
        edges = find_beads_brightness(
            np.atleast_2d(data["green"]).T,
            bead_diameter_pixels=4.84 / data["pixelsize"],
            plot=True,
            allow_negative_beads=allow_negative,
        )

        np.testing.assert_equal(plt.gca().lines[0].get_data()[1], data["green"])
        np.testing.assert_equal(plt.gca().lines[-2].get_data()[0], [edges[0], edges[0]])
        np.testing.assert_equal(plt.gca().lines[-1].get_data()[0], [edges[1], edges[1]])


@pytest.mark.parametrize(
    "bead_size_pixels, ref_values",
    [
        (5, (17, 37)),
        (7, (17, 37)),
        (8, (17, 36)),
        (10, (18, 35)),
        (11, (18, 35)),
        (12, (19, 34)),
    ],
)
def test_template_crop(bead_size_pixels, ref_values):
    peak1 = [6, 2, 3, 2, 3, 6, 9]
    peak2 = [5, 2, 3, 2, 12, 8, 2]
    image_line = [0] * 10 + peak1 + [0] * 20 + peak2 + [0] * 10
    im = np.tile(image_line, (3, 1)).T

    maxima = find_beads_template(im, bead_size_pixels, downsample_num_frames=10)
    np.testing.assert_equal(maxima, ref_values)


def test_template_crop_plot():
    peak1 = [6, 2, 3, 2, 3, 6, 9]
    peak2 = [5, 2, 3, 2, 12, 8, 2]
    image_line = [0] * 10 + peak1 + [0] * 20 + peak2 + [0] * 10
    im = np.tile(image_line, (3, 1)).T
    find_beads_template(im, 10, downsample_num_frames=10, plot=True, allow_movement=True)
    find_beads_template(im, 10, downsample_num_frames=10, plot=True, allow_movement=False)


def test_template_crop_errors():
    peak = [6, 2, 3, 2, 3, 6, 9]
    image_line = [0] * 10 + peak + [0] * 20

    with pytest.raises(RuntimeError, match="Did not find two beads"):
        find_beads_template(np.tile(image_line, (3, 1)).T, bead_diameter_pixels=5)

    with pytest.raises(ValueError, match="Kymograph image must be two dimensional"):
        find_beads_template(np.array([1, 2, 3]), bead_diameter_pixels=5)

    with pytest.raises(ValueError, match="Kymograph is too short to apply this method"):
        find_beads_template(np.tile(image_line, (2, 1)).T, bead_diameter_pixels=5)

    with pytest.raises(ValueError, match="Time axis needs to be divided in at least 3 sections"):
        find_beads_template(
            np.tile(image_line, (2, 1)).T, bead_diameter_pixels=5, downsample_num_frames=2
        )
