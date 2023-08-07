import re

import numpy as np
import pytest

from lumicks.pylake.adjustments import ColorAdjustment, colormaps, wavelength_to_xyz
from lumicks.pylake.detail.image import (
    histogram_rows,
    reconstruct_image,
    reconstruct_image_sum,
    reconstruct_num_frames,
    first_pixel_sample_indices,
)


def test_reconstruct():
    infowave = np.array([0, 1, 0, 1, 1, 0, 2, 1, 0, 1, 0, 0, 1, 2, 1, 1, 1, 2])
    the_data = np.array([1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3])

    image = reconstruct_image(the_data, infowave, (5,))
    assert image.shape == (1, 5)
    assert np.all(image == [4, 8, 12, 0, 0])

    image = reconstruct_image(the_data, infowave, (2,))
    assert image.shape == (2, 2)
    assert np.all(image == [[4, 8], [12, 0]])

    image = reconstruct_image_sum(the_data, infowave, (5,))
    assert image.shape == (1, 5)
    assert np.all(image == [4, 8, 12, 0, 0])

    image = reconstruct_image_sum(the_data, infowave, (2,))
    assert image.shape == (2, 2)
    assert np.all(image == [[4, 8], [12, 0]])


@pytest.mark.parametrize("reconstruction_func", [reconstruct_image_sum, reconstruct_image])
def test_unequal_length(reconstruction_func):
    with pytest.raises(
        ValueError, match=re.escape("Data size (3) must be the same as the infowave size (2)")
    ):
        reconstruction_func(np.array([1, 2, 3]), np.array([1, 1]), (2, 1))


def test_reconstruct_multiframe():
    size = 100
    infowave = np.ones(size)
    infowave[9::10] = 2
    the_data = np.arange(size)

    assert reconstruct_image(the_data, infowave, (5,)).shape == (2, 5)
    assert reconstruct_image(the_data, infowave, (2,)).shape == (5, 2)
    assert reconstruct_image(the_data, infowave, (1,)).shape == (10, 1)
    assert reconstruct_image(the_data, infowave, (2, 2)).shape == (3, 2, 2)
    assert reconstruct_image(the_data, infowave, (3, 2)).shape == (2, 3, 2)
    assert reconstruct_image(the_data, infowave, (5, 2)).shape == (1, 5, 2)

    assert reconstruct_image_sum(the_data, infowave, (5,)).shape == (2, 5)
    assert reconstruct_image_sum(the_data, infowave, (2,)).shape == (5, 2)
    assert reconstruct_image_sum(the_data, infowave, (1,)).shape == (10, 1)
    assert reconstruct_image_sum(the_data, infowave, (2, 2)).shape == (3, 2, 2)
    assert reconstruct_image_sum(the_data, infowave, (3, 2)).shape == (2, 3, 2)
    assert reconstruct_image_sum(the_data, infowave, (5, 2)).shape == (1, 5, 2)

    assert reconstruct_num_frames(infowave, 2, 2) == 3
    assert reconstruct_num_frames(infowave, 2, 3) == 2
    assert reconstruct_num_frames(infowave, 2, 5) == 1


def test_histogram_rows():
    data = np.arange(36).reshape((6, 6))

    e, h, w = histogram_rows(data, 1, 0.1)
    np.testing.assert_allclose(e, [0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
    assert np.all(np.equal(h, [15, 51, 87, 123, 159, 195]))
    np.testing.assert_allclose(w, 0.1)

    e, h, w = histogram_rows(data, 3, 0.1)
    np.testing.assert_allclose(e, [0.0, 0.3])
    assert np.all(np.equal(h, [153, 477]))
    np.testing.assert_allclose(w, 0.3)

    with pytest.warns(UserWarning):
        e, h, w = histogram_rows(data, 5, 0.1)
        np.testing.assert_allclose(e, [0.0, 0.5])
        assert np.all(np.equal(h, [435, 195]))
        np.testing.assert_allclose(w, [0.5, 0.1])


def test_partial_pixel_image_reconstruction():
    """This function tests whether a partial pixel at the end is fully dropped. This is important,
    since in the timestamp reconstruction, we subtract the minimum value prior to averaging (to
    allow taking averages of larger chunks). Without this functionality, the lowest timestamp to be
    reconstructed can be smaller than the first timestamp. This means that the value subtracted
    from the timestamps prior to summing is smaller. This means that the timestamps more quickly
    get into the range where they are at risk of overflow (and therefore have to be summed in
    smaller blocks). This in turn can lead to unnecessarily long reconstruction times."""

    def size_test(x, axis):
        assert len(x) == 4, "The last pixel should have been dropped since it was partial"
        return np.array([1, 1, 1, 1])

    iw = np.tile([1, 1, 1, 1, 2], (5,))
    iw[-1] = 0
    ts = np.arange(iw.size)
    reconstruct_image(ts, iw, (5, 1), reduce=size_test)


@pytest.mark.parametrize(
    "minimum, maximum, ref_minimum, ref_maximum",
    [
        (1.0, 5.0, [1.0, 1.0, 1.0], [5.0, 5.0, 5.0]),
        ([1.0, 2.0, 3.0], [3.0, 4.0, 5.0], [1.0, 2.0, 3.0], [3.0, 4.0, 5.0]),
        ([1.0], [3.0, 4.0, 5.0], [1.0, 1.0, 1.0], [3.0, 4.0, 5.0]),
        ([1.0, 2.0, 3.0], [5.0], [1.0, 2.0, 3.0], [5.0, 5.0, 5.0]),
        ([1.0, 2.0, 3.0], 5.0, [1.0, 2.0, 3.0], [5.0, 5.0, 5.0]),
    ],
)
def test_absolute_ranges(minimum, maximum, ref_minimum, ref_maximum):
    ref_img = np.array([np.reshape(np.arange(10), (2, 5)) for _ in np.arange(3)]).swapaxes(0, 2)

    ca = ColorAdjustment(minimum, maximum, mode="absolute")
    np.testing.assert_allclose(ca.minimum, ref_minimum)
    np.testing.assert_allclose(ca.maximum, ref_maximum)

    np.testing.assert_allclose(
        ca._get_data_rgb(ref_img), np.clip((ref_img - ca.minimum) / (ca.maximum - ca.minimum), 0, 1)
    )


@pytest.mark.parametrize(
    "minimum, maximum, gamma, ref_minimum, ref_maximum, ref_gamma",
    [
        # fmt: off
        (1.0, 5.0, [0.5, 2.0, 3.0], [1.0, 1.0, 1.0], [5.0, 5.0, 5.0], [0.5, 2.0, 3.0]),
        ([1.0, 2.0, 3.0], [3.0, 4.0, 5.0], 0.5, [1.0, 2.0, 3.0], [3.0, 4.0, 5.0], [0.5, 0.5, 0.5]),
        ([1.0], [3.0, 4.0, 5.0], [0.2, 1.0, 2.0], [1.0, 1.0, 1.0], [3.0, 4.0, 5.0], [0.2, 1.0, 2.0]),
        ([1.0, 2.0, 3.0], [5.0], None, [1.0, 2.0, 3.0], [5.0, 5.0, 5.0], [1, 1, 1]),  # none passed
        ([1.0, 2.0, 3.0], 5.0, 1, [1.0, 2.0, 3.0], [5.0, 5.0, 5.0], [1, 1, 1]),
        # fmt: on
    ],
)
def test_gamma(minimum, maximum, gamma, ref_minimum, ref_maximum, ref_gamma):
    ref_img = np.array([np.reshape(np.arange(10), (2, 5)) for _ in np.arange(3)]).swapaxes(0, 2)

    gamma_arg = {"gamma": gamma} if gamma is not None else {}  # Only pass gamma if we have one
    ca = ColorAdjustment(minimum, maximum, mode="absolute", **gamma_arg)
    np.testing.assert_allclose(ca.minimum, ref_minimum)
    np.testing.assert_allclose(ca.maximum, ref_maximum)
    np.testing.assert_allclose(ca.gamma, ref_gamma)

    expected = np.clip((ref_img - ca.minimum) / (ca.maximum - ca.minimum), 0, 1) ** ca.gamma
    np.testing.assert_allclose(ca._get_data_rgb(ref_img), expected)


@pytest.mark.parametrize(
    "minimum, maximum, ref_minimum, ref_maximum",
    [
        (25.0, 75.0, [25.0, 25.0, 25.0], [75.0, 75.0, 75.0]),
        ([25.0, 35.0, 45.0], [75.0, 85.0, 95.0], [25.0, 35.0, 45.0], [75.0, 85.0, 95.0]),
        ([25.0], [55.0, 65.0, 75.0], [25.0, 25.0, 25.0], [55.0, 65.0, 75.0]),
        ([25.0, 35.0, 45.0], [5.0], [25.0, 35.0, 45.0], [5.0, 5.0, 5.0]),
        ([25.0, 35.0, 45.0], 5.0, [25.0, 35.0, 45.0], [5.0, 5.0, 5.0]),
        (1.0, [3.0, 4.0, 5.0], [1.0, 1.0, 1.0], [3.0, 4.0, 5.0]),
    ],
)
def test_percentile_ranges(minimum, maximum, ref_minimum, ref_maximum):
    ref_img = np.array([np.reshape(np.arange(10), (2, 5)) for _ in np.arange(3)]).swapaxes(0, 2)

    ca = ColorAdjustment(minimum, maximum, mode="percentile")
    np.testing.assert_allclose(ca.minimum, ref_minimum)
    np.testing.assert_allclose(ca.maximum, ref_maximum)

    bounds = np.array(
        [
            np.percentile(img, [mini, maxi])
            for img, mini, maxi in zip(np.moveaxis(ref_img, 2, 0), ca.minimum, ca.maximum)
        ]
    )
    minimum, maximum = bounds.T

    np.testing.assert_allclose(
        ca._get_data_rgb(ref_img), np.clip((ref_img - minimum) / (maximum - minimum), 0, 1)
    )


def test_invalid_range():
    with pytest.raises(ValueError, match="Color value bounds and gamma should be of length 1 or 3"):
        ColorAdjustment([2.0, 3.0], [1.0, 2.0, 3.0], mode="absolute")

    with pytest.raises(ValueError, match="Mode must be percentile or absolute"):
        ColorAdjustment(1.0, 1.0, mode="spaghetti")


def test_no_adjust():
    ref_img = np.array([np.reshape(np.arange(10), (2, 5)) for _ in np.arange(3)]).swapaxes(0, 2)

    ca = ColorAdjustment.nothing()
    np.testing.assert_allclose(ca._get_data_rgb(ref_img), ref_img / ref_img.max())


@pytest.mark.parametrize(
    "wavelength, ref_xyz",
    [
        (300, [2.63637746e-14, 6.25334786e-08, 4.96638266e-09]),
        (488, [0.04306751, 0.19571404, 0.52012862]),
        (590, [1.02103920e00, 7.62586723e-01, 1.43479704e-04]),
        (650, [2.83631873e-01, 1.10045343e-01, 2.96115850e-08]),
        (850, [6.94669991e-15, 2.74627747e-11, 2.88661210e-29]),
    ],
)
def test_wavelength_to_xyz(wavelength, ref_xyz):
    xyz = wavelength_to_xyz(wavelength)
    np.testing.assert_allclose(xyz, ref_xyz)


@pytest.mark.parametrize(
    "wavelength, ref",
    [
        # fmt: off
        (300, [[0, 0, 0, 1], [0, 7.62146890e-07, 0, 1], [0, 1.51833951e-06, 0, 1]]),
        (488, [[0, 0, 0, 1], [0, 0.31312012, 0.3731908,  1], [0, 0.623794, 0.74346605, 1]]),
        (590, [[0, 0, 0, 1], [0.50196078, 0.34888505, 0, 1], [1, 0.69504443, 0, 1]]),
        (650, [[0, 0, 0, 1], [0.44212589, 0, 0, 1], [0.88079768,  0, 0, 1]]),
        (850, [[0, 0, 0, 1], [0, 3.34079999e-10, 0, 1], [0, 6.65549997e-10, 0, 1]]),
        # fmt: on
    ],
)
def test_wavelength_to_cmap(wavelength, ref):
    cmap = colormaps.from_wavelength(wavelength)
    np.testing.assert_allclose(cmap([0, 0.5, 1]), ref)


@pytest.mark.parametrize(
    "data, ref_start, ref_stop",
    [
        ([1, 1, 2], 0, 2),
        ([0, 0, 1, 1, 2], 2, 4),
        ([2, 2, 2, 2], 0, 0),
        ([0, 2, 2, 2, 2], 1, 1),
        ([0, 1, 2, 1, 2], 1, 2),
        ([], 0, 0),
    ],
)
def test_first_pixel_sample_indices(data, ref_start, ref_stop):
    start, stop = first_pixel_sample_indices(np.asarray(data))
    assert start == ref_start
    assert stop == ref_stop
