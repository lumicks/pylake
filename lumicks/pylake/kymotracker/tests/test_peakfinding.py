import re

import numpy as np
import pytest

from lumicks.pylake.kymotracker.detail.peakfinding import (
    KymoPeaks,
    peak_estimate,
    merge_close_peaks,
    unbiased_centroid,
    _clip_kernel_to_edge,
    bounds_to_centroid_data,
    refine_peak_based_on_moment,
)


@pytest.mark.parametrize("location", [12.3, 12.7, 11.7, 11.49, 11.51])
def test_peak_estimation(location):
    x = np.arange(25)
    data = np.tile(10000 * np.exp(-((x - location) ** 2)), (1, 1)).T

    position, time = peak_estimate(data, 7, thresh=0.4)
    assert position[0] == round(location)

    # Deliberately mis-shift the initial guess
    position = position + 5
    for input_dtype in (int, float):
        peaks = KymoPeaks(*refine_peak_based_on_moment(data.astype(input_dtype), position, time, 4))
        assert np.abs(peaks.frames[0].coordinates[0] - location) < 1e-3


def test_invalid_peak_construction():
    for coordinates, amplitudes in (([3, 4], [3, 4, 5]), ([3, 4, 5], [3, 4])):
        with pytest.raises(
            ValueError,
            match=re.escape(
                f"Number of time points (3), coordinates ({len(coordinates)}) and peak amplitudes "
                f"({len(amplitudes)}) must be equal"
            ),
        ):
            KymoPeaks(coordinates, np.array([1, 2, 3]), amplitudes)


def test_peak_refinement_input_validation():
    """When the kernel size is zero, then centroid refinement simply does nothing. Since this could
    lead to unexpected results (i.e. no subpixel accuracy), we throw."""
    data = np.tile(np.exp(-((np.arange(25) - 12.3) ** 2)), (1, 1)).T
    position, time = np.array([12]), np.array([0])
    with pytest.raises(ValueError, match="half_kernel_size may not be smaller than 1"):
        KymoPeaks(*refine_peak_based_on_moment(data, position, time, half_kernel_size=0))

    KymoPeaks(*refine_peak_based_on_moment(data, position, time, half_kernel_size=1))  # should work


def test_regression_peak_estimation():
    # This test tests a regression where a peak could be found adjacent to a very bright structure.
    # The error originated from the blurring used to get rid of pixelation noise being applied
    # in two directions rather than only one.
    data = np.array([[0, 0, 0, 0, 0], [255, 255, 255, 0, 0], [0, 0, 0, 0, 0]])

    position, time = peak_estimate(data, 1, thresh=10)
    assert len(position) == 3


def test_minimum_threshold_peakfinding():
    data = np.full((5, 5), fill_value=30)
    with pytest.raises(
        RuntimeError,
        match=re.escape(
            "Threshold (30) cannot be lower than or equal to the lowest filtered pixel (30)"
        ),
    ):
        _ = peak_estimate(data, 1, thresh=30)

    _, _ = peak_estimate(data, 1, thresh=30.1)


def test_kymopeaks():
    # First time frame we choose the right one first, then the second one. Second time frame vice
    # versa.
    coordinates = np.array([3.2, 4.1, 6.4, 8.2])
    time_points = np.array([0.0, 1.0, 0.5, 1.0])
    peak_amplitudes = np.array([2.0, 3.0, 3.0, 2.0])
    peaks = KymoPeaks(coordinates, time_points, peak_amplitudes)
    np.testing.assert_allclose(peaks.frames[0].coordinates, [3.2, 6.4])
    np.testing.assert_allclose(peaks.frames[1].coordinates, [4.1, 8.2])
    np.testing.assert_allclose(peaks.frames[0].time_points, [0.0, 0.5])
    np.testing.assert_allclose(peaks.frames[1].time_points, [1.0, 1.0])
    np.testing.assert_allclose(peaks.frames[0].peak_amplitudes, [2.0, 3.0])
    np.testing.assert_allclose(peaks.frames[1].peak_amplitudes, [3.0, 2.0])
    assert str(peaks) == "KymoPeaks(N=2)"

    # Indexing
    np.testing.assert_allclose(peaks[0].coordinates, [3.2, 6.4])
    np.testing.assert_allclose(peaks[1].coordinates, [4.1, 8.2])

    assert not KymoPeaks([], [], [])


def test_invalid_indexing():
    peaks = KymoPeaks([1, 2], [1, 2], [1, 2])

    with pytest.raises(IndexError, match="Only integer indexing is allowed"):
        peaks[0:2]


def test_peak_proximity_removal():
    # First time frame we choose the right one first, then the second one. Second time frame vice
    # versa.
    coordinates = np.array([3.2, 4.1, 6.4, 8.2, 12.1, 12.2, 3.2, 4.1, 6.4, 8.2, 12.1, 12.2])
    time_points = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    peak_amplitudes = np.array([2.0, 3.0, 3.0, 2.0, 3.0, 2.0, 3.0, 2.0, 0.0, 0.0, 2.0, 3.0])
    peaks = KymoPeaks(coordinates, time_points, peak_amplitudes)

    new_peaks = merge_close_peaks(peaks, 1.0)
    np.testing.assert_allclose(new_peaks.frames[0].coordinates, [4.1, 6.4, 8.2, 12.1])
    np.testing.assert_allclose(new_peaks.frames[1].coordinates, [3.2, 6.4, 8.2, 12.2])

    permute = [1, 2, 9, 5, 3, 11, 4, 8, 7, 0, 6, 10]
    coordinates = coordinates[permute]
    time_points = time_points[permute]
    peak_amplitudes = peak_amplitudes[permute]
    peaks = KymoPeaks(coordinates, time_points, peak_amplitudes)

    new_peaks = merge_close_peaks(peaks, 1.0)
    assert set(new_peaks.frames[0].coordinates) == {4.1, 6.4, 8.2, 12.1}
    assert set(new_peaks.frames[1].coordinates) == {3.2, 6.4, 8.2, 12.2}


@pytest.mark.parametrize(
    "bounds, selection_ref, center_ref, weights_ref",
    [
        # fmt:off
        (
            ((0, 0.5, 0.25, 1), (4, 4, 4, 4)),  # left bounds, right bounds
            np.array([[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [0, 1, 1, 1]]),  # selection mask
            np.array([[0.5, 1.5, 2.5, 3.5], [0.75, 1.5, 2.5, 3.5], [0.625, 1.5, 2.5, 3.5], [0, 1.5, 2.5, 3.5]]),  # centers
            np.array([[1, 1, 1, 1], [0.5, 1, 1, 1], [0.75, 1, 1, 1], [0, 1, 1, 1]]),  # weights
        ),
        (
            ((1.25, 0, 0, 0, 1.25), (4, 3, 2.5, 2.25, 3.75)),  # left bounds, right bounds
            np.array([[0, 1, 1, 1], [1, 1, 1, 0], [1, 1, 1, 0], [1, 1, 1, 0], [0, 1, 1, 1]]),  # selection mask
            np.array([[0, 1.625, 2.5, 3.5], [0.5, 1.5, 2.5, 0], [0.5, 1.5, 2.25, 0], [0.5, 1.5, 2.125, 0], [0, 1.625, 2.5, 3.375]]),  # centers
            np.array([[0, 0.75, 1, 1], [1, 1, 1, 0], [1, 1, 0.5, 0], [1, 1, 0.25, 0], [0, 0.75, 1, 0.75]])  # weights
        )
        # fmt:on
    ],
)
def test_bounds_to_centroid_data(bounds, selection_ref, center_ref, weights_ref):
    index_array = np.tile(np.arange(4), (len(bounds[0]), 1))
    result = bounds_to_centroid_data(index_array, *bounds)
    np.testing.assert_equal(result[0], selection_ref)
    np.testing.assert_equal(result[1], center_ref)
    np.testing.assert_equal(result[2], weights_ref)


@pytest.mark.parametrize(
    "data, ref_estimate",
    # fmt:off
    [
        (np.array([[0, 0, 3, 3, 3, 0, 0], [0, 0, 3, 3, 3, 3, 0]]), [3, 3.5]),  # No baseline (regular centroid would do fine)
        (np.array([[2, 2, 2, 2, 2, 2, 2], [2, 2, 3, 3, 3, 2, 2]]), [3, 3]),
        (np.array([[2, 2, 3, 3, 3, 3, 2], [2, 2, 2, 2, 2, 2, 2]]), [3.497509, 3]),  # Should be 3.5, 3
        (np.array([[2, 2, 3, 3, 3, 3, 2], [0, 0, 0, 0, 0, 0, 0]]), [3.497509, 3]),  # Should be 3.5, 3
        (np.array([[0, 0, 3, 3, 3, 0, 0], [0, 0, 3, 3, 3, 3, 0]]), [3, 3.5]),  # No baseline (regular centroid would do fine)
        (np.array([[2, 2, 2, 2, 2, 2, 2], [2, 2, 3, 3, 3, 2, 2]]), [3, 3]),
        (np.array([[2, 2, 3, 3, 3, 3, 2], [2, 2, 2, 2, 2, 2, 2]]), [3.497509, 3]),  # Should be 3.5, 3
        (np.array([[2, 2, 3, 3, 3, 3, 2], [0, 0, 0, 0, 0, 0, 0]]), [3.497509, 3]),  # Should be 3.5, 3
    ],
    # fmt:on
)
def test_unbiased_centroid_estimator(data, ref_estimate):
    for input_dtype in (int, float):
        np.testing.assert_allclose(
            unbiased_centroid(np.array((3.5, 3.5)), data.astype(input_dtype)), ref_estimate
        )


@pytest.mark.parametrize(
    "coords, data_shape, half_width, ref",
    [
        (np.arange(0, 11), 11, 3, np.array([0, 1, 2, 3, 3, 3, 3, 3, 2, 1, 0])),
        (np.arange(0, 11), 11, 4, np.array([0, 1, 2, 3, 4, 4, 4, 3, 2, 1, 0])),
        (np.arange(1, 12), 12, 3, np.array([1, 2, 3, 3, 3, 3, 3, 3, 2, 1, 0])),
        (np.arange(1, 12), 12, 4, np.array([1, 2, 3, 4, 4, 4, 4, 3, 2, 1, 0])),
        (np.array([3, 4, 8, 9, 28, 29]), 30, 4, np.array([3, 4, 4, 4, 1, 0])),
    ],
)
def test_clip_halfwidth(coords, data_shape, half_width, ref):
    np.testing.assert_allclose(_clip_kernel_to_edge(half_width, coords, data_shape), ref)


@pytest.mark.parametrize(
    "data, adjacency_half_width, ref_positions",
    [
        ([[255, 0, 0, 0, 0], [0, 0, 255, 0, 0], [0, 0, 0, 0, 0]], None, 2),  # No filter
        ([[255, 0, 0, 0, 0], [0, 0, 255, 0, 0], [0, 0, 0, 0, 0]], 1, 0),  # second peak out
        ([[255, 0, 0, 0, 0], [0, 0, 255, 0, 0], [0, 0, 0, 0, 0]], 2, 2),  # second peak in
        ([[255, 0, 0, 0, 0], [0, 0, 255, 0, 0], [0, 0, 0, 255, 0]], 1, 2),  # second peak out
        ([[255, 0, 0, 0, 0], [0, 0, 255, 0, 0], [0, 0, 0, 0, 255]], 1, 0),  # both out
        ([[255, 0, 0, 0, 0], [0, 0, 255, 0, 0], [0, 0, 0, 0, 255]], 2, 3),  # all in
    ],
)
def test_adjacency_filter(data, adjacency_half_width, ref_positions):
    position, time = peak_estimate(
        np.array(data).T, 0, thresh=128, adjacency_half_width=adjacency_half_width
    )
    assert len(position) == ref_positions
