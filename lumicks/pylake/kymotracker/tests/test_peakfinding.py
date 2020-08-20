from lumicks.pylake.kymotracker.detail.peakfinding import peak_estimate, refine_peak_based_on_moment, merge_close_peaks
import pytest
import numpy as np


@pytest.mark.parametrize("location", [12.3, 12.7, 11.7, 11.49, 11.51])
def test_peak_estimation(location):
    x = np.arange(25)
    data = np.tile(np.exp(-(x - location) ** 2), (1, 1)).T

    position, time = peak_estimate(data, 7, thresh=.4)
    assert position[0] == round(location)

    # Deliberately mis-shift the initial guess
    position = position + 5
    position2, time2, peak_amp = refine_peak_based_on_moment(data, position, time, 4)
    assert np.abs(position2[0] - location) < 1e-3


def test_peak_proximity_removal():
    # First time frame we choose the right one first, then the second one. Second time frame vice versa.
    coordinates = np.array([3.2, 4.1, 6.4, 8.2, 12.1, 12.2, 3.2, 4.1, 6.4, 8.2, 12.1, 12.2])
    time_points = np.array([0.0, 0.0, 0.0, 0.0,  0.0, 0.0,  1.0, 1.0, 1.0, 1.0, 1.0,  1.0])
    peak_amplitudes = np.array([2.0, 3.0, 3.0, 2.0,  3.0, 2.0,  3.0, 2.0, 0.0, 0.0, 2.0,  3.0])

    selection_mask = np.array([False, True, True, True, True, False, True, False, True, True, False, True])
    new_coords, new_time_points, new_peaks = merge_close_peaks(coordinates, time_points, peak_amplitudes, 1.0)

    assert np.allclose(new_coords, coordinates[selection_mask])
    assert np.allclose(new_time_points, time_points[selection_mask])
    assert np.allclose(new_peaks, peak_amplitudes[selection_mask])

    with pytest.raises(AssertionError):
        merge_close_peaks(np.array([1,2,3]), np.array([1,2]), np.array([1,2,3]), 1)

    with pytest.raises(AssertionError):
        merge_close_peaks(np.array([1, 2, 3]), np.array([1, 2, 3]), np.array([1, 2]), 1)
