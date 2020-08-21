from lumicks.pylake.kymotracker.detail.peakfinding import peak_estimate, refine_peak_based_on_moment, KymoPeaks, \
    merge_close_peaks
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
    peaks = refine_peak_based_on_moment(data, position, time, 4)
    assert np.abs(peaks.frames[0].coordinates[0] - location) < 1e-3


def test_kymopeaks():
    # First time frame we choose the right one first, then the second one. Second time frame vice versa.
    coordinates = np.array([3.2, 4.1, 6.4, 8.2])
    time_points = np.array([0.0, 1.0, 0.5, 1.0])
    peak_amplitudes = np.array([2.0, 3.0, 3.0, 2.0])
    peaks = KymoPeaks(coordinates, time_points, peak_amplitudes)
    assert np.allclose(peaks.frames[0].coordinates, [3.2, 6.4])
    assert np.allclose(peaks.frames[1].coordinates, [4.1, 8.2])
    assert np.allclose(peaks.frames[0].time_points, [0.0, 0.5])
    assert np.allclose(peaks.frames[1].time_points, [1.0, 1.0])
    assert np.allclose(peaks.frames[0].peak_amplitudes, [2.0, 3.0])
    assert np.allclose(peaks.frames[1].peak_amplitudes, [3.0, 2.0])


def test_peak_proximity_removal():
    # First time frame we choose the right one first, then the second one. Second time frame vice versa.
    coordinates = np.array([3.2, 4.1, 6.4, 8.2, 12.1, 12.2, 3.2, 4.1, 6.4, 8.2, 12.1, 12.2])
    time_points = np.array([0.0, 0.0, 0.0, 0.0,  0.0, 0.0,  1.0, 1.0, 1.0, 1.0, 1.0,  1.0])
    peak_amplitudes = np.array([2.0, 3.0, 3.0, 2.0,  3.0, 2.0,  3.0, 2.0, 0.0, 0.0, 2.0,  3.0])
    peaks = KymoPeaks(coordinates, time_points, peak_amplitudes)

    new_peaks = merge_close_peaks(peaks, 1.0)
    assert np.allclose(new_peaks.frames[0].coordinates, [4.1, 6.4, 8.2, 12.1])
    assert np.allclose(new_peaks.frames[1].coordinates, [3.2, 6.4, 8.2, 12.2])

    permute = [1, 2, 9, 5, 3, 11, 4, 8, 7, 0, 6, 10]
    coordinates = coordinates[permute]
    time_points = time_points[permute]
    peak_amplitudes = peak_amplitudes[permute]
    peaks = KymoPeaks(coordinates, time_points, peak_amplitudes)

    new_peaks = merge_close_peaks(peaks, 1.0)
    assert set(new_peaks.frames[0].coordinates) == {4.1, 6.4, 8.2, 12.1}
    assert set(new_peaks.frames[1].coordinates) == {3.2, 6.4, 8.2, 12.2}
