from lumicks.pylake.kymotracker.detail.peakfinding import peak_estimate, refine_peak_based_on_moment
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
