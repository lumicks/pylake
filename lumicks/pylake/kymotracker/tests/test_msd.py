from lumicks.pylake.kymotracker.detail.msd_estimation import *
import pytest


@pytest.mark.parametrize(
    "time,position,max_lag,lag,msd",
    [
        (np.arange(25), np.arange(25) * 2, 3, [1, 2, 3], [4.0, 16.0, 36.0]),
        (np.arange(25), np.arange(25) * 2, 1000, np.arange(1, 25), (np.arange(1, 25) * 2) ** 2),
        (np.arange(25), -np.arange(25) * 2, 3, [1, 2, 3], [4.0, 16.0, 36.0]),
        (np.arange(25), np.arange(25) * 3, 3, [1, 2, 3], [9.0, 36.0, 81.0]),
        (np.arange(25), np.arange(25) * 3, 2, [1, 2], [9.0, 36.0]),
        ([1, 3, 4], [0, 6, 9], 3, [1, 2, 3], [9.0, 36.0, 81.0]),
        ([1, 4, 6], [0, 9, 15], 3, [2, 3, 5], [36.0, 81.0, 225.0]),
    ],
)
def test_msd_estimation(time, position, max_lag, lag, msd):
    lag_est, msd_est = calculate_msd(time, position, max_lag)
    assert np.allclose(lag_est, lag)
    assert np.allclose(msd_est, msd)
