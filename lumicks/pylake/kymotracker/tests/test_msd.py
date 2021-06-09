from lumicks.pylake.kymotracker.detail.msd_estimation import *
import contextlib
import pytest


@contextlib.contextmanager
def temp_seed(seed):
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.seed(None)


def simulate_diffusion_1d(diffusion, steps, dt, observation_noise):
    """Simulate from a Wiener process

    Parameters
    ----------
    diffusion : float
        Diffusion constant.
    steps : int
        Number of steps to simulate.
    dt : float
        Time step.
    observation_noise : float
        Standard deviation of the observation noise.
    """

    def simulate_wiener(sigma, num_steps, time_step):
        return np.cumsum(np.random.normal(0, sigma * np.sqrt(time_step), size=(num_steps,)))

    return simulate_wiener(np.sqrt(2.0 * diffusion), steps, dt) + np.random.normal(
        0, observation_noise, (steps,)
    )


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
    np.testing.assert_allclose(lag_est, lag)
    np.testing.assert_allclose(msd_est, msd)


@pytest.mark.parametrize(
    "frame_idx, coordinate, time_step, max_lag, diffusion_const",
    [
        (np.array([1, 2, 3, 4, 5]), np.array([-1.0, 1.0, -1.0, -3.0, -5.0]), 0.5, 50, 4.53333333),
        (np.array([1, 2, 3, 4, 5]), np.array([-1.0, 1.0, -1.0, -3.0, -5.0]), 1.0, 50, 2.26666667),
        (np.array([1, 2, 3, 4, 5]), np.array([-1.0, 1.0, -1.0, -3.0, -5.0]), 1.0, 2, 3.33333333),
    ],
)
def test_estimate(frame_idx, coordinate, time_step, max_lag, diffusion_const):
    diffusion_est = estimate_diffusion_constant_simple(frame_idx, coordinate, time_step, max_lag)
    np.testing.assert_allclose(diffusion_est, diffusion_const)


def test_maxlag_asserts():
    # Max_lag has to be bigger than 2
    with pytest.raises(ValueError):
        estimate_diffusion_constant_simple(np.array([1]), None, None, 1)

    with pytest.raises(ValueError):
        estimate_diffusion_constant_simple(np.array([1]), None, None, -1)


@pytest.mark.parametrize(
    "intercept, slope, result",
    [(1, 1, 1), (2, 1, 2), (1, 2, 0.5), (-1, 1, 0), (-1, -1, 0), (1, -1, np.inf)],
)
def test_localization_error_calculation(intercept, slope, result):
    calculate_localization_error(np.arange(10), intercept + slope * np.arange(10))


def test_localization_eq():
    with pytest.raises(AssertionError):
        calculate_localization_error(np.array([1, 2, 3]), np.array([1, 2]))


@pytest.mark.parametrize(
    "N, ref_slopes, ref_intercepts",
    [
        (
            10,
            np.array([2, 2, 2, 2, 3, 4, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6]),
            np.array([2, 2, 2, 2, 3, 4, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6]),
        ),
        (
            100,
            np.array([2, 2, 2, 2, 3, 4, 7, 12, 22, 39, 52, 56, 57, 57, 57, 57, 57, 57, 57]),
            np.array([2, 2, 2, 2, 3, 4, 7, 11, 18, 27, 35, 38, 39, 39, 39, 39, 39, 39, 39]),
        ),
        (
            1000,
            np.array([2, 2, 2, 2, 3, 4, 7, 12, 23, 44, 87, 170, 319, 485, 551, 563, 564, 564, 564]),
            np.array([2, 2, 2, 2, 3, 4, 7, 11, 18, 32, 55, 90, 126, 142, 145, 146, 146, 146, 146]),
        ),
    ],
)
def test_optimal_points(N, ref_slopes, ref_intercepts):
    test_values = np.hstack((np.arange(-2, 7, 0.5), np.inf))
    np.testing.assert_allclose(ref_slopes, [optimal_points(10 ** x, N)[0] for x in test_values])
    np.testing.assert_allclose(ref_intercepts, [optimal_points(10 ** x, N)[1] for x in test_values])


@pytest.mark.parametrize(
    "diffusion, num_steps, step, noise, n_optimal",
    [
        (0.1, 30, 1, 0, 2),
        (0.1, 30, 1, 0.1, 2),
        (0.1, 30, 1, 0.5, 3),
        (0.1, 30, 1, 1.0, 4),
        (0.1, 50, 1, 1.0, 7),
        (1, 50, 1, 1.0, 3),
    ],
)
def test_determine_points_from_data(diffusion, num_steps, step, noise, n_optimal):
    with temp_seed(0):
        coordinate = simulate_diffusion_1d(diffusion, num_steps, step, noise)
        np.testing.assert_allclose(
            determine_optimal_points(np.arange(num_steps), coordinate, max_iterations=100),
            n_optimal,
        )


def test_integer_frame_indices():
    # It is important that these diffusion methods take integers as frame indices, because otherwise
    # round-off errors can occur. This test checks whether this criterion is actively checked.
    with pytest.raises(TypeError):
        estimate_diffusion_constant_simple(np.array([1.5]), None, None, -1)

    with pytest.raises(TypeError):
        determine_optimal_points(np.arange(0.0, 5.0, 1.0), np.arange(0.0, 5.0, 1.0))


def test_max_iterations():
    with pytest.warns(RuntimeWarning):
        determine_optimal_points(np.arange(0, 5, 1), np.arange(0.0, 5.0, 1.0), max_iterations=0)
