import pytest
import contextlib
from lumicks.pylake.kymotracker.detail.msd_estimation import *
from lumicks.pylake.kymotracker.detail.msd_estimation import (
    _msd_diffusion_covariance,
    _diffusion_ols,
)


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
    diffusion_est = estimate_diffusion_constant_simple(
        frame_idx,
        coordinate,
        time_step,
        max_lag,
        "ols",
        "au",
    )
    np.testing.assert_allclose(float(diffusion_est), diffusion_const)


def test_maxlag_asserts():
    # Max_lag has to be bigger than 2
    with pytest.raises(ValueError):
        estimate_diffusion_constant_simple(np.array([1]), None, 1.0, 1, "ols", "au")

    with pytest.raises(ValueError):
        estimate_diffusion_constant_simple(np.array([1]), None, 1.0, -1, "ols", "au")


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
    "num_points, ref_slopes, ref_intercepts",
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
def test_optimal_points(num_points, ref_slopes, ref_intercepts):
    test_values = np.hstack((np.arange(-2, 7, 0.5), np.inf))
    np.testing.assert_allclose(
        ref_slopes, [optimal_points(10**x, num_points)[0] for x in test_values]
    )
    np.testing.assert_allclose(
        ref_intercepts, [optimal_points(10**x, num_points)[1] for x in test_values]
    )


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
        estimate_diffusion_constant_simple(np.array([1.5]), None, 1.0, -1, method="ols")

    with pytest.raises(TypeError):
        determine_optimal_points(np.arange(0.0, 5.0, 1.0), np.arange(0.0, 5.0, 1.0))


def test_max_iterations():
    with pytest.warns(RuntimeWarning):
        determine_optimal_points(np.arange(0, 5, 1), np.arange(0.0, 5.0, 1.0), max_iterations=0)


@pytest.mark.parametrize(
    "lags, num_points, intercept, slope, ref_matrix",
    [
        (2, 2, 1.5, 2.5, [[16.5625, 21.125], [21.125, 84.5]]),
        (
            3,
            3,
            2.5,
            3.5,
            [[25.38888889, 31.125, 38.25], [31.125, 102.5, 136.125], [38.25, 136.125, 338.0]],
        ),
        (2, 2, 0.2, 0.5, [[0.5, 0.72], [0.72, 2.88]]),
        (2, 10, 0.2, 0.5, [[0.1016, 0.14755556], [0.14755556, 0.42222222]]),
    ],
)
def test_covariance_matrix(lags, num_points, intercept, slope, ref_matrix):
    cov = _msd_diffusion_covariance(lags, num_points, intercept, slope)
    np.testing.assert_allclose(cov, ref_matrix)


@pytest.mark.parametrize(
    "msd, num_points, ref_values",
    [
        (
            [29.41107065, 49.10010613, 50.82447159, 60.589703],
            10,
            [23.666272215, 9.526026251, 160.12828532251723],
        ),
        (
            [63.76387668, 113.59294396, 159.35299198, 199.92392186],
            50,
            [20.598387729999924, 45.42401835600003, 390.3951873987407],
        ),
        (
            [63.76387668, 113.59294396, 159.35299198, 199.92392186, 190.05758296],
            50,
            [43.66274634999994, 33.89183904600002, 276.2572452069317],
        ),
        (
            [86.95265981, 108.194084, 128.47057056, 146.25592639, 170.09528357, 185.86762662],
            100,
            [67.83297963266669, 19.94467967399999, 61.06172869734946],
        ),
    ],
)
def test_ols_results(msd, num_points, ref_values):
    np.testing.assert_allclose(_diffusion_ols(msd, num_points), ref_values)


@pytest.mark.parametrize(
    "diffusion,num_points,max_lag,time_step,obs_noise,diff_est,std_err_est",
    [
        (0, 30, 3, 3, 0.0, 0.0, 0.0),
        (2, 500, 5, 0.01, 1.0, 1.9971773659882501, 1.545715391995203),
        (1.5, 30, 3, 3, 1.0, 2.2169635763968656, 1.0967721686794563),
        (1.5, 30, 3, 3, 0.0, 2.3305086928094925, 1.1330532354550331),
        (0, 30, 3, 3, 1.0, 0.0667054748976413, 0.05257289344744834),
        (0.1, 30, 2, 3, 0.0, 0.15107861079551288, 0.06602813821039434),  # No noise is very off
        (0.1, 30, 2, 3, 1.0, 0.10575410549731264, 0.11092540144986164),
        (1.1, 80, 30, 3, 0.1, 2.4535935321735742, 2.1317623681445164),  # Too many points bad
    ],
)
def test_diffusion_estimate_ols(
    diffusion, num_points, max_lag, time_step, obs_noise, diff_est, std_err_est
):
    with temp_seed(0):
        trace = simulate_diffusion_1d(diffusion, num_points, time_step, obs_noise)
        diffusion_est = estimate_diffusion_constant_simple(
            np.arange(num_points), trace, time_step, max_lag, "ols", "mu^2/s", r"$\mu^2/s$"
        )

        np.testing.assert_allclose(float(diffusion_est), diff_est)
        np.testing.assert_allclose(diffusion_est.value, diff_est)
        np.testing.assert_allclose(diffusion_est.num_lags, max_lag)
        np.testing.assert_allclose(diffusion_est.num_points, num_points)
        np.testing.assert_allclose(diffusion_est.std_err, std_err_est)
        assert diffusion_est.method == "ols"
        assert diffusion_est.unit == "mu^2/s"
        assert diffusion_est._unit_label == r"$\mu^2/s$"


@pytest.mark.parametrize(
    "diffusion,num_points,max_lag,time_step,obs_noise,diff_est,std_err_est",
    [
        (2, 500, 5, 0.01, 1.0, 1.9834877726431195, 1.5462259288408835),
        (1.5, 30, 3, 3, 1.0, 2.0372156730720934, 0.9522810729354054),
        (1.5, 30, 3, 3, 0.0, 2.248704975395052, 0.9790286899037831),
        (0, 30, 3, 3, 1.0, 0.06404927480648823, 0.05210111486596876),
        (0.1, 30, 2, 3, 0.0, 0.15107861079551305, 0.0660281382103944),  # No noise is very off
        (0.1, 30, 2, 3, 1.0, 0.10575410549731236, 0.11092540144986159),
        (1.1, 80, 30, 3, 0.1, 1.254440632135123, 0.3288129370162493),  # Too many points ok
    ],
)
def test_diffusion_estimate_gls(
    diffusion, num_points, max_lag, time_step, obs_noise, diff_est, std_err_est
):
    with temp_seed(0):
        trace = simulate_diffusion_1d(diffusion, num_points, time_step, obs_noise)
        diffusion_est = estimate_diffusion_constant_simple(
            np.arange(num_points), trace, time_step, max_lag, "gls", "mu^2/s", r"$\mu^2/s$"
        )

        np.testing.assert_allclose(float(diffusion_est), diff_est)
        np.testing.assert_allclose(diffusion_est.value, diff_est)
        np.testing.assert_allclose(diffusion_est.num_lags, max_lag)
        np.testing.assert_allclose(diffusion_est.num_points, num_points)
        np.testing.assert_allclose(diffusion_est.std_err, std_err_est)
        assert diffusion_est.method == "gls"
        assert diffusion_est.unit == "mu^2/s"
        assert diffusion_est._unit_label == r"$\mu^2/s$"


def test_bad_input():
    with pytest.raises(ValueError, match="Invalid method selected."):
        estimate_diffusion_constant_simple(np.arange(5), np.arange(5), 1, 2, "glo", "unit")

    with pytest.raises(
        ValueError, match="You need at least two lags to estimate a diffusion constant"
    ):
        estimate_diffusion_constant_simple(np.arange(5), np.arange(5), 1, 1, "gls", "unit")


def test_singular_handling():
    with temp_seed(0):
        trace = simulate_diffusion_1d(0, 30, 3, 0)
        with pytest.warns(RuntimeWarning, match="Covariance matrix is singular"):
            estimate_diffusion_constant_simple(np.arange(len(trace)), trace, 1, 3, "gls", "unit")
