import re

import pytest
import matplotlib.pyplot as plt

from lumicks.pylake.detail.utilities import temp_seed
from lumicks.pylake.simulation.diffusion import _simulate_diffusion_1d
from lumicks.pylake.kymotracker.detail.msd_estimation import *
from lumicks.pylake.kymotracker.detail.msd_estimation import (
    _cve,
    _diffusion_ols,
    _var_cve_known_var,
    _var_cve_unknown_var,
    _msd_diffusion_covariance,
    _determine_optimal_points_ensemble,
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
    with pytest.raises(
        ValueError,
        match="To calculate a localization error, you need to supply an MSD estimate in msd for "
        "each lag time in frame_lag",
    ):
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
        coordinate = _simulate_diffusion_1d(diffusion, num_steps, step, noise)
        np.testing.assert_allclose(
            determine_optimal_points(np.arange(num_steps), coordinate, max_iterations=100),
            n_optimal,
        )
        np.testing.assert_allclose(
            _determine_optimal_points_ensemble(
                *calculate_msd(np.arange(num_steps), coordinate, max_lag=None),
                len(coordinate),
                max_iterations=100,
            ),
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
    np.testing.assert_allclose(_diffusion_ols(np.arange(len(msd)) + 1, msd, num_points), ref_values)


@pytest.mark.parametrize(
    "diffusion,num_points,max_lag,time_step,obs_noise,diff_est,std_err_est,loc_variance",
    # fmt:off
    [
        (0, 30, 3, 3, 0.0, 0.0, 0.0, 0.0),
        (2, 500, 5, 0.01, 1.0, 1.9971773659882501, 1.545715391995203, 0.9711582323236502),
        (1.5, 30, 3, 3, 1.0, 2.2169635763968656, 1.0967721686794563, -0.3239881906359111),
        (1.5, 30, 3, 3, 0.0, 2.3305086928094925, 1.1330532354550331, -1.1775008884354936),
        (0, 30, 3, 3, 1.0, 0.0667054748976413, 0.05257289344744834, 0.5904545191371421),
        (0.1, 30, 2, 3, 0.0, 0.15107861079551288, 0.06602813821039434, -0.061345517661886007),  # No noise is very off
        (0.1, 30, 2, 3, 1.0, 0.10575410549731264, 0.11092540144986164, 0.8589882417178512),
        (1.1, 80, 30, 3, 0.1, 2.4535935321735742, 2.1317623681445164, -24.181310284805875),  # Too many points bad
    ],
    # fmt:on
)
def test_diffusion_estimate_ols(
    diffusion, num_points, max_lag, time_step, obs_noise, diff_est, std_err_est, loc_variance
):
    with temp_seed(0):
        trace = _simulate_diffusion_1d(diffusion, num_points, time_step, obs_noise)
        diffusion_est = estimate_diffusion_constant_simple(
            np.arange(num_points), trace, time_step, max_lag, "ols", "mu^2/s", r"$\mu^2/s$"
        )

        np.testing.assert_allclose(float(diffusion_est), diff_est)
        np.testing.assert_allclose(diffusion_est.value, diff_est)
        np.testing.assert_allclose(diffusion_est.num_lags, max_lag)
        np.testing.assert_allclose(diffusion_est.num_points, num_points)
        np.testing.assert_allclose(diffusion_est.std_err, std_err_est)
        np.testing.assert_allclose(diffusion_est.localization_variance, loc_variance)
        assert diffusion_est.method == "ols"
        assert diffusion_est.unit == "mu^2/s"
        assert diffusion_est._unit_label == r"$\mu^2/s$"


@pytest.mark.parametrize(
    "diffusion, num_points, max_lag, time_step, obs_noise, diff_est, std_err_est, skip, shuffle",
    [
        (2.0, 1000, 5, 0.01, 0.5, 2.0191353993755534, 0.2691422691544549, 2, False),
        (2.0, 1000, 3, 0.01, 0.5, 1.5714322945079129, 0.8912916583320089, 2, True),
        (2.0, 5000, 5, 0.01, 0.5, 1.9352306588121024, 0.23809537086111288, 2, True),
    ],
)
def test_regression_ols_with_skipped_frames(
    diffusion,
    num_points,
    max_lag,
    time_step,
    obs_noise,
    diff_est,
    std_err_est,
    skip,
    shuffle,
):
    with temp_seed(0):
        trace = _simulate_diffusion_1d(diffusion, num_points, time_step, obs_noise)

        subsampling = np.zeros(skip - 1, dtype=bool)
        skipped_sampling = np.tile(np.hstack((True, subsampling)), num_points // skip)
        if shuffle:
            np.random.shuffle(skipped_sampling)

        frame_idx, trace = np.arange(num_points)[skipped_sampling], trace[skipped_sampling]

        with pytest.warns(RuntimeWarning, match="Your tracks have missing frames"):
            diffusion_est = estimate_diffusion_constant_simple(
                frame_idx, trace, time_step, max_lag, "ols", "mu^2/s", r"$\mu^2/s$"
            )

    np.testing.assert_allclose(float(diffusion_est), diff_est)
    np.testing.assert_allclose(diffusion_est.value, diff_est)
    np.testing.assert_allclose(diffusion_est.num_lags, max_lag)
    np.testing.assert_allclose(diffusion_est.num_points, num_points // skip)
    np.testing.assert_allclose(diffusion_est.std_err, std_err_est)


def test_skipped_sample_protection():
    lag_idx = np.asarray([1, 2, 4, 5, 6, 7, 8])
    with pytest.raises(
        RuntimeError,
        match=re.escape(
            "Your tracks cannot have missing frames when using the GLS estimator. Refine your "
            "tracks using `lk.refine_tracks_centroid()`"
        ),
    ):
        estimate_diffusion_constant_simple(lag_idx, lag_idx**2, 1, 2, method="gls")

    with pytest.warns(
        RuntimeWarning,
        match=re.escape(
            "Your tracks have missing frames. Note that this can lead to a suboptimal estimate "
            "of the optimal number of lags when using OLS."
        ),
    ):
        # We warn if the user tries to rely on the automatic lag estimation which according to the
        # paper may be of arguable quality.
        determine_optimal_points(lag_idx, lag_idx**2, max_iterations=100)

    with pytest.warns(
        RuntimeWarning,
        match=re.escape(
            "Your tracks have missing frames. Note that this results in a poor estimate of "
            "the standard error of the estimate. To avoid this warning, you can refine "
            "your tracks using `lk.refine_tracks_centroid()` or `lk.refine_tracks_gaussian()`. "
            "Please refer to `help(lk.refine_tracks_centroid)` or `help(lk.refine_tracks_gaussian)` "
            "for more information."
        ),
    ):
        estimate_diffusion_constant_simple(lag_idx, lag_idx**2, 1, 5, method="ols")


@pytest.mark.parametrize(
    "diffusion,num_points,max_lag,time_step,obs_noise,diff_est,std_err_est,loc_variance",
    # fmt:off
    [
        (2, 500, 5, 0.01, 1.0, 1.9834877726431195, 1.5462259288408835, 0.9725695521205839),
        (1.5, 30, 3, 3, 1.0, 2.0372156730720934, 0.9522810729354054, 0.4172355292491052),
        (1.5, 30, 3, 3, 0.0, 2.248704975395052, 0.9790286899037831, -0.8695045202096329),
        (0, 30, 3, 3, 1.0, 0.06404927480648823, 0.05210111486596876, 0.6132128433160494),
        (0.1, 30, 2, 3, 0.0, 0.15107861079551305, 0.0660281382103944, -0.06134551766188702),  # No noise is very off
        (0.1, 30, 2, 3, 1.0, 0.10575410549731236, 0.11092540144986159, 0.8589882417178527),
        (1.1, 80, 30, 3, 0.1, 1.254440632135123, 0.3288129370162493, -0.5158647586841503),  # Too many points ok
    ],
    # fmt:on
)
def test_diffusion_estimate_gls(
    diffusion, num_points, max_lag, time_step, obs_noise, diff_est, std_err_est, loc_variance
):
    with temp_seed(0):
        trace = _simulate_diffusion_1d(diffusion, num_points, time_step, obs_noise)
        diffusion_est = estimate_diffusion_constant_simple(
            np.arange(num_points), trace, time_step, max_lag, "gls", "mu^2/s", r"$\mu^2/s$"
        )

        np.testing.assert_allclose(float(diffusion_est), diff_est)
        np.testing.assert_allclose(diffusion_est.value, diff_est)
        np.testing.assert_allclose(diffusion_est.num_lags, max_lag)
        np.testing.assert_allclose(diffusion_est.num_points, num_points)
        np.testing.assert_allclose(diffusion_est.std_err, std_err_est)
        np.testing.assert_allclose(diffusion_est.localization_variance, loc_variance)
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
        trace = _simulate_diffusion_1d(0, 30, 3, 0)
        with pytest.warns(RuntimeWarning, match="Covariance matrix is singular"):
            estimate_diffusion_constant_simple(np.arange(len(trace)), trace, 1, 3, "gls", "unit")


@pytest.mark.parametrize(
    "diffusion,obs_noise,num_points,time_step, blur_constant,localization_var,"
    "var_of_localization_var,diffusion_ref,diffusion_var_ref,localization_var_ref",
    # fmt:off
    [
        (0.1, 0.01, 50, 0.1, 0, None, None, 0.09206101801689968, 0.0009716913812060779, -0.0008040516549331511),
        (0.1, 0.01, 50, 0.1, 1/6, None, None, 0.09206101801689968, 0.0009716913812060779, 0.0022646489456301716),
        (0.1, 0.01, 50, 0.1, 0, 0.0001, 0.0001, 0.08302050146756818, 0.010282397786674464, 0.0001),
        (0.1, 0.01, 50, 0.1, 1/6, 0.0001, 0.0001, 0.12453075220135225, 0.023209198638670672, 0.0001),
    ],
    # fmt:on
)
def test_cve(
    diffusion,
    obs_noise,
    num_points,
    time_step,
    blur_constant,
    localization_var,
    var_of_localization_var,
    diffusion_ref,
    diffusion_var_ref,
    localization_var_ref,
):
    with temp_seed(10):
        frame = np.arange(num_points)
        coord = _simulate_diffusion_1d(diffusion, num_points, time_step, obs_noise)
        diffusion_est, diffusion_var_est, localization_var_est = _cve(
            frame, coord, time_step, blur_constant, localization_var, var_of_localization_var
        )
        np.testing.assert_allclose(diffusion_est, diffusion_ref)
        np.testing.assert_allclose(diffusion_var_est, diffusion_var_ref)
        np.testing.assert_allclose(localization_var_est, localization_var_ref)


@pytest.mark.parametrize(
    "diffusion,obs_noise,num_points,time_step, blur_constant,localization_var,"
    "var_of_localization_var,diffusion_ref,diffusion_var_ref,localization_var_ref",
    # fmt:off
    [
        (0.1, 0.01, 50, 0.1, 0, None, None, 0.10637910762942307, 0.0015596341426062676, -0.0005047779987189253),
        (0.1, 0.01, 50, 0.1, 1/6, None, None, 0.10637910762942307, 0.0016112096114555851, 0.00363218729798086),
        (0.1, 0.01, 50, 0.1, 0, 0.0001, 0.0001, 0.10119529621183229, 0.007831360527457441, 0.0001),
        (0.1, 0.01, 50, 0.1, 1/6, 0.0001, 0.0001, 0.14167341469656522, 0.0154209878094207, 0.0001),
    ],
    # fmt:on
)
def test_cve_skipped_samples(
    diffusion,
    obs_noise,
    num_points,
    time_step,
    blur_constant,
    localization_var,
    var_of_localization_var,
    diffusion_ref,
    diffusion_var_ref,
    localization_var_ref,
):
    with temp_seed(10):
        frame = np.arange(num_points)
        coord = _simulate_diffusion_1d(diffusion, num_points, time_step, obs_noise)
        mask = np.full(len(frame), True, dtype=bool)
        mask[np.array([10, 20, 40, 41, 42, 43, 44])] = False
        diffusion_est, diffusion_var_est, localization_var_est = _cve(
            frame[mask],
            coord[mask],
            time_step,
            blur_constant,
            localization_var,
            var_of_localization_var,
        )
        np.testing.assert_allclose(diffusion_est, diffusion_ref)
        np.testing.assert_allclose(diffusion_var_est, diffusion_var_ref)
        np.testing.assert_allclose(localization_var_est, localization_var_ref)


@pytest.mark.parametrize(
    "diffusion,obs_noise,num_points,time_step,blur_constant,localization_var,"
    "var_of_localization_var,diffusion_ref,diffusion_var_ref,num_points_ref,localization_var_ref, "
    "var_of_localization_var_ref",
    # fmt:off
    [
        (2, 0.01, 50, 0.1, 0, None, None, 1.8984350363870435, 0.3972529037563432, 50, -0.028026554019435785, None),
        (2, 0.01, 50, 0.1, 1/6, None, None, 1.8984350363870435, 0.3972529037563432, 50, 0.03525461386013233, None),
        (2, 0.01, 50, 0.1, 0, 0.002, 0.0002, 1.5981694961926856, 0.12474690073633955, 50, 0.002, 0.0002),
        (2, 0.01, 50, 0.1, 1/6, 0.002, 0.0002, 2.397254244289028, 0.3079763136689993, 50, 0.002, 0.0002),
    ]
)
def test_estimate_diffusion_cve(
    diffusion,
    obs_noise,
    num_points,
    time_step,
    blur_constant,
    localization_var,
    var_of_localization_var,
    diffusion_ref,
    diffusion_var_ref,
    num_points_ref,
    localization_var_ref,
    var_of_localization_var_ref,
):
    with temp_seed(10):
        trace = _simulate_diffusion_1d(diffusion, num_points, time_step, obs_noise)
        diffusion_est = estimate_diffusion_cve(
            np.arange(num_points),
            trace,
            time_step,
            blur_constant,
            "mu^2/s",
            r"$\mu^2/s$",
            localization_var,
            var_of_localization_var,
        )

        np.testing.assert_allclose(float(diffusion_est), diffusion_ref)
        np.testing.assert_allclose(diffusion_est.value, diffusion_ref)
        assert diffusion_est.num_lags is None
        np.testing.assert_allclose(diffusion_est.num_points, num_points_ref)
        np.testing.assert_allclose(diffusion_est.std_err, np.sqrt(diffusion_var_ref))
        np.testing.assert_allclose(diffusion_est.localization_variance, localization_var_ref)
        if var_of_localization_var_ref:
            np.testing.assert_allclose(
                diffusion_est.variance_of_localization_variance, var_of_localization_var_ref
            )
        else:
            assert diffusion_est.variance_of_localization_variance is None
        assert diffusion_est.method == "cve"
        assert diffusion_est.unit == "mu^2/s"
        assert diffusion_est._unit_label == r"$\mu^2/s$"


@pytest.mark.parametrize(
    "diffusion_constant,localization_var,var_of_localization_var,dt,num_points,blur_constant,"
    "avg_frame_steps,ref_value_known,ref_value_unknown",
    [
        (2.0, 0.0001, 0.00005, 0.1, 50, 0, 0.9, 0.16635069135802472, 0.48658494024691357),
        (2.0, 0.0002, 0.0001, 0.1, 50, 0, 0.9, 0.17270153086419754, 0.4867699832098765),
        (2.0, 0.0001, 0.00005, 0.2, 50, 0, 0.9, 0.16163211728395063, 0.4864924572839506),
        (2.0, 0.0001, 0.00005, 0.1, 20, 0, 0.9, 0.40661746913580243, 1.2404890246913578),
        (2.0, 0.0001, 0.00005, 0.1, 50, 0.3, 0.9, 0.5355562222222221, 0.3385505698765432),
        (2.0, 0.0001, 0.00005, 0.1, 50, 0, 1.0, 0.16516005999999997, 0.48656644159999995),
        (2.5, 0.0001, 0.00005, 0.1, 50, 0, 0.9, 0.2563951358024691, 0.7602311624691358),
        (0, 0.0001, 0.00005, 0.1, 50, 0, 0.9, 0.006172913580246913, 5.1358024691358016e-08),
    ],
)
def test_cve_variances(
    diffusion_constant,
    localization_var,
    var_of_localization_var,
    dt,
    num_points,
    blur_constant,
    avg_frame_steps,
    ref_value_known,
    ref_value_unknown,
):
    shared_pars = {
        "diffusion_constant": diffusion_constant,
        "localization_var": localization_var,
        "dt": dt,
        "num_points": num_points,
        "blur_constant": blur_constant,
        "avg_frame_steps": avg_frame_steps,
    }
    np.testing.assert_allclose(
        _var_cve_known_var(**shared_pars, var_of_localization_var=var_of_localization_var),
        ref_value_known,
    )

    np.testing.assert_allclose(_var_cve_unknown_var(**shared_pars), ref_value_unknown)


@pytest.mark.parametrize(
    "means, counts, ref_weighted_mean, ref_weighted_variance, ref_counts, ref_ess",
    [
        (np.array([2.0, 2.0, 4.0, 4.0]), np.array([4, 4, 4, 4]), 3, 4 / 3, 16, 4),
        (np.array([2.0, 2.0, 4.0]), np.array([2, 2, 4]), 3, 1.6, 8, 8 / 3),
    ],
)
def test_weighted_variance(
    means, counts, ref_weighted_mean, ref_weighted_variance, ref_counts, ref_ess
):
    weighted_mean, weighted_var, counts, ess = weighted_mean_and_sd(means, counts)
    np.testing.assert_allclose(weighted_mean, ref_weighted_mean)
    np.testing.assert_allclose(weighted_var, ref_weighted_variance)
    np.testing.assert_allclose(counts, ref_counts)
    np.testing.assert_allclose(ess, ref_ess)


def test_weighted_variance_error_case():
    with pytest.raises(
        ValueError, match="Need more than one average to compute a weighted variance"
    ):
        weighted_mean_and_sd(np.array([1]), np.array([1]))

    with pytest.raises(ValueError, match="Mean and count arrays must be the same size"):
        weighted_mean_and_sd(np.array([2, 3]), np.array([2, 3, 4]))


@pytest.mark.parametrize(
    "frame_idx,position,max_lag,ref_lags,ref_msds,ref_samples",
    [
        [[1, 2, 3, 4], [1, 2, 3, 4], 1000, [1, 2, 3], [1.0, 4.0, 9.0], [3, 2, 1]],
        [[1, 2, 4], [1, 2, 4], 1000, [1, 2, 3], [1.0, 4.0, 9.0], [1, 1, 1]],
        [[1, 2, 5], [1, 2, 4], 1000, [1, 3, 4], [1.0, 4.0, 9.0], [1, 1, 1]],
        [[1, 2, 5, 6], [1, 2, 4, 5], 1000, [1, 3, 4, 5], [1.0, 4.0, 9.0, 16.0], [2, 1, 2, 1]],
        [
            [1, 2, 5, 6],
            [1.5, 0.5, 3.0, 5.5],
            1000,
            [1, 3, 4, 5],
            [3.625, 6.25, 13.625, 16.0],
            [2, 1, 2, 1],
        ],
        [[1, 2, 3, 4], [1, 2, 3, 4], 2, [1, 2], [1.0, 4.0], [3, 2]],  # test max_lag
        # max_lag refers to number of lags, not maximum lag
        [[1, 2, 5, 6], [1, 2, 4, 5], 3, [1, 3, 4], [1.0, 4.0, 9.0], [2, 1, 2]],
    ],
)
def test_msds_counts(frame_idx, position, max_lag, ref_lags, ref_msds, ref_samples):
    """Test function that computed squared displacement and sample counts"""
    lags, msds, num_samples = calculate_msd_counts(
        np.asarray(frame_idx), np.asarray(position), max_lag=max_lag
    )
    np.testing.assert_allclose(lags, ref_lags)
    np.testing.assert_allclose(msds, ref_msds)
    np.testing.assert_allclose(num_samples, ref_samples)


def test_merge_msds():
    """Test function which merges squared displacements obtained from various tracks"""

    # Tracks are given as a list of numpy arrays with [lags, msds, number of samples]
    trk = [np.array([1, 2, 3, 4]), np.array([1.0, 2.0, 3.0, 4.0]), np.array([4, 3, 2, 1])]
    trk2 = [np.array([1, 2, 3, 5]), np.array([3.0, 2.0, 3.0, 10.0]), np.array([5, 3, 2, 1])]

    lags, msds = merge_track_msds([trk, trk2, trk2], min_count=0)
    ref_msds = [
        [[1.0, 3.0, 3.0], [4, 5, 5]],
        [[2.0, 2.0, 2.0], [3, 3, 3]],
        [[3.0, 3.0, 3.0], [2, 2, 2]],
        [[4.0], [1]],
        [[10.0, 10.0], [1, 1]],
    ]
    np.testing.assert_allclose(lags, [1, 2, 3, 4, 5])
    assert len(msds) == len(ref_msds)
    for rho_count, ref_rho_count in zip(msds, ref_msds):
        np.testing.assert_allclose(rho_count[0], ref_rho_count[0])  # Compare rho
        np.testing.assert_allclose(rho_count[1], ref_rho_count[1])  # Compare count

    # Include a filter
    lags, msds = merge_track_msds([trk, trk2, trk2], min_count=2)
    ref_msds = [
        [[1.0, 3.0, 3.0], [4, 5, 5]],
        [[2.0, 2.0, 2.0], [3, 3, 3]],
        [[3.0, 3.0, 3.0], [2, 2, 2]],
        [[10.0, 10.0], [1, 1]],
    ]
    np.testing.assert_allclose(lags, [1, 2, 3, 5])
    assert len(msds) == len(ref_msds)
    for rho_count, ref_rho_count in zip(msds, ref_msds):
        np.testing.assert_allclose(rho_count[0], ref_rho_count[0])  # Compare rho
        np.testing.assert_allclose(rho_count[1], ref_rho_count[1])  # Compare count

    lags, msds = merge_track_msds([trk, trk2, trk2], min_count=6)
    np.testing.assert_equal(lags, [])
    np.testing.assert_equal(msds, [])


def test_ensemble_msd():
    frame_diffs = np.arange(1, 5, 1)

    # Lags, mean squared displacements and counts for 3 tracks
    track_msds = [
        [frame_diffs, frame_diffs**2 + 0.1, np.arange(len(frame_diffs), 0, -1)],
        [frame_diffs, frame_diffs**2 - 0.1, np.arange(len(frame_diffs), 0, -1)],
        [np.array([1, 2, 3, 5]), np.array([1, 2, 3, 5]) ** 2, np.arange(len(frame_diffs), 0, -1)],
    ]

    # By default, the single lag rho (5) should be ignored
    result = calculate_ensemble_msd(track_msds, 1.0, unit="what_a_unit", unit_label="label_ahoy")
    np.testing.assert_allclose(result.lags, frame_diffs)
    np.testing.assert_allclose(result.msd, frame_diffs**2)
    num_means = np.array([3, 3, 3, 2])  # number of means contributing to the estimate
    np.testing.assert_allclose(result.variance, 0.02 / (num_means - 1))
    np.testing.assert_allclose(result.counts, [12, 9, 6, 2])
    # Tracks are equal length, so the effective sample size is just the means that contributed
    np.testing.assert_allclose(result.effective_sample_size, num_means)
    np.testing.assert_allclose(result.sem, np.sqrt(0.02 / ((num_means - 1) * num_means)))
    assert result.unit == "what_a_unit^2"
    assert result._unit_label == "label_ahoy²"


def test_ensemble_msd_unequal_points():
    frame_diffs = np.arange(1, 6, 1)

    # Lags, mean squared displacements and counts for 3 tracks
    track_msds = [
        [frame_diffs, np.ones(frame_diffs.shape), [2, 2, 4, 4, 4]],
        [frame_diffs, 4 * np.ones(frame_diffs.shape), [4, 4, 2, 2, 2]],
    ]

    result = calculate_ensemble_msd(track_msds, 1.0)
    np.testing.assert_allclose(result.lags, frame_diffs)
    np.testing.assert_allclose(result.msd, np.array([3, 3, 2, 2, 2]))
    np.testing.assert_allclose(result.variance, np.ones(5) * 4.5)
    np.testing.assert_allclose(result.counts, np.ones(5) * 6)
    # ESS is less than 2 since we used weighting
    np.testing.assert_allclose(result.effective_sample_size, np.ones(5) * 9 / 5)
    np.testing.assert_allclose(result.sem, np.ones(5) * np.sqrt(5 / 2))


def test_ensemble_msd_little_data():
    frame_diffs = np.arange(1, 5, 1)
    trk1 = [frame_diffs, frame_diffs**2, np.arange(len(frame_diffs), 0, -1)]
    trk2 = [np.array([1, 2, 3, 5]), np.array([1, 2, 3, 5]) ** 2, np.arange(len(frame_diffs), 0, -1)]

    with pytest.raises(
        ValueError, match="Need more than one average to compute a weighted variance"
    ):
        calculate_ensemble_msd([trk1, trk1, trk2], 1.0, unit="au", unit_label="au", min_count=0)

    for msds in ([trk1], []):
        with pytest.raises(
            ValueError, match="You need at least two tracks to compute the ensemble MSD"
        ):
            calculate_ensemble_msd(msds, 1.0, unit="au", unit_label="au", min_count=0)


def test_ensemble_msd_plot():
    """Test whether the plot spins up"""
    frame_diffs = np.arange(1, 5, 1)
    trk1 = [frame_diffs, frame_diffs**2, np.arange(len(frame_diffs), 0, -1)]
    calculate_ensemble_msd([trk1, trk1, trk1], 1.0, unit="au", unit_label="label_unit").plot()
    axis = plt.gca()
    lines = axis.lines[0]
    np.testing.assert_allclose(lines.get_xdata(), frame_diffs)
    np.testing.assert_allclose(lines.get_ydata(), frame_diffs**2)
    assert axis.xaxis.get_label().get_text() == "Time [s]"
    assert axis.yaxis.get_label().get_text() == "Squared Displacement [label_unit²]"
