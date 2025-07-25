import re
from dataclasses import dataclass
from unittest.mock import Mock

import numpy as np
import pytest
import matplotlib.pyplot as plt

from lumicks.pylake import DwelltimeModel
from lumicks.pylake.detail.utilities import temp_seed
from lumicks.pylake.population.dwelltime import (
    DwelltimeBootstrap,
    _exponential_mle_optimize,
    _dwellcounts_from_statepath,
    _handle_amplitude_constraint,
    _exponential_mixture_log_likelihood,
    _exponential_mixture_log_likelihood_jacobian,
)
from lumicks.pylake.fitting.detail.derivative_manipulation import numerical_jacobian


def test_likelihood(exponential_data):
    # single exponential data
    dataset = exponential_data["dataset_1exp"]
    fit = DwelltimeModel(dataset["data"], 1, **dataset["parameters"].observation_limits)
    np.testing.assert_allclose(fit.log_likelihood, -1290.2636036948977, rtol=1e-5)
    np.testing.assert_allclose(fit.aic, 2582.5272073897954, rtol=1e-5)
    np.testing.assert_allclose(fit.bic, 2587.3815618920503, rtol=1e-5)

    # double exponential data
    dataset = exponential_data["dataset_2exp"]
    fit = DwelltimeModel(dataset["data"], 2, **dataset["parameters"].observation_limits)
    np.testing.assert_allclose(fit.log_likelihood, -2204.2954468395105, rtol=1e-5)
    np.testing.assert_allclose(fit.aic, 4414.590893679021, rtol=1e-5)
    np.testing.assert_allclose(fit.bic, 4429.232045925579, rtol=1e-5)


@pytest.mark.parametrize(
    "data, min_obs, max_obs, loglik",
    [
        # fmt:off
        (np.array([0.2, 0.3, 0.6, 1.2]), 0.1, 1.4, -0.604631649126058),
        (np.array([0.2, 0.3, 0.6, 1.2]), np.array([0.1, 0.1, 0.3, 0.3]), 1.3, 0.25784386656710745),
        (np.array([0.2, 0.3, 0.6, 1.2]), 0.1, np.array([0.5, 0.5, 1.3, 1.3]), 1.1932900895106002),
        (np.array([0.2, 0.3, 0.6, 1.2]), np.array([0.1, 0.1, 0.3, 0.3]), np.array([0.5, 0.5, 1.3, 1.3]), 1.7155483581674074),
        # fmt:on
    ],
)
def test_multi_observation_limits(data, min_obs, max_obs, loglik):
    fit = DwelltimeModel(data, 1, min_observation_time=min_obs, max_observation_time=max_obs)
    np.testing.assert_allclose(fit.log_likelihood, loglik, rtol=1e-5)


def test_invalid_multi_dwelltime_parameters():
    data = np.arange(1.0, 4.0, 1.0)
    with pytest.raises(
        ValueError,
        match=re.escape(
            r"Size of minimum observation time array (2) must be equal to that of dwelltimes (3)"
        ),
    ):
        DwelltimeModel(data, min_observation_time=np.array([0.1, 0.2]), max_observation_time=10.0)

    with pytest.raises(
        ValueError,
        match=re.escape(
            r"Size of maximum observation time array (4) must be equal to that of dwelltimes (3)"
        ),
    ):
        DwelltimeModel(data, min_observation_time=1.0, max_observation_time=np.array([1, 2, 3, 4]))

    with pytest.raises(
        ValueError,
        match=re.escape(
            r"When providing an array of discretization timesteps, the number of "
            r"discretization timesteps (2) should equal the number of dwell times provided (3)."
        ),
    ):
        DwelltimeModel(data, min_observation_time=10, discretization_timestep=np.array([10, 1]))

    with pytest.raises(
        ValueError,
        match=re.escape(
            r"To use a continuous model, specify a discretization timestep of None. Do not pass "
            r"zero as this leads to an invalid probability mass function."
        ),
    ):
        DwelltimeModel(data, min_observation_time=10, discretization_timestep=np.array([10, 0, 10]))

    for dt, min_obs in (
        (np.array([10, 10, 10]), np.array([0.0, 0.0, 0.0])),
        (10, np.array([0.0, 0.0, 0.0])),
        (np.array([10, 10, 10]), 0.0),
        (10, 0.0),
    ):
        with pytest.raises(
            ValueError,
            match=re.escape(
                r"The discretization timestep (10.0) cannot be larger than the minimum observable "
                r"time (0.0)."
            ),
        ):
            DwelltimeModel(data, discretization_timestep=dt, min_observation_time=min_obs)


@pytest.mark.parametrize(
    "model_args, scipy_args",
    [
        ({"tol": 1e-1}, {"ftol": 1e-1}),
        ({"max_iter": 2}, {"maxiter": 2}),
    ],
)
def test_optim_options(exponential_data, monkeypatch, model_args, scipy_args):
    @dataclass
    class MockResult:
        x: np.ndarray
        fun: float

    with monkeypatch.context() as m:
        dataset = exponential_data["dataset_1exp"]

        minimize = Mock(return_value=MockResult(np.array([5]), 5))
        m.setattr("scipy.optimize.minimize", minimize)
        _ = DwelltimeModel(
            dataset["data"], 1, **dataset["parameters"].observation_limits, **model_args
        )
        assert minimize.call_args.kwargs["options"] == scipy_args


def test_fit_parameters(exponential_data):
    # single exponential data
    dataset = exponential_data["dataset_1exp"]
    fit = DwelltimeModel(dataset["data"], 1, **dataset["parameters"].observation_limits, tol=1e-8)
    np.testing.assert_allclose(fit.amplitudes, [1])
    np.testing.assert_allclose(fit.lifetimes, [1.43481181], rtol=1e-5)
    np.testing.assert_allclose(fit.rate_constants, [1 / 1.43481181], rtol=1e-5)

    # double exponential data
    dataset = exponential_data["dataset_2exp"]
    fit = DwelltimeModel(dataset["data"], 2, **dataset["parameters"].observation_limits, tol=1e-8)
    np.testing.assert_allclose(fit.amplitudes, [0.46516346, 0.53483653], rtol=1e-5)
    np.testing.assert_allclose(fit.lifetimes, [1.50634996, 5.46227291], rtol=1e-5)
    np.testing.assert_allclose(fit.rate_constants, [1 / 1.50634996, 1 / 5.46227291], rtol=1e-5)


@pytest.mark.parametrize(
    "min_obs, max_obs, time_step, ref_ci",
    [
        (0.1, 1.4, None, (0.2663655, 1.4679362)),
        (np.array([0.1, 0.1, 0.3, 0.3]), 1.3, None, (0.2068400, 1.3463159)),
        (0.1, np.array([0.5, 0.5, 1.3, 1.3]), None, (0.7838319, 1.4147650)),
        (
            np.array([0.1, 0.1, 0.3, 0.3]),
            np.array([0.5, 0.5, 1.3, 1.3]),
            None,
            (0.3497069, 1.3528150),
        ),
        (0.2, 1.4, 0.2, (0.247748, 1.463575)),
        (np.array([0.2, 0.2, 0.4, 0.4]), 1.4, np.array([0.2, 0.2, 0.4, 0.4]), (0.21762, 1.448516)),
    ],
)
def test_bootstrap_multi(min_obs, max_obs, ref_ci, time_step):
    np.random.seed(123)
    data = np.array([0.2, 0.3, 0.6, 1.2])
    fit = DwelltimeModel(
        data,
        1,
        min_observation_time=min_obs,
        max_observation_time=max_obs,
        discretization_timestep=time_step,
    )
    bootstrap = fit.calculate_bootstrap(iterations=4)
    ci = bootstrap.get_interval("lifetime", 0)
    np.testing.assert_allclose(ci, ref_ci, rtol=1e-5)


@pytest.mark.parametrize("num_components", [1, 2])
@pytest.mark.filterwarnings("ignore:Values in x were outside bounds")
def test_bootstrap_parallel(monkeypatch, exponential_data, num_components):
    import multiprocessing.dummy

    dataset = exponential_data["dataset_2exp"]
    fit = DwelltimeModel(
        dataset["data"], num_components, **dataset["parameters"].observation_limits
    )

    with monkeypatch.context() as m:
        # Pipe communication doesn't seem to work on GitHub runners for macOS and Windows therefore
        # we fall back using a multiprocessing.dummy which emulates the pool with threads.
        m.setattr("lumicks.pylake.dwelltime.Pool", multiprocessing.dummy.Pool)

        bootstrap = fit.calculate_bootstrap(iterations=3, num_processes=2)
        assert bootstrap.n_samples == 3


@pytest.mark.slow
def test_bootstrap(exponential_data):
    # double exponential data
    dataset = exponential_data["dataset_2exp"]
    fit = DwelltimeModel(dataset["data"], 2, **dataset["parameters"].observation_limits)

    np.random.seed(123)
    bootstrap = fit.calculate_bootstrap(iterations=50)

    with pytest.warns(DeprecationWarning):
        mean, ci = bootstrap.calculate_stats("amplitude", 0)
        np.testing.assert_allclose(mean, 0.4642469883372174, rtol=1e-5)
        np.testing.assert_allclose(ci, (0.3647038711684928, 0.5979550940729152), rtol=1e-5)

    ci = bootstrap.get_interval("amplitude", 0, alpha=0.05)
    np.testing.assert_allclose(ci, (0.3647038711684928, 0.5979550940729152), rtol=1e-5)
    np.random.seed()

    more_bootstrap = bootstrap.extend(iterations=10)
    assert bootstrap.n_samples == 50
    assert more_bootstrap.n_samples == 60

    # TODO: delete after property removal
    with pytest.warns(DeprecationWarning):
        fit.bootstrap


@pytest.mark.slow
@pytest.mark.parametrize(
    # fmt:off
    "exp_name, reference_bounds, reinterpolated_bounds",
    [
        (
            "dataset_2exp",
            (
                (("amplitude", 0), (0.33042164281045583, 0.6134205866556652)),
                (("amplitude", 1), (0.38657420237892437, 0.6697233170539962)),
                (("lifetime", 0), (1.1248940731415664, 1.9643923058865285)),
                (("lifetime", 1), (4.713911531181084, 6.541327832464464)),
            ),
            (
                (("amplitude", 0, 0.1), (0.35110227911579833, 0.5887032423535256)),
                (("amplitude", 1, 0.1), (0.4113463262458407, 0.6488302504076651)),
                (("lifetime", 0, 0.1), (1.1820254668003933, 1.8848859261112236)),
                (("lifetime", 1, 0.1), (4.8197490469823405, 6.335898748874484)),
            )
        ),
        (
            "dataset_2exp_discrete",
            (
                (("amplitude", 0), (0.24136537566165966, 0.5142728935555203)),
                (("amplitude", 1), (0.48585243703506176, 0.7586685597179819)),
                (("lifetime", 0), (0.8840827827592247, 1.7536913419238422)),
                (("lifetime", 1), (4.3513926057240795, 5.7774431774946216)),
            ),
            (
                (("amplitude", 0, 0.1), (0.2595585323682838, 0.4866196042619292)),
                (("amplitude", 1, 0.1), (0.5132188827987454, 0.7403870281614857)),
                (("lifetime", 0, 0.1), (0.9394793422395256, 1.6606847211819782)),
                (("lifetime", 1, 0.1), (4.435259129659109, 5.617189775691198)),
            ),
        ),
    ],
    # fmt:on
)
def test_dwelltime_profiles(exponential_data, exp_name, reference_bounds, reinterpolated_bounds):
    dataset = exponential_data[exp_name]

    fit = DwelltimeModel(
        dataset["data"],
        n_components=2,
        **dataset["parameters"].observation_limits,
        discretization_timestep=dataset["parameters"].dt,
    )

    profiles = fit.profile_likelihood(max_chi2_step=0.25)
    for (name, component), ref_interval in reference_bounds:
        np.testing.assert_allclose(profiles.get_interval(name, component), ref_interval, rtol=1e-3)

    # Re-interpolated confidence level (different significance level than originally profiled).
    for (name, component, significance), ref_interval in reinterpolated_bounds:
        np.testing.assert_allclose(
            profiles.get_interval(name, component, significance), ref_interval, rtol=1e-3
        )

    # Significance level cannot be chosen lower than what we profiled.
    with pytest.raises(
        RuntimeError,
        match=re.escape(
            "Significance level (0.001) cannot be chosen lower or equal than the minimum profiled "
            "level (0.01)."
        ),
    ):
        profiles.get_interval("amplitude", 0, 0.001)


@pytest.mark.parametrize(
    # fmt:off
    "exp_name, n_components, ref_std_errs, tolerance",
    [
        ("dataset_2exp", 1, [np.nan, 0.117634], 1e-4),  # Amplitude is not fitted!
        ("dataset_2exp", 2, [0.072451, 0.072451, 0.212818, 0.4493979], 1e-3),
        ("dataset_2exp_discrete", 2, [0.068016, 0.068016, 0.213995 , 0.350305], 1e-3),
        # Over-fitted, hence coarse tolerances
        ("dataset_2exp_discrete", 3, [0.09755, 0.37899, 0.3933, 0.25203, 1.22622, 4.4805], 1e-1),
        ("dataset_2exp_discrete", 4, [0.097569, 3.5477e-05, 0.37874, 0.393413, 0.25203, 1.5543, 1.2278, 4.48905], 1e-1),
    ]
)
def test_std_errs(exponential_data, exp_name, n_components, ref_std_errs, tolerance):
    dataset = exponential_data[exp_name]

    fit = DwelltimeModel(
        dataset["data"],
        n_components=n_components,
        **dataset["parameters"].observation_limits,
        discretization_timestep=dataset["parameters"].dt,
    )
    np.testing.assert_allclose(fit._std_errs, ref_std_errs, rtol=tolerance)
    np.testing.assert_allclose(fit.err_amplitudes, ref_std_errs[:n_components], rtol=tolerance)
    np.testing.assert_allclose(fit.err_lifetimes, ref_std_errs[n_components:], rtol=tolerance)


@pytest.mark.parametrize("n_components", [2, 1])
def test_dwelltime_profile_plots(n_components):
    """Verify that the threshold moves appropriately"""
    fit = DwelltimeModel(
        np.array([10.0, 5.0, 4.0, 3.0, 3.0, 2.0, 2.0, 1.0]),
        n_components,
        min_observation_time=1e-4,
        max_observation_time=1e4,
    )
    profiles = fit.profile_likelihood(num_steps=2)  # Keep it short
    plt.close("all")
    profiles.plot()
    np.testing.assert_allclose(plt.gca().get_lines()[-1].get_data()[-1][-1], 40.989126)

    plt.close("all")
    profiles.plot(alpha=0.5)
    np.testing.assert_allclose(plt.gca().get_lines()[-1].get_data()[-1][-1], 37.602604)


@pytest.mark.filterwarnings("ignore:Values in x were outside bounds")
@pytest.mark.filterwarnings("ignore:divide by zero encountered in log")
def test_dwelltime_profiles_dunders(exponential_data):
    dataset = exponential_data["dataset_2exp"]
    fit = DwelltimeModel(dataset["data"], 2, **dataset["parameters"].observation_limits)
    profiles = fit.profile_likelihood(num_steps=2)  # Keep it short

    # Only check the first part of the repr, since the memory addresses change
    assert repr(profiles).startswith(
        r"DwelltimeProfiles({'amplitude 0': <lumicks.pylake.fitting.profile_likelihood"
    )

    ref_keys = ["amplitude 0", "amplitude 1", "lifetime 0", "lifetime 1"]
    for key, ref_key in zip(profiles.keys(), ref_keys):
        assert key == ref_key
        assert profiles[key] is profiles.profiles[key]
        assert profiles.get(key) is profiles.profiles.get(key)

    # Uses __iter__
    for key, ref_key in zip(list(profiles), ref_keys):
        assert key == ref_key

    for value, ref_value in zip(profiles.values(), profiles.profiles.values()):
        assert value is ref_value


# TODO: remove with deprecation
def test_empty_bootstrap(exponential_data):
    dataset = exponential_data["dataset_2exp"]

    for n_components in (1, 2):
        fit = DwelltimeModel(
            dataset["data"], n_components, **dataset["parameters"].observation_limits
        )

        bootstrap = fit._bootstrap
        assert bootstrap.n_components == fit.n_components
        assert bootstrap.n_samples == 0

    with pytest.warns(DeprecationWarning):
        with pytest.raises(
            RuntimeError,
            match=re.escape(
                "The bootstrap distribution is currently empty. Use `DwelltimeModel.calculate_bootstrap()` "
                "to sample a distribution before attempting downstream analysis."
            ),
        ):
            fit.bootstrap


def test_dwellcounts_from_statepath():
    def test_results(sp, exclude, ref_dwelltimes):
        dwells, ranges = _dwellcounts_from_statepath(sp, exclude_ambiguous_dwells=exclude)
        for key in ref_dwelltimes.keys():
            # test dwelltimes
            np.testing.assert_equal(dwells[key], ref_dwelltimes[key])
            # test slicing, ignore if empty
            if len(ranges[key]):
                values = np.hstack([sp[slice(*r)] for r in ranges[key]])
                np.testing.assert_equal(values, key)

    sp = [0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0]
    test_results(sp, True, {0: [2, 1], 1: [3, 1, 2]})
    test_results(sp, False, {0: [1, 2, 1, 5], 1: [3, 1, 2]})

    sp = [1, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1]
    test_results(sp, True, {0: [2, 1, 4], 1: [1, 2]})
    test_results(sp, False, {0: [2, 1, 4], 1: [3, 1, 2, 1]})

    sp = [0, 1, 1, 2, 0, 0, 1, 2, 2, 2, 0, 0, 0, 0, 1]
    test_results(sp, True, {0: [2, 4], 1: [2, 1], 2: [1, 3]})
    test_results(sp, False, {0: [1, 2, 4], 1: [2, 1, 1], 2: [1, 3]})

    sp = [0, 1, 1, 3, 0, 0, 1, 3, 3, 3, 0, 0, 0, 0, 1]
    test_results(sp, True, {0: [2, 4], 1: [2, 1], 3: [1, 3]})
    test_results(sp, False, {0: [1, 2, 4], 1: [2, 1, 1], 3: [1, 3]})

    sp = [0, 1, 1]
    test_results(sp, True, {0: [], 1: []})
    test_results(sp, False, {0: [1], 1: [2]})

    sp = [0, 0, 0]
    test_results(sp, True, {0: []})
    test_results(sp, False, {0: [3]})

    dwells, ranges = _dwellcounts_from_statepath([], exclude_ambiguous_dwells=True)
    assert dwells == {}
    assert ranges == {}


def test_plots(exponential_data):
    """Check if `DwelltimeModel` fits can be plotted without an exception"""
    dataset = exponential_data["dataset_2exp"]
    fit = DwelltimeModel(dataset["data"], 1, **dataset["parameters"].observation_limits)
    fit.hist()
    plt.close("all")

    np.random.seed(123)
    bootstrap = fit.calculate_bootstrap(iterations=2)
    bootstrap.hist()
    plt.close("all")

    with pytest.warns(DeprecationWarning):
        bootstrap.plot()


def test_invalid_bootstrap(exponential_data):
    dataset = exponential_data["dataset_2exp"]
    fit = DwelltimeModel(dataset["data"], 1, **dataset["parameters"].observation_limits)

    with pytest.raises(
        ValueError,
        match=re.escape("Number of parameters should be the same as the number of components (1)"),
    ):
        DwelltimeBootstrap(fit, np.zeros((3, 1)), np.zeros((3, 1)))

    with pytest.raises(
        ValueError,
        match=re.escape(
            "Number of amplitude samples (2) should be the same as number of lifetime samples (1)"
        ),
    ):
        DwelltimeBootstrap(fit, np.zeros((3, 2)), np.zeros((3, 1)))


def test_integration_dwelltime_fixing_parameters(exponential_data):
    dataset = exponential_data["dataset_2exp"]
    initial_params = np.array([0.2, 0.2, 0.5, 0.5])
    pars, log_likelihood, std_errs = _exponential_mle_optimize(
        2,
        dataset["data"],
        **dataset["parameters"].observation_limits,
        initial_guess=initial_params,
        fixed_param_mask=[False, True, False, True],
    )
    np.testing.assert_allclose(pars, [0.8, 0.2, 4.27753, 0.5], rtol=1e-4)
    np.testing.assert_allclose(std_errs, [np.nan, np.nan, 0.15625, np.nan], rtol=1e-4)


@pytest.mark.parametrize(
    "dataset, n_components, ref_discrete, ref_continuous",
    [
        (
            "dataset_1exp_discrete",
            1,
            [1, 1.512417],
            [1, 1.462961],  # True value: [1, 1.5]
        ),
        (
            "dataset_2exp_discrete",  # True values: [0.4, 0.6, 1.5, 5]
            2,
            [0.36318, 0.63682, 1.257651, 4.943089],
            [0.346897, 0.653103, 1.039719, 4.830995],
        ),
    ],
)
def test_discrete_dwelltimes(exponential_data, dataset, n_components, ref_discrete, ref_continuous):
    dataset = exponential_data[dataset]

    def fit_data(dt):
        return _exponential_mle_optimize(
            n_components,
            dataset["data"],
            **dataset["parameters"].observation_limits,
            discretization_timestep=dt,
        )

    np.testing.assert_allclose(fit_data(dataset["parameters"].dt)[0], ref_discrete, rtol=1e-4)
    np.testing.assert_allclose(fit_data(None)[0], ref_continuous, rtol=1e-4)


@pytest.mark.parametrize(
    "n_components,params,fixed_param_mask,ref_fitted,ref_const_fun,free_amplitudes,ref_par,ref_amp",
    [
        # fmt:off
        # 2 components, fix one amplitude => everything fixed in the end
        [
            2, np.array([0.3, 0.4, 0.3, 0.3]), [True, False, False, False],
            [False, False, True, True], None, 0, [0.3, 0.7, 0.3, 0.3], 0,
        ],
        # 2 components, fix both amplitudes
        [
            2, np.array([0.3, 0.7, 0.3, 0.3]), [True, True, False, False],
            [False, False, True, True], 0, 0, [0.3, 0.7, 0.3, 0.3], 0,
        ],
        # 2 components, free amplitudes
        [
            2, np.array([0.3, 0.7, 0.3, 0.3]), [False, False, True, False],
            [True, True, False, True], 0.75, 2, [0.3, 0.7, 0.3, 0.3], 2,
        ],
        # 3 components, fix one amplitude => End up with two free ones
        [
            3, np.array([0.3, 0.4, 0.2, 0.3, 0.3, 0.3]), [True, False, False, False, False, False],
            [False, True, True, True, True, True], 1.6 / 3, 2, [0.3, 0.4, 0.2, 0.3, 0.3, 0.3], 2,
        ],
        # 3 components, fix two amplitudes => Amplitudes are now fully determined
        [
            3, np.array([0.3, 0.4, 0.2, 0.3, 0.3, 0.3]), [True, True, False, False, False, False],
            [False, False, False, True, True, True], 0, 0, [0.3, 0.4, 0.3, 0.3, 0.3, 0.3], 0,
        ],
        # 1 component, no amplitudes required
        [
            1, np.array([0.3, 0.5]), [False, False],
            [False, True], None, 0, [1.0, 0.5], 0,
        ],
        # fmt:on
    ],
)
def test_parameter_fixing(
    n_components,
    params,
    fixed_param_mask,
    ref_fitted,
    ref_const_fun,
    free_amplitudes,
    ref_par,
    ref_amp,
):
    old_params = np.copy(params)
    fitted_param_mask, constraints, out_params, free_amplitudes = _handle_amplitude_constraint(
        n_components, params, np.array(fixed_param_mask)
    )
    assert free_amplitudes == ref_amp

    # Verify that we didn't modify the input
    np.testing.assert_allclose(params, old_params)

    assert np.all(fitted_param_mask == ref_fitted)
    np.testing.assert_allclose(out_params, ref_par)

    if free_amplitudes:
        np.testing.assert_allclose(constraints["args"], free_amplitudes)
        np.testing.assert_allclose(
            constraints["fun"](np.arange(2 * n_components) / (2 * n_components), free_amplitudes),
            ref_const_fun,
        )
    else:
        assert not constraints


def test_invalid_models():
    with pytest.raises(
        ValueError,
        match=re.escape(
            "Length of fixed parameter mask (4) is not equal to the number of model parameters (6)"
        ),
    ):
        _handle_amplitude_constraint(
            2, np.array([1, 0.0001, 0.0, 0.3, 0.3, 0.3]), np.array([True, True, False, False])
        )

    with pytest.raises(ValueError, match="Sum of the fixed amplitudes is bigger than 1"):
        _handle_amplitude_constraint(
            3,
            np.array([1, 0.0001, 0.1, 0.3, 0.3, 0.3]),
            np.array([True, True, False, False, False, False]),
        )

    # If all amplitudes are fixed, they have to be 1.
    with pytest.raises(ValueError, match="Sum of the provided amplitudes has to be 1"):
        _handle_amplitude_constraint(
            2, np.array([0.1, 0.1, 0.3, 0.3]), np.array([True, True, False, False])
        )

    # This should be OK though (sum to 1).
    _handle_amplitude_constraint(
        3,
        np.array([0.25, 0.25, 0.5, 0.3, 0.3, 0.3]),
        np.array([True, True, True, False, True, True]),
    )


@pytest.mark.parametrize(
    "multiple_dt",
    [False, True],
)
@pytest.mark.parametrize(
    "dt",
    [None, 0.01, 0.0001],  # 0.1,
)
@pytest.mark.parametrize(
    "params, t, min_observation_time, max_observation_time",
    [
        [[1.0, 0.4], np.arange(0.0, 10.0, 0.1), 0, np.inf],
        [[0.25, 0.75, 0.4, 1.0], np.arange(0.0, 10.0, 0.1), 0, np.inf],
        [[0.25, 0.75, 0.4, 1.0], np.arange(0.0, 10.0, 0.1), 0.5, np.inf],
        [[0.25, 0.75, 0.4, 1.0], np.arange(0.0, 10.0, 0.1), 0.5, 1e6],
        [[0.25, 0.75, 0.4, 1.0], np.arange(0.0, 10.0, 0.1), 0, 5],
        [[0.3, 0.3, 0.1, 0.4, 1.0, 10.0], np.arange(1.0, 50.0, 0.1), 2, 5],
        # Test "zero" parameters. Because we use a central differencing scheme for validating the
        # gradient we have to set the amplitude at least the finite differencing stepsize away
        # from the bound (otherwise we'd only observe half the gradient in the numerical scheme).
        [[1e-4, 1.0, 0.4, 1.0], np.arange(0.0, 10.0, 0.1), 0, np.inf],
        # Zero lifetime is problematic because of all the reciprocals.
        [[0.4, 0.6, 1e-2, 1.0], np.arange(0.0, 10.0, 0.1), 0, np.inf],
        # Multi-t_min case
        [[0.4, 0.6, 1e-2, 1.0], np.arange(0.0, 10.0, 0.1), np.arange(0.0, 10.0, 0.1) / 10, np.inf],
        # Multi-t_max case
        [[0.4, 0.6, 1e-2, 1.0], np.arange(0.0, 10.0, 0.1), 0, 1.0 + np.arange(0.0, 10.0, 0.1) * 2],
        # Mix of infinity and values for t_max
        [[0.4, 0.6, 1e-2, 1.0], np.array([1.0, 2.0, 3.0]), 0, np.array([3.0, np.inf, 5.0])],
    ],
)
def test_analytic_gradient_exponential(
    params, t, min_observation_time, max_observation_time, dt, multiple_dt
):
    dt = dt * np.arange(1, len(t) + 1) if (multiple_dt and dt is not None) else dt

    def fn(params):
        return np.atleast_1d(
            _exponential_mixture_log_likelihood(
                np.array(params), t, min_observation_time, max_observation_time, dt
            )
        )

    np.testing.assert_allclose(
        _exponential_mixture_log_likelihood_jacobian(
            np.array(params), t, min_observation_time, max_observation_time, dt
        ),
        numerical_jacobian(fn, params, dx=1e-5).flatten(),
        rtol=1e-5,
        atol=1e-7,
    )


def test_analytic_gradient_exponential_used(monkeypatch):
    """Verify that the dwell time model actually uses the gradient"""

    def store_args(*args, **kwargs):
        raise StopIteration

    with monkeypatch.context() as m:
        # Jacobian should be passed
        model = DwelltimeModel(np.arange(0.0, 10.0, 0.1), n_components=2, use_jacobian=True)

        m.setattr(
            "lumicks.pylake.population.dwelltime._exponential_mixture_log_likelihood_jacobian",
            store_args,
        )

        # Jacobian should be passed
        with pytest.raises(StopIteration):
            model.calculate_bootstrap(1)

        # Jacobian should not be passed
        model = DwelltimeModel(np.arange(0.0, 10.0, 0.1), n_components=2, use_jacobian=False)
        model.calculate_bootstrap(1)

        # Jacobian should be passed
        with pytest.raises(StopIteration):
            DwelltimeModel(np.arange(0.0, 10.0, 0.1), n_components=2, use_jacobian=True)


def test_dwelltime_exponential_no_free_params(monkeypatch):
    """When fitting a single parameter, we don't need to optimize"""

    def stop(*args, **kwargs):
        raise StopIteration

    with monkeypatch.context() as m:
        m.setattr("scipy.optimize.minimize", stop)

        def quick_fit(fixed_params):
            return _exponential_mle_optimize(
                1,
                np.arange(5),
                min_observation_time=0,
                max_observation_time=5,
                initial_guess=np.array([1.0, 2.0]),
                fixed_param_mask=fixed_params,
            )

        x, cost, std_err = quick_fit(
            [False, True]
        )  # Amplitude is 1 -> Problem fully determined -> No fit
        np.testing.assert_allclose(x, np.array([1.0, 2.0]))
        np.testing.assert_equal(std_err, [np.nan, np.nan])  # no uncertainty estimates

        with pytest.raises(StopIteration):
            quick_fit([True, False])  # Lifetime unknown -> Need to fit


def test_bad_std_err():
    """This tests a case where one amplitude is specifically expected to collapse. If not handled
    appropriately, this will spit out many errors."""
    with temp_seed(10):
        dwells = np.hstack((np.random.exponential(1, 10000), 5000))
        _ = DwelltimeModel(dwells, n_components=2)
