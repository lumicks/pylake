import pytest
import numpy as np
import re

from lumicks.pylake import DwelltimeModel
from lumicks.pylake.fitting.detail.derivative_manipulation import numerical_jacobian
from lumicks.pylake.population.dwelltime import (
    _dwellcounts_from_statepath,
    DwelltimeBootstrap,
    _handle_amplitude_constraint,
    _exponential_mle_optimize,
    exponential_mixture_log_likelihood,
    _exponential_mixture_log_likelihood_gradient
)


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


def test_optim_options(exponential_data):
    dataset = exponential_data["dataset_1exp"]

    fit = DwelltimeModel(dataset["data"], 1, **dataset["parameters"].observation_limits, tol=1e-1)
    np.testing.assert_allclose(fit.lifetimes, [1.442235], rtol=1e-5)

    fit = DwelltimeModel(dataset["data"], 1, **dataset["parameters"].observation_limits, max_iter=2)
    np.testing.assert_allclose(fit.lifetimes, [1.382336], rtol=1e-5)


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

    np.random.seed(123)
    bootstrap = fit.calculate_bootstrap(iterations=2)
    bootstrap.hist()

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
    pars, log_likelihood = _exponential_mle_optimize(
        2, dataset["data"],
        **dataset["parameters"].observation_limits,
        initial_guess=initial_params,
        fixed_param_mask=[False, True, False, True],
    )
    np.testing.assert_allclose(pars, [0.8, 0.2, 4.27753, 0.5], rtol=1e-4)


@pytest.mark.parametrize(
    "n_components,params,fixed_param_mask,ref_fitted,ref_const_fun,free_amplitudes,ref_par",
    [
        # fmt:off
        # 2 components, fix one amplitude => everything fixed in the end
        [
            2, [0.3, 0.4, 0.3, 0.3], [True, False, False, False],
            [False, False, True, True], None, 0, [0.3, 0.7, 0.3, 0.3],
        ],
        # 2 components, fix both amplitudes
        [
            2, [0.3, 0.7, 0.3, 0.3], [True, True, False, False],
            [False, False, True, True], 0, 0, [0.3, 0.7, 0.3, 0.3]
        ],
        # 2 components, free amplitudes
        [
            2, [0.3, 0.7, 0.3, 0.3], [False, False, True, False],
            [True, True, False, True], 0.75, 2, [0.3, 0.7, 0.3, 0.3],
        ],
        # 3 components, fix one amplitude => End up with two free ones
        [
            3, [0.3, 0.4, 0.2, 0.3, 0.3, 0.3], [True, False, False, False, False, False],
            [False, True, True, True, True, True], 1.6 / 3, 2, [0.3, 0.4, 0.2, 0.3, 0.3, 0.3],
        ],
        # 3 components, fix two amplitudes => Amplitudes are now fully determined
        [
            3, [0.3, 0.4, 0.2, 0.3, 0.3, 0.3], [True, True, False, False, False, False],
            [False, False, False, True, True, True], 0, 0, [0.3, 0.4, 0.3, 0.3, 0.3, 0.3],
        ],
        # 1 component, no amplitudes required
        [
            1, [0.3, 0.5], [False, False],
            [False, True], None, 0, [1.0, 0.5],
        ],
        # fmt:off
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
):
    fitted_param_mask, constraints, params = _handle_amplitude_constraint(
        n_components, np.array(params), np.array(fixed_param_mask)
    )

    assert np.all(fitted_param_mask == ref_fitted)
    np.testing.assert_allclose(params, ref_par)

    if free_amplitudes:
        np.testing.assert_allclose(constraints["args"], free_amplitudes)
        np.testing.assert_allclose(
            constraints["fun"](np.arange(2 * n_components) / (2 * n_components), free_amplitudes),
            ref_const_fun
        )
    else:
        assert not constraints


def test_invalid_models():
    with pytest.raises(
        ValueError,
        match=re.escape(
            "Length of fixed parameter mask (4) is not equal to the number of model parameters (6)"
        )
    ):
        _handle_amplitude_constraint(
            2, np.array([1, 0.0001, 0.0, 0.3, 0.3, 0.3]), np.array([True, True, False, False])
        )

    with pytest.raises(ValueError, match="Sum of the fixed amplitudes is bigger than 1"):
        _handle_amplitude_constraint(
            3,
            np.array([1, 0.0001, 0.1, 0.3, 0.3, 0.3]),
            np.array([True, True, False, False, False, False])
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
        np.array([True, True, True, False, True, True])
    )


@pytest.mark.parametrize(
    "params, t, min_observation_time, max_observation_time",
    [
        [[1.0, 0.4], np.arange(0.0, 10.0, 0.1), 0, np.inf],
        [[0.25, 0.75, 0.4, 1.0], np.arange(0.0, 10.0, 0.1), 0, np.inf],
        [[0.25, 0.75, 0.4, 1.0], np.arange(0.0, 10.0, 0.1), 0.5, np.inf],
        [[0.25, 0.75, 0.4, 1.0], np.arange(0.0, 10.0, 0.1), 0.5, 1e6],
        [[0.25, 0.75, 0.4, 1.0], np.arange(0.0, 10.0, 0.1), 0, 5],
        [[0.3, 0.3, 0.1, 0.4, 1.0, 10.0], np.arange(1.0, 10.0, 0.1), 2, 5],
        # Test "zero" parameters. Because we use a central differencing scheme for validating the
        # gradient we have to set the amplitude at least the finite differencing stepsize away
        # from the bound (otherwise we'd only observe half the gradient in the numerical scheme).
        [[1e-4, 1.0, 0.4, 1.0], np.arange(0.0, 10.0, 0.1), 0, np.inf],
        # Zero lifetime is problematic because of all the reciprocals.
        [[0.4, 0.6, 1e-2, 1.0], np.arange(0.0, 10.0, 0.1), 0, np.inf],
    ]
)
def test_analytic_gradient_exponential(params, t, min_observation_time, max_observation_time):
    def fn(params):
        return np.atleast_1d(
            exponential_mixture_log_likelihood(
                np.array(params), t, min_observation_time, max_observation_time
            )
        )

    np.testing.assert_allclose(
        _exponential_mixture_log_likelihood_gradient(
            np.array(params), t, min_observation_time, max_observation_time
        ),
        numerical_jacobian(fn, params, dx=1e-5).flatten(),
        rtol=1e-5,
    )


def test_analytic_gradient_exponential_used(monkeypatch):
    """Verify that the dwell time model actually uses the gradient"""

    def store_args(*args, **kwargs):
        raise StopIteration

    with monkeypatch.context() as m:
        # Jacobian should be passed
        model = DwelltimeModel(np.arange(0.0, 10.0, 0.1), n_components=2, use_jacobian=True)

        m.setattr(
            "lumicks.pylake.population.dwelltime._exponential_mixture_log_likelihood_gradient",
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
        m.setattr("lumicks.pylake.population.dwelltime.minimize", stop)

        def quick_fit(fixed_params):
            return _exponential_mle_optimize(
                1,
                np.arange(5),
                min_observation_time=0,
                max_observation_time=5,
                initial_guess=np.array([1.0, 2.0]),
                fixed_param_mask=fixed_params,
            )

        x, cost = quick_fit([False, True])  # Amplitude is 1 -> Problem fully determined -> No fit
        np.testing.assert_allclose(x, np.array([1.0, 2.0]))

        with pytest.raises(StopIteration):
            quick_fit([True, False])  # Lifetime unknown -> Need to fit
