import pytest
import numpy as np
import re

from lumicks.pylake import DwelltimeModel
from lumicks.pylake.population.dwelltime import _dwellcounts_from_statepath


@pytest.mark.filterwarnings("ignore:Values in x were outside bounds")
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


@pytest.mark.filterwarnings("ignore:Values in x were outside bounds")
def test_optim_options(exponential_data):
    dataset = exponential_data["dataset_1exp"]

    fit = DwelltimeModel(dataset["data"], 1, **dataset["parameters"].observation_limits, tol=1e-1)
    np.testing.assert_allclose(fit.lifetimes, [1.442235], rtol=1e-5)

    fit = DwelltimeModel(dataset["data"], 1, **dataset["parameters"].observation_limits, max_iter=2)
    np.testing.assert_allclose(fit.lifetimes, [1.382336], rtol=1e-5)


@pytest.mark.filterwarnings("ignore:Values in x were outside bounds")
def test_fit_parameters(exponential_data):
    # single exponential data
    dataset = exponential_data["dataset_1exp"]
    fit = DwelltimeModel(dataset["data"], 1, **dataset["parameters"].observation_limits)
    np.testing.assert_allclose(fit.amplitudes, [1])
    np.testing.assert_allclose(fit.lifetimes, [1.43481181], rtol=1e-5)
    np.testing.assert_allclose(fit.rate_constants, [1 / 1.43481181], rtol=1e-5)

    # double exponential data
    dataset = exponential_data["dataset_2exp"]
    fit = DwelltimeModel(dataset["data"], 2, **dataset["parameters"].observation_limits)
    np.testing.assert_allclose(fit.amplitudes, [0.46513486, 0.53486514], rtol=1e-5)
    np.testing.assert_allclose(fit.lifetimes, [1.50630877, 5.46212603], rtol=1e-5)
    np.testing.assert_allclose(fit.rate_constants, [1 / 1.50630877, 1 / 5.46212603], rtol=1e-5)


@pytest.mark.slow
@pytest.mark.filterwarnings("ignore:Values in x were outside bounds")
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
@pytest.mark.filterwarnings("ignore:Values in x were outside bounds")
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
            )
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


@pytest.mark.filterwarnings("ignore:Values in x were outside bounds")
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
