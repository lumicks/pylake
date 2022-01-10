import pytest
import numpy as np

from lumicks.pylake.kymotracker.detail.binding_times import _kinetic_mle_optimize


@pytest.mark.filterwarnings("ignore:Values in x were outside bounds")
def test_likelihood(exponential_data):
    # single exponential data
    dataset = exponential_data["dataset_1exp"]
    fit = _kinetic_mle_optimize(1, dataset["data"], *dataset["parameters"].observation_limits)
    np.testing.assert_allclose(fit.log_likelihood, -1290.2636036948977, rtol=1e-5)
    np.testing.assert_allclose(fit.aic, 2582.5272073897954, rtol=1e-5)
    np.testing.assert_allclose(fit.bic, 2587.3815618920503, rtol=1e-5)

    # double exponential data
    dataset = exponential_data["dataset_2exp"]
    fit = _kinetic_mle_optimize(2, dataset["data"], *dataset["parameters"].observation_limits)
    np.testing.assert_allclose(fit.log_likelihood, -2204.2954468395105, rtol=1e-5)
    np.testing.assert_allclose(fit.aic, 4414.590893679021, rtol=1e-5)
    np.testing.assert_allclose(fit.bic, 4429.232045925579, rtol=1e-5)


@pytest.mark.filterwarnings("ignore:Values in x were outside bounds")
def test_fit_parameters(exponential_data):
    # single exponential data
    dataset = exponential_data["dataset_1exp"]
    fit = _kinetic_mle_optimize(1, dataset["data"], *dataset["parameters"].observation_limits)
    np.testing.assert_allclose(fit.amplitudes, [1])
    np.testing.assert_allclose(fit.lifetimes, [1.43481181], rtol=1e-5)

    # double exponential data
    dataset = exponential_data["dataset_2exp"]
    fit = _kinetic_mle_optimize(2, dataset["data"], *dataset["parameters"].observation_limits)
    np.testing.assert_allclose(fit.amplitudes, [0.46513486, 0.53486514], rtol=1e-5)
    np.testing.assert_allclose(fit.lifetimes, [1.50630877, 5.46212603], rtol=1e-5)


@pytest.mark.slow
@pytest.mark.filterwarnings("ignore:Values in x were outside bounds")
def test_bootstrap(exponential_data):
    # double exponential data
    dataset = exponential_data["dataset_2exp"]
    fit = _kinetic_mle_optimize(2, dataset["data"], *dataset["parameters"].observation_limits)

    np.random.seed(123)
    fit.calculate_bootstrap(iterations=50)
    mean, ci = fit.bootstrap.calculate_stats("amplitude", 0)
    np.testing.assert_allclose(mean, 0.4642469883372174, rtol=1e-5)
    np.testing.assert_allclose(ci, (0.3647038711684928, 0.5979550940729152), rtol=1e-5)
    np.random.seed()
