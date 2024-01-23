import numpy as np
import pytest

from lumicks.pylake import GaussianMixtureModel
from lumicks.pylake.channel import Slice, Continuous
from lumicks.pylake.population.detail.fit_info import PopulationFitInfo


def make_channel(data, labels=None):
    return Slice(
        Continuous(data, np.int64(20e9), np.int64(1 / 78125 * 1e9)),
        labels={} if labels is None else labels,
    )


def test_gmm(trace_lownoise):
    data, statepath, params = trace_lownoise
    for data in (data, make_channel(data)):
        m = GaussianMixtureModel(data, params["n_states"])
        weights = np.array([np.sum(statepath == j) for j in np.arange(params["n_states"])])
        weights = weights / weights.sum()

        np.testing.assert_allclose(m.means, params["means"], atol=0.05)
        np.testing.assert_allclose(m.std, params["st_devs"], atol=0.02)
        np.testing.assert_allclose(m.weights, weights)


def test_gmm_from_slice(trace_simple):
    data, _, params = trace_simple
    trace = make_channel(data)

    with pytest.warns(DeprecationWarning):
        m = GaussianMixtureModel.from_channel(trace, params["n_states"])
        np.testing.assert_allclose(m.means, params["means"], atol=0.05)


def test_labels(trace_simple):
    data, statepath, params = trace_simple
    m = GaussianMixtureModel(
        data, params["n_states"], init_method="kmeans", n_init=1, tol=1e-3, max_iter=100
    )
    with pytest.warns(DeprecationWarning):
        labels = m.label(make_channel(data))
        np.testing.assert_equal(labels, statepath)


def test_state_path(trace_simple):
    data, ref_statepath, params = trace_simple
    trace = make_channel(data, labels={"title": "mock", "y": "Force (pN)"})
    m = GaussianMixtureModel(
        data, params["n_states"], init_method="kmeans", n_init=1, tol=1e-3, max_iter=100
    )

    statepath = m.state_path(trace)
    np.testing.assert_equal(statepath.data, ref_statepath)

    np.testing.assert_equal(trace.data, data)

    assert statepath.labels["y"] == "state"
    assert trace.labels["y"] == "Force (pN)"
    assert statepath.labels["title"] == "mock"
    assert trace.labels["title"] == "mock"


def test_dwelltimes(trace_simple):
    data, _, params = trace_simple
    m = GaussianMixtureModel(
        data, params["n_states"], init_method="kmeans", n_init=1, tol=1e-3, max_iter=100
    )

    channel = make_channel(data)

    dwell_times = m.extract_dwell_times(channel, exclude_ambiguous_dwells=False)
    np.testing.assert_allclose(dwell_times[0], [3.200e-04, 2.432e-04, 5.120e-05])
    np.testing.assert_allclose(dwell_times[1], [1.664e-04, 3.456e-04, 3.840e-05, 1.152e-04])

    dwell_times = m.extract_dwell_times(channel, exclude_ambiguous_dwells=True)
    np.testing.assert_allclose(dwell_times[0], [3.200e-04, 2.432e-04, 5.120e-05])
    np.testing.assert_allclose(dwell_times[1], [3.456e-04, 3.840e-05])


def test_information_criteria(trace_simple):
    data, _, params = trace_simple
    m = GaussianMixtureModel(
        data, params["n_states"], init_method="kmeans", n_init=1, tol=1e-3, max_iter=100
    )

    with pytest.warns(DeprecationWarning):
        np.testing.assert_allclose(m.bic, -20.04115465)

    with pytest.warns(DeprecationWarning):
        np.testing.assert_allclose(m.aic, -33.06700559)


def test_exit_flag(trace_simple):
    data, _, params = trace_simple
    m = GaussianMixtureModel(
        data, params["n_states"], init_method="kmeans", n_init=1, tol=1e-3, max_iter=100
    )

    with pytest.warns(DeprecationWarning):
        ef = m.exit_flag
        assert ef["converged"] == True
        assert ef["n_iter"] == 2
        np.testing.assert_allclose(ef["lower_bound"], 0.215335, rtol=1e-5)


def test_log_likelihood_calculation(trace_simple):
    """We calculate the log likelihood ourselves using the sklearn model's `score_samples` method
    (which according to their docs calculates the log likelihood for each data point) and then take
    the sum. A simple way to validate this is to use the value to calculate the BIC and compare
    it to the BIC returned from the sklearn model instance."""

    data, _, params = trace_simple
    model = GaussianMixtureModel(
        data, params["n_states"], init_method="kmeans", n_init=1, tol=1e-3, max_iter=100
    )

    n_states = model.n_states
    k = n_states * 3 - 1  # number of parameters
    n = len(data)  # num data points
    bic = k * np.log(n) - (2 * model.fit_info.log_likelihood)
    np.testing.assert_equal(bic, model.fit_info.bic)


def test_fit_info(trace_simple):
    data, _, params = trace_simple
    model = GaussianMixtureModel(
        data, params["n_states"], init_method="kmeans", n_init=1, tol=1e-3, max_iter=100
    )

    assert isinstance(model.fit_info, PopulationFitInfo)
    assert model.fit_info.converged
    assert model.fit_info.n_iter == 2
    np.testing.assert_allclose(model.fit_info.bic, -20.04115465)
    np.testing.assert_allclose(model.fit_info.aic, -33.06700559)
    np.testing.assert_allclose(model.fit_info.log_likelihood, 21.533503, rtol=1e-5)


def test_pdf(trace_simple):
    data, _, params = trace_simple
    m = GaussianMixtureModel(
        data, params["n_states"], init_method="kmeans", n_init=1, tol=1e-3, max_iter=100
    )
    np.testing.assert_allclose(
        m.pdf(np.array([10, 11])), [[1.857758, 5e-26], [5e-21, 2.222931]], rtol=1e-5, atol=1e-13
    )


def test_gmm_plots(trace_simple):
    import matplotlib.pyplot as plt

    data, _, params = trace_simple
    m = GaussianMixtureModel(
        data, params["n_states"], init_method="kmeans", n_init=1, tol=1e-3, max_iter=100
    )
    trace = make_channel(data)

    m.hist(trace)
    plt.close()

    m.plot(trace)
    plt.close()
