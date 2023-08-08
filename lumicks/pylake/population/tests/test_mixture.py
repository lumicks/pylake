import numpy as np
import pytest
import matplotlib.pyplot as plt

from lumicks.pylake import GaussianMixtureModel
from lumicks.pylake.channel import Slice, Continuous


def test_gmm(trace_lownoise):
    data, statepath, params = trace_lownoise
    m = GaussianMixtureModel(
        data, params["n_states"], init_method="kmeans", n_init=1, tol=1e-3, max_iter=100
    )
    weights = np.array([np.sum(statepath == j) for j in np.arange(params["n_states"])])
    weights = weights / weights.sum()

    # assert np.all(np.equal(m.label(trace), statepath))
    np.testing.assert_allclose(m.means, params["means"], atol=0.05)
    np.testing.assert_allclose(m.std, params["st_devs"], atol=0.02)
    np.testing.assert_allclose(m.weights, weights)


def test_gmm_from_slice(trace_simple):
    data, statepath, params = trace_simple
    trace = Slice(Continuous(data, 20000, 12800))
    m = GaussianMixtureModel.from_channel(trace, params["n_states"])
    np.testing.assert_allclose(m.means, params["means"], atol=0.05)


def test_labels(trace_simple):
    data, statepath, params = trace_simple
    m = GaussianMixtureModel(
        data, params["n_states"], init_method="kmeans", n_init=1, tol=1e-3, max_iter=100
    )
    labels = m.label(Slice(Continuous(data, 20000, 12800)))
    np.testing.assert_equal(labels, statepath)


def test_dwelltimes(trace_simple):
    data, statepath, params = trace_simple
    m = GaussianMixtureModel(
        data, params["n_states"], init_method="kmeans", n_init=1, tol=1e-3, max_iter=100
    )

    channel = Slice(Continuous(data, 20000, 12800))

    dwell_times = m.extract_dwell_times(channel, exclude_ambiguous_dwells=False)
    np.testing.assert_allclose(dwell_times[0], [3.200e-04, 2.432e-04, 5.120e-05])
    np.testing.assert_allclose(dwell_times[1], [1.664e-04, 3.456e-04, 3.840e-05, 1.152e-04])

    dwell_times = m.extract_dwell_times(channel, exclude_ambiguous_dwells=True)
    np.testing.assert_allclose(dwell_times[0], [3.200e-04, 2.432e-04, 5.120e-05])
    np.testing.assert_allclose(dwell_times[1], [3.456e-04, 3.840e-05])


def test_information_criteria(trace_simple):
    data, statepath, params = trace_simple
    m = GaussianMixtureModel(
        data, params["n_states"], init_method="kmeans", n_init=1, tol=1e-3, max_iter=100
    )

    np.testing.assert_allclose(m.bic, -20.04115465)
    np.testing.assert_allclose(m.aic, -33.06700559)


def test_exit_flag(trace_simple):
    data, statepath, params = trace_simple
    m = GaussianMixtureModel(
        data, params["n_states"], init_method="kmeans", n_init=1, tol=1e-3, max_iter=100
    )

    ef = m.exit_flag
    assert ef["converged"] == True
    assert ef["n_iter"] == 2
    np.testing.assert_allclose(ef["lower_bound"], 0.215335, rtol=1e-5)


def test_pdf(trace_simple):
    data, statepath, params = trace_simple
    m = GaussianMixtureModel(
        data, params["n_states"], init_method="kmeans", n_init=1, tol=1e-3, max_iter=100
    )
    np.testing.assert_allclose(
        m.pdf(np.array([10, 11])), [[1.857758, 5e-26], [5e-21, 2.222931]], rtol=1e-5, atol=1e-13
    )


def test_gmm_plots(trace_simple):
    data, statepath, params = trace_simple
    m = GaussianMixtureModel(
        data, params["n_states"], init_method="kmeans", n_init=1, tol=1e-3, max_iter=100
    )
    trace = Slice(Continuous(data, 20000, 12800))

    m.hist(trace)
    plt.close()

    m.plot(trace)
    plt.close()
