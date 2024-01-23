from dataclasses import dataclass

import numpy as np
import pytest

from lumicks.pylake import HiddenMarkovModel, GaussianMixtureModel
from lumicks.pylake.channel import Slice, Continuous
from lumicks.pylake.population.detail.hmm import ClassicHmm
from lumicks.pylake.population.detail.fit_info import PopulationFitInfo


def make_channel(data):
    return Slice(Continuous(data, np.int64(20e9), np.int64(1 / 78125 * 1e9)))


def test_hmm(trace_lownoise):
    data, statepath, params = trace_lownoise
    n_states = params["n_states"]

    # fmt: off
    ref_transition_matrices = {
        2: [
            [0.9375, 0.0625],
            [0.05882353, 0.94117647],
        ],
        3: [
            [0.85714286, 0.07142857, 0.07142857],
            [0.04545455, 0.90909091, 0.04545455],
            [0.07407407, 0.11111111, 0.81481481],
        ],
        4: [
            [0.86363636, 0.0, 0.04545455, 0.09090909],
            [0.0, 0.85185185, 0.14814815, 0.0],
            [0.03448276, 0.06896552, 0.79310345, 0.10344828],
            [0.0952381, 0.14285714, 0.0, 0.76190476],
        ],
    }
    # fmt: on

    for data in (data, make_channel(data)):
        model = HiddenMarkovModel(data, n_states)

        # test model emission parameters converge to model used to generate test data
        np.testing.assert_allclose(model.means, params["means"], atol=0.032)
        np.testing.assert_allclose(model.std, params["st_devs"], atol=0.02)

        # there are too few transitions in the trace for the model to converge to the transition
        # matrix used to generate the data; would need to use an atol on the order of 0.1
        # test against hard-coded values instead to guard against unintended changes to algo
        np.testing.assert_allclose(model.transition_matrix, ref_transition_matrices[n_states])

        # because only one trace is fit, the initial state proability collapses to the
        # actual observed first state
        ref_pi = np.zeros(n_states)
        ref_pi[statepath[0]] = 1
        np.testing.assert_allclose(model.initial_state_probability, ref_pi, atol=1e-10)


def test_initial_guess(trace_simple):
    data, statepath, params = trace_simple
    n_states = params["n_states"]
    trace = make_channel(data)

    # no initial guess
    model = HiddenMarkovModel(trace, n_states)

    # GMM initial guess
    gmm_guess = GaussianMixtureModel(trace, n_states)
    model = HiddenMarkovModel(trace, n_states, initial_guess=gmm_guess)

    # HMM initial guess
    model._model = ClassicHmm.guess(trace.data, n_states, gmm=gmm_guess)
    model = HiddenMarkovModel(trace, n_states, initial_guess=model)

    # anything else raises
    with pytest.raises(
        TypeError,
        match=(
            "if provided, `initial_guess` must be either GaussianMixtureModel or "
            "HiddenMarkovModel, got `str`."
        ),
    ):
        model = HiddenMarkovModel(trace, n_states, initial_guess="hello")

    # initial guess must have correct number of states
    with pytest.raises(
        ValueError,
        match=(
            "Initial guess must have the same number of states as requested for the current model; "
            f"expected {n_states} got {n_states + 1}."
        ),
    ):
        bad_gmm_guess = GaussianMixtureModel(trace, n_states + 1)
        model = HiddenMarkovModel(trace, n_states, initial_guess=bad_gmm_guess)

    with pytest.raises(
        ValueError,
        match=(
            "Initial guess must have the same number of states as requested for the current model; "
            f"expected {n_states} got {n_states+1}."
        ),
    ):
        bad_hmm_guess = HiddenMarkovModel(trace, n_states + 1)
        model = HiddenMarkovModel(trace, n_states, initial_guess=bad_hmm_guess)


def test_state_path(trace_simple):
    data, ref_statepath, params = trace_simple
    trace = make_channel(data)
    trace.labels = {"title": "mock", "y": "Force (pN)"}
    original_labels = trace.labels.copy()

    model = HiddenMarkovModel(trace, params["n_states"])
    state_path = model.state_path(trace)
    np.testing.assert_equal(state_path.data, ref_statepath)

    # no changes to original channel instance
    np.testing.assert_equal(trace.data, data)
    assert id(trace.labels) != id(state_path.labels)
    for key, original_value in original_labels.items():
        if key == "y":
            assert original_value == "Force (pN)"
            assert state_path.labels[key] == "state"
        else:
            assert state_path.labels[key] == original_value


def test_emission_path(trace_simple):
    data, statepath, params = trace_simple
    trace = make_channel(data)
    trace.labels = {"title": "mock", "y": "Force (pN)"}
    original_labels = trace.labels.copy()

    model = HiddenMarkovModel(trace, params["n_states"])
    emission_path = model.emission_path(trace)
    np.testing.assert_equal(emission_path.data, model.means[statepath])

    # no changes to original channel instance
    np.testing.assert_equal(trace.data, data)
    assert id(trace.labels) != id(emission_path.labels)
    for key, original_value in original_labels.items():
        assert emission_path.labels[key] == original_value


def test_dwelltimes(trace_simple):
    data, _, params = trace_simple
    trace = make_channel(data)

    model = HiddenMarkovModel(trace, params["n_states"])

    dwell_times = model.extract_dwell_times(trace, exclude_ambiguous_dwells=False)
    np.testing.assert_allclose(dwell_times[0], [3.200e-04, 2.432e-04, 5.120e-05])
    np.testing.assert_allclose(dwell_times[1], [1.664e-04, 3.456e-04, 3.840e-05, 1.152e-04])

    dwell_times = model.extract_dwell_times(trace, exclude_ambiguous_dwells=True)
    np.testing.assert_allclose(dwell_times[0], [3.200e-04, 2.432e-04, 5.120e-05])
    np.testing.assert_allclose(dwell_times[1], [3.456e-04, 3.840e-05])


def test_fit_info(trace_simple):
    data, _, params = trace_simple

    model = HiddenMarkovModel(data, params["n_states"])

    assert isinstance(model.fit_info, PopulationFitInfo)
    assert model.fit_info.converged
    assert model.fit_info.n_iter == 2
    np.testing.assert_allclose(model.fit_info.bic, -104.03696527084878)
    np.testing.assert_allclose(model.fit_info.aic, -122.27315657276543)
    np.testing.assert_allclose(model.fit_info.log_likelihood, 68.13657828638271)


def test_plots(trace_simple):
    import matplotlib.pyplot as plt

    data, _, params = trace_simple
    trace = make_channel(data)
    model = HiddenMarkovModel(trace, params["n_states"])

    model.hist(trace)
    plt.close()

    model.plot(trace)
    plt.close()
