import numpy as np
import pytest

from lumicks.pylake import GaussianMixtureModel
from lumicks.pylake.population.detail.hmm import (
    ClassicHmm,
    viterbi,
    normalize_rows,
    forward_backward,
    calculate_temporary_variables,
)


@pytest.fixture(scope="module")
def test_data():
    return np.array([1, 1, 1, 2, 2, 1, 1, 1]) + np.random.normal(0, scale=0.01, size=8)


def test_classic_hmm(trace_simple):
    data, _, params = trace_simple

    for gmm in (
        None,
        GaussianMixtureModel(
            data, params["n_states"], init_method="kmeans", n_init=1, tol=1e-3, max_iter=100
        ),
    ):
        guess = ClassicHmm.guess(data, params["n_states"], gmm=gmm)

        assert guess.K == params["n_states"]
        np.testing.assert_allclose(guess.mu, [9.99752139, 11.00902636])
        np.testing.assert_allclose(guess.tau, [94.173207, 115.911592])
        np.testing.assert_allclose(guess.pi, [0.5, 0.5])
        np.testing.assert_allclose(guess.A, [[0.9375, 0.0625], [0.05882353, 0.94117647]])

        np.testing.assert_allclose(
            guess.state_log_likelihood([10, 11]),
            [[1.35334005, -45.96668197], [-57.54930288, 1.45275339]],
        )


def test_forward_backward_vectorization(test_data):
    model = ClassicHmm.guess(test_data, 2)
    alpha, beta, c, B = forward_backward(test_data, model)

    # B comes from the internal model instance, needs to be exponentiated for this algo
    np.testing.assert_allclose(B, np.exp(model.state_log_likelihood(test_data).T))

    # alpha is row-wise normalized for scaling
    np.testing.assert_equal(alpha.sum(axis=1), 1)

    # beta is initialized (from the end) to 1
    # Rabiner Eqn 24
    np.testing.assert_equal(beta[-1], 1)

    # test against non-vectorized equation for initialization, first observation (t=0)
    # Rabiner Eqn 18 (unscaled)
    fwd_init = np.array([model.pi[i] * B[0, i] for i in range(model.K)])
    fwd_init = fwd_init / fwd_init.sum()
    np.testing.assert_allclose(alpha[0], fwd_init)

    # test against non-vectorized equation for induction, first time step (t=1)
    # Rabiner Eqn 19 (unscaled)
    fwd_ind = np.array(
        [
            np.sum([alpha[0, j] * model.A[j, i] for j in range(model.K)]) * B[1, i]
            for i in range(model.K)
        ]
    )
    fwd_ind = fwd_ind / fwd_ind.sum()
    np.testing.assert_allclose(alpha[1], fwd_ind)

    # test against non-vectorized equation for backward induction (t=T-1)
    # Rabiner Eqn 25 (unscaled)
    back_ind = [
        np.sum([model.A[i, j] * 1 * B[-1, j] for j in range(model.K)]) / c[-1]
        for i in range(model.K)
    ]
    np.testing.assert_allclose(beta[-2], back_ind)


def test_temp_variables(test_data):
    model = ClassicHmm.guess(test_data, 2)
    alpha, beta, c, B = forward_backward(test_data, model)
    gamma, xi, log_likelihood = calculate_temporary_variables(model, alpha, beta, c, B)

    np.testing.assert_equal(gamma.sum(axis=1), 1)

    # test again non-vectorized equation
    # Rabiner Eqn 37 (unscaled)
    xi_ref = np.stack(
        [
            np.vstack(
                [
                    [
                        alpha[t, i] * model.A[i, j] * B[t + 1, j] * beta[t + 1, j] / c[t + 1]
                        for j in range(model.K)
                    ]
                    for i in range(model.K)
                ]
            )
            for t in range(len(test_data) - 1)
        ],
        axis=0,
    )
    np.testing.assert_allclose(xi, xi_ref)

    # Rabiner Eqn 38
    np.testing.assert_allclose(gamma[:-1], np.sum(xi, axis=2))

    # Rabiner Eqn 103
    np.testing.assert_allclose(log_likelihood, np.sum(np.log(c)))


def test_viterbi_with_zeros(trace_simple):
    """test viterbi algorithm with zeros in initial state probability or transition matrix.
    Should not warn (invalid divide from log(0))"""
    data, _, params = trace_simple
    model = ClassicHmm(
        params["n_states"],
        params["means"],
        params["st_devs"],
        [1, 0],
        normalize_rows(params["transition_prob"]),
    )

    viterbi(data, model)

    model = ClassicHmm(
        params["n_states"],
        params["means"],
        params["st_devs"],
        params["initial_state_prob"],
        [[1, 0], [0.1, 0.9]],
    )

    viterbi(data, model)
