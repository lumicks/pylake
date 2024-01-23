"""
Parameters for internal API here conform to notation in Rabiner

References
----------
.. [1] Rabiner, L. A Tutorial on Hidden Markov Models and Selected Applications in Speech
        Recognition. Proceedings of IEEE. 77, 257-286 (1989).
"""

import warnings
from dataclasses import dataclass

import numpy as np

from .mixin import LatentVariableModel
from ..mixture import GaussianMixtureModel
from .fit_info import PopulationFitInfo
from ...channel import Slice, Continuous
from .validators import col, row


def normalize_rows(matrix):
    return matrix / col(np.sum(matrix, axis=1))


@dataclass(frozen=True)
class ClassicHmm(LatentVariableModel):
    """Model parameters for classic Hidden Markov Model.

    Parameters
    ----------
    K : int
        number of states
    mu : np.ndarray
        state means, shape [K, ]
    tau : np.ndarray
        state precision (1 / variance), shape [K, ]
    pi : np.ndarray
        initial state probabilities, shape [K, ]
    A : np.ndarray
        state transition probability matrix, shape [K, K]
    """

    pi: np.ndarray
    A: np.ndarray

    @classmethod
    def guess(cls, data, n_states, gmm=None):
        """Calculate an initial guess for the model from a GMM.

        Parameters
        ----------
        data : np.ndarray
            Training data
        n_states : int
            Number of hidden states
        gmm : GaussianMixtureModel
            Pre-trained GMM
        """
        data = Slice(Continuous(data, 0, 1))
        if gmm is None:
            gmm = GaussianMixtureModel(data, n_states)
        statepath = gmm.state_path(data).data
        A = normalize_rows(
            np.vstack(
                [
                    [
                        np.logical_and(statepath[:-1] == j, statepath[1:] == k).sum()
                        for k in range(n_states)
                    ]
                    for j in range(n_states)
                ]
            )
        )
        pi = np.ones(n_states) / n_states

        return cls(n_states, gmm.means, 1 / gmm.variances, pi, A)

    def state_log_likelihood(self, x):
        """Calculate the state likelihood of the observation data `x`. Work in log space to avoid
        underflow.
        """
        x2 = (row(x) - col(self.mu)) ** 2
        tau = col(self.tau)
        return -0.5 * (np.log(2 * np.pi) - np.log(tau) + x2 * tau)

    def update(self, data, gamma, xi):
        """Update model parameters from the temporary variables of the Baum Welch algorithm.

        Parameters
        ----------
        data : np.ndarray
            observed data, shape [T, ]
        gamma : np.ndarray
            probability of being in state i at time t, shape [T, K]
        xi : np.ndarray
            probability of being in state i at time t and state j at time t + 1, shape [T, K, K]
        """
        pi = gamma[0]  # Eq 40a
        A = xi.sum(axis=0) / col(gamma[:-1].sum(axis=0))  # Eq 40b
        x_bar = np.sum(gamma * col(data), axis=0) / gamma.sum(axis=0)  # Eq 53
        variance = np.sum(gamma * (col(data) - row(x_bar)) ** 2, axis=0) / gamma.sum(axis=0)  # Eq54

        return ClassicHmm(self.K, x_bar, 1 / variance, pi, A)


def baum_welch(data, model, tol, max_iter):
    """Specialized form of the expectation maximization algorithm to train HMMs.

    Described in detail in Section I in Rabiner (specific discussion on the full Baul-Welch procedure
    given on page 265).

    Parameters
    ----------
    data : np.ndarray
        Observation data, shape [T, ]
    model : ClassicHmm
        Model parameters with K states
    tol : float
        The tolerance for training convergence
    max_iter : int
        The maximum number of iterations to perform

    Returns
    -------
    model : ClassicHmm
        Optimized model instance
    """
    # initial E-step
    converged = False
    gamma, xi, previous_log_likelihood = calculate_temporary_variables(
        model, *forward_backward(data, model)
    )

    with warnings.catch_warnings():
        warnings.simplefilter("always")
        for _itr in range(1, max_iter + 1):
            # M-step
            model = model.update(data, gamma, xi)

            # E-step
            gamma, xi, current_log_likelihood = calculate_temporary_variables(
                model, *forward_backward(data, model)
            )

            # check for convergence
            delta = current_log_likelihood - previous_log_likelihood
            if np.abs(delta) < tol:
                converged = True
                break
            previous_log_likelihood = current_log_likelihood

        if not converged:
            warnings.warn(
                f"Model has not converged after {_itr} iterations. Last log likelihood step "
                f"was {delta:0.4e}."
            )

    # free parameters; pi and each row of A constrained to sum to 1
    # (K - 1) + (K**2 - K) + K + K
    #   pi          A        mu  tau
    k = model.K**2 + 2 * model.K - 1
    bic = k * np.log(len(data)) - 2 * current_log_likelihood
    aic = 2 * k - 2 * current_log_likelihood

    fit_info = PopulationFitInfo(converged, _itr, bic, aic, current_log_likelihood)

    return model, fit_info


def forward_backward(data, model):
    """Recursive algorithm to calculate state path probabilities from observations.

    Use scaled version to avoid underflow (Section V-A in Rabiner; the unscaled versions fournd in
    Equations 18-21 and 24-25 in Section I).

    Parameters
    ----------
    data : np.ndarray
        Observation data, shape [T, ]
    model : ClassicHmm
        Model parameters with K states

    Returns
    -------
    alpha : np.ndarray
        probability of the sequence y[0]...y[t] and being in state i at time t; shape [T, K]
    beta : np.ndarray
        probability of the ending partial sequence y[t+1]...y[T] given starting state i at time t;
        shape [T, K]
    c : np.ndarray
        scaling factors; shape [T, ]
    B : np.ndarray
        state likelihoods; shape[T, K]
    """
    # for here, simpler if states as columns, observations as rows
    B = np.exp(model.state_log_likelihood(data)).T
    T = data.size
    K = model.K

    alpha = np.zeros((T, K))
    beta = np.ones((T, K))
    c = np.zeros(T)

    # forward loop, initialize
    alpha[0] = model.pi * B[0]
    c[0] = np.sum(alpha[0])
    alpha[0] = alpha[0] / c[0]

    # forward loop, induction
    for t in range(1, T):
        tmp = col(alpha[t - 1]) * model.A
        alpha[t] = np.sum(tmp, axis=0) * B[t]
        c[t] = np.sum(alpha[t])
        alpha[t] = alpha[t] / c[t]

    # backward loop
    for t in range(1, T):
        beta[-(t + 1)] = np.sum(model.A * row(B[-t]) * row(beta[-t]), axis=1) / c[-t]

    return alpha, beta, c, B


def calculate_temporary_variables(model, alpha, beta, c, B):
    """Calculate temporary variables for model update and log likelihood of current model.

    Parameters
    ----------
    model : ClassicHmm
        Current model parameters with K states
    alpha : np.ndarray
        probability of the sequence y[0]...y[t] and being in state i at time t; shape [T, K]
    beta : np.ndarray
        probability of the ending partial sequence y[t+1]...y[T] given starting state i at time t;
        shape [T, K]
    c : np.ndarray
        scaling factors (to avoid underflow); shape [T, ]
    B : np.ndarray
        state likelihoods; shape[T, K]

    Returns
    -------
    gamma : np.ndarray
        Probability of being in state i at time t; shape [T, K]
    xi : np.ndarray
        Probability of being in state i at time t and transitioning to state j at time t + 1;
        shape [T, K, K]
    log_likelihood : float
    """
    gamma = alpha * beta  # Eq 27

    # Eq 37
    xi = (
        alpha[:-1][:, :, np.newaxis]
        * model.A
        * B[1:][:, np.newaxis, :]
        * beta[1:][:, np.newaxis, :]
    ) / c[1:][:, np.newaxis, np.newaxis]

    # Eq 103
    log_likelihood = np.sum(np.log(c))

    return gamma, xi, log_likelihood


def viterbi(data, model):
    """The Viterbi algorithm to assign most probable hidden state path based on observation
    sequence.

    Details in Equations 29-35.

    Parameters
    ----------
    data : np.ndarray
        Observation data, shape [T, ]
    model : ClassicHmm
        Model parameters with K states

    Returns
    -------
    statepath : np.ndarray
        Hidden state path, shape [T, ]
    """
    T = data.size
    B = model.state_log_likelihood(data).T
    psi = np.zeros(B.shape, dtype=int)
    statepath = np.zeros(T, dtype=int)

    # - for models trained on a single trace you can end up with zeros
    #   if states are well separated or observations have very low noise
    # - you can also end up with zeros in the transition matrix for states
    #   which are not connected to each other
    with np.errstate(divide="ignore"):
        log_pi = np.log(model.pi)
        log_A = np.log(model.A)

    # initialization
    delta = col(log_pi + B[0])
    # recursion
    for t in range(1, T):
        R = delta + log_A
        psi[t] = np.argmax(R, axis=0)
        delta = col(np.max(R, axis=0) + B[t])

    # termination
    statepath[-1] = np.argmax(delta)
    # path backtracking
    for t in range(1, T):
        statepath[-(t + 1)] = psi[-t, statepath[-t]]

    return statepath
