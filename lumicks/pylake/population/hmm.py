import numpy as np

from .mixture import GaussianMixtureModel
from ..channel import Slice
from .detail.hmm import ClassicHmm, viterbi, baum_welch
from .detail.mixin import TimeSeriesMixin
from .detail.validators import col


class HiddenMarkovModel(TimeSeriesMixin):
    """A Hidden Markov Model describing hidden state occupancy and state transitions of observed
    time series data (force, fluorescence, etc.)

    A detailed description of the model properties and training algorithms can be found in [1]_.

    .. warning::

        This is early access alpha functionality. While usable, this has not yet been tested in a
        large number of different scenarios. The API can still be subject to change without any
        prior deprecation notice! If you use this functionality keep a close eye on the changelog
        for any changes that may affect your analysis.

    Parameters
    ----------
    data : numpy.ndarray | Slice
        Data array used for model training.
    n_states : int
        The number of hidden states in the model.
    tol : float
        The tolerance for training convergence.
    max_iter : int
        The maximum number of iterations to perform.
    initial_guess : HiddenMarkovModel | GaussianMixtureModel | None
        Initial guess for the observation model parameters.

    References
    ----------
    .. [1] Rabiner, L. A Tutorial on Hidden Markov Models and Selected Applications in Speech
           Recognition. Proceedings of IEEE. 77, 257-286 (1989).
    """

    def __init__(self, data, n_states, *, tol=1e-3, max_iter=250, initial_guess=None):
        if isinstance(data, Slice):
            data = data.data

        if (
            isinstance(initial_guess, (GaussianMixtureModel, HiddenMarkovModel))
            and initial_guess.n_states != n_states
        ):
            raise ValueError(
                "Initial guess must have the same number of states as requested "
                f"for the current model; expected {n_states} got {initial_guess.n_states}."
            )
        self.n_states = n_states

        if isinstance(initial_guess, GaussianMixtureModel) or initial_guess is None:
            initial_guess = ClassicHmm.guess(data, n_states, gmm=initial_guess)
        elif isinstance(initial_guess, HiddenMarkovModel):
            initial_guess = initial_guess._model
        else:
            raise TypeError(
                f"if provided, `initial_guess` must be either GaussianMixtureModel or "
                f"HiddenMarkovModel, got `{type(initial_guess).__name__}`."
            )

        self._model, self._fit_info = baum_welch(data, initial_guess, tol=tol, max_iter=max_iter)

    @property
    def initial_state_probability(self) -> np.ndarray:
        """Model initial state probability."""
        return self._model.pi

    @property
    def transition_matrix(self) -> np.ndarray:
        """Model state transition matrix.

        Element `i, j` gives the probability of transitioning from state `i` at time point `t`
        to state `j` at time point `t+1`.
        """
        return self._model.A

    def _calculate_state_path(self, trace):
        return viterbi(trace.data, self._model)

    def _calculate_gaussian_components(self, x, trace):
        state_path = self.state_path(trace)
        fractions = col(
            [np.sum(state_path.data == j) / len(state_path) for j in np.unique(state_path.data)]
        )
        pdf = np.exp(self._model.state_log_likelihood(x))
        return pdf * fractions
