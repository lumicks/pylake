import numpy as np

from .mixture import GaussianMixtureModel
from ..channel import Slice
from .dwelltime import _dwellcounts_from_statepath
from .detail.hmm import ClassicHmm, viterbi, baum_welch
from .detail.fit_info import HmmFitInfo
from .detail.validators import col


class HiddenMarkovModel:
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
    def fit_info(self) -> HmmFitInfo:
        """Information about the model training exit conditions."""
        return self._fit_info

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

    @property
    def means(self) -> np.ndarray:
        """Model state means."""
        return self._model.mu

    @property
    def variances(self) -> np.ndarray:
        """Model state variances."""
        return 1 / self._model.tau

    @property
    def std(self) -> np.ndarray:
        """Model state standard deviations."""
        return np.sqrt(self.variances)

    def state_path(self, trace):
        """Calculate the state emission path for a given data trace.

        Parameters
        ----------
        trace : Slice
            Channel data to determine path.

        Returns
        -------
        state_path : Slice
            Estimated state path
        """
        state_path = viterbi(trace.data, self._model)
        src = trace._src._with_data(state_path)
        labels = trace.labels.copy()
        labels["y"] = "state"
        return Slice(src, labels=labels)

    def emission_path(self, trace):
        """Calculate the emission path for a given data trace.

        Parameters
        ----------
        trace : Slice
            Channel data to determine path.

        Returns
        -------
        emission_path : Slice
            Estimated emission path
        """
        emission_path = self.means[self.state_path(trace).data]
        return Slice(trace._src._with_data(emission_path), labels=trace.labels.copy())

    def extract_dwell_times(self, trace, *, exclude_ambiguous_dwells=True):
        """Calculate lists of dwelltimes for each state in a time-ordered statepath array.

        Parameters
        ----------
        trace : Slice
            Channel data to be analyzed.
        exclude_ambiguous_dwells : bool
            Determines whether to exclude dwelltimes which are not exactly determined. If `True`, the first
            and last dwells are not used in the analysis, since the exact start/stop times of these events are
            not definitively known.

        Returns
        -------
        dict:
            Dictionary of all dwell times (in seconds) for each state. Keys are state labels.
        """
        state_path = self.state_path(trace).data
        dt_seconds = 1.0 / trace.sample_rate

        dwell_counts, _ = _dwellcounts_from_statepath(
            state_path, exclude_ambiguous_dwells=exclude_ambiguous_dwells
        )
        dwell_times = {key: counts * dt_seconds for key, counts in dwell_counts.items()}
        return dwell_times

    def hist(self, trace, n_bins=100, plot_kwargs=None, hist_kwargs=None):
        """Plot a histogram of the data overlaid with the model PDF.

        Parameters
        ----------
        trace : Slice
            Data object to histogram.
        n_bins : int
            Number of histogram bins.
        plot_kwargs : Optional[dict]
            Plotting keyword arguments passed to the PDF line plot.
        hist_kwargs : Optional[dict]
            Plotting keyword arguments passed to the histogram plot.
        """
        import matplotlib.pyplot as plt

        hist_kwargs = {"facecolor": "#c5c5c5", **(hist_kwargs or {})}

        lims = (np.min(trace.data), np.max(trace.data))
        bins = np.linspace(*lims, num=n_bins)
        x = np.linspace(*lims, num=(n_bins * 5))

        state_path = self.state_path(trace)
        fractions = col(
            [np.sum(state_path.data == j) / len(state_path) for j in np.unique(state_path.data)]
        )
        pdf = np.exp(self._model.state_log_likelihood(x))
        g_components = pdf * fractions

        plt.hist(trace.data, bins=bins, density=True, **hist_kwargs)
        # reset color cycle
        plt.gca().set_prop_cycle(None)
        plt.plot(x, g_components.T, **(plot_kwargs or {}))
        plt.ylabel("density")
        plt.xlabel(trace.labels.get("y", "signal"))

    def plot(self, trace, *, trace_kwargs=None, path_kwargs=None):
        """Plot a time trace with each data point labeled with the state assignment.

        Parameters
        ----------
        trace : Slice
            Data object to histogram.
        trace_kwargs : Optional[dict]
            Plotting keyword arguments passed to the data line plot.
        path_kwargs : Optional[dict]
            Plotting keyword arguments passed to the state path line plot.
        """

        trace_kwargs = {"c": "#c5c5c5", **(trace_kwargs or {})}
        path_kwargs = {"c": "tab:blue", "lw": 2, **(path_kwargs or {})}

        trace.plot(**trace_kwargs)
        emission_path = self.emission_path(trace)
        emission_path.plot(**path_kwargs)
