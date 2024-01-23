from dataclasses import dataclass

import numpy as np

from .fit_info import PopulationFitInfo
from ...channel import Slice
from ..dwelltime import _dwellcounts_from_statepath


class TimeSeriesMixin:
    @property
    def fit_info(self) -> PopulationFitInfo:
        """Information about the model training exit conditions."""
        return self._fit_info

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

    def extract_dwell_times(self, trace, *, exclude_ambiguous_dwells=True):
        """Calculate lists of dwelltimes for each state in a time-ordered state path array.

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
        state_path = self.state_path(trace)
        dt_seconds = 1.0 / state_path.sample_rate

        dwell_counts, _ = _dwellcounts_from_statepath(
            state_path.data, exclude_ambiguous_dwells=exclude_ambiguous_dwells
        )
        dwell_times = {key: counts * dt_seconds for key, counts in dwell_counts.items()}
        return dwell_times

    def _calculate_state_path(self):
        raise NotImplementedError(
            f"`{self.__module__}.{self.__class__.__name__}` does not implement "
            "`_calculate_state_path()`."
        )

    def state_path(self, trace):
        """Calculate the state path for a given data trace.

        Parameters
        ----------
        trace : Slice
            Channel data to determine path.

        Returns
        -------
        state_path : Slice
            Estimated state path
        """
        state_path = self._calculate_state_path(trace)
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

    def _calculate_gaussian_components(self, x, trace):
        raise NotImplementedError(
            f"`{self.__module__}.{self.__class__.__name__}` does not implement "
            "`_calculate_gaussian_components()`."
        )

    def hist(self, trace, n_bins=100, plot_kwargs=None, hist_kwargs=None):
        """Plot a histogram of the trace data overlaid with the model state path.

        Parameters
        ----------
        trace : Slice
            Data object to histogram.
        n_bins : int
            Number of histogram bins.
        plot_kwargs : Optional[dict]
            Plotting keyword arguments passed to the state path line plot.
        hist_kwargs : Optional[dict]
            Plotting keyword arguments passed to the histogram plot.
        """
        import matplotlib.pyplot as plt

        hist_kwargs = {"facecolor": "#c5c5c5", **(hist_kwargs or {})}

        lims = (np.min(trace.data), np.max(trace.data))
        bins = np.linspace(*lims, num=n_bins)
        x = np.linspace(*lims, num=(n_bins * 5))

        g_components = self._calculate_gaussian_components(x, trace)

        plt.hist(trace.data, bins=bins, density=True, **hist_kwargs)
        # reset color cycle
        plt.gca().set_prop_cycle(None)
        plt.plot(x, g_components.T, **(plot_kwargs or {}))
        plt.ylabel("density")
        plt.xlabel(trace.labels.get("y", "signal"))

    def plot(self, trace, *, trace_kwargs=None, label_kwargs=None):
        """Plot a histogram of the trace data with data points classified in states.

        Parameters
        ----------
        trace : Slice
            Data object to histogram.
        trace_kwargs : Optional[dict]
            Plotting keyword arguments passed to the data line plot.
        label_kwargs : Optional[dict]
            Plotting keyword arguments passed to the state labels plot.
        """
        import matplotlib.pyplot as plt

        trace_kwargs = {"c": "#c5c5c5", **(trace_kwargs or {})}
        label_kwargs = {"marker": "o", "ls": "none", "ms": 3, **(label_kwargs or {})}

        trace.plot(**trace_kwargs)

        state_path = self.state_path(trace)
        for k in range(self.n_states):
            ix = np.argwhere(state_path.data == k)
            plt.plot(trace.seconds[ix], trace.data[ix], **label_kwargs)

    def plot_path(self, trace, *, trace_kwargs=None, path_kwargs=None):
        """Plot a histogram of the trace data overlaid with the model path.

        Parameters
        ----------
        trace : Slice
            Data object to histogram.
        trace_kwargs : Optional[dict]
            Plotting keyword arguments passed to the data line plot.
        path_kwargs : Optional[dict]
            Plotting keyword arguments passed to the path line plot.
        """
        trace_kwargs = {"c": "#c5c5c5", **(trace_kwargs or {})}
        trace.plot(**trace_kwargs)

        path_kwargs = {"c": "tab:blue", "lw": 2, **(path_kwargs or {})}
        emission_path = self.emission_path(trace)
        emission_path.plot(**path_kwargs)


@dataclass(frozen=True)
class LatentVariableModel:
    K: int
    mu: np.ndarray
    tau: np.ndarray
