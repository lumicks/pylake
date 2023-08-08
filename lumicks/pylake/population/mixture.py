import numpy as np
import scipy

from .dwelltime import _dwellcounts_from_statepath


def as_sorted(fcn):
    """Decorator to return results sorted according to mapping array.

    To be used as a method decorator in a class that supplies an index
    mapping array via the `._map` attribute.
    """

    def wrapper(self, *args, **kwargs) -> np.ndarray:
        result = fcn(self, *args, **kwargs)
        return result[self._map]

    wrapper.__doc__ = fcn.__doc__

    return wrapper


class GaussianMixtureModel:
    """A wrapper around :class:`sklearn.mixture.GaussianMixture`.

    This model accepts a 1D array as training data. *The state parameters are sorted according
    to state mean* in order to facilitate comparison of models with different number of states
    or trained on different datasets. As the current implementation is designed to specifically
    handle 1D data, model parameters are also returned as 1D arrays (:func:`numpy.squeeze()` is applied to
    the results) so that users do not have to be concerned with the shape of the output results.

    .. warning::

        This is early access alpha functionality. While usable, this has not yet been tested in a large number of
        different scenarios. The API can still be subject to change without any prior deprecation notice! If you
        use this functionality keep a close eye on the changelog for any changes that may affect your analysis.

    Parameters
    ----------
    data : numpy.ndarray
        Data array used for model training.
    n_states : int
        The number of Gaussian components in the model.
    init_method : {'kmeans', 'random'}
        - "kmeans" : parameters are initialized via k-means algorithm
        - "random" : parameters are initialized randomly
    n_init : int
        The number of initializations to perform.
    tol : float
        The tolerance for training convergence.
    max_iter : int
        The maximum number of iterations to perform.
    """

    def __init__(self, data, n_states, init_method, n_init, tol, max_iter):
        from sklearn.mixture import GaussianMixture

        self.n_states = n_states
        self._model = GaussianMixture(
            n_components=n_states,
            init_params=init_method,
            n_init=n_init,
            tol=tol,
            max_iter=max_iter,
        )
        data = np.reshape(data, (-1, 1))
        self._model.fit(data)
        self._bic = self._model.bic(data)
        self._aic = self._model.aic(data)

    @classmethod
    def from_channel(cls, slc, n_states, init_method="kmeans", n_init=1, tol=1e-3, max_iter=100):
        """Initialize a model from channel data.

        Parameters
        ----------
        slc : Slice
            Channel data used for model training.
        n_states : int
            The number of Gaussian components in the model.
        init_method : {'kmeans', 'random'}
            - "kmeans" : parameters are initialized via k-means algorithm
            - "random" : parameters are initialized randomly
        n_init : int
            The number of initializations to perform.
        tol : float
            The tolerance for training convergence.
        max_iter : int
            The maximum number of iterations to perform.
        """
        return cls(
            slc.data, n_states, init_method=init_method, n_init=n_init, tol=tol, max_iter=max_iter
        )

    @property
    def exit_flag(self) -> dict:
        """Model optimization information."""
        return {
            "converged": self._model.converged_,
            "n_iter": self._model.n_iter_,
            "lower_bound": self._model.lower_bound_,
        }

    @property
    def _map(self) -> np.ndarray:
        """Indices of sorted means."""
        return np.argsort(self._model.means_.squeeze())

    @property
    @as_sorted
    def weights(self):
        """Model state weights."""
        return self._model.weights_

    @property
    @as_sorted
    def means(self):
        """Model state means."""
        return self._model.means_.squeeze()

    @property
    @as_sorted
    def variances(self):
        """Model state variances."""
        return self._model.covariances_.squeeze()

    @property
    def std(self) -> np.ndarray:
        """Model state standard deviations."""
        return np.sqrt(self.variances)

    def label(self, trace):
        """Label channel data as states.

        Parameters
        ----------
        trace : Slice
            Channel data to label.
        """
        data = trace.data.reshape((-1, 1))
        labels = self._model.predict(data)  # wrapped model labels
        output_states = np.argsort(self._map)  # output model state labels in wrapped model order
        return output_states[labels]  # output model labels

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
        statepath = self.label(trace)
        dt_seconds = 1.0 / trace.sample_rate

        dwell_counts, _ = _dwellcounts_from_statepath(
            statepath, exclude_ambiguous_dwells=exclude_ambiguous_dwells
        )
        dwell_times = {key: counts * dt_seconds for key, counts in dwell_counts.items()}
        return dwell_times

    @property
    def bic(self) -> float:
        r"""Calculates the Bayesian Information Criterion:

        .. math::
            BIC = k \ln{(n)} - 2 \ln{(L)}

        Where k refers to the number of parameters, n to the number of observations (or data points)
        and L to the maximized value of the likelihood function
        """
        return self._bic

    @property
    def aic(self) -> float:
        r"""Calculates the Akaike Information Criterion:

        .. math::
            AIC = 2 k - 2 \ln{(L)}

        Where k refers to the number of parameters, n to the number of observations (or data
        points) and L to the maximized value of the likelihood function.
        """
        return self._aic

    def pdf(self, x):
        """Calculate the Probability Distribution Function (PDF) given the independent data array `x`.

        Parameters
        ----------
        x : numpy.ndarray
            Array of independent variable values at which to calculate the PDF.

        Returns
        -------
        numpy.ndarray:
            PDF array split into components for each state with shape (n_states, x.size).
            The full normalized PDF can be calculated by summing across rows.
        """
        components = np.vstack(
            [scipy.stats.norm(m, s).pdf(x) for m, s in zip(self.means, self.std)]
        )
        return self.weights.reshape((-1, 1)) * components

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

        g_components = self.pdf(x)

        plt.hist(trace.data, bins=bins, density=True, **hist_kwargs)
        # reset color cycle
        plt.gca().set_prop_cycle(None)
        plt.plot(x, g_components.T, **(plot_kwargs or {}))
        plt.ylabel("density")
        plt.xlabel(trace.labels.get("y", "signal"))

    def plot(self, trace, trace_kwargs=None, label_kwargs=None):
        """Plot a time trace with each data point labeled with the state assignment.

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

        labels = self.label(trace)
        trace.plot(**trace_kwargs)
        for k in range(self.n_states):
            ix = np.argwhere(labels == k)
            plt.plot(trace.seconds[ix], trace.data[ix], **label_kwargs)
