import numpy as np
from sklearn.mixture import GaussianMixture
from scipy import stats


def as_sorted(fcn):
    """Decorator to return results sorted according to mapping array."""

    def wrapper(self, *args, **kwargs):
        result = fcn(self, *args, **kwargs)
        return result[self._map]

    wrapper.__doc__ = fcn.__doc__

    return wrapper


class GaussianMixtureModel:
    """A wrapper around scikit-learn's GMM.

    This model accepts a 1D array as training data. The state parameters are sorted according
    to state mean in order to facilitate comparison of models with different number of states
    or trained on different datasets. As the current implementation is designed to specifically
    handle 1D data, model parameters are also returned as 1D arrays (np.squeeze() is applied to the results)
    so that users do not have to be concerned with the shape of the output results.

    Parameters
    ----------
    data : array-like
        Data object used for model training.
    n_states : int
        The number of Gaussian components in the model.
    init_method : 'kmeans' or 'random'
        The method used to initialize parameters.
    n_init : int
        The number of initializations to perform.
    tol : float
        The tolerance for training convergence.
    max_iter : int
        The maximum number of iterations to perform.
    """

    def __init__(self, data, n_states, init_method, n_init, tol, max_iter):
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
        init_method : 'kmeans' or 'random'
            The method used to initialize parameters.
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
    def exit_flag(self):
        """Model optimization information."""
        return {
            "converged": self._model.converged_,
            "n_iter": self._model.n_iter_,
            "lower_bound": self._model.lower_bound_,
        }

    @property
    def _map(self):
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
    def std(self):
        """Model state standard deviations."""
        return np.sqrt(self.variances)

    def label(self, trace):
        """Label channel trace data as states."""
        data = trace.data.reshape((-1, 1))
        labels = self._model.predict(data)  # wrapped model labels
        output_states = np.argsort(self._map)  # output model state labels in wrapped model order
        return output_states[labels]  # output model labels

    @property
    def bic(self):
        """Bayesian Information Criterion."""
        return self._bic

    @property
    def aic(self):
        """Akaike Information Criterion."""
        return self._aic

    def pdf(self, x):
        """Probability Distribution Function (states as rows)."""
        components = np.vstack([stats.norm(m, s).pdf(x) for m, s in zip(self.means, self.std)])
        return self.weights.reshape((-1, 1)) * components

    def hist(self, trace, n_bins=100, plot_kwargs={}, hist_kwargs={}):
        """Plot a histogram of the data overlaid with the model PDF.

        Parameters
        ----------
        trace : Slice-like
            Data object to histogram.
        n_bins : int
            Number of histogram bins.
        plot_kwargs : dict
            Plotting keyword arguments passed to the PDF line plot.
        hist_kwargs : dict
            Plotting keyword arguments passed to the histogram plot.
        """
        import matplotlib.pyplot as plt

        hist_kwargs = {"facecolor": "#c5c5c5", **hist_kwargs}

        lims = (np.min(trace.data), np.max(trace.data))
        bins = np.linspace(*lims, num=n_bins)
        x = np.linspace(*lims, num=(n_bins * 5))

        g_components = self.pdf(x)

        plt.hist(trace.data, bins=bins, density=True, **hist_kwargs)
        # reset color cycle
        plt.gca().set_prop_cycle(None)
        plt.plot(x, g_components.T, **plot_kwargs)
        plt.ylabel("density")
        plt.xlabel(trace.labels.get("y", "signal"))

    def plot(self, trace, trace_kwargs={}, label_kwargs={}):
        """Plot a time trace with each data point labeled with the state assignment.

        Parameters
        ----------
        trace : Slice-like
            Data object to histogram.
        trace_kwargs : dict
            Plotting keyword arguments passed to the data line plot.
        label_kwargs : dict
            Plotting keyword arguments passed to the state labels plot.
        """
        import matplotlib.pyplot as plt

        trace_kwargs = {"c": "#c5c5c5", **trace_kwargs}
        label_kwargs = {"marker": "o", "ls": "none", "ms": 3, **label_kwargs}

        labels = self.label(trace)
        trace.plot(**trace_kwargs)
        for k in range(self.n_states):
            ix = np.argwhere(labels == k)
            plt.plot(trace.seconds[ix], trace.data[ix], **label_kwargs)
