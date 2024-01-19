import numpy as np
import scipy
from deprecated.sphinx import deprecated

from ..channel import Slice
from .detail.mixin import TimeSeriesMixin
from .detail.fit_info import GmmFitInfo


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


class GaussianMixtureModel(TimeSeriesMixin):
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
    data : numpy.ndarray | Slice
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

    def __init__(self, data, n_states, init_method="kmeans", n_init=1, tol=1e-3, max_iter=100):
        from sklearn.mixture import GaussianMixture

        if isinstance(data, Slice):
            data = data.data

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

        self._fit_info = GmmFitInfo(
            self._model.converged_,
            self._model.n_iter_,
            self._model.bic(data),
            self._model.aic(data),
            self._model.lower_bound_,
        )

    @classmethod
    @deprecated(
        reason=(
            "This method has been deprecated and will be removed in a future version. You can now "
            "use `Slice` instances to construct this class directly."
        ),
        action="always",
        version="1.4.0",
    )
    def from_channel(cls, slc, n_states, init_method="kmeans", n_init=1, tol=1e-3, max_iter=100):
        """Initialize a model from channel data."""
        return cls(
            slc, n_states, init_method=init_method, n_init=n_init, tol=tol, max_iter=max_iter
        )

    @property
    @deprecated(
        reason=(
            "This property has been replaced with `GaussianMixtureModel.fit_info` and will be removed "
            "in a future release."
        ),
        action="always",
        version="1.4.0",
    )
    def exit_flag(self) -> dict:
        """Model optimization information."""
        return {
            "converged": self.fit_info.converged,
            "n_iter": self.fit_info.n_iter,
            "lower_bound": self.fit_info.lower_bound,
        }

    @property
    def fit_info(self) -> GmmFitInfo:
        """Information about the model training exit conditions."""
        return self._fit_info

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

    @deprecated(
        reason=(
            "This method has been replaced with `GaussianMixtureModel.state_path()` and will be "
            "removed in a future release."
        ),
        action="always",
        version="1.4.0",
    )
    def label(self, trace):
        """Label channel data as states.

        Parameters
        ----------
        trace : Slice
            Channel data to label.
        """
        return self.state_path(trace).data

    def _calculate_state_path(self, trace):
        labels = self._model.predict(trace.data.reshape((-1, 1)))  # wrapped model labels
        output_states = np.argsort(self._map)  # output model state labels in wrapped model order
        return output_states[labels]  # output model labels

    @property
    @deprecated(
        reason=(
            "This property has been deprecated and will be removed in a future version. Use "
            "`GaussianMixtureModel.fit_info.bic` instead."
        ),
        action="always",
        version="1.4.0",
    )
    def bic(self) -> float:
        r"""Calculates the Bayesian Information Criterion:"""
        return self.fit_info.bic

    @property
    @deprecated(
        reason=(
            "This method has been deprecated and will be removed in a future version. Use "
            "`GaussianMixtureModel.fit_info.aic` instead."
        ),
        action="always",
        version="1.4.0",
    )
    def aic(self) -> float:
        r"""Calculates the Akaike Information Criterion:"""
        return self.fit_info.aic

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

    def _calculate_gaussian_components(self, x, trace):
        return self.pdf(x)
