from dataclasses import dataclass

import numpy as np
import scipy
from deprecated.sphinx import deprecated

from ..channel import Slice
from .detail.mixin import TimeSeriesMixin, LatentVariableModel
from .detail.fit_info import PopulationFitInfo


@dataclass(frozen=True)
class ClassicGmm(LatentVariableModel):
    """Model parameters for classic Gaussian Mixture Model.

    Parameters
    ----------
    K : int
        number of states
    mu : np.ndarray
        state means, shape [K, ]
    tau : np.ndarray
        state precision (1 / variance), shape [K, ]
    weights: np.ndarray
        state fractional weights
    """

    weights: np.ndarray


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
        model = GaussianMixture(
            n_components=n_states,
            init_params=init_method,
            n_init=n_init,
            tol=tol,
            max_iter=max_iter,
        )
        data = np.reshape(data, (-1, 1))
        model.fit(data)

        # todo: remove when exit_flag is removed
        self._deprecated_lower_bound = model.lower_bound_

        idx = np.argsort(model.means_.squeeze())
        self._model = ClassicGmm(
            K=n_states,
            mu=model.means_.squeeze()[idx],
            tau=1 / model.covariances_.squeeze()[idx],
            weights=model.weights_[idx],
        )

        self._fit_info = PopulationFitInfo(
            converged=model.converged_,
            n_iter=model.n_iter_,
            bic=model.bic(data),
            aic=model.aic(data),
            log_likelihood=np.sum(model.score_samples(data)),
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
            "lower_bound": self._deprecated_lower_bound,
        }

    @property
    def weights(self):
        """Model state weights."""
        return self._model.weights

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
        return np.argmax(self.pdf(trace.data), axis=0)

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
