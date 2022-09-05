import numpy as np
from scipy import integrate
from dataclasses import dataclass
from typing import ClassVar

__all__ = ["LocalizationModel", "GaussianLocalizationModel"]


@dataclass(frozen=True)
class LocalizationModel:
    """Helper (base) class to hold refinement optimization parameters.

    Parameters
    ----------
    position : np.array
        Spatial coordinates in physical units.
    """

    position: np.ndarray

    def _calculate_position_variance(self, pixelsize):
        return np.full(self.position.size, np.nan)


@dataclass(frozen=True)
class GaussianLocalizationModel(LocalizationModel):
    """Helper (base) class to hold refinement optimization parameters.

    Parameters
    ----------
    position : np.array
        Spatial coordinates in physical units.
    total_photons : np.array
        Array of integrated photon counts for each time point.
    sigma : np.array
        Array of gaussian sigma parameters; sqrt(variance)
    background : np.array
        Average background photon count per pixel for each time point.
    _overlap_fit: np.ndarray
        Boolean array of whether time point was fit with overlapping gaussian model.
    """

    total_photons: np.ndarray
    sigma: np.ndarray
    background: np.ndarray
    _overlap_fit: np.ndarray

    def _calculate_position_variance(self, pixelsize):
        # Mortensen et al. SI Eq. 54
        var = []
        for photons, sigma, background, overlap in zip(
            self.total_photons, self.sigma, self.background, self._overlap_fit
        ):
            if overlap:
                var.append(np.nan)
            else:
                tau = np.sqrt(2 * np.pi * sigma**2) * background / (photons * pixelsize)
                fn = lambda t: np.sqrt(-np.log(t)) / (1 + tau / t)
                integral, _ = integrate.quad(fn, 0, 1)
                var.append(sigma**2 / photons * np.sqrt(np.pi) / 2 * integral**-1)
        return var
