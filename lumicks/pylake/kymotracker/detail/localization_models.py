import numpy as np
from dataclasses import dataclass

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

    def _split(self, indices_or_sections):
        return [LocalizationModel(p) for p in np.array_split(self.position, indices_or_sections)]


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

    def _split(self, indices_or_sections):
        return [
            GaussianLocalizationModel(position, total_photons, sigma, background, _overlap_fit)
            for position, total_photons, sigma, background, _overlap_fit in zip(
                np.array_split(self.position, indices_or_sections),
                np.array_split(self.total_photons, indices_or_sections),
                np.array_split(self.sigma, indices_or_sections),
                np.array_split(self.background, indices_or_sections),
                np.array_split(self._overlap_fit, indices_or_sections),
            )
        ]
