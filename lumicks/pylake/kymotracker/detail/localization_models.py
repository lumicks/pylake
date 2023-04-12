import numpy as np
from dataclasses import dataclass, replace

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

    def with_position(self, position):
        """Return a copy with a new set of positions"""
        return replace(self, position=position)

    def _flip(self, size):
        """Flip the localization

        Parameters
        ----------
        size : float
            Size of the kymograph in physical units.
        """
        return self.with_position(size - self.position)


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
