import numpy as np
from dataclasses import dataclass, replace, fields

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

    def __add__(self, other):
        if other.__class__ is not self.__class__:
            raise TypeError(
                f"Incompatible localization models {self.__class__.__name__} and "
                f"{other.__class__.__name__}."
            )

        init_kwargs = {
            f.name: np.hstack((getattr(self, f.name), getattr(other, f.name))) for f in fields(self)
        }
        return self.__class__(**init_kwargs)

    def __getitem__(self, item):
        return self.__class__(**{f.name: getattr(self, f.name)[item] for f in fields(self)})


@dataclass(frozen=True)
class CentroidLocalizationModel(LocalizationModel):
    """Helper class to hold refinement optimization parameters.

    Parameters
    -----------
    position : numpy.ndarray
        Spatial coordinates in physical units.
    total_photons : numpy.ndarray
        Array of integrated photon counts for each time point.
    """

    total_photons: np.ndarray


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
