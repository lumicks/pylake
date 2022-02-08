import numpy as np
from scipy import integrate
from dataclasses import dataclass
from typing import ClassVar


@dataclass(frozen=True)
class LocalizationModel:
    """Data structure to handle localization parameters.

    All position/spatial values are in physical units (microns, kbp, etc).
    """

    _name: ClassVar[str] = "default"
    pixel_size: float
    position: np.ndarray

    @property
    def pixel_coordinate(self):
        return self.position / self.pixel_size

    @property
    def loc_variance(self):
        return np.full(self.position.size, np.nan)


@dataclass(frozen=True)
class GaussianLocalizationModel(LocalizationModel):
    """
    Mortensen, K. I., Churchman, L. S., Spudich, J. A., & Flyvbjerg, H. (2010).
    Optimized localization analysis for single-molecule tracking and super-resolution microscopy.
    Nature Methods, 7(5), 377-381.
    """

    _name: ClassVar[str] = "gaussian"
    total_photons: np.ndarray
    sigma: np.ndarray
    background: np.ndarray
    _overlap_fit: np.ndarray

    @property
    def loc_variance(self):
        # Mortensen et al. SI Eq. 54
        var = []
        for photons, sigma, background, overlap in zip(
            self.total_photons, self.sigma, self.background, self._overlap_fit
        ):
            if overlap:
                var.append(np.nan)
            else:
                tau = 2 * np.pi * sigma**2 * background / (photons * self.pixel_size)
                fn = lambda t: np.log(t) / (1 + t / tau)
                integral, _ = integrate.quad(fn, 0, 1)
                var.append(sigma**2 / photons * (1 + integral) ** -1)
        return var
