import numpy as np
from scipy import integrate
from dataclasses import dataclass
from typing import ClassVar


@dataclass(frozen=True)
class LocalizationModel:
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
    _name: ClassVar[str] = "gaussian"
    total_photons: np.ndarray
    sigma: np.ndarray
    background: np.ndarray

    @property
    def loc_variance(self):
        """
        [ ] Mortensen, K. I., Churchman, L. S., Spudich, J. A., & Flyvbjerg, H. (2010).
        Optimized localization analysis for single-molecule tracking and super-resolution microscopy.
        Nature Methods, 7(5), 377-381.
        SI Eq. 54
        """

        var = []
        for p, s, b in zip(self.total_photons, self.sigma, self.background):
            tau = 2 * np.pi * s ** 2 * b / (p * self.pixel_size)
            fn = lambda t: np.log(t) / (1 + t / tau)
            integral, _ = integrate.quad(fn, 0, 1)
            var.append(s ** 2 / p * (1 + integral) ** -1)
        return var
