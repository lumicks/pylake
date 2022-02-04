import numpy as np
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

