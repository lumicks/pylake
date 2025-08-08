from enum import Enum
from collections import namedtuple
from dataclasses import dataclass

import numpy as np

UnitInfo = namedtuple("UnitInfo", ["name", "label"])


class PositionUnit(Enum):
    um = UnitInfo(name="um", label=r"μm")
    kbp = UnitInfo(name="kbp", label="kbp")
    pixel = UnitInfo(name="pixel", label="pixels")
    au = UnitInfo(name="au", label="au")

    def __str__(self):
        return self.value.name

    def __hash__(self):
        return hash(self.value)

    @property
    def label(self):
        return self.value.label

    def get_diffusion_labels(self) -> dict:
        return {
            "unit": f"{self}^2 / s",
            "_unit_label": f"{self.label}²/s",
        }

    def get_squared_labels(self) -> dict:
        return {
            "unit": f"{self}^2",
            "_unit_label": f"{self.label}²",
        }


@dataclass(frozen=True)
class PositionCalibration:
    unit: PositionUnit = PositionUnit.pixel
    scale: float = 1.0
    origin: float = 0.0

    def __post_init__(self):
        if not isinstance(self.unit, PositionUnit):
            raise TypeError("`unit` must be a PositionUnit instance")

    def from_pixels(self, pixels):
        """Convert coordinates from pixel values to calibrated values"""
        return self.scale * (np.array(pixels) - self.origin)

    def to_pixels(self, calibrated):
        """Convert coordinates from calibrated values to pixel values"""
        return np.array(calibrated) / self.scale + self.origin

    @property
    def pixelsize(self):
        return np.abs(self.scale)

    def downsample(self, factor):
        return (
            self
            if self.unit == PositionUnit.pixel
            else PositionCalibration(self.unit, self.scale * factor)
        )
