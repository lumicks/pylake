import numpy as np
from pathlib import Path
from dataclasses import dataclass, asdict
import json


save_path = Path(__file__).parent


@dataclass
class GaussianParameters1D:
    total_photons: float
    center: float
    width: float
    background: float
    pixel_size: float

    @property
    def amplitude(self):
        return self.total_photons * self.pixel_size * 1/np.sqrt(2*np.pi*self.width**2)

    def generate_coordinates(self, n_pixels):
        return np.arange(n_pixels) * self.pixel_size

    def generate_peak(self, x):
        return self.amplitude * np.exp(-0.5 * ((x - self.center) / self.width) ** 2) + self.background


def make_dataset(parameters, n_pixels=100, n_samples=1):
    coordinates = parameters[0].generate_coordinates(n_pixels)
    expectation = np.vstack([p.generate_peak(coordinates) for p in parameters]).sum(axis=0)
    photon_count = np.vstack([np.random.poisson(expectation) for _ in range(n_samples)]).T
    return {
        "parameters": json.dumps([asdict(p) for p in parameters]),
        "parameters_class": parameters[0].__class__.__name__,
        "coordinates": coordinates,
        "expectation": expectation,
        "photon_count": photon_count
    }


def read_dataset(filename):
    dataset = np.load(save_path / filename)
    parameters = json.loads(str(dataset["parameters"]))
    return dataset["coordinates"], dataset["expectation"], dataset["photon_count"], [GaussianParameters1D(**p) for p in parameters]


if __name__ == "__main__":
    np.random.seed(10071985)

    # generate single gaussian peak dataset
    parameters = [GaussianParameters1D(50, 3.500, 0.250, 1, 0.100)]
    dataset = make_dataset(parameters, n_pixels=100, n_samples=3)
    np.savez(save_path / "gaussian_data_1d.npz", **dataset)

    # generate kymo dataset
    parameters = [GaussianParameters1D(50, 3.500, 0.200, 1, 0.100),
                  GaussianParameters1D(60, 5.000, 0.200, 1, 0.100)]
    dataset = make_dataset(parameters, n_pixels=100, n_samples=5)
    np.savez(save_path / "kymo_data_2lines.npz", **dataset)
