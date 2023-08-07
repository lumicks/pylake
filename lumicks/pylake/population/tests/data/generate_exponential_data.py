import json
from pathlib import Path
from dataclasses import asdict, dataclass

import numpy as np

save_path = Path(__file__).parent


@dataclass
class ExponentialParameters:
    amplitudes: list
    lifetimes: list
    _observation_limits: tuple
    dt: float or None

    @property
    def observation_limits(self):
        return {
            "min_observation_time": self._observation_limits[0],
            "max_observation_time": self._observation_limits[1],
        }


def make_dataset(parameters, n_samples=1000):
    dwells = [
        np.random.exponential(lifetime, int(np.floor(n_samples * amplitude)))
        for amplitude, lifetime in zip(parameters.amplitudes, parameters.lifetimes)
    ]
    dwells = np.hstack(dwells)

    if parameters.dt:
        # Discretize dwells
        start_point = np.random.rand(dwells.size) * parameters.dt
        end_point = start_point + dwells

        start_point, end_point = (
            np.floor(s / parameters.dt) * parameters.dt for s in (start_point, end_point)
        )
        dwells = end_point - start_point

    dwells = dwells[
        np.logical_and(
            dwells >= parameters.observation_limits["min_observation_time"],
            dwells <= parameters.observation_limits["max_observation_time"],
        )
    ]

    return {"parameters": json.dumps(asdict(parameters)), "data": dwells}


def read_dataset(filename):
    """Read in the data from .npz file

    Each dataset consists of two items in the saved file with keys
    'parameters_{name}' and 'data_{name}'
    `ExponentialParameters` instance is reconstructed from json serialized string

    Output is a dictionary with keys 'dataset_{name}` and items as dictionaries
    containing parameters and data.
    """

    data = np.load(save_path / filename)
    names = data["names"]

    output = {}
    for name in names:
        parameters = json.loads(str(data[f"parameters_{name}"]))
        output[f"dataset_{name}"] = {
            "parameters": ExponentialParameters(**parameters),
            "data": data[f"data_{name}"],
        }
    return output


if __name__ == "__main__":
    np.random.seed(10071985)

    parameters = [
        ExponentialParameters([1], [1.5], (0.1, 120), None),
        ExponentialParameters([0.4, 0.6], [1.5, 5], (0.1, 120), None),
        ExponentialParameters([1], [1.5], (0.1, 120), 0.1),
        ExponentialParameters([0.4, 0.6], [1.5, 5], (0.1, 120), 0.1),
    ]
    names = ["1exp", "2exp", "1exp_discrete", "2exp_discrete"]

    data = {"names": names}
    for p, name in zip(parameters, names):
        output = make_dataset(p, n_samples=1000)
        data[f"parameters_{name}"] = output["parameters"]
        data[f"data_{name}"] = output["data"]

    np.savez(save_path / "exponential_data.npz", **data)
