import numpy as np
from pathlib import Path
from dataclasses import dataclass, asdict
import json


save_path = Path(__file__).parent


@dataclass
class ExponentialParameters:
    amplitudes: np.ndarray
    lifetimes: np.ndarray
    observation_limits: tuple


def make_dataset(parameters, n_samples=1000):
    dwells = []
    for amplitude, lifetime in zip(parameters.amplitudes, parameters.lifetimes):
        n = int(np.floor(n_samples * amplitude))
        tmp = np.random.exponential(lifetime, n)
        idx = np.logical_and(tmp >= parameters.observation_limits[0],
                             tmp <= parameters.observation_limits[1])
        dwells.append(tmp[idx])
    dwells = np.hstack(dwells)
    return {"parameters": json.dumps(asdict(parameters)),
            "data": dwells}


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
            "data": data[f"data_{name}"]
        }
    return output


if __name__ == "__main__":
    np.random.seed(10071985)

    parameters = [ExponentialParameters([1], [1.5], (0.1, 120)),
                  ExponentialParameters([0.4, 0.6], [1.5, 5], (0.1, 120))]
    names = ["1exp", "2exp"]

    data = {"names": names}
    for p, name in zip(parameters, names):
        output = make_dataset(p, n_samples=1000)
        data[f"parameters_{name}"] = output["parameters"]
        data[f"data_{name}"] = output["data"]

    np.savez(save_path / "exponential_data.npz", **data)
