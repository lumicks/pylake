import pytest
import numpy as np
from pathlib import Path


def read_dataset(name):
    data = np.load(Path(__file__).parent / "data/gaussian_track_data.npz")
    position = data["position"]
    pixel_size = data["pixel_size"]
    n_frames = data["n_frames"]

    true_params = np.array([data[f"{name}_{key}"] for key in ("amplitude", "center", "scale", "offset")])
    stacker = lambda a, b: np.hstack([np.full(n_frames, true_params[0]),
                                      np.full(n_frames, true_params[1]),
                                      np.full(1 if a else n_frames, true_params[2]),
                                      np.full(1 if b else n_frames, true_params[3])])
    image_params = {(a, b): stacker(a, b) for a in (False, True) for b in (False, True)}

    expectation = data[f"{name}_expectation"]
    photon_count = data[f"{name}_photon_count"]
    line = photon_count[:,0][:, np.newaxis]

    return position, pixel_size, line, photon_count, true_params, image_params


@pytest.fixture(scope="module")
def high_intensity():
    return read_dataset("high_intensity")
