import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def generate_peak(x, amplitude, center, scale, offset):
    return amplitude * np.exp(-0.5 * ((x - center) / scale) ** 2) + offset


def sample_image(expectation, n_frames):
    expectation = np.repeat(expectation[:, np.newaxis], n_frames, axis=1)
    return np.random.poisson(expectation)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    np.random.seed(10071985)
    save_path = Path(__file__).parent / "gaussian_track_data.npz"

    pixel_size = 0.1
    n_frames = 3

    position = np.arange(0, 5, 0.1)
    params = {
        "high_intensity": {"amplitude": 1500, "center": 2.6, "scale": 0.35, "offset": 2},
    }

    data = {"position": position,
            "pixel_size": pixel_size,
            "n_frames": n_frames}
    for key, p in params.items():
        expectation = generate_peak(position, **p)
        photon_count = sample_image(expectation, n_frames)

        data[f"{key}_expectation"] = expectation
        data[f"{key}_photon_count"] = photon_count
        for pkey, val in p.items():
            data[f"{key}_{pkey}"] = val

        plt.plot(position, photon_count)
        plt.show()

    np.savez(save_path, **data)
