import json
import math
import numpy as np


def reconstruct_image(data, infowave, pixels_per_line, reduce=np.sum):
    """Reconstruct a scan or kymograph image from raw data

    Parameters
    ----------
    data : array_like
        Raw data to use for the reconstruction. E.g. photon counts or force samples.
    infowave : array_like
        The famous infowave.
    pixels_per_line : int
        The number of pixels on the fast axis of the scan.
    reduce : callable
        A function which reduces multiple sample into a pixel. Usually `np.sum`
        for photon counts and `np.mean` for force samples.

    Returns
    -------
    np.ndarray
    """
    assert data.size == infowave.size
    code = {"discard": 0, "use": 1, "pixel_boundary": 2}

    # Example infowave:
    #  1 0 0 1 0 1 2 0 1 0 0 1 0 1 0 1 0 0 1 0 2 0 1 0 1 0 0 1 0 1 0 1 2 1 0 0 1
    #              ^ <-----------------------> ^                       ^
    #                       one pixel
    valid_idx = infowave != code["discard"]
    infowave = infowave[valid_idx]

    # After discard:
    #  1 1 1 2 1 1 1 1 1 2 1 1 1 1 1 2 1 1 1
    #        ^ <-------> ^           ^
    #         pixel_size (i.e. data samples per pixel)
    pixel_sizes = np.diff(np.flatnonzero(infowave == code["pixel_boundary"]))
    pixel_size = pixel_sizes[0]
    # For now we assume that every pixel consists of the same number of samples
    assert np.all(pixel_sizes == pixel_size)

    def round_up(size, n):
        """Round up `size` to the nearest multiple of `n`"""
        return int(math.ceil(size / n)) * n

    data = data[valid_idx]
    data.resize(round_up(data.size, pixel_size))

    pixels = reduce(data.reshape(-1, pixel_size), axis=1)
    pixels.resize(round_up(pixels.size, pixels_per_line))
    return pixels.reshape(-1, pixels_per_line)


class Kymo:
    """A Kymograph exported from Bluelake

    Parameters
    ----------
    h5py_dset : h5py.Dataset
        The original HDF5 dataset containing kymo information
    file : lumicks.hdf5.File
        The parent file. Used to loop up channel data
    """
    def __init__(self, h5py_dset, file):
        self.start = h5py_dset.attrs["Start time (ns)"]
        self.stop = h5py_dset.attrs["Stop time (ns)"]
        self.name = h5py_dset.name.split("/")[-1]
        self.json = json.loads(h5py_dset.value)["value0"]
        self.file = file

    @property
    def has_fluorescence(self) -> bool:
        return self.json["fluorescence"]

    @property
    def has_force(self) -> bool:
        return self.json["force"]

    @property
    def infowave(self):
        return self.file["Info wave"]["Info wave"][self.start:self.stop]

    @property
    def pixels_per_line(self):
        return self.json["scan volume"]["scan axes"][0]["num of pixels"]

    @property
    def red_photons(self):
        return self.file.red_photons[self.start:self.stop]

    @property
    def green_photons(self):
        return self.file.green_photons[self.start:self.stop]

    @property
    def blue_photons(self):
        return self.file.blue_photons[self.start:self.stop]

    @property
    def red_image(self):
        return reconstruct_image(self.red_photons.data, self.infowave.data, self.pixels_per_line)

    @property
    def green_image(self):
        return reconstruct_image(self.green_photons.data, self.infowave.data, self.pixels_per_line)

    @property
    def blue_image(self):
        return reconstruct_image(self.blue_photons.data, self.infowave.data, self.pixels_per_line)

    @property
    def rgb_image(self):
        color_channels = [getattr(self, f"{color}_image") for color in ("red", "green", "blue")]
        return np.stack(color_channels).T

    def _plot(self, image, **kwargs):
        import matplotlib.pyplot as plt

        width_um = self.json["scan volume"]["scan axes"][0]["scan width (um)"]
        duration = (self.stop - self.start) / 1e9

        default_kwargs = dict(
            extent=[0, duration, 0, width_um],
            aspect=(image.shape[0] / image.shape[1]) * (duration / width_um)
        )

        plt.imshow(image, **{**default_kwargs, **kwargs})
        plt.xlabel("time (s)")
        plt.ylabel(r"position ($\mu$m)")
        plt.title(self.name)

    def _plot_color(self, color, **kwargs):
        from matplotlib.colors import LinearSegmentedColormap

        linear_colormaps = {
            "red": LinearSegmentedColormap.from_list("red", colors=[(0, 0, 0), (1, 0, 0)]),
            "green": LinearSegmentedColormap.from_list("green", colors=[(0, 0, 0), (0, 1, 0)]),
            "blue": LinearSegmentedColormap.from_list("blue", colors=[(0, 0, 0), (0, 0, 1)]),
        }

        image = getattr(self, f"{color}_image").T
        self._plot(image, **{"cmap": linear_colormaps[color], **kwargs})

    def plot_red(self, **kwargs):
        """Plot an image of the red photon channel

        Parameters
        ----------
        **kwargs
            Forwarded to `~matplotlib.pyplot.imshow`.
        """
        self._plot_color("red", **kwargs)

    def plot_green(self, **kwargs):
        """Plot an image of the green photon channel

        Parameters
        ----------
        **kwargs
            Forwarded to `~matplotlib.pyplot.imshow`.
        """
        self._plot_color("green", **kwargs)

    def plot_blue(self, **kwargs):
        """Plot an image of the blue photon channel

        Parameters
        ----------
        **kwargs
            Forwarded to `~matplotlib.pyplot.imshow`.
        """
        self._plot_color("blue", **kwargs)

    def plot_rgb(self, **kwargs):
        """Plot a full rbg kymograph image

        Parameters
        ----------
        **kwargs
            Forwarded to `~matplotlib.pyplot.imshow`.
        """
        image = self.rgb_image
        image = image / np.max(image)
        self._plot(image, **kwargs)
