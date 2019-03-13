import json
import numpy as np

from .detail.mixin import PhotonCounts
from .detail.image import reconstruct_image, save_tiff


class Kymo(PhotonCounts):
    """A Kymograph exported from Bluelake

    Parameters
    ----------
    h5py_dset : h5py.Dataset
        The original HDF5 dataset containing kymo information
    file : lumicks.pylake.File
        The parent file. Used to loop up channel data
    """
    def __init__(self, h5py_dset, file):
        self.start = h5py_dset.attrs["Start time (ns)"]
        self.stop = h5py_dset.attrs["Stop time (ns)"]
        self.name = h5py_dset.name.split("/")[-1]
        self.json = json.loads(h5py_dset[()])["value0"]
        self.file = file
        self._cache = {}

    def __repr__(self):
        name = self.__class__.__name__
        return f"{name}(pixels={self.pixels_per_line})"

    def _get_photon_count(self, name):
        return getattr(self.file, f"{name}_photon_count".lower())[self.start:self.stop]

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

    def _image(self, color):
        if color not in self._cache:
            photon_counts = getattr(self, f"{color}_photon_count").data
            self._cache[color] = reconstruct_image(photon_counts, self.infowave.data,
                                                   self.pixels_per_line).T
        return self._cache[color]

    def _timestamps(self, sample_timestamps):
        return reconstruct_image(sample_timestamps, self.infowave.data,
                                 self.pixels_per_line, reduce=np.mean).T

    @property
    def red_image(self):
        return self._image("red")

    @property
    def green_image(self):
        return self._image("green")

    @property
    def blue_image(self):
        return self._image("blue")

    @property
    def rgb_image(self):
        color_channels = [getattr(self, f"{color}_image").T for color in ("red", "green", "blue")]
        return np.stack(color_channels).T

    @property
    def timestamps(self) -> np.ndarray:
        """Timestamps for image pixels, not for samples

        The returned array has the same shape as the `*_image` arrays.
        """
        # Uses the timestamps from the first non-zero-sized photon channel
        photon_counts = self.red_photon_count, self.green_photon_count, self.blue_photon_count
        for photon_count in photon_counts:
            if len(photon_count) == 0:
                continue
            return self._timestamps(photon_count.timestamps)
        raise RuntimeError("Can't get pixel timestamps if there are no pixels")

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

        image = getattr(self, f"{color}_image")
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

    def save_tiff(self, filename, dtype=np.float32, clip=False):
        """Save the RGB photon counts to a TIFF image

        Parameters
        ----------
        filename : str
            The name of the TIFF file where the image will be saved.
        dtype : np.dtype
            The data type of a single color channel in the resulting image.
        clip : bool
            If enabled, the photon count data will be clipped to fit into the desired `dtype`.
            This option is disabled by default: an error will be raise if the data does not fit.
        """
        save_tiff(self.rgb_image, filename, dtype, clip)
