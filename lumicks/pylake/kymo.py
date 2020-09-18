import json
import numpy as np
import warnings

from .detail.mixin import PhotonCounts
from .detail.mixin import ExcitationLaserPower
from .detail.image import reconstruct_image_sum, reconstruct_image, save_tiff, ImageMetadata, line_timestamps_image, \
    seek_timestamp_next_line
from .detail.timeindex import to_timestamp


"""Axis label used for plotting"""
axis_label = ("x", "y", "z")


class Kymo(PhotonCounts, ExcitationLaserPower):
    """A Kymograph exported from Bluelake

    Parameters
    ----------
    name : str
        Kymograph name
    file : lumicks.pylake.File
        Parent file. Contains the channel data.
    start : int
        Start point in the relevant info wave.
    stop : int
        End point in the relevant info wave.
    json : dict
        Dictionary containing kymograph-specific metadata.
    """
    def __init__(self, name, file, start, stop, json):
        self.start = start
        self.stop = stop
        self.name = name
        self.json = json
        self.file = file
        self._cache = {}

    def __repr__(self):
        name = self.__class__.__name__
        return f"{name}(pixels={self.pixels_per_line})"

    def __getitem__(self, item):
        """All indexing is in timestamp units (ns)"""
        if not isinstance(item, slice):
            raise IndexError("Scalar indexing is not supported, only slicing")
        if item.step is not None:
            raise IndexError("Slice steps are not supported")

        start = self.start if item.start is None else item.start
        stop = self.stop if item.stop is None else item.stop
        start, stop = (to_timestamp(v, self.start, self.stop) for v in (start, stop))

        timestamps = self.infowave.timestamps
        line_timestamps = line_timestamps_image(timestamps, self.infowave.data, self.pixels_per_line)
        line_timestamps = np.append(line_timestamps, timestamps[-1])

        i_min = np.searchsorted(line_timestamps, start, side='left')
        i_max = np.searchsorted(line_timestamps, stop, side='left')

        if i_min >= len(line_timestamps):
            return EmptyKymo(self.name, self.file, line_timestamps[-1], line_timestamps[-1], self.json)

        if i_min >= i_max:
            return EmptyKymo(self.name, self.file, line_timestamps[i_min], line_timestamps[i_min], self.json)

        if i_max < len(line_timestamps):
            stop = line_timestamps[i_max]

        start = line_timestamps[i_min]

        return Kymo(self.name, self.file, start, stop, self.json)

    def _get_photon_count(self, name):
        photon_count = getattr(self.file, f"{name}_photon_count".lower())[self.start:self.stop]

        if self._has_incorrect_start(photon_count._src.start):
            photon_count = getattr(self.file, f"{name}_photon_count".lower())[self.start:self.stop]

        return photon_count

    def _ordered_axes(self):
        """Returns axis indices in spatial order"""
        return sorted(self.json["scan volume"]["scan axes"], key=lambda x: x["axis"])

    @property
    def _fast_axis_metadata(self):
        return self.json["scan volume"]["scan axes"][0]

    @property
    def fast_axis(self):
        return "X" if self._fast_axis_metadata["axis"] == 0 else "Y"

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
        return self._fast_axis_metadata["num of pixels"]

    def _image(self, color):
        if color not in self._cache:
            photon_counts = getattr(self, f"{color}_photon_count").data
            self._cache[color] = reconstruct_image_sum(photon_counts, self.infowave.data, self.pixels_per_line).T
        return self._cache[color]

    def _has_incorrect_start(self, timeline_start):
        """Checks whether the scan or kymograph starts before the timeline information. If this is the case, it will
        lead to an incorrect reconstruction. For kymographs this is recoverable by omitting the first line. For scans
        it is currently unrecoverable."""
        if timeline_start > self.start:
            if type(self) == Kymo:
                self.start = seek_timestamp_next_line(self.infowave[self.start:])
                self._cache = {}
                warnings.warn("Start of the kymograph was truncated. Omitting the truncated first line.",
                              RuntimeWarning)
                return True
            else:
                raise RuntimeError("Start of the scan was truncated. Reconstruction cannot proceed. Did you export the "
                                   "entire scan time in Bluelake?")

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

        width_um = self._ordered_axes()[0]["scan width (um)"]
        duration = (self.stop - self.start) / 1e9

        default_kwargs = dict(
            # With origin set to upper (default) bounds should be given as (0, n, n, 0)
            extent=[0, duration, width_um, 0],
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
            Forwarded to :func:`matplotlib.pyplot.imshow`.
        """
        self._plot_color("red", **kwargs)

    def plot_green(self, **kwargs):
        """Plot an image of the green photon channel

        Parameters
        ----------
        **kwargs
            Forwarded to :func:`matplotlib.pyplot.imshow`.
        """
        self._plot_color("green", **kwargs)

    def plot_blue(self, **kwargs):
        """Plot an image of the blue photon channel

        Parameters
        ----------
        **kwargs
            Forwarded to :func:`matplotlib.pyplot.imshow`.
        """
        self._plot_color("blue", **kwargs)

    def plot_rgb(self, **kwargs):
        """Plot a full rbg kymograph image

        Parameters
        ----------
        **kwargs
            Forwarded to :func:`matplotlib.pyplot.imshow`.
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
        if self.rgb_image.size > 0:
            save_tiff(self.rgb_image, filename, dtype, clip, ImageMetadata.from_dataset(self.json))
        else:
            raise RuntimeError("Can't export TIFF if there are no pixels")

    @classmethod
    def from_dataset(cls, h5py_dset, file):
        """
        Construct Kymograph class from dataset.

        Parameters
        ----------
        h5py_dset : h5py.Dataset
            The original HDF5 dataset containing kymo information
        file : lumicks.pylake.File
            The parent file. Used to loop up channel data
        """
        start = h5py_dset.attrs["Start time (ns)"]
        stop = h5py_dset.attrs["Stop time (ns)"]
        name = h5py_dset.name.split("/")[-1]
        json_data = json.loads(h5py_dset[()])["value0"]
        return cls(name, file, start, stop, json_data)


class EmptyKymo(Kymo):
    def plot_rgb(self):
        raise RuntimeError("Cannot plot empty kymograph")

    def _plot(self, image, **kwargs):
        raise RuntimeError("Cannot plot empty kymograph")

    def _image(self, color):
        return np.empty((self.pixels_per_line, 0))

