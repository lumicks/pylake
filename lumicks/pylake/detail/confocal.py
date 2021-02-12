import json
import numpy as np
import cachetools
from deprecated.sphinx import deprecated

from .mixin import PhotonCounts
from .mixin import ExcitationLaserPower
from .image import reconstruct_image_sum, reconstruct_image, save_tiff, ImageMetadata


class BaseScan(PhotonCounts, ExcitationLaserPower):
    """Base class for confocal scans

    Parameters
    ----------
    name : str
        confocal scan name
    file : lumicks.pylake.File
        Parent file. Contains the channel data.
    start : int
        Start point in the relevant info wave.
    stop : int
        End point in the relevant info wave.
    json : dict
        Dictionary containing metadata.
    """

    def __init__(self, name, file, start, stop, json):
        self.start = start
        self.stop = stop
        self.name = name
        self._json = json
        self.file = file
        self._cache = {}

    @classmethod
    def from_dataset(cls, h5py_dset, file):
        """
        Construct a confocal class from dataset.

        Parameters
        ----------
        h5py_dset : h5py.Dataset
            The original HDF5 dataset containing confocal scan information
        file : lumicks.pylake.File
            The parent file. Used to loop up channel data
        """
        start = h5py_dset.attrs["Start time (ns)"]
        stop = h5py_dset.attrs["Stop time (ns)"]
        name = h5py_dset.name.split("/")[-1]
        try:
            json_data = json.loads(h5py_dset[()])["value0"]
        except KeyError:
            raise KeyError(f"{cls.__name__} '{name}' is missing metadata and cannot be loaded")

        return cls(name, file, start, stop, json_data)

    @property
    @deprecated(reason=("Access to raw metadata will be removed in a future release. "
                        "Use accessor properties instead. (see docs)"),
                        action="always", version="0.8.0")
    def json(self):
        return self._json

    def _get_photon_count(self, name):
        """Grab the portion of the photon count that overlaps with the scan."""
        photon_count = getattr(self.file, f"{name}_photon_count".lower())[self.start:self.stop]
        timeline_start = photon_count._src.start
        timeline_dt = photon_count._src.dt

        # Workaround for a bug in the STED delay mechanism which could result in scan start times ending up within
        # the sample time.
        if timeline_start - timeline_dt < self.start < timeline_start:
            self.start = timeline_start

        # Checks whether the scan or kymograph starts before the timeline information.
        # If this is the case, it will lead to an incorrect reconstruction.
        # If implemented, resolve the problem.
        if timeline_start > self.start:
            self._fix_incorrect_start()
            photon_count = getattr(self.file, f"{name}_photon_count".lower())[self.start:self.stop]

        return photon_count

    def _fix_incorrect_start(self):
        """Resolve error when confocal scan starts before the timeline information."""
        raise NotImplementedError

    def _plot_color(self, color, **kwargs):
        """Implementation of plotting for selected color."""
        raise NotImplementedError

    def plot_red(self, **kwargs):
        """Plot an image of the red photon channel

        Parameters
        ----------
        **kwargs
            Forwarded to :func:`matplotlib.pyplot.imshow`.
        """
        return self._plot_color("red", **kwargs)

    def plot_green(self, **kwargs):
        """Plot an image of the green photon channel

        Parameters
        ----------
        **kwargs
            Forwarded to :func:`matplotlib.pyplot.imshow`.
        """
        return self._plot_color("green", **kwargs)

    def plot_blue(self, **kwargs):
        """Plot an image of the blue photon channel

        Parameters
        ----------
        **kwargs
            Forwarded to :func:`matplotlib.pyplot.imshow`.
        """
        return self._plot_color("blue", **kwargs)

    def plot_rgb(self, **kwargs):
        """Plot all color channels

        Parameters
        ----------
        **kwargs
            Forwarded to `~matplotlib.pyplot.plot`.
        """
        raise NotImplementedError

    @property
    @deprecated(reason="By definition, confocal images always have fluorescence data.",
                version="0.8.0", action="always")
    def has_fluorescence(self) -> bool:
        return True

    @property
    @deprecated(reason="This property is always False and therefore not needed.",
                version="0.8.0", action="always")
    def has_force(self) -> bool:
        return False

    @property
    def center_point_um(self):
        """Returns a dictionary of the x/y/z center coordinates of the scan (w.r.t. brightfield field of view) """
        return self._json["scan volume"]["center point (um)"]


class ConfocalImage(BaseScan):

    def _ordered_axes(self):
        """Returns axis indices in spatial order"""
        return sorted(self._json["scan volume"]["scan axes"], key=lambda x: x["axis"])

    def _to_spatial(self, data):
        """Implements any necessary post-processing actions after image reconstruction from infowave """
        raise NotImplementedError

    @cachetools.cachedmethod(lambda self: self._cache)
    def _image(self, channel):
        channel_data = getattr(self, f"{channel}_photon_count").data
        raw_image = reconstruct_image_sum(channel_data, self.infowave.data, self._shape)
        return self._to_spatial(raw_image)

    @cachetools.cachedmethod(lambda self: self._cache)
    def _timestamps(self, channel):
        assert channel == "timestamps"
        # Uses the timestamps from the first non-zero-sized photon channel
        for color in ("red", "green", "blue"):
            channel_data = getattr(self, f"{color}_photon_count").timestamps
            if len(channel_data) != 0:
                break
        else:
            raise RuntimeError("Can't get pixel timestamps if there are no pixels")
        raw_image = reconstruct_image(channel_data, self.infowave.data, self._shape, reduce=np.mean)
        return self._to_spatial(raw_image)

    def _plot_color(self, color, **kwargs):
        from matplotlib.colors import LinearSegmentedColormap

        linear_colormaps = {
            "red": LinearSegmentedColormap.from_list("red", colors=[(0, 0, 0), (1, 0, 0)]),
            "green": LinearSegmentedColormap.from_list("green", colors=[(0, 0, 0), (0, 1, 0)]),
            "blue": LinearSegmentedColormap.from_list("blue", colors=[(0, 0, 0), (0, 0, 1)]),
        }

        image = getattr(self, f"{color}_image")
        return self._plot(image, **{"cmap": linear_colormaps[color], **kwargs})

    def _plot(self, image, **kwargs):
        raise NotImplementedError

    def plot_rgb(self, **kwargs):
        """Plot a full rbg kymograph image

        Parameters
        ----------
        **kwargs
            Forwarded to :func:`matplotlib.pyplot.imshow`.
        """
        image = self.rgb_image
        image = image / np.max(image)
        return self._plot(image, **kwargs)

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
            save_tiff(self.rgb_image, filename, dtype, clip, ImageMetadata.from_dataset(self._json))
        else:
            raise RuntimeError("Can't export TIFF if there are no pixels")

    @property
    def infowave(self):
        return self.file["Info wave"]["Info wave"][self.start:self.stop]

    @property
    def _shape(self):
        """The shape of the image ([optional: pixels on slow axis], pixels on fast axis)"""
        raise NotImplementedError

    @property
    def pixels_per_line(self):
        return self._fast_axis_metadata["num of pixels"]

    @property
    def _fast_axis_metadata(self):
        return self._json["scan volume"]["scan axes"][0]

    @property
    def fast_axis(self):
        return "X" if self._fast_axis_metadata["axis"] == 0 else "Y"

    @property
    def timestamps(self) -> np.ndarray:
        """Timestamps for image pixels, not for samples

        The returned array has the same shape as the `*_image` arrays.
        """
        return self._timestamps("timestamps")

    @property
    def pixelsize_um(self):
        """Returns a `List` of axes dimensions in um. The length of the list corresponds to the number of scan axes."""
        return [axes["pixel size (nm)"] / 1000 for axes in self._ordered_axes()]

    @property
    def scan_width_um(self):
        """Returns a `List` of scan widths in um along axes. The length of the list corresponds to the number of scan axes."""
        return [axes["scan width (um)"] for axes in self._ordered_axes()]

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
