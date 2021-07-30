import json
import numpy as np
import cachetools
from deprecated.sphinx import deprecated

from .mixin import PhotonCounts
from .mixin import ExcitationLaserPower
from .image import reconstruct_image_sum, reconstruct_image, save_tiff, ImageMetadata


def _default_image_factory(self: "ConfocalImage", color):
    channel_data = getattr(self, f"{color}_photon_count").data
    raw_image = reconstruct_image_sum(channel_data, self.infowave.data, self._shape)
    return self._to_spatial(raw_image)


def _default_timestamp_factory(self: "ConfocalImage", reduce=np.mean):
    # Uses the timestamps from the first non-zero-sized photon channel
    for color in ("red", "green", "blue"):
        channel_data = getattr(self, f"{color}_photon_count").timestamps
        if len(channel_data) != 0:
            break
    else:
        raise RuntimeError("Can't get pixel timestamps if there are no pixels")
    raw_image = reconstruct_image(channel_data, self.infowave.data, self._shape, reduce=reduce)
    return self._to_spatial(raw_image)


def _default_pixelsize_factory(self: "ConfocalImage"):
    return [axes["pixel size (nm)"] / 1000 for axes in self._ordered_axes()]


def _default_pixelcount_factory(self: "ConfocalImage"):
    return [axes["num of pixels"] for axes in self._ordered_axes()]


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
        self._image_factory = _default_image_factory
        self._timestamp_factory = _default_timestamp_factory
        self._pixelsize_factory = _default_pixelsize_factory
        self._pixelcount_factory = _default_pixelcount_factory
        self._cache = {}

    def _has_default_factories(self):
        return (
            self._image_factory == _default_image_factory
            and self._timestamp_factory == _default_timestamp_factory
        )

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
    @deprecated(
        reason=(
            "Access to raw metadata will be removed in a future release. "
            "Use accessor properties instead. (see docs)"
        ),
        action="always",
        version="0.8.0",
    )
    def json(self):
        return self._json

    def _get_photon_count(self, name):
        """Grab the portion of the photon count that overlaps with the scan."""
        photon_count = getattr(self.file, f"{name}_photon_count".lower())[self.start : self.stop]
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
            photon_count = getattr(self.file, f"{name}_photon_count".lower())[
                self.start : self.stop
            ]

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
    @deprecated(
        reason="By definition, confocal images always have fluorescence data.",
        version="0.8.0",
        action="always",
    )
    def has_fluorescence(self) -> bool:
        return True

    @property
    @deprecated(
        reason="This property is always False and therefore not needed.",
        version="0.8.0",
        action="always",
    )
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

    @property
    def _scan_order(self):
        """Order in which the axes are scanned. Assume we have an Y, X scan, then the physical
        axis order is X, Y. In that case, this function would return [1, 0]. For an X, Z scan
        physical axes order would be X, Z, so in that case this function would return [0, 1]."""
        return np.argsort([x["axis"] for x in self._json["scan volume"]["scan axes"]])

    def _to_spatial(self, data):
        """Implements any necessary post-processing actions after image reconstruction from infowave """
        raise NotImplementedError

    @cachetools.cachedmethod(lambda self: self._cache)
    def _image(self, channel):
        assert channel in ("red", "green", "blue")
        return self._image_factory(self, channel)

    @cachetools.cachedmethod(lambda self: self._cache)
    def _timestamps(self, channel, reduce=np.mean):
        assert channel == "timestamps"
        return self._timestamp_factory(self, reduce)

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
        return self.file["Info wave"]["Info wave"][self.start : self.stop]

    @property
    def _shape(self):
        """The shape of the image ([optional: pixels on slow axis], pixels on fast axis)"""
        raise NotImplementedError

    @property
    def pixels_per_line(self):
        return self._num_pixels[self._scan_order[0]]

    @property
    def _num_pixels(self):
        return self._pixelcount_factory(self)

    @property
    def fast_axis(self):
        return "X" if self._json["scan volume"]["scan axes"][0]["axis"] == 0 else "Y"

    @property
    def timestamps(self) -> np.ndarray:
        """Timestamps for image pixels, not for samples

        The returned array has the same shape as the `*_image` arrays.
        """
        return self._timestamps("timestamps")

    @property
    def pixelsize_um(self):
        """Returns a `List` of axes dimensions in um. The length of the list corresponds to the
        number of scan axes."""
        return self._pixelsize_factory(self)

    @property
    def size_um(self):
        """Returns a `List` of scan sizes in um along axes. The length of the list corresponds to
        the number of scan axes."""
        return list(
            map(
                lambda pixel_size, num_pixels: pixel_size * num_pixels,
                self.pixelsize_um,
                self._num_pixels,
            )
        )

    @property
    @deprecated(
        reason=(
            "The property `scan_width_um` has been deprecated. Use `size_um` to get the actual "
            "size of the scan. When performing a scan, Bluelake determines an appropriate scan "
            "width based on the desired pixel size and the desired scan width. This means that the "
            "size of the performed scan could deviate from the width provided in this property."
        ),
        action="always",
        version="0.8.2",
    )
    def scan_width_um(self):
        """Returns a `List` of scan widths as configured in the Bluelake UI. The length of the list
        corresponds to the number of scan axes. Note that these widths can deviate from the actual
        scan widths performed in practice"""
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
