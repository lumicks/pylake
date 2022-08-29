import json
import numpy as np
import cachetools
from deprecated.sphinx import deprecated
from dataclasses import dataclass
from typing import List
import warnings

from .mixin import PhotonCounts
from .mixin import ExcitationLaserPower
from .image import reconstruct_image_sum, reconstruct_image, save_tiff
from .utilities import could_sum_overflow
from ..adjustments import ColorAdjustment
from matplotlib.colors import LinearSegmentedColormap

linear_colormaps = {
    "red": LinearSegmentedColormap.from_list("red", colors=[(0, 0, 0), (1, 0, 0)]),
    "green": LinearSegmentedColormap.from_list("green", colors=[(0, 0, 0), (0, 1, 0)]),
    "blue": LinearSegmentedColormap.from_list("blue", colors=[(0, 0, 0), (0, 0, 1)]),
    "rgb": None,
}


def _int_mean(a, total_size, axis):
    """Compute the `mean` for `a`.

    When the mean can safely be computed for the entire block, we simply do so in a single step.
    The early return will actually be called in most cases. When an overflow would occur, we
    split the blocks to sum over recursively until the individual blocks can be summed without
    incurring an overflow."""
    if not could_sum_overflow(a, axis):
        return np.sum(a, axis) // total_size

    # Swap axis is used so the dimension we average over is the first. This makes it easier to
    # divide it into blocks and sum over them. It also generalizes between nD and 1D.
    b = a if axis is None else np.swapaxes(a, axis, 0)
    n = b.shape[0] // 2
    return _int_mean(b[:n], total_size, 0) + _int_mean(b[n:], total_size, 0)


def timestamp_mean(a, axis=None):
    """An overflow protected `mean` for `timestamps`."""
    # By subtracting the minimal timestamp first, we greatly reduce the magnitude of the values we
    # we have to sum over to obtain the mean (since now we are not dealing with timestamps since
    # epoch, but timestamps since start of this timestamp array).
    minimum = np.min(a)
    return minimum + _int_mean(a - minimum, a.size if axis is None else a.shape[axis], axis)


def _default_image_factory(self: "ConfocalImage", color):
    channel = getattr(self, f"{color}_photon_count")
    raw_image = reconstruct_image_sum(
        channel.data.astype(float) if channel else np.zeros(self.infowave.data.size),
        self.infowave.data,
        self._reconstruction_shape,
    )
    return self._to_spatial(raw_image)


def _default_timestamp_factory(self: "ConfocalImage", reduce=timestamp_mean):
    # Uses the timestamps from the first non-zero-sized photon channel
    for color in ("red", "green", "blue"):
        channel_data = getattr(self, f"{color}_photon_count").timestamps
        if len(channel_data) != 0:
            break
    else:
        raise RuntimeError("Can't get pixel timestamps if there are no pixels")

    raw_image = reconstruct_image(
        channel_data, self.infowave.data, self._reconstruction_shape, reduce=reduce
    )
    return self._to_spatial(raw_image)


def _default_pixelsize_factory(self: "ConfocalImage"):
    return [axes.pixel_size_um for axes in self._metadata.ordered_axes]


def _default_pixelcount_factory(self: "ConfocalImage"):
    return [axes.num_pixels for axes in self._metadata.ordered_axes]


@dataclass(frozen=True)
class ScanAxis:
    axis: int
    num_pixels: int
    pixel_size_um: float

    @property
    def axis_label(self):
        return ("X", "Y", "Z")[self.axis]


@dataclass(frozen=True)
class ScanMetaData:
    """Scan metadata

    Parameters
    ----------
    scan_axes : list[ScanAxis]
        Scan axes ordered by scanning speed (first axis being the fast axis).
    center_point_um : np.ndarray
        Center point of the scan.
    num_frames : int
        Number of frames in the scan.
    """

    scan_axes: List[ScanAxis]
    center_point_um: np.ndarray
    num_frames: int  # can be 0 for Scan, in which case it needs to be reconstructed from infowave

    def with_num_frames(self, num_frames):
        """Returns new scan metadata with different number of frames"""
        return ScanMetaData(self.scan_axes, self.center_point_um, num_frames)

    @property
    def num_axes(self):
        return len(self.scan_axes)

    @property
    def ordered_axes(self):
        """Axis ordered by spatial axis"""
        return sorted(self.scan_axes, key=lambda x: x.axis)

    @property
    def fast_axis(self):
        return self.scan_axes[0].axis_label

    @property
    def scan_order(self):
        """Order in which the axes are scanned.

        Assume we have an Y, X scan, then the physical axis order is X, Y. In that case, this
        function would return [1, 0]. For an X, Z scan physical axes order would be X, Z, so in
        that case this function would return [0, 1]."""
        return np.argsort([x.axis for x in self.scan_axes])

    @classmethod
    def from_json(cls, json_string):
        json_dict = json.loads(json_string)["value0"]

        axes = [
            ScanAxis(ax["axis"], ax["num of pixels"], ax["pixel size (nm)"] / 1000)
            for ax in json_dict["scan volume"]["scan axes"]
        ]

        return cls(axes, json_dict["scan volume"]["center point (um)"], json_dict["scan count"])


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
    metadata : ScanMetaData
        Metadata.
    """

    def __init__(self, name, file, start, stop, metadata):
        self.start = start
        self.stop = stop
        self.name = name
        self._metadata = metadata
        self._file = file
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
            metadata = ScanMetaData.from_json(h5py_dset[()])
        except KeyError:
            raise KeyError(f"{cls.__name__} '{name}' is missing metadata and cannot be loaded")

        return cls(name, file, start, stop, metadata)

    @property
    def file(self):
        if self._file is None:
            raise ValueError(f"There is no .h5 file associated with this {self.__class__.__name__}")
        else:
            return self._file

    @property
    def pixel_time_seconds(self):
        """Pixel dwell time in seconds"""
        raise NotImplementedError("Pixel dwell times have not been implemented for this class.")

    def __copy__(self):
        instance = self.__class__(
            name=self.name,
            file=self._file,
            start=self.start,
            stop=self.stop,
            metadata=self._metadata,
        )

        # Preserve custom factories
        instance._image_factory = self._image_factory
        instance._timestamp_factory = self._timestamp_factory
        instance._pixelsize_factory = self._pixelsize_factory
        instance._pixelcount_factory = self._pixelcount_factory
        return instance

    def _get_photon_count(self, name):
        """Grab the portion of the photon count that overlaps with the scan."""
        photon_count = getattr(self.file, f"{name}_photon_count".lower())[self.start : self.stop]

        # Channel `name` does not exist
        if not photon_count:
            return photon_count

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

    @deprecated(
        reason=("`plot_red()` is deprecated. Use `plot(channel='red')` instead."),
        version="0.11.1",
        action="always",
    )
    def plot_red(self, **kwargs):
        """Plot an image of the red photon channel"""
        return self.plot(channel="red", **kwargs)

    @deprecated(
        reason="`plot_green()` is deprecated. Use `plot(channel='green')` instead.",
        version="0.11.1",
        action="always",
    )
    def plot_green(self, **kwargs):
        """Plot an image of the green photon channel"""
        return self.plot(channel="green", **kwargs)

    @deprecated(
        reason="`plot_blue()` is deprecated. Use `plot(channel='blue')` instead.",
        version="0.11.1",
        action="always",
    )
    def plot_blue(self, **kwargs):
        """Plot an image of the blue photon channel"""
        return self.plot(channel="blue", **kwargs)

    @deprecated(
        reason="`plot_rgb()` is deprecated. Use `plot(channel='rgb')` instead.",
        version="0.11.1",
        action="always",
    )
    def plot_rgb(self, **kwargs):
        """Plot an image of all color channels."""
        return self.plot(channel="rgb", **kwargs)

    def _get_plot_data(self, channel):
        """Get data for plotting requested channel."""
        raise NotImplementedError

    def _plot(self, channel, axes, **kwargs):
        """Internal implementation of the plotting."""
        raise NotImplementedError

    def plot(self, channel, axes=None, **kwargs):
        """Show a formatted plot for the requested color channel.

        Parameters
        ----------
        channel : {'red', 'green', 'blue', 'rgb'}
            Color channel to plot.
        axes : mpl.axes.Axes or None
            If supplied, the axes instance in which to plot.
        **kwargs
            Forwarded to :func:`matplotlib.pyplot.plot` or :func:`matplotlib.pyplot.imshow`
        """
        import matplotlib.pyplot as plt

        if axes is None:
            axes = plt.gca()
        return self._plot(channel, axes=axes, **kwargs)

    @property
    def center_point_um(self):
        """Returns a dictionary of the x/y/z center coordinates of the scan (w.r.t. brightfield
        field of view)"""
        return self._metadata.center_point_um


class ConfocalImage(BaseScan):
    def _to_spatial(self, data):
        """Implements any necessary post-processing actions after image reconstruction from infowave"""
        raise NotImplementedError

    @cachetools.cachedmethod(lambda self: self._cache)
    def _image(self, channel):
        assert channel in ("red", "green", "blue")
        return self._image_factory(self, channel)

    @cachetools.cachedmethod(lambda self: self._cache)
    def _timestamps(self, channel, reduce=timestamp_mean):
        assert channel == "timestamps"
        return self._timestamp_factory(self, reduce)

    def _get_plot_data(self, channel, adjustment=ColorAdjustment.nothing(), frame=None):
        """Get image data for plotting requested channel.

        Parameters
        ----------
        channel : str
            Which channel to return. Options are: "red", "green", "blue" or "rgb".
        adjustment : lk.ColorAdjustment
            Color adjustments to apply to the output image if channel is set to "rgb".
        """
        image = self.get_image(channel)
        frame_image = image if frame is None else image[frame]

        if channel == "rgb":
            frame_image = adjustment._get_data_rgb(frame_image)

        return frame_image

    def export_tiff(self, filename, *, dtype=np.float32, clip=False):
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
        try:
            pixel_time_seconds = self.pixel_time_seconds
        except NotImplementedError:
            warnings.warn(
                f"Pixel times are not defined for this {self.__class__.__name__}. "
                "The corresponding metadata in the output file is set to `None`."
            )
            pixel_time_seconds = None
        if self.get_image("rgb").size > 0:
            save_tiff(
                self.get_image("rgb"),
                filename,
                dtype,
                clip,
                pixel_sizes_um=self.pixelsize_um,
                pixel_time_seconds=pixel_time_seconds,
            )
        else:
            raise RuntimeError("Can't export TIFF if there are no pixels")

    @deprecated(
        reason=(
            "This method has been renamed to `export_tiff` to more accurately reflect that it is "
            "exporting to a different format."
        ),
        action="always",
        version="0.13.0",
    )
    def save_tiff(self, filename, dtype=np.float32, clip=False):
        return self.export_tiff(filename, dtype=dtype, clip=clip)

    save_tiff.__doc__ = export_tiff.__doc__

    @property
    def infowave(self):
        return self.file["Info wave"]["Info wave"][self.start : self.stop]

    @property
    def _reconstruction_shape(self):
        """The shape of the image ([optional: pixels on slow axis], pixels on fast axis). This
        property is purely used for reconstruction from photon counts."""
        raise NotImplementedError

    @property
    def pixels_per_line(self):
        return self._num_pixels[self._metadata.scan_order[0]]

    @property
    def _num_pixels(self):
        return self._pixelcount_factory(self)

    @property
    def fast_axis(self):
        return self._metadata.fast_axis

    @property
    def timestamps(self) -> np.ndarray:
        """Timestamps for each image pixel.

        The returned array has the same shape as the `{color}_image` arrays. Timestamps are defined
        at the mean of the timestamps.
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
            "This property will be removed in a future release. Use `get_image('red')` instead."
        ),
        action="always",
        version="0.12.0",
    )
    def red_image(self):
        """Returns an image representing the red channel"""
        return self.get_image("red")

    @property
    @deprecated(
        reason=(
            "This property will be removed in a future release. Use `get_image('green')` instead."
        ),
        action="always",
        version="0.12.0",
    )
    def green_image(self):
        """Returns an image representing the green channel"""
        return self.get_image("green")

    @property
    @deprecated(
        reason=(
            "This property will be removed in a future release. Use `get_image('blue')` instead."
        ),
        action="always",
        version="0.12.0",
    )
    def blue_image(self):
        """Returns an image representing the blue channel"""
        return self.get_image("blue")

    @property
    @deprecated(
        reason=(
            "This property will be removed in a future release. Use `get_image('rgb')` instead."
        ),
        action="always",
        version="0.12.0",
    )
    def rgb_image(self):
        """Returns an rgb image"""
        return self.get_image("rgb")

    def get_image(self, channel="rgb") -> np.ndarray:
        """Get image data for the full stack as an `np.ndarray`.

        Parameters
        ----------
        channel : {'red', 'green', 'blue', 'rgb'}
            The color channel of the requested data.
        """
        if channel == "rgb":
            return np.stack([self.get_image(color) for color in ("red", "green", "blue")], axis=-1)
        else:
            return self._image(channel)
