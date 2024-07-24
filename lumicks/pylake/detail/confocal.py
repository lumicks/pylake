import json
import warnings
from typing import List
from dataclasses import dataclass

import numpy as np
from numpy import typing as npt

from .image import reconstruct_image, reconstruct_image_sum
from .mixin import PhotonCounts, ExcitationLaserPower
from .plotting import parse_color_channel
from .utilities import method_cache, could_sum_overflow
from ..adjustments import no_adjustment
from .imaging_mixins import TiffExport


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


def _get_confocal_data(self: "ConfocalImage", color):
    channel = getattr(self, f"{color}_photon_count")
    infowave = self.infowave

    # Early out for an empty channel
    if not channel:
        return channel, infowave

    if infowave.stop and channel.stop and len(infowave.data) != len(channel.data):
        if channel.stop < infowave.stop:
            warnings.warn(
                RuntimeWarning(
                    f"Warning: {self.__class__.__name__} is truncated. Photon count data ends "
                    f"{(infowave.stop - channel.stop) / 1e9:.2g} seconds before the end of the "
                    "info wave (which encodes how the data should be read)."
                )
            )

        return (
            channel[: min(infowave.stop, channel.stop)],
            infowave[: min(infowave.stop, channel.stop)],
        )
    else:
        return channel, infowave


def _default_image_factory(self: "ConfocalImage", color):
    channel, infowave = _get_confocal_data(self, color)
    raw_image = reconstruct_image_sum(
        channel.data.astype(float) if channel else np.zeros(infowave.data.size),
        infowave.data,
        self._reconstruction_shape,
    )
    return self._to_spatial(raw_image)


def _default_timestamp_factory(self: "ConfocalImage", reduce=timestamp_mean):
    # Uses the timestamps from the first non-zero-sized photon channel
    for color in ("red", "green", "blue"):
        channel_data, infowave = _get_confocal_data(self, color)
        channel_data = channel_data.timestamps
        if len(channel_data) != 0:
            break
    else:
        raise RuntimeError("Can't get pixel timestamps if there are no pixels")

    raw_image = reconstruct_image(
        channel_data, infowave.data, self._reconstruction_shape, reduce=reduce
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

    def _get_plot_data(self, channel):
        """Get data for plotting requested channel."""
        raise NotImplementedError

    @property
    def center_point_um(self):
        """Returns a dictionary of the x/y/z center coordinates of the scan (w.r.t. brightfield
        field of view)"""
        return self._metadata.center_point_um


class ConfocalImage(BaseScan, TiffExport):
    def _to_spatial(self, data):
        """Implements any necessary post-processing actions after image reconstruction from infowave"""
        raise NotImplementedError

    @method_cache("_image")
    def _image(self, channel) -> np.ndarray:
        """Returns a (read-only) reconstructed image.

        Parameters
        ----------
        channel : str
            Channel to return. Must be "red", "green" or "blue".
        """
        if channel not in ("red", "green", "blue"):
            raise ValueError(f'Channel must be "red", "green" or "blue", got "{channel}".')

        image = self._image_factory(self, channel)
        image.flags.writeable = False

        return image

    @method_cache("_timestamps")
    def _timestamps(self, reduce=timestamp_mean) -> np.ndarray:
        """Returns (read-only) reconstructed timestamps.

        Parameters
        ----------
        reduce : callable
            Function to reduce sample timestamps into aggregate timestamps.
        """
        timestamps = self._timestamp_factory(self, reduce)
        timestamps.flags.writeable = False

        return timestamps

    def _get_plot_data(self, channel, adjustment=no_adjustment, frame=None):
        """Get image data for plotting requested channel.

        Parameters
        ----------
        channel : str
            Which channel to return. Options are: "red", "green", "blue" or "rgb".
        adjustment : lk.ColorAdjustment
            Color adjustments to apply to the output image if channel is set to "rgb".
        """
        channel = parse_color_channel(channel)
        image = self.get_image(channel)
        frame_image = image if frame is None else image[frame]

        if len(channel) > 1:
            frame_image = adjustment._get_data_rgb(frame_image, channel=channel)

        return frame_image

    def export_tiff(self, filename, *, dtype=np.float32, clip=False):
        """Save the RGB photon counts to a TIFF image

        Parameters
        ----------
        filename : str | os.PathLike
            The name of the TIFF file where the image will be saved.
        dtype : np.dtype
            The data type of a single color channel in the resulting image.
        clip : bool
            If enabled, the photon count data will be clipped to fit into the desired `dtype`.
            This option is disabled by default: an error will be raise if the data does not fit.
        """
        super().export_tiff(filename, dtype=dtype, clip=clip)

    def _tiff_frames(self, iterator=False) -> npt.ArrayLike:
        """Create frames of TIFFs used by `export_tiff().`"""
        image = self.get_image()
        return image if image.ndim >= 4 else np.expand_dims(image, axis=0)

    def _tiff_image_metadata(self) -> dict:
        """Create metadata stored in the ImageDescription field of TIFFs used by `export_tiff()`"""
        # Try to get the pixel time
        try:
            pixel_time_seconds = self.pixel_time_seconds
        except NotImplementedError:
            warnings.warn(
                f"Pixel times are not defined for this {self.__class__.__name__}. "
                "The corresponding metadata in the output file is set to `None`."
            )
            pixel_time_seconds = None

        # Build metadata dict
        metadata = {
            "Camera": f"Confocal{self.__class__.__name__}",
            "Scan axes": [
                {
                    "Axis": sa.axis,
                    "Label": sa.axis_label.lower(),
                    "Number of pixels": sa.num_pixels,
                    "Pixel size (um)": sa.pixel_size_um,
                }
                for sa in self._metadata.scan_axes
            ],
            "Fast axis": self.fast_axis.lower(),
            "Center point (um)": self.center_point_um,
            "Pixel time (s)": pixel_time_seconds,
        }

        return metadata

    def _tiff_writer_kwargs(self) -> dict:
        """Create keyword arguments used for `TiffWriter.write()` in `self.export_tiff()`."""
        from .. import __version__ as version

        write_kwargs = {
            "software": f"Pylake v{version}",
            "photometric": "rgb",
        }

        # Add resolution if pixelsize is available
        pixel_sizes_um = self.pixelsize_um
        pixel_size_x, pixel_size_y = (
            pixel_sizes_um[0],
            pixel_sizes_um[1] if len(pixel_sizes_um) == 2 else pixel_sizes_um[0],
        )
        if pixel_size_x:
            write_kwargs["resolution"] = (1e4 / pixel_size_x, 1e4 / pixel_size_y)
            write_kwargs["resolutionunit"] = "CENTIMETER"

        return write_kwargs

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
        """Number of pixels in each line"""
        return self._num_pixels[self._metadata.scan_order[0]]

    @property
    def _num_pixels(self):
        return self._pixelcount_factory(self)

    @property
    def fast_axis(self):
        """The axis that was scanned ("X" or "Y")"""
        return self._metadata.fast_axis

    @property
    def timestamps(self) -> np.ndarray:
        """Timestamps for each image pixel.

        The returned array has the same shape as the `{color}_image` arrays. Timestamps are defined
        at the mean of the timestamps.
        """
        return self._timestamps()

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

    def get_image(self, channel="rgb") -> np.ndarray:
        """Get image data for the full stack as an :class:`~numpy.ndarray`.

        Parameters
        ----------
        channel : {'red', 'green', 'blue', 'rgb'}
            The color channel of the requested data.
        """
        if channel in (color_shorthand := {"r": "red", "g": "green", "b": "blue"}):
            channel = color_shorthand[channel]

        if channel not in ("red", "green", "blue"):
            return np.stack([self.get_image(color) for color in ("red", "green", "blue")], axis=-1)
        else:
            # Make sure we don't return a reference to our cache
            return self._image(channel)
