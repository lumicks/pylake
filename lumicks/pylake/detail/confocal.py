import json
import numpy as np
import cachetools
import warnings
from deprecated.sphinx import deprecated

from .mixin import PhotonCounts
from .mixin import ExcitationLaserPower
from .image import (
    reconstruct_image_sum,
    reconstruct_image,
    reconstruct_num_frames,
    seek_timestamp_next_line,
    line_timestamps_image,
    save_tiff,
    ImageMetadata,
)


class DeprecatedBooleanProperties:
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


class ConfocalPlotting:
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


class BaseScan(PhotonCounts, DeprecatedBooleanProperties):
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
            The parent file. Used to look up channel data
        """
        start = h5py_dset.attrs["Start time (ns)"]
        stop = h5py_dset.attrs["Stop time (ns)"]
        name = h5py_dset.name.split("/")[-1]
        try:
            json_data = json.loads(h5py_dset[()])["value0"]
        except KeyError:
            raise KeyError(f"{cls.__name__} '{name}' is missing metadata and cannot be loaded")

        return cls(name, file, start, stop, json_data)

    def _get_photon_count(self, name):
        """Grab the portion of the photon count that overlaps with the scan."""
        photon_count = getattr(self.file, f"{name}_photon_count".lower())[self.start : self.stop]
        timeline_start = photon_count._src.start
        timeline_dt = photon_count._src.dt

        # Workaround for a bug in the STED delay mechanism which could result in scan start times
        # ending up within the sample time.
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

    @property
    def center_point_um(self):
        """Returns a dictionary of the x/y/z center coordinates of the scan (w.r.t. brightfield
        field of view)"""
        return self._json["scan volume"]["center point (um)"]

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


class ScannedImage(BaseScan):
    def __init__(self, name, file, start, stop, json):
        super().__init__(name, file, start, stop, json)
        self._num_frames = self._json["scan count"]
        if len(self._json["scan volume"]["scan axes"]) > 2:
            raise RuntimeError("3D scans are not supported")

    @cachetools.cachedmethod(lambda self: self._cache)
    def _line_start_timestamps(self):
        """Compute starting timestamp of each line (first DAQ sample corresponding to that line),
        not the first pixel timestamp."""
        timestamps = self.infowave.timestamps
        line_timestamps = line_timestamps_image(
            timestamps, self.infowave.data, self.pixels_per_line
        )
        return np.append(line_timestamps, timestamps[-1])

    def slice(self, start, stop):
        """All indexing is in timestamp units (ns)"""
        line_timestamps = self._line_start_timestamps()
        i_min = np.searchsorted(line_timestamps, start, side="left")
        i_max = np.searchsorted(line_timestamps, stop, side="left")

        if i_min >= len(line_timestamps) or i_min >= i_max:
            return None

        if i_max < len(line_timestamps):
            stop = line_timestamps[i_max]

        start = line_timestamps[i_min]
        return ScannedImage(self.name, self.file, start, stop, self._json)

    def _ordered_axes(self):
        """Returns axis indices in spatial order"""
        return sorted(self._json["scan volume"]["scan axes"], key=lambda x: x["axis"])

    def _to_spatial(self, data):
        """If the first axis of the reconstruction has a higher physical axis number than the second, we flip the axes.

        Checks whether the axes should be flipped w.r.t. the reconstruction. Reconstruction always produces images
        with the slow axis first, and the fast axis second. Depending on the order of axes scanned, this may not
        coincide with physical axes. The axes should always be ordered from the lowest physical axis number to higher.
        Here X, Y, Z correspond to axis number 0, 1 and 2. So for an YZ scan, we'd want Y on the X axis."""
        physical_axis = [axis["axis"] for axis in self._json["scan volume"]["scan axes"]]
        if len(physical_axis) == 1:
            return data.T

        data = data.squeeze()

        if physical_axis[0] > physical_axis[1]:
            new_axis_order = np.arange(len(data.shape), dtype=int)
            new_axis_order[-1], new_axis_order[-2] = new_axis_order[-2], new_axis_order[-1]
            return np.transpose(data, new_axis_order)
        else:
            return data

    def _fix_incorrect_start(self):
        """Resolve error when confocal scan starts before the timeline information.

        For 1D scans this is recoverable by omitting the first line."""
        if len(self._shape) > 1:
            raise RuntimeError(
                "Start of the scan was truncated. Reconstruction cannot proceed. Did you export the "
                "entire scan time in Bluelake?"
            )

        self.start = seek_timestamp_next_line(self.infowave[self.start :])
        self._cache = {}
        warnings.warn(
            "Start of the kymograph was truncated. Omitting the truncated first line.",
            RuntimeWarning,
        )

    @cachetools.cachedmethod(lambda self: self._cache)
    def image(self, channel):
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

    @property
    def timestamps(self) -> np.ndarray:
        """Timestamps for image pixels, not for samples

        The returned array has the same shape as the `*_image` arrays.
        """
        return self._timestamps("timestamps")

    @property
    def infowave(self):
        return self.file["Info wave"]["Info wave"][self.start : self.stop]

    @property
    def _fast_axis_metadata(self):
        return self._json["scan volume"]["scan axes"][0]

    @property
    def fast_axis(self):
        return "X" if self._fast_axis_metadata["axis"] == 0 else "Y"

    @property
    def pixels_per_line(self):
        return self._fast_axis_metadata["num of pixels"]

    @property
    def num_frames(self):
        if self._num_frames == 0:
            self._num_frames = reconstruct_num_frames(
                self.infowave.data, self.pixels_per_line, self.lines_per_frame
            )
        return self._num_frames

    @property
    def lines_per_frame(self):
        return self._json["scan volume"]["scan axes"][1]["num of pixels"]

    @property
    def _shape(self):
        return tuple(ax["num of pixels"] for ax in reversed(self._json["scan volume"]["scan axes"]))

    @property
    def pixelsize_um(self):
        """Returns a `List` of axes dimensions in um. The length of the list corresponds to the
        number of scan axes."""
        return [axes["pixel size (nm)"] / 1000 for axes in self._ordered_axes()]

    @property
    def size_um(self):
        """Returns a `List` of scan sizes in um along axes. The length of the list corresponds to
        the number of scan axes."""
        return [
            axes["pixel size (nm)"] * axes["num of pixels"] / 1000 for axes in self._ordered_axes()
        ]

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


class ConfocalImageAPI(DeprecatedBooleanProperties, ConfocalPlotting, ExcitationLaserPower):
    def __init__(self, image, file):
        self._src = image
        self.file = file
        self._cache = {}

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
            save_tiff(
                self.rgb_image, filename, dtype, clip, ImageMetadata.from_dataset(self._src._json)
            )
        else:
            raise RuntimeError("Can't export TIFF if there are no pixels")

    @property
    def infowave(self):
        return self._src.infowave

    @property
    def _shape(self):
        """The shape of the image ([optional: pixels on slow axis], pixels on fast axis)"""
        return self._src._shape

    @property
    def fast_axis(self):
        return self._src.fast_axis

    @property
    def pixels_per_line(self):
        return self._src.pixels_per_line

    @property
    def timestamps(self) -> np.ndarray:
        """Timestamps for image pixels, not for samples

        The returned array has the same shape as the `*_image` arrays.
        """
        return self._src.timestamps

    @property
    def pixelsize_um(self):
        """Returns a `List` of axes dimensions in um. The length of the list corresponds to the
        number of scan axes."""
        return self._src.pixelsize_um

    @property
    def size_um(self):
        """Returns a `List` of scan sizes in um along axes. The length of the list corresponds to
        the number of scan axes."""
        return self._src.size_um

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
        return self._src.scan_width_um

    @property
    def red_image(self):
        return self._src.image("red")

    @property
    def green_image(self):
        return self._src.image("green")

    @property
    def blue_image(self):
        return self._src.image("blue")

    @property
    def red_photon_count(self):
        return self._src.red_photon_count

    @property
    def green_photon_count(self):
        return self._src.green_photon_count

    @property
    def blue_photon_count(self):
        return self._src.blue_photon_count

    @property
    def rgb_image(self):
        color_channels = [getattr(self, f"{color}_image").T for color in ("red", "green", "blue")]
        return np.stack(color_channels).T

    @classmethod
    def from_dataset(cls, h5py_dset, file):
        return cls(ScannedImage.from_dataset(h5py_dset, file), file)

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
        return self._src.json

    @property
    def center_point_um(self):
        return self._src.center_point_um

    @property
    def start(self):
        return self._src.start

    @property
    def stop(self):
        return self._src.stop

    @property
    def name(self):
        return self._src.name
