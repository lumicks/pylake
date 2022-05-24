import numpy as np
import warnings
import cachetools
from dataclasses import dataclass
from copy import copy
from skimage.measure import block_reduce
from deprecated.sphinx import deprecated
from .adjustments import ColorAdjustment
from .detail.confocal import ConfocalImage, linear_colormaps, ScanMetaData
from .detail.image import (
    line_timestamps_image,
    seek_timestamp_next_line,
    histogram_rows,
    round_down,
)
from .detail.timeindex import to_timestamp


def _default_line_time_factory(self: "Kymo"):
    """Line time in seconds"""
    if self.timestamps.shape[1] > 1:
        ns_to_sec = 1e-9
        return (self.timestamps[0, 1] - self.timestamps[0, 0]) * ns_to_sec
    else:
        raise RuntimeError(
            "This kymograph consists of only a single line. It is not possible to determine the "
            "kymograph line time for a kymograph consisting only of a single line."
        )


class Kymo(ConfocalImage):
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
    metadata : ScanMetaData
        Metadata for this Kymo.
    position_offset : float
        Coordinate position offset with respect to the original raw data.
    calibration : PositionCalibration
        Class defining calibration from microns to desired position units.
    """

    def __init__(self, name, file, start, stop, metadata, position_offset=0, calibration=None):
        super().__init__(name, file, start, stop, metadata)
        self._line_time_factory = _default_line_time_factory
        self._position_offset = position_offset
        self._calibration = PositionCalibration() if calibration is None else calibration

    def _has_default_factories(self):
        return (
            super()._has_default_factories()
            and self._line_time_factory == _default_line_time_factory
        )

    def __repr__(self):
        name = self.__class__.__name__
        return f"{name}(pixels={self.pixels_per_line})"

    def __copy__(self):
        kymo_copy = super().__copy__()
        kymo_copy._position_offset = self._position_offset
        kymo_copy._calibration = self._calibration
        return kymo_copy

    def __getitem__(self, item):
        """All indexing is in timestamp units (ns)"""
        if not isinstance(item, slice):
            raise IndexError("Scalar indexing is not supported, only slicing")
        if item.step is not None:
            raise IndexError("Slice steps are not supported")
        if not self._has_default_factories():
            raise NotImplementedError(
                "Slicing is not implemented for processed kymographs. Please slice prior to "
                "processing the data."
            )

        start = self.start if item.start is None else item.start
        stop = self.stop if item.stop is None else item.stop
        start, stop = (to_timestamp(v, self.start, self.stop) for v in (start, stop))

        line_timestamps = self._line_start_timestamps()
        i_min = np.searchsorted(line_timestamps, start, side="left")
        i_max = np.searchsorted(line_timestamps, stop, side="left")

        if i_min >= len(line_timestamps):
            return EmptyKymo(
                self.name, self.file, line_timestamps[-1], line_timestamps[-1], self._metadata
            )

        if i_min >= i_max:
            return EmptyKymo(
                self.name, self.file, line_timestamps[i_min], line_timestamps[i_min], self._metadata
            )

        if i_max < len(line_timestamps):
            stop = line_timestamps[i_max]

        start = line_timestamps[i_min]

        sliced_kymo = copy(self)
        sliced_kymo.start = start
        sliced_kymo.stop = stop

        return sliced_kymo

    @property
    def pixel_time_seconds(self):
        """Pixel dwell time in seconds"""
        return (self.timestamps[1, 0] - self.timestamps[0, 0]) / 1e9

    @cachetools.cachedmethod(lambda self: self._cache)
    def _line_start_timestamps(self):
        """Compute starting timestamp of each line (first DAQ sample corresponding to that line),
        not the first pixel timestamp."""
        timestamps = self.infowave.timestamps
        line_timestamps = line_timestamps_image(
            timestamps, self.infowave.data, self.pixels_per_line
        )
        return np.append(line_timestamps, timestamps[-1])

    def _fix_incorrect_start(self):
        """Resolve error when confocal scan starts before the timeline information.
        For kymographs this is recoverable by omitting the first line."""
        self.start = seek_timestamp_next_line(self.infowave[self.start :])
        self._cache = {}
        warnings.warn(
            "Start of the kymograph was truncated. Omitting the truncated first line.",
            RuntimeWarning,
        )

    def _to_spatial(self, data):
        """Spatial data as rows, time as columns"""
        return data.T

    @property
    def _shape(self):
        return (self.pixels_per_line,)

    @property
    def line_time_seconds(self):
        """Line time in seconds"""
        return self._line_time_factory(self)

    @property
    def pixelsize(self):
        """Returns a `List` of axes dimensions in calibrated units. The length of the
        list corresponds to the number of scan axes."""
        pixelsize = self.pixelsize_um
        pixelsize[0] = self._calibration.from_um(pixelsize[0])
        return pixelsize

    def _plot(self, channel, axes, adjustment=ColorAdjustment.nothing(), **kwargs):
        """Plot a kymo for requested color channel(s).

        Parameters
        ----------
        channel : {'red', 'green', 'blue', 'rgb'}
            Color channel to plot.
        axes : mpl.axes.Axes
            The axes instance in which to plot.
        adjustment : lk.ColorAdjustment
            Color adjustments to apply to the output image.

        **kwargs
            Forwarded to :func:`matplotlib.pyplot.imshow`
        """
        image = self._get_plot_data(channel, adjustment)
        size_calibrated = self._calibration.from_um(self.size_um[0])
        duration = self.line_time_seconds * image.shape[1]

        default_kwargs = dict(
            # With origin set to upper (default) bounds should be given as (0, n, n, 0)
            # pixel center aligned with mean time per line
            extent=[
                -0.5 * self.line_time_seconds,
                duration - 0.5 * self.line_time_seconds,
                size_calibrated - 0.5 * self.pixelsize[0],
                -0.5 * self.pixelsize[0],
            ],
            aspect=(image.shape[0] / image.shape[1]) * (duration / size_calibrated),
            cmap=linear_colormaps[channel],
        )

        image_handle = axes.imshow(image, **{**default_kwargs, **kwargs})
        axes.set_xlabel("time (s)")
        axes.set_ylabel(f"position ({self._calibration.unit_label})")
        axes.set_title(self.name)
        adjustment._update_limits(image_handle, image, channel)

    def _downsample_channel(self, channel, reduce=np.mean):
        # downsample exactly over the scanline time range
        min_times = self._timestamps("timestamps", reduce=np.min)[0, :].astype(np.int64)
        max_times = (
            # + 1 since we want inclusive end
            np.max(self._timestamps("timestamps", reduce=np.max).astype(np.int64), axis=0)
            + 1
        )

        time_ranges = [(mini, maxi) for mini, maxi in zip(min_times, max_times)]
        return channel.downsampled_over(time_ranges, reduce=reduce, where="center")

    def plot_with_force(
        self,
        force_channel,
        color_channel,
        aspect_ratio=0.25,
        reduce=np.mean,
        kymo_args={},
        adjustment=ColorAdjustment.nothing(),
        **kwargs,
    ):
        """Plot kymo with force channel downsampled over scan lines

        Note that high frequency channel data must be available for this function to work.

        Parameters
        ----------
        force_channel: str
            name of force channel to downsample and plot (e.g. '1x')
        color_channel: str
            color channel of kymo to plot ('red', 'green', 'blue', 'rgb')
        aspect_ratio: float
            aspect ratio of the axes (i.e. ratio of y-unit to x-unit)
        reduce : callable
            The `numpy` function which is going to reduce multiple samples into one.
            Forwarded to :func:`Slice.downsampled_over`
        kymo_args : dict
            Forwarded to :func:`matplotlib.pyplot.imshow`
        adjustment : lk.ColorAdjustment
            Color adjustments to apply to the output image.
        **kwargs
            Forwarded to :func:`Slice.plot`.
        """

        def set_aspect_ratio(axis, ar):
            """This function forces a specific aspect ratio, can be useful when aligning figures"""
            axis.set_aspect(ar * np.abs(np.diff(axis.get_xlim())[0] / np.diff(axis.get_ylim()))[0])

        import matplotlib.pyplot as plt

        _, (ax1, ax2) = plt.subplots(2, 1, sharex="all")

        # plot kymo
        self.plot(channel=color_channel, axes=ax1, adjustment=adjustment, **kymo_args)
        ax1.set_xlabel(None)
        xlim_kymo = ax1.get_xlim()  # Stored since plotting the force channel will change the limits

        # plot force channel
        plt.sca(ax2)
        channel = getattr(self.file, f"force{force_channel}")
        if not channel:
            channel = getattr(self.file, f"downsampled_force{force_channel}")
            if not channel:
                raise RuntimeError(
                    f"Desired force channel {force_channel} not available in h5 file"
                )

            warnings.warn(
                RuntimeWarning("Using downsampled force since high frequency force is unavailable.")
            )
        force = self._downsample_channel(channel, reduce=reduce)
        force.plot(**kwargs)
        ax2.set_xlim(xlim_kymo)

        set_aspect_ratio(ax1, aspect_ratio)
        set_aspect_ratio(ax2, aspect_ratio)

    def plot_with_position_histogram(
        self,
        color_channel,
        pixels_per_bin=1,
        hist_ratio=0.25,
        adjustment=ColorAdjustment.nothing(),
        **kwargs,
    ):
        """Plot kymo with histogram along position axis

        Parameters
        ----------
        color_channel: str
            color channel of kymo to plot ('red', 'green', 'blue', 'rgb').
        pixels_per_bin: int
            number of pixels along position axis to bin together.
        hist_ratio: float
            width of the histogram with respect to the kymo image.
        adjustment : lk.ColorAdjustment
            Color adjustments to apply to the output image.
        **kwargs
            Forwarded to histogram bar plot.
        """
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec

        image = self.get_image(color_channel)
        pixel_width = self.pixelsize[0]
        edges, counts, bin_widths = histogram_rows(image, pixels_per_bin, pixel_width)

        gs = GridSpec(1, 2, width_ratios=(1, hist_ratio))
        ax_kymo = plt.subplot(gs[0])
        self.plot(channel=color_channel, axes=ax_kymo, adjustment=adjustment, aspect="auto")

        ax_hist = plt.subplot(gs[1])
        ax_hist.barh(edges, counts, bin_widths, align="edge", **kwargs)
        ax_hist.invert_yaxis()
        ax_hist.set_ylim(ax_kymo.get_ylim())
        ax_hist.set_xlabel("counts")

    def plot_with_time_histogram(
        self,
        color_channel,
        pixels_per_bin=1,
        hist_ratio=0.25,
        adjustment=ColorAdjustment.nothing(),
        **kwargs,
    ):
        """Plot kymo with histogram along time axis

        Parameters
        ----------
        color_channel: str
            color channel of kymo to plot ('red', 'green', 'blue', 'rgb').
        pixels_per_bin: int
            number of pixels along time axis to bin together.
        hist_ratio: float
            height of the histogram with respect to the kymo image.
        adjustment : lk.ColorAdjustment
            Color adjustments to apply to the output image.
        **kwargs
            Forwarded to histogram bar plot.
        """
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec

        image = self.get_image(color_channel).T
        pixel_width = self.line_time_seconds
        edges, counts, bin_widths = histogram_rows(image, pixels_per_bin, pixel_width)
        # time points are defined at center of pixel
        edges = edges - pixel_width / 2

        gs = GridSpec(2, 1, height_ratios=(hist_ratio, 1))
        ax_kymo = plt.subplot(gs[1])
        self.plot(channel=color_channel, axes=ax_kymo, aspect="auto", adjustment=adjustment)
        ax_kymo.set_title("")

        ax_hist = plt.subplot(gs[0])
        ax_hist.bar(edges, counts, bin_widths, align="edge", **kwargs)
        ax_hist.set_xlim(ax_kymo.get_xlim())
        ax_hist.set_ylabel("counts")
        ax_hist.set_title(self.name)

    def crop_by_distance(self, lower, upper):
        """Crop the kymo by position.

        Crops the kymograph down to lower <= x < upper.

        Parameters
        ----------
        lower : float
            Lower bound in physical units.
        upper : float
            Upper bound in physical units.
        """
        if lower < 0 or upper < 0:
            raise ValueError("Cropping by negative positions not allowed")

        lower_pixels = int(lower / self.pixelsize[0])
        upper_pixels = int(np.ceil(upper / self.pixelsize[0]))
        n_pixels = len(np.arange(self._num_pixels[0])[lower_pixels:upper_pixels])
        if n_pixels == 0:
            raise IndexError("Cropped image would be empty")

        result = copy(self)
        result._position_offset = self._position_offset + lower_pixels * self.pixelsize[0]

        def image_factory(_, channel):
            return self._image(channel)[lower_pixels:upper_pixels, :]

        def timestamp_factory(_, reduce):
            return self._timestamps("timestamps", reduce)[lower_pixels:upper_pixels, :]

        def pixelcount_factory(_):
            num_pixels = self._num_pixels
            num_pixels[0] = n_pixels
            return num_pixels

        result._image_factory = image_factory
        result._timestamp_factory = timestamp_factory
        result._line_time_factory = lambda _: self.line_time_seconds
        result._pixelsize_factory = lambda _: self.pixelsize_um
        result._pixelcount_factory = pixelcount_factory
        return result

    def downsampled_by(
        self,
        time_factor=1,
        position_factor=1,
        reduce=np.sum,
    ):
        """Return a copy of this Kymograph which is downsampled by `time_factor` in time and
        `position_factor` in space.

        Parameters
        ----------
        time_factor : int
            The number of pixels that will be averaged in time (default: 1).
        position_factor : int
            The number of pixels that will be averaged in space (default: 1).
        reduce : callable
            The `numpy` function which is going to reduce multiple pixels into one.
            The default is `np.sum`.
        """
        result = copy(self)

        def image_factory(_, channel):
            data = self._image(channel)
            return block_reduce(data, (position_factor, time_factor), func=reduce)[
                : data.shape[0] // position_factor, : data.shape[1] // time_factor
            ]

        def timestamp_factory_ill_defined(_, reduce_timestamps=np.mean):
            raise AttributeError(
                "Per-pixel timestamps are no longer available after downsampling a kymograph in "
                "time since they are not well defined (the downsampling occurs over a "
                "non-contiguous time window). Line timestamps are still available, however. See: "
                "`Kymo.line_time_seconds`."
            )

        def timestamp_factory(_, reduce_timestamps):
            ts = self._timestamps("timestamps", reduce_timestamps)
            full_blocks = ts[: round_down(ts.shape[0], position_factor), :]
            reshaped = full_blocks.reshape(-1, position_factor, ts.shape[1])
            return reduce_timestamps(reshaped, axis=1)

        def line_time_factory(_):
            return self.line_time_seconds * time_factor

        def pixelsize_factory(_):
            pixelsizes = self.pixelsize_um
            pixelsizes[0] = pixelsizes[0] * position_factor
            return pixelsizes

        def pixelcount_factory(_):
            num_pixels = self._num_pixels
            num_pixels[0] = num_pixels[0] // position_factor
            return num_pixels

        result._image_factory = image_factory
        result._timestamp_factory = (
            timestamp_factory if time_factor == 1 else timestamp_factory_ill_defined
        )
        result._line_time_factory = line_time_factory
        result._pixelsize_factory = pixelsize_factory
        result._pixelcount_factory = pixelcount_factory
        return result

    def calibrate_to_kbp(self, length_kbp):
        """Calibrate from microns to other units.

        Parameters
        ----------
        length : float
            length of the kymo in kilobase pairs
        """
        if self._calibration.unit == "kbp":
            raise RuntimeError("kymo is already calibrated in base pairs.")

        result = copy(self)
        result._calibration = PositionCalibration("kbp", length_kbp / self.size_um[0], "kbp")
        result._image_factory = self._image_factory
        result._timestamp_factory = self._timestamp_factory
        result._line_time_factory = self._line_time_factory
        result._pixelsize_factory = self._pixelsize_factory
        result._pixelcount_factory = self._pixelcount_factory

        return result

    def crop_and_calibrate(self, channel="rgb", tether_length_kbp=None, **kwargs):
        """Open a widget to interactively edit the image stack.

        Actions include:
            * left-click and drag to define the cropped ROI

        Parameters
        ----------
        channel : 'rgb', 'red', 'green', 'blue', None; optional
            Channel to plot for RGB images (None defaults to 'rgb')
            Not used for grayscale images
        tether_length_kbp : float
            Length of the tether in the cropped region in kilobase pairs.
            If provided, the kymo returned from the `image` property will be automatically
            calibrated to this tether length.
        **kwargs
            Forwarded to :func:`Kymo.plot()`.

        Examples
        --------
        ::

        from lumicks import pylake
        import matplotlib.pyplot as plt

        # Loading a stack.
        h5_file = pylake.File("example.h5")
        _, kymo = h5_file.kymos.popitem()
        widget = kymo.crop_and_calibrate("green", 48.502)
        plt.show()

        # Select cropping ROI by left-click drag

        # Grab the updated image stack
        new_kymo = widget.kymo
        """
        from .nb_widgets.image_editing import KymoEditorWidget

        return KymoEditorWidget(self, channel, tether_length_kbp, **kwargs)


class EmptyKymo(Kymo):
    def plot_rgb(self):
        raise RuntimeError("Cannot plot empty kymograph")

    def _plot(self, image, **kwargs):
        raise RuntimeError("Cannot plot empty kymograph")

    def _image(self):
        return np.empty((self.pixels_per_line, 0))

    def get_image(self, channel="rgb"):
        return self._image()

    @property
    @deprecated(
        reason=(
            "This property will be removed in a future release. Use `get_image('red')` instead."
        ),
        action="always",
        version="0.12.0",
    )
    def red_image(self):
        return self._image()

    @property
    @deprecated(
        reason=(
            "This property will be removed in a future release. Use `get_image('green')` instead."
        ),
        action="always",
        version="0.12.0",
    )
    def green_image(self):
        return self._image()

    @property
    @deprecated(
        reason=(
            "This property will be removed in a future release. Use `get_image('blue')` instead."
        ),
        action="always",
        version="0.12.0",
    )
    def blue_image(self):
        return self._image()


@dataclass(frozen=True)
class PositionCalibration:
    unit: str = "um"
    calibration_per_um: float = 1.0
    unit_label: str = r"$\mu$m"

    def from_um(self, um):
        return um * self.calibration_per_um
