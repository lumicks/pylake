import warnings
from copy import copy
from dataclasses import dataclass

import numpy as np

from .channel import Slice
from .adjustments import colormaps, no_adjustment
from .detail.image import (
    InfowaveCode,
    round_down,
    histogram_rows,
    seek_timestamp_next_line,
    first_pixel_sample_indices,
)
from .detail.confocal import ScanAxis, ScanMetaData, ConfocalImage
from .detail.plotting import get_axes, show_image
from .detail.timeindex import to_timestamp
from .detail.utilities import method_cache
from .detail.bead_cropping import find_beads_template, find_beads_brightness


def _default_line_time_factory(self: "Kymo"):
    """Line time in seconds

    The line time is defined as the time between frames (including the dead-time between frames).
    For a single-line scan, the line time is defined as the time without dead-time.
    """
    ns_to_sec = 1e-9

    if self._has_default_factories():
        infowave = self.infowave  # Make sure we pull this out only once, the slice is not free
        start, stop = first_pixel_sample_indices(infowave.data)
        pixel_samples = stop - start + 1
        scan_time = self.pixels_per_line * pixel_samples

        beyond_first_line = infowave.data[start + scan_time :] != InfowaveCode.discard
        if not len(beyond_first_line):  # First line is empty
            return scan_time * infowave._src.dt * ns_to_sec

        dead_time = np.argmax(beyond_first_line)

        return (scan_time + dead_time) * infowave._src.dt * ns_to_sec

    elif self.timestamps.shape[1] > 1:
        return (self.timestamps[0, 1] - self.timestamps[0, 0]) * ns_to_sec
    else:
        return self.pixels_per_line * self.pixel_time_seconds


def _default_line_timestamp_ranges_factory(self: "Kymo", exclude: bool):
    """Get start and stop timestamp of each line in the kymo."""

    ts_min = self._timestamps(reduce=np.min)[0]

    if exclude:
        # Take the max value of each line to account for unfinished final line
        # and add one sample to have proper slicing
        delta_ts = int(1e9 / self.infowave.sample_rate)
        ts_max = self._timestamps(reduce=np.max).max(axis=0) + delta_ts
    else:
        line_time = ts_min[1] - ts_min[0]
        ts_max = ts_min + line_time
    return [(t1, t2) for t1, t2 in zip(ts_min, ts_max)]


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
        self._line_timestamp_ranges_factory = _default_line_timestamp_ranges_factory
        self._position_offset = position_offset
        self._contiguous = True
        self._motion_blur_constant = 0

        self._calibration = (
            calibration
            if calibration is not None
            else (
                PositionCalibration()
                if self.pixelsize_um[0] is None
                else PositionCalibration("um", self.pixelsize_um[0], r"μm")
            )
        )

    @property
    def contiguous(self):
        """Are the pixels integrated over a contiguous period of time

        If this flag is false then pixels have been integrated over disjoint sections of time. This
        can be the case when a kymograph has been downsampled over time."""
        return self._contiguous

    @property
    def motion_blur_constant(self) -> float:
        r"""Motion blur as defined by the shutter function.

        The normalized shutter function is defined as :math:`c(t)`, where :math:`c(t)` represents
        whether the shutter is open or closed. :math:`c(t)` is normalized w.r.t. area. For no
        motion blur, :math:`c(t) = \delta(t_\mathrm{exposure})`, whereas for a constantly open
        shutter it is defined as :math:`c(t) = 1 / \Delta t`.

        The motion blur constant is defined as:

        .. math::

            R = \frac{1}{\Delta t} \int_{0}^{\Delta t}S(t) \left(1 - S(t)\right)dt

        with

        .. math::

            S(t) = \int_{0}^{t} c(t') dt'

        When there is no motion blur, we obtain: R = 0, whereas a continuously open shutter over the
        exposure time results in R = 1/6. Note that when estimating both localization uncertainty
        and the diffusion constant, the motion blur factor has no effect on the estimate of the
        diffusion constant itself, but it does affect the calculated uncertainties. In the case of
        a provided localization uncertainty, it does impact the estimate of the diffusion constant.

        Raises
        ------
        NotImplementedError
            if the motion blur is not defined for this kymograph.
        """
        if self._motion_blur_constant is None:
            raise NotImplementedError("No motion blur constant was defined for this kymograph.")

        return self._motion_blur_constant

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

        # Copy the default factories
        kymo_copy._line_time_factory = self._line_time_factory
        kymo_copy._line_timestamp_ranges_factory = self._line_timestamp_ranges_factory
        kymo_copy._motion_blur_constant = self._motion_blur_constant
        return kymo_copy

    @property
    def _id(self):
        return id(self)

    def _check_is_sliceable(self):
        if self._file is None:
            raise NotImplementedError(
                "Slicing is not implemented for kymographs derived from image stacks."
            )
        if not self._has_default_factories():
            raise NotImplementedError(
                "Slicing is not implemented for processed kymographs. Please slice prior to "
                "processing the data."
            )

    def __getitem__(self, item):
        """All indexing is in timestamp units (ns)"""
        if not isinstance(item, slice):
            raise IndexError("Scalar indexing is not supported, only slicing")
        if item.step is not None:
            raise IndexError("Slice steps are not supported")
        self._check_is_sliceable()

        start = self.start if item.start is None else item.start
        stop = self.stop if item.stop is None else item.stop
        start, stop = (to_timestamp(v, self.start, self.stop) for v in (start, stop))

        # Find the index of the first line where `start` (or `stop`) <= the start timestamp of the
        # line. If there is no such line, the result will contain the number of lines/timestamps.
        line_timestamp_ranges = np.array(self.line_timestamp_ranges(include_dead_time=False))
        line_timestamp_starts = line_timestamp_ranges[:, 0]
        i_min = np.searchsorted(line_timestamp_starts, start, side="left")
        i_max = np.searchsorted(line_timestamp_starts, stop, side="left")

        if i_min == len(line_timestamp_starts):
            return EmptyKymo(
                self.name,
                self.file,
                line_timestamp_starts[-1],
                line_timestamp_starts[-1],
                self._metadata,
            )

        if i_min >= i_max:
            return EmptyKymo(
                self.name,
                self.file,
                line_timestamp_starts[i_min],
                line_timestamp_starts[i_min],
                self._metadata,
            )

        if i_max < len(line_timestamp_starts):
            stop = line_timestamp_starts[i_max]
        else:
            # Set `stop` to at least the stop timestamp of the very last line
            stop = max(stop, line_timestamp_ranges[-1, 1])

        start = line_timestamp_starts[i_min]

        sliced_kymo = copy(self)
        sliced_kymo.start = start
        sliced_kymo.stop = stop

        return sliced_kymo

    @property
    @method_cache("pixel_time_seconds")
    def pixel_time_seconds(self):
        """Pixel dwell time in seconds"""
        if self._has_default_factories():
            infowave = self.infowave  # Make sure we pull this out only once
            start, stop = first_pixel_sample_indices(infowave.data)
            return (stop - start + 1) * infowave._src.dt * 1e-9
        else:
            return (self.timestamps[1, 0] - self.timestamps[0, 0]) / 1e9

    def line_timestamp_ranges(self, *, include_dead_time=False):
        """Get start and stop timestamp of each line in the kymo.

        Note: The stop timestamp for each line is defined as the first sample past the end of the
        relevant data such that the timestamps can be used for slicing directly.

        Parameters
        ----------
        include_dead_time : bool
            Include dead time at the end of each frame (default: False).
        """
        return self._line_timestamp_ranges_factory(self, not include_dead_time)

    def _tiff_image_metadata(self) -> dict:
        """Create metadata for the ImageDescription field of TIFFs used by `export_tiff()`."""
        metadata = super()._tiff_image_metadata()
        metadata["Line time (s)"] = self.line_time_seconds
        # Cast numpy.int64 into Python int to make it compatible to json
        metadata["Start pixel timestamp (ns)"] = int(self.line_timestamp_ranges()[0][0])
        metadata["Stop pixel timestamp (ns)"] = int(self.line_timestamp_ranges()[-1][1])
        return metadata

    def _tiff_timestamp_ranges(self, include_dead_time) -> list:
        """Create Timestamp ranges for the DateTime field of TIFFs used by `export_tiff`."""
        # As `Kymo` has only one frame, return a list with one timestamp range
        ts_ranges = np.array(self.line_timestamp_ranges(include_dead_time=include_dead_time))
        return [(np.min(ts_ranges), np.max(ts_ranges))]

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
    def _reconstruction_shape(self):
        """Shape used when reconstructing the image from raw photon counts (ordered by axis scan
        speed slow to fast)."""
        return (self.pixels_per_line,)

    @property
    def shape(self):
        """Shape of the reconstructed :class:`~lumicks.pylake.kymo.Kymo` image"""
        # For a kymo the only way to find the shape is to perform the reconstruction. While one
        # could traverse the info-wave for this purpose, we may as well just reconstruct the image
        # since that still has the potential to be useful in the future (since it is cached).
        for color in ("red", "green", "blue"):
            img = self.get_image(color)
            shape = (*img.shape, 3)
            if img.size:  # Early out if we got one
                return shape
        else:
            return shape

    @property
    @method_cache("line_time_seconds")
    def line_time_seconds(self):
        """Line time in seconds

        Raises
        ------
        RuntimeError
            If line time is not defined because the kymograph only has a single line.
        """
        return self._line_time_factory(self)

    @property
    def duration(self):
        """Duration of the kymograph in seconds. This value is equivalent to the number of scan
        lines times the line time in seconds. It does not take into account incomplete scan
        lines or mirror fly-in/out time.
        """
        return self.line_time_seconds * self.shape[1]

    @property
    def pixelsize(self):
        """Returns a `List` of axes dimensions in calibrated units. The length of the
        list corresponds to the number of scan axes."""
        return [self._calibration.value]

    def plot(
        self,
        channel="rgb",
        *,
        adjustment=no_adjustment,
        axes=None,
        image_handle=None,
        show_title=True,
        show_axes=True,
        scale_bar=None,
        **kwargs,
    ):
        """Plot a kymo for the requested color channel(s).

        Parameters
        ----------
        channel : {"red", "green", "blue", "rgb"}, optional
            Color channel to plot.
        adjustment : lk.ColorAdjustment
            Color adjustments to apply to the output image.
        axes : matplotlib.axes.Axes, optional
            If supplied, the axes instance in which to plot.
        image_handle : matplotlib.image.AxesImage or None
            Optional image handle which is used to update plots with new data rather than
            reconstruct them (better for performance).
        show_title : bool, optional
            Controls display of auto-generated plot title
        show_axes : bool, optional
            Setting show_axes to False hides the axes.
        scale_bar : lk.ScaleBar, optional
            Scale bar to add to the figure.
        **kwargs
            Forwarded to :func:`matplotlib.pyplot.imshow`. These arguments are ignored if
            `image_handle` is provided.

        Returns
        -------
        matplotlib.image.AxesImage
            The image handle representing the plotted image.
        """
        axes = get_axes(axes=axes, image_handle=image_handle)

        if show_axes is False:
            axes.set_axis_off()

        if scale_bar and not image_handle:
            scale_bar._attach_scale_bar(axes, 60.0, 1.0, "s", self._calibration.unit_label)

        image = self._get_plot_data(channel, adjustment)

        size_calibrated = self._calibration.value * self._num_pixels[0]

        default_kwargs = dict(
            # With origin set to upper (default) bounds should be given as (0, n, n, 0)
            # pixel center aligned with mean time per line
            extent=[
                -0.5 * self.line_time_seconds,
                self.duration - 0.5 * self.line_time_seconds,
                size_calibrated - 0.5 * self.pixelsize[0],
                -0.5 * self.pixelsize[0],
            ],
            aspect=(image.shape[0] / image.shape[1]) * (self.duration / size_calibrated),
            cmap=colormaps._get_default_colormap(channel),
        )

        image_handle = show_image(
            image,
            adjustment,
            channel,
            image_handle=image_handle,
            axes=axes,
            **{**default_kwargs, **kwargs},
        )
        axes.set_xlabel("time (s)")
        axes.set_ylabel(f"position ({self._calibration.unit_label})")
        if show_title:
            axes.set_title(self.name)

        return image_handle

    def plot_with_channels(
        self,
        channels,
        color_channel="rgb",
        *,
        aspect_ratio=0.25,
        kymo_args=None,
        adjustment=no_adjustment,
        title_vertical=False,
        labels=None,
        colors=None,
        scale_bar=None,
        **kwargs,
    ):
        """Plot kymo with channel data.

        Parameters
        ----------
        channels : Slice | List[Slice]
            data slice or list of slices
        color_channel : str
            color channel of kymo to plot ('red', 'green', 'blue', 'rgb')
        aspect_ratio: float
            aspect ratio of the axes (i.e. ratio of y-unit to x-unit)
        kymo_args : Optional[dict]
            Forwarded to :func:`matplotlib.pyplot.imshow()`
        adjustment : lk.ColorAdjustment
            Color adjustments to apply to the output image.
        title_vertical : bool
            Place channel title on vertical axis
        labels : str | List[str]
            Custom labels to plot with the channels
        colors : List[matplotlib.color]
            Forwarded to color argument of :func:`matplotlib.pyplot.plot()`.
        scale_bar : lk.ScaleBar, optional
            Scale bar to add to the kymograph.
        **kwargs
            Forwarded to :meth:`Slice.plot() <lumicks.pylake.channel.Slice.plot()>`.

        Examples
        --------
        ::

            import lumicks.pylake as lk
            import matplotlib.pyplot as plt
            import numpy as np

            h5_file = lk.File("example.h5")
            _, kymo = h5_file.kymos.popitem()

            kymo.plot_with_channels(
                [
                    h5_file.force1x.downsampled_by(100),
                    h5_file["Photon count"]["Green"].downsampled_over(kymo.line_timestamp_ranges(), reduce=np.sum),
                ],
                "rgb",
                adjustment=lk.ColorAdjustment(5, 98, "percentile"),
                aspect_ratio=0.2,
                title_vertical=True,
            )
        """

        def set_aspect_ratio(axis, ar):
            """This function forces a specific aspect ratio, can be useful when aligning figures"""
            axis.set_aspect(ar * np.abs(np.diff(axis.get_xlim())[0] / np.diff(axis.get_ylim()))[0])

        def check_length(items, item_type):
            if len(items) != len(channels):
                raise ValueError(
                    f"When a list of {item_type} is provided, it needs to have the same length as "
                    f"the number of channels provided. Expected {len(channels)}, got: "
                    f"{len(items)}."
                )

        import matplotlib.pyplot as plt
        from matplotlib.colors import is_color_like

        channels = [channels] if isinstance(channels, Slice) else channels
        for channel in channels:
            if not isinstance(channel, Slice):
                raise ValueError(
                    "channel is not a Slice or list of Slice objects. "
                    f"Got {type(channel).__name__} instead."
                )

        if labels:
            labels = [labels] if isinstance(labels, str) else labels
            check_length(labels, "labels")

        if colors:
            colors = [colors] if is_color_like(colors) else colors
            check_length(colors, "colors")

        _, axes = plt.subplots(len(channels) + 1, 1, sharex="all")

        # plot kymo
        scale_bar = {"scale_bar": scale_bar} if scale_bar is not None else {}
        kymo_args = ({} if kymo_args is None else kymo_args) | scale_bar
        self.plot(channel=color_channel, axes=axes[0], adjustment=adjustment, **kymo_args)

        # Storing these since plotting the data channels will change the limits
        xlim_kymo = axes[0].get_xlim()

        # plot data channels
        for idx, (ax, channel) in enumerate(zip(axes[1:], channels)):
            plt.sca(ax)
            plot_args = kwargs if not colors else kwargs | {"color": colors[idx]}
            # Cropping the channel to the kymograph time range leads to better axis limits
            channel[self.start : self.stop + 1].plot(**plot_args)
            ax.set_xlim(xlim_kymo)

            if title_vertical:
                ax.set_title(None)
                y_label = channel.labels.get("y", "")

                # Labelling with y is unnecessary.
                y_label = f"\n{y_label}" if y_label != "y" else ""

                ax.set_ylabel(f"{channel.labels.get('title', '').split('/')[-1]}{y_label}")

            if labels:
                ax.set_ylabel(labels[idx])

        for ax in axes:
            set_aspect_ratio(ax, aspect_ratio)

        for ax in axes[:-1]:
            ax.set_xlabel(None)

    def plot_with_force(
        self,
        force_channel,
        color_channel,
        aspect_ratio=0.25,
        reduce=np.mean,
        kymo_args=None,
        adjustment=no_adjustment,
        **kwargs,
    ):
        """Plot kymo with force channel downsampled over scan lines

        Note that high frequency channel data must be available for this function to work.

        Parameters
        ----------
        force_channel : str
            name of force channel to downsample and plot (e.g. '1x')
        color_channel : str
            color channel of kymo to plot ('red', 'green', 'blue', 'rgb')
        aspect_ratio : float
            aspect ratio of the axes (i.e. ratio of y-unit to x-unit)
        reduce : callable
            The :mod:`numpy` function which is going to reduce multiple samples into one. Forwarded
            to :meth:`Slice.downsampled_over() <lumicks.pylake.channel.Slice.downsampled_over()>`
        kymo_args : Optional[dict]
            Forwarded to :func:`matplotlib.pyplot.imshow()`
        adjustment : lk.ColorAdjustment
            Color adjustments to apply to the output image.
        **kwargs
            Forwarded to :meth:`Slice.plot() <lumicks.pylake.channel.Slice.plot()>`.
        """
        try:
            channel = getattr(self.file, f"force{force_channel}")
        except ValueError:
            raise ValueError(
                f"There is no force data associated with this {self.__class__.__name__}"
            )
        if not channel:
            channel = getattr(self.file, f"downsampled_force{force_channel}")
            if not channel:
                raise RuntimeError(
                    f"Desired force channel {force_channel} not available in h5 file"
                )

            warnings.warn(
                RuntimeWarning("Using downsampled force since high frequency force is unavailable.")
            )

        self.plot_with_channels(
            channel.downsampled_over(
                self.line_timestamp_ranges(include_dead_time=False), reduce=reduce, where="center"
            ),
            color_channel,
            aspect_ratio=aspect_ratio,
            kymo_args=kymo_args,
            adjustment=adjustment,
            **kwargs,
        )

    def plot_with_position_histogram(
        self,
        color_channel,
        pixels_per_bin=1,
        hist_ratio=0.25,
        adjustment=no_adjustment,
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
        edges = edges - pixel_width / 2  # pixel centers are defined at the center of a pixel

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
        adjustment=no_adjustment,
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

    def estimate_bead_edges(
        self,
        bead_diameter,
        algorithm,
        *,
        channel="green",
        extra_cropping=0,
        plot=False,
        threshold_percentile=70,
        allow_movement=False,
        downsample_num_frames=5,
    ):
        """Determine approximate positional coordinates of the bead edges. Only intended for
        stationary beads.

        .. warning::

            This is early access alpha functionality. While usable, this has not yet been tested in
            a large number of different scenarios. The API can still be subject to change without
            any prior deprecation notice! If you use this functionality keep a close eye on the
            changelog for any changes that may affect your analysis.

        There are two algorithms to determine bead edges:

            - "brightness": brightness based bead edge detection
            - "template": template correlation-based bead edge detection

        * *Brightness: brightness based bead edge detection.*
          Searches for bead edges by summing along the temporal direction and removing small
          features. The result is smoothed and peaks are detected. Bead edges are found by checking
          where the fluorescence drops below a threshold.

        * *Template: template correlation-based bead edge detection.*
          Downsamples the kymograph to a specific number of frames (specified with the optional
          parameter `downsample_num_frames`). For each frame, we traverse the scan line pixel by
          pixel, extracting a template at each pixel. This template is then correlated with the
          next frame. This process is repeated until all frames have been processed and a 2D matrix
          is acquired. This matrix is then summed over the temporal axis obtaining a similarity
          score over time. This score can be interpreted as a measure for how long a template
          taken at that particular location is present. The outer maxima of this curve provide
          the bead centers.

        Parameters
        ----------
        bead_diameter : float
            Rough estimate for the bead size (microns).
        algorithm : 'brightness', 'template'
            Which algorithm to use.
        channel : 'red', 'green', 'blue', optional
            Channel to use for bead detection.
        extra_cropping : float, optional
            How much to move the returned edge inwards from the detected bead edge in microns.
        plot : optional[bool]
            Plot result
        threshold_percentile : optional[int]
            Percentile to drop down to before accepting that we have left the bead area. Higher
            values will make the bounds go closer to the bead edge but risk the algorithm failing
            or having high background (default: 70).
        allow_movement : optional[bool]
            Allow movement of the template between frames. Only relevant for `algorithm="template"`.
            When this is enabled, the maximum correlation along the spatial axis is selected
            between frames. This allows the template to move which can lead to better detection
            for cases where the beads may be moving slightly (default: False).
        downsample_num_frames : optional[int]
            Number of time frames to downsample to (must be larger than 3). Only relevant for
            `algorithm="template"`. Default: 5.

        Returns
        -------
        bead_edges : list[float]
            List of the two edge positions in microns.

        Raises
        ------
        RuntimeError
            When the algorithm fails to locate two edges during the bead finding stage.
        """
        if self._calibration.unit != "um":
            raise RuntimeError(
                f"This kymograph is not calibrated in um but in {self._calibration.unit}. "
                f"Please make sure the kymograph is calibrated to microns before using this "
                f"functionality."
            )

        if algorithm not in ("brightness", "template"):
            raise ValueError(
                f'Unrecognized algorithm {algorithm} selected. Choose "brightness" or "template"'
            )

        shared_parameters = {
            "kymograph_image": self.get_image(channel),
            "bead_diameter_pixels": bead_diameter / self.pixelsize_um[0],
            "plot": plot,
            "threshold_percentile": threshold_percentile,
        }

        edges_pixels = (
            find_beads_brightness(**shared_parameters)
            if algorithm == "brightness"
            else find_beads_template(
                **shared_parameters,
                allow_movement=allow_movement,
                downsample_num_frames=downsample_num_frames,
            )
        )

        bead_edges = np.array(edges_pixels) * self.pixelsize_um[0] + np.array(
            [extra_cropping, -extra_cropping]
        )

        if np.diff(bead_edges) <= self.pixelsize_um[0]:
            raise RuntimeError(
                f"Detected bead edges in combination with chosen extra cropping "
                f"({extra_cropping:.2f}) lead to empty kymograph. Cropping region: "
                f"{bead_edges[0]:.2f}, {bead_edges[1]:.2f}."
            )

        return bead_edges

    def crop_beads(self, bead_diameter, algorithm, *, channel="green", extra_cropping=0, **kwargs):
        """Estimates the edges of stationary beads and returns a copy of the kymograph cropped to
        these limits.

        .. warning::

            This is early access alpha functionality. While usable, this has not yet been tested in
            a large number of different scenarios. The API can still be subject to change without
            any prior deprecation notice! If you use this functionality keep a close eye on the
            changelog for any changes that may affect your analysis.

        Parameters
        ----------
        bead_diameter : float
            Rough estimate for the bead size (microns).
        algorithm : 'brightness', 'template'
            Which algorithm to use. See :meth:`Kymo.estimate_bead_edges()` for more information.
        channel : 'red', 'green', 'blue', optional
            Channel to use for bead detection.
        extra_cropping : float, optional
            How much to move the returned edge inwards from the detected bead edge in microns.
        **kwargs
            Forwarded to :meth:`Kymo.estimate_bead_edges()`.
        """
        to_current_units = self.pixelsize[0] / self.pixelsize_um[0]
        crop_locations = np.asarray(
            self.estimate_bead_edges(
                bead_diameter,
                algorithm=algorithm,
                channel=channel,
                extra_cropping=extra_cropping,
                **kwargs,
            )
        )
        return self.crop_by_distance(*(crop_locations * to_current_units))

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
            return self._timestamps(reduce)[lower_pixels:upper_pixels, :]

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
            The :mod:`numpy` function which is going to reduce multiple pixels into one. The default
            is :func:`np.sum <numpy.sum()>`.
        """
        result = copy(self)

        def image_factory(_, channel):
            from skimage.measure import block_reduce

            data = self._image(channel)

            return block_reduce(data, (position_factor, time_factor), func=reduce)[
                : data.shape[0] // position_factor, : data.shape[1] // time_factor
            ]

        def ill_defined(thing):
            raise NotImplementedError(
                f"{thing} are no longer available after downsampling a kymograph in "
                "time since they are not well defined (the downsampling occurs over a "
                "non-contiguous time window). Line timestamps are still available, however. See: "
                "`Kymo.line_time_seconds`."
            )

        def timestamp_factory_ill_defined(_, reduce_timestamps=np.mean):
            ill_defined("Per-pixel timestamps")

        def line_timestamp_ranges_factory_ill_defined(_, exclude: bool):
            ill_defined("Line timestamp ranges")

        def timestamp_factory(_, reduce_timestamps):
            ts = self._timestamps(reduce_timestamps)
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
        if time_factor == 1:
            result._timestamp_factory = timestamp_factory
            result._line_timestamp_ranges_factory = self._line_timestamp_ranges_factory
        else:
            result._timestamp_factory = timestamp_factory_ill_defined
            result._line_timestamp_ranges_factory = line_timestamp_ranges_factory_ill_defined

        result._image_factory = image_factory
        result._line_time_factory = line_time_factory
        result._pixelsize_factory = pixelsize_factory
        result._pixelcount_factory = pixelcount_factory
        result._calibration = self._calibration.downsample(position_factor)
        result._contiguous = time_factor == 1 and self.contiguous
        result._motion_blur_constant = None
        return result

    def flip(self):
        """Flip the kymograph along the positional axis"""
        result = copy(self)

        def image_factory(_, channel):
            return np.flip(self._image(channel), axis=0)

        result._image_factory = image_factory

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
        result._calibration = PositionCalibration("kbp", length_kbp / self._num_pixels[0], "kbp")
        result._image_factory = self._image_factory
        result._timestamp_factory = self._timestamp_factory
        result._line_time_factory = self._line_time_factory
        result._pixelsize_factory = self._pixelsize_factory
        result._pixelcount_factory = self._pixelcount_factory

        return result

    def crop_and_calibrate(self, channel="rgb", tether_length_kbp=None, **kwargs):
        """Open a widget to interactively edit the image stack.

        Actions
        -------
        left-click and drag
            Define the cropped ROI.

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
            Forwarded to :meth:`Kymo.plot()`.

        Examples
        --------
        ::

            import lumicks.pylake as lk
            import matplotlib.pyplot as plt

            # Loading a kymograph.
            h5_file = lk.File("example.h5")
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
    def plot(self, channel="rgb", **kwargs):
        raise RuntimeError("Cannot plot empty kymograph")

    def _image(self, channel):
        shape = (self.pixels_per_line, 0, 3) if channel == "rgb" else (self.pixels_per_line, 0)
        return np.empty(shape)

    def get_image(self, channel="rgb"):
        return self._image(channel)

    def _has_default_factories(self):
        return False

    @property
    def duration(self):
        return 0

    def __bool__(self):
        return False


@dataclass(frozen=True)
class PositionCalibration:
    unit: str = "pixel"
    value: float = 1.0
    unit_label: str = "pixels"

    def downsample(self, factor):
        return (
            self
            if self.unit == "pixel"
            else PositionCalibration(self.unit, self.value * factor, self.unit_label)
        )


def _kymo_from_array(
    image,
    color_format,
    line_time_seconds,
    exposure_time_seconds=None,
    start=0,
    pixel_size_um=None,
    name="",
):
    """Generate a :class:`~lumicks.pylake.kymo.Kymo` instance from an image array.

    Parameters
    ----------
    image : np.ndarray
        Image data.
    color_format : str
        String indicating the order of the color channels in the image; combination of
        'r', 'g', 'b'. For example 'r' for red channel only, 'bg' for blue, green data.
    line_time_seconds : float
        Line time in seconds.
    exposure_time_seconds : float
        Line exposure time in seconds. If `None`, the exposure time is set equal to
        the line time.
    start : int
        Start timestamp of the kymo.
    pixel_size_um : float
        Pixel spatial size in microns. If `None`, the kymo will be calibrated in pixel units.
    name : string
        Kymo name.
    """

    image = np.atleast_3d(image)
    n_pixels, n_lines = image.shape[:2]

    line_delta = np.int64(line_time_seconds * 1e9)
    exposure_delta = (
        line_delta if exposure_time_seconds is None else np.int64(exposure_time_seconds * 1e9)
    )

    metadata = ScanMetaData(
        scan_axes=[ScanAxis(axis=0, num_pixels=image.shape[0], pixel_size_um=pixel_size_um)],
        center_point_um={"x": None, "y": None, "z": None},
        num_frames=0,
    )

    if not all([char in ["r", "g", "b"] for char in list(color_format)]):
        raise ValueError(
            f"Invalid color format '{color_format}'. Only 'r', 'g', and 'b' are valid components."
        )
    if len(color_format) != image.shape[-1]:
        sizes = (len(color_format), image.shape[-1])
        raise ValueError(
            f"Color format '{color_format}' specifies {sizes[0]} "
            f"channel{'s' if sizes[0] > 1 else ''} for a {sizes[1]} channel image."
        )
    rgb_image = np.zeros([n_pixels, n_lines, 3])
    for j, char in enumerate(color_format):
        channel = "rgb".find(char)
        rgb_image[:, :, channel] = image[:, :, j]

    kymo = Kymo(
        name,
        file=None,
        start=np.int64(start),
        stop=start + (n_lines * line_delta),
        metadata=metadata,
        position_offset=0,
    )

    def image_factory(_, channel):
        if channel == "rgb":
            return rgb_image
        else:
            index = ("red", "green", "blue").index(channel)
            return rgb_image[:, :, index]

    def timestamp_factory_ill_defined(_, reduce_timestamps=np.mean):
        raise NotImplementedError(
            "Per-pixel timestamps are not implemented. Line timestamps are "
            "still available, however. See: `Kymo.line_time_seconds`."
        )

    def line_timestamp_ranges_factory(_, exclude=bool):
        starts = kymo.start + np.arange(n_lines, dtype=np.int64) * line_delta
        stops = starts + (exposure_delta if exclude else line_delta)
        return [(start, stop) for start, stop in zip(starts, stops)]

    kymo._image_factory = image_factory
    kymo._timestamp_factory = timestamp_factory_ill_defined
    kymo._line_time_factory = lambda _: line_time_seconds
    kymo._line_timestamp_ranges_factory = line_timestamp_ranges_factory

    # For a Kymograph generated from an array, it is not possible to know what kind of motion blur
    # to apply be default. Setting the motion blur constant to zero explicitly disallows
    # calculations where motion blur has to be considered.
    kymo._motion_blur_constant = None

    return kymo


def _kymo_from_image_stack(
    stack, adjacent_lines=0, pixel_size_um=None, name="", reduce=np.mean
) -> Kymo:
    """Generate a :class:`~lumicks.pylake.kymo.Kymo` instance from an image stack.

    Parameters
    ----------
    stack : ImageStack
        An instance of a ImageStack. The frame rate and the exposure time of the images need
        to be constant and `corrstack` needs to have a tether. The data for the kymograph will be
        taken along the line of the tether, including the pixels of the ends of the tether.
    adjacent_lines : int
        Number of adjacent lines to the line of the tether on both sides in pixels. The data for the
        kymograph will be calculated from the pixel values reduced to a one pixel line given by
        :func:`reduce`.
    pixel_size_um : float
        Pixel spatial size in microns. If `None`, the kymo will be calibrated in pixel units.
    name : str
        Kymo name.
    reduce : callable
        The function which is going to reduce multiple pixels into one. The default is
        :func:`np.mean <numpy.mean()>, but :func:`np.max <numpy.max()>` could also be appropriate
        for some cases.
    """
    # Ensure constant frame rate of the whole stack
    ts_ranges = np.array(stack.frame_timestamp_ranges())
    line_times = np.diff(ts_ranges[:, 0])
    line_time = line_times[0]
    if not np.all(line_times == line_time):
        raise ValueError("The frame rate of the images of the image stack is not constant.")
    line_time_s = line_time * 1e-9

    # Ensure constant exposure time of the whole stack
    exp_times = ts_ranges[:, 1] - ts_ranges[:, 0]
    exp_time = exp_times[0]
    if not np.all(exp_times == exp_time):
        raise ValueError("The exposure time of the images of the image stack is not constant.")
    exp_time_s = exp_time * 1e-9

    # Start timestamp of the kymo is start timestamp of first image
    start = ts_ranges[0, 0]

    # Ensure image stack has proper tether
    if not stack._src._tether:
        raise ValueError("The image stack does not have a tether.")
    (x1, y1), (x2, y2) = stack._src._tether.ends
    if np.floor(y1) != np.floor(y2):
        raise ValueError("The image stack is not aligned along the tether axis.")

    # Extract the kymograph data along the line of the tether
    if adjacent_lines < 0:
        raise ValueError("The requested number of `adjacent_lines` must not be negative.")
    xmin = int(np.floor(x1))
    xmax = int(np.floor(x2)) + 1
    ymin = int(np.floor(y1)) - adjacent_lines
    ymax = int(np.floor(y2)) + adjacent_lines + 1
    if ymin < 0 or ymax > stack.shape[1]:
        raise ValueError(
            "The number of `adjacent_lines` exceed the size of the image stack images."
        )
    kymostack = stack.crop_by_pixels(xmin, xmax, ymin, ymax)
    image = kymostack.get_image(channel="rgb")  # time, (y,) x, (c)
    if adjacent_lines > 0:
        image = reduce(image, axis=1)  # time, x, (c)
    image = np.swapaxes(image, 0, 1)  # x, time, (c)

    # If image stack has only one channel, triplicate data to create "rgb" kymo
    if image.ndim == 2:
        image = np.repeat(image, 3, axis=1).reshape(*image.shape, 3)  # x, time, c

    return _kymo_from_array(
        image=image,
        color_format="rgb",
        line_time_seconds=line_time_s,
        exposure_time_seconds=exp_time_s,
        start=start,
        pixel_size_um=pixel_size_um,
        name=name,
    )
