import numpy as np
from copy import copy
from deprecated import deprecated

from .detail.imaging_mixins import VideoExport
from .adjustments import ColorAdjustment
from .detail.confocal import ConfocalImage, linear_colormaps
from .detail.image import reconstruct_num_frames, make_image_title


class Scan(ConfocalImage, VideoExport):
    """A confocal scan exported from Bluelake

    Parameters
    ----------
    name : str
        Scan name
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
        super().__init__(name, file, start, stop, metadata)
        if self._metadata.num_axes > 2:
            raise RuntimeError("3D scans are not supported")

    def __repr__(self):
        name = self.__class__.__name__
        return f"{name}(pixels=({self.pixels_per_line}, {self.lines_per_frame}))"

    def __getitem__(self, item):
        """All indexing is in frames"""
        ts_ranges = self.frame_timestamp_ranges(exclude=False)

        if isinstance(item, slice):
            if item.step is not None:
                raise IndexError("Slice steps are not supported")

            sliced_ts_ranges = ts_ranges[item]
            if not sliced_ts_ranges:
                raise NotImplementedError("Slice is empty.")

            start, stop = sliced_ts_ranges[0][0], sliced_ts_ranges[-1][-1]
            num_frames = len(sliced_ts_ranges)
        else:
            start, stop = ts_ranges[item]
            num_frames = 1

        new_scan = copy(self)
        new_scan.start = start
        new_scan.stop = stop
        new_scan._metadata = self._metadata.with_num_frames(num_frames)

        return new_scan

    @property
    def pixel_time_seconds(self):
        """Pixel dwell time in seconds"""
        indices = np.zeros(self.timestamps.ndim, dtype=int)

        # We want the difference along the fast axis. The first (optional) index corresponds to the
        # frame. The last two indices correspond to the imaging axes.
        indices[-2 if self._metadata.scan_order[0] > self._metadata.scan_order[1] else -1] = 1
        return (self.timestamps.item(tuple(indices)) - self.timestamps.item(0)) / 1e9

    @property
    def num_frames(self):
        if self._metadata.num_frames == 0:
            self._metadata = self._metadata.with_num_frames(
                reconstruct_num_frames(
                    self.infowave.data, self.pixels_per_line, self.lines_per_frame
                )
            )
        return self._metadata.num_frames

    def frame_timestamp_ranges(self, exclude=True):
        """Get start and stop timestamp of each frame in the scan.

        Note: The stop timestamp for each frame is defined as the first sample past the end of the
        relevant data such that the timestamps can be used for slicing directly.

        Parameters
        ----------
        exclude : bool
            Exclude dead time at the end of each frame.


        Examples
        --------
        ::

            from lumicks import pylake

            file = pylake.File("example.h5")
            scan = file.scans["my scan"]

            # Grab start and stop timestamp of the first scan.
            start, stop = scan.frame_timestamp_ranges()[0]

            # Plot the force data corresponding to the first scan.
            file.force1x[start:stop].plot()
        """
        ts_min = self._timestamps("timestamps", reduce=np.min)
        ts_max = self._timestamps("timestamps", reduce=np.max)
        delta_ts = int(1e9 / self.infowave.sample_rate)  # We want the sample beyond the end
        if ts_min.ndim == 2:
            return [(np.min(ts_min), np.max(ts_max) + delta_ts)]
        else:
            if exclude:
                maximum_timestamp = np.max(ts_max, axis=tuple(range(1, ts_max.ndim)))
                return [(t1, t2) for t1, t2 in zip(ts_min[:, 0, 0], maximum_timestamp + delta_ts)]
            else:
                frame_time = ts_min[1, 0, 0] - ts_min[0, 0, 0]
                return [(t, t + frame_time) for t in ts_min[:, 0, 0]]

    def plot_correlated(
        self,
        channel_slice,
        frame=0,
        reduce=np.mean,
        channel="rgb",
        figure_scale=0.75,
        adjustment=ColorAdjustment.nothing(),
    ):
        """Downsample channel on a frame by frame basis and plot the results. The downsampling
        function (e.g. np.mean) is evaluated for the time between a start and end time of a frame.
        Note: In environments which support interactive figures (e.g. jupyter notebook with
        ipywidgets or interactive python) this plot will be interactive.

        Parameters
        ----------
        channel_slice : pylake.channel.Slice
            Data slice that we with to downsample.
        frame : int
            Frame to show.
        reduce : callable
            The function which is going to reduce multiple samples into one. The default is
            :func:`numpy.mean`, but :func:`numpy.sum` could also be appropriate for some cases
            e.g. photon counts.
        channel : 'rgb', 'red', 'green', 'blue', None; optional
            Channel to plot for RGB images (None defaults to 'rgb')
            Not used for grayscale images
        figure_scale : float
            Scaling of the figure width and height. Values greater than one increase the size of the
            figure.
        adjustment : lk.ColorAdjustment
            Color adjustments to apply to the output image.


        Examples
        --------
        ::

            from lumicks import pylake

            file = pylake.File("example.h5")
            scan = file.scans["my scan"]
            scan.plot_correlated(file.force1x, channel="red")
        """
        from lumicks.pylake.nb_widgets.correlated_plot import plot_correlated
        from .detail.confocal import linear_colormaps

        def plot_channel(frame):
            if channel in ("red", "green", "blue", "rgb"):
                return self._get_plot_data(channel, frame=frame, adjustment=adjustment)
            else:
                raise RuntimeError("Invalid channel selected")

        def post_update(image_handle, image):
            return adjustment._update_limits(image_handle, image, channel)

        title_factory = lambda frame: make_image_title(self, frame, show_name=False)
        frame_timestamps = self.frame_timestamp_ranges()

        plot_correlated(
            channel_slice,
            frame_timestamps,
            plot_channel,
            title_factory,
            frame,
            reduce,
            colormap=linear_colormaps[channel],
            figure_scale=figure_scale,
            post_update=post_update,
        )

    @property
    def lines_per_frame(self):
        return self._num_pixels[self._metadata.scan_order[1]]

    @property
    def _shape(self):
        return (self.lines_per_frame, self.pixels_per_line)

    def _fix_incorrect_start(self):
        """Resolve error when confocal scan starts before the timeline information.
        For scans, this is currently unrecoverable."""
        raise RuntimeError(
            "Start of the scan was truncated. Reconstruction cannot proceed. Did you export the "
            "entire scan time in Bluelake?"
        )

    def _to_spatial(self, data):
        """If the first axis of the reconstruction has a higher physical axis number than the second, we flip the axes.

        Checks whether the axes should be flipped w.r.t. the reconstruction. Reconstruction always produces images
        with the slow axis first, and the fast axis second. Depending on the order of axes scanned, this may not
        coincide with physical axes. The axes should always be ordered from the lowest physical axis number to higher.
        Here X, Y, Z correspond to axis number 0, 1 and 2. So for an YZ scan, we'd want Y on the X axis."""
        data = data.squeeze()

        if self._metadata.scan_order[0] > self._metadata.scan_order[1]:
            new_axis_order = np.arange(len(data.shape), dtype=int)
            new_axis_order[-1], new_axis_order[-2] = new_axis_order[-2], new_axis_order[-1]
            return np.transpose(data, new_axis_order)
        else:
            return data

    def _plot(
        self,
        channel,
        axes,
        frame=0,
        image_handle=None,
        adjustment=ColorAdjustment.nothing(),
        **kwargs,
    ):
        """Plot a scan frame for requested color channel(s).

        Parameters
        ----------
        channel : {'red', 'green', 'blue', 'rgb'}
            Color channel to plot.
        axes : mpl.axes.Axes
            The axes instance in which to plot.
        frame : int
            Frame index.
        adjustment : lk.ColorAdjustment
            Color adjustments to apply to the output image.
        image_handle : `matplotlib.image.AxesImage` or None
            Optional image handle which is used to update plots with new data rather than
            reconstruct them (better for performance).
        **kwargs
            Forwarded to :func:`matplotlib.pyplot.imshow`
        """
        if frame < 0:
            raise IndexError("negative indexing is not supported.")

        image = self._get_plot_data(
            channel, adjustment, frame=frame if self.num_frames != 1 else None
        )

        x_um, y_um = self.size_um
        default_kwargs = dict(
            # With origin set to upper (default) bounds should be given as (0, n, n, 0)
            extent=[0, x_um, y_um, 0],
            aspect=(image.shape[0] / image.shape[1]) * (x_um / y_um),
            cmap=linear_colormaps[channel],
        )

        if not image_handle:
            image_handle = axes.imshow(image, **{**default_kwargs, **kwargs})
        else:
            # Updating the image data in an existing plot is a lot faster than re-plotting with
            # `imshow`.
            image_handle.set_data(image)

        adjustment._update_limits(image_handle, image, channel)

        scan_axes = self._metadata.ordered_axes
        axes.set_xlabel(rf"{scan_axes[0].axis_label.lower()} ($\mu$m)")
        axes.set_ylabel(rf"{scan_axes[1].axis_label.lower()} ($\mu$m)")
        axes.set_title(make_image_title(self, frame))

        return image_handle

    @deprecated(
        reason=(
            "This property will be removed in a future release. Use `export_video('rgb')` instead."
        ),
        action="always",
        version="0.13.0",
    )
    def export_video_rgb(
        self,
        file_name,
        start_frame=None,
        end_frame=None,
        fps=15,
        adjustment=ColorAdjustment.nothing(),
        **kwargs,
    ):
        """Export multi-frame scan as video.

        Parameters
        ----------
        file_name : str
            File name to export to.
        start_frame : int
            Initial frame.
        end_frame : int
            Last frame.
        fps : int
            Frames per second.
        adjustment : lk.ColorAdjustment
            Color adjustments to apply to the output image.
        **kwargs
            Forwarded to :func:`matplotlib.pyplot.imshow`.
        """
        self.export_video(
            "rgb",
            file_name,
            start_frame=start_frame,
            stop_frame=end_frame,
            fps=fps,
            adjustment=adjustment,
            **kwargs,
        )

    @deprecated(
        reason=(
            "This property will be removed in a future release. Use `export_video('red')` instead."
        ),
        action="always",
        version="0.13.0",
    )
    def export_video_red(
        self,
        file_name,
        start_frame=None,
        end_frame=None,
        fps=15,
        adjustment=ColorAdjustment.nothing(),
        **kwargs,
    ):
        """Export multi-frame scan as video.

        Parameters
        ----------
        file_name : str
            File name to export to.
        start_frame : int
            Initial frame.
        end_frame : int
            Last frame.
        fps : int
            Frames per second.
        adjustment : lk.ColorAdjustment
            Color adjustments to apply to the output image.
        **kwargs
            Forwarded to :func:`matplotlib.pyplot.imshow`.
        """
        self.export_video(
            "red",
            file_name,
            start_frame=start_frame,
            stop_frame=end_frame,
            fps=fps,
            adjustment=adjustment,
            **kwargs,
        )

    @deprecated(
        reason=(
            "This property will be removed in a future release. Use `export_video('green')` instead."
        ),
        action="always",
        version="0.13.0",
    )
    def export_video_green(
        self,
        file_name,
        start_frame=None,
        end_frame=None,
        fps=15,
        adjustment=ColorAdjustment.nothing(),
        **kwargs,
    ):
        """Export multi-frame scan as video.

        Parameters
        ----------
        file_name : str
            File name to export to.
        start_frame : int
            Initial frame.
        end_frame : int
            Last frame.
        fps : int
            Frames per second.
        adjustment : lk.ColorAdjustment
            Color adjustments to apply to the output image.
        **kwargs
            Forwarded to :func:`matplotlib.pyplot.imshow`.
        """
        self.export_video(
            "green",
            file_name,
            start_frame=start_frame,
            stop_frame=end_frame,
            fps=fps,
            adjustment=adjustment,
            **kwargs,
        )

    @deprecated(
        reason=(
            "This property will be removed in a future release. Use `export_video('blue')` instead."
        ),
        action="always",
        version="0.13.0",
    )
    def export_video_blue(
        self,
        file_name,
        start_frame=None,
        end_frame=None,
        fps=15,
        adjustment=ColorAdjustment.nothing(),
        **kwargs,
    ):
        """Export multi-frame scan as video.

        Parameters
        ----------
        file_name : str
            File name to export to.
        start_frame : int
            Initial frame.
        end_frame : int
            Last frame.
        fps : int
            Frames per second.
        adjustment : lk.ColorAdjustment
            Color adjustments to apply to the output image.
        **kwargs
            Forwarded to :func:`matplotlib.pyplot.imshow`.
        """
        self.export_video(
            "blue",
            file_name,
            start_frame=start_frame,
            stop_frame=end_frame,
            fps=fps,
            adjustment=adjustment,
            **kwargs,
        )
