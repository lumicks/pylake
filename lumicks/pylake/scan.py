from copy import copy
from itertools import zip_longest

import numpy as np

from .adjustments import colormaps, no_adjustment
from .detail.image import make_image_title, reconstruct_num_frames, first_pixel_sample_indices
from .detail.confocal import ConfocalImage
from .detail.plotting import get_axes, show_image
from .detail.utilities import method_cache
from .detail.imaging_mixins import FrameIndex, VideoExport


class Scan(ConfocalImage, VideoExport, FrameIndex):
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
        if self._metadata.num_axes == 1:
            raise RuntimeError("1D scans are not supported")
        if self._metadata.num_axes > 2:
            raise RuntimeError("3D scans are not supported")

    def crop_by_pixels(self, x_min, x_max, y_min, y_max) -> "Scan":
        """Crop the image stack by pixel values.

        Parameters
        ----------
        x_min : int
            minimum x pixel (inclusive, optional)
        x_max : int
            maximum x pixel (exclusive, optional)
        y_min : int
            minimum y pixel (inclusive, optional)
        y_max : int
            maximum y pixel (exclusive, optional)
        """
        spatial_axes = (slice(y_min, y_max), slice(x_min, x_max))
        return self._scan_with_sliced_factories(frame_axis=slice(None), spatial_axes=spatial_axes)

    def _scan_with_sliced_factories(self, frame_axis, spatial_axes) -> "Scan":
        """Return a scan with sliced factories

        Parameters
        ----------
        frame_axis : slice or int
            How to slice the frame axis
        spatial_axes : () or (slice) or (slice, slice)
            Slices with which to slice the Scan. The order should be the order in which the numpy
            array is sliced, with an optional first dimension reflecting frame slices if we are
            dealing with a multi-frame scan.
        """
        if isinstance(frame_axis, int):
            num_frames = 1
            frame_axis = frame_axis if frame_axis >= 0 else self.num_frames + frame_axis
            if not 0 <= frame_axis < self.num_frames:
                raise IndexError("Frame index out of range")
        else:
            frame_indices = range(*frame_axis.indices(self.num_frames))
            num_frames = len(frame_indices)

            if not frame_indices:
                return EmptyScan(self.name, self.file, self.start, self.start, self._metadata)
            elif num_frames == 1:
                # Convert single frame to direct index (amounts to squeezing that dimension)
                frame_axis = frame_indices[0]

        slices = (frame_axis, *spatial_axes) if self.num_frames > 1 else spatial_axes

        def pixelcount_factory(_):
            # We need to make sure we have all spatial axes as we need to correct each axis since
            # they appear in reverse order in `_num_pixels`.
            full_spatial_axes = [
                ax if ax is not None else slice(None)
                for _, ax in zip_longest(range(2), spatial_axes)
            ]

            return tuple(
                [
                    np.arange(n_pixels)[s].size
                    for n_pixels, s in zip(self._num_pixels, reversed(full_spatial_axes))
                ]
            )

        def image_factory(_, channel):
            return self._image(channel)[slices]

        def timestamp_factory(_, reduce_timestamps):
            ts = self._timestamps(reduce_timestamps)
            return ts[slices]

        result = copy(self)
        result._image_factory = image_factory
        result._timestamp_factory = timestamp_factory
        result._pixelcount_factory = pixelcount_factory

        # Force reconstruction number of frames now
        result._metadata = self._metadata.with_num_frames(num_frames)

        # The metadata has an ugly hack when the number of frames metadata is missing
        # this means that zero cannot be expressed correctly at this time since it
        # would trigger a reconstruction. For now, we raise on an empty scan.
        if result._metadata.num_frames == 0:
            raise NotImplementedError("Slice is empty.")

        # Verify that none of the axis end up empty
        resulting_shape = pixelcount_factory(None)
        if np.any(np.asarray(resulting_shape) == 0):
            raise NotImplementedError("Slice is empty.")

        return result

    def __repr__(self):
        name = self.__class__.__name__
        return f"{name}(pixels=({self.pixels_per_line}, {self.lines_per_frame}))"

    def __getitem__(self, item):
        """Returns specific scan frame(s) and/or cropped scans.

        The first item refers to the scan frames; both indexing (by integer) and slicing are
        allowed, but using steps or slicing by list is not.
        The last two items refer to the spatial dimensions in pixel rows and columns. Only full
        slices (without steps) are allowed. All values should be given in pixels.

        Examples
        --------
        ::

            import lumicks.pylake as lk

            file = lk.File("example.h5")
            scan = file.scans["my_scan"]

            scan[5]  # Gets the 6th frame of the scan (0 is the first).
            scan[1:5]  # Get scan frames 1, 2, 3 and 4.
            scan[:, 10:50, 20:50]  # Gets all frames cropped from row 11 to 50 and column 21 to 50.
            scan[:, 10:50]  # Gets all frames and all columns but crops from row 11 to 50.
            scan[5, 10:20, 10:20]  # Obtains the 6th frame and crops it.

            scan[[1, 3, 4]]  # Produces an error, lists are not allowed.
            scan[1:5:2]  # Produces an error, steps are not allowed.
            scan[1, 3, 5]   # Error, integer indices are not allowed for the spatial dimensions.
            scan[1, 3:5:2]  # Produces an error, steps are not allowed when slicing.

            scan["1s":"5s"]  # Gets the scan frames from first to the fifth second
            scan[:"-5s"]  # Gets the scan frames up to the last 5 seconds

            scan[file.force1x.start:file.force1x.stop]  # Gets frames overlapping with force1x
        """

        def check_item(item, slicing_frames):
            """We don't allow slice steps, and we don't allow slicing with lists. Slicing with
            single integer value is only allowed for frames."""
            if isinstance(item, slice):
                if item.step is not None:
                    raise IndexError("Slice steps are not supported when indexing")

                return (
                    slice(
                        self._time_to_frame_index(item.start, is_start=True),
                        self._time_to_frame_index(item.stop, is_start=False),
                    )
                    if slicing_frames
                    else item
                )

            if isinstance(item, int):
                if not slicing_frames:
                    raise IndexError("Scalar indexing is not supported for spatial coordinates")
                return item

            raise IndexError(f"Indexing by {type(item).__name__} is not allowed.")

        try:
            frame_slice, *spatial_slices = item
        except TypeError:  # Unpack fails if not iterable => only single axis is sliced
            frame_slice, spatial_slices = item, []

        new_scan = self._scan_with_sliced_factories(
            check_item(frame_slice, True),
            tuple([check_item(item, False) for item in spatial_slices]),
        )

        if not new_scan:
            return new_scan

        # Excluding the dead time between frames only makes sense if we have more than one frame
        ts_ranges = new_scan.frame_timestamp_ranges(include_dead_time=new_scan.num_frames > 1)
        start, stop = ts_ranges[0][0], ts_ranges[-1][-1]
        new_scan.start = start
        new_scan.stop = stop

        return new_scan

    @property
    @method_cache("pixel_time_seconds")
    def pixel_time_seconds(self):
        """Pixel dwell time in seconds"""
        if self._has_default_factories():
            infowave = self.infowave  # Make sure we pull this out only once
            start, stop = first_pixel_sample_indices(infowave.data)
            return (stop - start + 1) * infowave._src.dt * 1e-9
        else:
            indices = np.zeros(self.timestamps.ndim, dtype=int)

            # We want the difference along the fast axis. The first (optional) index corresponds to
            # the frame. The last two indices correspond to the imaging axes.
            indices[-2 if self._metadata.scan_order[0] > self._metadata.scan_order[1] else -1] = 1
            return (self.timestamps.item(tuple(indices)) - self.timestamps.item(0)) / 1e9

    @property
    def num_frames(self):
        """Number of available frames"""
        if self._metadata.num_frames == 0:
            self._metadata = self._metadata.with_num_frames(
                reconstruct_num_frames(
                    self.infowave.data, self.pixels_per_line, self.lines_per_frame
                )
            )
        return self._metadata.num_frames

    def _tiff_image_metadata(self) -> dict:
        """Create metadata for the ImageDescription field of TIFFs used by `export_tiff()`."""
        metadata = super()._tiff_image_metadata()
        metadata["Number of frames"] = self.num_frames
        return metadata

    def _tiff_timestamp_ranges(self, include_dead_time) -> list:
        """Create Timestamp ranges for the DateTime field of TIFFs used by `export_tiff()`."""
        return self.frame_timestamp_ranges(include_dead_time=include_dead_time)

    def frame_timestamp_ranges(self, *, include_dead_time=False):
        """Get start and stop timestamp of each frame in the scan.

        Note: The stop timestamp for each frame is defined as the first sample past the end of the
        relevant data such that the timestamps can be used for slicing directly.

        Parameters
        ----------
        include_dead_time : bool
            Include dead time at the end of each frame (default: False).


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

        ts_min = self._timestamps(reduce=np.min)
        ts_max = self._timestamps(reduce=np.max)
        delta_ts = int(1e9 / self.infowave.sample_rate)  # We want the sample beyond the end
        if ts_min.ndim == 2:
            return [(np.min(ts_min), np.max(ts_max) + delta_ts)]
        else:
            if include_dead_time:
                frame_time = ts_min[1, 0, 0] - ts_min[0, 0, 0]
                return [(t, t + frame_time) for t in ts_min[:, 0, 0]]
            else:
                maximum_timestamp = np.max(ts_max, axis=tuple(range(1, ts_max.ndim)))
                return [(t1, t2) for t1, t2 in zip(ts_min[:, 0, 0], maximum_timestamp + delta_ts)]

    def plot_correlated(
        self,
        channel_slice,
        frame=0,
        reduce=np.mean,
        channel="rgb",
        figure_scale=0.75,
        adjustment=no_adjustment,
        *,
        vertical=False,
        include_dead_time=False,
        return_frame_setter=False,
        downsample_to_frames=True,
    ):
        """Downsample channel on a frame by frame basis and plot the results. The downsampling
        function (e.g. np.mean) is evaluated for the time between a start and end time of a frame.
        Note: In environments which support interactive figures (e.g. jupyter notebook with
        ipywidgets or interactive python) this plot will be interactive.

        Actions
        -------
        left-click in the left axes
            Show the corresponding image frame in the right axes.

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
        vertical : bool, optional
            Align plots vertically.
        include_dead_time : bool, optional
            Include dead time between scan frames.
        return_frame_setter : bool, optional
            Whether to return a handle that allows updating the plotted frame.
        downsample_to_frames : bool, optional
            Downsample the channel data over frame timestamp ranges (default: True).

        Examples
        --------
        ::

            from lumicks import pylake

            file = pylake.File("example.h5")
            scan = file.scans["my scan"]
            scan.plot_correlated(file.force1x, channel="red")
        """
        from lumicks.pylake.nb_widgets.correlated_plot import plot_correlated

        def plot_channel(frame):
            try:
                return self._get_plot_data(channel, frame=frame, adjustment=adjustment)
            except ValueError:
                raise RuntimeError("Invalid channel selected")

        def post_update(image_handle, image):
            return adjustment._update_limits(image_handle, image, channel)

        def title_factory(frame):
            return make_image_title(self, frame, show_name=False)

        frame_timestamps = self.frame_timestamp_ranges(include_dead_time=include_dead_time)

        frame_setter = plot_correlated(
            channel_slice,
            frame_timestamps,
            plot_channel,
            title_factory,
            frame,
            reduce,
            colormap=colormaps._get_default_colormap(channel),
            figure_scale=figure_scale,
            post_update=post_update,
            vertical=vertical,
            downsample_to_frames=downsample_to_frames,
        )
        if return_frame_setter:
            return frame_setter

    @property
    def lines_per_frame(self):
        """Number of scanned lines in each frame"""
        return self._num_pixels[self._metadata.scan_order[1]]

    @property
    def _reconstruction_shape(self):
        """Shape used when reconstructing the image from raw photon counts (ordered by axis scan
        speed slow to fast)."""
        return (self.lines_per_frame, self.pixels_per_line)

    @property
    def shape(self):
        """Shape of the reconstructed :class:`~lumicks.pylake.scan.Scan` image"""
        shape = reversed(self._num_pixels)
        return (self.num_frames, *shape, 3) if self.num_frames > 1 else (*shape, 3)

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
        Here X, Y, Z correspond to axis number 0, 1 and 2. So for an YZ scan, we'd want Y on the X axis.
        """
        data = data.squeeze()

        if self._metadata.scan_order[0] > self._metadata.scan_order[1]:
            new_axis_order = np.arange(len(data.shape), dtype=int)
            new_axis_order[-1], new_axis_order[-2] = new_axis_order[-2], new_axis_order[-1]
            return np.transpose(data, new_axis_order)
        else:
            return data

    def plot(
        self,
        channel="rgb",
        *,
        frame=0,
        adjustment=no_adjustment,
        axes=None,
        image_handle=None,
        show_title=True,
        show_axes=True,
        scale_bar=None,
        **kwargs,
    ):
        """Plot a scan frame for the requested color channel(s).

        Parameters
        ----------
        channel : {"red", "green", "blue", "rgb"}, optional
            Color channel to plot.
        frame : int, optional
            Index of the frame to plot.
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
        if frame < 0:
            raise IndexError("negative indexing is not supported.")
        axes = get_axes(axes=axes, image_handle=image_handle)

        if show_axes is False:
            axes.set_axis_off()

        positional_unit = r"μm"
        if scale_bar and not image_handle:
            scale_bar._attach_scale_bar(axes, 1.0, 1.0, positional_unit, positional_unit)

        image = self._get_plot_data(
            channel, adjustment, frame=frame if self.num_frames != 1 else None
        )

        x_um, y_um = self.size_um
        default_kwargs = dict(
            # With origin set to upper (default) bounds should be given as (0, n, n, 0)
            extent=[0, x_um, y_um, 0],
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
        scan_axes = self._metadata.ordered_axes
        axes.set_xlabel(rf"{scan_axes[0].axis_label.lower()} ({positional_unit})")
        axes.set_ylabel(rf"{scan_axes[1].axis_label.lower()} ({positional_unit})")
        if show_title:
            axes.set_title(make_image_title(self, frame))

        return image_handle


class EmptyScan(Scan):
    def plot(self, channel="rgb", **kwargs):
        raise RuntimeError("Cannot plot empty Scan")

    def _image(self, channel):
        shape = (self.pixels_per_line, 0, 3) if channel == "rgb" else (self.pixels_per_line, 0)
        return np.empty(shape)

    def _has_default_factories(self):
        return False

    def get_image(self, channel="rgb"):
        return self._image(channel)

    def __bool__(self):
        return False
