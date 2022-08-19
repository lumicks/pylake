import numpy as np
import os
import tifffile
from deprecated.sphinx import deprecated
from .detail.imaging_mixins import VideoExport, FrameIndex
from .adjustments import ColorAdjustment
from .detail.widefield import TiffStack
from .detail.image import make_image_title


class CorrelatedStack(VideoExport, FrameIndex):
    """CorrelatedStack acquired with Bluelake. Bluelake can export stacks of images to various
    formats. These can be opened and correlated to timeline data using CorrelatedStack.

    Parameters
    ----------
    *image_names : str
        Filenames for the image stack. Typically a TIFF file recorded from a camera in Bluelake.
    align : bool
        If enabled, multi-channel images will be reconstructed from the image alignment metadata
        from Bluelake. The default value is `True`.

    Examples
    --------
    ::

        from lumicks import pylake

        # Loading a stack.
        stack = pylake.CorrelatedStack("example.tiff")

        # Making a plot where force is correlated to images in the stack.
        file = pylake.File("example.h5")
        stack.plot_correlated(file.force1x)

        # Determine the force trace averaged over frame 2...9.
        file.force1x.downsampled_over(stack[2:10].frame_timestamp_ranges())

        # Loading multiple TIFFs into a stack.
        stack = pylake.CorrelatedStack("example.tiff", "example2.tiff", "example3.tiff")
    """

    def __init__(self, *image_names, align=True):
        self.src = TiffStack.from_file(list(image_names), align_requested=align)
        self.name = (
            os.path.splitext(os.path.basename(str(image_names)))[0]
            if len(image_names) == 1
            else "Multi-file stack"
        )
        self._start_idx = 0
        self._stop_idx = self.src.num_frames
        self._step = 1

    def _handle_cropping(self, item):
        """Crop the stack based on tuple of slices"""

        def interpret_crop(item):
            if isinstance(item, slice):
                if item.step is not None:
                    raise IndexError("Slice steps are not supported when indexing")
                return item.start, item.stop
            else:
                return item, item + 1

        if len(item) > 3:
            raise IndexError("Only three indices are accepted when slicing CorrelatedStacks.")

        rows = interpret_crop(item[1]) if len(item) >= 2 else (None, None)
        columns = interpret_crop(item[2]) if len(item) >= 3 else (None, None)
        return self.src.with_roi(np.array([columns, rows]).flatten()), item[0]

    def __getitem__(self, item):
        """Returns specific frame(s) and/or cropped stacks.

        The first item refers to the camera frames; both indexing (by integer) and slicing are
        allowed, but using steps or slicing by a list is not.
        The last two items refer to the spatial dimensions in pixel rows and columns. Only full
        slices (without steps) are allowed. All values should be given in pixels.

        Examples
        --------
        ::

            import lumicks.pylake as lk

            stack = lk.CorrelatedStack("test.tiff")

            stack[5]  # Gets the 6th frame of the scan (0 is the first).
            stack[1:5]  # Get scan frames 1, 2, 3 and 4.
            stack[:, 10:50, 20:50]  # Gets all frames cropped from row 11 to 50 and column 21 to 50.
            stack[:, 10:50]  # Gets all frames and all columns, but crops from row 11 to 50.
            stack[5, 10:20, 10:20]  # Obtains the 6th frame and crops it.

            stack[[1, 3, 4]]  # Produces an error, lists are not allowed.
            stack[1:5:2]  # Produces an error, steps are not allowed.
            stack[1, 3, 5]   # Error, integer indices are not allowed for the spatial dimensions.
            stack[1, 3:5:2]  # Produces an error, steps are not allowed when slicing.

            stack["1s":"5s"]  # Slice the stack from 1 to 5 seconds
            stack[:"-5s"]  # Slice the stack up to the last 5 seconds
        """
        src, item = self._handle_cropping(item) if isinstance(item, tuple) else (self.src, item)

        if isinstance(item, slice):
            item = slice(
                self._time_to_frame_index(item.start, is_start=True),
                self._time_to_frame_index(item.stop, is_start=False),
                item.step,
            )
            start, stop, step = item.indices(self.num_frames)
            new_start = self._start_idx + self._step * start
            new_stop = self._start_idx + self._step * stop
            new_step = self._step * step

            # To have analogous behaviour to indexing of numpy arrays, first check if slice would be
            # empty and then check if slice would be reversed
            if new_stop == new_start or np.sign(new_stop - new_start) != np.sign(new_step):
                raise NotImplementedError("Slice is empty")
            if new_step < 0:
                raise NotImplementedError("Reverse slicing is not supported")

            return CorrelatedStack.from_dataset(src, self.name, new_start, new_stop, new_step)
        else:
            idx = item if item >= 0 else item + self.num_frames
            new_start = self._start_idx + self._step * idx
            new_stop = new_start + self._step
            new_step = self._step

            if new_start < self._start_idx or new_start >= self._stop_idx:
                raise IndexError("Index out of bounds")
            return CorrelatedStack.from_dataset(src, self.name, new_start, new_stop, new_step)

    def __iter__(self):
        idx = 0
        while idx < self.num_frames:
            yield self._get_frame(idx)
            idx += 1

    @classmethod
    @deprecated(
        reason=("Renamed to `from_dataset()` for consistency with `Kymo` and `Scan`."),
        action="always",
        version="0.10.1",
    )
    def from_data(cls, data, name=None, start_idx=0, stop_idx=None):
        return cls.from_dataset(data, name, start_idx, stop_idx)

    @property
    def shape(self):
        base_shape = (self.num_frames, *self.src._shape)
        return (*base_shape, 3) if self.src.is_rgb else base_shape

    @classmethod
    def from_dataset(cls, data, name=None, start_idx=0, stop_idx=None, step=1) -> "CorrelatedStack":
        """Construct CorrelatedStack from image stack object

        Parameters
        ----------
        data : TiffStack
            TiffStack object.
        name : str
            Plot label of the correlated stack
        start_idx : int
            Index at the first frame.
        stop_idx : int
            Index beyond the last frame.
        step : int
            Step value for slicing frames.
        """
        new_correlated_stack = cls.__new__(cls)
        new_correlated_stack.src = data
        new_correlated_stack.name = name
        new_correlated_stack._start_idx = start_idx
        new_correlated_stack._stop_idx = (
            new_correlated_stack.src.num_frames if stop_idx is None else stop_idx
        )
        new_correlated_stack._step = step
        return new_correlated_stack

    def crop_by_pixels(self, x_min, x_max, y_min, y_max):
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
        data = self.src.with_roi(np.array([x_min, x_max, y_min, y_max]))
        return self.from_dataset(data, self.name, self._start_idx, self._stop_idx)

    def define_tether(self, point1, point2):
        """Returns a copy of the stack rotated such that the tether defined by `point_1` and
        `point_2` is horizontal.

        Parameters
        ----------
        point_1 : (float, float)
            (x, y) coordinates of the tether start point
        point_2 : (float, float)
            (x, y) coordinates of the tether end point
        """
        data = self.src.with_tether((point1, point2))
        return self.from_dataset(data, self.name, self._start_idx, self._stop_idx)

    def get_image(self, channel="rgb"):
        """Get image data for the full stack as an `np.ndarray`.

        Parameters
        ----------
        channel : {'red', 'green', 'blue', 'rgb'}
            The color channel of the requested data.
            For single-color data, this argument is ignored.
        """
        if self.src._description.is_rgb:
            channel_indices = {"red": 0, "green": 1, "blue": 2, "rgb": slice(None)}
            slc = (slice(None), slice(None), channel_indices[channel])
        else:
            slc = (slice(None),)

        return np.stack([frame.data[slc] for frame in self], axis=0).squeeze()

    def plot(
        self,
        frame=0,
        channel="rgb",
        show_title=True,
        axes=None,
        adjustment=ColorAdjustment.nothing(),
        *,
        image_handle=None,
        **kwargs,
    ):
        """Plot image from image stack

        Parameters
        ----------
        frame : int, optional
            Index of the frame to plot.
        channel : 'rgb', 'red', 'green', 'blue', None; optional
            Channel to plot for RGB images (None defaults to 'rgb')
            Not used for grayscale images
        show_title : bool, optional
            Controls display of auto-generated plot title
        axes : mpl.axes.Axes or None
            If supplied, the axes instance in which to plot.
        image_handle : `matplotlib.image.AxesImage` or None
            Optional image handle which is used to update plots with new data rather than
            reconstruct them (better for performance).
        **kwargs
            Forwarded to :func:`matplotlib.pyplot.imshow`.
        """
        import matplotlib.pyplot as plt

        if axes is None:
            axes = plt.gca() if image_handle is None else image_handle.axes
        elif image_handle is not None and id(axes) != id(image_handle.axes):
            raise ValueError("Supplied image_handle with a different axes than the provided axes")

        default_kwargs = dict(cmap="gray")
        kwargs = {**default_kwargs, **kwargs}

        image = self._get_frame(frame)._get_plot_data(channel, adjustment=adjustment)

        if not image_handle:
            image_handle = axes.imshow(image, **kwargs)
        else:
            # Updating the image data in an existing plot is a lot faster than re-plotting with
            # `imshow`.
            image_handle.set_data(image)

        adjustment._update_limits(image_handle, image, channel)

        if show_title:
            axes.set_title(make_image_title(self, frame))

        return image_handle

    def plot_tether(self, axes=None, **kwargs):
        """Plot a line at the tether position.

        Parameters
        ----------
        axes : mpl.axes.Axes or None
            If supplied, the axes instance in which to plot.
        **kwargs
            Forwarded to :func:`matplotlib.pyplot.plot`.
        """
        import matplotlib.pyplot as plt

        if not self.src._tether:
            raise ValueError("A tether is not defined yet for this image stack.")

        if axes is None:
            axes = plt.gca()

        x, y = np.vstack(self.src._tether.ends).T
        tether_kwargs = {"c": "w", "marker": "o", "mfc": "none", "ls": ":", **kwargs}
        axes.plot(x, y, **tether_kwargs)

    def crop_and_rotate(self, frame=0, channel="rgb", show_title=True, **kwargs):
        """Open a widget to interactively edit the image stack.

        Actions
        -------

            * scrolling through frames using the mouse wheel
            * left-click to define the location of the tether

        Parameters
        ----------
        frame : int, optional
            Index of the frame to plot.
        channel : 'rgb', 'red', 'green', 'blue', None; optional
            Channel to plot for RGB images (None defaults to 'rgb')
            Not used for grayscale images
        show_title : bool, optional
            Controls display of auto-generated plot title
        **kwargs
            Forwarded to :func:`matplotlib.pyplot.imshow`.

        Examples
        --------
        ::

            from lumicks import pylake
            import matplotlib.pyplot as plt

            # Loading a stack.
            stack = pylake.CorrelatedStack("example.tiff")
            widget = stack.crop_and_rotate()
            plt.show()

            # Select cropping ROI by right-click drag
            # Select tether ends by left-click

            # Grab the updated image stack
            new_stack = widget.image
        """
        from .nb_widgets.image_editing import ImageEditorWidget

        return ImageEditorWidget(self, frame, channel, show_title, **kwargs)

    def _get_frame(self, frame=0):
        if frame >= self.num_frames or frame < 0:
            raise IndexError("Frame index out of range")
        return self.src.get_frame(self._start_idx + frame * self._step)

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
        function (e.g. `np.mean`) is evaluated for the time between a start and end time of a
        frame.

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

        Note
        ----
        In environments which support interactive figures (e.g. `jupyter notebook` with
        `ipywidgets` or interactive python) this plot will be interactive.

        Examples
        --------
        ::

            from lumicks import pylake

            file = pylake.File("example.h5")
            stack = pylake.CorrelatedStack("example.tiff")
            stack.plot_correlated(file.force1x, frame=5)
        """
        from lumicks.pylake.nb_widgets.correlated_plot import plot_correlated

        title_factory = lambda frame: make_image_title(self, frame, show_name=False)
        frame_timestamps = self.frame_timestamp_ranges()

        def frame_grabber(frame_idx):
            return self._get_frame(frame_idx)._get_plot_data(channel, adjustment=adjustment)

        def post_update(image_handle, image):
            return adjustment._update_limits(image_handle, image, channel)

        plot_correlated(
            channel_slice,
            frame_timestamps,
            frame_grabber,
            title_factory,
            frame,
            reduce,
            figure_scale=figure_scale,
            post_update=post_update,
        )

    def export_tiff(self, file_name):
        """Export a video of a particular scan plot

        Parameters
        ----------
        file_name : str
            File name to export to.
        """
        from . import __version__ as version

        def parse_tags(frame):
            # Parse original file tags into list of tuples
            #   [(code, dtype, count, value, writeonce)]
            #   code is defined by the TIFF specification
            #   dtype is defined in tifffile
            # Only tags that are not resolved automatically by `TiffWriter.save()` are needed

            # Orientation, uint16, len, ORIENTATION.TOPLEFT
            orientation = (274, "H", 1, 1)

            # SampleFormat, uint16, len, number of channels
            n_channels = 3 if frame.is_rgb else 1
            sample_format = (339, "H", n_channels, (1,) * n_channels)

            # DateTime, str, len, start:stop
            datetime = frame._page.tags["DateTime"].value
            datetime = (306, "s", len(datetime), datetime)

            return (orientation, sample_format, datetime)

        # re-name alignment matrices fields in image description
        # to reflect the fact that the image has already been processed
        description = self.src._description.for_export

        # add pylake to Software tag
        software = self.src._description.software
        if "pylake" not in software:
            software += f", pylake v{version}"

        # write frames sequentially
        with tifffile.TiffWriter(file_name) as tif:
            for frame in self:
                tif.write(
                    frame.data,
                    description=description,
                    software=software,
                    metadata=None,  # suppress tifffile default ImageDescription tag
                    contiguous=False,  # needed to write tags on each page
                    extratags=parse_tags(frame),
                    photometric="rgb" if frame.is_rgb else "minisblack",
                )

    @property
    def num_frames(self):
        """Number of frames in the stack."""
        return max(-1, (self._stop_idx - self._start_idx - 1)) // self._step + 1

    @property
    @deprecated(
        reason=(
            "Access to raw frame instances will be removed in a future release. "
            "All operations on these objects should be handled through the `CorrelatedStack` public API. "
            "For example, to retrieve the image data as an `np.ndarray` please use `CorrelatedStack.get_image()`."
        ),
        action="always",
        version="0.10.1",
    )
    def raw(self):
        """Raw frame data."""
        if self.num_frames > 1:
            return [self._get_frame(idx) for idx in range(self.num_frames)]
        else:
            return self._get_frame(0)

    @property
    def start(self):
        """Starting time stamp of the stack."""
        return self._get_frame(0).start

    @property
    def stop(self):
        """Final time stamp of the stack."""
        return self._get_frame(self.num_frames - 1).stop

    @property
    @deprecated(
        reason=(
            "For camera based images only the integration start/stop timestamps are defined. "
            "Use `CorrelatedStack.frame_timestamp_ranges()` instead."
        ),
        action="always",
        version="0.11.1",
    )
    def timestamps(self):
        """Get start and stop timestamp of each frame in the stack."""
        return self.frame_timestamp_ranges()

    def frame_timestamp_ranges(self, *, include_dead_time=False):
        """Get start and stop timestamp of each frame in the stack.

        Parameters
        ----------
        include_dead_time : bool
            Include dead time between frames.
        """
        ts_ranges = [frame.frame_timestamp_range for frame in self]
        if include_dead_time:
            frame_ts = [
                (leading[0], trailing[0]) for leading, trailing in zip(ts_ranges, ts_ranges[1:])
            ]
            if len(ts_ranges) >= 2:
                dt = ts_ranges[-1][0] - ts_ranges[-2][0]
                stop = ts_ranges[-1][0] + dt
            else:
                stop = ts_ranges[-1][1]
            frame_ts.append((ts_ranges[-1][0], stop))
            return frame_ts
        else:
            return ts_ranges
