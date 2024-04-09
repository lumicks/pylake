import os
import warnings
from typing import Union, Iterator, Optional

import numpy as np
import numpy.typing as npt
from deprecated.sphinx import deprecated

from .kymo import Kymo, _kymo_from_image_stack
from .adjustments import no_adjustment
from .detail.image import make_image_title
from .detail.plotting import get_axes, show_image
from .detail.widefield import TiffStack, _frame_timestamps_from_exposure_timestamps
from .detail.imaging_mixins import FrameIndex, TiffExport, VideoExport


class ImageStack(FrameIndex, TiffExport, VideoExport):
    """Open a TIF file acquired with Bluelake. Bluelake can export stacks of images from various
    cameras and these can be opened and correlated to timeline data.

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
        stack = pylake.ImageStack("example.tiff")

        # Making a plot where force is correlated to images in the stack.
        file = pylake.File("example.h5")
        stack.plot_correlated(file.force1x)

        # Determine the force trace averaged over frame 2...9.
        file.force1x.downsampled_over(stack[2:10].frame_timestamp_ranges())

        # Loading multiple TIFFs into a stack.
        stack = pylake.ImageStack("example.tiff", "example2.tiff", "example3.tiff")
    """

    def __init__(self, *image_names, align=True):
        self._src = TiffStack.from_file(list(image_names), align_requested=align)
        self.name = (
            os.path.splitext(os.path.basename(str(image_names[0])))[0]
            if len(image_names) == 1
            else "Multi-file stack"
        )
        self._start_idx = 0
        self._stop_idx = self._src.num_frames
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
            raise IndexError("Only three indices are accepted when slicing an ImageStack.")

        rows = interpret_crop(item[1]) if len(item) >= 2 else (None, None)
        columns = interpret_crop(item[2]) if len(item) >= 3 else (None, None)
        return self._src.with_roi(np.array([columns, rows]).flatten()), item[0]

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

            stack = lk.ImageStack("test.tiff")

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
        src, item = self._handle_cropping(item) if isinstance(item, tuple) else (self._src, item)

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

            return ImageStack.from_dataset(src, self.name, new_start, new_stop, new_step)
        else:
            idx = item if item >= 0 else item + self.num_frames
            new_start = self._start_idx + self._step * idx
            new_stop = new_start + self._step
            new_step = self._step

            if new_start < self._start_idx or new_start >= self._stop_idx:
                raise IndexError("Index out of bounds")
            return ImageStack.from_dataset(src, self.name, new_start, new_stop, new_step)

    def __iter__(self):
        idx = 0
        while idx < self.num_frames:
            yield self._get_frame(idx)
            idx += 1

    @property
    def shape(self):
        base_shape = (self.num_frames, *self._src._shape)
        return (*base_shape, 3) if self._src.is_rgb else base_shape

    @classmethod
    def from_dataset(cls, data, name=None, start_idx=0, stop_idx=None, step=1) -> "ImageStack":
        """Construct ImageStack from image stack object

        Parameters
        ----------
        data : TiffStack
            TiffStack object.
        name : str
            Plot label of the image stack
        start_idx : int
            Index at the first frame.
        stop_idx : int
            Index beyond the last frame.
        step : int
            Step value for slicing frames.
        """
        new_image_stack = cls.__new__(cls)
        new_image_stack._src = data
        new_image_stack.name = name
        new_image_stack._start_idx = start_idx
        new_image_stack._stop_idx = (
            new_image_stack._src.num_frames if stop_idx is None else stop_idx
        )
        new_image_stack._step = step
        return new_image_stack

    @property
    def _pixel_calibration_factors(self):
        return np.asarray(self.pixelsize_um) if self.pixelsize_um else np.array((1.0, 1.0))

    def _crop(self, x_min, x_max, y_min, y_max):
        """Crop the image stack in the units the image is calibrated in."""
        return self.crop_by_pixels(
            int(x_min / self._pixel_calibration_factors[0]) if x_min else None,
            int(x_max / self._pixel_calibration_factors[0]) if x_max else None,
            int(y_min / self._pixel_calibration_factors[1]) if y_min else None,
            int(y_max / self._pixel_calibration_factors[1]) if y_max else None,
        )

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
        data = self._src.with_roi(np.array([x_min, x_max, y_min, y_max]))
        return self.from_dataset(data, self.name, self._start_idx, self._stop_idx)

    def define_tether(self, point1, point2):
        """Returns a copy of the stack rotated such that the tether defined by `point_1` and
        `point_2` is horizontal.

        Note that the tether position is given in the image units (microns).

        Parameters
        ----------
        point_1 : (float, float)
            (x, y) coordinates of the tether start point
        point_2 : (float, float)
            (x, y) coordinates of the tether end point
        """
        data = self._src.with_tether(np.asarray((point1, point2)) / self._pixel_calibration_factors)
        return self.from_dataset(data, self.name, self._start_idx, self._stop_idx)

    def to_kymo(self, half_window, reduce=np.sum) -> Kymo:
        """Convert this :class:`ImageStack` to a :class:`~lumicks.pylake.kymo.Kymo` by sampling
        along the tether.

        Parameters
        ----------
        half_window : int
            Number of pixels adjacent to the tether on either side to average over. The kymograph
            will be calculated from the pixel values reduced to a one pixel line using the
            downsampling function provided in :func:`reduce`.
        reduce : callable
            The :mod:`numpy` function which is going to reduce multiple samples into one. The
            default is :func:`np.sum <numpy.sum>`.

        Returns
        -------
        Kymo

        Raises
        ------
        ValueError
            If the frame rate or exposure time are not constant.
        ValueError
            If the image stack does not have a tether defined.
        ValueError
            If the number of adjacent lines are negative.
        ValueError
            If the number of adjacent lines exceed the image dimensions.

        Notes
        -----
        In order to convert a :class:`ImageStack` to a :class:`~lumicks.pylake.kymo.Kymo`, a tether
        needs to be defined first. See the example below for more information.

        Examples
        --------
        ::

            import lumicks.pylake as lk
            import matplotlib.pyplot as plt

            stack = lk.ImageStack("camera_recording.tiff")

            # Plot the stack
            plt.figure()
            stack.plot()

            # Define tether by a pair of X, Y coordinates.
            stack = stack.define_tether((7.19, 4.07), (22.74, 4.15))

            # Plot the tether to verify.
            stack.plot_tether()
            kymo = stack.to_kymo(half_window=3)

            # Plot the kymograph just created
            plt.figure()
            kymo.plot()

            # Or alternatively, use the interactive widget to define the tether graphically
            %matplotlib widget  # Enable interactive widgets
            stack = lk.ImageStack("camera_recording.tiff")

            # Opens an interactive widget. See help(lk.ImageStack.crop_and_rotate) for more info.
            widget = stack.crop_and_rotate()
            kymo = widget.image.to_kymo(half_window=3)
            kymo.plot()
        """
        return _kymo_from_image_stack(
            self,
            half_window,
            self.pixelsize_um[0] if self.pixelsize_um else None,
            self.name,
            reduce,
        )

    def get_image(self, channel="rgb"):
        """Get image data for the full stack as an `np.ndarray`.

        Parameters
        ----------
        channel : {'red', 'green', 'blue', 'rgb'}
            The color channel of the requested data.
            For single-color data, this argument is ignored.
        """
        if self._src.is_rgb:
            channel_indices = {"red": 0, "green": 1, "blue": 2, "rgb": slice(None)}
            slc = (slice(None), slice(None), channel_indices[channel])
        else:
            slc = (slice(None),)

        return np.stack([frame.data[slc] for frame in self], axis=0).squeeze()

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
        """Plot a frame from the image stack for the requested color channel(s)

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
        axes = get_axes(axes=axes, image_handle=image_handle)

        if show_axes is False:
            axes.set_axis_off()

        positional_unit = "px" if not self.pixelsize_um else r"μm"
        if scale_bar and not image_handle:
            scale = 1 if self.pixelsize_um else 100
            scale_bar._attach_scale_bar(axes, scale, scale, positional_unit, positional_unit)

        image = self._get_frame(frame)._get_plot_data(channel, adjustment=adjustment)

        default_kwargs = dict(cmap="gray")
        if self.pixelsize_um:
            x_um, y_um = self.size_um
            pixel_size_x, pixel_size_y = self.pixelsize_um

            # With origin set to upper (default) bounds are defined as: (min_x, max_x, max_y, min_y)
            default_kwargs["extent"] = [
                -pixel_size_x,
                x_um - pixel_size_x,
                y_um - pixel_size_y,
                -pixel_size_y,
            ]
            default_kwargs["aspect"] = (image.shape[0] / image.shape[1]) * (x_um / y_um)

        kwargs = {**default_kwargs, **kwargs}

        image_handle = show_image(
            image, adjustment, channel, image_handle=image_handle, axes=axes, **kwargs
        )

        axes.set_xlabel(rf"x ({positional_unit})")
        axes.set_ylabel(rf"y ({positional_unit})")

        if show_title:
            axes.set_title(make_image_title(self, frame))

        return image_handle

    def plot_tether(self, axes=None, **kwargs):
        """Plot a line at the tether position.

        Parameters
        ----------
        axes : matplotlib.axes.Axes or None
            If supplied, the axes instance in which to plot.
        **kwargs
            Forwarded to :func:`matplotlib.pyplot.plot`.
        """
        import matplotlib.pyplot as plt

        if not self._src._tether:
            raise ValueError("A tether is not defined yet for this image stack.")

        if axes is None:
            axes = plt.gca()

        x, y = np.vstack(self._src._tether.ends * self._pixel_calibration_factors).T
        tether_kwargs = {"c": "w", "marker": "o", "mfc": "none", "ls": ":", **kwargs}
        axes.plot(x, y, **tether_kwargs)

    def crop_and_rotate(self, frame=0, channel="rgb", show_title=True, **kwargs):
        """Open a widget to interactively edit the image stack.

        Actions
        -------
        mouse wheel
            Scroll through frames of the stack.
        left-click
            Define the location of the tether. First click defines the start of the tether
            and second click defines the end. Subsequent clicks will cycle back to re-defining
            the start, etc.
        right-click and drag
            Define the ROI to be cropped.

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

        Returns
        -------
        :class:`~lumicks.pylake.nb_widgets.image_editing.ImageEditorWidget`

        Examples
        --------
        ::

            from lumicks import pylake
            import matplotlib.pyplot as plt

            # Loading a stack.
            stack = pylake.ImageStack("example.tiff")
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
        return self._src.get_frame(self._start_idx + frame * self._step)

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
        function (e.g. `np.mean`) is evaluated for the time between a start and end time of a
        frame.

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
            :func:`np.mean <numpy.mean>`, but :func:`np.sum <numpy.sum>` could also be appropriate
            for some cases e.g. photon counts.
        channel : 'rgb', 'red', 'green', 'blue', None; optional
            Channel to plot for RGB images (None defaults to 'rgb')
            Not used for grayscale images
        figure_scale : float
            Scaling of the figure width and height. Values greater than one increase the size of the
            figure.
        adjustment : lk.ColorAdjustment
            Color adjustments to apply to the output image.
        vertical : bool, optional
            Align plots vertically, default: False.
        include_dead_time : bool, optional
            Include dead time between frames, default: False.
        return_frame_setter : bool, optional
            Whether to return a handle that allows updating the plotted frame, default: False.
        downsample_to_frames : bool, optional
            Downsample the channel data over frame timestamp ranges (default: True).

        Note
        ----
        In environments which support interactive figures (e.g. `jupyter notebook` with
        `ipywidgets` or interactive python) this plot will be interactive.

        Examples
        --------
        ::

            from lumicks import pylake

            file = pylake.File("example.h5")
            stack = pylake.ImageStack("example.tiff")
            stack.plot_correlated(file.force1x, frame=5)
        """
        from lumicks.pylake.nb_widgets.correlated_plot import plot_correlated

        def frame_grabber(frame_idx):
            return self._get_frame(frame_idx)._get_plot_data(channel, adjustment=adjustment)

        def post_update(image_handle, image):
            return adjustment._update_limits(image_handle, image, channel)

        frame_setter = plot_correlated(
            channel_slice=channel_slice,
            frame_timestamps=self.frame_timestamp_ranges(include_dead_time=include_dead_time),
            get_plot_data=frame_grabber,
            title_factory=lambda frame: make_image_title(self, frame, show_name=False),
            frame=frame,
            reduce=reduce,
            figure_scale=figure_scale,
            post_update=post_update,
            vertical=vertical,
            downsample_to_frames=downsample_to_frames,
        )

        if return_frame_setter:
            return frame_setter

    @property
    def size_um(self) -> Optional[list]:
        """Returns a `list` of axis sizes in um. Returns `None` if calibration is unavailable."""
        if self.pixelsize_um:
            return list(
                map(
                    lambda pixel_size, num_pixels: pixel_size * num_pixels,
                    self.pixelsize_um,
                    reversed(self._src._shape),
                )
            )

    @property
    def pixelsize_um(self) -> Optional[list]:
        """Returns a `list` of pixel sizes along each axis in um. Returns `None` if calibration is
        unavailable."""
        if self._src._description.pixelsize_um:
            return [self._src._description.pixelsize_um] * 2

    def export_tiff(self, file_name):
        """Export a video of a particular scan plot

        Parameters
        ----------
        file_name : str
            The name of the TIFF file where the image will be saved.
        """
        super().export_tiff(file_name, dtype=None, clip=False)

    def _tiff_frames(self, iterator=False) -> Union[npt.ArrayLike, Iterator]:
        """Create images for frames of TIFFs used by `export_tiff().`"""
        return (frame.data for frame in self) if iterator else self.get_image()

    def _tiff_image_metadata(self) -> dict:
        """Create metadata stored in the ImageDescription field of TIFFs used by `export_tiff()`"""
        # re-name alignment matrices fields in image description
        # to reflect the fact that the image has already been processed
        return self._src._description.for_export

    def _tiff_timestamp_ranges(self, include_dead_time) -> list:
        """Create Timestamp ranges for DateTime field of TIFFs used by `export_tiff().`"""
        return self.frame_timestamp_ranges(include_dead_time=include_dead_time)

    def _tiff_writer_kwargs(self) -> dict:
        """Create keyword arguments used for `TiffWriter.write()` in `self.export_tiff()`."""
        from . import __version__ as version

        # add pylake to Software tag
        software = self._src._description.software
        if "pylake" not in software.lower():
            software += (", " if len(software) else "") + f"Pylake v{version}"

        write_kwargs = {
            "software": software,
            "photometric": "rgb" if self._src.is_rgb else "minisblack",
        }

        return write_kwargs

    @property
    def num_frames(self):
        """Number of frames in the stack."""
        return max(-1, (self._stop_idx - self._start_idx - 1)) // self._step + 1

    @property
    def start(self):
        """Starting time stamp of the stack."""
        return self._get_frame(0).start

    @property
    def stop(self):
        """Final time stamp of the stack."""
        return self._get_frame(self.num_frames - 1).stop

    def frame_timestamp_ranges(self, *, include_dead_time=False):
        """Get start and stop timestamp of each frame in the stack.

        Parameters
        ----------
        include_dead_time : bool
            Include dead time between frames.
        """
        if include_dead_time:
            ts_ranges = [frame.frame_timestamp_range for frame in self]

            if self._src._description._legacy_exposure:
                return _frame_timestamps_from_exposure_timestamps(ts_ranges)

            return ts_ranges

        frame_timestamps = [frame.exposure_timestamp_range for frame in self]
        if len(np.unique(np.diff(np.asarray(frame_timestamps), 1))) != 1:
            warnings.warn(
                RuntimeWarning(
                    "This image stack contains a non-constant exposure time. While the start time "
                    "of each frame is synchronized, the update of the exposure metadata can "
                    "lag behind. This means that when you average data over the frame, some frames "
                    "after the switch may take an incorrect exposure time into account in the "
                    "averaging."
                )
            )

        return frame_timestamps

    def close(self):
        """Closes `ImageStack` file handle.

        Note
        ----
        Attempting to access image data after closing the file handle will result in an `IOError` for both this
        `ImageStack` and any `ImageStack` derived from it through e.g. `ImageStack.crop_by_pixels()` or
        `ImageStack.define_tether()`
        """
        self._src.close()


@deprecated(
    reason="`CorrelatedStack` has been renamed to `ImageStack` and will be removed in the next release.",
    version="0.13.3",
    action="always",
)
def CorrelatedStack(*image_names, align=True):
    """Open a TIF file acquired with Bluelake. Bluelake can export stacks of images from various
    cameras and these can be opened and correlated to timeline data.

    Parameters
    ----------
    *image_names : str
        Filenames for the image stack. Typically a TIFF file recorded from a camera in Bluelake.
    align : bool
        If enabled, multi-channel images will be reconstructed from the image alignment metadata
        from Bluelake. The default value is `True`.
    """

    return ImageStack(*image_names, align=align)
