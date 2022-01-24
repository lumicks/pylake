import numpy as np
import os
import tifffile
import warnings
from deprecated.sphinx import deprecated
from .detail.widefield import TiffStack
from .detail.image import make_image_title


class CorrelatedStack:
    """CorrelatedStack acquired with Bluelake. Bluelake can export stacks of images to various formats. These can be
    opened and correlated to timeline data using CorrelatedStack.

    Parameters
    ----------
    image_name : str
        Filename for the image stack. Typically a TIFF file recorded from a camera in Bluelake.

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
        file.force1x.downsampled_over(stack[2:10].frame_timestamp_ranges)
    """

    def __init__(self, image_name, align=True):
        self.src = TiffStack.from_file(image_name, align_requested=align)
        self.name = os.path.splitext(os.path.basename(image_name))[0]
        self.start_idx = 0
        self.stop_idx = self.src.num_frames

    def __getitem__(self, item):
        """All indexing is in frames"""
        if isinstance(item, slice):
            if item.step is not None:
                raise IndexError("Slice steps are not supported")

            start, stop, _ = item.indices(self.num_frames)
            new_start = self.start_idx + start
            new_stop = self.start_idx + stop

            if new_stop - new_start < 0:
                raise NotImplementedError("Reverse slicing is not supported")
            if new_stop == new_start:
                raise NotImplementedError("Slice is empty")

            return CorrelatedStack.from_dataset(self.src, self.name, new_start, new_stop)
        else:
            item = self.start_idx + item if item >= 0 else self.stop_idx + item
            if item >= self.stop_idx or item < self.start_idx:
                raise IndexError("Index out of bounds")
            return CorrelatedStack.from_dataset(self.src, self.name, item, item + 1)

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

    @classmethod
    def from_dataset(cls, data, name=None, start_idx=0, stop_idx=None):
        """Construct CorrelatedStack from image stack object

        Parameters
        ----------
        data : TiffStack
            TiffStack object.
        name : str
            Plot label of the correlated stack
        start_idx : int
            Index at the first frame.
        stop_idx: int
            Index beyond the last frame.
        """
        new_correlated_stack = cls.__new__(cls)
        new_correlated_stack.src = data
        new_correlated_stack.name = name
        new_correlated_stack.start_idx = start_idx
        new_correlated_stack.stop_idx = (
            new_correlated_stack.src.num_frames if stop_idx is None else stop_idx
        )
        return new_correlated_stack

    def crop_by_pixels(self, x_min, x_max, y_min, y_max):
        """Crop the image stack by pixel values.

        Parameters
        ----------
        x_min : int
            minimum x pixel (inclusive)
        x_max : int
            maximum x pixel (exclusive)
        y_min : int
            minimum y pixel (inclusive)
        y_max : int
            maximum y pixel (exclusive)
        """
        data = self.src.with_roi(np.array([x_min, x_max, y_min, y_max]))
        return self.from_dataset(data, self.name, self.start_idx, self.stop_idx)

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
        return self.from_dataset(data, self.name, self.start_idx, self.stop_idx)

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

    def plot(self, frame=0, channel="rgb", show_title=True, axes=None, **kwargs):
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
        **kwargs
            Forwarded to :func:`matplotlib.pyplot.imshow`.
        """
        import matplotlib.pyplot as plt

        if axes is None:
            axes = plt.gca()

        default_kwargs = dict(cmap="gray", vmax=None)
        kwargs = {**default_kwargs, **kwargs}

        image = self._get_frame(frame)._get_plot_data(channel, vmax=kwargs["vmax"])
        axes.imshow(image, **kwargs)

        if show_title:
            axes.set_title(make_image_title(self, frame))

    def plot_tether(self, axes=None, **kwargs):
        """Plot a line at the tether position.

        Parameters
        ----------
        axes : mpl.axes.Axes or None
            If supplied, the axes instance in which to plot.
        **kwargs
            Forwarded to :fun:`matplotlib.pyplot.plot`.
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

        Actions include:
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
        return self.src.get_frame(self.start_idx + frame)

    def plot_correlated(
        self, channel_slice, frame=0, reduce=np.mean, channel="rgb", figure_scale=0.75
    ):
        """Downsample channel on a frame by frame basis and plot the results. The downsampling function (e.g. np.mean)
        is evaluated for the time between a start and end time of a frame. Note: In environments which support
        interactive figures (e.g. jupyter notebook with ipywidgets or interactive python) this plot will be interactive.

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
        frame_timestamps = self.frame_timestamp_ranges

        plot_correlated(
            channel_slice,
            frame_timestamps,
            lambda frame_idx: self._get_frame(frame_idx)._get_plot_data(channel),
            title_factory,
            frame,
            reduce,
            figure_scale=figure_scale,
        )

    def export_tiff(self, file_name, roi=None):
        """Export a video of a particular scan plot

        Parameters
        ----------
        file_name : str
            File name to export to.
        roi : list_like
            region of interest in pixel values [xmin, xmax, ymin, ymax]
            *Deprecated since v0.10.1* Instead, use `CorrelatedStack.crop_by_pixels()` to select the ROI before exporting."
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

        # crop to ROI if applicable
        if roi is not None:
            warnings.warn(
                (
                    "The `roi` argument is deprecated since v0.10.1 "
                    "Instead, use `CorrelatedStack.crop_by_pixels()` to select the ROI before exporting."
                ),
                DeprecationWarning,
            )
            to_save = self.crop_by_pixels(*roi)
        else:
            to_save = self

        # re-name alignment matrices fields in image description
        # to reflect the fact that the image has already been processed
        description = to_save.src._description.for_export

        # add pylake to Software tag
        software = to_save.src._tags["Software"].value
        if "pylake" not in software:
            software += f", pylake v{version}"

        # write frames sequentially
        with tifffile.TiffWriter(file_name) as tif:
            for frame in to_save:
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
        return self.stop_idx - self.start_idx

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
            "Use `CorrelatedStack.frame_timestamp_ranges` instead."
        ),
        action="always",
        version="0.11.1",
    )
    def timestamps(self):
        return self.frame_timestamp_ranges

    @property
    def frame_timestamp_ranges(self):
        """List of time stamps."""
        return [
            (self._get_frame(idx).start, self._get_frame(idx).stop)
            for idx in range(self.num_frames)
        ]
